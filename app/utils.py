import json, time, re, functools, shutil, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Any, Optional, Dict, Tuple
from tqdm import tqdm
import numpy as np
import requests

import psycopg
from psycopg_pool import ConnectionPool

from app.logger import get_logger
from app.settings import get_settings

log = get_logger(__name__)
_db_pool: Optional[ConnectionPool] = None
_embedder: Optional["_BaseEmbedder"] = None
_embedder_lock = threading.Lock()


# ---------------- JSON ----------------


def safe_json_loads(text: str) -> dict:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
    return {}


# ---------------- DB (standalone PostgreSQL, no pgvector) ----------------


def get_db_pool() -> ConnectionPool:
    global _db_pool
    if _db_pool is not None:
        return _db_pool

    cfg = get_settings()
    conninfo = (
        f"host={cfg.db_host} port={cfg.db_port} dbname={cfg.db_name} "
        f"user={cfg.db_user} password={cfg.db_password}"
    )
    _db_pool = ConnectionPool(
        conninfo=conninfo,
        min_size=cfg.db_pool_min,
        max_size=cfg.db_pool_max,
        open=True,
        kwargs={"autocommit": True},
    )
    log.info(
        f"PostgreSQL ConnectionPool 初始化完成 "
        f"({cfg.db_host}:{cfg.db_port}/{cfg.db_name})"
    )
    return _db_pool


def get_db_conn() -> psycopg.Connection:
    global _db_pool
    if _db_pool is None:
        get_db_pool()
    conn = _db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
        return conn
    except Exception as e:
        log.warning(f"数据库连接心跳异常，正在尝试重连: {e}")
        _db_pool.putconn(conn, close=True)
        return _db_pool.getconn()


def release_db_conn(conn: psycopg.Connection) -> None:
    if _db_pool and conn:
        _db_pool.putconn(conn)


# ---------------- embedding <-> BYTEA ----------------


def emb_to_bytes(vec: np.ndarray) -> bytes:
    """numpy 向量 → BYTEA 存储格式（float32 原始字节）。"""
    return np.asarray(vec, dtype=np.float32).ravel().tobytes()


def bytes_to_emb(b: bytes) -> np.ndarray:
    """BYTEA → numpy 向量（float32）。"""
    return np.frombuffer(b, dtype=np.float32)


# ---------------- DB lifecycle ----------------


def clear_system_data() -> None:
    cfg = get_settings()
    log.warning("⚠️ CLEAR_CACHE_ON_START=True，正在清空本地缓存...")

    for p in [cfg.cache_path, cfg.output_path]:
        if p.exists():
            for item in p.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
    log.info("本地目录 (cache/, output/) 已清空")

    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE qa_pairs RESTART IDENTITY;")
        conn.commit()
        release_db_conn(conn)
        log.info("PostgreSQL 表 (qa_pairs) 已清空")
    except Exception as e:
        log.warning(f"清空数据库失败: {e}")

    invalidate_embedding_index()


def init_db() -> None:
    """
    初始化表结构（无 pgvector）。

    embedding 以 BYTEA 存储（float32 原始字节），应用层计算相似度。
    自动迁移：
      - 旧 vector 类型 → DROP 重建为 BYTEA（向量丢失，需重新 encode）
      - 旧 jsonb 类型  → 逐行转为 BYTEA（数据保留）
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS qa_pairs (
                    id SERIAL PRIMARY KEY,
                    call_id VARCHAR(64),
                    stage VARCHAR(64) NOT NULL,
                    primary_intent VARCHAR(64) NOT NULL,
                    tags JSONB,
                    risk VARCHAR(16),
                    session_ptp BOOLEAN,
                    session_outcomes JSONB,
                    rewritten_question TEXT NOT NULL,
                    rewritten_answer TEXT NOT NULL,
                    embedding BYTEA,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            # ---- embedding 列类型迁移 ----
            cur.execute(
                "SELECT udt_name FROM information_schema.columns "
                "WHERE table_name='qa_pairs' AND column_name='embedding';"
            )
            row = cur.fetchone()
            if row is None:
                cur.execute("ALTER TABLE qa_pairs ADD COLUMN embedding BYTEA;")
            elif row[0] == "vector":
                log.warning(
                    "embedding 列为旧 pgvector 类型，DROP 并重建为 BYTEA"
                    "（旧向量丢失，需重新 encode 后写入）"
                )
                cur.execute("ALTER TABLE qa_pairs DROP COLUMN embedding;")
                cur.execute("ALTER TABLE qa_pairs ADD COLUMN embedding BYTEA;")
            elif row[0] == "jsonb":
                log.info("embedding 列从 JSONB 迁移到 BYTEA ...")
                cur.execute("ALTER TABLE qa_pairs ADD COLUMN embedding_v2 BYTEA;")
                cur.execute(
                    "SELECT id, embedding FROM qa_pairs WHERE embedding IS NOT NULL"
                )
                migrate = cur.fetchall()
                for rid, payload in migrate:
                    arr = np.asarray(payload, dtype=np.float32)
                    cur.execute(
                        "UPDATE qa_pairs SET embedding_v2 = %s WHERE id = %s",
                        (arr.tobytes(), rid),
                    )
                cur.execute("ALTER TABLE qa_pairs DROP COLUMN embedding;")
                cur.execute(
                    "ALTER TABLE qa_pairs RENAME COLUMN embedding_v2 TO embedding;"
                )
                log.info(f"JSONB → BYTEA 迁移完成 ({len(migrate)} 行)")
            # bytea：无需处理

            # ---- session_ptp 兼容补列 ----
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
                "WHERE table_name='qa_pairs' AND column_name='session_ptp');"
            )
            if not cur.fetchone()[0]:
                cur.execute("ALTER TABLE qa_pairs ADD COLUMN session_ptp BOOLEAN;")

            # ---- 索引 ----
            cur.execute("DROP INDEX IF EXISTS idx_qa_emb;")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_qa_stage_intent "
                "ON qa_pairs (stage, primary_intent);"
            )

        conn.commit()
        log.info("数据库初始化完成（embedding 以 BYTEA 存储，应用层计算相似度）")
    finally:
        release_db_conn(conn)


# ---------------- 进程级 embedding 索引缓存 ----------------


class _EmbeddingIndex:
    """
    进程级 embedding 矩阵缓存，替代 pgvector 的 ANN 索引。

    设计要点：
      - 懒加载：首次访问时全量拉取，之后常驻内存
      - 预归一化：存储时即归一化，查询时矩阵乘法一次完成
      - 向量化掩码：stage/intent/risk 预存为 ndarray，过滤用 np.where
      - 线程安全：加载互斥（RLock），加载完成后读路径无锁
      - 写后失效：写入侧调用 invalidate_embedding_index()，下次读时重建

    内存估算：N × D × 4 字节。768 维时，10 万行 ≈ 300MB，100 万行 ≈ 3GB。
    数据量非常大时，建议改用专用向量库（Qdrant / Milvus / FAISS）。
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._loaded = False
        self._matrix: Optional[np.ndarray] = None  # [N, D] float32 已归一化
        self._ids: Optional[np.ndarray] = None  # [N] int64
        self._stages: List[str] = []
        self._intents: List[str] = []
        self._payloads: List[Tuple[str, str, str, str]] = []
        self._risk_ok: Optional[np.ndarray] = None  # [N] bool

    def invalidate(self) -> None:
        with self._lock:
            self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._load_locked()
            self._loaded = True

    def _load_locked(self) -> None:
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, stage, primary_intent, rewritten_question,
                           rewritten_answer, risk, embedding
                    FROM qa_pairs
                    WHERE embedding IS NOT NULL
                """)
                rows = cur.fetchall()
        finally:
            release_db_conn(conn)

        N = len(rows)
        if N == 0:
            self._matrix = np.empty((0, 0), dtype=np.float32)
            self._ids = np.empty((0,), dtype=np.int64)
            self._stages = []
            self._intents = []
            self._payloads = []
            self._risk_ok = np.empty((0,), dtype=bool)
            return

        # 探测维度 & 存储格式
        first = rows[0][6]
        if isinstance(first, (bytes, bytearray, memoryview)):
            dim = len(first) // 4
            is_bytes = True
        else:
            dim = len(first)
            is_bytes = False

        # 预分配，避免 vstack 类型推断开销
        mat = np.empty((N, dim), dtype=np.float32)
        ids = np.empty((N,), dtype=np.int64)
        stages: List[str] = []
        intents: List[str] = []
        payloads: List[Tuple[str, str, str, str]] = []
        risk_ok = np.empty((N,), dtype=bool)

        for i, r in enumerate(rows):
            v = r[6]
            if is_bytes:
                mat[i] = np.frombuffer(v, dtype=np.float32)
            else:
                mat[i] = np.asarray(v, dtype=np.float32)
            ids[i] = r[0]
            stages.append(r[1])
            intents.append(r[2])
            payloads.append((r[1], r[2], r[3], r[4]))
            risk_ok[i] = r[5] != "High"

        # 一次性归一化，零向量保护
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        mat /= norms

        self._matrix = mat
        self._ids = ids
        self._stages = stages
        self._intents = intents
        self._payloads = payloads
        self._risk_ok = risk_ok
        log.info(
            f"embedding 索引加载完成 | N={N} D={dim} "
            f"约 {mat.nbytes / 1024 / 1024:.1f} MB"
        )

    # ---- 掩码构造 ----

    def stage_mask(self, stages: List[str]) -> Optional[np.ndarray]:
        if not stages:
            return None
        self._ensure_loaded()
        wanted = set(stages)
        return np.fromiter(
            (s in wanted for s in self._stages),
            dtype=bool,
            count=len(self._stages),
        )

    def node_mask(self, nodes: Dict[Tuple[str, str], int]) -> Optional[np.ndarray]:
        if not nodes:
            return None
        self._ensure_loaded()
        wanted = set(nodes.keys())
        return np.fromiter(
            ((s, i) in wanted for s, i in zip(self._stages, self._intents)),
            dtype=bool,
            count=len(self._stages),
        )

    # ---- 查询 ----

    def query(
        self,
        emb: np.ndarray,
        mask: Optional[np.ndarray],
        top_k: int,
    ) -> List[Tuple[str, str, str, str, float]]:
        self._ensure_loaded()
        N = self._matrix.shape[0] if self._matrix is not None else 0
        if N == 0:
            return []

        q = np.asarray(emb, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(q))
        if norm == 0:
            return []
        q = q / norm

        sims = self._matrix @ q  # [N]

        effective = self._risk_ok if mask is None else (mask & self._risk_ok)
        valid = int(effective.sum())
        if valid == 0:
            return []

        k = min(max(1, int(top_k)), valid)

        if not effective.all():
            sims = np.where(effective, sims, -np.inf)

        if k < N:
            idx = np.argpartition(-sims, k - 1)[:k]
        else:
            idx = np.arange(N)
        idx = idx[np.argsort(-sims[idx])]

        out: List[Tuple[str, str, str, str, float]] = []
        for i in idx:
            s = float(sims[i])
            if not np.isfinite(s):
                break
            out.append((*self._payloads[i], s))
        return out


_index: Optional[_EmbeddingIndex] = None
_index_lock = threading.Lock()


def get_embedding_index() -> _EmbeddingIndex:
    global _index
    if _index is not None:
        return _index
    with _index_lock:
        if _index is None:
            _index = _EmbeddingIndex()
    return _index


def invalidate_embedding_index() -> None:
    """写入 / 删除 embedding 后调用，下次查询时自动重建。"""
    if _index is not None:
        _index.invalidate()


# ---------------- Embedder (remote TEI only) ----------------


class _BaseEmbedder:
    def encode(self, texts, normalize_embeddings: bool = True) -> np.ndarray:
        raise NotImplementedError

    def get_sentence_embedding_dimension(self) -> int:
        raise NotImplementedError

    def health_check(self) -> None:
        """发一次最小请求验证服务可用；失败抛异常。"""
        self.encode(["ping"], normalize_embeddings=False)


class _TEIEmbedder(_BaseEmbedder):
    def __init__(self, cfg):
        self._cfg = cfg
        self._model = cfg.embedding_model
        self._endpoint, self._style = self._resolve_endpoint(
            cfg.embedding_tei_url, cfg.embedding_tei_style
        )
        self._timeout = cfg.embedding_tei_timeout
        self._batch_size = max(1, cfg.embedding_tei_batch_size)
        self._concurrency = max(1, cfg.embedding_tei_concurrency)
        self._retry_max = max(1, cfg.embedding_tei_retry_max)
        self._retry_backoff = cfg.embedding_tei_retry_backoff
        self._truncate = cfg.embedding_tei_truncate
        self._dims: Optional[int] = None

        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=self._concurrency, pool_maxsize=self._concurrency
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        # ⚠️ 顺序关键：先建线程池，再探测维度
        # _probe_dims() 在 /info 失败时会走 encode() 探针，
        # encode() 需要 self._pool；否则会 AttributeError。
        self._pool = ThreadPoolExecutor(
            max_workers=self._concurrency, thread_name_prefix="emb-tei"
        )

        self._probe_dims()

        log.info(
            f"Embedding 客户端就绪 | style={self._style} endpoint={self._endpoint} "
            f"model={self._model} dims={self._dims} batch={self._batch_size} "
            f"concurrency={self._concurrency}"
        )

    @staticmethod
    def _resolve_endpoint(url: str, style: str) -> Tuple[str, str]:
        """
        返回 (完整 endpoint URL, 实际风格)。
        style 传入 auto 时按 URL 后缀判断；显式 native/openai 时补全路径。
        """
        u = (url or "").strip().rstrip("/")
        if not u:
            raise ValueError("EMBEDDING_TEI_URL 为空")

        # 显式指定风格：优先补全路径
        if style == "openai":
            if u.endswith("/v1/embeddings") or u.endswith("/embeddings"):
                return u, "openai"
            return f"{u}/v1/embeddings", "openai"

        if style == "native":
            if u.endswith("/embed"):
                return u, "native"
            return f"{u}/embed", "native"

        # auto：根据后缀判断
        if u.endswith("/v1/embeddings") or u.endswith("/embeddings"):
            return u, "openai"
        if u.endswith("/embed"):
            return u, "native"
        # base URL 默认走 native（TEI 官方部署最常见）
        return f"{u}/embed", "native"

    # ---------- 维度探测 ----------
    def _probe_dims(self) -> None:
        """
        探测维度：
          - native 风格优先 GET {base}/info（零推理成本）
          - 其余或 /info 不可用时，走一次真实 encode 探针
        探测失败不影响功能：get_sentence_embedding_dimension() 会再次尝试。
        """
        if self._style == "native":
            base = (
                self._endpoint[: -len("/embed")]
                if self._endpoint.endswith("/embed")
                else self._endpoint
            )
            try:
                r = self._session.get(f"{base}/info", timeout=self._timeout)
                if r.ok:
                    info = r.json()
                    for key in (
                        "dimensions",
                        "model_dims",
                        "hidden_size",
                        "dim",
                        "embedding_dimension",
                    ):
                        if key in info:
                            self._dims = int(info[key])
                            log.info(
                                f"Embedding dims={self._dims}（来自 /info, "
                                f"model={info.get('model_id')}）"
                            )
                            return
            except Exception as e:
                log.debug(f"/info 探测失败，将回退 probe embed: {e}")

        # 兜底：发一次真实 embed 探针
        try:
            v = self.encode("probe", normalize_embeddings=False)
            self._dims = int(np.asarray(v).shape[-1])
            log.info(f"Embedding dims={self._dims}（probe embed, style={self._style}）")
        except Exception as e:
            # 非致命：get_sentence_embedding_dimension() 会再次尝试
            log.debug(f"dim 探测失败（首次 encode 时会重试）: {e}")

    # ---------- HTTP 请求 ----------
    def _post_embed(self, batch: List[str]) -> List[List[float]]:
        last_exc: Optional[Exception] = None
        for attempt in range(self._retry_max):
            try:
                if self._style == "openai":
                    payload = {"input": batch, "model": self._model}
                else:
                    payload = {"inputs": batch, "truncate": self._truncate}

                r = self._session.post(
                    self._endpoint, json=payload, timeout=self._timeout
                )
                r.raise_for_status()
                body = r.json()

                if self._style == "openai":
                    items = sorted(
                        body.get("data", []),
                        key=lambda x: x.get("index", 0),
                    )
                    if len(items) != len(batch):
                        raise RuntimeError(
                            f"OpenAI embedding 返回条数不匹配: "
                            f"请求 {len(batch)} 条，返回 {len(items)} 条"
                        )
                    return [item["embedding"] for item in items]

                return body
            except Exception as e:
                last_exc = e
                if attempt < self._retry_max - 1:
                    wait = self._retry_backoff**attempt
                    log.debug(
                        f"embed 失败 (attempt {attempt + 1}/{self._retry_max})，"
                        f"{wait:.1f}s 后重试: {e}"
                    )
                    time.sleep(wait)
        assert last_exc is not None
        raise last_exc

    # ---------- encode ----------

    def encode(self, texts, normalize_embeddings: bool = True) -> np.ndarray:
        single = isinstance(texts, str)
        batch_texts = [texts] if single else list(texts)
        if not batch_texts:
            return np.empty((0, self._dims or 0), dtype=np.float32)

        bs = self._batch_size
        order = list(range(len(batch_texts)))
        if self._cfg.embedding_sort_by_length and len(batch_texts) > bs:
            order.sort(key=lambda i: len(batch_texts[i]))
        chunks = [order[i : i + bs] for i in range(0, len(order), bs)]

        results: List[Optional[List[float]]] = [None] * len(batch_texts)
        futures = [
            (
                chunk,
                self._pool.submit(self._post_embed, [batch_texts[i] for i in chunk]),
            )
            for chunk in chunks
        ]
        for chunk, fut in futures:
            for orig_idx, emb in zip(chunk, fut.result()):
                results[orig_idx] = emb

        arr = np.asarray(results, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=-1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            arr = arr / norms
        return arr[0] if single else arr

    def get_sentence_embedding_dimension(self) -> int:
        if self._dims is None:
            # 再试一次（可能是首次构造时 TEI 未就绪）
            self._probe_dims()
        if self._dims is None:
            v = self.encode("probe", normalize_embeddings=False)
            self._dims = int(np.asarray(v).shape[-1])
        return self._dims


def get_embedder() -> _BaseEmbedder:
    """
    仅支持远程 embedding 模式：
      - 服务可以是 TEI 原生 /embed 或任意 OpenAI 兼容 /v1/embeddings
      - 进程内不加载任何本地模型
    """
    global _embedder
    if _embedder is not None:
        return _embedder
    with _embedder_lock:
        if _embedder is None:
            cfg = get_settings()
            if not cfg.embedding_tei_url:
                raise ValueError(
                    "本版本只支持远程 embedding，必须设置 EMBEDDING_TEI_URL"
                )
            _embedder = _TEIEmbedder(cfg)
    return _embedder


# ---------------- 并发工具 ----------------


def retry(max_retries: int = 3, backoff: float = 2.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(backoff**attempt)

        return wrapper

    return decorator


def parallel_map(
    func: Callable, items: List[Any], max_workers: int = 8, desc: str = "Processing"
) -> List[Any]:
    if not items:
        return []
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, item): idx for idx, item in enumerate(items)}
        with tqdm(total=len(items), desc=desc) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result(timeout=120)
                except Exception as e:
                    log.error(f"[{desc}] 任务 {idx} 失败: {e}")
                finally:
                    pbar.update(1)
    return results


# ---------------- Dialog parsing ----------------


def _build_dialog_patterns(cfg) -> Tuple[re.Pattern, str]:
    role_pattern = "|".join(re.escape(r) for r in cfg.dialog_role_prefixes)
    turn_re = re.compile(rf"^({role_pattern}){cfg.dialog_colon_pattern}(.*)$")
    sep_class = "[" + "".join(re.escape(s) for s in cfg.dialog_turn_separators) + "]"
    return turn_re, sep_class


def _auto_detect_mode(text: str, cfg) -> str:
    if cfg.dialog_parse_mode != "auto":
        return cfg.dialog_parse_mode
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 2:
        return "separator"
    turn_re, _ = _build_dialog_patterns(cfg)
    matching = sum(1 for l in lines if turn_re.match(l))
    return "line" if matching >= len(lines) * 0.8 else "separator"


def _parse_separator(text: str, cfg) -> List[Dict]:
    turn_re, sep_class = _build_dialog_patterns(cfg)
    turns, turn_id = [], 0
    for part in re.split(sep_class, text):
        part = part.strip()
        if not part:
            continue
        m = turn_re.match(part)
        if not m:
            continue
        role = cfg.dialog_role_aliases.get(m.group(1), m.group(1))
        content = m.group(2).strip()
        if content:
            turns.append({"turn_id": turn_id, "role": role, "text": content})
            turn_id += 1
    return turns


def _parse_line(text: str, cfg) -> List[Dict]:
    turn_re, _ = _build_dialog_patterns(cfg)
    turns, turn_id = [], 0
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = turn_re.match(line)
        if m:
            role = cfg.dialog_role_aliases.get(m.group(1), m.group(1))
            content = m.group(2).strip()
            if content:
                turns.append({"turn_id": turn_id, "role": role, "text": content})
                turn_id += 1
        elif turns:
            turns[-1]["text"] += " " + line
    return turns


def parse_dialog(text: str, mode: Optional[str] = None, cfg=None) -> List[Dict]:
    cfg = cfg or get_settings()
    if not text:
        return []
    effective = mode or _auto_detect_mode(text, cfg)
    if effective == "line":
        return _parse_line(text, cfg)
    return _parse_separator(text, cfg)


def format_turns(turns: List[Dict], sep: str = ": ") -> str:
    return "\n".join([f"{t['role']}{sep}{t['text']}" for t in turns])


def merge_consecutive(turns: List[Dict], sep: str = "") -> List[Dict]:
    blocks: List[Dict] = []
    for idx, t in enumerate(turns):
        if blocks and blocks[-1]["role"] == t["role"]:
            blocks[-1]["text"] += sep + t["text"]
            blocks[-1]["end_idx"] = idx
        else:
            blocks.append(
                {"role": t["role"], "text": t["text"], "start_idx": idx, "end_idx": idx}
            )
    return blocks


def pair_turns(
    turns: List[Dict], sep: str = "", customer_role: Optional[str] = None
) -> List[Dict]:
    cfg = get_settings()
    customer_role = customer_role or cfg.dialog_customer_role
    blocks = merge_consecutive(turns, sep=sep)
    pairs: List[Dict] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if b["role"] != customer_role:
            i += 1
            continue
        if i + 1 < len(blocks) and blocks[i + 1]["role"] != customer_role:
            a = blocks[i + 1]
            pairs.append(
                {
                    "q_idx": (b["start_idx"], b["end_idx"]),
                    "a_idx": (a["start_idx"], a["end_idx"]),
                    "question": b["text"],
                    "answer": a["text"],
                }
            )
            i += 2
        else:
            pairs.append(
                {
                    "q_idx": (b["start_idx"], b["end_idx"]),
                    "a_idx": None,
                    "question": b["text"],
                    "answer": None,
                }
            )
            i += 1
    return pairs
