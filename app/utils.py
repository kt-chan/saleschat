import json, time, re, functools, os, shutil, queue, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Any, Optional, Dict, Tuple
from tqdm import tqdm
import numpy as np
import requests
import psycopg2
import pgembed
from psycopg2 import pool as pg_pool
from pgvector.psycopg2 import register_vector

from app.logger import get_logger
from app.settings import Settings, get_settings

log = get_logger(__name__)
_db_pool = None
_embedded_pg_server = None
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


# ---------------- DB ----------------

def get_db_pool():
    global _db_pool, _embedded_pg_server
    if _db_pool is None:
        cfg = get_settings()
        if cfg.use_embedded_db:
            try:
                pg_data_dir = str(Path(cfg.cache_dir) / "local_pgdata")
                os.makedirs(pg_data_dir, exist_ok=True)
                _embedded_pg_server = pgembed.get_server(pg_data_dir)
                dsn = _embedded_pg_server.get_uri()
                _db_pool = pg_pool.ThreadedConnectionPool(
                    cfg.db_pool_min, cfg.db_pool_max, dsn=dsn
                )
                log.info("本地嵌入式 PostgreSQL 启动成功")
                if not pgembed.has_extension("pgvector"):
                    log.error("pgembed-pgvector 未正确安装，向量检索将不可用")
                else:
                    log.info("pgvector 扩展已就绪")
            except Exception as e:
                log.error(f"本地数据库启动失败: {e}")
                raise
        else:
            _db_pool = pg_pool.ThreadedConnectionPool(
                cfg.db_pool_min, cfg.db_pool_max,
                host=cfg.db_host, port=cfg.db_port,
                dbname=cfg.db_name, user=cfg.db_user, password=cfg.db_password,
            )
            log.info("远程 Pgvector ThreadedConnectionPool 初始化完成")
    return _db_pool


def get_db_conn():
    global _db_pool
    if _db_pool is None:
        get_db_pool()
    conn = _db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
        try:
            register_vector(conn)
        except psycopg2.ProgrammingError as e:
            if "vector type not found" not in str(e):
                raise
        return conn
    except Exception as e:
        log.warning(f"数据库连接心跳异常，正在尝试重连: {e}")
        _db_pool.putconn(conn, close=True)
        conn = _db_pool.getconn()
        try:
            register_vector(conn)
        except Exception:
            pass
        return conn


def release_db_conn(conn):
    if _db_pool and conn:
        _db_pool.putconn(conn)


def clear_system_data():
    global _db_pool, _embedded_pg_server
    cfg = get_settings()
    log.warning("⚠️ CLEAR_CACHE_ON_START=True，正在清空本地缓存...")
    if cfg.use_embedded_db and _db_pool is not None:
        _db_pool.closeall()
        _db_pool = None
        _embedded_pg_server = None
        log.info("嵌入式数据库连接已释放")
    for p in [cfg.cache_path, cfg.output_path]:
        if p.exists():
            for item in p.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
    log.info("本地目录 (cache/, output/) 已彻底清空")
    if not cfg.use_embedded_db:
        try:
            conn = get_db_conn()
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE qa_pairs RESTART IDENTITY;")
            conn.commit()
            release_db_conn(conn)
            log.info("外部 Pgvector 数据库表 (qa_pairs) 已清空")
        except Exception as e:
            log.warning(f"清空外部数据库失败: {e}")


def init_db():
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        register_vector(conn)
        dim = get_embedder().get_sentence_embedding_dimension()
        with conn.cursor() as cur:
            cur.execute(f"""
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
                    embedding vector({dim}),
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
                "WHERE table_name='qa_pairs' AND column_name='session_ptp');"
            )
            if not cur.fetchone()[0]:
                cur.execute("ALTER TABLE qa_pairs ADD COLUMN session_ptp BOOLEAN;")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_qa_emb ON qa_pairs "
                "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_qa_stage_intent ON qa_pairs (stage, primary_intent);"
            )
        conn.commit()
        log.info(f"数据库平滑升级检查完成 (embedding dim={dim})")
    finally:
        release_db_conn(conn)


# ---------------- Embedder ----------------

class _BaseEmbedder:
    """本项目使用的最小 embedder 接口，与 SentenceTransformer 兼容。"""

    def encode(self, texts, normalize_embeddings: bool = True) -> np.ndarray:
        raise NotImplementedError

    def get_sentence_embedding_dimension(self) -> int:
        raise NotImplementedError


class _LocalEmbedder(_BaseEmbedder):
    """
    本地异构加速器 embedder：CUDA / Ascend NPU / CPU。

    设计要点：
      - 每设备一份模型副本（非互联卡的数据并行）。
      - ThreadPoolExecutor(max_workers=N) + device 令牌队列：
        每卡同一时刻至多一个 encode，杜绝显存/device 上下文竞争。
      - 长度排序分 chunk，减少 padding 浪费。
      - 单 chunk 级重试 + 手动 HF 推理兜底，防止 sentence-transformers
        在 NPU 上的 encode() 兼容性问题导致整批失败。
      - 设备切换统一使用 torch.accelerator.set_device_index（PyTorch ≥2.10），
        不再依赖 torch.cuda.device 或任何社区补丁。
    """

    def __init__(self, cfg: Settings):
        self._cfg = cfg
        self._backend = self._detect_backend(cfg)
        self._devices = self._resolve_devices(cfg, self._backend)
        self._num_devices = len(self._devices)

        log.info(f"Local embedder 初始化 | backend={self._backend} devices={self._devices}")

        t0 = time.time()
        self._models = []
        for d in self._devices:
            self._models.append(self._load_model(cfg.embedding_model, d))
        self._dims = self._models[0].get_sentence_embedding_dimension()
        log.info(
            f"模型加载完成 | devices={self._devices} dims={self._dims} "
            f"batch={cfg.embedding_batch_size} 耗时={time.time()-t0:.2f}s"
        )

        # 设备令牌队列：队列中有什么 = 哪些设备空闲
        self._device_queue: "queue.Queue[int]" = queue.Queue()
        for i in range(self._num_devices):
            self._device_queue.put(i)

        self._pool = ThreadPoolExecutor(
            max_workers=self._num_devices, thread_name_prefix="emb"
        )

    # ---------- 模型加载 ----------

    @staticmethod
    def _load_model(model_name: str, device: str):
        """
        加载 SentenceTransformer。

        若目标加速器原生加载失败，回退到 CPU 加载后**整体**迁移到目标设备。
        —— 必须用 model.to(device)，不能只移 auto_model：
           SentenceTransformer 内部包含 auto_model / pooling / normalize 等多个
           nn.Module，只移 auto_model 会导致前向时 pooling 参数在 CPU、
           token_embeddings 在 NPU，触发设备不匹配。
        """
        from sentence_transformers import SentenceTransformer
        try:
            return SentenceTransformer(model_name, device=device)
        except Exception as e:
            if device.startswith("npu") or device.startswith("cuda"):
                log.warning(
                    f"原生设备加载失败 ({e})，尝试 CPU 加载后整体迁移到 {device}"
                )
                model = SentenceTransformer(model_name, device="cpu")
                model.to(device)   # 递归迁移所有子模块
                return model
            raise

    # ---------- 后端探测与设备枚举 ----------

    @staticmethod
    def _detect_backend(cfg) -> str:
        forced = (cfg.embedding_backend or "auto").lower()
        if forced in ("cuda", "npu", "cpu"):
            return forced

        # auto: NPU 优先 → CUDA → CPU
        try:
            import torch_npu  # noqa: F401
            import torch
            if torch.npu.is_available():
                return "npu"
        except ImportError:
            pass
        import torch
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @classmethod
    def _resolve_devices(cls, cfg, backend: str) -> List[str]:
        if backend == "cpu":
            return ["cpu"]

        if backend == "npu":
            try:
                import torch_npu  # noqa: F401
                import torch
                total = torch.npu.device_count()
            except Exception:
                log.warning("NPU 后端已选但 torch_npu 不可用，降级 CPU")
                return ["cpu"]
            n = cfg.embedding_num_gpus if cfg.embedding_num_gpus > 0 else total
            n = max(1, min(n, total))
            return [f"npu:{i}" for i in range(n)]

        # cuda
        import torch
        if not torch.cuda.is_available():
            log.warning("CUDA 后端已选但不可用，降级 CPU")
            return ["cpu"]
        total = torch.cuda.device_count()
        n = cfg.embedding_num_gpus if cfg.embedding_num_gpus > 0 else total
        n = max(1, min(n, total))
        return [f"cuda:{i}" for i in range(n)]

    # ---------- 设备切换 ----------

    @staticmethod
    def _set_device(device_str: str):
        """
        在 worker 线程内设置当前设备。
        device 队列保证线程独占设备，因此直接 set_device_index 即可，无需上下文管理器。
        统一走 torch.accelerator，NPU / CUDA / CPU 均兼容。
        """
        if device_str == "cpu":
            return
        import torch
        idx = int(device_str.split(":")[1])
        try:
            torch.accelerator.set_device_index(idx)
        except Exception:
            # 兜底：按后端前缀直接调用
            if device_str.startswith("npu"):
                import torch_npu  # noqa: F401
                torch.npu.set_device(idx)
            elif device_str.startswith("cuda"):
                torch.cuda.set_device(idx)

    # ---------- 编码（含重试与手动兜底） ----------

    def _encode_with_retry(self, model, texts: List[str], device_str: str) -> np.ndarray:
        """
        单 chunk 编码，带重试。
        若 sentence-transformers 的 encode() 在 NPU 上失败，
        回退到手动 HuggingFace 前向 + pooling。
        """
        last_exc = None
        for attempt in range(self._cfg.embedding_retry_max):
            try:
                return model.encode(
                    texts,
                    batch_size=len(texts),
                    normalize_embeddings=False,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            except Exception as e:
                last_exc = e
                if attempt < self._cfg.embedding_retry_max - 1:
                    wait = self._cfg.embedding_retry_backoff ** attempt
                    log.warning(
                        f"[{device_str}] encode 失败 (attempt {attempt+1}/"
                        f"{self._cfg.embedding_retry_max})，{wait:.1f}s 后重试: {e}"
                    )
                    time.sleep(wait)

        # 重试耗尽 → 手动 HF 推理兜底
        log.warning(f"[{device_str}] encode 重试耗尽，切换手动 HF 推理兜底")
        return self._manual_hf_encode(model, texts, device_str)

    @staticmethod
    def _manual_hf_encode(model, texts: List[str], device_str: str) -> np.ndarray:
        """
        绕过 sentence-transformers 的 encode()，手动走 forward → pooling。
        —— 复用 model.pooling 子模块，保持与正常 encode 路径的 pooling 语义一致
           （bge-base 默认 CLS pooling，而非 mean pooling）。
        —— 归一化不在本函数内做，交由上层 encode(normalize_embeddings=True) 统一处理。
        """
        import torch

        # 定位底层 HF 模型与 tokenizer
        auto_model = None
        tokenizer = getattr(model, "tokenizer", None)
        for module in model.modules():
            if hasattr(module, "auto_model"):
                auto_model = module.auto_model
                break
        if auto_model is None or tokenizer is None:
            raise RuntimeError("无法定位底层 HF 模型/tokenizer，手动推理不可用")

        auto_model.eval()
        pooling = getattr(model, "pooling", None)

        with torch.no_grad():
            enc = tokenizer(
                texts, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            )
            enc = {k: v.to(device_str) for k, v in enc.items()}
            out = auto_model(**enc)
            token_embs = out.last_hidden_state        # [B, L, H]

            if pooling is not None:
                # 复用 ST 的 Pooling 模块（CLS / mean / max 由模型配置决定）
                pooled = pooling({
                    "token_embeddings": token_embs,
                    "attention_mask": enc["attention_mask"],
                })["sentence_embedding"]              # [B, H]
            else:
                # 兜底：attention_mask 加权 mean pooling
                mask = enc["attention_mask"].unsqueeze(-1)
                summed = (token_embs * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts

            return pooled.cpu().numpy()

    def _encode_chunk(
        self, indices: List[int], texts: List[str]
    ) -> List[Tuple[int, np.ndarray]]:
        """
        编码一个 chunk。从设备令牌队列借用一张设备，完成后归还。

        防御式设计：
          - dev_idx 初始化为 None，避免 get() 异常时 finally 引用未绑定变量。
          - queue.Empty 保留独立的明确消息，便于定位死锁。
          - 错误日志不引用 device_str，避免其在赋值前抛异常时
            触发 UnboundLocalError 掩盖原始异常。
        """
        dev_idx = None
        try:
            try:
                dev_idx = self._device_queue.get(
                    timeout=self._cfg.embedding_queue_timeout
                )
            except queue.Empty:
                raise RuntimeError(
                    f"获取设备令牌超时 ({self._cfg.embedding_queue_timeout}s)，"
                    f"可能存在 worker 泄漏或设备卡死"
                )
            model = self._models[dev_idx]
            device_str = self._devices[dev_idx]
            self._set_device(device_str)
            embs = self._encode_with_retry(model, texts, device_str)
            return list(zip(indices, embs))
        except Exception as e:
            # 刻意不引用 device_str：它可能在赋值前抛异常（如 _models 越界）
            log.error(f"chunk 编码失败 ({len(texts)} 条, dev_idx={dev_idx}): {e}")
            raise
        finally:
            if dev_idx is not None:
                self._device_queue.put(dev_idx)

    def encode(self, texts, normalize_embeddings: bool = True) -> np.ndarray:
        single = isinstance(texts, str)
        batch_texts = [texts] if single else list(texts)
        if not batch_texts:
            return np.empty((0, self._dims), dtype=np.float32)

        bs = max(1, self._cfg.embedding_batch_size)

        # 长度排序：长度相近的进入同一 chunk，减少 padding
        order = list(range(len(batch_texts)))
        if self._cfg.embedding_sort_by_length and len(batch_texts) > bs:
            order.sort(key=lambda i: len(batch_texts[i]))
        chunks = [order[i:i + bs] for i in range(0, len(order), bs)]

        # 并发编码
        results: List[Optional[np.ndarray]] = [None] * len(batch_texts)
        futures = [
            self._pool.submit(
                self._encode_chunk, chunk, [batch_texts[i] for i in chunk]
            )
            for chunk in chunks
        ]
        for fut in futures:
            for orig_idx, emb in fut.result():
                results[orig_idx] = emb

        arr = np.asarray(results, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=-1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            arr = arr / norms
        return arr[0] if single else arr

    def get_sentence_embedding_dimension(self) -> int:
        return self._dims


class _TEIEmbedder(_BaseEmbedder):
    """远程 TEI HTTP 服务 embedder。并发请求 + 客户端分块 + 重试。"""

    def __init__(self, cfg):
        self._cfg = cfg
        self._url = cfg.embedding_tei_url.rstrip("/")
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

        self._probe_dims()
        self._pool = ThreadPoolExecutor(
            max_workers=self._concurrency, thread_name_prefix="emb-tei"
        )
        log.info(
            f"TEI embedder ready | url={self._url} dims={self._dims} "
            f"batch={self._batch_size} concurrency={self._concurrency}"
        )

    def _probe_dims(self):
        try:
            r = self._session.get(f"{self._url}/info", timeout=self._timeout)
            r.raise_for_status()
            info = r.json()
            for key in ("dimensions", "model_dims", "hidden_size", "dim",
                        "embedding_dimension"):
                if key in info:
                    self._dims = int(info[key])
                    log.info(f"TEI /info dims={self._dims} (model={info.get('model_id')})")
                    return
        except Exception as e:
            log.warning(f"TEI /info 探测失败: {e}")

    def _post_embed(self, batch: List[str]) -> List[List[float]]:
        last_exc = None
        for attempt in range(self._retry_max):
            try:
                r = self._session.post(
                    f"{self._url}/embed",
                    json={"inputs": batch, "truncate": self._truncate},
                    timeout=self._timeout,
                )
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_exc = e
                if attempt < self._retry_max - 1:
                    time.sleep(self._retry_backoff ** attempt)
        raise last_exc  # type: ignore

    def encode(self, texts, normalize_embeddings: bool = True) -> np.ndarray:
        single = isinstance(texts, str)
        batch_texts = [texts] if single else list(texts)
        if not batch_texts:
            return np.empty((0, self._dims or 0), dtype=np.float32)

        bs = self._batch_size
        order = list(range(len(batch_texts)))
        if self._cfg.embedding_sort_by_length and len(batch_texts) > bs:
            order.sort(key=lambda i: len(batch_texts[i]))
        chunks = [order[i:i + bs] for i in range(0, len(order), bs)]

        results: List[Optional[List[float]]] = [None] * len(batch_texts)
        futures = [
            (chunk, self._pool.submit(self._post_embed,
                                      [batch_texts[i] for i in chunk]))
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
            v = self.encode("probe", normalize_embeddings=False)
            self._dims = int(np.asarray(v).shape[-1])
        return self._dims


def get_embedder() -> _BaseEmbedder:
    global _embedder
    if _embedder is not None:
        return _embedder
    with _embedder_lock:
        if _embedder is None:
            cfg = get_settings()
            mode = (cfg.embedding_mode or "local").lower()
            if mode == "tei":
                if not cfg.embedding_tei_url:
                    raise ValueError("EMBEDDING_MODE=tei 需要设置 EMBEDDING_TEI_URL")
                _embedder = _TEIEmbedder(cfg)
            elif mode == "local":
                _embedder = _LocalEmbedder(cfg)
            else:
                raise ValueError(f"未知 EMBEDDING_MODE={cfg.embedding_mode!r}")
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
                    time.sleep(backoff ** attempt)
        return wrapper
    return decorator


def parallel_map(func: Callable, items: List[Any], max_workers: int = 8,
                 desc: str = "Processing") -> List[Any]:
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
                {"role": t["role"], "text": t["text"],
                 "start_idx": idx, "end_idx": idx}
            )
    return blocks


def pair_turns(turns: List[Dict], sep: str = "",
               customer_role: Optional[str] = None) -> List[Dict]:
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
            pairs.append({
                "q_idx": (b["start_idx"], b["end_idx"]),
                "a_idx": (a["start_idx"], a["end_idx"]),
                "question": b["text"],
                "answer": a["text"],
            })
            i += 2
        else:
            pairs.append({
                "q_idx": (b["start_idx"], b["end_idx"]),
                "a_idx": None,
                "question": b["text"],
                "answer": None,
            })
            i += 1
    return pairs