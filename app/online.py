import json, time, threading
from typing import Dict, List, Optional, Tuple
import numpy as np
import re
import app.config as raw_config
from app.settings import Settings, get_settings
from app.logger import get_logger
from app.utils import get_db_conn, release_db_conn, get_embedder, parse_dialog

_SENT_END_RE = re.compile(r"[。！？；;!?\n]")

log = get_logger(__name__)


class HybridRetriever:
    """
    两段式检索（golden 优先 + pgvector 兜底）：

      ① 定位 (stage, intent)
          - 有 hint：直接用 hint（enforce 下必须匹配）
          - 无 hint：pgvector ORDER BY sim 取 top-1 的 (stage, intent)

      ② 在定位到的节点里查 golden QA
          - top1.sim >= τ → 直接返回（source="golden"）

      ③ golden 未命中 → pgvector 兜底
          - 有 hint：hint stage 邻域（+N hop）检索
          - 无 hint：pgvector 全库 top-k 直接返回

    模式：
      off     — 纯 pgvector，不查 golden
      guide   — golden 优先，未命中回退 pgvector
      enforce — 锁定 hint stage，golden 未命中时只查 hint stage 内的 pgvector
    """

    def __init__(self, sop_graph_path: str):
        self.cfg = get_settings()
        self.model = get_embedder()

        self._cache_lock = threading.Lock()
        self._emb_cache: Dict[str, np.ndarray] = {}

        with open(sop_graph_path, "r", encoding="utf-8") as f:
            self.sop_graph = json.load(f)

        self.allowed_intents: Dict[str, List[str]] = {}
        for stage, sd in self.sop_graph.get("nodes", {}).items():
            self.allowed_intents[stage] = list(sd.get("intents", {}).keys())
        for stage, intents in raw_config.SOP_FLOW.items():
            self.allowed_intents.setdefault(stage, list(intents))

        self.transition_sources: Dict[str, List[Tuple[str, str]]] = {}
        for edge in self.sop_graph.get("edges", []):
            src = edge["source"]
            self.transition_sources.setdefault(src, []).append(
                (edge["trigger_intent"], edge["target"])
            )

        self.golden_vectors: Dict[Tuple[str, str], Dict] = {}
        self._precompute_golden_qa()

        self.golden_min_sim_with_hint = float(
            getattr(self.cfg, "golden_min_sim_with_hint", 0.75)
        )
        self.golden_min_sim_no_hint = float(
            getattr(self.cfg, "golden_min_sim_no_hint", 0.70)
        )

        log.info(
            f"HybridRetriever ready | stages={len(self.allowed_intents)} "
            f"edges={sum(len(v) for v in self.transition_sources.values())} "
            f"golden_qa={sum(len(v['qa_pairs']) for v in self.golden_vectors.values())} "
            f"mode={self.cfg.state_machine_mode} "
            f"tau_hint={self.golden_min_sim_with_hint} "
            f"tau_nohint={self.golden_min_sim_no_hint}"
        )

    # ---------------- golden QA 预计算 ----------------

    def _precompute_golden_qa(self):
        t0 = time.time()
        total = 0
        for stage, sd in self.sop_graph.get("nodes", {}).items():
            for intent, nd in sd.get("intents", {}).items():
                if not nd:
                    continue
                qa_pairs = nd.get("golden_qa_pairs", []) or []
                if not qa_pairs:
                    continue
                embs = self.model.encode(
                    [q["question"] for q in qa_pairs], normalize_embeddings=True
                )
                self.golden_vectors[(stage, intent)] = {
                    "qa_pairs": qa_pairs,
                    "embeddings": embs,
                }
                total += len(qa_pairs)
        log.info(
            f"Golden QA precomputed: {total} pairs in "
            f"{(time.time() - t0) * 1000:.1f}ms"
        )

    # ---------------- 编码缓存 ----------------

    def _encode(self, text: str) -> np.ndarray:
        cached = self._emb_cache.get(text)
        if cached is not None:
            return cached
        emb = self.model.encode(text, normalize_embeddings=True)
        with self._cache_lock:
            existing = self._emb_cache.get(text)
            if existing is not None:
                return existing
            if len(self._emb_cache) >= self.cfg.embed_cache_size:
                self._emb_cache.clear()
            self._emb_cache[text] = emb
            return emb

    # ---------------- 图邻域 ----------------

    def _hop_decay(self, hop: int) -> float:
        decay = self.cfg.hybrid_hop_decay
        return decay[hop] if 0 <= hop < len(decay) else decay[-1]

    def _candidate_nodes(
        self, current_stage: str, max_hops: int
    ) -> Dict[Tuple[str, str], int]:
        result: Dict[Tuple[str, str], int] = {}
        frontier = [(current_stage, 0)]
        visited = {current_stage}
        while frontier:
            stage, hop = frontier.pop(0)
            for intent in self.allowed_intents.get(stage, []):
                key = (stage, intent)
                if key not in result or result[key] > hop:
                    result[key] = hop
            if hop >= max_hops:
                continue
            for _intent, tgt in self.transition_sources.get(stage, []):
                if tgt not in visited:
                    visited.add(tgt)
                    frontier.append((tgt, hop + 1))
        return result

    # ---------------- pgvector 查询 ----------------

    def _pgvector_hybrid(
        self,
        emb: np.ndarray,
        nodes: Dict[Tuple[str, str], int],
        top_k: int,
    ) -> List[Tuple]:
        if not nodes:
            return []
        pairs = list(nodes.keys())
        placeholders = ",".join(["(%s,%s)"] * len(pairs))
        params = [v for pair in pairs for v in pair]
        sql = f"""
            SELECT stage, primary_intent, rewritten_question, rewritten_answer,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM qa_pairs
            WHERE (stage, primary_intent) IN ({placeholders})
              AND risk != 'High'
              AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (emb, *params, emb, top_k))
                rows = cur.fetchall()
        finally:
            release_db_conn(conn)
        return [(s, i, q, a, float(sim)) for s, i, q, a, sim in rows]

    def _pgvector_stage_fallback(
        self,
        emb: np.ndarray,
        stages: List[str],
        top_k: int,
    ) -> List[Tuple]:
        if not stages:
            return []
        placeholders = ",".join(["%s"] * len(stages))
        sql = f"""
            SELECT stage, primary_intent, rewritten_question, rewritten_answer,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM qa_pairs
            WHERE stage IN ({placeholders})
              AND risk != 'High'
              AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (emb, *stages, emb, top_k))
                rows = cur.fetchall()
        finally:
            release_db_conn(conn)
        return [(s, i, q, a, float(sim)) for s, i, q, a, sim in rows]

    def _pgvector_unrestricted(self, emb: np.ndarray, top_k: int) -> List[Tuple]:
        sql = """
            SELECT stage, primary_intent, rewritten_question, rewritten_answer,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM qa_pairs
            WHERE risk != 'High'
              AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (emb, emb, top_k))
                rows = cur.fetchall()
        finally:
            release_db_conn(conn)
        return [(s, i, q, a, float(sim)) for s, i, q, a, sim in rows]

    # ---------------- golden 节点查询 ----------------

    def _golden_node_query(
        self,
        emb: np.ndarray,
        stage: str,
        intent: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Dict]:
        hits: List[Dict] = []

        if intent and (stage, intent) in self.golden_vectors:
            gd = self.golden_vectors[(stage, intent)]
            sims = gd["embeddings"] @ emb
            for i in np.argsort(-sims)[:top_k]:
                hits.append(
                    {
                        "stage": stage,
                        "intent": intent,
                        "question": gd["qa_pairs"][i]["question"],
                        "answer": gd["qa_pairs"][i]["answer"],
                        "score": float(sims[i]),
                        "raw_sim": float(sims[i]),
                        "source": "golden",
                    }
                )
            return hits

        for (s, i_), gd in self.golden_vectors.items():
            if s != stage:
                continue
            sims = gd["embeddings"] @ emb
            for j in np.argsort(-sims)[:1]:
                hits.append(
                    {
                        "stage": s,
                        "intent": i_,
                        "question": gd["qa_pairs"][j]["question"],
                        "answer": gd["qa_pairs"][j]["answer"],
                        "score": float(sims[j]),
                        "raw_sim": float(sims[j]),
                        "source": "golden",
                    }
                )
        hits.sort(key=lambda x: -x["raw_sim"])
        return hits[:top_k]

    # ---------------- 结果格式化 ----------------

    def _format_result(
        self,
        recs: List[Dict],
        stage: Optional[str],
        intent: Optional[str],
        source: str,
        t0: float,
        trace: Dict,
    ) -> Dict:
        return {
            "stage": stage,
            "intent": intent,
            "recs": recs[: self.cfg.online_top_k],
            "fallback": [],
            "reject": False,
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "debug": {**trace, "source": source, "n_recs": len(recs)},
        }

    def _empty(
        self,
        stage: Optional[str],
        reason: str,
        t0: float,
        trace: Optional[Dict] = None,
    ) -> Dict:
        return {
            "stage": stage,
            "intent": None,
            "recs": [],
            "fallback": [],
            "reject": True,
            "reject_reason": reason,
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "debug": {**(trace or {}), "mode": self.cfg.state_machine_mode},
        }

    def _pg_rows_to_recs(self, pg_rows: List[Tuple]) -> List[Dict]:
        return [
            {
                "question": r[2],
                "answer": r[3],
                "score": float(r[4]),
                "raw_sim": float(r[4]),
                "source": "pgvector",
            }
            for r in pg_rows
        ]

    # ---------------- 主入口 ----------------

    def retrieve_batch(self, query: str, stage_hint: Optional[str] = None) -> Dict:
        t0 = time.time()
        mode = self.cfg.state_machine_mode
        hint_valid = (stage_hint in self.allowed_intents) if stage_hint else None
        trace: Dict = {
            "mode": mode,
            "stage_hint": stage_hint,
            "hint_valid": hint_valid,
        }

        if mode == "off":
            return self._retrieve_off(query, t0, trace)

        emb = self._encode(query)  # 单次 TEI，无前缀

        if hint_valid:
            return self._retrieve_with_hint(emb, stage_hint, t0, trace)

        if mode == "enforce" and stage_hint and not hint_valid:
            return self._empty(
                stage_hint, f"enforce 无效 stage_hint: {stage_hint}", t0, trace
            )

        return self._retrieve_without_hint(emb, t0, trace)

    # ---------------- 有 hint ----------------

    def _retrieve_with_hint(
        self, emb: np.ndarray, stage_hint: str, t0: float, trace: Dict
    ) -> Dict:
        trace["path"] = "hint"

        golden = self._golden_node_query(
            emb, stage_hint, intent=None, top_k=self.cfg.online_top_k
        )
        if golden:
            trace["golden_sim"] = round(golden[0]["raw_sim"], 4)
        if golden and golden[0]["raw_sim"] >= self.golden_min_sim_with_hint:
            trace["sub_path"] = "golden"
            return self._format_result(
                golden, stage_hint, golden[0]["intent"], "golden", t0, trace
            )

        max_hops = 0 if trace["mode"] == "enforce" else self.cfg.hybrid_max_hops
        nodes = self._candidate_nodes(stage_hint, max_hops)
        pg_rows = self._pgvector_hybrid(emb, nodes, self.cfg.hybrid_pg_top_k)

        if not pg_rows:
            stages = sorted({s for (s, _i) in nodes.keys()})
            if stages:
                pg_rows = self._pgvector_stage_fallback(
                    emb, stages, self.cfg.hybrid_pg_top_k
                )

        if not pg_rows:
            return self._empty(
                stage_hint, "hint 邻域与 stage 级回退均无结果", t0, trace
            )

        top_intent = pg_rows[0][1]
        trace["sub_path"] = "pgvector"
        trace["pg_top_sim"] = round(float(pg_rows[0][4]), 4)
        return self._format_result(
            self._pg_rows_to_recs(pg_rows),
            stage_hint,
            top_intent,
            "pgvector",
            t0,
            trace,
        )

    # ---------------- 无 hint ----------------

    def _retrieve_without_hint(self, emb: np.ndarray, t0: float, trace: Dict) -> Dict:
        trace["path"] = "infer"

        pg_rows = self._pgvector_unrestricted(emb, self.cfg.hybrid_pg_top_k)
        if not pg_rows:
            return self._empty(None, "全库检索为空", t0, trace)

        top = pg_rows[0]
        voted_stage, voted_intent, top_sim = top[0], top[1], float(top[4])
        top2_sim = float(pg_rows[1][4]) if len(pg_rows) > 1 else 0.0
        margin = top_sim - top2_sim

        trace.update(
            {
                "stage": voted_stage,
                "intent": voted_intent,
                "top_sim": round(top_sim, 4),
                "margin": round(margin, 4),
            }
        )

        golden = self._golden_node_query(
            emb, voted_stage, intent=voted_intent, top_k=self.cfg.online_top_k
        )
        if golden:
            trace["golden_sim"] = round(golden[0]["raw_sim"], 4)
        if golden and golden[0]["raw_sim"] >= self.golden_min_sim_no_hint:
            trace["sub_path"] = "golden"
            return self._format_result(
                golden, voted_stage, voted_intent, "golden", t0, trace
            )

        trace["sub_path"] = "pgvector"
        return self._format_result(
            self._pg_rows_to_recs(pg_rows),
            voted_stage,
            voted_intent,
            "pgvector",
            t0,
            trace,
        )

    # ---------------- off 模式 ----------------

    def _retrieve_off(self, query: str, t0: float, trace: Dict) -> Dict:
        trace["path"] = "off"
        emb = self._encode(query)
        pg_rows = self._pgvector_unrestricted(emb, self.cfg.hybrid_pg_top_k)
        if not pg_rows:
            return self._empty(None, "unrestricted 检索为空", t0, trace)
        top = pg_rows[0]
        trace["top_sim"] = round(float(top[4]), 4)
        return self._format_result(
            self._pg_rows_to_recs(pg_rows),
            top[0],
            top[1],
            "off",
            t0,
            trace,
        )


# ---------------- 单例 ----------------

_retriever_lock = threading.Lock()
_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is not None:
        return _retriever
    with _retriever_lock:
        if _retriever is None:
            path = get_settings().sop_graph_file
            if not path.exists():
                raise FileNotFoundError(
                    f"SOP 图谱未找到 ({path})。请先执行 `python main.py --mode offline`"
                )
            _retriever = HybridRetriever(str(path))
    return _retriever


# ---------------- 外层：对话解析 + query 构造 ----------------


def _empty_result(
    reason: str,
    turns: Optional[List[Dict]] = None,
    query: str = "",
    debug: Optional[Dict] = None,
    latency_ms: Optional[float] = None,
) -> Dict:
    cfg = get_settings()
    return {
        "script": cfg.suggest_fallback_script,
        "question": "",
        "stage": None,
        "intent": None,
        "source": None,
        "score": 0.0,
        "alternates": [],
        "reject": True,
        "reject_reason": reason,
        "is_fallback": True,
        "turns": turns or [],
        "query": query,
        "latency_ms": latency_ms,
        "debug": debug,
    }


def _truncate_on_sentence(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    ell = "…"
    cut = text[: max(0, max_chars - len(ell))]
    matches = list(_SENT_END_RE.finditer(cut))
    if matches:
        cut = cut[: matches[-1].end()]
    return cut + ell


def _build_suggest_query(turns: List[Dict], cfg) -> Optional[Tuple[str, int]]:
    cust_role = cfg.dialog_customer_role
    max_chars = cfg.suggest_max_query_chars
    max_rounds = max(1, cfg.suggest_context_customer_turns)
    if max_chars <= 0:
        return None

    last_customer_idx = None
    for i in range(len(turns) - 1, -1, -1):
        if turns[i]["role"] == cust_role:
            last_customer_idx = i
            break
    if last_customer_idx is None:
        return None

    customer_indices = [i for i, t in enumerate(turns) if t["role"] == cust_role]
    if len(customer_indices) <= max_rounds:
        start_idx = 0
    else:
        start_idx = customer_indices[-max_rounds]
        if start_idx > 0 and turns[start_idx - 1]["role"] != cust_role:
            start_idx -= 1

    window = list(turns[start_idx : last_customer_idx + 1])

    def render(items: List[Dict]) -> str:
        return "\n".join(f"{t['role']}: {t['text']}" for t in items)

    while len(window) > 1 and len(render(window)) > max_chars:
        if (
            len(window) >= 2
            and window[0]["role"] != cust_role
            and window[1]["role"] == cust_role
        ):
            window = window[2:]
        else:
            window = window[1:]

    query = render(window)
    if len(query) <= max_chars:
        return query, last_customer_idx

    t = window[0]
    budget = max(50, max_chars - (len(t["role"]) + 2))
    truncated = _truncate_on_sentence(t["text"], budget)
    return f"{t['role']}: {truncated}"[:max_chars], last_customer_idx


def suggest_script_for_conversation(
    text: str,
    stage_hint: Optional[str] = None,
    retriever: Optional[HybridRetriever] = None,
) -> Dict:
    t_start = time.time()
    cfg = get_settings()

    turns = parse_dialog(text)
    if not turns:
        return _empty_result(
            "无法解析对话文本",
            latency_ms=round((time.time() - t_start) * 1000, 1),
        )

    built = _build_suggest_query(turns, cfg)
    if built is None:
        return _empty_result(
            "对话中无客户发言",
            turns=turns,
            latency_ms=round((time.time() - t_start) * 1000, 1),
        )
    query, _ = built

    ret = retriever or get_retriever()
    res = ret.retrieve_batch(query, stage_hint=stage_hint)
    e2e_ms = round((time.time() - t_start) * 1000, 1)

    if res.get("reject") or not res.get("recs"):
        return _empty_result(
            res.get("reject_reason") or "无可用推荐",
            turns=turns,
            query=query,
            debug=res.get("debug"),
            latency_ms=e2e_ms,
        )

    top = res["recs"][0]
    return {
        "script": top["answer"],
        "question": top["question"],
        "stage": res.get("stage"),
        "intent": res.get("intent"),
        "source": top["source"],
        "score": top["score"],
        "alternates": [
            {
                "question": r["question"],
                "answer": r["answer"],
                "source": r["source"],
                "score": r["score"],
            }
            for r in res["recs"][1:]
        ],
        "reject": False,
        "reject_reason": None,
        "is_fallback": False,
        "turns": turns,
        "query": query,
        "latency_ms": e2e_ms,
        "debug": res.get("debug"),
    }


def suggest_script(text: str, stage_hint: Optional[str] = None) -> str:
    return (
        suggest_script_for_conversation(text, stage_hint=stage_hint).get("script") or ""
    )
