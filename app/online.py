import json, time, threading
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
import re
import app.config as raw_config
from app.settings import Settings, get_settings
from app.logger import get_logger
from app.utils import get_db_conn, release_db_conn, get_embedder, parse_dialog

_MIN_CUSTOMER_ROUNDS = 3  # 硬编码：目标窗口 = 最后 3 条客户话术
_SENT_END_RE = re.compile(r"[。！？；;!?\n]")


log = get_logger(__name__)


class HybridRetriever:
    """
    Two-branch hybrid retrieval:
      Branch A: pgvector with (stage, primary_intent) metadata filter
      Branch B: in-memory golden QA from sop_graph
    Merged with graph-hop decay; voted for stage & intent.

    Modes:
      off     — unrestricted pgvector only
      guide   — graph-aware steering, allows cross-stage jumps (max_hops)
      enforce — pins to hint/pass-1 stage (max_hops=0), hard-rejects on any failure
    """

    def __init__(self, sop_graph_path: str):
        self.cfg = get_settings()
        self.model = get_embedder()

        # Thread-safe embedding cache (double-checked locking in _encode)
        self._cache_lock = threading.Lock()
        self._emb_cache: Dict[str, np.ndarray] = {}

        with open(sop_graph_path, "r", encoding="utf-8") as f:
            self.sop_graph = json.load(f)

        self.allowed_intents: Dict[str, List[str]] = {}
        for stage, sd in self.sop_graph.get("nodes", {}).items():
            self.allowed_intents[stage] = list(sd.get("intents", {}).keys())
        for stage, intents in raw_config.SOP_FLOW.items():
            self.allowed_intents.setdefault(stage, list(intents))

        self.full_transition_map: Dict[Tuple[str, str], str] = {}
        self.transition_sources: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for edge in self.sop_graph.get("edges", []):
            src, tgt, intent = edge["source"], edge["target"], edge["trigger_intent"]
            self.full_transition_map[(src, intent)] = tgt
            self.transition_sources[src].append((intent, tgt))

        self.golden_vectors: Dict[Tuple[str, str], Dict] = {}
        self._precompute_golden_qa()

        log.info(
            f"HybridRetriever ready | stages={len(self.allowed_intents)} "
            f"edges={len(self.full_transition_map)} "
            f"golden_qa={sum(len(v['qa_pairs']) for v in self.golden_vectors.values())} "
            f"mode={self.cfg.state_machine_mode}"
        )

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
            f"Golden QA precomputed: {total} pairs in {(time.time() - t0) * 1000:.1f}ms"
        )

    # ---------- Thread-safe encode (double-checked locking) ----------

    def _encode(self, text: str) -> np.ndarray:
        # Fast path — atomic dict.get in CPython
        cached = self._emb_cache.get(text)
        if cached is not None:
            return cached

        # Encode without holding the lock — model.encode releases the GIL
        emb = self.model.encode(text, normalize_embeddings=True)

        # Double-check + write under lock
        with self._cache_lock:
            existing = self._emb_cache.get(text)
            if existing is not None:
                return existing
            if len(self._emb_cache) >= self.cfg.embed_cache_size:
                self._emb_cache.clear()
            self._emb_cache[text] = emb
            return emb

    def _hop_decay(self, hop: int) -> float:
        decay = self.cfg.hybrid_hop_decay
        return decay[hop] if 0 <= hop < len(decay) else decay[-1]

    # ---------- Graph BFS ----------

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

    # ---------- Branch A: pgvector (numpy → list conversion) ----------

    def _pgvector_hybrid(
        self, emb: np.ndarray, nodes: Dict[Tuple[str, str], int], top_k: int
    ) -> List[Tuple]:
        if not nodes:
            return []
        pairs = list(nodes.keys())
        placeholders = ",".join(["(%s,%s)"] * len(pairs))
        flat = [x for pair in pairs for x in pair]
        emb_list = emb.tolist()  # explicit conversion for psycopg2
        sql = f"""
            SELECT stage, primary_intent, rewritten_question, rewritten_answer,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM qa_pairs
            WHERE (stage, primary_intent) IN ({placeholders})
              AND risk != 'High'
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params = [emb_list] + flat + [emb_list, top_k]
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()
        finally:
            release_db_conn(conn)

    def _pgvector_stage_fallback(
        self, emb: np.ndarray, stages: List[str], top_k: int
    ) -> List[Tuple]:
        emb_list = emb.tolist()
        sql = """
            SELECT stage, primary_intent, rewritten_question, rewritten_answer,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM qa_pairs
            WHERE stage = ANY(%s) AND risk != 'High'
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (emb_list, stages, emb_list, top_k))
                return cur.fetchall()
        finally:
            release_db_conn(conn)

    def _pgvector_unrestricted(self, emb: np.ndarray, top_k: int) -> List[Tuple]:
        emb_list = emb.tolist()
        sql = """
            SELECT stage, primary_intent, rewritten_question, rewritten_answer,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM qa_pairs
            WHERE risk != 'High'
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (emb_list, emb_list, top_k))
                return cur.fetchall()
        finally:
            release_db_conn(conn)

    # ---------- Branch B: golden QA ----------

    def _golden_in_memory(
        self, emb: np.ndarray, nodes: Dict[Tuple[str, str], int], top_per_node: int
    ) -> List[Dict]:
        hits = []
        for (stage, intent), hop in nodes.items():
            gd = self.golden_vectors.get((stage, intent))
            if not gd:
                continue
            sims = gd["embeddings"] @ emb
            for i in np.argsort(-sims)[:top_per_node]:
                hits.append(
                    {
                        "stage": stage,
                        "intent": intent,
                        "question": gd["qa_pairs"][i]["question"],
                        "answer": gd["qa_pairs"][i]["answer"],
                        "raw_sim": float(sims[i]),
                        "hop": hop,
                    }
                )
        return hits

    # ---------- Merge / vote / format ----------

    def _merge(
        self,
        pg_rows: List[Tuple],
        golden_hits: List[Dict],
        nodes: Dict[Tuple[str, str], int],
    ) -> List[Dict]:
        merged = []
        for stage, intent, q, a, sim in pg_rows:
            if sim is None or np.isnan(sim):
                continue
            hop = nodes.get((stage, intent), 99)
            merged.append(
                {
                    "source": "pgvector",
                    "stage": stage,
                    "intent": intent,
                    "question": q,
                    "answer": a,
                    "raw_sim": float(sim),
                    "score": float(sim)
                    * self.cfg.hybrid_pg_weight
                    * self._hop_decay(hop),
                }
            )
        for h in golden_hits:
            merged.append(
                {
                    "source": "golden",
                    "stage": h["stage"],
                    "intent": h["intent"],
                    "question": h["question"],
                    "answer": h["answer"],
                    "raw_sim": h["raw_sim"],
                    "score": h["raw_sim"]
                    * self.cfg.hybrid_golden_weight
                    * self._hop_decay(h["hop"]),
                }
            )

        seen: Dict[Tuple[str, str, str], Dict] = {}
        for m in merged:
            key = (m["stage"], m["intent"], m["question"].strip())
            if key not in seen or m["score"] > seen[key]["score"]:
                seen[key] = m
        return sorted(seen.values(), key=lambda x: x["score"], reverse=True)

    def _vote(self, merged: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
        if not merged:
            return None, None
        top_n = self.cfg.hybrid_vote_top_n
        recent = merged[:top_n]

        # 先定 stage
        s_score: Dict[str, float] = defaultdict(float)
        for m in recent:
            s_score[m["stage"]] += m["score"] ** 2
        if not s_score:
            return None, None
        voted_stage = max(s_score, key=s_score.get)

        # 再在该 stage 内投 intent
        i_score: Dict[str, float] = defaultdict(float)
        for m in recent:
            if m["stage"] == voted_stage:
                i_score[m["intent"]] += m["score"] ** 2
        if not i_score:
            voted_intent = raw_config.SOP_FLOW[voted_stage][0]
        else:
            voted_intent = max(i_score, key=i_score.get)

        return voted_stage, voted_intent

    def _format(
        self,
        merged: List[Dict],
        stage: Optional[str],
        intent: Optional[str],
        t0: float,
        trace: Dict,
        source: str,
    ) -> Dict:
        golden_recs = [m for m in merged if m["source"] == "golden"][
            : self.cfg.online_top_k
        ]
        pg_recs = [m for m in merged if m["source"] == "pgvector"][
            : self.cfg.online_top_k
        ]
        recs = golden_recs or pg_recs
        fallback = pg_recs if golden_recs else []

        return {
            "stage": stage,
            "intent": intent,
            "recs": [
                {
                    "question": r["question"],
                    "answer": r["answer"],
                    "score": round(r["score"], 4),
                    "raw_sim": round(r["raw_sim"], 4), 
                    "source": r["source"],
                }
                for r in recs
            ],
            "fallback": [
                {
                    "question": r["question"],
                    "answer": r["answer"],
                    "score": round(r["score"], 4),
                    "raw_sim": round(r["raw_sim"], 4), 
                    "source": r["source"],
                }
                for r in fallback
            ],
            "reject": False,
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "debug": {
                **trace,
                "source": source,
                "merged": len(merged),
                "golden": len(golden_recs),
                "pg": len(pg_recs),
            },
        }

    def _empty(
        self, stage: Optional[str], reason: str, t0: float, trace: Optional[Dict] = None
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

    # ---------- Pass types ----------

    def _retrieve_off(self, query: str, t0: float, trace: Dict) -> Dict:
        trace["path"] = "off"
        trace["passes"] = 1
        emb = self._encode(query)
        pg_rows = self._pgvector_unrestricted(emb, self.cfg.hybrid_pg_top_k)
        if not pg_rows:
            return self._empty(None, "unrestricted 检索为空", t0, trace)
        merged = self._merge(pg_rows, [], {})
        voted_stage, voted_intent = self._vote(merged)
        if voted_stage is None:
            return self._empty(None, "vote 失败", t0, trace)
        return self._format(merged, voted_stage, voted_intent, t0, trace, "off")

    def _do_restricted(
        self,
        query: str,
        stage: str,
        nodes: Dict[Tuple[str, str], int],
        t0: float,
        trace: Dict,
        source: str,
        allow_fallback: bool = True,
        fallback_merged: Optional[List[Dict]] = None,
        fallback_stage: Optional[str] = None,
        fallback_intent: Optional[str] = None,
    ) -> Dict:
        """
        Restricted retrieval within `nodes`. Always re-encodes with `[stage]` prefix.
        When `allow_fallback=False` (enforce mode), never falls back to pass-1 results.
        """
        eq = f"[{stage}] {query}"
        emb = self._encode(eq)

        pg_rows = self._pgvector_hybrid(emb, nodes, self.cfg.hybrid_pg_top_k)
        if not pg_rows:
            stages = sorted({s for (s, _i) in nodes.keys()})
            if stages:
                pg_rows = self._pgvector_stage_fallback(
                    emb, stages, self.cfg.hybrid_pg_top_k
                )

        golden_hits = self._golden_in_memory(
            emb, nodes, self.cfg.hybrid_golden_top_k_per_node
        )
        merged = self._merge(pg_rows, golden_hits, nodes)

        if not merged:
            if allow_fallback and fallback_merged:
                log.warning(f"[{source}] restricted empty → fallback to pass 1")
                return self._format(
                    fallback_merged,
                    fallback_stage,
                    fallback_intent,
                    t0,
                    trace,
                    source + "-fallback",
                )
            return self._empty(stage, "候选节点检索为空", t0, trace)

        voted_stage, voted_intent = self._vote(merged)
        if voted_stage is None:
            if allow_fallback and fallback_merged:
                return self._format(
                    fallback_merged,
                    fallback_stage,
                    fallback_intent,
                    t0,
                    trace,
                    source + "-vote-fallback",
                )
            return self._empty(stage, "vote 失败", t0, trace)

        return self._format(merged, voted_stage, voted_intent, t0, trace, source)

    # ---------- Public entry ----------

    def retrieve_batch(self, query: str, stage_hint: Optional[str] = None) -> Dict:
        t0 = time.time()
        mode = self.cfg.state_machine_mode
        hint_short = stage_hint[:64] if stage_hint else None
        hint_valid = (stage_hint in self.allowed_intents) if stage_hint else None

        trace: Dict = {
            "mode": mode,
            "stage_hint": hint_short,
            "hint_valid": hint_valid,
            "passes": 0,
        }

        # ---- OFF ----
        if mode == "off":
            return self._retrieve_off(query, t0, trace)

        # ---- FAST PATH (hint given and valid) ----
        if hint_valid:
            trace["path"] = "hint"
            trace["passes"] = 1
            max_hops = 0 if mode == "enforce" else self.cfg.hybrid_max_hops
            nodes = self._candidate_nodes(stage_hint, max_hops)
            fast = self._do_restricted(
                query,
                stage_hint,
                nodes,
                t0,
                trace,
                "hint",
                allow_fallback=(mode != "enforce"),
            )
            if not fast.get("reject"):
                return fast

            # Enforce: hard reject on fast-path failure — do NOT try slow path
            if mode == "enforce":
                return self._empty(
                    stage_hint,
                    f"enforce 模式快速通道失败: {fast.get('reject_reason')}",
                    t0,
                    trace,
                )

            # Guide: fall through to slow path
            log.info(
                f"[retrieve_batch] fast path rejected ({fast.get('reject_reason')}), "
                f"falling back to slow path"
            )
            trace["fast_rejected"] = True
            trace["fast_reject_reason"] = fast.get("reject_reason")

        # Enforce: hint provided but invalid → hard reject
        if mode == "enforce" and stage_hint and not hint_valid:
            return self._empty(
                stage_hint,
                f"enforce 模式无效 stage_hint: {stage_hint}",
                t0,
                trace,
            )

        # ---- SLOW PATH ----
        trace["path"] = "infer"
        trace["passes"] = 1

        emb = self._encode(query)
        pg_rows = self._pgvector_unrestricted(emb, self.cfg.hybrid_pg_top_k)
        if not pg_rows:
            return self._empty(None, "unrestricted 检索为空", t0, trace)

        merged_1 = self._merge(pg_rows, [], {})
        voted_stage_1, voted_intent_1 = self._vote(merged_1)
        if voted_stage_1 is None:
            return self._empty(None, "第一轮 vote 失败", t0, trace)
        trace["stage_vote_1"] = voted_stage_1
        trace["intent_vote_1"] = voted_intent_1

        trace["passes"] = 2
        max_hops = 0 if mode == "enforce" else self.cfg.hybrid_max_hops
        nodes = self._candidate_nodes(voted_stage_1, max_hops)

        if mode == "enforce":
            # Enforce: pass 2 must resolve within stage_1, no pass-1 fallback
            return self._do_restricted(
                query,
                voted_stage_1,
                nodes,
                t0,
                trace,
                "infer-2pass-enforce",
                allow_fallback=False,
            )

        # Guide: pass 2 allows fallback to pass 1
        return self._do_restricted(
            query,
            voted_stage_1,
            nodes,
            t0,
            trace,
            "infer-2pass",
            allow_fallback=True,
            fallback_merged=merged_1,
            fallback_stage=voted_stage_1,
            fallback_intent=voted_intent_1,
        )


# ============================================================
#  Singleton
# ============================================================
import threading

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
                raise FileNotFoundError(...)
            _retriever = HybridRetriever(str(path))
    return _retriever


# ============================================================
#  Suggest API
# ============================================================


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
    """在句末标点处截断，返回带 '…' 的省略版本。"""
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
    """
    构造检索查询。

    规则：
      1. 客户话术 ≤ 3 条 → 全量返回
      2. 客户话术 > 3 条 → 窗口 = 最后 3 条客户话术 + 中间所有坐席（保持原顺序）
      3. 若渲染后 > suggest_max_query_chars：
         a. 从最早的 (坐席, 客户) 对话对开始整组丢弃
         b. 只剩 1 个 turn 仍超 → 句末截断该条文本
      4. 窗口起点向前对齐到坐席，保证 (坐席, 客户) 结构完整
    """
    cust_role = cfg.dialog_customer_role
    max_chars = cfg.suggest_max_query_chars
    if max_chars <= 0:
        return None

    # 触发点：最后一条客户话术
    last_customer_idx = None
    for i in range(len(turns) - 1, -1, -1):
        if turns[i]["role"] == cust_role:
            last_customer_idx = i
            break
    if last_customer_idx is None:
        return None

    # 窗口起点
    customer_indices = [i for i, t in enumerate(turns) if t["role"] == cust_role]
    if len(customer_indices) <= _MIN_CUSTOMER_ROUNDS:
        start_idx = 0
    else:
        start_idx = customer_indices[-_MIN_CUSTOMER_ROUNDS]
        if start_idx > 0 and turns[start_idx - 1]["role"] != cust_role:
            start_idx -= 1

    window = list(turns[start_idx : last_customer_idx + 1])

    def render(items: List[Dict]) -> str:
        return "\n".join(f"{t['role']}: {t['text']}" for t in items)

    # 超限：整组丢弃最早的对话对
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

    # 只剩 1 个 turn 仍超 → 句末截断
    t = window[0]
    prefix_len = len(t["role"]) + 2
    budget = max(50, max_chars - prefix_len)
    truncated = _truncate_on_sentence(t["text"], budget)
    return f"{t['role']}: {truncated}"[:max_chars], last_customer_idx


def suggest_script_for_conversation(
    text: str,
    stage_hint: Optional[str] = None,
    retriever: Optional[HybridRetriever] = None,
) -> Dict:
    """
    Stateless: full multi-turn conversation → best next utterance for 催收员.
    """
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
            for r in res["recs"][1:] + res.get("fallback", [])
        ],
        "reject": False,
        "reject_reason": None,
        "is_fallback": False,
        "turns": turns,
        "query": query,
        "latency_ms": e2e_ms,
        "debug": res.get("debug"),
    }


def suggest_script(
    text: str,
    stage_hint: Optional[str] = None,
) -> str:
    return (
        suggest_script_for_conversation(text, stage_hint=stage_hint).get("script") or ""
    )
