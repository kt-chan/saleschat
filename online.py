import json, time
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer

import config as raw_config
from settings import get_settings
from logger import get_logger
from utils import get_db_conn, release_db_conn, SessionStore

log = get_logger(__name__)


class PgvectorRetriever:
    def __init__(self, sop_graph_path: str):
        self.cfg = get_settings()
        self.model = SentenceTransformer(
            self.cfg.embedding_model, device=self.cfg.embedding_device
        )

        with open(sop_graph_path, "r", encoding="utf-8") as f:
            self.sop_graph = json.load(f)

        self.transition_map = defaultdict(set)
        for edge in self.sop_graph["edges"]:
            self.transition_map[edge["source"]].add(edge["target"])

        self._emb_cache = {}

        self.golden_vectors = {}
        self._precompute_golden_qa()
        log.info("在线实时检索引擎已就绪 (内存矩阵点积重排)")

    def _precompute_golden_qa(self):
        t0 = time.time()
        total_qs = 0
        for stage, stage_data in self.sop_graph.get("nodes", {}).items():
            for intent, intent_data in stage_data.get("intents", {}).items():
                qa_pairs = intent_data.get("golden_qa_pairs", []) if intent_data else []
                if not qa_pairs:
                    continue

                questions = [qa["question"] for qa in qa_pairs]
                embs = self.model.encode(questions, normalize_embeddings=True)

                self.golden_vectors[(stage, intent)] = {
                    "qa_pairs": qa_pairs,
                    "embeddings": embs,
                }
                total_qs += len(qa_pairs)
        log.info(
            f"Golden QA 预计算完成: 共加载 {total_qs} 条经典话术，耗时 {(time.time()-t0)*1000:.1f}ms"
        )

    def _encode(self, text: str):
        if text not in self._emb_cache:
            if len(self._emb_cache) > 2000:
                self._emb_cache.clear()
            self._emb_cache[text] = self.model.encode(text, normalize_embeddings=True)
        return self._emb_cache[text]

    def _search_similar(self, query: str) -> List[Tuple]:
        emb = self._encode(query)
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT stage, primary_intent, rewritten_question, rewritten_answer, 1 - (embedding <=> %s::vector) AS similarity
                    FROM qa_pairs WHERE risk != 'High' ORDER BY embedding <=> %s::vector LIMIT %s
                """,
                    (emb, emb, self.cfg.online_recall_top_k * 2),
                )
                coarse = cur.fetchall()
                if not coarse:
                    return []

                enhanced_query = f"[{coarse[0][0]}][{coarse[0][1]}] {query}"
                emb_enhanced = self._encode(enhanced_query)
                cur.execute(
                    """
                    SELECT stage, primary_intent, rewritten_question, rewritten_answer, 1 - (embedding <=> %s::vector) AS similarity
                    FROM qa_pairs WHERE risk != 'High' ORDER BY embedding <=> %s::vector LIMIT %s
                """,
                    (emb_enhanced, emb_enhanced, self.cfg.online_recall_top_k),
                )
                return cur.fetchall()
        finally:
            release_db_conn(conn)

    def _vote_labels(self, rows: List[Tuple]) -> Tuple[Optional[str], Optional[str]]:
        if not rows:
            return None, None
        top = rows[0]
        if top[4] is None or np.isnan(top[4]):
            return None, None
        if top[4] >= 0.68 or (len(rows) >= 2 and (top[4] - rows[1][3]) >= 0.03):
            return top[0], top[1]

        s_score, i_score = defaultdict(float), defaultdict(float)
        for r in rows[:4]:
            if r[4] is not None and not np.isnan(r[4]):
                s_score[r[0]] += r[4] ** 2
                i_score[r[1]] += r[4] ** 2
        if not s_score:
            return None, None
        return max(s_score, key=s_score.get), max(i_score, key=i_score.get)

    def retrieve(self, query: str, current_stage: str) -> Dict:
        t0 = time.time()
        rows = self._search_similar(query)

        # 如果开启状态机，且没查到结果，则拒答
        if not rows and self.cfg.enable_state_machine:
            return self._reject(current_stage, "无检索结果", t0)

        if not rows:
            return {
                "stage": current_stage,
                "intent": None,
                "recs": [],
                "fallback": [{"question": "", "answer": "未找到相关话术", "score": 0}],
                "reject": False,
                "latency_ms": round((time.time() - t0) * 1000, 1),
            }

        top_sim = rows[0][4]

        # ★ 只有开启状态机时，才做严格的相似度拒答
        if self.cfg.enable_state_machine:
            if (
                top_sim is None
                or np.isnan(top_sim)
                or top_sim < self.cfg.online_sim_threshold
            ):
                return self._reject(current_stage, f"相似度不足({top_sim:.3f})", t0)

        pred_stage, intent = self._vote_labels(rows)
        if not pred_stage:
            pred_stage = current_stage
        if not intent:
            intent = "其他"

        final_stage = current_stage

        # ★ 状态机拦截逻辑
        if self.cfg.enable_state_machine:
            if pred_stage != current_stage:
                if pred_stage in self.transition_map.get(current_stage, set()):
                    final_stage = pred_stage
                else:
                    allowed_intents = raw_config.SOP_FLOW.get(current_stage, [])
                    if intent not in allowed_intents:
                        return self._reject(
                            current_stage, f"非法跨阶段意图({intent})拦截", t0
                        )
                    final_stage = current_stage
        else:
            # 状态机关闭，无条件相信检索结果
            final_stage = pred_stage

        recs = []
        golden_data = self.golden_vectors.get((final_stage, intent))

        if golden_data:
            qa_pairs = golden_data["qa_pairs"]
            qa_embs = golden_data["embeddings"]
            q_emb = self._encode(query)
            sims = qa_embs @ q_emb

            for i, score in enumerate(sims):
                # 状态机关闭时，放宽阈值（比如 0.3），确保 Demo 一定有输出
                min_score = (
                    self.cfg.online_sim_threshold
                    if self.cfg.enable_state_machine
                    else 0.30
                )
                if score >= min_score:
                    recs.append(
                        {
                            "question": qa_pairs[i]["question"],
                            "answer": qa_pairs[i]["answer"],
                            "score": round(float(score), 4),
                        }
                    )
            recs.sort(key=lambda x: x["score"], reverse=True)

        fallback = []
        if not recs:
            # 兜底逻辑
            fallback = [
                {
                    "question": "标准库匹配",
                    "answer": r[3],
                    "score": round(float(r[4]), 4),
                }
                for r in rows
                if r[0] == final_stage
            ][:2]
            if not fallback:
                fallback = [
                    {
                        "question": "全库兜底",
                        "answer": rows[0][3],
                        "score": round(float(rows[0][4]), 4),
                    }
                ]

            if not fallback and self.cfg.enable_state_machine:
                return self._reject(final_stage, "图谱无话术且兜底失效", t0)

        latency = (time.time() - t0) * 1000
        return {
            "stage": final_stage,
            "intent": intent,
            "recs": recs[: self.cfg.online_top_k],
            "fallback": fallback,
            "reject": False,
            "latency_ms": round(latency, 1),
        }

    def _reject(self, stage: str, reason: str, t0: float) -> Dict:
        return {
            "stage": stage,
            "intent": None,
            "recs": [],
            "fallback": [],
            "reject": True,
            "reject_reason": reason,
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }


class RealtimeRecommender:
    def __init__(self):
        self.retriever = PgvectorRetriever(str(get_settings().sop_graph_file))
        self.store = SessionStore()

    def on_asr(self, sid: str, role: str, text: str):
        """标准多轮实时推理接口（包含状态机限制）"""
        state = self.store.get(sid)
        if not state.get("stage"):
            state["stage"] = raw_config.SOP_STAGES[0]
        if "history" not in state:
            state["history"] = []

        state["history"].append({"role": role, "text": text.strip()})

        if role != "客户":
            self.store.set(sid, state)
            return None

        cust_texts = [h["text"] for h in state["history"] if h["role"] == "客户"]
        recent = cust_texts[-self.retriever.cfg.online_context_window :]
        query = " ".join(recent)

        res = self.retriever.retrieve(query, state["stage"])
        state["stage"] = res["stage"]
        self.store.set(sid, state)
        return res
