# online.py
"""
在线层：
1. 状态跟踪（StateTracker）
2. 融合检索（FusionRetriever）
3. 实时推荐（RealtimeRecommender）
4. 测试集构建（build_testset / save_testset）
"""
import re
import json
import time
import random
from pathlib import Path
from typing import List, Dict

import jieba
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

from settings import get_settings
from logger import get_logger
from offline import get_runtime_config

log = get_logger(__name__)


# ============================================================
# 1. 状态跟踪
# ============================================================
class StateTracker:
    """规则状态机（O(1) 跳转）"""

    def __init__(self, graph: Dict):
        self.rcfg = get_runtime_config()
        self.transition = self._build_transition(graph)
        self.sessions: Dict[str, Dict] = {}

    def _build_transition(self, graph: Dict) -> Dict:
        t = {}
        for e in graph.get("edges", []):
            t[(e["source"], e["intent"])] = e["target"]
        for key, tgt in self.rcfg.transition_table.items():
            if key not in t:
                t[key] = tgt
        log.debug(f"跳转表: {len(t)} 条")
        return t

    def init_session(self, session_id: str):
        self.sessions[session_id] = {
            "phase": "身份核实",
            "intents": [], "emotions": [],
            "history": [], "ptp": None,
        }
        log.debug(f"会话初始化: {session_id}")

    def update(self, session_id: str, role: str, text: str) -> Dict:
        state = self.sessions.setdefault(session_id, {
            "phase": "身份核实", "intents": [], "emotions": [],
            "history": [], "ptp": None,
        })
        state["history"].append({"role": role, "text": text})

        if role == "客户":
            intents = self._match_intents(text)
            emotions = self._match_emotions(text)
            state["intents"] = list(set(state["intents"] + intents))
            state["emotions"] = list(set(state["emotions"] + emotions))

            for intent in intents:
                key = (state["phase"], intent)
                if key in self.transition:
                    state["phase"] = self.transition[key]
                    break

            if "承诺还款" in intents:
                state["ptp"] = {"time": self._extract_time(text), "raw": text}
        return state

    def _match_intents(self, text: str) -> List[str]:
        matched = []
        for intent, kws in self.rcfg.intent_keywords.items():
            if any(kw in text for kw in kws):
                matched.append(intent)
        return matched

    def _match_emotions(self, text: str) -> List[str]:
        matched = []
        for emo, kws in self.rcfg.emotion_keywords.items():
            if any(kw in text for kw in kws):
                matched.append(emo)
        return matched

    def _extract_time(self, text: str) -> str:
        patterns = [
            r"(今天|明天|后天|这周|下周|本月|下个月)",
            r"(周[一二三四五六日末])",
            r"(\d{1,2}[月号日])",
            r"(\d{1,2}[点:：]\d{0,2})",
        ]
        for p in patterns:
            m = re.search(p, text)
            if m:
                return m.group(1)
        return ""


# ============================================================
# 2. 融合检索
# ============================================================
class FusionRetriever:
    """多路融合检索（主题/向量/标签/图谱）"""

    def __init__(self, bertopic_dir: Path, faiss_dir: Path, graph_path: Path):
        self.cfg = get_settings()
        t0 = time.time()

        log.info("加载 BERTopic...")
        self.topic_model = BERTopic.load(str(bertopic_dir))

        log.info("加载 Faiss 索引...")
        self.q_index = faiss.read_index(str(faiss_dir / "question.faiss"))
        self.l_index = faiss.read_index(str(faiss_dir / "label.faiss"))
        with open(faiss_dir / "qa_meta.json", "r", encoding="utf-8") as f:
            self.qa_meta = json.load(f)

        log.info("加载 SOP 图谱...")
        with open(graph_path, "r", encoding="utf-8") as f:
            self.graph = json.load(f)

        log.info("加载 Embedding 模型...")
        self.model = SentenceTransformer(
            self.cfg.embedding_model, device=self.cfg.embedding_device)
        self._emb_cache: Dict[str, np.ndarray] = {}

        log.info(f"检索器初始化完成: QA={len(self.qa_meta)}, "
                 f"耗时={time.time() - t0:.1f}s")

    def _encode_cached(self, text: str) -> np.ndarray:
        if text not in self._emb_cache:
            if len(self._emb_cache) > self.cfg.online_embedding_cache_size:
                self._emb_cache.clear()
            vec = self.model.encode(
                [text], normalize_embeddings=True,
                max_length=self.cfg.embedding_max_length,
            ).astype(np.float32)
            self._emb_cache[text] = vec
        return self._emb_cache[text]

    # ---------- 四路召回 ----------
    def recall_topic(self, text: str) -> List[Dict]:
        """
        BERTopic 主题召回
        增加多重防御：
        1. 输入校验（空、纯符号、过短）
        2. transform 异常捕获
        3. 返回主题有效性校验
        """
        # ---------- 1. 输入校验 ----------
        if not text or not text.strip():
            return []

        # 去掉纯标点/数字后，要求至少有 2 个中文字符或 1 个有效词
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', '', text)
        if len(cleaned) < 2:
            log.debug(f"[topic] 输入过短，跳过: {text!r}")
            return []

        # ---------- 2. 分词 + 长度校验 ----------
        try:
            words_list = [w for w in jieba.cut(text) if w.strip()]
        except Exception as e:
            log.warning(f"[topic] 分词失败: {e}, text={text!r}")
            return []

        if len(words_list) < 2:
            log.debug(f"[topic] 分词结果过短，跳过: {words_list}")
            return []

        words = " ".join(words_list)

        # ---------- 3. transform 异常捕获 ----------
        try:
            topics, _ = self.topic_model.transform([words])
        except Exception as e:
            log.warning(f"[topic] transform 失败: {e}, text={text!r}")
            return []

        if not topics or len(topics) == 0:
            log.debug(f"[topic] transform 未返回主题: {text!r}")
            return []

        tid = topics[0]
        if tid == -1:
            log.debug(f"[topic] 命中噪声主题 (-1): {text!r}")
            return []

        # ---------- 4. 主题名有效性校验 ----------
        custom = getattr(self.topic_model, "custom_labels_", None)
        name = None
        if custom:
            try:
                if 0 <= tid < len(custom):
                    name = custom[tid]
            except (IndexError, TypeError):
                name = None

        if not name:
            try:
                info = self.topic_model.get_topic(tid)
                name = info[0][0] if info else None
            except Exception as e:
                log.debug(f"[topic] get_topic 失败: {e}, tid={tid}")
                name = None

        if not name:
            return []

        # ---------- 5. 从 QA 库中筛选该主题 ----------
        results = []
        for idx, qa in enumerate(self.qa_meta):
            if name in qa.get("sop_phases", []):
                results.append({"idx": idx, "score": 0.9, "source": "topic"})
                if len(results) >= self.cfg.online_recall_top_k:
                    break
        return results

    def recall_vector(self, text: str) -> List[Dict]:
        vec = self._encode_cached(text)
        scores, indices = self.q_index.search(vec, self.cfg.online_recall_top_k)
        return [
            {"idx": int(i), "score": float(s), "source": "vector"}
            for i, s in zip(indices[0], scores[0]) if i >= 0
        ]

    def recall_label(self, intents: List[str], emotions: List[str]) -> List[Dict]:
        label_text = " ".join(intents + emotions)
        if not label_text.strip():
            return []
        vec = self._encode_cached(label_text)
        scores, indices = self.l_index.search(vec, self.cfg.online_recall_top_k)
        return [
            {"idx": int(i), "score": float(s), "source": "label"}
            for i, s in zip(indices[0], scores[0]) if i >= 0
        ]

    def recall_graph(self, phase: str) -> List[Dict]:
        node = self.graph["nodes"].get(phase)
        if not node:
            return []
        scripts = node.get("scripts", [])
        results = []
        for s in scripts[:self.cfg.online_recall_top_k]:
            for idx, qa in enumerate(self.qa_meta):
                if qa["question"] == s["question"] and qa["answer"] == s["answer"]:
                    results.append({"idx": idx, "score": s.get("quality", 0.5),
                                    "source": "graph"})
                    break
        return results

    # ---------- RRF 融合 ----------
    def fuse(self, recalls: Dict[str, List[Dict]], top_k: int) -> List[Dict]:
        K = self.cfg.online_rrf_k
        scores: Dict[int, float] = {}
        source_map: Dict[int, List[str]] = {}
        for source, items in recalls.items():
            for rank, item in enumerate(items):
                idx = item["idx"]
                scores[idx] = scores.get(idx, 0) + 1.0 / (K + rank + 1)
                source_map.setdefault(idx, []).append(source)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k * 2]
        results = []
        for idx, score in ranked:
            qa = self.qa_meta[idx]
            if qa.get("compliance") == "禁止":
                continue
            results.append({
                "question": qa["question"],
                "answer": qa["answer"],
                "score": score,
                "sources": source_map[idx],
                "quality": qa.get("quality_score", 0),
                "intents": qa.get("customer_intents", []),
                "phases": qa.get("sop_phases", []),
            })
        return results[:top_k]

    def retrieve(self, query: str, state: Dict, top_k: int = None) -> List[Dict]:
        top_k = top_k or self.cfg.online_top_k
        recalls = {}
        for name, func, args in [
            ("topic", self.recall_topic, (query,)),
            ("vector", self.recall_vector, (query,)),
            ("label", self.recall_label,
             (state.get("intents", []), state.get("emotions", []))),
            ("graph", self.recall_graph, (state.get("phase", ""),)),
        ]:
            try:
                recalls[name] = func(*args)
            except Exception as e:
                log.warning(f"召回 [{name}] 失败: {e}")
                recalls[name] = []
        return self.fuse(recalls, top_k)


# ============================================================
# 3. 实时推荐
# ============================================================
class RealtimeRecommender:
    """实时推荐入口"""

    def __init__(self):
        self.cfg = get_settings()
        t0 = time.time()
        log.info("=" * 50)
        log.info("初始化实时推荐器")
        log.info("=" * 50)

        self.retriever = FusionRetriever(
            bertopic_dir=self.cfg.bertopic_model_dir,
            faiss_dir=self.cfg.faiss_dir,
            graph_path=self.cfg.sop_graph_file,
        )
        with open(self.cfg.sop_graph_file, "r", encoding="utf-8") as f:
            graph = json.load(f)
        self.tracker = StateTracker(graph)
        log.info(f"推荐器初始化完成，总耗时 {time.time() - t0:.1f}s")

    def init_session(self, session_id: str):
        self.tracker.init_session(session_id)

    def on_asr(self, session_id: str, role: str, text: str) -> Dict:
        t0 = time.time()
        state = self.tracker.update(session_id, role, text)
        t1 = time.time()

        if role != "客户":
            return None

        results = self.retriever.retrieve(
            query=text,
            state={
                "phase": state["phase"],
                "intents": state["intents"],
                "emotions": state["emotions"],
            },
        )
        t2 = time.time()

        result = {
            "session_id": session_id,
            "phase": state["phase"],
            "intents": state["intents"],
            "emotions": state["emotions"],
            "ptp": state.get("ptp"),
            "recommendations": results,
            "latency_ms": round((t2 - t0) * 1000, 1),
            "state_latency_ms": round((t1 - t0) * 1000, 1),
            "retrieval_latency_ms": round((t2 - t1) * 1000, 1),
        }
        log.info(f"[{session_id}] 主题={state['phase']} | "
                 f"意图={state['intents']} | 情绪={state['emotions']} | "
                 f"延迟={result['latency_ms']}ms")
        return result


# ============================================================
# 4. 测试集
# ============================================================
def build_testset(labels: List[Dict], calls: List[Dict],
                  test_ratio: float = 0.2, seed: int = 42) -> List[Dict]:
    random.seed(seed)
    call_map = {c["call_id"]: c for c in calls}

    call_ids = [lb["call_id"] for lb in labels]
    random.shuffle(call_ids)
    n_test = int(len(call_ids) * test_ratio)
    test_ids = set(call_ids[:n_test])

    samples = []
    for lb in labels:
        if lb["call_id"] not in test_ids:
            continue
        call = call_map.get(lb["call_id"])
        if not call:
            continue
        turns = call["turns"]
        for i in range(len(turns) - 1):
            if turns[i]["role"] != "客户":
                continue
            for j in range(i + 1, min(i + 3, len(turns))):
                if turns[j]["role"] == "催收员":
                    samples.append({
                        "call_id": lb["call_id"],
                        "query": turns[i]["text"],
                        "context": turns[max(0, i - 2):i],
                        "ground_truth_answer": turns[j]["text"],
                        "ground_truth_phases": lb.get("sop_phases", []),
                        "ground_truth_intents": lb.get("customer_intents", []),
                        "ground_truth_emotions": lb.get("customer_emotions", []),
                        "ptp": lb.get("ptp", False),
                    })
                    break
    log.info(f"测试集: {len(samples)} 条")
    return samples


def save_testset(samples: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    log.info(f"测试集已保存: {path}")