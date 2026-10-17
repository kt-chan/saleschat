import json, time
from typing import List, Dict, Tuple
import numpy as np
import faiss
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

import config as raw_config
from settings import get_settings
from logger import get_logger

log = get_logger(__name__)

class StateTracker:
    def __init__(self, graph: Dict):
        self.cfg = get_settings()
        self.rcfg = raw_config
        
        self.model = SentenceTransformer(self.cfg.embedding_model, device=self.cfg.embedding_device)
        self.stage_emb = np.load(self.cfg.stage_emb_file)
        self.intent_emb = np.load(self.cfg.intent_emb_file)
        with open(self.cfg.intent_labels_file, "r", encoding="utf-8") as f:
            self.intent_labels = json.load(f)
            
        self.adj = self.rcfg.TRANSITION_ADJ
        self.sessions = {}

    def _get_scores(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        emb = self.model.encode([text], normalize_embeddings=True)[0]
        return np.dot(self.stage_emb, emb), np.dot(self.intent_emb, emb)

    def _match_rules(self, text: str) -> str:
        best_intent, highest = None, 0
        for intent, kws in self.rcfg.INTENT_KEYWORDS.items():
            for kw in kws:
                score = fuzz.partial_ratio(kw, text)
                if score > highest:
                    highest, best_intent = score, intent
        return best_intent if highest >= self.cfg.online_rule_cutoff else None

    def update(self, session_id: str, role: str, text: str) -> Dict:
        state = self.sessions.setdefault(session_id, {
            "stage": self.rcfg.SOP_STAGES[0], 
            "current_intent": "",
            "history": [], "last_decision": {}
        })
        if role != "客户": return state

        stage_scores, intent_scores = self._get_scores(text)
        rule_intent = self._match_rules(text)
        
        top_i_idx = np.argmax(intent_scores)
        vec_intent = self.intent_labels[top_i_idx]
        vec_score = float(intent_scores[top_i_idx])
        
        allowed = self.rcfg.STAGE_GATED_INTENTS.get(state["stage"], [])
        triggered = None
        
        if rule_intent and rule_intent in allowed:
            triggered = rule_intent
        elif vec_score >= self.cfg.online_intent_min and vec_intent in allowed:
            triggered = vec_intent

        decision = "保持"
        if triggered:
            state["current_intent"] = triggered
            next_stages = self.adj.get((state["stage"], triggered), [])
            
            if len(next_stages) == 1:
                state["stage"] = next_stages[0]
                decision = f"阶段转移至({state['stage']})"
            elif len(next_stages) > 1:
                best_target, best_score = None, -1
                for tgt in next_stages:
                    tgt_idx = self.rcfg.SOP_STAGES.index(tgt)
                    if stage_scores[tgt_idx] > best_score:
                        best_score = stage_scores[tgt_idx]
                        best_target = tgt
                state["stage"] = best_target
                decision = f"分支转移至({state['stage']})"

        state["last_decision"] = {"text": text, "rule": rule_intent, "vec": vec_intent, "triggered": triggered, "decision": decision}
        return state

class FusionRetriever:
    def __init__(self):
        self.cfg = get_settings()
        self.q_index = faiss.read_index(str(self.cfg.faiss_dir / "question.faiss"))
        self.l_index = faiss.read_index(str(self.cfg.faiss_dir / "label.faiss"))
        with open(self.cfg.faiss_dir / "qa_meta.json", "r", encoding="utf-8") as f:
            self.qa_meta = json.load(f)
        with open(self.cfg.sop_graph_file, "r", encoding="utf-8") as f:
            self.graph = json.load(f)
        self.model = SentenceTransformer(self.cfg.embedding_model)
        self.cache = {}

    def _encode(self, text):
        if text not in self.cache:
            if len(self.cache) > self.cfg.online_embedding_cache_size: self.cache.clear()
            self.cache[text] = self.model.encode([text], normalize_embeddings=True).astype(np.float32)
        return self.cache[text]

    def retrieve(self, query: str, state: Dict) -> List[Dict]:
        recalls = {"graph": [], "label": [], "vector": []}
        
        stage_data = self.graph["nodes"].get(state["stage"])
        if stage_data and state.get("current_intent"):
            intent_scripts = stage_data.get("intents", {}).get(state["current_intent"], {}).get("scripts", [])
            for qa in intent_scripts:
                for idx, meta in enumerate(self.qa_meta):
                    if meta["question"] == qa["question"]:
                        recalls["graph"].append({"idx": idx, "score": 1.0})
                        break
                        
        if state.get("current_intent"):
            label_vec = self._encode(state["current_intent"])
            scores, indices = self.l_index.search(label_vec, self.cfg.online_recall_top_k)
            for i, s in zip(indices[0], scores[0]):
                if i >= 0: recalls["label"].append({"idx": int(i), "score": float(s) * 0.6})
                
        vec = self._encode(query)
        scores, indices = self.q_index.search(vec, self.cfg.online_recall_top_k)
        for i, s in zip(indices[0], scores[0]):
            if i >= 0: recalls["vector"].append({"idx": int(i), "score": float(s) * 0.4})

        final_scores = {}
        for src, items in recalls.items():
            for item in items:
                idx = item["idx"]
                final_scores[idx] = max(final_scores.get(idx, 0), item["score"])
                
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:self.cfg.online_top_k]
        return [{"answer": self.qa_meta[idx]["answer"], "score": round(sc, 3), "question": self.qa_meta[idx]["question"]} for idx, sc in ranked]

class RealtimeRecommender:
    def __init__(self):
        with open(get_settings().sop_graph_file, "r", encoding="utf-8") as f:
            graph = json.load(f)
        self.tracker = StateTracker(graph)
        self.retriever = FusionRetriever()

    def init_session(self, sid):
        self.tracker.sessions[sid] = {"stage": raw_config.SOP_STAGES[0], "current_intent": "", "history": [], "last_decision": {}}

    def on_asr(self, sid: str, role: str, text: str):
        t0 = time.time()
        state = self.tracker.update(sid, role, text)
        if role != "客户": return None
        recs = self.retriever.retrieve(text, state)
        latency = (time.time() - t0) * 1000
        log.info(f"[{sid}] Stage: {state['stage']} | Intent: {state['current_intent']} | 判决: {state['last_decision']['decision']} | 延迟: {latency:.1f}ms")
        return {"stage": state["stage"], "intent": state["current_intent"], "decision": state["last_decision"], "recs": recs, "latency": latency}