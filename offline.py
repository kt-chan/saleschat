from datetime import datetime
import re, json, time, yaml
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
import jieba
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from hdbscan import HDBSCAN

import config as raw_config
from settings import get_settings
from logger import get_logger
from utils import retry, parallel_map, FileCache

log = get_logger(__name__)

ROLE_PATTERN = re.compile(r'^(催收员|客户|客服|坐席|用户)[：:](.*)$')
SPLIT_PATTERN = re.compile(r'[；;]')
ROLE_NORMALIZE = {"客服": "催收员", "坐席": "催收员", "用户": "客户"}

def parse_dialog(text: str) -> List[Dict]:
    if not text: return []
    turns, turn_id = [], 0
    for part in SPLIT_PATTERN.split(text):
        part = part.strip()
        if not part: continue
        m = ROLE_PATTERN.match(part)
        if not m: continue
        role = ROLE_NORMALIZE.get(m.group(1), m.group(1))
        content = m.group(2).strip()
        if content:
            turns.append({"turn_id": turn_id, "role": role, "text": content})
            turn_id += 1
    return turns

def load_calls(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists(): return []
    calls, skipped, raw_objs = [], 0, []
    if p.suffix.lower() == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): raw_objs.append(json.loads(line.strip()))
    else:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            raw_objs = data if isinstance(data, list) else data.get("calls", [])
            
    for obj in raw_objs:
        obj["turns"] = parse_dialog(obj.get("text", ""))
        if len(obj["turns"]) >= 2: calls.append(obj)
        else: skipped += 1
    log.info(f"加载完成: 有效 {len(calls)} 通, 跳过 {skipped} 通")
    return calls

def pair_turns(turns: List[Dict]) -> List[Dict]:
    pairs = []
    for i, t in enumerate(turns):
        if t["role"] != "客户": continue
        for j in range(i + 1, min(i + 3, len(turns))):
            if turns[j]["role"] == "催收员":
                pairs.append({"q_idx": i, "a_idx": j, "question": t["text"], "answer": turns[j]["text"]})
                break
    return pairs

class PromptLoader:
    def __init__(self):
        self.cfg = get_settings()
        with open(self.cfg.prompt_file, "r", encoding="utf-8") as f:
            self.tpls = yaml.safe_load(f)

    def get_session(self):
        return self.tpls["session_prompt"] \
            .replace("{sop_stages}", "、".join(raw_config.SOP_STAGES)) \
            .replace("{session_reasons}", "、".join(raw_config.SESSION_REASONS)) \
            .replace("{session_tactics}", "、".join(raw_config.SESSION_TACTICS)) \
            .replace("{session_objections}", "、".join(raw_config.SESSION_OBJECTIONS)) \
            .replace("{session_emotions}", "、".join(raw_config.SESSION_EMOTIONS)) \
            .replace("{session_compliance}", "、".join(raw_config.SESSION_COMPLIANCE)) \
            .replace("{session_outcomes}", "、".join(raw_config.SESSION_OUTCOMES))

    def get_pair(self):
        return self.tpls["pair_prompt"] \
            .replace("{sop_stages}", "、".join(raw_config.SOP_STAGES)) \
            .replace("{pair_intents}", "、".join(raw_config.UNIFIED_INTENTS)) \
            .replace("{pair_emotions}", "、".join(raw_config.PAIR_EMOTIONS)) \
            .replace("{pair_actions}", "、".join(raw_config.PAIR_ACTIONS)) \
            .replace("{pair_keywords}", "、".join(raw_config.PAIR_KEYWORDS)) \
            .replace("{pair_tags}", "、".join(raw_config.PAIR_TAGS)) \
            .replace("{pair_risk_levels}", "、".join(raw_config.PAIR_RISK_LEVELS)) \
            .replace("{pair_risk_types}", "、".join(raw_config.PAIR_RISK_TYPES))

class LLMLabeler:
    def __init__(self):
        self.cfg = get_settings()
        self.client = OpenAI(api_key=self.cfg.llm_api_key, base_url=self.cfg.llm_base_url, timeout=self.cfg.llm_timeout)
        self.prompts = PromptLoader()
        self.sess_cache = FileCache(self.cfg.cache_path, "llm_sessions")
        self.pair_cache = FileCache(self.cfg.cache_path, "llm_pairs")

    @retry()
    def _call(self, prompt):
        resp = self.client.chat.completions.create(
            model=self.cfg.llm_model, messages=[{"role": "user", "content": prompt}],
            temperature=self.cfg.llm_temperature, response_format={"type": "json_object"}
        )
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log.debug(f"@{ts}：LLM 打标 Prompt: {prompt[:16000]}...")
        log.debug(f"@{ts}：Result: {resp.choices[0].message.content[:2000]}...")
        return json.loads(resp.choices[0].message.content)

    def _fmt(self, turns): return "\n".join([f"{t['role']}: {t['text']}" for t in turns])

    def run_sessions(self, calls, out_path):
        log.info(f"Session 打标: {len(calls)} 通")
        def label(call):
            if (c := self.sess_cache.get(call["call_id"])): return c
            prompt = self.prompts.get_session().replace("{session_text}", self._fmt(call["turns"]))
            try: res = self._call(prompt)
            except Exception as e:
                log.error(f"Session 打标失败 {call['call_id']}: {e}")
                res = {"ptp": False, "sop_stages": [], "final_outcomes": []}
            res["call_id"] = call["call_id"]
            self.sess_cache.set(call["call_id"], res)
            return res
        res = parallel_map(label, calls, self.cfg.llm_max_concurrency, "Session")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in res:
                if r: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return res

    def run_pairs(self, calls, out_path):
        log.info(f"Pair 打标与重写: {len(calls)} 通")
        tasks = []
        for c in calls:
            turns = c["turns"]
            for p in pair_turns(turns):
                chunk = f"客户: {p['question']}\n催收员: {p['answer']}"
                # sess 只保留当前 chunk 之前的对话内容
                prev_turns = turns[:p["q_idx"]]
                sess_text = self._fmt(prev_turns) if prev_turns else ""
                tasks.append({
                    "call_id": c["call_id"],
                    "q_idx": p["q_idx"],
                    "a_idx": p["a_idx"],
                    "sess": sess_text,
                    "chunk": chunk,
                })

        def label(t):
            key = f"{t['call_id']}_{t['q_idx']}"
            if (c := self.pair_cache.get(key)): return c
            prompt = (self.prompts.get_pair()
                    .replace("{session_content}", t["sess"])
                    .replace("{chunk_content}", t["chunk"]))
            try:
                res = self._call(prompt)
            except Exception as e:
                log.error(f"Pair 打标失败 {key}: {e}")
                q_line, a_line = t["chunk"].split("\n")
                res = {
                    "stage": raw_config.SOP_STAGES[0],
                    "customer_intents": ["其他"],
                    "customer_emotions": [], "agent_actions": [], "keywords": [],
                    "tags": [], "risk": "None",
                    "rewritten_question": q_line.replace("客户: ", ""),
                    "rewritten_answer": a_line.replace("催收员: ", ""),
                }
            res.update({"call_id": t["call_id"], "q_idx": t["q_idx"], "a_idx": t["a_idx"]})
            self.pair_cache.set(key, res)
            return res

        res = parallel_map(label, tasks, self.cfg.llm_max_concurrency, "Pair")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in res:
                if r:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return res

class BERTopicBuilder:
    def __init__(self):
        self.cfg = get_settings()
        self.rcfg = raw_config
        self.model = SentenceTransformer(self.cfg.embedding_model, device=self.cfg.embedding_device)

    def build_and_export(self, docs: List[str]):
        log.info("构建 BERTopic 并导出标签向量...")
        if not docs: return
        
        # 使用环境变量初始化 BERTopic 依赖
        ctfidf = ClassTfidfTransformer(seed_words=["催收", "逾期", "还款"], seed_multiplier=self.cfg.bertopic_seed_multiplier) 
        umap_m = UMAP(
            n_neighbors=self.cfg.bertopic_umap_neighbors, 
            n_components=self.cfg.bertopic_umap_components, 
            metric="cosine", random_state=42
        )
        hdbscan_m = HDBSCAN(
            min_cluster_size=self.cfg.bertopic_hdbscan_min_size, 
            metric="euclidean", prediction_data=True
        )
        
        topic_model = BERTopic(
            embedding_model=self.model, 
            umap_model=umap_m, 
            hdbscan_model=hdbscan_m, 
            ctfidf_model=ctfidf, 
            zeroshot_topic_list=self.rcfg.SOP_STAGES, 
            zeroshot_min_similarity=self.cfg.bertopic_zeroshot_similarity,
            seed_topic_list=self.rcfg.SEED_TOPIC_LIST, 
            min_topic_size=self.cfg.bertopic_min_topic_size
        )
        topic_model.fit_transform(docs)
        
        # 导出 Stage 向量 (7 x D)
        stage_emb = self.model.encode(self.rcfg.SOP_STAGES, normalize_embeddings=True)
        np.save(self.cfg.stage_emb_file, stage_emb)
        
        # 导出 Intent 向量 (N x D)
        intent_emb = self.model.encode(self.rcfg.UNIFIED_INTENTS, normalize_embeddings=True)
        np.save(self.cfg.intent_emb_file, intent_emb)
        with open(self.cfg.intent_labels_file, "w", encoding="utf-8") as f:
            json.dump(self.rcfg.UNIFIED_INTENTS, f, ensure_ascii=False)
        log.info("标签嵌入导出完成")

class QAExtractor:
    def __init__(self, calls, sess_labels, pair_labels):
        self.call_map = {c["call_id"]: c for c in calls}
        self.sess_map = {s["call_id"]: s for s in sess_labels if s}
        self.pair_map = {(p["call_id"], p["q_idx"]): p for p in pair_labels if p}
        self.cfg = get_settings()

    def extract(self):
        qa_pairs = []
        for cid, sess in self.sess_map.items():
            if not sess.get("ptp"): continue
            call = self.call_map.get(cid)
            if not call: continue
            for p in pair_turns(call["turns"]):
                pl = self.pair_map.get((cid, p["q_idx"]))
                if not pl or pl.get("risk") == "High": continue
                intents = pl.get("customer_intents", [])
                primary_intent = intents[0] if intents else "其他"
                
                qa_pairs.append({
                    "question": pl.get("rewritten_question", p["question"]),
                    "answer": pl.get("rewritten_answer", p["answer"]),
                    "stage": pl.get("stage"),
                    "intent": primary_intent,
                    "quality": 0.8
                })
        unique_qa = list({q["question"]: q for q in qa_pairs}.values())
        log.info(f"抽取脱敏 QA 对: {len(qa_pairs)} -> {len(unique_qa)}")
        return unique_qa

class SOPGraphBuilder:
    def __init__(self, qa_pairs):
        self.rcfg = raw_config
        self.qa_pairs = qa_pairs

    def build(self):
        nodes = {}
        for stage in self.rcfg.SOP_STAGES:
            intents_in_stage = self.rcfg.STAGE_GATED_INTENTS.get(stage, [])
            stage_data = {}
            for intent in intents_in_stage:
                scripts = [q for q in self.qa_pairs if q.get("stage") == stage and q.get("intent") == intent]
                stage_data[intent] = {"scripts": scripts[:10]}
            nodes[stage] = {"type": "SOP_Stage", "intents": stage_data}
            
        edges = []
        for (src, intent), targets in self.rcfg.TRANSITION_ADJ.items():
            for tgt in targets:
                edges.append({"source": src, "target": tgt, "intent": intent})
        return {"nodes": nodes, "edges": edges}

class VectorIndexBuilder:
    def __init__(self):
        self.cfg = get_settings()
        self.model = SentenceTransformer(self.cfg.embedding_model)

    def build(self, qa_pairs):
        if not qa_pairs: return
        self.cfg.faiss_dir.mkdir(parents=True, exist_ok=True)
        q_texts = [q["question"] for q in qa_pairs]
        l_texts = [" ".join([q["intent"]]) for q in qa_pairs]
        
        q_emb = self.model.encode(q_texts, batch_size=self.cfg.embedding_batch_size, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
        l_emb = self.model.encode(l_texts, batch_size=self.cfg.embedding_batch_size, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
        
        q_idx, l_idx = faiss.IndexFlatIP(q_emb.shape[1]), faiss.IndexFlatIP(l_emb.shape[1])
        q_idx.add(q_emb); l_idx.add(l_emb)
        
        faiss.write_index(q_idx, str(self.cfg.faiss_dir / "question.faiss"))
        faiss.write_index(l_idx, str(self.cfg.faiss_dir / "label.faiss"))
        with open(self.cfg.faiss_dir / "qa_meta.json", "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, ensure_ascii=False)