# offline.py
"""
离线层：
1. 会话解析（parse_dialog / load_calls / pair_turns）
2. LLM 批量打标（LLMLabeler）
3. 配置优化（ConfigOptimizer）
4. 运行时配置（RuntimeConfig / get_runtime_config）
5. BERTopic 构建（BERTopicBuilder）
6. QA 抽取（QAExtractor）
7. 向量索引（VectorIndexBuilder）
8. SOP 图谱（SOPGraphBuilder）
"""
import re
import json
import time
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

import jieba
import jieba.analyse           # ★ 显式导入
import numpy as np
import faiss
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import yaml

import config as raw_config
from settings import get_settings
from logger import get_logger
from utils import retry, parallel_map, FileCache

log = get_logger(__name__)

STOPWORDS = {
    "这个", "那个", "什么", "我们", "你们", "他们", "可以", "如果",
    "现在", "今天", "明天", "就是", "还有", "然后", "但是", "因为",
}

# ============================================================
# Prompt 加载器（从 prompt.yaml 读模板，从 config.py 注入清单）
# ============================================================
class PromptLoader:
    """从 YAML 读模板 + 从 config 注入候选清单"""

    def __init__(self):
        self.cfg = get_settings()
        self._templates = self._load_yaml()
        self._session_prompt_cache = None
        self._pair_prompt_cache = None

    def _load_yaml(self) -> Dict:
        path = Path(self.cfg.prompt_file)
        if not path.exists():
            raise FileNotFoundError(f"Prompt 文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for key in ("session_prompt", "pair_prompt"):
            if key not in data:
                raise KeyError(f"YAML 缺少 '{key}'")
        return data

    def get_session_prompt(self) -> str:
        """
        渲染 session_prompt（注入 config.py 的候选清单）
        注意：使用 str.replace 而非 str.format，避免与 JSON Schema 花括号冲突；
              {session_text} 保留为占位符，由 label_one_session 填充。
        """
        if self._session_prompt_cache is None:
            tpl = self._templates["session_prompt"]
            tpl = tpl.replace("{sop_stages}", "、".join(raw_config.SOP_STAGES))
            tpl = tpl.replace("{session_reasons}", "、".join(raw_config.SESSION_REASONS))
            tpl = tpl.replace("{session_tactics}", "、".join(raw_config.SESSION_TACTICS))
            tpl = tpl.replace("{session_objections}", "、".join(raw_config.SESSION_OBJECTIONS))
            tpl = tpl.replace("{session_compliance}", "、".join(raw_config.SESSION_COMPLIANCE))
            tpl = tpl.replace("{session_outcomes}", "、".join(raw_config.SESSION_OUTCOMES))
            # 注意：{session_text} 不动，留给调用方填充
            self._session_prompt_cache = tpl
        return self._session_prompt_cache

    def get_pair_prompt(self) -> str:
        """渲染 pair_prompt（注入扁平候选清单）"""
        if self._pair_prompt_cache is None:
            tpl = self._templates["pair_prompt"]
            tpl = tpl.replace("{sop_stages}", "、".join(raw_config.SOP_STAGES))
            tpl = tpl.replace("{pair_intents}", "、".join(raw_config.PAIR_INTENTS))
            tpl = tpl.replace("{pair_actions}", "、".join(raw_config.PAIR_ACTIONS))
            tpl = tpl.replace("{pair_keywords}", "、".join(raw_config.PAIR_KEYWORDS))
            tpl = tpl.replace("{pair_tags}", "、".join(raw_config.PAIR_TAGS))
            tpl = tpl.replace("{pair_risk_levels}", "、".join(raw_config.PAIR_RISK_LEVELS))
            tpl = tpl.replace("{pair_risk_types}", "、".join(raw_config.PAIR_RISK_TYPES))
            # {session_content} 和 {chunk_content} 保留，留给调用方填充
            self._pair_prompt_cache = tpl
        return self._pair_prompt_cache


# ============================================================
# 会话解析（保持原有）
# ============================================================
ROLE_PATTERN = re.compile(r'^(催收员|客户|客服|坐席|用户)[：:](.*)$')
SPLIT_PATTERN = re.compile(r'[；;]')
ROLE_NORMALIZE = {"客服": "催收员", "坐席": "催收员", "用户": "客户"}


def parse_dialog(text: str) -> List[Dict]:
    """解析 ASR 文本为结构化对话轮次（带 turn_id，从 0 开始）"""
    if not text:
        return []
    turns = []
    turn_id = 0
    for part in SPLIT_PATTERN.split(text):
        part = part.strip()
        if not part:
            continue
        m = ROLE_PATTERN.match(part)
        if not m:
            continue
        role = ROLE_NORMALIZE.get(m.group(1), m.group(1))
        content = m.group(2).strip()
        if content:
            turns.append({
                "turn_id": turn_id,
                "role": role,
                "text": content,
            })
            turn_id += 1
    return turns


def save_calls(calls: List[Dict], output_path: Path) -> None:
    """
    将解析后的会话（含 turns 字段）保存为 JSONL
    用途：
      - 调试时可直接查看解析结果，无需重复解析原文
      - 后续 pipeline 可复用该文件，跳过 parse_dialog 步骤
    """
    if not calls:
        log.warning("无会话可保存")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(output_path, "w", encoding="utf-8") as f:
        for c in calls:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    size_mb = output_path.stat().st_size / 1024 / 1024
    log.info(f"已保存 {len(calls)} 条解析后会话 -> {output_path} "
             f"({size_mb:.1f} MB, 耗时 {time.time() - t0:.1f}s)")
    
def load_calls(path: str) -> List[Dict]:
    """加载会话（支持 JSON 与 JSONL）"""
    path = Path(path)
    if not path.exists():
        log.error(f"会话文件不存在: {path}")
        return []

    log.info(f"加载会话数据: {path}")
    calls = []
    skipped = 0
    raw_objs: List[Dict] = []
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_objs.append(json.loads(line))
                except json.JSONDecodeError as e:
                    log.warning(f"第 {line_no} 行解析失败: {e}")
    else:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            log.warning(f"会话文件为空: {path}")
            return []
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                raw_objs = parsed
            elif isinstance(parsed, dict):
                for key in ("calls", "data", "records", "items"):
                    if key in parsed and isinstance(parsed[key], list):
                        raw_objs = parsed[key]
                        log.info(f"从顶层键 '{key}' 读取 {len(raw_objs)} 条")
                        break
                else:
                    raw_objs = [parsed]
        except json.JSONDecodeError:
            log.warning(f"整体 JSON 解析失败，回退按 JSONL 逐行解析: {path}")
            for line_no, line in enumerate(content.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_objs.append(json.loads(line))
                except json.JSONDecodeError as e:
                    log.warning(f"第 {line_no} 行解析失败: {e}")

    for obj in raw_objs:
        if not isinstance(obj, dict):
            skipped += 1
            continue
        obj["turns"] = parse_dialog(obj.get("text", ""))
        if len(obj["turns"]) < 2:
            skipped += 1
            continue
        calls.append(obj)

    log.info(f"加载完成: 有效 {len(calls)} 通, 跳过 {skipped} 通")
    return calls


def pair_turns(turns: List[Dict]) -> List[Dict]:
    """配对 (客户问题, 催收员回答)"""
    pairs = []
    for i, turn in enumerate(turns):
        if turn["role"] != "客户":
            continue
        for j in range(i + 1, min(i + 3, len(turns))):
            if turns[j]["role"] == "催收员":
                pairs.append({
                    "question": turn["text"],
                    "answer": turns[j]["text"],
                    "q_idx": i,
                    "a_idx": j,
                })
                break
    return pairs


# ============================================================
# LLM 打标器（Session 级 + Pair 级）
# ============================================================
class LLMLabeler:
    """LLM 并发打标器（Session + Pair 两级）"""

    def __init__(self):
        self.cfg = get_settings()
        if not self.cfg.llm_api_key:
            log.warning("⚠️ LLM_API_KEY 未配置，打标将失败")
        self.client = OpenAI(
            api_key=self.cfg.llm_api_key,
            base_url=self.cfg.llm_base_url,
            timeout=self.cfg.llm_timeout,
        )
        self.prompts = PromptLoader()
        self.session_cache = FileCache(self.cfg.cache_path, namespace="llm_sessions")
        self.pair_cache = FileCache(self.cfg.cache_path, namespace="llm_pairs")
        self.api_calls = 0

    @retry(max_retries=3, backoff=2.0)
    def _call_llm(self, prompt: str) -> Dict:
        self.api_calls += 1
        resp = self.client.chat.completions.create(
            model=self.cfg.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.cfg.llm_temperature,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)

    @staticmethod
    def _format_dialog(turns: List[Dict]) -> str:
        return "\n".join([f"{t['role']}: {t['text']}" for t in turns])

    # ---------- Session 级 ----------
    def _empty_session_label(self, call_id: str) -> Dict:
        return {
            "call_id": call_id,
            "session_summary": "",
            "ptp": False,
            "sop_stages": [],
            "reason_for_delinquency": [],
            "agent_tactics": [],
            "customer_objections": [],
            "compliance_checkpoints": [],
            "final_outcomes": [],
        }

    def label_one_session(self, call: Dict) -> Dict:
        """
        对单通会话打 Session 级标签
        - 缓存命中直接返回
        - 用 str.replace 填充 {session_text}，避免 format 花括号冲突
        - LLM 失败时回退空标签
        - 补齐 call_id / mobtyp / colluserid 元数据
        """
        call_id = call["call_id"]

        # 1) 缓存命中
        cached = self.session_cache.get(call_id)
        if cached:
            log.debug(f"[Session] 命中缓存: {call_id}")
            return cached

        # 2) 渲染 Prompt（replace 而非 format）
        session_text = self._format_dialog(call["turns"])
        prompt = self.prompts.get_session_prompt().replace(
            "{session_text}", session_text
        )

        # 3) 调用 LLM
        try:
            label = self._call_llm(prompt)
        except Exception as e:
            log.error(f"Session 打标失败 {call_id}: {e}")
            label = self._empty_session_label(call_id)

        # 4) 注入元数据
        label["call_id"] = call_id
        label["mobtyp"] = call.get("mobtyp")
        label["colluserid"] = call.get("colluserid")

        # 5) 写缓存
        self.session_cache.set(call_id, label)
        return label

    def run_sessions(self, calls: List[Dict], output_path: Path) -> List[Dict]:
        log.info(f"开始 Session 级打标: {len(calls)} 通, "
                 f"并发={self.cfg.llm_max_concurrency}")
        t0 = time.time()
        all_labels = []
        for i in range(0, len(calls), self.cfg.llm_batch_size):
            batch = calls[i:i + self.cfg.llm_batch_size]
            log.info(f"Session 批次 {i // self.cfg.llm_batch_size + 1}, 大小 {len(batch)}")
            results = parallel_map(
                self.label_one_session, batch,
                max_workers=self.cfg.llm_max_concurrency,
                desc=f"Session-{i // self.cfg.llm_batch_size + 1}",
            )
            all_labels.extend([r for r in results
                               if isinstance(r, dict) and r.get("call_id")])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for lb in all_labels:
                f.write(json.dumps(lb, ensure_ascii=False) + "\n")
        log.info(f"Session 打标完成: {len(all_labels)} 条, "
                 f"耗时 {time.time() - t0:.1f}s, API={self.api_calls}")
        return all_labels

    # ---------- Pair 级 ----------
    def _empty_pair_label(self, call_id: str, q_idx: int, a_idx: int) -> Dict:
        return {
            "call_id": call_id,
            "q_idx": q_idx,
            "a_idx": a_idx,
            "stage": "专业开场及身份核实",
            "customer_intents": [],
            "agent_actions": [],
            "keywords": [],
            "tags": [],
            "risk": "None",
        }

    def label_one_pair(self, task: Dict) -> Dict:
        """task = {call_id, q_idx, a_idx, session_content, chunk_content}"""
        key = f"{task['call_id']}::{task['q_idx']}::{task['a_idx']}"
        cached = self.pair_cache.get(key)
        if cached:
            return cached

        prompt = self.prompts.get_pair_prompt() \
            .replace("{session_content}", task["session_content"]) \
            .replace("{chunk_content}", task["chunk_content"])
        try:
            label = self._call_llm(prompt)
        except Exception as e:
            log.error(f"Pair 打标失败 {key}: {e}")
            label = self._empty_pair_label(task["call_id"],
                                            task["q_idx"], task["a_idx"])

        label["call_id"] = task["call_id"]
        label["q_idx"] = task["q_idx"]
        label["a_idx"] = task["a_idx"]
        self.pair_cache.set(key, label)
        return label

    def run_pairs(self, calls: List[Dict], output_path: Path) -> List[Dict]:
        log.info("开始 Pair 级打标...")
        t0 = time.time()

        # 展平所有 pair 任务
        tasks = []
        for call in calls:
            session_content = self._format_dialog(call["turns"])
            for pair in pair_turns(call["turns"]):
                chunk = f"客户: {pair['question']}\n催收员: {pair['answer']}"
                tasks.append({
                    "call_id": call["call_id"],
                    "q_idx": pair["q_idx"],
                    "a_idx": pair["a_idx"],
                    "session_content": session_content,
                    "chunk_content": chunk,
                })

        log.info(f"Pair 任务总数: {len(tasks)}")

        all_labels = []
        for i in range(0, len(tasks), self.cfg.llm_batch_size):
            batch = tasks[i:i + self.cfg.llm_batch_size]
            log.info(f"Pair 批次 {i // self.cfg.llm_batch_size + 1}, 大小 {len(batch)}")
            results = parallel_map(
                self.label_one_pair, batch,
                max_workers=self.cfg.llm_max_concurrency,
                desc=f"Pair-{i // self.cfg.llm_batch_size + 1}",
            )
            all_labels.extend([r for r in results
                               if isinstance(r, dict) and r.get("call_id")])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for lb in all_labels:
                f.write(json.dumps(lb, ensure_ascii=False) + "\n")
        log.info(f"Pair 打标完成: {len(all_labels)} 条, "
                 f"耗时 {time.time() - t0:.1f}s, API={self.api_calls}")
        return all_labels
    
# ============================================================
# 3. 配置优化
# ============================================================
class ConfigOptimizer:
    """从 LLM 标签挖掘 + 合并人工配置"""

    def __init__(self, labels: List[Dict], calls: List[Dict]):
        self.cfg = get_settings()
        self.labels = labels
        self.calls = calls
        self.call_map = {c["call_id"]: c for c in calls}
        self.min_quality = self.cfg.config_min_quality_score

    def _valid_labels(self):
        return [lb for lb in self.labels
                if lb.get("quality_score", 0) >= self.min_quality]

    def optimize_seed_topics(self, top_k: int = 10) -> List[List[str]]:
        topic_texts = defaultdict(list)
        for lb in self._valid_labels():
            call = self.call_map.get(lb["call_id"])
            if not call:
                continue
            best = lb.get("best_cs_script", "")
            intents = lb.get("customer_intents", [])
            for phase in lb.get("sop_phases", []):
                if best:
                    topic_texts[phase].append(best)
                for t in call["turns"]:
                    if t["role"] == "客户":
                        for intent in intents:
                            kws = raw_config.INTENT_KEYWORDS.get(intent, [])
                            if any(kw in t["text"] for kw in kws):
                                topic_texts[phase].append(t["text"])
                                break

        result = []
        for i, phase in enumerate(raw_config.SOP_STAGES):
            base = raw_config.SEED_TOPIC_LIST[i].copy()
            texts = topic_texts.get(phase, [])
            if len(texts) < 10:
                result.append(base)
                continue
            auto = jieba.analyse.extract_tags(" ".join(texts), topK=top_k * 2)
            auto = [w for w in auto
                    if len(w) >= 2 and not w.isdigit() and w not in STOPWORDS]
            merged = base.copy()
            for kw in auto:
                if kw not in merged:
                    merged.append(kw)
            result.append(merged[:top_k + 5])
        log.info(f"[优化] SEED_TOPIC_LIST 词数: {[len(s) for s in result]}")
        return result

    def optimize_seed_words(self, top_k: int = 40) -> List[str]:
        texts = []
        for lb in self._valid_labels():
            if lb.get("best_cs_script"):
                texts.append(lb["best_cs_script"])
            for v in lb.get("key_slots", {}).values():
                if v:
                    texts.append(str(v))
            call = self.call_map.get(lb["call_id"])
            if call:
                for t in call["turns"]:
                    if t["role"] == "客户" and len(t["text"]) >= 4:
                        texts.append(t["text"])

        if not texts:
            log.warning("[优化] SEED_WORDS: 无有效文本，保留原始")
            return raw_config.SEED_WORDS.copy()

        auto = jieba.analyse.extract_tags(" ".join(texts), topK=top_k * 2)
        auto = [w for w in auto
                if len(w) >= 2 and not w.isdigit() and w not in STOPWORDS]
        merged = raw_config.SEED_WORDS.copy()
        for w in auto:
            if w not in merged:
                merged.append(w)
        log.info(f"[优化] SEED_WORDS: {len(raw_config.SEED_WORDS)} → {len(merged)}")
        return merged[:top_k + 20]

    def optimize_intent_keywords(self, top_k: int = 8) -> Dict[str, List[str]]:
        intent_texts = defaultdict(list)
        for lb in self._valid_labels():
            call = self.call_map.get(lb["call_id"])
            if not call:
                continue
            for t in call["turns"]:
                if t["role"] != "客户":
                    continue
                for intent, kws in raw_config.INTENT_KEYWORDS.items():
                    if any(kw in t["text"] for kw in kws):
                        intent_texts[intent].append(t["text"])

        result = {}
        for intent, base in raw_config.INTENT_KEYWORDS.items():
            texts = intent_texts.get(intent, [])
            if len(texts) < 10:
                result[intent] = base
                continue
            auto = jieba.analyse.extract_tags(" ".join(texts), topK=top_k * 3)
            auto = [w for w in auto
                    if len(w) >= 2 and not w.isdigit() and w not in STOPWORDS]
            merged = base.copy()
            for kw in auto:
                if kw not in merged:
                    merged.append(kw)
            result[intent] = merged[:top_k + len(base)]
        log.info(f"[优化] INTENT_KEYWORDS: {len(result)} 个意图")
        return result

    def optimize_emotion_keywords(self, top_k: int = 12) -> Dict[str, List[str]]:
        emo_texts = defaultdict(list)
        for lb in self._valid_labels():
            call = self.call_map.get(lb["call_id"])
            if not call:
                continue
            for t in call["turns"]:
                if t["role"] != "客户":
                    continue
                for emo in lb.get("customer_emotions", []):
                    emo_texts[emo].append(t["text"])

        result = {}
        for emo, base in raw_config.EMOTION_KEYWORDS.items():
            texts = emo_texts.get(emo, [])
            if len(texts) < 10:
                result[emo] = base
                continue
            auto = jieba.analyse.extract_tags(" ".join(texts), topK=top_k * 2)
            auto = [w for w in auto
                    if len(w) >= 2 and not w.isdigit() and w not in STOPWORDS]
            merged = base.copy()
            for kw in auto:
                if kw not in merged:
                    merged.append(kw)
            result[emo] = merged[:top_k]
        log.info(f"[优化] EMOTION_KEYWORDS: {len(result)} 类")
        return result

    def optimize_transition_table(self) -> Dict[Tuple[str, str], str]:
        transitions = Counter()
        for lb in self._valid_labels():
            explicit = lb.get("phase_transitions", [])
            if explicit:
                for t in explicit:
                    src, tgt = t.get("from"), t.get("to")
                    intent = t.get("trigger_intent", "其他")
                    if src in raw_config.SOP_STAGES and tgt in raw_config.SOP_STAGES:
                        transitions[(src, intent, tgt)] += 1
            else:
                phases = lb.get("sop_phases", [])
                for i in range(len(phases) - 1):
                    for intent in lb.get("customer_intents", []):
                        transitions[(phases[i], intent, phases[i + 1])] += 1

        table = {}
        for (src, intent, tgt), _ in transitions.most_common():
            key = (src, intent)
            if key not in table:
                table[key] = tgt

        filled = 0
        for key, tgt in raw_config.TRANSITION_TABLE.items():
            if key not in table:
                table[key] = tgt
                filled += 1
        log.info(f"[优化] TRANSITION_TABLE: 数据 {len(table) - filled} + 兜底 {filled} = {len(table)}")
        return table

    def run(self, output_path: Path) -> Dict:
        log.info("=" * 50)
        log.info("开始配置优化")
        log.info("=" * 50)
        t0 = time.time()
        optimized = {
            "sop_stages": raw_config.SOP_STAGES,
            "seed_topic_list": self.optimize_seed_topics(),
            "seed_words": self.optimize_seed_words(),
            "intent_keywords": self.optimize_intent_keywords(),
            "emotion_keywords": self.optimize_emotion_keywords(),
            "transition_table": {
                f"{src}|{intent}": tgt
                for (src, intent), tgt in self.optimize_transition_table().items()
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(optimized, f, ensure_ascii=False, indent=2)
        log.info(f"优化配置已保存: {output_path} (耗时 {time.time() - t0:.1f}s)")
        return optimized


# ============================================================
# 4. 运行时配置
# ============================================================
class RuntimeConfig:
    """运行时配置（优先优化版本，回退原始 config）"""

    def __init__(self):
        self.cfg = get_settings()
        path = self.cfg.optimized_config_file

        if path.exists():
            try:
                log.info(f"加载优化配置: {path}")
                with open(path, "r", encoding="utf-8") as f:
                    opt = json.load(f)
                self.sop_stages = opt["sop_stages"]
                self.seed_topic_list = opt["seed_topic_list"]
                self.seed_words = opt["seed_words"]
                self.intent_keywords = opt["intent_keywords"]
                self.emotion_keywords = opt["emotion_keywords"]
                self.transition_table = {}
                for k, v in opt["transition_table"].items():
                    src, intent = k.split("|", 1)
                    self.transition_table[(src, intent)] = v
                self.source = "optimized"
                return
            except (json.JSONDecodeError, KeyError) as e:
                log.error(f"优化配置损坏，回退原始: {e}")

        log.info("使用原始 config.py")
        self.sop_stages = raw_config.SOP_STAGES
        self.seed_topic_list = raw_config.SEED_TOPIC_LIST
        self.seed_words = raw_config.SEED_WORDS
        self.intent_keywords = raw_config.INTENT_KEYWORDS
        self.emotion_keywords = raw_config.EMOTION_KEYWORDS
        self.transition_table = raw_config.TRANSITION_TABLE
        self.source = "raw"


_runtime_config = None


def get_runtime_config() -> RuntimeConfig:
    global _runtime_config
    if _runtime_config is None:
        _runtime_config = RuntimeConfig()
    return _runtime_config


# ============================================================
# 5. BERTopic
# ============================================================
class BERTopicBuilder:
    def __init__(self):
        self.cfg = get_settings()
        self.rcfg = get_runtime_config()
        self.topic_model = None

    def build(self, docs: List[str]) -> Tuple[List[int], List[float]]:
        
        if len(docs) < 30:
            log.warning(f"文档数量过少 ({len(docs)} 条)，BERTopic 可能无法生成有效主题。"
                        f"建议至少提供 100 条以上文档。")
            
        log.info(f"构建 BERTopic: 文档={len(docs)}, 主题={len(self.rcfg.sop_stages)}, "
                 f"配置来源={self.rcfg.source}")
        t0 = time.time()

        ctfidf = ClassTfidfTransformer(
            seed_words=self.rcfg.seed_words,
            seed_multiplier=self.cfg.bertopic_seed_multiplier,
        )
        umap_model = UMAP(
            n_neighbors=self.cfg.bertopic_umap_neighbors,
            n_components=self.cfg.bertopic_umap_components,
            min_dist=0.0, metric="cosine", random_state=42,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.cfg.bertopic_hdbscan_min_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,        # ★ 必须：允许对新数据预测
            gen_min_span_tree=True,      # ★ 必须：生成最小生成树
        )

        self.topic_model = BERTopic(
            embedding_model=SentenceTransformer(
                self.cfg.embedding_model, device=self.cfg.embedding_device),
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            ctfidf_model=ctfidf,
            zeroshot_topic_list=self.rcfg.sop_stages,
            zeroshot_min_similarity=self.cfg.bertopic_zeroshot_similarity,
            seed_topic_list=self.rcfg.seed_topic_list,
            min_topic_size=self.cfg.bertopic_min_topic_size,
            verbose=True,
        )

        topics, probs = self.topic_model.fit_transform(docs)
        log.info(f"BERTopic 拟合完成: {len(set(topics))} 主题, "
                 f"耗时 {time.time() - t0:.1f}s")

        # ★ 修复主题映射逻辑
        topic_labels = {}
        for _, row in self.topic_model.get_topic_info().iterrows():
            tid = row["Topic"]
            if tid == -1:
                continue
            info = self.topic_model.get_topic(tid)
            if info and len(info) > 0:
                # 优先使用自定义标签（零样本主题名）
                custom = getattr(self.topic_model, "custom_labels_", None)
                if custom and 0 <= tid < len(custom) and custom[tid]:
                    topic_labels[tid] = custom[tid]
                else:
                    # 回退到关键词匹配
                    matched = self._match_sop(info[0][0])
                    if matched:
                        topic_labels[tid] = matched
        
        if topic_labels:
            self.topic_model.set_topic_labels(topic_labels)
            log.info(f"主题映射: {topic_labels}")
        else:
            log.warning("⚠️ 无法生成主题映射，BERTopic 可能未生成有效主题。"
                       f"当前主题数: {len(self.topic_model.get_topic_info()) - 1}")
        
        return topics, probs

    def _match_sop(self, keyword: str) -> str:
        for i, sop in enumerate(self.rcfg.sop_stages):
            if keyword in self.rcfg.seed_topic_list[i]:
                return sop
        for i, sop in enumerate(self.rcfg.sop_stages):
            if any(kw in keyword or keyword in kw
                   for kw in self.rcfg.seed_topic_list[i]):
                return sop
        return ""

    def save(self):
        path = self.cfg.bertopic_model_dir
        path.parent.mkdir(parents=True, exist_ok=True)
        self.topic_model.save(str(path))
        log.info(f"BERTopic 已保存: {path}")


# ============================================================
# 6. QA 抽取
# ============================================================
class QAExtractor:
    def __init__(self, calls: List[Dict], session_labels: List[Dict],
                 pair_labels: List[Dict]):
        self.cfg = get_settings()
        self.call_map = {c["call_id"]: c for c in calls}
        self.session_labels = session_labels
        self.pair_labels = pair_labels
        # 建索引：(call_id, q_idx) -> pair label
        self.pair_map = {
            (p["call_id"], p["q_idx"]): p for p in pair_labels
        }
        # call_id -> session label
        self.session_map = {s["call_id"]: s for s in session_labels}

    def extract(self, min_quality: float = 0.7) -> List[Dict]:
        """
        从 PTP 会话中抽取 QA 对
        标注来源：Pair 级标签（意图、动作、标签）
        """
        pairs = []
        for sess in self.session_labels:
            if not sess.get("ptp"):
                continue
            call_id = sess["call_id"]
            call = self.call_map.get(call_id)
            if not call:
                continue
            for pair in pair_turns(call["turns"]):
                pair_label = self.pair_map.get((call_id, pair["q_idx"]))
                if not pair_label:
                    continue
                pairs.append({
                    "call_id": call_id,
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "context": call["turns"][max(0, pair["q_idx"] - 2):pair["q_idx"]],
                    "stage": pair_label.get("stage", "专业开场及身份核实"),
                    "customer_intents": pair_label.get("customer_intents", []),
                    "agent_actions": pair_label.get("agent_actions", []),
                    "keywords": pair_label.get("keywords", []),
                    "tags": pair_label.get("tags", []),
                    "risk": pair_label.get("risk", "None"),
                    "session_outcomes": sess.get("final_outcomes", []),
                    "mobtyp": sess.get("mobtyp"),
                    "quality_score": self._score_quality(pair_label, sess),
                    "source": "ptp_pair_extracted",
                    "compliance": "安全" if pair_label.get("risk") in ("None", "Low") else "警告",
                })
        log.info(f"抽取 {len(pairs)} 个 QA 对")
        return pairs

    @staticmethod
    def _score_quality(pair_label: Dict, sess: Dict) -> float:
        """综合 Pair 级与 Session 级信号计算质量分"""
        score = 0.5
        # Session 级 PTP 是强正向信号
        if sess.get("ptp"):
            score += 0.2
        # Pair 级风险越低越好
        risk = pair_label.get("risk", "None")
        if risk == "None":
            score += 0.15
        elif risk == "High":
            score -= 0.3
        # 有正向动作加分
        positive_actions = {"锁定PTP", "复述PTP", "确认还款日期",
                            "同理心安抚", "引导分期", "引导减免"}
        if positive_actions & set(pair_label.get("agent_actions", [])):
            score += 0.1
        return max(0.0, min(1.0, score))

    def dedup(self, pairs: List[Dict], threshold: float = 0.92) -> List[Dict]:
        if not pairs:
            log.warning("QA 去重: 输入为空")
            return []
        log.info(f"开始去重: {len(pairs)} 条")
        t0 = time.time()
        model = SentenceTransformer(
            self.cfg.embedding_model, device=self.cfg.embedding_device)
        questions = [p["question"] for p in pairs]
        embs = model.encode(
            questions,
            batch_size=self.cfg.embedding_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        used = np.zeros(len(pairs), dtype=bool)
        order = sorted(range(len(pairs)),
                       key=lambda i: pairs[i]["quality_score"], reverse=True)
        keep = []
        for i in order:
            if used[i]:
                continue
            keep.append(pairs[i])
            used[i] = True
            sims = cosine_similarity([embs[i]], embs)[0]
            used |= (sims > threshold)
        log.info(f"去重后: {len(keep)} 条, 耗时 {time.time() - t0:.1f}s")
        return keep



# ============================================================
# 7. 向量索引
# ============================================================
class VectorIndexBuilder:
    def __init__(self):
        self.cfg = get_settings()
        self.model = SentenceTransformer(
            self.cfg.embedding_model, device=self.cfg.embedding_device)

    def build(self, qa_pairs: List[Dict], save_dir: Path):
        if not qa_pairs:
            log.warning("向量索引: QA 为空，跳过")
            return
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        q_texts = [qa["question"] for qa in qa_pairs]
        l_texts = [
            " ".join(qa.get("customer_intents", []) +
                     qa.get("customer_emotions", []) +
                     qa.get("sop_phases", []))
            for qa in qa_pairs
        ]

        log.info(f"编码问题向量: {len(q_texts)} 条")
        q_vecs = self.model.encode(
            q_texts, batch_size=self.cfg.embedding_batch_size,
            show_progress_bar=True, normalize_embeddings=True,
        ).astype(np.float32)

        log.info(f"编码标签向量: {len(l_texts)} 条")
        l_vecs = self.model.encode(
            l_texts, batch_size=self.cfg.embedding_batch_size,
            show_progress_bar=True, normalize_embeddings=True,
        ).astype(np.float32)

        dim = q_vecs.shape[1]
        q_idx = faiss.IndexFlatIP(dim)
        l_idx = faiss.IndexFlatIP(dim)
        q_idx.add(q_vecs)
        l_idx.add(l_vecs)

        faiss.write_index(q_idx, str(save_dir / "question.faiss"))
        faiss.write_index(l_idx, str(save_dir / "label.faiss"))
        with open(save_dir / "qa_meta.json", "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, ensure_ascii=False)
        log.info(f"向量索引已保存: {save_dir}, 维度={dim}, "
                 f"耗时 {time.time() - t0:.1f}s")


# ============================================================
# 8. SOP 图谱
# ============================================================
class SOPGraphBuilder:
    def __init__(self, calls, session_labels, pair_labels, qa_pairs):
        self.cfg = get_settings()
        self.rcfg = get_runtime_config()
        self.calls = calls
        self.session_labels = session_labels
        self.pair_labels = pair_labels
        self.qa_pairs = qa_pairs

    def build(self) -> Dict:
        log.info("构建 SOP 图谱")
        phase_stats = defaultdict(lambda: {
            "intents": Counter(), "emotions": Counter(),
            "slots": Counter(), "scripts": [],
        })
        # 从 Pair 标签统计每个 stage 下的意图分布
        for p in self.pair_labels:
            stage = p.get("stage", "专业开场及身份核实")
            if stage not in self.rcfg.sop_stages:
                stage = "专业开场及身份核实"
            for intent in p.get("customer_intents", []):
                phase_stats[stage]["intents"][intent] += 1
            for kw in p.get("keywords", []):
                phase_stats[stage]["slots"][kw] += 1

        # 从 QA 对补充话术
        for qa in self.qa_pairs:
            stage = qa.get("stage", "专业开场及身份核实")
            if stage not in self.rcfg.sop_stages:
                stage = "专业开场及身份核实"
            phase_stats[stage]["scripts"].append({
                "question": qa["question"], "answer": qa["answer"],
                "quality": qa["quality_score"], "source": qa["source"],
            })

        graph = {"nodes": {}, "edges": []}
        for stage in self.rcfg.sop_stages:
            stats = phase_stats.get(stage, {
                "intents": Counter(), "emotions": Counter(),
                "slots": Counter(), "scripts": [],
            })
            scripts = sorted(stats["scripts"],
                             key=lambda x: x["quality"], reverse=True)[:10]
            graph["nodes"][stage] = {
                "type": "SOP阶段",
                "order": self.rcfg.sop_stages.index(stage),
                "intents": [i for i, _ in stats["intents"].most_common(10)],
                "slots": [s for s, _ in stats["slots"].most_common(10)],
                "scripts": scripts,
            }
        for (src, intent), tgt in self.rcfg.transition_table.items():
            if src in self.rcfg.sop_stages and tgt in self.rcfg.sop_stages:
                graph["edges"].append({
                    "source": src, "target": tgt,
                    "type": "意图跳转", "intent": intent,
                })
        log.info(f"SOP 图谱: {len(graph['nodes'])} 节点, {len(graph['edges'])} 边")
        return graph

    def save(self, graph: Dict, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)
        log.info(f"图谱已保存: {path}")


# ============================================================
# 9. 文档准备
# ============================================================
def prepare_docs_from_labels(labels: List[Dict], calls: List[Dict],
                              min_quality: float = 0.6):
    """从 LLM 标签中提取客户问题文本，分词后用于 BERTopic"""
    call_map = {c["call_id"]: c for c in calls}
    docs, meta = [], []
    for lb in labels:
        if lb.get("quality_score", 0) < min_quality:
            continue
        call = call_map.get(lb["call_id"])
        if not call:
            continue
        for i, t in enumerate(call["turns"]):
            if t["role"] == "客户" and len(t["text"]) >= 4:
                docs.append(" ".join(jieba.cut(t["text"])))
                meta.append({"call_id": lb["call_id"], "turn_idx": i})
    log.info(f"准备 {len(docs)} 条文档")
    return docs, meta