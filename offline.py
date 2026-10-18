import re, json, yaml
from pathlib import Path
from typing import List, Dict
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from psycopg2.extras import execute_values

import config as raw_config
from settings import get_settings
from logger import get_logger
from utils import retry, parallel_map, get_db_conn, release_db_conn, safe_json_loads

log = get_logger(__name__)


def parse_dialog(text: str) -> List[Dict]:
    if not text:
        return []
    turns, turn_id = [], 0
    for part in re.split(r"[；;]", text):
        part = part.strip()
        m = re.match(r"^(催收员|客户|客服|坐席|用户)[：:](.*)$", part)
        if m:
            role = {"客服": "催收员", "坐席": "催收员", "用户": "客户"}.get(
                m.group(1), m.group(1)
            )
            content = m.group(2).strip()
            if content:
                turns.append({"turn_id": turn_id, "role": role, "text": content})
                turn_id += 1
    return turns


def load_calls(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        log.error(f"文件不存在: {path}")
        return []
    calls, skipped, raw_objs = [], 0, []
    if p.suffix.lower() == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_objs.append(json.loads(line.strip()))
    else:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            raw_objs = data if isinstance(data, list) else data.get("calls", [])

    for obj in raw_objs:
        obj["turns"] = parse_dialog(obj.get("text", ""))
        if len(obj["turns"]) >= 2:
            calls.append(obj)
        else:
            skipped += 1
    log.info(f"加载完成: 有效 {len(calls)} 通, 跳过 {skipped} 通")
    return calls


def merge_consecutive(turns: List[Dict], sep: str = "") -> List[Dict]:
    """把连续同角色的 turn 合并成块，保留原始索引范围。"""
    blocks: List[Dict] = []
    for idx, t in enumerate(turns):
        if blocks and blocks[-1]["role"] == t["role"]:
            blocks[-1]["text"] += sep + t["text"]
            blocks[-1]["end_idx"] = idx
        else:
            blocks.append(
                {
                    "role": t["role"],
                    "text": t["text"],
                    "start_idx": idx,
                    "end_idx": idx,
                }
            )
    return blocks


def pair_turns(turns: List[Dict], sep: str = "") -> List[Dict]:
    """合并连续对话，输出 QA 对。"""
    blocks = merge_consecutive(turns, sep=sep)
    pairs: List[Dict] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if b["role"] != "客户":
            i += 1  # 开头的孤立催收员块，跳过
            continue

        # 客户块后面紧跟催收员块 → 组成一对
        if i + 1 < len(blocks) and blocks[i + 1]["role"] == "催收员":
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
            # 客户块后面没有回答（会话结尾）
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


class LLMLabeler:
    def __init__(self):
        cfg = get_settings()
        self.client = OpenAI(
            api_key=cfg.llm_api_key, base_url=cfg.llm_base_url, timeout=60
        )
        with open(cfg.prompt_file, "r", encoding="utf-8") as f:
            self.prompts = yaml.safe_load(f)

    @retry()
    def _call(self, prompt):
        resp = self.client.chat.completions.create(
            model=get_settings().llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        log.debug(f"prompt: {prompt}")
        log.debug(f"response: {resp.choices[0].message.content}")
        return safe_json_loads(resp.choices[0].message.content)

    def _fmt(self, turns):
        return "\n".join([f"{t['role']}: {t['text']}" for t in turns])

    def run_sessions(self, calls):
        def label(call):
            text = "\n".join([f"{t['role']}: {t['text']}" for t in call["turns"]])
            try:
                prompt = (
                    self.prompts["session_prompt"]
                    .replace("{session_text}", text)
                    .replace("{sop_stages}", "、".join(raw_config.SOP_STAGES))
                    .replace("{session_reasons}", "、".join(raw_config.SESSION_REASONS))
                    .replace("{unified_intents}", "、".join(raw_config.UNIFIED_INTENTS))
                    .replace(
                        "{session_emotions}", "、".join(raw_config.SESSION_EMOTIONS)
                    )
                    .replace(
                        "{session_compliance}", "、".join(raw_config.SESSION_COMPLIANCE)
                    )
                    .replace(
                        "{session_outcomes}", "、".join(raw_config.SESSION_OUTCOMES)
                    )
                )
                res = self._call(prompt)
            except:
                res = {"ptp": False, "final_outcomes": []}
            res["call_id"] = call["call_id"]
            return res

        results = parallel_map(
            label, calls, get_settings().llm_max_concurrency, "Session打标"
        )

        # === 补回：保存 Session Labels 到 output ===
        out_file = get_settings().session_labels_file
        with open(out_file, "w", encoding="utf-8") as f:
            for r in results:
                if r:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info(f"✅ Session Labels 已保存至 {out_file}")

        return results

    def run_pairs(self, calls):
        log.info(f"Pair 打标与重写: {len(calls)} 通")
        tasks = []
        for c in calls:
            turns = c["turns"]
            for p in pair_turns(turns):
                chunk = f"客户: {p['question']}\n催收员: {p['answer']}"
                # sess 只保留当前 chunk 之前的对话内容
                prev_turns = turns[: min(p["q_idx"])]
                sess_text = self._fmt(prev_turns) if prev_turns else ""
                tasks.append(
                    {
                        "cid": c["call_id"],
                        "q_idx": min(p["q_idx"]) if p["q_idx"] else 0,
                        "a_idx": min(p["a_idx"]) if p["a_idx"] else 0,
                        "sess": sess_text,
                        "chunk": chunk,
                        "q": p["question"],
                        "a": p["answer"],
                    }
                )

        def label(t):
            try:
                prompt = (
                    self.prompts["pair_prompt"]
                    .replace("{session_content}", t["sess"])
                    .replace("{chunk_content}", t["chunk"])
                    .replace("{sop_stages}", "、".join(raw_config.SOP_STAGES))
                    .replace("{unified_intents}", "、".join(raw_config.UNIFIED_INTENTS))
                    .replace(
                        "{pair_risk_levels}", "、".join(raw_config.PAIR_RISK_LEVELS)
                    )
                    .replace("{pair_risk_types}", "、".join(raw_config.PAIR_RISK_TYPES))
                    .replace("{pair_keywords}", "、".join(raw_config.PAIR_KEYWORDS))
                )
                res = self._call(prompt)
            except:
                res = {
                    "stage": raw_config.SOP_STAGES[0],
                    "customer_intents": ["其他"],
                    "risk": "None",
                    "rewritten_question": t["q"],
                    "rewritten_answer": t["a"],
                }

            if res.get("stage") not in raw_config.SOP_STAGES:
                res["stage"] = raw_config.SOP_STAGES[0]

            intents = res.get("customer_intents", [])
            valid_intents = [it for it in intents if it in raw_config.UNIFIED_INTENTS]
            if not valid_intents:
                valid_intents = ["其他"]

            res["customer_intents"] = list(set(valid_intents))
            res["primary_intent"] = res["customer_intents"][0]

            rq = res.get("rewritten_question", "").strip()
            ra = res.get("rewritten_answer", "").strip()
            if not rq.startswith("客户："):
                rq = "客户：" + rq.replace("客户:", "").strip()
            if not ra.startswith("催收员："):
                ra = "催收员：" + ra.replace("催收员:", "").strip()

            res["rewritten_question"] = rq
            res["rewritten_answer"] = ra
            res["call_id"] = t["cid"]
            return res

        results = [
            r
            for r in parallel_map(
                label, tasks, get_settings().llm_max_concurrency, "Pair打标"
            )
            if r
        ]

        # === 补回：保存 Pair Labels 到 output ===
        out_file = get_settings().pair_labels_file
        with open(out_file, "w", encoding="utf-8") as f:
            for r in results:
                if r:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info(f"✅ Pair Labels 已保存至 {out_file}")

        return results


class PgvectorWriter:
    def __init__(self):
        cfg = get_settings()
        self.model = SentenceTransformer(
            cfg.embedding_model, device=cfg.embedding_device
        )

    def write(self, pairs, sessions):
        sess_map = {s["call_id"]: s for s in sessions if s}
        ptp_strict_stages = {raw_config.S.NEGOTIATE, raw_config.S.PTP}

        valid = []
        for p in pairs:
            if not p or p.get("risk") == "High" or not p.get("rewritten_question"):
                continue
            sess = sess_map.get(p["call_id"], {})
            is_ptp = sess.get("ptp", False)
            if p.get("stage") in ptp_strict_stages and not is_ptp:
                continue
            p["session_ptp"] = is_ptp
            p["session_outcomes"] = sess.get("final_outcomes", [])
            valid.append(p)

        log.info(f"[入库] 过滤后保留 {len(valid)} 条 Pairs 话术")

        for i in range(0, len(valid), 500):
            batch = valid[i : i + 500]
            texts = [
                f"[{p['stage']}][{p['primary_intent']}] {p['rewritten_question']}"
                for p in batch
            ]
            embs = self.model.encode(texts, normalize_embeddings=True)
            data = [
                (
                    p["call_id"],
                    p["stage"],
                    p["primary_intent"],
                    json.dumps(p.get("tags", []), ensure_ascii=False),
                    p.get("risk", "None"),
                    p["session_ptp"],
                    json.dumps(p["session_outcomes"], ensure_ascii=False),
                    p["rewritten_question"],
                    p["rewritten_answer"],
                    embs[j],
                )
                for j, p in enumerate(batch)
            ]

            log.debug(f"[入库] 批量入库 {len(data)} 条数据")
            for idx, row in enumerate(data):
                # row[-1] 是 embedding，日志里跳过
                log.debug(f"[{idx}] {row[:-1]}")

            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    execute_values(
                        cur,
                        """
                        INSERT INTO qa_pairs (call_id, stage, primary_intent, tags, risk, session_ptp, session_outcomes, rewritten_question, rewritten_answer, embedding) VALUES %s
                    """,
                        data,
                    )
                conn.commit()
            except Exception as e:
                log.error(f"入库失败: {e}")
                conn.rollback()
            finally:
                release_db_conn(conn)


class GoldenScriptGenerator:
    def __init__(self):
        cfg = get_settings()
        self.client = OpenAI(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url)
        with open(cfg.prompt_file, "r", encoding="utf-8") as f:
            self.prompts = yaml.safe_load(f)

    @retry()
    def _call(self, prompt):
        resp = self.client.chat.completions.create(
            model=get_settings().llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        log.debug(f"prompt: {prompt}")
        log.debug(f"response: {resp.choices[0].message.content}")
        return safe_json_loads(resp.choices[0].message.content)

    def synthesize(self, stage, intent):
        conn = get_db_conn()
        mode = None
        try:
            with conn.cursor() as cur:
                # 一级：精确匹配
                cur.execute(
                    """
                    SELECT rewritten_question, rewritten_answer, tags FROM (
                        SELECT *, ROW_NUMBER() OVER(PARTITION BY tags::text ORDER BY RANDOM()) as rn
                        FROM qa_pairs WHERE stage=%s AND primary_intent=%s AND risk != 'High'
                    ) sub WHERE rn <= 3 LIMIT 20
                """,
                    (stage, intent),
                )
                rows = cur.fetchall()
                mode = "exact"

                # 二级：阶段模糊匹配
                if not rows:
                    log.warning(
                        f"  ⚠️ [{stage}] 下的 [{intent}] 无专属语料，触发模糊兜底..."
                    )
                    cur.execute(
                        """
                        SELECT rewritten_question, rewritten_answer, tags FROM (
                            SELECT *, ROW_NUMBER() OVER(PARTITION BY primary_intent ORDER BY RANDOM()) as rn
                            FROM qa_pairs WHERE stage=%s AND risk != 'High'
                        ) sub WHERE rn <= 2 LIMIT 10
                    """,
                        (stage,),
                    )
                    rows = cur.fetchall()
                    mode = "fuzzy"

                # # 三级：全库随机兜底
                # if not rows:
                #     log.error(f"  🚨 [{stage}] 整个阶段完全无数据！触发全库随机兜底...")
                #     cur.execute("""
                #         SELECT rewritten_question, rewritten_answer, tags FROM qa_pairs
                #         WHERE risk = 'None' ORDER BY RANDOM() LIMIT 5
                #     """)
                #     rows = cur.fetchall()
                #     mode = "random"

        finally:
            release_db_conn(conn)

        if not rows:
            return mode, []

        samples = "\n".join(
            [f"- 客户: {r[0]}\n  坐席: {r[1]}\n  标签: {r[2]}" for r in rows]
        )

        try:
            res = self._call(
                self.prompts["golden_script_prompt"]
                .replace("{stage}", stage)
                .replace("{intent}", intent)
                .replace("{samples}", samples)
            )
            return mode, res.get("golden_qa_pairs", [])
        except:
            return mode, []

    def build_graph(self):
        graph = {"nodes": {}, "edges": []}
        for stage, intents in raw_config.SOP_FLOW.items():
            node = {}
            for intent in intents:
                mode, data = self.synthesize(stage, intent)
                node[intent] = {"mode": mode, "golden_qa_pairs": data} if data else None
            graph["nodes"][stage] = {"intents": node}
        for (src, intent), tgt in raw_config.SOP_TRANSITIONS.items():
            graph["edges"].append(
                {"source": src, "target": tgt, "trigger_intent": intent}
            )
        return graph
