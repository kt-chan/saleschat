import json
from pathlib import Path
from typing import List, Dict, Optional

from psycopg2.extras import execute_values

import config as raw_config
from settings import get_settings
from logger import get_logger
from utils import (
    parallel_map, get_db_conn, release_db_conn, get_embedder,
    parse_dialog, format_turns, pair_turns,
)
from llm import LLMClient, get_llm_client

log = get_logger(__name__)


def load_calls(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        log.error(f"文件不存在: {path}")
        return []
    raw_objs: List[Dict] = []
    if p.suffix.lower() == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_objs.append(json.loads(line.strip()))
    else:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            raw_objs = data if isinstance(data, list) else data.get("calls", [])

    calls, skipped = [], 0
    for obj in raw_objs:
        obj["turns"] = parse_dialog(obj.get("text", ""))
        if len(obj["turns"]) >= 2:
            calls.append(obj)
        else:
            skipped += 1
    log.info(f"加载完成: 有效 {len(calls)} 通, 跳过 {skipped} 通")
    return calls


class LLMLabeler:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or get_llm_client()
        self.cfg = self.llm.cfg

    def run_sessions(self, calls: List[Dict]) -> List[Dict]:
        def label(call):
            text = format_turns(call["turns"])
            try:
                prompt = self.llm.render(
                    "session_prompt",
                    session_text=text,
                    sop_stages="、".join(raw_config.SOP_STAGES),
                    session_reasons="、".join(raw_config.SESSION_REASONS),
                    unified_intents="、".join(raw_config.UNIFIED_INTENTS),
                    session_emotions="、".join(raw_config.SESSION_EMOTIONS),
                    session_compliance="、".join(raw_config.SESSION_COMPLIANCE),
                    session_outcomes="、".join(raw_config.SESSION_OUTCOMES),
                )
                res = self.llm.call(prompt, temperature=self.cfg.llm_temperature_labeling)
            except Exception as e:
                log.warning(f"Session labeling failed: {e}")
                res = {"ptp": False, "final_outcomes": []}
            res["call_id"] = call["call_id"]
            return res

        results = parallel_map(label, calls, self.cfg.llm_max_concurrency, "Session打标")

        out_file = self.cfg.session_labels_file
        with open(out_file, "w", encoding="utf-8") as f:
            for r in results:
                if r:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info(f"✅ Session Labels 已保存至 {out_file}")
        return results

    def run_pairs(self, calls: List[Dict], sessions: Optional[List[Dict]] = None) -> List[Dict]:
        log.info(f"Pair 打标与重写: {len(calls)} 通")
        sess_map = {s["call_id"]: s for s in (sessions or []) if s}
        cust_role = self.cfg.dialog_customer_role
        agent_role = self.cfg.dialog_agent_role

        tasks: List[Dict] = []
        for c in calls:
            turns = c["turns"]
            sess_data = sess_map.get(c["call_id"], {})
            sess_stages = sess_data.get("sop_stages", []) or []

            for p in pair_turns(turns, customer_role=cust_role):
                chunk = f"{cust_role}: {p['question']}\n{agent_role}: {p['answer']}"
                prev_turns = turns[: min(p["q_idx"])]
                sess_text = format_turns(prev_turns) if prev_turns else ""

                pair_start = min(p["q_idx"])
                pair_end = max(p["a_idx"]) if p["a_idx"] else max(p["q_idx"])
                overlapping = []
                for st in sess_stages:
                    ts = st.get("turn_start", 0)
                    te = st.get("turn_end", ts)
                    if not (te < pair_start or ts > pair_end):
                        overlapping.append(st.get("stage_name", ""))
                session_stage_labels = "、".join([x for x in overlapping if x]) or "无"

                tasks.append({
                    "cid": c["call_id"],
                    "q_idx": pair_start,
                    "a_idx": min(p["a_idx"]) if p["a_idx"] else 0,
                    "sess": sess_text,
                    "chunk": chunk,
                    "q": p["question"],
                    "a": p["answer"],
                    "session_stage_labels": session_stage_labels,
                })

        def label(t):
            try:
                prompt = self.llm.render(
                    "pair_prompt",
                    session_content=t["sess"],
                    chunk_content=t["chunk"],
                    sop_stages="、".join(raw_config.SOP_STAGES),
                    unified_intents="、".join(raw_config.UNIFIED_INTENTS),
                    sop_flow=raw_config.SOP_FLOW_JSON,
                    session_stage_labels=t["session_stage_labels"],
                    pair_risk_levels="、".join(raw_config.PAIR_RISK_LEVELS),
                    pair_risk_types="、".join(raw_config.PAIR_RISK_TYPES),
                    pair_keywords="、".join(raw_config.PAIR_KEYWORDS),
                    pair_emotions="、".join(raw_config.PAIR_EMOTIONS),
                )
                res = self.llm.call(prompt, temperature=self.cfg.llm_temperature_labeling)
            except Exception as e:
                log.warning(f"Pair labeling failed: {e}")
                res = {
                    "stage": raw_config.SOP_STAGES[0],
                    "customer_intents": [raw_config.I.OTHER], "risk": "None",
                    "rewritten_question": t["q"], "rewritten_answer": t["a"],
                }

            stage = res.get("stage", "")
            if stage not in raw_config.SOP_STAGES:
                stage = raw_config.SOP_STAGES[0]

            allowed = raw_config.ALLOWED_INTENTS_BY_STAGE.get(stage, set())
            raw_intents = res.get("customer_intents", []) or []
            valid_intents = [it for it in raw_intents if it in allowed]

            if not valid_intents and raw_intents:
                candidate_stages = set()
                for it in raw_intents:
                    candidate_stages.update(raw_config.INTENT_TO_STAGES.get(it, []))
                reachable = {stage} | {
                    tgt for (src, _), tgt in raw_config.SOP_TRANSITIONS.items() if src == stage
                }
                for cs in (candidate_stages & reachable):
                    aligned = [it for it in raw_intents
                               if it in raw_config.ALLOWED_INTENTS_BY_STAGE.get(cs, set())]
                    if aligned:
                        log.info(f"[align] stage {stage} → {cs} (intents: {aligned})")
                        stage, allowed, valid_intents = cs, raw_config.ALLOWED_INTENTS_BY_STAGE[cs], aligned
                        break

            if not valid_intents:
                valid_intents = [raw_config.SOP_FLOW[stage][0]]

            res["stage"] = stage
            res["customer_intents"] = list(dict.fromkeys(valid_intents))
            res["primary_intent"] = res["customer_intents"][0]
            res["session_stage_labels"] = t["session_stage_labels"]

            cpre, apre = self.cfg.prompt_customer_prefix, self.cfg.prompt_agent_prefix
            rq = (res.get("rewritten_question") or "").strip()
            ra = (res.get("rewritten_answer") or "").strip()
            if not rq.startswith(cpre):
                rq = cpre + rq.replace("客户:", "").replace("客户：", "").strip()
            if not ra.startswith(apre):
                ra = apre + ra.replace("催收员:", "").replace("催收员：", "").strip()
            res["rewritten_question"] = rq
            res["rewritten_answer"] = ra
            res["call_id"] = t["cid"]
            return res

        results = [r for r in parallel_map(label, tasks, self.cfg.llm_max_concurrency, "Pair打标") if r]

        out_file = self.cfg.pair_labels_file
        with open(out_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info(f"✅ Pair Labels 已保存至 {out_file}")
        return results


class PgvectorWriter:
    def __init__(self):
        self.cfg = get_settings()
        self.model = get_embedder()

    def write(self, pairs, sessions):
        sess_map = {s["call_id"]: s for s in sessions if s}
        ptp_strict_stages = {raw_config.S.NEGOTIATE, raw_config.S.PTP}
        ptp_strict_intents = {
            raw_config.I.SIGN, raw_config.I.PLAN_INSTALL,
            raw_config.I.PLAN_SETTLE, raw_config.I.VOICE_CONFIRM,
        }

        valid = []
        for p in pairs:
            if not p or p.get("risk") == "High" or not p.get("rewritten_question"):
                continue
            sess = sess_map.get(p["call_id"], {})
            is_ptp = sess.get("ptp", False)
            if (p.get("stage") in ptp_strict_stages
                    and p.get("primary_intent") in ptp_strict_intents
                    and not is_ptp):
                continue
            p["session_ptp"] = is_ptp
            p["session_outcomes"] = sess.get("final_outcomes", [])
            valid.append(p)

        log.info(f"[入库] 过滤后保留 {len(valid)} 条 Pairs 话术")

        bs = self.cfg.db_write_batch_size
        for i in range(0, len(valid), bs):
            batch = valid[i: i + bs]
            texts = [f"[{p['stage']}][{p['primary_intent']}] {p['rewritten_question']}" for p in batch]
            embs = self.model.encode(texts, normalize_embeddings=True)
            data = [
                (p["call_id"], p["stage"], p["primary_intent"],
                 json.dumps(p.get("tags", []), ensure_ascii=False),
                 p.get("risk", "None"), p["session_ptp"],
                 json.dumps(p["session_outcomes"], ensure_ascii=False),
                 p["rewritten_question"], p["rewritten_answer"], embs[j])
                for j, p in enumerate(batch)
            ]
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    execute_values(
                        cur,
                        """
                        INSERT INTO qa_pairs
                          (call_id, stage, primary_intent, tags, risk, session_ptp,
                           session_outcomes, rewritten_question, rewritten_answer, embedding)
                        VALUES %s
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
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or get_llm_client()
        self.cfg = self.llm.cfg

    def synthesize(self, stage: str, intent: str):
        conn = get_db_conn()
        mode, rows = None, []
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT rewritten_question, rewritten_answer, tags FROM (
                        SELECT *, ROW_NUMBER() OVER(PARTITION BY tags::text ORDER BY RANDOM()) as rn
                        FROM qa_pairs WHERE stage=%s AND primary_intent=%s AND risk != 'High'
                    ) sub WHERE rn <= %s LIMIT %s
                    """,
                    (stage, intent, self.cfg.golden_exact_per_tag, self.cfg.golden_exact_limit),
                )
                rows = cur.fetchall()
                mode = "exact"

                if not rows:
                    cur.execute(
                        """
                        SELECT rewritten_question, rewritten_answer, tags FROM (
                            SELECT *, ROW_NUMBER() OVER(PARTITION BY primary_intent ORDER BY RANDOM()) as rn
                            FROM qa_pairs WHERE stage=%s AND risk != 'High'
                        ) sub WHERE rn <= %s LIMIT %s
                        """,
                        (stage, self.cfg.golden_fuzzy_per_intent, self.cfg.golden_fuzzy_limit),
                    )
                    rows = cur.fetchall()
                    mode = "fuzzy"
        finally:
            release_db_conn(conn)

        if not rows:
            return "empty", []

        samples = "\n".join([f"- 客户: {r[0]}\n  坐席: {r[1]}\n  标签: {r[2]}" for r in rows])
        try:
            prompt = self.llm.render(
                "golden_script_prompt", stage=stage, intent=intent, samples=samples,
            )
            res = self.llm.call(prompt, temperature=self.cfg.llm_temperature_synthesis)
            return mode, res.get("golden_qa_pairs", [])
        except Exception as e:
            log.warning(f"Golden synthesis failed for {stage}/{intent}: {e}")
            return mode, []

    def build_graph(self):
        graph = {"nodes": {}, "edges": [], "retrieval_index": {}}
        for stage, intents in raw_config.SOP_FLOW.items():
            node = {}
            for intent in intents:
                mode, data = self.synthesize(stage, intent)
                node[intent] = {"mode": mode, "golden_qa_pairs": data} if data else None
            graph["nodes"][stage] = {"intents": node}
        for (src, intent), tgt in raw_config.SOP_TRANSITIONS.items():
            graph["edges"].append({"source": src, "target": tgt, "trigger_intent": intent})

        graph["retrieval_index"] = {
            "stage_to_intents": {s: list(its) for s, its in raw_config.SOP_FLOW.items()},
            "intent_to_stages": {i: raw_config.INTENT_TO_STAGES.get(i, []) for i in raw_config.UNIFIED_INTENTS},
            "legal_edges": [{"from_stage": s, "intent": i, "to_stage": t}
                            for (s, i), t in raw_config.SOP_TRANSITIONS.items()],
        }
        return graph