import json
from pathlib import Path
from typing import Iterator, List, Dict, Optional

import app.config as raw_config
from app.settings import get_settings
from app.logger import get_logger
from app.utils import (
    emb_to_bytes,
    invalidate_embedding_index,
    parallel_map,
    get_db_conn,
    release_db_conn,
    get_embedder,
    parse_dialog,
    format_turns,
    pair_turns,
)
from app.llm import LLMClient, get_llm_client

log = get_logger(__name__)

VALID_SUFFIXES = (".json", ".jsonl")
DEFAULT_CALL_BATCH_SIZE = 1000


# ---------------------------------------------------------------------------
# Streaming readers (replaces _read_raw_objs)
# ---------------------------------------------------------------------------
def _iter_raw_objs(f: Path) -> Iterator[Dict]:
    """逐条产出 raw obj。jsonl 流式；.json 仍整文件读（建议改 jsonl 或 ijson）。"""
    try:
        if f.suffix.lower() == ".jsonl":
            with open(f, "r", encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        yield json.loads(s)
                    except json.JSONDecodeError as e:
                        log.warning(f"跳过非法行 {f}:{lineno}: {e}")
        else:  # .json
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            objs = data if isinstance(data, list) else data.get("calls", [])
            for obj in objs:
                yield obj
    except Exception as e:
        log.error(f"解析文件失败 {f}: {e}")


def _collect_files(
    p: Path, pattern: Optional[str] = None, recursive: bool = False
) -> List[Path]:
    """把 path 归一为文件列表。目录模式下按 pattern（如 part_*.jsonl）过滤。"""
    if p.is_file():
        if p.suffix.lower() not in VALID_SUFFIXES:
            log.warning(f"非 json/jsonl 文件: {p}")
            return []
        return [p]
    if p.is_dir():
        pats = [pattern] if pattern else ["*.json", "*.jsonl"]
        it = p.rglob if recursive else p.glob
        seen: set[Path] = set()
        files: List[Path] = []
        for pat in pats:
            for f in it(pat):
                if f.is_file() and f.suffix.lower() in VALID_SUFFIXES and f not in seen:
                    seen.add(f)
                    files.append(f)
        files.sort()
        if not files:
            log.warning(
                f"目录下未找到匹配文件: {p} (pattern={pattern or '*.json|*.jsonl'})"
            )
        return files
    log.error(f"无效路径: {p}")
    return []


def load_calls(
    path: Optional[str] = None,
    pattern: Optional[str] = None,
    recursive: bool = False,
) -> Iterator[List[Dict]]:
    """
    签名与旧版一致；现在返回的是 **分批迭代器**：每批 <= call_batch_size 条。
    用法：
        for batch in load_calls(path, pattern, recursive):
            labeler.run_sessions(batch)
            ...
    如果旧代码是 calls = load_calls(...)，请改成 for batch in load_calls(...)。
    """
    target = path
    if not target:
        log.error("未提供 path 且 .env 中未配置 DATA_DIR")
        return
    p = Path(target)
    if not p.exists():
        log.error(f"文件/目录不存在: {target}")
        return

    files = _collect_files(p, pattern=pattern, recursive=recursive)
    if not files:
        return

    try:
        batch_size = int(
            getattr(get_settings(), "call_batch_size", DEFAULT_CALL_BATCH_SIZE)
            or DEFAULT_CALL_BATCH_SIZE
        )
    except Exception:
        batch_size = DEFAULT_CALL_BATCH_SIZE
    if batch_size <= 0:
        batch_size = DEFAULT_CALL_BATCH_SIZE

    log.info(f"流式加载: {len(files)} 个文件, batch_size={batch_size}")

    batch: List[Dict] = []
    total_valid = 0
    total_skipped = 0
    try:
        for f in files:
            for obj in _iter_raw_objs(f):
                try:
                    obj["turns"] = parse_dialog(obj.get("dialog", ""))
                except Exception as e:
                    log.warning(f"parse_dialog 失败 {f}: {e}")
                    total_skipped += 1
                    continue
                if len(obj["turns"]) >= 2:
                    batch.append(obj)
                    total_valid += 1
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
                else:
                    total_skipped += 1
        if batch:
            yield batch
            batch = []
    finally:
        log.info(
            f"加载完成: 有效 {total_valid} 通, 跳过 {total_skipped} 通, "
            f"文件 {len(files)} 个 (来源: {target}, pattern={pattern or '默认'})"
        )


# ---------------------------------------------------------------------------
# Labelers — signatures unchanged, output files now append (first call truncates)
# ---------------------------------------------------------------------------
class LLMLabeler:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or get_llm_client()
        self.cfg = self.llm.cfg
        # 首次写文件用 "w" 截断，之后批次用 "a" 追加
        self._session_file_initialized = False
        self._pair_file_initialized = False

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
                res = self.llm.call(
                    prompt, temperature=self.cfg.llm_temperature_labeling
                )
            except Exception as e:
                log.warning(f"Session labeling failed: {e}")
                res = {"ptp": False, "final_outcomes": []}
            res["call_id"] = call["call_id"]
            return res

        results = parallel_map(
            label, calls, self.cfg.llm_max_concurrency, "Session打标"
        )

        out_file = self.cfg.session_labels_file
        try:
            Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        mode = "a" if self._session_file_initialized else "w"
        with open(out_file, mode, encoding="utf-8") as f:
            n = 0
            for r in results:
                if r:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n += 1
        self._session_file_initialized = True
        log.info(f"Session Labels 追加 {n} 条 -> {out_file}")
        return results

    def run_pairs(
        self, calls: List[Dict], sessions: Optional[List[Dict]] = None
    ) -> List[Dict]:
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

                tasks.append(
                    {
                        "cid": c["call_id"],
                        "q_idx": pair_start,
                        "a_idx": min(p["a_idx"]) if p["a_idx"] else 0,
                        "sess": sess_text,
                        "chunk": chunk,
                        "q": p["question"],
                        "a": p["answer"],
                        "session_stage_labels": session_stage_labels,
                    }
                )

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
                res = self.llm.call(
                    prompt, temperature=self.cfg.llm_temperature_labeling
                )
            except Exception as e:
                log.warning(f"Pair labeling failed: {e}")
                res = {
                    "stage": raw_config.SOP_STAGES[0],
                    "customer_intents": [raw_config.I.OTHER],
                    "risk": "None",
                    "rewritten_question": t["q"],
                    "rewritten_answer": t["a"],
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
                    tgt
                    for (src, _), tgt in raw_config.SOP_TRANSITIONS.items()
                    if src == stage
                }
                for cs in candidate_stages & reachable:
                    aligned = [
                        it
                        for it in raw_intents
                        if it in raw_config.ALLOWED_INTENTS_BY_STAGE.get(cs, set())
                    ]
                    if aligned:
                        log.info(f"[align] stage {stage} → {cs} (intents: {aligned})")
                        stage, allowed, valid_intents = (
                            cs,
                            raw_config.ALLOWED_INTENTS_BY_STAGE[cs],
                            aligned,
                        )
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

        results = [
            r
            for r in parallel_map(
                label, tasks, self.cfg.llm_max_concurrency, "Pair打标"
            )
            if r
        ]

        out_file = self.cfg.pair_labels_file
        try:
            Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        mode = "a" if self._pair_file_initialized else "w"
        with open(out_file, mode, encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self._pair_file_initialized = True
        log.info(f"Pair Labels 追加 {len(results)} 条 -> {out_file}")
        return results


# ---------------------------------------------------------------------------
# Pgvector writer — unchanged
# ---------------------------------------------------------------------------
class PgvectorWriter:
    def __init__(self):
        self.cfg = get_settings()
        self.model = get_embedder()

    def write(self, pairs, sessions):
        sess_map = {s["call_id"]: s for s in sessions if s}
        ptp_strict_stages = {raw_config.S.NEGOTIATE, raw_config.S.PTP}
        ptp_strict_intents = {
            raw_config.I.SIGN,
            raw_config.I.PLAN_INSTALL,
            raw_config.I.PLAN_SETTLE,
            raw_config.I.VOICE_CONFIRM,
        }

        valid = []
        for p in pairs:
            if not p or p.get("risk") == "High" or not p.get("rewritten_question"):
                continue
            sess = sess_map.get(p["call_id"], {})
            is_ptp = sess.get("ptp", False)
            if (
                p.get("stage") in ptp_strict_stages
                and p.get("primary_intent") in ptp_strict_intents
                and not is_ptp
            ):
                continue
            p["session_ptp"] = is_ptp
            p["session_outcomes"] = sess.get("final_outcomes", [])
            valid.append(p)

        log.info(f"[入库] 过滤后保留 {len(valid)} 条 Pairs 话术")
        if not valid:
            return

        bs = max(1, self.cfg.db_write_batch_size)
        inserted = 0
        for i in range(0, len(valid), bs):
            batch = valid[i : i + bs]
            texts = [
                f"[{p['stage']}][{p['primary_intent']}] {p['rewritten_question']}"
                for p in batch
            ]
            embs = self.model.encode(texts, normalize_embeddings=True)
            if embs.ndim == 1:
                embs = embs.reshape(1, -1)

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
                    emb_to_bytes(embs[j]),
                )
                for j, p in enumerate(batch)
            ]

            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO qa_pairs
                            (call_id, stage, primary_intent, tags, risk, session_ptp,
                             session_outcomes, rewritten_question, rewritten_answer, embedding)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        data,
                    )
                conn.commit()
                inserted += len(data)
                log.info(f"[入库] batch {i // bs + 1} 写入 {len(data)} 条")
            except Exception as e:
                log.error(f"入库失败 (batch {i // bs + 1}): {e}")
                try:
                    conn.rollback()
                except Exception:
                    pass
            finally:
                release_db_conn(conn)

        if inserted > 0:
            invalidate_embedding_index()
            log.info(f"[入库] 本批写入 {inserted} 条，embedding 索引已失效")


# ---------------------------------------------------------------------------
# Golden script / SOP graph — unchanged
# ---------------------------------------------------------------------------
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
                    (
                        stage,
                        intent,
                        self.cfg.golden_exact_per_tag,
                        self.cfg.golden_exact_limit,
                    ),
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
                        (
                            stage,
                            self.cfg.golden_fuzzy_per_intent,
                            self.cfg.golden_fuzzy_limit,
                        ),
                    )
                    rows = cur.fetchall()
                    mode = "fuzzy"
        finally:
            release_db_conn(conn)

        if not rows:
            return "empty", []

        samples = "\n".join(
            [f"- 客户: {r[0]}\n  坐席: {r[1]}\n  标签: {r[2]}" for r in rows]
        )
        try:
            prompt = self.llm.render(
                "golden_script_prompt",
                stage=stage,
                intent=intent,
                samples=samples,
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
            graph["edges"].append(
                {"source": src, "target": tgt, "trigger_intent": intent}
            )

        graph["retrieval_index"] = {
            "stage_to_intents": {
                s: list(its) for s, its in raw_config.SOP_FLOW.items()
            },
            "intent_to_stages": {
                i: raw_config.INTENT_TO_STAGES.get(i, [])
                for i in raw_config.UNIFIED_INTENTS
            },
            "legal_edges": [
                {"from_stage": s, "intent": i, "to_stage": t}
                for (s, i), t in raw_config.SOP_TRANSITIONS.items()
            ],
        }
        return graph


# ---------------------------------------------------------------------------
# Optional convenience entry point. If you already have a main(), just copy
# the loop below — nothing new to import.
# ---------------------------------------------------------------------------
def run(
    path: Optional[str] = None,
    pattern: Optional[str] = None,
    recursive: bool = False,
):
    labeler = LLMLabeler()
    writer = PgvectorWriter()
    generator = GoldenScriptGenerator()

    total_calls = 0
    total_pairs = 0
    batch_idx = 0

    # Phase 1 — 流式分批：load_calls → session → pair → pgvector
    for batch in load_calls(path, pattern, recursive):
        batch_idx += 1
        log.info(f"[批次 {batch_idx}] 载入 {len(batch)} 通 calls")
        sessions = pairs = None
        try:
            sessions = labeler.run_sessions(batch)
            pairs = labeler.run_pairs(batch, sessions)
            total_calls += len(batch)
            total_pairs += len(pairs or [])
            writer.write(pairs, sessions)
        except Exception as e:
            log.exception(f"[批次 {batch_idx}] 处理失败: {e}")
        finally:
            # 打破引用，让本批内存尽快被 GC 回收
            del batch, sessions, pairs

    log.info(
        f"✅ 第 1 阶段完成: {batch_idx} 批, {total_calls} 通 calls, "
        f"{total_pairs} 条 pairs 已入库"
    )

    # Phase 2 — 全部入库完成之后，构建 SOP 图（读取 pgvector）
    log.info("开始第 2 阶段: SOP 图构建 ...")
    graph = generator.build_graph()
    log.info("✅ 第 2 阶段完成: SOP 图构建完成")
    return graph
