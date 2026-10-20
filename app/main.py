import json, time, argparse
from pathlib import Path
from typing import Optional

from app.settings import get_settings
from app.logger import get_logger
from app.utils import init_db, clear_system_data, parse_dialog
from app.offline import load_calls, LLMLabeler, PgvectorWriter, GoldenScriptGenerator
from app.online import HybridRetriever, get_retriever, suggest_script_for_conversation

log = get_logger(__name__)
retriever: Optional[HybridRetriever] = None


_DEFAULT_DEMO_TEXT = (
    "催收员：您好，请问是张先生吗？\n"
    "客户：是我。\n"
    "催收员：您名下信用卡本期逾期1900元，请问能处理吗？\n"
    "客户：我最近被裁员了，暂时没有收入。\n"
    "坐席：理解您的困难。那能不能找朋友先凑一下呢？\n"
    "客户：朋友也不宽裕。你们刚说可以延期几天并减免违约金，后天我也不敢保证能借到。\n"
    "坐席：那我们先登记诉求，后续再回访确认，可以吗？\n"
    "客户：可以的。"
)


def _load_demo_text() -> str:
    cfg = get_settings()
    if cfg.demo_text:
        return cfg.demo_text
    p = Path(cfg.demo_text_file)
    if p.exists():
        return p.read_text(encoding="utf-8")
    log.warning(f"Demo 文本文件不存在 ({p})，使用内置默认示例")
    return _DEFAULT_DEMO_TEXT


def offline_pipeline():
    cfg = get_settings()
    if cfg.clear_cache_on_start:
        clear_system_data()

    init_db()

    log.info("=== 阶段1: 解析原始会话 ===")
    calls = load_calls(cfg.calls_file)
    if not calls:
        return

    log.info("=== 阶段2: LLM 双级打标 (自动落盘) ===")
    labeler = LLMLabeler()
    sessions = labeler.run_sessions(calls)
    pairs = labeler.run_pairs(calls, sessions)

    log.info("=== 阶段3: 向量化入库 ===")
    PgvectorWriter().write(pairs, sessions)

    log.info("=== 阶段4: 图谱合成经典 Q&A (自动落盘) ===")
    graph = GoldenScriptGenerator().build_graph()
    with open(cfg.sop_graph_file, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    log.info(f"✅ SOP 图谱已保存至 {cfg.sop_graph_file}")
    log.info("🎉 离线流程全部完成")


def setup_retriever():
    global retriever

    if retriever is None:
        try:
            retriever = get_retriever()
        except FileNotFoundError as e:
            log.error(str(e))
            return


def online_pipeline(text, label, hint):
    log.info("=== suggest_script_for_conversation Demo ===")
    setup_retriever()

    print("\n" + "=" * 72)
    print(f"🔎 {label}")
    print("=" * 72)

    t0 = time.time()
    res = suggest_script_for_conversation(text, stage_hint=hint)
    wall = (time.time() - t0) * 1000

    if res["reject"]:
        print(f"  🚫 reject={res['reject_reason']}")
        print(f"  💬 Fallback: {res['script']}")
        print(f"  ⏱  {wall:.1f}ms")
        if res.get("debug"):
            print(f"  🔍 debug: {res['debug']}")
        return

    d = res.get("debug") or {}
    print(f"  📍 Stage:    {res['stage']}")
    print(f"  📍 Intent:   {res['intent']}")
    print(f"  📍 Source:   {res['source']} (score={res['score']})")
    print(f"  📍 Path:     {d.get('path')} | passes={d.get('passes')}")
    if d.get("stage_vote_1"):
        print(
            f"  📍 Pass 1:   stage={d.get('stage_vote_1')} intent={d.get('intent_vote_1')}"
        )
    print(f"\n  💬 Script for 催收员:")
    print(f"     {res['script']}")
    if res["question"]:
        print(f"\n  ❓ Reference Q:")
        print(f"     {res['question']}")
    if res["alternates"]:
        print(f"\n  🔄 Alternates ({len(res['alternates'])}):")
        for i, alt in enumerate(res["alternates"][:3], 1):
            print(f"     [{i}] {alt['answer']}")
            print(f"         (source={alt['source']} score={alt['score']})")
    print(f"\n  ⏱  {wall:.1f}ms")


def online_demo():

    text = _load_demo_text()
    turns = parse_dialog(text)

    print("\n" + "=" * 72)
    print("📞 Input conversation (from call-center detector):")
    print("=" * 72)
    for t in turns:
        print(f"  [{t['turn_id']:>2}] {t['role']}: {t['text']}")

    for label, hint in [
        ("Fast path (stage_hint='04 方案协商与签署')", "04 方案协商与签署"),
        ("Slow path (no hint — infer from conversation)", None),
    ]:
        online_pipeline(text, label, hint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offline", "online"], default="offline")
    args = parser.parse_args()
    if args.mode == "offline":
        offline_pipeline()
    elif args.mode == "online":
        online_demo()
