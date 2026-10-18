import json, shutil, argparse
from settings import get_settings
from logger import get_logger
from utils import init_db, clear_system_data
from offline import load_calls, LLMLabeler, PgvectorWriter, GoldenScriptGenerator
from online import RealtimeRecommender
import config as raw_config

log = get_logger(__name__)


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
    pairs = labeler.run_pairs(calls)

    log.info("=== 阶段3: 向量化入库 ===")
    PgvectorWriter().write(pairs, sessions)

    log.info("=== 阶段4: 图谱合成经典 Q&A (自动落盘) ===")
    graph = GoldenScriptGenerator().build_graph()
    with open(cfg.sop_graph_file, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    log.info(f"✅ SOP 图谱已保存至 {cfg.sop_graph_file}")

    log.info("🎉 离线流程全部完成")


def online_demo():
    log.info("=== 在线实时推理测试 (多轮合并 & 配置控制状态机) ===")
    rec = RealtimeRecommender()

    # 客户：
    testquestion = "催收员告知可申请延期3天处理剩余1900元相关金额并申请减免违约金）后天？后天我也不敢保证能借到。"
    dialog = [
        ("坐席", "您好，请问是张先生吗？"),
        ("客户", "是我。"),
        ("坐席", "您名下信用卡本期逾期1900元，请问能处理吗？"),
        ("客户", "我最近被裁员了，没钱。"),
        ("坐席", "理解。那能不能找朋友凑一下？"),
        ("客户", testquestion),
    ]

    print("\n--- 历史会话过程 ---")
    for role, text in dialog:
        print(f"[{role}] {text}")

    # 提取客户的每一句话，合并成一个大 Query
    customer_statements = [f"{role}:{text}" for role, text in dialog]
    merged_query = " ".join(customer_statements)

    print(f"\n🔍 触发多轮合并一次查询: {merged_query}")

    sid = "demo_session"
    # 初始化会话状态为起点
    rec.store.set(sid, {"stage": raw_config.SOP_STAGES[0], "history": []})

    # 把合并后的客户发言，作为单次客户输入丢给 on_asr
    res = rec.on_asr(sid, "客户", merged_query)

    if res:
        print(
            f"\n👉 最终命中阶段: {res['stage']} | 意图: {res['intent']} | 耗时: {res['latency_ms']}ms"
        )
        if res.get("reject"):
            print(f"  🚫 拒答原因: {res['reject_reason']}")
        elif res["recs"]:
            print(f"  🏆 推荐经典 Q&A:")
            print(f"     Q: {res['recs'][0]['question']}")
            print(f"     A: {res['recs'][0]['answer']}")
        elif res["fallback"]:
            print(f"  ⚠️ 兜底话术: {res['fallback'][0]['answer']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offline", "online"], default="offline")
    args = parser.parse_args()
    if args.mode == "offline":
        offline_pipeline()
    elif args.mode == "online":
        online_demo()
