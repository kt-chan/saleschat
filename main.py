import os, time, json, shutil, argparse
import jieba
from settings import get_settings
from logger import get_logger
from offline import (
    load_calls,
    LLMLabeler,
    BERTopicBuilder,
    QAExtractor,
    SOPGraphBuilder,
    VectorIndexBuilder,
)
from online import RealtimeRecommender

log = get_logger(__name__)


def clear_cache():
    cfg = get_settings()
    for p in [cfg.cache_path, cfg.output_path]:
        if p.exists():
            for item in p.iterdir():
                shutil.rmtree(item) if item.is_dir() else item.unlink()
    log.info("系统缓存已清空")


def offline_pipeline():
    cfg = get_settings()
    if cfg.clear_cache_on_start:
        clear_cache()

    log.info("阶段1: 加载解析会话")
    calls = load_calls(cfg.calls_file)
    if not calls:
        return
    if cfg.save_parsed_calls:
        with open(cfg.parsed_calls_path, "w", encoding="utf-8") as f:
            for c in calls:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    log.info("阶段2: LLM 双级打标与脱敏重写")
    labeler = LLMLabeler()
    sess_labels = labeler.run_sessions(calls, cfg.session_labels_file)
    pair_labels = labeler.run_pairs(calls, cfg.pair_labels_file)

    log.info("阶段3: 构建 BERTopic 导出向量")
    docs = [
        " ".join(jieba.cut(p["rewritten_question"]))
        for p in pair_labels
        if p and p.get("risk") != "High" and len(p.get("rewritten_question", "")) > 3
    ]
    if docs:
        BERTopicBuilder().build_and_export(docs)

    log.info("阶段4: 抽取脱敏 QA 库")
    qa_pairs = QAExtractor(calls, sess_labels, pair_labels).extract()
    with open(cfg.qa_pairs_file, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    log.info("阶段5: 构建 Node 图谱与向量索引")
    graph = SOPGraphBuilder(qa_pairs).build()
    with open(cfg.sop_graph_file, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    VectorIndexBuilder().build(qa_pairs)
    log.info("✅ 离线流程完成")


def online_demo():
    log.info("启动在线推荐演示")
    rec = RealtimeRecommender()
    dialog = [
        ("催收员", "喂，您好，招商银行。请问是张先生吗？"),
        ("客户", "是我。"),
        ("催收员", "您名下信用卡本期逾期了，最低还款6800元。"),
        ("客户", "我上个月被裁员了，现在没钱，能下个月还吗？"),
        ("催收员", "理解。那您可以先还一点表达意愿。"),
        ("客户", "那我找朋友凑一下，明天给你答复行不行？"),
    ]

    rec.init_session("demo")

    # 1) 把整段会话拼接成一个查询问题（保留角色标签，便于模型区分谁在说）
    query = "\n".join(f"{role}：{text}" for role, text in dialog)
    # 若希望更贴近“一个问题”，也可以在最前面加一句总结性提问，例如：
    # query = "以下是催收员与客户的完整对话，请判断当前阶段并推荐催收员下一句应说的话：\n" + query

    log.info(f"\n[拼接后的会话查询]\n{query}")

    res = rec.on_asr("demo", "客户", query)

    # 3) 统一输出一次结果
    if res:
        log.info(
            f"  👉 Stage: {res['stage']} | Intent: {res['intent']} "
            f"(判决: {res['decision']['decision']})"
        )
        for i, r in enumerate(res["recs"], 1):
            log.info(f"     {i}. [{r['score']}] {r['answer']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offline", "online", "all"], default="all")
    args = parser.parse_args()
    if args.mode in ["offline", "all"]:
        offline_pipeline()
    if args.mode in ["online", "all"]:
        online_demo()
