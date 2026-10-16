# main.py
"""主流程入口"""

import os
import sys
import time
import json
import shutil
import argparse
from typing import Dict, List

import jieba

from settings import get_settings
from logger import get_logger
from offline import (
    load_calls,
    save_calls,
    LLMLabeler,
    ConfigOptimizer,
    BERTopicBuilder,
    QAExtractor,
    VectorIndexBuilder,
    SOPGraphBuilder,
    prepare_docs_from_labels,
)
from online import build_testset, save_testset, RealtimeRecommender

log = get_logger(__name__)


def load_labels(path) -> list:
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(json.loads(line))
    return labels


def offline_pipeline():
    cfg = get_settings()
    t_total = time.time()

    # 检查并清空缓存    
    if cfg.clear_cache_on_start:
        clear_system_cache()
    
    log.info("=" * 60)
    log.info("阶段1: 加载会话")
    log.info("=" * 60)
    calls = load_calls(cfg.calls_file)
    if not calls:
        log.error("无会话数据，退出")
        return
    else:
        log.info(f"加载到 {len(calls)} 条会话")
        save_calls(calls, cfg.parsed_calls_path)
        
    labeler = LLMLabeler()

    # ---------- 阶段2a: Session 级打标 ----------
    log.info("=" * 60)
    log.info("阶段2a: Session 级打标")
    log.info("=" * 60)
    if cfg.session_labels_file.exists():
        session_labels = load_labels(cfg.session_labels_file)
        if not session_labels:
            session_labels = labeler.run_sessions(calls, cfg.session_labels_file)
    else:
        session_labels = labeler.run_sessions(calls, cfg.session_labels_file)

    # ---------- 阶段2b: Pair 级打标 ----------
    log.info("=" * 60)
    log.info("阶段2b: Pair 级打标")
    log.info("=" * 60)
    if cfg.pair_labels_file.exists():
        pair_labels = load_labels(cfg.pair_labels_file)
        if not pair_labels:
            pair_labels = labeler.run_pairs(calls, cfg.pair_labels_file)
    else:
        pair_labels = labeler.run_pairs(calls, cfg.pair_labels_file)

    # ---------- 阶段3: 配置优化（用 Pair 标签） ----------
    if cfg.config_optimize_enabled:
        log.info("=" * 60)
        log.info("阶段3: 配置优化")
        log.info("=" * 60)
        ConfigOptimizer(pair_labels, calls).run(cfg.optimized_config_file)

    # ---------- 阶段4: BERTopic（用 Pair 标签的 question 文本） ----------
    log.info("=" * 60)
    log.info("阶段4: BERTopic 构建")
    log.info("=" * 60)
    docs, _ = prepare_docs_from_pairs(calls, pair_labels)
    if not docs:
        log.error("文档为空，无法构建 BERTopic")
        return
    bt_builder = BERTopicBuilder()
    bt_builder.build(docs)
    bt_builder.save()

    # ---------- 阶段5: QA 抽取 ----------
    log.info("=" * 60)
    log.info("阶段5: QA 抽取（Session + Pair 标签）")
    log.info("=" * 60)
    extractor = QAExtractor(calls, session_labels, pair_labels)
    qa_pairs = extractor.extract()
    qa_pairs = extractor.dedup(qa_pairs)
    with open(cfg.qa_pairs_file, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    log.info(f"QA 库已保存: {cfg.qa_pairs_file} ({len(qa_pairs)} 条)")

    # ---------- 阶段6: SOP 图谱 ----------
    log.info("=" * 60)
    log.info("阶段6: SOP 图谱构建")
    log.info("=" * 60)
    graph_builder = SOPGraphBuilder(calls, session_labels, pair_labels, qa_pairs)
    graph_builder.save(graph_builder.build(), cfg.sop_graph_file)

    # ---------- 阶段7: 向量索引 ----------
    log.info("=" * 60)
    log.info("阶段7: 向量索引构建")
    log.info("=" * 60)
    VectorIndexBuilder().build(qa_pairs, cfg.faiss_dir)

    # ---------- 阶段8: 测试集 ----------
    log.info("=" * 60)
    log.info("阶段8: 构建测试集")
    log.info("=" * 60)
    save_testset(build_testset(pair_labels, calls), cfg.testset_file)

    log.info(f"✅ 离线流程完成，总耗时 {(time.time() - t_total) / 60:.1f} 分钟")


def prepare_docs_from_pairs(
    calls: List[Dict], pair_labels: List[Dict], min_quality: float = 0.5
):
    """从 Pair 标签中提取客户问题文本，用于 BERTopic"""
    call_map = {c["call_id"]: c for c in calls}
    docs = []
    for p in pair_labels:
        call = call_map.get(p["call_id"])
        if not call:
            continue
        # 用 QA 抽取时的质量分过滤
        if p.get("risk") == "High":
            continue
        q_idx = p.get("q_idx")
        if q_idx is None or q_idx >= len(call["turns"]):
            continue
        t = call["turns"][q_idx]
        if t["role"] != "客户" or len(t["text"]) < 4:
            continue
        docs.append(" ".join(jieba.cut(t["text"])))
    log.info(f"准备 {len(docs)} 条文档（来自 Pair 级）")
    return docs, None


def online_demo():
    log.info("=" * 60)
    log.info("在线推荐演示")
    log.info("=" * 60)

    recommender = RealtimeRecommender()
    demo_dialog = [
        ("催收员", "喂，您好，招商银行信用卡中心，请问是张先生吗？"),
        ("客户", "是，你好。"),
        (
            "催收员",
            "张先生您好，您名下信用卡本期已经逾期了，最低还款是6800元，请问今天能处理吗？",
        ),
        ("客户", "我上个月被裁员了，N+1还没发，现在没收入，真的还不上。"),
        ("催收员", "理解，您之前还款一直比较稳定，这次是遇到困难了。"),
        ("客户", "6800我拿不出来，我现在手里只有2000多，还要交房租。"),
        ("催收员", "2000多不够最低还款，但您可以先还进来表达还款意愿。"),
        ("客户", "能不能把我整个账单做分期？"),
        ("催收员", "目前没有总账单分期方案。"),
        ("客户", "那我找朋友凑一下，明天给你答复行不行？"),
    ]
    session_id = "demo_001"
    recommender.init_session(session_id)

    for role, text in demo_dialog:
        log.info(f"[{role}] {text}")
        result = recommender.on_asr(session_id, role, text)
        if result:
            if result.get("ptp"):
                log.info(f"  ⭐ PTP: {result['ptp']}")
            for i, r in enumerate(result["recommendations"], 1):
                log.info(
                    f"  #{i} [{r['score']:.3f}] {r['answer'][:60]}... "
                    f"(来源={r['sources']})"
                )


def clear_system_cache():
    """清空系统缓存和输出目录"""
    cfg = get_settings()
    log.info("=" * 60)
    log.info("⚠️ CLEAR_CACHE_ON_START 为 True，正在清空系统缓存和输出...")

    # 需要清空的目录列表：缓存目录和输出目录
    target_dirs = [cfg.cache_path, cfg.output_path]

    for target_dir in target_dirs:
        if target_dir.exists():
            # 遍历目录下的所有内容并删除
            for item in target_dir.iterdir():
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                except Exception as e:
                    log.error(f"清空 {item} 失败: {e}")
            log.info(f"  ✅ 已清空目录: {target_dir}")
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"  ✅ 目录不存在，已创建: {target_dir}")

    log.info("系统缓存清空完成，开始重新构建...")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offline", "online", "all"], default="all")
    args = parser.parse_args()

    if args.mode in ("offline", "all"):
        offline_pipeline()
    if args.mode in ("online", "all"):
        online_demo()
