# evaluate.py
"""在线检索评测"""
import json
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from app.settings import get_settings
from app.logger import get_logger
from app.online import RealtimeRecommender

log = get_logger(__name__)


def evaluate(K: int = 3, sim_threshold: float = 0.75):
    cfg = get_settings()
    if not cfg.testset_file.exists():
        log.error(f"测试集不存在: {cfg.testset_file}，请先运行离线流程")
        return

    samples = []
    with open(cfg.testset_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if not samples:
        log.error("测试集为空")
        return
    log.info(f"评测样本: {len(samples)}")

    rec = RealtimeRecommender()
    model = SentenceTransformer(cfg.embedding_model, device=cfg.embedding_device)

    log.info("编码 ground truth...")
    gt_embs = model.encode(
        [s["ground_truth_answer"] for s in samples],
        normalize_embeddings=True, show_progress_bar=True,
    )

    hits, mrrs, lats, covered = 0, 0.0, [], 0
    for i, s in enumerate(samples):
        rec.init_session(f"eval_{i}")
        t0 = time.time()
        res = rec.on_asr(f"eval_{i}", "客户", s["query"])
        lats.append((time.time() - t0) * 1000)

        if not res or not res.get("recommendations"):
            continue
        covered += 1
        recs = res["recommendations"][:K]
        rec_embs = model.encode([r["answer"] for r in recs],
                                normalize_embeddings=True)
        sims = cosine_similarity([gt_embs[i]], rec_embs)[0]
        if (sims > sim_threshold).any():
            hits += 1
            mrrs += 1.0 / (int(sims.argmax()) + 1)

    n = len(samples)
    lats.sort()
    log.info("=" * 50)
    log.info(f"Recall@{K}: {hits / n:.4f}")
    log.info(f"MRR@{K}: {mrrs / n:.4f}")
    log.info(f"覆盖率: {covered / n:.4f}")
    log.info(f"延迟 P50/P95/P99: "
             f"{lats[len(lats)//2]:.1f}/"
             f"{lats[int(len(lats) * 0.95)]:.1f}/"
             f"{lats[int(len(lats) * 0.99)]:.1f} ms")
    log.info("=" * 50)


if __name__ == "__main__":
    evaluate()