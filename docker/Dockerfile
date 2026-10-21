# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ARG TARGETARCH

# HF_HOME / TORCH_HOME 必须在顶部：构建期下载与运行期加载都用同一路径。
# 只写到 .../huggingface，hub/ 由库自动派生。
ENV HF_HOME=/app/models/huggingface \
    TORCH_HOME=/app/models/torch

WORKDIR /app

# 系统依赖：cache mount 让 apt 包跨构建复用，但不进镜像层
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev wget curl vim iputils-ping net-tools \
    build-essential g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 分架构安装 PyTorch（requirements.txt 里不要再含 torch 系列）
RUN --mount=type=cache,target=/root/.cache/pip \
    case "${TARGETARCH}" in \
      amd64) pip install --no-cache-dir \
          torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
          --index-url https://download.pytorch.org/whl/cu129 ;; \
      arm64) pip install --no-cache-dir \
          torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
          --index-url https://pypi.jetson-ai-lab.io/jp6/cu129 ;; \
      *) echo "Unsupported arch: ${TARGETARCH}" >&2; exit 1 ;; \
    esac

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# === 模型预下载层 ===
# models.txt 单独 COPY：只有清单变化才重跑此层，业务代码改动不触发重下
COPY models.txt download_models.sh ./
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    chmod +x download_models.sh && \
    ./download_models.sh models.txt

# === 构建期 smoke test ===
# 强制从缓存加载，能在构建时暴露「路径不一致导致重新下载」的问题
RUN python - <<'PY'
import os
from sentence_transformers import SentenceTransformer

hf_home = os.environ["HF_HOME"]
assert os.path.isdir(os.path.join(hf_home, "hub")), f"hub dir missing under {hf_home}"

model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
vec = model.encode(["offline smoke test"], normalize_embeddings=True)
print(f"smoke ok: dim={vec.shape[-1]}, cache={hf_home}")
PY

COPY . .

RUN rm -f .env .env.* && rm -rf data output log cache

# === 纯运行时 ENV 放最后：修改不会使上面的 apt/pip/model 层失效 ===
# HF_HUB_OFFLINE 在此才打开，避免阻断上面的模型下载
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000
CMD ["python", "main.py", "--mode", "offline"]