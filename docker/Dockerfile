# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ARG TARGETARCH

ENV HF_HOME=/app/models/huggingface \
    TORCH_HOME=/app/models/torch

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev wget curl vim iputils-ping net-tools \
    build-essential g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

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
# COPY 自动创建 /app/docker，无需 mkdir
COPY docker/models.txt docker/download_models.sh ./docker/
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    chmod +x ./docker/download_models.sh && \
    ./docker/download_models.sh docker/models.txt

# === 构建期 smoke test ===
RUN python - <<'PY'
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

hf_home = os.environ["HF_HOME"]
assert os.path.isdir(os.path.join(hf_home, "hub")), f"hub dir missing under {hf_home}"

first = next(
    line.split("#", 1)[0].strip()
    for line in Path("docker/models.txt").read_text().splitlines()
    if line.split("#", 1)[0].strip()
)
model = SentenceTransformer(first, device="cpu")
vec = model.encode(["offline smoke test"], normalize_embeddings=True)
print(f"smoke ok: {first}, dim={vec.shape[-1]}, cache={hf_home}")
PY

COPY . .

RUN rm -f .env .env.* && rm -rf data output log cache docker

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000
CMD ["python", "main.py", "--mode", "offline"]