# syntax=docker/dockerfile:1.7
#
# 双目标 Dockerfile：
#   --target nvidia  → linux/amd64，NVIDIA A1000 (sm_86) + CUDA 12.8 + Python 3.12
#   --target ascend  → linux/arm64，昇腾 310P + CANN + Python 3.12
#
# 两个 target 都安装了 GPU/NPU 运行时；如果运行环境无卡，应用会回退 CPU。
# 构建期 smoke test 只验证「软件栈完整 + 无卡回退 CPU + CPU 推理」，不做真实加速器推理。
# 真实加速器验证请用 scripts/verify_accel.py（见仓库）。

ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG TORCHAUDIO_VERSION=2.8.0
ARG TORCH_NPU_VERSION=2.8.0
ARG CANN_IMAGE=ascendai/cann:8.2.rc1-310p-ubuntu22.04-py3.11
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-runtime-ubuntu22.04
# ============================================================
# Target: nvidia  (linux/amd64)
# ============================================================
FROM ${CUDA_IMAGE} AS nvidia
ARG PYTHON_VERSION
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/models/huggingface \
    TORCH_HOME=/app/models/torch \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    PATH="/opt/venv/bin:$PATH"

# Ubuntu 22.04 上没有 Python 3.12 官方包，用 deadsnakes PPA
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
        gcc libpq-dev wget curl vim iputils-ping net-tools build-essential g++ \
    && rm -rf /var/lib/apt/lists/*

RUN python${PYTHON_VERSION} -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app
COPY requirements.txt .

# CUDA 12.8 wheel：pip 会自动把 nvidia-*-cu12 运行时作为依赖装进 site-packages
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} \
        --index-url https://download.pytorch.org/whl/cu128

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# === 预下载模型 ===
COPY docker/models.txt docker/download_models.sh ./docker/
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    chmod +x ./docker/download_models.sh \
    && ./docker/download_models.sh docker/models.txt

# === 构建期 smoke test（无卡） ===
RUN python - <<'PY'
import os, ctypes, torch
from pathlib import Path

print(f"torch: {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
assert torch.version.cuda is not None, "torch was built without CUDA support"

# 关键：确认 CUDA 运行时库存在（此前 arm64 失败就出在这里）
ctypes.CDLL("libcudart.so.12")
print("libcudart.so.12: OK")

# 设备探测：无卡时应返回 cpu，验证回退逻辑
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"cuda.is_available(): {torch.cuda.is_available()} -> device={device}")

# CPU 回退推理
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from sentence_transformers import SentenceTransformer

first = next(
    l.split("#", 1)[0].strip()
    for l in Path("docker/models.txt").read_text().splitlines()
    if l.split("#", 1)[0].strip()
)
model = SentenceTransformer(first, device="cpu")
vec = model.encode(["offline smoke test"], normalize_embeddings=True)
print(f"smoke ok (cpu fallback): {first}, dim={vec.shape[-1]}, cache={os.environ['HF_HOME']}")
PY

COPY . .
RUN rm -f .env .env.* && rm -rf data output log cache docker

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8000
CMD ["python", "main.py", "--mode", "offline"]


# ============================================================
# Target: ascend  (linux/arm64)
# ============================================================
# 官方 310P CANN 基础镜像（含 CANN runtime + ascendcl）。
# 常用来源：
#   ascendai/cann:<ver>-310p-ubuntu22.04-py3.11
#   swr.cn-north-4.myhuaweicloud.com/ascendhub/cann:<ver>-310p-ubuntu22.04-py3.11
# 若你所在环境使用其它版本，通过 --build-arg CANN_IMAGE=... 覆盖。
# 注意：此基础镜像可能不带 Python 3.12，下面会用 deadsnakes 补装。
FROM ${CANN_IMAGE} AS ascend

ARG PYTHON_VERSION
ARG TORCH_VERSION
ARG TORCH_NPU_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/models/huggingface \
    TORCH_HOME=/app/models/torch \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    PATH="/opt/venv/bin:$PATH"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
        gcc libpq-dev wget curl vim iputils-ping net-tools build-essential g++ \
    && rm -rf /var/lib/apt/lists/*

RUN python${PYTHON_VERSION} -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app
COPY requirements.txt .

# ARM64 上 pip 的 torch wheel 是 CPU 版本（不带 CUDA）
# torch_npu 作为后端插件，通过 CANN 驱动昇腾 NPU
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir torch==${TORCH_VERSION} \
    && pip install --no-cache-dir torch_npu==${TORCH_NPU_VERSION}

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# === 预下载模型 ===
COPY docker/models.txt docker/download_models.sh ./docker/
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    chmod +x ./docker/download_models.sh \
    && ./docker/download_models.sh docker/models.txt

# === 构建期 smoke test（无 NPU 卡） ===
RUN python - <<'PY'
import os, torch
from pathlib import Path

print(f"torch: {torch.__version__}")
import torch_npu
print(f"torch_npu: {torch_npu.__version__}")
print(f"torch.npu.is_available(): {torch.npu.is_available()}")

# CANN 目录存在性（基础镜像应已安装）
ascend_home = os.environ.get("ASCEND_HOME") or "/usr/local/Ascend"
assert os.path.isdir(ascend_home), f"CANN home not found: {ascend_home}"
print(f"CANN home: {ascend_home}")

# 设备探测：无 NPU 时应返回 cpu
device = "cpu"
try:
    if torch.npu.is_available():
        device = "npu:0"
except Exception as e:
    print(f"npu probe raised (expected without card): {e!r}")
print(f"detected device: {device} (expected cpu on build machine)")

# CPU 回退推理
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from sentence_transformers import SentenceTransformer

first = next(
    l.split("#", 1)[0].strip()
    for l in Path("docker/models.txt").read_text().splitlines()
    if l.split("#", 1)[0].strip()
)
model = SentenceTransformer(first, device="cpu")
vec = model.encode(["offline smoke test"], normalize_embeddings=True)
print(f"smoke ok (cpu fallback): {first}, dim={vec.shape[-1]}, cache={os.environ['HF_HOME']}")
PY

COPY . .
RUN rm -f .env .env.* && rm -rf data output log cache docker

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8000
CMD ["python", "main.py", "--mode", "offline"]