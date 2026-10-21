# syntax=docker/dockerfile:1.7
#
# 双目标 Dockerfile：
#   --target nvidia  → linux/amd64，NVIDIA A1000 (sm_86) + CUDA 12.8 + Python 3.12
#   --target ascend  → linux/arm64，昇腾 310P + CANN 8.5.0 + openEuler 24.03 + Python 3.12
#
# 两个 target 都安装了 GPU/NPU 运行时；如果运行环境无卡，应用会回退 CPU。
# 构建期 smoke test 只验证「软件栈完整 + 无卡回退 CPU + CPU 推理」，不做真实加速器推理。

ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG TORCHAUDIO_VERSION=2.8.0
ARG TORCH_NPU_VERSION=2.8.0
# 310P + openEuler 24.03 实际可用标签为 py3.11；Python 3.12 在下方通过 EPOL 安装
ARG CANN_IMAGE=quay.io/ascend/cann:8.5.0-310p-openeuler24.03-py3.11
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

# Ubuntu 22.04 无 Python 3.12 官方包，用 deadsnakes PPA
# 关键修复：apt-get update 加 Retries/Timeout，抵御 NVIDIA 源同步瞬时不一致
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
        -o Acquire::Retries=10 \
        -o Acquire::http::Timeout=30 \
        -o Acquire::https::Timeout=30 \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
        -o Acquire::Retries=10 \
        -o Acquire::http::Timeout=30 \
        -o Acquire::https::Timeout=30 \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
        gcc libpq-dev wget curl vim iputils-ping net-tools build-essential g++ \
    && rm -rf /var/lib/apt/lists/*

RUN python${PYTHON_VERSION} -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} \
        --index-url https://download.pytorch.org/whl/cu128

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY docker/models.txt docker/download_models.sh ./docker/
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    chmod +x ./docker/download_models.sh \
    && ./docker/download_models.sh docker/models.txt

RUN python - <<'PY'
import os, ctypes, torch
from pathlib import Path

print(f"torch: {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
assert torch.version.cuda is not None, "torch was built without CUDA support"

ctypes.CDLL("libcudart.so.12")
print("libcudart.so.12: OK")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"cuda.is_available(): {torch.cuda.is_available()} -> device={device}")

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
# 基础镜像：CANN 8.5.0 + openEuler 24.03 (Euler 2.0 SP15 对应版本)，芯片 310P
# 基础镜像自带 py3.11；Python 3.12 通过启用 EPOL 扩展源安装
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

# 关键修复：python3.12 位于 EPOL 扩展源，需先启用
# 1. 安装 dnf-plugins-core 以获得 config-manager
# 2. 启用 epol 仓库
# 3. 安装 python3.12 及构建工具
RUN --mount=type=cache,target=/var/cache/dnf,sharing=locked \
    --mount=type=cache,target=/var/lib/dnf,sharing=locked \
    dnf install -y dnf-plugins-core \
    && dnf config-manager --set-enabled epol \
    && dnf install -y --setopt=install_weak_deps=False \
        python3.12 python3.12-devel \
        gcc gcc-c++ make cmake libpq-devel \
        wget curl vim iputils net-tools \
    && dnf clean all

RUN python3.12 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app
COPY requirements.txt .

# 官方配套：CANN 8.5.0 + torch 2.8.0 + torch_npu 2.8.0.post2 + Python 3.12 + AArch64
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
        torch==${TORCH_VERSION}+cpu \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.8.0/torch_npu-2.8.0.post2-cp312-cp312-manylinux_2_28_aarch64.whl

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY docker/models.txt docker/download_models.sh ./docker/
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    chmod +x ./docker/download_models.sh \
    && ./docker/download_models.sh docker/models.txt

RUN python - <<'PY'
import os, torch
from pathlib import Path

print(f"torch: {torch.__version__}")
import torch_npu
print(f"torch_npu: {torch_npu.__version__}")
print(f"torch.npu.is_available(): {torch.npu.is_available()}")

ascend_home = os.environ.get("ASCEND_HOME") or "/usr/local/Ascend"
assert os.path.isdir(ascend_home), f"CANN home not found: {ascend_home}"
print(f"CANN home: {ascend_home}")

device = "cpu"
try:
    if torch.npu.is_available():
        device = "npu:0"
except Exception as e:
    print(f"npu probe raised (expected without card): {e!r}")
print(f"detected device: {device} (expected cpu on build machine)")

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