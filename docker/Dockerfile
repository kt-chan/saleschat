# syntax=docker/dockerfile:1.7
#
# 双目标 Dockerfile：
#   --target nvidia  → linux/amd64，NVIDIA (sm_86+) + CUDA 12.8 + Python 3.12
#   --target ascend  → linux/arm64，昇腾 310P + CANN 9.1.0 + openEuler 24.03 + Python 3.12
#
# 版本配套（ascend）：
#   torch 2.10.0+cpu  ↔  torch_npu 2.10.0  ↔  CANN 9.1.0  ↔  Python 3.12
#   wheel: torch_npu-2.10.0-cp312-cp312-manylinux_2_28_aarch64.whl
#   来源: https://gitcode.com/Ascend/pytorch/releases/download/v26.0.0-pytorch2.10.0/
#
# 两个 target 都安装了加速器运行时；无卡环境会回退 CPU。
# 构建期 smoke test 验证软件栈完整性 + CPU 推理可用性，不做真实加速器推理。

ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.10.0
ARG TORCHVISION_VERSION=0.25.0
ARG TORCHAUDIO_VERSION=2.10.0
ARG TORCH_NPU_VERSION=2.10.0

# 基础镜像标签
ARG CUDA_IMAGE=nvidia/cuda:12.8.0-runtime-ubuntu22.04
ARG CANN_IMAGE=quay.io/ascend/cann:9.1.0-310p-openeuler24.03-py3.12

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
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
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

# torch 家族从 cu128 索引安装（与 CUDA 12.8 runtime 匹配）
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} \
        --index-url https://download.pytorch.org/whl/cu128

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY docker/models.txt docker/download_models.sh ./docker/
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    chmod +x ./docker/download_models.sh \
    && ./docker/download_models.sh docker/models.txt

# Smoke test：CUDA 链接 + CPU fallback 推理
RUN /opt/venv/bin/python - <<'PY'
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
# 基础镜像：官方预打包 CANN 9.1.0 + openEuler 24.03 + Python 3.12
# ============================================================
FROM ${CANN_IMAGE} AS ascend
ARG TORCH_VERSION
ARG TORCH_NPU_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/models/huggingface \
    TORCH_HOME=/app/models/torch \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/devlib \
    TORCH_DEVICE_BACKEND_AUTOLOAD=1

WORKDIR /app
COPY requirements.txt .

# 1) torch CPU 版本（2.10.0 系列的 +cpu wheel 仅在 PyTorch 官方 CPU 源提供）
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==${TORCH_VERSION}+cpu \
        --index-url https://download.pytorch.org/whl/cpu

# 2) 从 GitCode Release 下载并安装 torch_npu（aarch64 wheel）
#    - 标签: v26.0.0-pytorch2.10.0
#    - 文件: torch_npu-2.10.0-cp312-cp312-manylinux_2_28_aarch64.whl
#    - 使用 curl -fL: -f 使 4xx/5xx 报错退出, -L 跟随重定向
RUN set -eux; \
    TORCH_NPU_URL="https://gitcode.com/Ascend/pytorch/releases/download/v26.0.0-pytorch2.10.0/torch_npu-${TORCH_NPU_VERSION}-cp312-cp312-manylinux_2_28_aarch64.whl"; \
    WHL="$(basename "${TORCH_NPU_URL}")"; \
    echo "Downloading ${WHL}"; \
    curl -fL --retry 5 --retry-delay 3 --connect-timeout 30 \
        "${TORCH_NPU_URL}" -o "/tmp/${WHL}"; \
    pip install --no-cache-dir "/tmp/${WHL}"; \
    rm -f "/tmp/${WHL}"

# 3) 验证 torch_npu 可导入
RUN python -c "import torch, torch_npu; print(f'torch: {torch.__version__}, torch_npu: {torch_npu.__version__}')"

# 4) 安装项目其余依赖
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY docker/models.txt docker/download_models.sh ./docker/
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    chmod +x ./docker/download_models.sh \
    && ./docker/download_models.sh docker/models.txt

# 5) Smoke test：torch_npu 导入 + NPU 可用性探测 + CPU fallback 推理
RUN python - <<'PY'
import os, torch

print(f"torch: {torch.__version__}")
assert "+cpu" in torch.__version__, "ascend target must use torch CPU build"

import torch_npu
print(f"torch_npu: {torch_npu.__version__}")
print(f"npu.is_available(): {torch.npu.is_available()}")

ascend_home = os.environ.get(
    "ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"
)
if os.path.isdir(ascend_home):
    print(f"ASCEND_HOME_PATH: {ascend_home} (present)")
else:
    print(f"WARNING: {ascend_home} not found — CANN runtime may be incomplete")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from sentence_transformers import SentenceTransformer
from pathlib import Path

first = next(
    l.split("#", 1)[0].strip()
    for l in Path("docker/models.txt").read_text().splitlines()
    if l.split("#", 1)[0].strip()
)
model = SentenceTransformer(first, device="cpu")
vec = model.encode(["offline smoke test"], normalize_embeddings=True)
print(f"smoke ok (cpu fallback): {first}, dim={vec.shape[-1]}")
PY

COPY . .
RUN rm -f .env .env.* && rm -rf data output log cache docker

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8000
CMD ["python", "main.py", "--mode", "offline"]
