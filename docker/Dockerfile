# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

# BuildKit 在多架构构建时自动注入：amd64 或 arm64
ARG TARGETARCH

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/cache/huggingface \
    TORCH_HOME=/app/cache/torch \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# 系统依赖：两种架构下 Debian 仓库均提供对应二进制
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    wget curl \
    vim iputils-ping net-tools \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 显式按架构安装 PyTorch CPU 版本
# 注意：请将 torch / torchvision / torchaudio 从 requirements.txt 中移除，
# 避免重复安装或版本冲突
RUN case "${TARGETARCH}" in \
        amd64) \
            pip install --no-cache-dir \
                torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
                --index-url https://download.pytorch.org/whl/cu124 ;; \
        arm64) \
            # arm64 直接使用默认 PyPI 源（官方在 PyPI 上维护了 arm64 CPU 版本的 wheel）
            pip install --no-cache-dir \
                torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 ;; \
        *) echo "Unsupported architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && pip install --no-cache-dir -r requirements.txt



COPY . .

# 兜底清理，防止敏感文件或运行时数据被打入镜像
RUN rm -f .env .env.* && \
    rm -rf data output log cache

EXPOSE 8000

CMD ["python", "main.py", "--mode", "offline"]