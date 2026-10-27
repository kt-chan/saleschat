# syntax=docker/dockerfile:1.7
#
# 多阶段 CPU 应用镜像（Python 3.11）
#
#   Stage 1 builder：完整编译链，装所有依赖到 /opt/venv
#                    （hdbscan 在 arm64 无 wheel，必须源码编译）
#   Stage 2 runtime：只装 .so 运行时，拷贝 /opt/venv
#                    编译工具链(约 450MB)不进入最终镜像
#
# 架构定位：
#   - embedding 由独立 TEI 容器提供（EMBEDDING_TEI_URL）
#   - PostgreSQL 独立容器（DB_HOST / DB_PORT）
#   - 本进程纯 CPU，不加载任何模型，不依赖 CUDA / CANN
#
# 构建：
#   docker build -f docker/dockerfile -t cca-app:latest .
#   docker buildx build --platform linux/amd64,linux/arm64 \
#     -f docker/dockerfile -t cca-app:latest --push .

ARG PYTHON_VERSION=3.11

# ============================================================
# Stage 1: builder
# ============================================================
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 编译 hdbscan 及部分 native 扩展所需的完整工具链
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

WORKDIR /build
COPY requirements.txt .

# pip 缓存走 BuildKit mount，不进最终镜像
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# ============================================================
# Stage 2: runtime
# ============================================================
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    PATH="/opt/venv/bin:$PATH" \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    NLTK_DATA=/app/.nltk_data \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken_cache \
    MPLCONFIGDIR=/app/.matplotlib \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

# 仅安装运行期 .so 依赖，不含编译工具链
#   libgomp1     hdbscan/numpy/scipy/sklearn/faiss 的 OpenMP 运行时（必须）
#   libopenblas0 scipy/sklearn 若动态链接 BLAS 时需要（保守装上）
#   liblapack3   同上
#   tini         PID 1 信号转发
#   curl/wget    TEI 探活 / 排障
#   jq           排障查看 JSON
#   locales      zh_CN.UTF-8 避免中文编码异常
#   tzdata       日志时间戳正确
#   procps       ps/pgrep/free 排障
#   iproute2     ss -tlnp 查端口
#   less         查看大 jsonl
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        libopenblas0 \
        liblapack3 \
        curl \
        wget \
        tini \
        locales \
        tzdata \
        procps \
        iproute2 \
        less \
        jq \
        vim-tiny \
        iputils-ping \
        netcat-openbsd \
        postgresql-client \
        dos2unix \
    && sed -i 's/^# *en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && sed -i 's/^# *zh_CN.UTF-8 UTF-8/zh_CN.UTF-8 UTF-8/' /etc/locale.gen \
    && locale-gen \
    && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# 拷贝 builder 编译好的 venv（绝对路径一致，venv 可直接复用）
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY . .

# ---- 预缓存离线数据（tiktoken / NLTK，构建期一次性完成） ----
RUN mkdir -p "$NLTK_DATA" "$TIKTOKEN_CACHE_DIR" "$MPLCONFIGDIR" && python - <<'PY'
import os

# tiktoken BPE 文件，避免离线环境调用时尝试下载
try:
    import tiktoken
    for enc in ("cl100k_base", "o200k_base", "p50k_base", "r50k_base"):
        try:
            tiktoken.get_encoding(enc)
            print(f"tiktoken: {enc} OK")
        except Exception as e:
            print(f"tiktoken: {enc} SKIP ({e})")
except ImportError:
    print("tiktoken not installed, skipping")

# NLTK 语料，仅当 nltk 存在时下载
try:
    import nltk
    for pkg in (
        "punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    ):
        try:
            nltk.download(pkg, download_dir=os.environ["NLTK_DATA"], quiet=True)
            print(f"nltk: {pkg} OK")
        except Exception as e:
            print(f"nltk: {pkg} SKIP ({e})")
except ImportError:
    print("nltk not installed, skipping")
PY

# 清理本地开发残留；同时删除 docker/ 目录（生产镜像不需要 compose 与开发脚本）
RUN rm -f .env .env.* \
    && rm -rf data output log cache pg-data tei-data docker .vscode

# 预编译字节码，启动更快；unchecked-hash 适合不可变镜像
RUN python -m compileall -q --invalidation-mode unchecked-hash /app \
    && python -m compileall -q --invalidation-mode unchecked-hash \
        "/opt/venv/lib/python${PYTHON_VERSION}/site-packages" \
    || true

# 校验 native 扩展在 runtime 环境下可加载（缺 .so 会在此失败）
RUN python -c "import numpy, scipy, sklearn, hdbscan, faiss; print('native extensions OK')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD pgrep -f "python .*main\.py" >/dev/null || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "main.py", "--mode", "offline"]