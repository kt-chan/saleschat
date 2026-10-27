# syntax=docker/dockerfile:1.7
#
# Offline-ready CPU application image (Python 3.11).
#
# Architecture:
#   - this container:  Python app only (no inference, no DB)
#   - embedding:       remote TEI service  (EMBEDDING_TEI_URL)
#   - PostgreSQL:      standalone container (DB_HOST / DB_PORT)
#
# Build-time pre-cache ensures the container works with no internet at runtime:
#   - tiktoken BPE files   → TIKTOKEN_CACHE_DIR
#   - NLTK corpora         → NLTK_DATA  (only if nltk is installed)
#   - Python bytecode      → precompiled .pyc (unchecked-hash)
#
# Multi-arch build:
#   docker buildx build \
#     --platform linux/amd64,linux/arm64 \
#     -t cca-app:latest --push .

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS runtime

# ---- Runtime env ---------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
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

# ---- System packages -----------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        curl \
        wget \
        tini \
        locales \
        tzdata \
        procps \
        iproute2 \
        less \
        jq \
    && sed -i 's/^# *en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && sed -i 's/^# *zh_CN.UTF-8 UTF-8/zh_CN.UTF-8 UTF-8/' /etc/locale.gen \
    && locale-gen \
    && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# ---- Python venv ---------------------------------------------------------
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel

WORKDIR /app
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# ---- Pre-cache offline data (build-time only) ----------------------------
RUN mkdir -p "$NLTK_DATA" "$TIKTOKEN_CACHE_DIR" "$MPLCONFIGDIR" && python - <<'PY'
import os

# tiktoken: BPE files for GPT-3.5/4 and GPT-4o must be cached
# or every call tries to download at runtime (fails in offline env).
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

# NLTK: only meaningful if nltk is a direct/transitive dependency
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

# ---- Application ---------------------------------------------------------
COPY . .
RUN rm -f .env .env.* \
    && rm -rf data output log cache pg-data tei-data docker .vscode

# Precompile bytecode; unchecked-hash is safe because the image is immutable.
RUN python -m compileall -q --invalidation-mode unchecked-hash /app \
    && python -m compileall -q --invalidation-mode unchecked-hash \
        "/opt/venv/lib/python${PYTHON_VERSION}/site-packages" \
    || true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD pgrep -f "python .*main\.py" >/dev/null || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "main.py", "--mode", "offline"]