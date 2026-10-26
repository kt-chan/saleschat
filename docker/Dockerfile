# ------------------------------------------------------------------
# Base image: Python 3.11 on Debian Slim (smaller than the full image,
# but still contains everything needed to run pip / venv).
# ------------------------------------------------------------------
FROM python:3.11-slim

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ------------------------------------------------------------------
# Install "basic commands" + useful tooling.
# Everything is installed in ONE layer, then the apt cache is
# removed to keep the image small.
# ------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        # --- editors ---
        vim \
        nano \
        less \
        # --- networking / download ---
        curl \
        wget \
        ca-certificates \
        net-tools \
        iputils-ping \
        dnsutils \
        # --- archives / compression ---
        unzip \
        zip \
        tar \
        gzip \
        xz-utils \
        # --- source control / build essentials ---
        git \
        build-essential \
        # --- process / system inspection ---
        procps \
        htop \
        tree \
        # --- misc ---
        bash-completion \
        jq \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------
# Create a non-root user (optional, but recommended).
# ------------------------------------------------------------------
ARG UID=1000
ARG GID=1000
RUN groupadd --gid ${GID} appuser \
    && useradd --uid ${UID} --gid ${GID} --create-home --shell /bin/bash appuser

WORKDIR /app
RUN chown -R appuser:appuser /app
USER appuser

# ------------------------------------------------------------------
# Install Python packages (optional)
# ------------------------------------------------------------------
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# Default command – override at runtime as needed
CMD ["python3"]
