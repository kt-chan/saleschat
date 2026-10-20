# docker/Dockerfile
FROM python:3.12-slim

# 设置环境变量，确保 Python 输出不被缓冲，且不生成 .pyc 文件
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # 指定 HuggingFace 缓存目录，方便挂载持久化
    HF_HOME=/app/cache/huggingface \
    TORCH_HOME=/app/cache/torch

WORKDIR /app

# 安装系统底层依赖 (psycopg2 编译需要, sentence-transformers 可能依赖)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    wget curl vim \
    iputils-ping net-tools\
    && rm -rf /var/lib/apt/lists/*

# 先复制 requirements.txt 并安装依赖，利用 Docker 镜像层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 兜底删除（尽管 .dockerignore 已经拦截了，但为了绝对安全）
RUN rm -f .env .env.* && \
    rm -rf data output log cache

# 暴露端口 (如果你的项目未来封装了 FastAPI 接口，可在此处开放)
EXPOSE 8000

# 默认启动命令：执行离线构建 (可通过 docker-compose 覆盖为 --mode online)
CMD ["python", "-m", "app.main", "--mode", "offline"]