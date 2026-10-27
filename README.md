# 催收话术智能推荐系统 (CCA)

> **Collections Conversation Assistant** — 面向催收坐席的实时话术推荐引擎。
> 基于 SOP 图谱的双分支混合检索（应用层余弦 + Golden QA）。
> 应用纯 CPU，embedding 由独立 TEI 容器提供，数据库为独立 PostgreSQL 容器。

---

## 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [架构与数据流](#架构与数据流)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [使用方式](#使用方式)
- [配置说明](#配置说明)
- [目录结构](#目录结构)
- [TEI 部署](#tei-部署)
- [性能调优](#性能调优)
- [故障排查](#故障排查)
- [开发与测试](#开发与测试)
- [变更记录](#变更记录)

---

## 项目简介

CCA 是一个**离线建库 + 在线检索**的两阶段系统，用于在催收通话过程中，根据当前对话上下文为坐席推荐下一句最合适的应答话术。

- **离线阶段**：解析历史通话 → LLM 双级打标（会话级 + 话对级）→ 调用 TEI 生成向量并写入 PostgreSQL → 图谱合成"经典 Q&A"。
- **在线阶段**：解析实时对话 → 双分支混合检索（进程内 embedding 索引 + 图谱 golden QA）→ 图感知投票 → 返回最佳话术。

系统不依赖在线 LLM 推理，在线延迟稳定在 **毫秒级**（典型 5–50 ms，视候选集大小而定）。

**部署拓扑**：

```
┌─────────────┐    HTTP     ┌─────────────┐
│  App (CPU)  │ ──────────► │  TEI (GPU)  │  embedding
│  Python 3.11│             │  bge-base-zh│
└──────┬──────┘             └─────────────┘
       │ psycopg3
       ▼
┌─────────────┐
│ PostgreSQL  │  qa_pairs / 状态机
└─────────────┘
```

---

## 核心特性

| 特性 | 说明 |
|---|---|
| **SOP 图谱驱动** | 阶段（stage）/ 意图（intent）/ 转移（transition）显式建模，检索时按图跳数衰减，避免跨阶段语义漂移 |
| **双分支混合检索** | Branch A: 进程内 embedding 矩阵（BLAS matmul）；Branch B: 内存 golden QA 点积；分数融合 + 平方加权投票 |
| **三种检索模式** | `off`（无约束）/ `guide`（图引导，允许跳转）/ `enforce`（锁定阶段，失败即拒绝） |
| **远程 Embedding** | 统一通过 TEI HTTP 服务获取向量，进程内**不加载模型**；模型权重与计算全在 TEI 容器 |
| **进程级索引缓存** | `_EmbeddingIndex` 首次查询时全量加载 embedding 到 numpy 矩阵并预归一化；查询退化为一次 matmul；写入后懒重建 |
| **离线-在线语义一致** | 离线入库与在线检索共用同一 TEI 服务与同一 `normalize_embeddings=True` 口径；L2 归一化后点积 ≡ 余弦相似度 |
| **独立数据库** | PostgreSQL 16 独立容器，embedding 以 `BYTEA`（float32 原始字节）存储，无 pgvector 依赖 |
| **FastAPI 服务** | `app/server.py` 提供 `/suggest`、`/health`、`/health/ready`；`fastapi-offline` 保证离线环境 `/docs` 可渲染 |
| **多架构镜像** | amd64 / arm64 双架构构建；多阶段 Dockerfile 隔离编译链（arm64 上 hdbscan 需源码编译） |

---

## 架构与数据流

```
┌──────────────────────── 离线阶段 (--mode offline) ────────────────────────┐
│                                                                          │
│  calls.jsonl                                                             │
│      │                                                                   │
│      ▼  parse_dialog                                                     │
│  calls[] ──► LLMLabeler.run_sessions ──► session_labels.jsonl           │
│      │                                     │                             │
│      │                                     ▼                             │
│      └──► LLMLabeler.run_pairs ────────► pair_labels.jsonl              │
│                                              │                           │
│                                              ▼  PgvectorWriter          │
│                                        qa_pairs (embedding BYTEA)       │
│                                              │                           │
│                                              ▼  GoldenScriptGenerator   │
│                                        sop_graph.json                   │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────── 在线阶段 (--mode online) ─────────────────────────┐
│                                                                          │
│  raw text ──► parse_dialog ──► _build_suggest_query (窗口裁剪)          │
│                                              │                           │
│                                              ▼                           │
│                                    HybridRetriever                       │
│                              ┌───────────────┴───────────────┐          │
│                              ▼                               ▼          │
│                    _EmbeddingIndex                    golden QA          │
│                 (预归一化矩阵 matmul)                  (内存点积)        │
│                              └───────────────┬───────────────┘          │
│                                              ▼                           │
│                              merge (hop decay) ──► vote (stage+intent)  │
│                                              ▼                           │
│                                    best script + alternates              │
└──────────────────────────────────────────────────────────────────────────┘
```

**关键设计点**：

1. **阶段/意图约束**：检索前先用向量化 mask 过滤 `(stage, primary_intent) ∈ nodes`，候选节点由 SOP 图 BFS 决定（`hybrid_max_hops` 控制跳数）。
2. **跳数衰减**：`score = raw_sim × weight × hop_decay[hop]`，默认 `[1.0, 0.95, 0.85]`。
3. **投票**：对 top-N 结果按 `score²` 加权投票，先定 stage 再定 intent，避免弱证据干扰。
4. **Golden QA 优先**：`golden` 命中时作为主推荐，`pgvector` 作为 alternate，兼顾精度与召回。
5. **应用层余弦**：DB 只做行过滤，相似度计算在进程内 BLAS 完成；`_EmbeddingIndex` 缓存命中后每次查询 <5ms（万行级）。

---

## 环境要求

### 应用（本仓库）

| 组件 | 版本 | 说明 |
|---|---|---|
| **Python** | 3.11.x | 与依赖的 wheel 兼容性最好 |
| **Docker** | 24.0+ | Compose V2 集成 |
| **Docker Compose** | v2.20+ | 支持 `profiles` |
| **NVIDIA Container Toolkit** | 任意近版 | 仅当运行 TEI 容器且需要 GPU 加速时必需 |

应用本身**不依赖 CUDA / Ascend NPU / 模型权重**，纯 CPU 即可运行。

### 外部服务

| 组件 | 版本 | 说明 |
|---|---|---|
| **TEI** | `ghcr.io/huggingface/text-embeddings-inference:89-1.9` | Ada Lovelace GPU 优化；Ampere 用 `86-`，Hopper 用 `90-`，混卡用 `cuda-1.9` |
| **PostgreSQL** | 16 | `postgres:16` 官方镜像，无需 pgvector |
| **Embedding 模型** | `BAAI/bge-base-zh-v1.5` | 768 维；与 `.env` 的 `EMBEDDING_MODEL` 对齐 |
| **LLM** | OpenAI 兼容 API | 离线打标与图谱合成时使用；在线检索不需要 |

### Python 依赖（精简后）

```
openai / requests / loguru / tqdm / PyYAML / nanoid
pydantic / pydantic-settings
numpy / scipy / scikit-learn / pandas / networkx
bertopic / faiss-cpu / umap-learn / hdbscan
jieba / rapidfuzz
psycopg[binary,pool]
fastapi / uvicorn / python-multipart / fastapi-offline
tiktoken
debugpy
```

完整列表见 `requirements.txt`。已移除 `torch` / `sentence-transformers` / `transformers` / `pgvector` / `embedded-postgres` 等本地推理相关依赖。

---

## 快速开始

### 1. 克隆与配置

```bash
git clone <repo> cca && cd cca

# 环境变量
cp .env-sample .env
# 至少填写：
#   LLM_API_KEY=sk-...
#   EMBEDDING_TEI_URL=http://127.0.0.1:8080
#   EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5
```

### 2. 启动依赖栈

```bash
# 拉起 postgres + tei（不含 app）
docker compose -f docker/docker-compose.yml up -d

# 或用封装脚本（幂等，带就绪探测）
python scripts/stack.py up
```

首次启动 TEI 会下载模型（约 500MB 到 `./tei-data`），需 3~5 分钟。后续重启秒起。

验证：

```bash
curl -s http://localhost:8080/health
docker exec cca-postgres pg_isready -U cca_user -d cca_db
```

### 3. 安装 Python 依赖

```bash
python3.11 -m venv .venv
source .venv/bin/activate           # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4. 准备数据

将历史通话数据放入 `./data/calls.jsonl`，每行一个 JSON：

```jsonl
{"call_id": "C001", "text": "催收员：您好，请问是张先生吗？\n客户：是我。\n催收员：您名下信用卡本期逾期1900元，请问能处理吗？\n客户：我最近被裁员了，暂时没有收入。"}
{"call_id": "C002", "text": "..."}
```

对话文本支持 **行模式**（每行一条 `角色:内容`）或 **分隔符模式**（`；`/`;` 分隔），自动检测。

### 5. 离线建库

```bash
python app/main.py --mode offline
```

输出：

- `output/session_labels.jsonl` — 会话级标签
- `output/pair_labels.jsonl` — 话对级标签
- `output/sop_graph.json` — SOP 图谱 + golden QA
- `qa_pairs` 表（PostgreSQL，embedding 为 BYTEA）

### 6. 在线演示

```bash
python app/main.py --mode online
```

演示会分别以 **fast path**（给定 `stage_hint`）和 **slow path**（自动推断）两条路径检索，并打印匹配结果、耗时与 debug 信息。

### 7. 启动 FastAPI 服务（可选）

```bash
python -m app.server
# 或
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

访问 `http://localhost:8000/docs`（离线可渲染）测试 `/suggest`。

---

## 使用方式

### CLI

```bash
python app/main.py --mode offline      # 离线建库
python app/main.py --mode online       # 在线演示
python app/main.py --mode all          # 顺序执行
```

### FastAPI

```bash
# 启动
python -m app.server

# 请求
curl -s http://localhost:8000/suggest \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": "催收员：您好，您的账单已逾期三天。\n客户：我最近资金周转困难。",
    "stage_hint": "04 方案协商与签署"
  }' | jq .
```

端点：

| 端点 | 说明 |
|---|---|
| `POST /suggest` | 话术推荐主接口；`conversation` 必填，`stage_hint` 可选 |
| `GET /health` | liveness，仅检查进程存活 |
| `GET /health/ready` | readiness，探测 PG + TEI，任一不通返回 503 |
| `GET /docs` | Swagger UI（fastapi-offline，无外网可用） |

### 编程接口

```python
from app.online import suggest_script_for_conversation

text = (
    "催收员：您好，请问是张先生吗？\n"
    "客户：是我。\n"
    "催收员：您名下信用卡本期逾期1900元，请问能处理吗？\n"
    "客户：我最近被裁员了，暂时没有收入。"
)

# 方式一：带 stage hint（快速通道）
res = suggest_script_for_conversation(text, stage_hint="04 方案协商与签署")
print(res["script"], res["stage"], res["intent"], res["score"])

# 方式二：自动推断（两趟检索）
res = suggest_script_for_conversation(text)

# 方式三：只要话术字符串
from app.online import suggest_script
print(suggest_script(text))
```

返回值结构：

```python
{
    "script":     "客户：...",           # 推荐话术（主）
    "question":   "客户：...",           # 对应参考客户问句
    "stage":      "04 方案协商与签署",   # 推断的 SOP 阶段
    "intent":     "PLAN_INSTALL",       # 推断的客户意图
    "source":     "golden" | "pgvector",
    "score":      0.8734,
    "alternates": [ {"question": "...", "answer": "...", "score": 0.82, "source": "pgvector"}, ... ],
    "reject":     False,
    "latency_ms": 34.2,
    "debug":      {"mode": "guide", "path": "hint", "passes": 1, ...}
}
```

---

## 配置说明

### 关键配置项（`.env`）

| 变量 | 默认值 | 说明 |
|---|---|---|
| **Embedding** | | |
| `EMBEDDING_MODEL` | `BAAI/bge-base-zh-v1.5` | 与 TEI 容器 `--model-id` 对齐（维度必须一致） |
| `EMBEDDING_TEI_URL` | `http://127.0.0.1:8080` | **必填**，指向 TEI 服务根地址 |
| `EMBEDDING_TEI_TIMEOUT` | `30` | 单次 HTTP 超时（秒） |
| `EMBEDDING_TEI_BATCH_SIZE` | `16` | 每次 POST /embed 的条数，建议 ≤ TEI `--max-batch-requests` |
| `EMBEDDING_TEI_CONCURRENCY` | `8` | 并发 HTTP 连接数 |
| `EMBEDDING_TEI_RETRY_MAX` | `3` | 失败重试次数 |
| `EMBEDDING_SORT_BY_LENGTH` | `true` | 按长度排序分块，减少 padding |
| **检索** | | |
| `STATE_MACHINE_MODE` | `guide` | `off` / `guide` / `enforce` |
| `HYBRID_MAX_HOPS` | `2` | 图 BFS 最大跳数 |
| `HYBRID_PG_TOP_K` | `20` | 进程内索引 top-k 深度 |
| **Database** | | |
| `DB_HOST` | `localhost` | compose 内用 `postgres` |
| `DB_PORT` | `5432` | |
| `DB_NAME` | `cca_db` | |
| `DB_USER` / `DB_PASSWORD` | `cca_user` / `cca_pass` | |
| `DB_WRITE_BATCH_SIZE` | `500` | 每批写库条数 |
| **Runtime** | | |
| `CLEAR_CACHE_ON_START` | `false` | `true` 则清空 `cache/`、`output/`、`qa_pairs` |

完整配置见 `app/settings.py` 与 `.env-sample`。

### 三种检索模式

| 模式 | 行为 | 适用场景 |
|---|---|---|
| `off` | 无约束 embedding 检索 | 冷启动 / 数据探索 |
| `guide` | 图引导，允许跨 stage 跳转，失败时回退 pass 1 | 生产默认 |
| `enforce` | 锁定 hint 或 pass-1 推断的 stage，任何失败硬拒绝 | 合规严格场景 |

---

## 目录结构

```
.
├── app/
│   ├── settings.py           # Pydantic Settings（.env 加载）
│   ├── logger.py             # 日志
│   ├── config.py             # SOP 阶段/意图/转移静态定义
│   ├── llm.py                # LLMClient（OpenAI 兼容）
│   ├── utils.py              # DB / TEI Embedder / _EmbeddingIndex / 对话解析
│   ├── offline.py            # LLMLabeler / PgvectorWriter / GoldenScriptGenerator
│   ├── online.py             # HybridRetriever / suggest_script*
│   └── server.py             # FastAPI 服务
├── docker/
│   ├── dockerfile            # 多阶段构建（builder + runtime）
│   └── docker-compose.yml    # postgres + tei (+ app profile)
├── scripts/
│   └── stack.py              # compose 管理封装（供 VS Code 任务调用）
├── data/
│   ├── calls.jsonl           # 历史通话（输入）
│   └── demo_text.txt         # 演示对话
├── output/                   # 离线产物（自动创建）
│   ├── session_labels.jsonl
│   ├── pair_labels.jsonl
│   └── sop_graph.json
├── cache/                    # 缓存目录
├── log/
├── pg-data/                  # PostgreSQL 数据卷（.gitignore）
├── tei-data/                 # TEI 模型缓存（.gitignore）
├── prompts.yaml              # LLM prompt 模板
├── main.py                   # CLI 入口（可选）
├── requirements.txt
├── .dockerignore
├── .env-sample
└── README.md
```

---

## TEI 部署

### GPU 镜像选择

TEI 官方镜像按 **Compute Capability** 分类，不是按 CUDA 版本：

| 前缀 | 架构 | 代表 GPU |
|---|---|---|
| `86-` | Ampere (CC 8.6) | A1000、A10、RTX 30xx |
| `89-` | Ada Lovelace (CC 8.9) | **RTX 2000 Ada**、RTX 40xx、L4 |
| `90-` | Hopper (CC 9.0) | H100 |
| `cuda-` | 多合一（启动时自动探测） | 任意 |

**RTX 2000 Ada Generation 用 `89-1.9`**。选错会导致容器内 `CUDA_ERROR_NO_DEVICE`，回退 CPU。

### 容器配置

`docker/docker-compose.yml` 中的 `tei` 服务：

```yaml
  tei:
    image: ghcr.io/huggingface/text-embeddings-inference:89-1.9
    entrypoint: ["text-embeddings-router"]      # 绕过 entrypoint.sh 包装
    command:
      - "--model-id"
      - "BAAI/bge-base-zh-v1.5"
      - "--max-batch-tokens"
      - "4096"
      - "--max-batch-requests"
      - "8"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

**关键点**：

- `entrypoint: ["text-embeddings-router"]`：部分 TEI 版本的 `entrypoint.sh` 会吞或改写参数；直接调二进制保证行为确定。
- `deploy.resources.reservations.devices`：Compose V2 官方 GPU 语法，所有版本支持。
- `count: all` 或 `count: 1`：单卡用 `count: 1` 更明确。

### 验证 GPU 生效

```bash
docker compose -f docker/docker-compose.yml logs tei | grep -E "Starting Bert model"

# 期望：INFO ... Starting Bert model on Cuda { ordinal: 0, ... }
# 而非：WARN ... Could not find a compatible CUDA device
#      INFO ... Starting Bert model on Cpu
```

### 宿主机前置检查

```bash
nvidia-smi                                                  # 宿主机可见 GPU
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

第二条命令失败说明 **NVIDIA Container Toolkit** 未装好，参考：

```bash
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 性能调优

| 参数 | 影响 | 建议 |
|---|---|---|
| `DB_WRITE_BATCH_SIZE` | 单次写库条数 | 500 起步；batch 越大写入吞吐越高但单次事务耗时增加 |
| `EMBEDDING_TEI_BATCH_SIZE` | 单次 HTTP 请求条数 | 建议 ≤ TEI `--max-batch-requests`，过大会削弱 TEI 内部 batching |
| `EMBEDDING_TEI_CONCURRENCY` | 客户端并发 | 与 TEI `--max-concurrent-requests` 协调，过高会导致排队 |
| `EMBEDDING_SORT_BY_LENGTH` | padding 浪费 | 保持 `true` |
| `HYBRID_PG_TOP_K` | 候选深度 | 20 → 50 提升召回，矩阵 matmul 成本几乎不变 |
| `HYBRID_MAX_HOPS` | 图 BFS 半径 | 2 平衡召回与噪声；严格场景降至 1 |
| TEI `--max-batch-tokens` | GPU 显存占用 | RTX 2000 Ada 8GB 建议 4096；A1000 4GB 降到 2048 |

**应用层缓存效果**（BGE-base 768 维）：

| 候选集大小 | 首次加载 | 单次查询 |
|---|---|---|
| 1 万行 | ~0.3s | ~1 ms |
| 10 万行 | ~2s | ~8 ms |
| 100 万行 | ~20s | ~80 ms |

缓存常驻内存 ≈ `N × 768 × 4` 字节。10 万行 ≈ 300MB。

---

## 故障排查

### 启动类

| 现象 | 排查 |
|---|---|
| `ValueError: EMBEDDING_TEI_URL 为必填项` | `.env` 未填或为空；`Settings()` 校验器会在启动时立即失败 |
| `TEI /info 探测失败` | `curl http://localhost:8080/health` 检查 TEI 是否就绪；冷启动首次需等 3~5 分钟 |
| `PostgreSQL 连接失败` | `docker exec cca-postgres pg_isready -U cca_user -d cca_db`；确认 `DB_HOST` 在 compose 内为 `postgres` |
| `qa_pairs.embedding 类型不匹配` | 旧库有 `vector` 或 `jsonb` 列；`init_db()` 会自动迁移，但 `vector` → `BYTEA` 会丢旧向量，需重跑 offline |
| `libgomp.so.1: cannot open shared object file` | runtime 阶段缺 `libgomp1`，检查 Dockerfile apt 列表 |

### 运行类

| 现象 | 排查 |
|---|---|
| `SOP 图谱未找到` | 先执行 `python app/main.py --mode offline` |
| TEI 返回 `CUDA_ERROR_NO_DEVICE` | 镜像 tag 与 GPU 架构不匹配（Ada 用 `89-`）；或 `entrypoint.sh` 干扰，覆写 `entrypoint: ["text-embeddings-router"]` |
| TEI 日志 `Backend does not support a batch size > 4` | 回退 CPU 后强制降级；确认 GPU 实际生效 |
| 在线结果与离线预期不符 | 检查离线/在线是否指向**同一** `EMBEDDING_TEI_URL` |
| 检索结果为空 | `_EmbeddingIndex` 缓存可能过期；重启服务或触发 `invalidate_embedding_index()` |

### 调试工具

```bash
# 检查 TEI 服务
curl -s http://localhost:8080/info | jq .
curl -s -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs":["测试"]}' | head -c 200

# 检查 PG
psql -h localhost -U cca_user -d cca_db -c '\d qa_pairs'
psql -h localhost -U cca_user -d cca_db -c 'SELECT count(*), octet_length(embedding) FROM qa_pairs GROUP BY 2'

# 检查容器 GPU
docker inspect cca-tei --format '{{json .HostConfig.DeviceRequests}}'

# 清空全部状态重来
CLEAR_CACHE_ON_START=true python app/main.py --mode offline
```

---

## 开发与测试

### VS Code 任务（一键起栈）

`.vscode/tasks.json` 提供：

- `Stack: Up` — 启动 postgres + tei，等待就绪
- `Stack: Down` — 停止
- `Stack: Restart` / `Stack: Status` / `Stack: Logs`

`.vscode/launch.json` 中每个配置带 `preLaunchTask: "Stack: Up"`，F5 调试前自动拉起依赖栈（幂等，已就绪时秒过）。

### 静态检查与测试

```bash
# 静态检查
ruff check app/

# 单元测试
pytest tests/ -v

# 快速回归
python -c "
from app.online import suggest_script_for_conversation
r = suggest_script_for_conversation('客户：我最近被裁员了。', stage_hint='04 方案协商与签署')
assert r['script'] and not r['reject']
print('OK:', r['latency_ms'], 'ms')
"
```

### Docker 构建

```bash
# 单架构本地构建
docker build -f docker/dockerfile -t cca-app:latest .

# 多架构构建（推送到 registry）
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f docker/dockerfile \
  -t ghcr.io/<owner>/<repo>:dev --push .
```

### CI

GitHub Actions workflow (`.github/workflows/build.yml`) 在 `v*` tag / release / 手动触发时：

1. 并行在 `ubuntu-latest`（amd64）与 `ubuntu-24.04-arm`（arm64）构建
2. 分别推送 by-digest 到 GHCR
3. 合并多架构 manifest 并打语义版本 tag
4. 可选导出单架构 `.tar` 供离线交付

---

## 变更记录

### 本轮（TEI + 独立 PG 架构）

- **Embedding**：移除进程内 `_LocalEmbedder`（CUDA/NPU/CPU 多后端），只保留远程 TEI HTTP 客户端 `_TEIEmbedder`。删除 `EMBEDDING_BACKEND` / `EMBEDDING_NUM_GPUS` / `EMBEDDING_DEVICE` / `EMBEDDING_BATCH_SIZE` / `EMBEDDING_RETRY_MAX` / `EMBEDDING_RETRY_BACKOFF` / `EMBEDDING_QUEUE_TIMEOUT` / `EMBEDDING_MODE` 配置项。
- **Database**：移除 `pgvector` 与 `embedded-postgres`。`qa_pairs.embedding` 改为 `BYTEA`（float32 原始字节，自动从 `jsonb` 迁移）。相似度检索从 SQL `<=>` 算子改为进程内 `_EmbeddingIndex`（预归一化矩阵 + 向量化 mask + `argpartition` top-k）。
- **新增**：`search_top_k` 应用层余弦检索；`invalidate_embedding_index()` 写后懒失效。
- **FastAPI**：新增 `app/server.py`，`fastapi-offline` 保证离线 `/docs` 可渲染；`/health`（liveness）与 `/health/ready`（readiness）分离；`lifespan` 替代 `on_event`；统一异常响应结构。
- **Docker**：多阶段 Dockerfile（`docker/dockerfile`），builder 安装编译链（arm64 上 `hdbscan` 无 wheel 需源码编译），runtime 只保留 `.so`。移除 CUDA / CANN / torch / torch_npu 安装，镜像从 ~6.5GB 缩至 ~750MB。
- **Compose**：`docker/docker-compose.yml` 新增 TEI 服务，`app` 使用 `profiles: ["app"]` 隔离（避免与宿主机 F5 调试端口冲突）；GPU 使用 `deploy.resources.reservations.devices`（Compose V2 标准语法）。
- **TEI**：镜像 tag 从 `86-1.9` 改为 `89-1.9`（RTX 2000 Ada = Ada Lovelace CC 8.9）；`entrypoint: ["text-embeddings-router"]` 绕过 shell 包装，确保参数与 GPU 初始化行为确定。
- **依赖**：`requirements.txt` 移除 `torch` / `sentence-transformers` / `transformers` / `pgvector` / `embedded-postgres` / `Pillow` / `plotly` / `tenacity` / `psutil` / `structlog` / `prometheus-client` 等。保留 `bertopic`（暂未确认可删）。
- **CI**：workflow 从双 target（`nvidia`/`ascend`）改为单 target 双架构（amd64 + arm64），移除 `HF_TOKEN` secret 依赖；新增 `Assert both platforms in manifest` 步骤防止单架构 manifest 静默通过。

### 历史（本地推理 + pgvector 架构，已废弃）

- `psycopg2-binary` → `psycopg[binary,pool]`（psycopg3）
- `pgembed` → `embedded-postgres`（PostgreSQL 18.6 + pgvector 0.8.6）
- `torch` 2.10.0 / `torchvision` 0.25.0 / `torchaudio` 2.10.0 / `torch_npu` 2.10.0 / CANN 9.0.0 宿主机与 Docker 严格对齐
- 多卡数据并行（设备令牌队列 + 长度排序分 chunk）

---

## License

内部项目，未公开发布。使用前请联系项目维护者。