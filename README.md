# 催收话术智能推荐系统 (CCA)

> **Collections Conversation Assistant** — 面向催收坐席的实时话术推荐引擎。
>
> 基于 SOP 图谱的双分支混合检索（应用层余弦 + Golden QA）。
> 应用纯 CPU，embedding 由独立 TEI 服务提供，数据库为独立 PostgreSQL 服务。
> 测试环境与生产环境通过一致的 DNS 路径 + 端口对齐。

---

## 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [架构](#架构)
- [目录结构](#目录结构)
- [环境要求](#环境要求)
- [生产环境对齐](#生产环境对齐)
- [配置说明](#配置说明)
- [快速开始](#快速开始)
- [部署脚本](#部署脚本)
- [使用流程](#使用流程)
- [Kubernetes 布局](#kubernetes-布局)
- [TEI 部署](#tei-部署)
- [故障排查](#故障排查)
- [变更记录](#变更记录)

---

## 项目简介

CCA 是一个**离线建库 + 在线检索**的两阶段系统，用于在催收通话过程中根据当前对话上下文为坐席推荐下一句最合适的应答话术。

- **离线阶段**：解析历史通话 → LLM 双级打标（会话级 + 话对级）→ 调用 TEI 生成向量并写入 PostgreSQL → 图谱合成"经典 Q&A"。
- **在线阶段**：解析实时对话 → 双分支混合检索（进程内 embedding 索引 + 图谱 golden QA）→ 图感知投票 → 返回最佳话术。

系统不依赖在线 LLM 推理，在线延迟稳定在 **毫秒级**（典型 5–50 ms）。

---

## 核心特性

| 特性 | 说明 |
|---|---|
| **SOP 图谱驱动** | 阶段（stage）/ 意图（intent）/ 转移（transition）显式建模，检索时按图跳数衰减，避免跨阶段语义漂移 |
| **双分支混合检索** | Branch A: 进程内 embedding 矩阵（BLAS matmul）；Branch B: 内存 golden QA 点积；分数融合 + 平方加权投票 |
| **三种检索模式** | `off`（无约束）/ `guide`（图引导）/ `enforce`（锁定阶段，失败拒绝） |
| **远程 Embedding** | 统一通过 TEI 服务获取向量，进程内**不加载模型**；模型权重与计算全在 TEI 容器 |
| **进程级索引缓存** | `_EmbeddingIndex` 首次查询时全量加载到 numpy 矩阵并预归一化；查询退化为一次 matmul；写入后懒重建 |
| **独立数据库** | PostgreSQL 独立容器，embedding 以 `BYTEA`（float32 原始字节）存储，无 pgvector 依赖 |
| **FastAPI 服务** | `app/server.py` 提供 `/suggest`、`/health`、`/health/ready`；`fastapi-offline` 保证离线环境 `/docs` 可渲染 |
| **生产对齐** | 测试环境与生产使用完全相同的 namespace / Service 名 / DNS 路径 / 端口，可无缝迁移 |

---

## 架构

### 部署拓扑

```text
┌─────────────────────────────────────────────────────────────┐
│  Kubernetes 集群                                             │
│                                                              │
│  ┌───────────────────┐    ┌───────────────────┐             │
│  │  namespace: gv    │    │ namespace:         │             │
│  │                   │    │   model-service    │             │
│  │  StatefulSet:     │    │                   │             │
│  │    postgres       │    │  Deployment:      │             │
│  │                   │    │    tei-svc        │             │
│  │  Service:         │    │                   │             │
│  │    gv-sql-service │    │  Service:         │             │
│  │    :14003→5432    │    │    tei-svc        │             │
│  └─────────┬─────────┘    │    :8010→80       │             │
│            │              └─────────┬─────────┘             │
│            │                        │                        │
│            │  psycopg3              │  HTTP embedding        │
│            │                        │                        │
│            ▼                        ▼                        │
│  ┌──────────────────────────────────────────────┐           │
│  │  namespace: cca                               │           │
│  │                                                │           │
│  │  Deployment: cca-app (FastAPI + 检索引擎)     │           │
│  │  Job:        cca-offline (离线建库)           │           │
│  │  Service:    cca-app:8000                     │           │
│  └────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

### 数据流

```text
离线阶段 (--mode offline)：
  calls.jsonl → parse_dialog → LLMLabeler → PgvectorWriter → qa_pairs
                                                          ↓
                                                  GoldenScriptGenerator
                                                          ↓
                                                    sop_graph.json

在线阶段 (--mode online)：
  raw text → parse_dialog → _build_suggest_query
                                 ↓
                          HybridRetriever
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
              _EmbeddingIndex              golden QA
            (预归一化矩阵 matmul)          (内存点积)
                    └─────────────┬─────────────┘
                                  ▼
                    merge (hop decay) → vote (stage+intent)
                                  ▼
                        best script + alternates
```

---

## 目录结构

```text
.
├── app/
│   ├── settings.py             # Pydantic Settings（.env 加载）
│   ├── logger.py               # 日志
│   ├── config.py               # SOP 阶段/意图/转移静态定义
│   ├── llm.py                  # LLMClient（OpenAI 兼容）
│   ├── utils.py                # DB / TEI Embedder / _EmbeddingIndex / 对话解析
│   ├── offline.py              # LLMLabeler / PgvectorWriter / GoldenScriptGenerator
│   ├── online.py               # HybridRetriever / suggest_script*
│   └── server.py               # FastAPI 服务
├── docker/
│   ├── dockerfile              # 多阶段构建（builder + runtime）
│   └── docker-compose.yml      # 本地 compose（postgres + tei + app profile）
├── scripts/k8s/
│   ├── config.env              # 唯一配置真相源
│   ├── lib/common.sh           # 公共函数（含 sync_app_config）
│   ├── install.sh              # 部署 infra + 模型 + app
│   ├── upload.sh               # 上传模型 + 数据到 PVC
│   ├── offline.sh              # 上传数据 + 跑离线建库
│   ├── uninstall.sh            # 仅清理 cca namespace
│   ├── status.sh               # 状态一览
│   └── manifests/
│       ├── infra-gv.yaml       # PG StatefulSet + Service
│       ├── infra-ms.yaml       # TEI Deployment + Service
│       ├── cca-secret.yaml     # LLM_API_KEY / DB_PASSWORD
│       ├── cca.yaml            # ConfigMap + PVC + Deployment + Service
│       └── offline-job.yaml    # 离线建库 Job
├── data/
│   ├── 202610/                 # 通话数据（批次）
│   └── demo_text.txt
├── models/
│   └── huggingface/
│       └── hub/
│           └── models--BAAI--bge-base-zh-v1.5/   # 本地 HF 缓存
├── output/                     # 离线产物（自动创建）
├── prompts.yaml                # LLM prompt 模板
├── requirements.txt
├── .env-sample
└── README.md
```

---

## 环境要求

### 本地开发

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.11.x | 与依赖 wheel 兼容性最好 |
| Docker | 24.0+ | 构建和运行容器 |
| Docker Compose | v2.20+ | 本地编排 |

### Kubernetes（minikube / 生产）

| 组件 | 版本 | 说明 |
|---|---|---|
| minikube | v1.32+ | 本地测试集群 |
| kubectl | 1.28+ | 建议与集群版本匹配 |
| NVIDIA Container Toolkit | 任意近版 | TEI 需要 GPU（WSL2 + Docker 驱动） |
| StorageClass | `standard` | minikube 默认，动态供应 PV |

### 外部服务

| 服务 | 版本 | 位置 |
|---|---|---|
| TEI | `ghcr.io/huggingface/text-embeddings-inference:89-1.9` | namespace `model-service` |
| PostgreSQL | 16 | namespace `gv` |
| Embedding 模型 | `BAAI/bge-base-zh-v1.5`（768 维） | TEI 容器内加载 |
| LLM | OpenAI 兼容 API | 智谱 / OpenAI / 本地 vLLM 等 |

---

## 生产环境对齐

**测试环境必须与生产环境的接口契约完全一致**——只有 ClusterIP（由集群动态分配）可以不同。

### 对齐参数表

| 项 | 生产 | 测试 | 一致 |
|---|---|---|---|
| PG namespace | `gv` | `gv` | ✅ |
| PG Service 名 | `gv-sql-service` | `gv-sql-service` | ✅ |
| **PG Service port** | **14003** | **14003** | ✅ |
| PG targetPort | 5432 | 5432 | ✅ |
| TEI namespace | `model-service` | `model-service` | ✅ |
| TEI Service 名 | `tei-svc` | `tei-svc` | ✅ |
| **TEI Service port** | **8010** | **8010** | ✅ |
| TEI targetPort | 80 | 80 | ✅ |
| App `DB_HOST` | `gv-sql-service.gv.svc.cluster.local` | 同左 | ✅ |
| App `DB_PORT` | `14003` | `14003` | ✅ |
| App `EMBEDDING_TEI_URL` | `http://tei-svc.model-service.svc.cluster.local:8010/v1/embeddings` | 同左 | ✅ |
| PG ClusterIP | `172.16.193.228` | `10.x.x.x` | ⚠️ 由集群分配，**app 不直接用** |
| TEI ClusterIP | `172.16.193.151` | `10.x.x.x` | ⚠️ 由集群分配，**app 不直接用** |

**ClusterIP 变化不影响 app**——所有连接走 Service DNS，不是 IP。

---

## 配置说明

### `scripts/k8s/config.env`（唯一真相源）

所有脚本 `source` 此文件。修改值后重跑 `install.sh` / `offline.sh` 会自动同步到 k8s ConfigMap。

```bash
# ===== App =====
NS="cca"
IMAGE="ghcr.io/kt-chan/saleschat-amd64:latest"
PF_PORT="8000"
PF_ADDRESS="127.0.0.1"

# ===== Infra 位置（生产对齐）=====
INFRA_PG_NS="gv"
INFRA_PG_SVC="gv-sql-service"
INFRA_PG_PORT="14003"

INFRA_TEI_NS="model-service"
INFRA_TEI_SVC="tei-svc"
INFRA_TEI_PORT="8010"

# ===== App 连接串（写入 cca-config ConfigMap）=====
DB_HOST="gv-sql-service.gv.svc.cluster.local"
DB_PORT="14003"
DB_NAME="cca_db"
DB_USER="cca_user"

EMBEDDING_TEI_URL="http://tei-svc.model-service.svc.cluster.local:8010/v1/embeddings"
EMBEDDING_MODEL="BAAI/bge-base-zh-v1.5"
EMBEDDING_TEI_STYLE="auto"

# ===== LLM（写入 cca-config ConfigMap）=====
LLM_BASE_URL="https://open.bigmodel.cn/api/paas/v4"
LLM_MODEL="GLM-4-FlashX-250414"
# LLM_API_KEY 在 manifests/cca-secret.yaml

# ===== 模型（upload.sh 用）=====
MODEL_ID="BAAI/bge-base-zh-v1.5"
MODEL_DIR_NAME="models--BAAI--bge-base-zh-v1.5"
TEI_PVC="tei-data"
TEI_HF_HOME="/data"
TEI_HUB_SUBDIR="hub"
HF_CACHE_DIR="$HOME/app/saleschat/models/huggingface"

# ===== 数据（upload.sh / offline.sh 用）=====
APP_DATA_PVC="app-data"
APP_OUTPUT_PVC="app-output"
DATA_DIR_LOCAL="$HOME/app/saleschat/data/202610"
CALLS_FILE_PATTERN="*.jsonl"
```

### 配置同步机制

`config.env` 是**唯一真相源**。`install.sh` 和 `offline.sh` 都会调用 `sync_app_config` 将 `LLM_BASE_URL` / `LLM_MODEL` / `DB_HOST` / `DB_PORT` / `EMBEDDING_TEI_URL` 等关键字段 patch 到 `cca-config` ConfigMap。

```text
config.env  →  sync_app_config()  →  ConfigMap cca-config  →  Pod env
```

修改配置只需改 `config.env` 一处，重跑脚本即可。

### 敏感配置

| 项 | 位置 |
|---|---|
| `LLM_API_KEY` | `manifests/cca-secret.yaml` |
| `DB_PASSWORD` | `manifests/cca-secret.yaml`（必须与 `infra-gv.yaml` 里的 `POSTGRES_PASSWORD` 一致） |

---

## 快速开始

### 1. 前置准备

```bash
# 克隆
git clone <repo> saleschat && cd saleschat

# 本地 Python 环境
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 准备敏感配置
vim scripts/k8s/manifests/cca-secret.yaml
# 填入真实 LLM_API_KEY / DB_PASSWORD
```

### 2. 准备本地数据

```bash
# 通话数据
ls data/202610/
# part_001.jsonl  part_002.jsonl  ...

# 本地 HF 模型缓存
ls models/huggingface/hub/
# models--BAAI--bge-base-zh-v1.5/
```

### 3. 部署服务

```bash
cd scripts/k8s
chmod +x *.sh lib/*.sh

# 部署 infra（PG + TEI）+ 模型 + app
./install.sh
```

**`install.sh` 会自动完成**：

- 处理 StatefulSet 不可变字段冲突（自动删除重建，PVC 保留）
- 部署 `gv/postgres` + `model-service/tei-svc`
- 上传模型（若缺，调 `upload.sh --model-only`）
- 构建/加载 app 镜像到 minikube
- 部署 `cca/cca-app`
- **从 `config.env` 同步关键配置到 ConfigMap**
- 验证 DB / TEI 连通性
- 启动后台 port-forward

### 4. 上传数据 + 离线建库

```bash
./offline.sh
# 上传 data/202610/ → PVC /app/data/202610/
# 跑 Job，产出 qa_pairs + sop_graph.json
# 完成后自动拉起 app（默认至少 1 副本）
```

### 5. 验证

```bash
./status.sh

# 或手动
curl -s http://127.0.0.1:8000/health/ready | jq .
# 期望：{"status":"ready","detail":{"postgres":"ok","embedding":"ok"}}
```

---

## 部署脚本

### `install.sh` — 部署服务

```bash
./install.sh                       # 标准安装（infra + 模型 + app）
./install.sh --dev                 # 开发模式（跳过 infra 检查 + 构建镜像）
./install.sh --skip-infra          # 只部署 app（假定 infra 就绪）
./install.sh --infra-only          # 只部署 infra
./install.sh --upload-data         # 部署完自动上传数据
./install.sh --no-upload           # 不上传模型
./install.sh --no-pf               # 不启动 port-forward
./install.sh --pf-port 8001        # 自定义端口
```

### `upload.sh` — 上传模型 / 数据

```bash
./upload.sh                        # 模型 + 数据
./upload.sh --model-only           # 只上传模型
./upload.sh --data-only            # 只上传数据
./upload.sh --data-dir data/202611 # 指定数据源目录
./upload.sh --batch 202611-v2      # 自定义批次名
./upload.sh --force                # 强制重传
./upload.sh --dry-run              # 只打印不执行
```

### `offline.sh` — 跑离线建库

```bash
./offline.sh                       # 上传数据 + 跑 Job + 拉起 app
./offline.sh 202611                # 指定批次
./offline.sh --no-upload           # 跳过上传（数据已在 PVC）
./offline.sh --data-dir <path>     # 明确数据源
./offline.sh --force-upload        # 强制重传
./offline.sh --keep-down           # 跑完不拉起 app
./offline.sh --replicas 2          # 跑完恢复指定副本数
```

**默认行为**：无论 app 之前是运行还是停止，`offline.sh` 结束后**必拉起 app 至少 1 副本**。

### `uninstall.sh` — 卸载

```bash
./uninstall.sh                     # 停 cca workload，保留 PVC/config
./uninstall.sh --purge             # 额外删 cca PVC/config（数据丢失）
./uninstall.sh --all               # 删除整个 cca namespace
```

**安全约束**：`uninstall.sh` **只操作 `cca` namespace**，绝不触碰 `gv` 和 `model-service`（共享基础设施）。

### `status.sh` — 状态一览

```bash
./status.sh
# 显示：所有 namespace 的 Pod / Service / app env / 健康检查 / port-forward
```

---

## 使用流程

### 首次部署

```bash
cd scripts/k8s

./install.sh               # 部署服务
./offline.sh               # 上传数据 + 建库
./status.sh                # 验证
```

### 日常迭代

| 场景 | 命令 |
|---|---|
| 改代码，重新部署 app | `./install.sh --skip-infra` |
| 改配置（`config.env`），重新部署 | `./install.sh --skip-infra` |
| 换数据批次 | `./offline.sh 202611` |
| 只重跑建库（数据已在 PVC） | `./offline.sh --no-upload` |
| 换模型 | `./upload.sh --model-only --force` |
| 停止所有服务 | `./uninstall.sh` |
| 清数据重来 | `./uninstall.sh --purge && ./install.sh && ./offline.sh` |

### 编程接口

```python
from app.online import suggest_script_for_conversation

text = (
    "催收员：您好，请问是张先生吗？\n"
    "客户：是我。\n"
    "催收员：您名下信用卡本期逾期1900元，请问能处理吗？\n"
    "客户：我最近被裁员了，暂时没有收入。"
)

# 带 stage hint（快速通道）
res = suggest_script_for_conversation(text, stage_hint="04 方案协商与签署")
print(res["script"], res["stage"], res["intent"], res["score"])

# 自动推断（两趟检索）
res = suggest_script_for_conversation(text)

# 只要话术
from app.online import suggest_script
print(suggest_script(text))
```

### FastAPI 端点

| 端点 | 说明 |
|---|---|
| `POST /suggest` | 话术推荐主接口；`conversation` 必填，`stage_hint` 可选 |
| `GET /health` | liveness，仅检查进程存活 |
| `GET /health/ready` | readiness，探测 PG + TEI，任一不通返回 503 |
| `GET /docs` | Swagger UI（fastapi-offline，无外网可用） |

```bash
curl -s http://localhost:8000/suggest \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": "催收员：您好，您的账单已逾期三天。\n客户：我最近资金周转困难。",
    "stage_hint": "04 方案协商与签署"
  }' | jq .
```

---

## Kubernetes 布局

### Namespace 职责

| Namespace | 用途 | 维护方 | 脚本权限 |
|---|---|---|---|
| `gv` | PostgreSQL（共享） | 运维 / 其他团队 | **只读** |
| `model-service` | TEI（共享） | 运维 / 其他团队 | **只读** |
| `cca` | 本应用（app + Job + 数据） | 本应用 | 完全控制 |

**`install.sh` / `uninstall.sh` 绝不修改 `gv` / `model-service` 的 Pod / Service / PVC**，只 `kubectl get` 检查就绪状态。

### Service 端口映射

| Service | port（对外） | targetPort（容器内） |
|---|---|---|
| `gv/gv-sql-service` | 14003 | 5432 |
| `model-service/tei-svc` | 8010 | 80 |
| `cca/cca-app` | 8000 | 8000 |

### PVC

| PVC | namespace | 大小 | 用途 |
|---|---|---|---|
| `postgres-data` | gv | 5Gi | PG 数据 |
| `tei-data` | model-service | 2Gi | HF 缓存 / 模型 |
| `app-data` | cca | 2Gi | 通话数据 `/app/data/<batch>/` |
| `app-output` | cca | 2Gi | 离线产物 `/app/output/` |

---

## TEI 部署

### GPU 镜像选择

TEI 官方镜像按 **Compute Capability**（不是 CUDA 版本）：

| 前缀 | 架构 | 代表 GPU |
|---|---|---|
| `86-` | Ampere (CC 8.6) | A1000、A10、RTX 30xx |
| `89-` | Ada Lovelace (CC 8.9) | **RTX 2000 Ada**、RTX 40xx、L4 |
| `90-` | Hopper (CC 9.0) | H100 |
| `cuda-` | 多合一（启动时自动探测） | 任意 |

**RTX 2000 Ada Generation 使用 `89-1.9`**。选错 tag 会导致容器内 `CUDA_ERROR_NO_DEVICE`，回退 CPU。

### Deployment 关键配置

```yaml
spec:
  containers:
    - name: tei
      image: ghcr.io/huggingface/text-embeddings-inference:89-1.9
      command: ["text-embeddings-router"]   # ⚠️ 绕过 entrypoint.sh
      args:
        - "--model-id"
        - "BAAI/bge-base-zh-v1.5"
        - "--served-model-name"
        - "BAAI/bge-base-zh-v1.5"
        - "--port"
        - "80"
      env:
        - name: HF_HOME
          value: /data
        - name: HF_HUB_OFFLINE
          value: "1"
      resources:
        limits:
          nvidia.com/gpu: 1
```

### 验证 GPU 生效

```bash
# 日志应显示 "Starting Bert model on Cuda"
kubectl -n model-service logs deploy/tei-svc | grep -E "Bert model|Cuda|Cpu"

# 从容器内确认 GPU 可见
kubectl -n model-service exec deploy/tei-svc -- nvidia-smi -L
```

### 宿主机前置检查

```bash
nvidia-smi                                     # 宿主机可见 GPU
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

第二条命令失败说明 **NVIDIA Container Toolkit** 未装好：

```bash
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 故障排查

### App 启动失败

| 现象 | 排查 |
|---|---|
| `CreateContainerConfigError` | 缺 `cca-secret` 或 `cca-config`。检查 `kubectl -n cca get cm,secret` |
| `failed to resolve host 'postgres'` | ConfigMap 的 `DB_HOST` 是旧值。检查 `kubectl -n cca get cm cca-config -o jsonpath='{.data.DB_HOST}'`，应为 `gv-sql-service.gv.svc.cluster.local` |
| `ImagePullBackOff` | 镜像不在 minikube。`minikube image load $IMAGE` 或 `./install.sh --dev` |
| `Pending: unbound PVC` | `storageClassName` 错误或 storage-provisioner 未跑。`kubectl get sc` / `minikube addons enable storage-provisioner` |

### Infra 部署失败

| 现象 | 排查 |
|---|---|
| `StatefulSet "postgres" is invalid: spec.serviceName: Invalid value: field is immutable` | `install.sh` 的 `pre_apply_infra` 会自动处理；手动：`kubectl -n gv delete sts postgres && kubectl apply -f manifests/infra-gv.yaml` |
| TEI `spec.replicas=0` | 上次 upload 未恢复。`kubectl -n model-service scale deploy/tei-svc --replicas=1` |
| TEI 一直 `0/1`（未 Ready） | 模型缺失。`./upload.sh --model-only` 后重试 |

### 离线建库失败

| 现象 | 排查 |
|---|---|
| Job 找不到数据 | `kubectl -n cca exec deploy/cca-app -- ls /app/data/<batch>/`。若无数据：`./upload.sh --data-only` |
| LLM `403 Country, region, or territory not supported` | 打到了错误 endpoint 或地区限制。检查 `kubectl -n cca get cm cca-config -o jsonpath='{.data.LLM_BASE_URL}'`；如仍 403，说明该 LLM 服务商在你主机所在地区不可用 |
| TEI 连接超时 | `kubectl -n cca exec deploy/cca-app -- curl -sf http://tei-svc.model-service:8010/health` |

### 常见误用

| 错误操作 | 后果 | 正确做法 |
|---|---|---|
| 手动 `kubectl delete ns gv` / `kubectl delete ns model-service` | **破坏整个生产环境的共享 PG / TEI** | 只在本地 minikube 测试集群使用；生产环境**永久禁止** |
| 改 `manifests/cca.yaml` 的 ConfigMap 但不重启 Pod | 新配置不生效 | `./install.sh --skip-infra` 或 `kubectl rollout restart deploy/cca-app` |
| `kubectl cp` 上传大目录到 PVC | 易失败或超时 | `./upload.sh`（内部用 tar 管道） |

---

## 变更记录

### 本轮（生产对齐 + 完整部署脚本）

- **架构**：从"进程内推理"改为"应用纯 CPU + 独立 TEI + 独立 PostgreSQL"三服务架构
- **Embedding**：移除进程内 `_LocalEmbedder`（CUDA/NPU/CPU 多后端），只保留 `_TEIEmbedder`（HTTP 客户端）
- **Database**：移除 `pgvector` 与 `embedded-postgres`；`qa_pairs.embedding` 改为 `BYTEA`（float32 原始字节），自动从 `jsonb` 迁移
- **检索**：从 SQL `<=>` 算子改为进程内 `_EmbeddingIndex`（预归一化矩阵 + `argpartition` top-k）
- **FastAPI**：新增 `app/server.py`，`fastapi-offline` 保证离线 `/docs` 可渲染，`/health` 与 `/health/ready` 分离
- **Docker**：多阶段 Dockerfile，builder 安装编译链（arm64 上 hdbscan 需源码编译），runtime 只保留 `.so`；镜像从 ~6.5GB 缩至 ~750MB
- **Kubernetes 生产对齐**：
  - PG：`gv/gv-sql-service:14003 → targetPort 5432`
  - TEI：`model-service/tei-svc:8010 → targetPort 80`
  - App：`cca/cca-app:8000`
- **脚本**：
  - `install.sh` 自动处理 StatefulSet 不可变字段冲突（PVC Retain 保护，自动删除重建）
  - `upload.sh` 用 tar 管道替代 `kubectl cp`（大目录稳定），`scale_down` 输出到 stderr（修复 integer expression bug）
  - `offline.sh` 结束时**强制拉起 app**（默认至少 1 副本），支持 `--keep-down` / `--replicas`
  - `uninstall.sh` **只清 cca**，绝不触碰 gv / model-service
  - `lib/common.sh` 新增 `sync_app_config()`：从 `config.env` 同步到 `cca-config` ConfigMap
- **CI**：双架构（amd64 + arm64）原生 runner 构建；新增 `Assert both platforms in manifest` 步骤

### 历史（已废弃）

- 本地推理（CUDA / NPU / CPU 多后端）+ pgvector + `embedded-postgres`
- 多卡数据并行（设备令牌队列 + 长度排序分 chunk）
- 双 target Dockerfile（`nvidia` / `ascend`）

---

## License

内部项目，未公开发布。使用前请联系项目维护者。