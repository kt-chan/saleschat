# CCA Kubernetes 部署

`start.sh` / `stop.sh` 双脚本方案，零外部依赖（`envsubst` 可选）。默认行为是生产部署（跳过 infra，用已有 PG/TEI），本地开发用 `dev` 模式全量部署。

---

## 目录

- [前置条件](#前置条件)
- [首次部署](#首次部署)
- [命令速查](#命令速查)
- [常见场景](#常见场景)
- [配置修改生效规则](#配置修改生效规则)
- [目录结构](#目录结构)
- [排障](#排障)
- [无 GPU 集群](#无-gpu-集群)
- [清理](#清理)
- [一句话速记](#一句话速记)

---

## 前置条件

一次性检查，全部通过才能开始。

```bash
# 1. kubectl 可用且能连上集群
kubectl cluster-info

# 2. 集群能识别 GPU（TEI 需要）
kubectl get node -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}'
# 期望：1（无 GPU 见「无 GPU 集群」）

# 3. app 镜像已存在于集群
docker images | grep saleschat

# 4. 本地已有模型和数据（仅 dev 模式需要模型）
ls ~/app/saleschat/models/huggingface/hub/
ls ~/app/saleschat/data/
```

---

## 首次部署

### 步骤 1：准备目录

```bash
cd ~/app/saleschat/scripts/k8s
ls
```

应包含：

```
.env.example
start.sh
stop.sh
README.md
manifests/
└── template/                  # 模板（manifests/runtime/ 由 start.sh 生成）
    ├── 00-namespace-app.yaml
    ├── 00-namespace-infra.yaml
    ├── 01-pvc-app.yaml
    ├── 01-pvc-infra.yaml
    ├── 02-secret-app.yaml
    ├── 02-secret-infra.yaml
    ├── 03-configmap.yaml
    ├── 04-postgres.yaml
    ├── 05-tei.yaml
    ├── 06-app.yaml
    ├── 07-offline-job.yaml
    └── upload/
        ├── app-upload.yaml
        └── tei-upload.yaml
```

### 步骤 2：创建 `.env`

```bash
cp .env.example .env
vim .env
```

**⚠️ 必须修改的 4 项**（不填无法启动）：

| 变量 | 说明 | 示例 |
|---|---|---|
| `LLM_API_KEY` | LLM 的 API Key | `sk-xxxxxxxx` |
| `HF_CACHE_DIR` | 本地 HF 缓存根目录（含 `hub/` 子目录） | `$HOME/app/saleschat/models/huggingface` |
| `DATA_DIR_LOCAL` | 本地数据目录（含 `*.jsonl`） | `$HOME/app/saleschat/data/202610` |
| `DATA_DIR_BATCH` | 容器内批次子目录名，决定 `DATA_DIR=/app/data/<BATCH>` | `202610` |

**按需修改**：

| 变量 | 默认值 | 何时改 |
|---|---|---|
| `LLM_BASE_URL` / `LLM_MODEL` | 智谱 GLM | 用其他 LLM |
| `APP_IMAGE` | `ghcr.io/kt-chan/saleschat-amd64:latest` | 镜像在别处 |
| `IMAGE_PULL_POLICY` | `Never` | 镜像在 registry 时改 `IfNotPresent` |
| `NS` / `INFRA_PG_NS` / `INFRA_TEI_NS` | cca / gv / model-service | 集群布局不同 |
| `PF_PORT` | `8000` | 端口冲突 |

### 步骤 3：准备镜像

```bash
# minikube 场景
minikube image ls | grep saleschat

# 若不存在，构建 + 导入
cd ~/app/saleschat
docker build -f docker/dockerfile -t ghcr.io/kt-chan/saleschat-amd64:latest .
minikube image load ghcr.io/kt-chan/saleschat-amd64:latest
```

### 步骤 4：一键部署

#### 生产环境（默认，推荐）

```bash
cd ~/app/saleschat/scripts/k8s
chmod +x start.sh stop.sh

./start.sh
```

`./start.sh` 无参数执行**生产部署**（默认）：

```
1. render    从 .env + manifests/template/ 渲染出 manifests/runtime/*.yaml
2. upload    只上传数据到 app-data PVC
             （跳过模型上传，由 infra 团队管理；已就绪自动跳过）
3. up        只 apply app 相关资源：
                00-namespace-app / 01-pvc-app / 02-secret-app
                03-configmap / 06-app
             跳过 04-postgres / 05-tei
             等待已有 PG / TEI 就绪（只读 wait）
4. offline   暂停 app → 跑 cca-offline Job → 恢复 app
5. pf        后台 port-forward :8000
```

**生产模式的安全保证**：

- 不创建 `gv` / `model-service` namespace
- 不 apply 任何 infra yaml
- 不碰 `tei-data` PVC
- 不重启生产 TEI
- 所有 infra 操作都是 `kubectl wait`（只读）

**预期输出**：

```
[16:35:01] render → manifests/runtime/
[16:35:01]   ✅ 12 文件
[16:35:01] upload
[16:35:01]   SKIP_MODEL_UPLOAD=1 → 跳过模型上传（infra 团队负责）
[16:35:02]   数据已就绪 (5/5) → 跳过
[16:35:02]   ✅ 模型和数据均已在 PVC，跳过 upload
[16:35:02] apply manifests
[16:35:02]   SKIP_INFRA=1 → 跳过 PG/TEI 部署（使用已有 Service）
[16:35:03]   配置未变 → 跳过 app 重启
[16:35:03]   等待 postgres
[16:35:15]   等待 TEI
[16:35:47]   等待 app
[16:35:52]   ✅
[16:35:52]   暂停 app (1 → 0)
[16:35:54]   运行 Job
[16:44:30]   恢复 app (1)
[16:44:55]   ✅
[16:44:55] port-forward 127.0.0.1:8000
[16:44:57]   ✅ http://127.0.0.1:8000
```

#### 本地 / 测试环境（全量）

```bash
./start.sh dev
```

`dev` 模式执行**完整部署**（约 **10~40 分钟**）：

```
1. render    渲染 manifests/runtime/*.yaml
2. upload    上传模型 + 数据到 PVC
             （会暂停 app 和 TEI 释放 RWO PVC；已就绪则跳过）
3. up        部署所有资源：infra + app
             等 PG Ready → 等 TEI Ready → 重启 app
4. offline   暂停 app → 跑 cca-offline Job → 恢复 app
5. pf        后台 port-forward :8000
```

**首次运行**（PVC 全空）会看到真正的上传过程：

```
[16:35:01] upload
[16:35:10]   暂停 app (0 → 0)
[16:35:10]   暂停 TEI (0 → 0)
[16:35:18]   起 upload pods
[16:35:25]   模型: bge-base-zh-v1.5 (781M)
[16:36:45]   数据: → /app/data/202610/ (5 个)
[16:36:52]   恢复 TEI (0)
[16:36:53]   恢复 app (0)
[16:36:53]   ✅
```

### 步骤 5：验证

```bash
# 三个 Pod 都是 Running
kubectl get pods -n gv
kubectl get pods -n model-service
kubectl get pods -n cca

# 健康检查
curl -s http://127.0.0.1:8000/health/ready | jq .
# 期望：{"status":"ready","detail":{"postgres":"ok","embedding":"ok"}}

# API 文档
open http://127.0.0.1:8000/docs

# 验证数据已入库
kubectl -n cca exec deploy/cca-app -- python -c '
from app.utils import get_db_conn, release_db_conn
c = get_db_conn()
c.cursor().execute("SELECT count(*) FROM qa_pairs")
print(c.cursor().fetchone()[0])
release_db_conn(c)
'
# 期望：非零数字

# 在线接口测试
curl -s -X POST http://127.0.0.1:8000/suggest \
  -H "Content-Type: application/json" \
  -d '{"conversation":"客户：我最近资金困难。"}' | jq .
```

**部署完成，系统可用。**

---

## 命令速查

### `start.sh`（部署 / 更新 / 运行）

| 命令 | 用途 | 耗时 |
|---|---|---|
| `./start.sh` | **生产部署**（默认）：跳过 infra + 跳过模型 | 5~30 分钟 |
| `./start.sh dev` | **完整部署**：本地/测试环境（部署 infra + 上传模型） | 10~40 分钟 |
| `./start.sh online` | 只部署服务（迭代用） | 3~60 秒 |
| `./start.sh render` | 只渲染 manifests/runtime/ | 秒级 |
| `./start.sh upload` | 上传模型/数据（已就绪自动跳过） | 0~5 分钟 |
| `./start.sh up` | 只部署服务（配置未变不重启） | 3~60 秒 |
| `./start.sh offline` | 跑离线建库 Job | 5~30 分钟 |
| `./start.sh pf` | 起 port-forward | 秒级 |
| `./start.sh pf-stop` | 停 port-forward | 秒级 |
| `./start.sh status` | 查看部署状态 | 秒级 |
| `./start.sh help` | 帮助 | 秒级 |

### `stop.sh`（停止 / 清理）

| 命令 | 删除内容 | 数据 |
|---|---|---|
| `./stop.sh` | cca 内 workload（app + config + secret） | ✅ PVC 保留 |
| `./stop.sh --purge` | cca 内 workload + PVC + config | ❌ cca 内数据丢失 |
| `./stop.sh --all` | 整个 cca namespace | ❌ cca 内全部删除 |

**⚠️ 安全保证**：`stop.sh` 的所有模式都**永不触碰** `gv` / `model-service` namespace 及其中的任何资源。

### 智能跳过机制

`start.sh` 内部有三处幂等优化，稳态下不会做多余工作：

| 操作 | 判断依据 | 已就绪时 |
|---|---|---|
| **模型上传** | `exec deploy/tei-svc -- test -f /data/models/<short>/config.json` | 不停 TEI，跳过上传 |
| **数据上传** | `exec deploy/cca-app -- find /app/data/<batch> \| wc -l` 与本地数量比对 | 不停 app，跳过上传 |
| **app 重启** | ConfigMap + Secret 内容 hash 前后对比 | 配置未变不重启 |

---

## 常见场景

### 场景 1：改了业务配置（LLM、DB 等）

```bash
vim .env             # 改配置
./start.sh online    # 重新渲染 + 部署（配置变了会自动重启 app）
```

`online` 会检测配置变化，只有内容真变了才 `rollout restart` app。**不用重传模型/数据。**

---

### 场景 2：换数据批次

```bash
# 假设新数据在 ~/app/saleschat/data/202611/
vim .env
# DATA_DIR_LOCAL=$HOME/app/saleschat/data/202611
# DATA_DIR_BATCH=202611

./start.sh upload offline
```

`upload` 检测到远端文件数与本地不一致，会停 app、上传新数据、重启 app。**TEI 全程在线**（不传模型）。

`offline` 会用新数据重建 SOP 图谱。

---

### 场景 3：改了业务代码，重建镜像

```bash
cd ~/app/saleschat
docker build -f docker/dockerfile -t ghcr.io/kt-chan/saleschat-amd64:latest .
minikube image load ghcr.io/kt-chan/saleschat-amd64:latest

cd scripts/k8s
FORCE_RESTART=1 ./start.sh online    # 强制重启 app 拉新镜像
```

> 因为 ConfigMap 内容未变，`online` 不会自动重启 app。`FORCE_RESTART=1` 强制拉新镜像。

---

### 场景 4：临时停止，稍后重启

```bash
./stop.sh            # 停 workload（保留所有数据）
./start.sh online    # 重新起来（约 30 秒）
```

**关键**：`./stop.sh` 保留 PVC，模型和数据都不用重传，`up` 时 TEI 从 PVC 加载模型（30 秒内就绪）。

---

### 场景 5：清空 cca 内数据重建

```bash
./stop.sh --purge    # 删 cca 内 workload + PVC
./start.sh           # 生产模式重新部署（会重新上传数据 + 建库）
```

> `--purge` 只删 `cca` namespace 内的 PVC（`app-data`、`app-output`），不删 `postgres-data` / `tei-data`。

---

### 场景 6：彻底重置 cca（含 namespace）

```bash
./stop.sh --all      # 删 cca namespace 本身
./start.sh           # 生产模式重新部署
```

> `--all` 只删 `cca` namespace，**不删 `gv` / `model-service`**。

---

### 场景 7：换 embedding 模型（仅 dev 模式）

```bash
# 1. 本地准备新模型
ls ~/app/saleschat/models/huggingface/hub/models--BAAI--bge-large-zh-v1.5/

# 2. 改 .env
vim .env
# MODEL_ID=BAAI/bge-large-zh-v1.5
# MODEL_DIR_NAME=models--BAAI--bge-large-zh-v1.5
# MODEL_SHORT_NAME=bge-large-zh-v1.5

# 3. 上传 + 部署 + 重建库
./start.sh upload up offline
```

**警告**：换模型会改变向量维度（bge-base=768，bge-large=1024），**旧的 embedding 数据全部失效**。必须重新 `offline` 建库。

**生产换模型**：由 infra 团队更新 TEI，app 侧无需操作（`EMBEDDING_MODEL` 与 TEI 侧对齐即可）。

---

### 场景 8：只重建库（模型数据都没变）

```bash
./start.sh offline
```

例如：改了 `prompts.yaml`，或想用同样的数据重新生成 SOP 图谱。

---

### 场景 9：查看日志

```bash
# app 日志
kubectl -n cca logs -f deploy/cca-app

# TEI 日志
kubectl -n model-service logs -f deploy/tei-svc

# PG 日志
kubectl -n gv logs -f postgres-0

# offline Job 日志
kubectl -n cca logs -f job/cca-offline

# port-forward 日志
tail -f scripts/k8s/.pf.log
```

---

### 场景 10：进入容器调试

```bash
kubectl -n cca exec -it deploy/cca-app -- bash
```

容器内：

```bash
# 查看环境变量
env | grep -E "DB_|LLM_|EMBEDDING_|DATA_DIR"

# 测 DB
python -c "from app.utils import get_db_conn; print(get_db_conn())"

# 测 TEI
python -c "from app.utils import get_embedder; print(get_embedder().encode('test'))"

# 查数据
ls -la /app/data/
ls -la /app/output/
```

---

### 场景 11：查看部署状态

```bash
./start.sh status
```

输出示例：

```
=== gv ===
NAME             READY   STATUS    AGE
pod/postgres-0   1/1     Running   10m

NAME                     TYPE        PORT(S)
service/gv-sql-service   ClusterIP   ...:14003/TCP

=== model-service ===
...

=== cca ===
...

PF: Running → http://127.0.0.1:8000
```

---

## 配置修改生效规则

改完 `.env` 后，用对应命令应用：

| 修改项 | 生效命令 | 是否重启 app |
|---|---|---|
| `LLM_BASE_URL` / `LLM_MODEL` / `EMBEDDING_TEI_URL` 等 ConfigMap 内容 | `./start.sh online` | ✅ 自动 |
| `DB_PASSWORD` / `LLM_API_KEY`（Secret） | `./start.sh online` | ✅ 自动 |
| `APP_IMAGE`（镜像地址） | `FORCE_RESTART=1 ./start.sh online` | ✅ 强制 |
| `DATA_DIR_LOCAL` / `DATA_DIR_BATCH`（数据源） | `./start.sh upload offline` | ✅ 上传时停/启 |
| `HF_CACHE_DIR` / `MODEL_DIR_NAME` / `MODEL_SHORT_NAME`（模型源） | `./start.sh upload up offline` | ✅ 上传时停/启 |
| `NS` / `INFRA_PG_NS` / `INFRA_TEI_NS` | 先 `./stop.sh --all`，再 `./start.sh` | 全新部署 |
| `PF_PORT` | `./start.sh pf-stop && ./start.sh pf` | 只重启 PF |

**核心原则**：改 `.env` → `./start.sh online`（配置类）或 `./start.sh upload offline`（数据/模型类），脚本会判断需要做什么。

---

## 目录结构

```
scripts/k8s/
├── .env.example
├── .env                    # cp .env.example .env 后修改（gitignore）
├── start.sh
├── stop.sh
├── README.md
├── .pf.pid                 # start.sh 生成（gitignore）
├── .pf.log                 # start.sh 生成（gitignore）
└── manifests/
    ├── template/           # 模板（源，git 跟踪）
    │   ├── 00-namespace-app.yaml
    │   ├── 00-namespace-infra.yaml
    │   ├── 01-pvc-app.yaml
    │   ├── 01-pvc-infra.yaml
    │   ├── 02-secret-app.yaml
    │   ├── 02-secret-infra.yaml
    │   ├── 03-configmap.yaml
    │   ├── 04-postgres.yaml
    │   ├── 05-tei.yaml
    │   ├── 06-app.yaml
    │   ├── 07-offline-job.yaml
    │   └── upload/
    │       ├── app-upload.yaml
    │       └── tei-upload.yaml
    └── runtime/            # start.sh render 生成（gitignore）
        ├── 00-namespace-app.yaml
        ├── ...
        ├── 07-offline-job.yaml
        └── upload/
```

`.gitignore`：

```gitignore
.env
manifests/runtime/
.pf.pid
.pf.log
```

---

## 排障

### 常见错误

| 现象 | 原因 | 修法 |
|---|---|---|
| `StatefulSet "postgres" is invalid: spec.serviceName ... immutable` | 之前 sts 的 `serviceName` 不同 | `kubectl delete sts postgres -n gv`，再 `./start.sh dev` |
| `Multi-Attach error for volume ...` | PVC 被其他 Pod 占用 | `start.sh upload` 会自动停 app/tei，无需手动 |
| `TEI Pod Pending`（`0/1 nodes are available: nvidia.com/gpu`） | 无 GPU 或 device plugin 未就绪 | 见「无 GPU 集群」 |
| `CreateContainerConfigError` | Secret 或 ConfigMap 缺失 | `./start.sh online`（会自动 apply） |
| `ImagePullBackOff` | 镜像不在集群里 | `minikube image load xxx`，或改 `IMAGE_PULL_POLICY=IfNotPresent` |
| `pf 30s 未就绪` | app 还没 Ready | `kubectl -n cca get pods` 查 app |
| offline Job `CrashLoopBackOff` | 数据目录不存在 / 配置错误 | `kubectl -n cca logs job/cca-offline --tail=100` |
| `/suggest` 返回 `SOP 图谱未找到` | 没跑过 offline 或 offline 失败 | `./start.sh offline`，见下节 |
| offline 报 `目录下未找到匹配文件: /app/data` | `DATA_DIR_BATCH` 未配或模板没改 | 见下节 |
| 生产模式 `forbidden` 报错 | 无 `gv` / `model-service` 写权限 | 默认模式应该已跳过；检查 `SKIP_INFRA` 是否被环境变量覆盖 |

### `/suggest` 返回 `SOP 图谱未找到`

```bash
# 1. 确认 SOP 图谱是否存在
POD=$(kubectl -n cca get pod -l app=cca-app -o jsonpath='{.items[0].metadata.name}')
kubectl -n cca exec $POD -- ls -la /app/output/

# 2. 若没有，检查 offline 是否跑过
kubectl -n cca get job cca-offline
kubectl -n cca get pods -l job-name=cca-offline

# 3. 没有就重新跑
./start.sh offline

# 4. 若已有 job 但失败，看日志
kubectl -n cca logs -l job-name=cca-offline --tail=100
```

### offline 报 `目录下未找到匹配文件: /app/data`

说明 `DATA_DIR` 没指向数据子目录。检查：

```bash
# 1. ConfigMap 的 DATA_DIR
kubectl -n cca get cm cca-config -o jsonpath='{.data.DATA_DIR}{"\n"}'
# 期望：/app/data/<DATA_DIR_BATCH>

# 2. 数据实际在哪
kubectl -n cca exec deploy/cca-app -- ls /app/data/

# 3. 若 DATA_DIR 是 /app/data，说明 .env 的 DATA_DIR_BATCH 未生效
grep DATA_DIR_BATCH .env
# 期望：DATA_DIR_BATCH=202610

# 4. 修完重新渲染 + 部署 + 建库
./start.sh online offline
```

### TEI 仍在下载模型

如果日志看到 `Starting download`，说明 TEI 没认出本地模型：

```bash
# 1. 确认模型在扁平路径
kubectl -n model-service run check --image=ghcr.io/kt-chan/saleschat-amd64:latest \
  --rm -it --restart=Never --image-pull-policy=Never \
  --overrides='{"spec":{"containers":[{"name":"c","image":"ghcr.io/kt-chan/saleschat-amd64:latest","imagePullPolicy":"Never","command":["ls","-la","/data/models/bge-base-zh-v1.5/"],"volumeMounts":[{"name":"d","mountPath":"/data"}]}],"volumes":[{"name":"d","persistentVolumeClaim":{"claimName":"tei-data"}}]}}'
# 期望看到实体文件（非软链）

# 2. 确认 TEI 的 --model-id
kubectl -n model-service get deploy tei-svc -o yaml | grep -A2 "model-id"
# 期望：- "--model-id" - "/data/models/bge-base-zh-v1.5"

# 3. 若不对，重新上传模型
./start.sh dev   # 或 ./start.sh upload（若 model 已被清除）
```

### 诊断命令

```bash
# 三个 namespace 全览
kubectl get all -n cca
kubectl get all -n gv
kubectl get all -n model-service

# PVC 状态
kubectl get pvc -n cca
kubectl get pvc -n gv
kubectl get pvc -n model-service

# Pod 详细事件
kubectl -n cca describe pod <pod-name> | tail -30

# app 实际 env（确认 ConfigMap 生效）
POD=$(kubectl -n cca get pod -l app=cca-app -o jsonpath='{.items[0].metadata.name}')
kubectl -n cca exec $POD -- env | grep -E "DB_|LLM_|EMBEDDING_|DATA_DIR"

# 端到端健康检查
curl -s http://127.0.0.1:8000/health/ready | jq .
```

---

## 无 GPU 集群

如果集群没有 NVIDIA GPU：

```bash
# 1. 修改模板
vim manifests/template/05-tei.yaml

# 2. 删除 nvidia.com/gpu: 1 这一行
#    并把 image 改为 CPU 版本：
#    image: ghcr.io/huggingface/text-embeddings-inference:cpu-1.9

# 3. 重新部署（仅 dev 模式需要）
./start.sh dev
```

CPU 版 TEI：加载模型慢（5~10 分钟），吞吐低（~30 q/s），但功能完整。

生产环境：由 infra 团队决定 TEI 的 GPU 配置，app 侧无需关心。

---

## 清理

```bash
# 只停 app（保留 PVC / config / secret，30 秒重启）
./stop.sh

# 清 cca 内所有资源（含 PVC，cca 内数据丢失）
./stop.sh --purge

# 删 cca namespace（cca 内所有资源消失）
./stop.sh --all

# 确认 gv / model-service 完全不受影响
kubectl -n gv get pods
kubectl -n model-service get pods

# 本地清理（可选）
rm -rf manifests/runtime/ .pf.pid .pf.log
```

**如需清理本地测试用的 infra**，手动执行（`stop.sh` 不提供此能力）：

```bash
# 确认当前是测试集群
kubectl config current-context

# 手动删除（生产绝对禁止）
kubectl delete ns gv model-service
```

---

## 一句话速记

| 需求 | 命令 |
|---|---|
| **生产首次部署** | `cp .env.example .env && vim .env && ./start.sh` |
| **本地/测试部署** | `cp .env.example .env && vim .env && ./start.sh dev` |
| **改配置** | `vim .env && ./start.sh online` |
| **改代码** | `docker build ... && minikube image load ... && FORCE_RESTART=1 ./start.sh online` |
| **换数据** | 改 `.env` 的 `DATA_DIR_LOCAL` 和 `DATA_DIR_BATCH` → `./start.sh upload offline` |
| **只重建库** | `./start.sh offline` |
| **临时停** | `./stop.sh` → `./start.sh online` |
| **清 cca 数据重建** | `./stop.sh --purge && ./start.sh` |
| **彻底重置 cca** | `./stop.sh --all && ./start.sh` |
| **看状态** | `./start.sh status` |
| **看日志** | `kubectl -n cca logs -f deploy/cca-app` |
