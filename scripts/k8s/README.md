# SalesChat / CCA Kubernetes 部署与运维指南

本项目采用 `start.sh` / `stop.sh` 双脚本方案，实现了从本地开发到  内网生产的全流程闭环。零外部强依赖（`envsubst` 可选，自带 Python 降级替换），集成了 **双层网络探针（Host + Pod）** 和 **动态文件注入**，确保在高度隔离的内网环境下也能一键部署。

默认行为是生产部署（跳过 infra，使用已有 PG/TEI）；本地开发使用 `dev` 模式全量重建基础设施。

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

一次性检查，全部通过才能开始：

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

```text
.env.example
start.sh
stop.sh
healthcheck.py               # 网络探针脚本
prompts.yaml                 # Prompt 模板
manifests/
├── probe-pod.yaml           # 探针 Pod 模板
└── template/                # K8s 模板（runtime/ 由 start.sh 自动生成）
```

### 步骤 2：配置 `.env`

```bash
cp .env.example .env
vim .env
```

**⚠️ 必须修改的核心项**：

| 变量 | 说明 | 示例 |
|---|---|---|
| `LLM_API_KEY` | 大模型 API Key | `sk-xxxxxxxx` |
| `APP_IMAGE` | 业务镜像地址 | `ghcr.io/kt-chan/saleschat-amd64:latest` |
| `HF_CACHE_DIR` | 本地 HF 缓存根目录 | `$HOME/app/saleschat/models/huggingface` |
| `DATA_DIR_LOCAL` | 本地数据目录 | `$HOME/app/saleschat/data/202610` |
| `DATA_DIR_BATCH` | 容器内批次目录名 | `202610` |

**双环境网络适配（按需取消注释）**：

```dotenv
# 【场景 A：测试环境（智谱公网）】
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_MODEL=GLM-4-FlashX-250414
LLM_HOST=                       # 公网无需 Host
LLM_TIMEOUT=60

# 【场景 B：生产环境（ 内网）】
# LLM_BASE_URL=http://55.1.102.156/llm/deepseek-v4-flash-284b-a13b-w8a8/v1
# LLM_MODEL=deepseek-v4-flash-284b-a13b-w8a8
# LLM_HOST=open-llm.uat.company.cn # 必须注入 Host 头进行网关路由
# LLM_TIMEOUT=240
```

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
chmod +x start.sh stop.sh
./start.sh
```
**生产模式流程**：
1. `render`：渲染模板，自动打包 `prompts.yaml` 和 `healthcheck.py`。
2. `precheck`：**Host 层测 LLM** + **Pod 层拉起临时探针测 LLM/PG/TEI**（测完自动销毁）。
3. `upload`：只上传数据（跳过模型，由 infra 团队管理）。
4. `up`：只 apply app 资源，跳过 PG/TEI 部署（只读 `wait` 等待其就绪）。
5. `offline`：跑建库 Job。
6. `pf`：后台启动端口转发。

#### 本地 / 测试环境（全量）
```bash
./start.sh dev
```
**Dev 模式流程**：与生产模式基本一致，但会**全量部署 Infra（PG/TEI）并上传模型**。同时，预检阶段会**跳过 PG/TEI 的探针**（因为即将被重建）。

### 步骤 5：验证

```bash
# 查看三个命名空间状态
./start.sh status

# 健康检查（需要 pf 就绪）
curl -s http://127.0.0.1:8000/health/ready | jq .
# 期望：{"status":"ready","detail":{"postgres":"ok","embedding":"ok"}}

# 在线接口测试
curl -s -X POST http://127.0.0.1:8000/suggest \
  -H "Content-Type: application/json" \
  -d '{"conversation":"客户：我最近资金困难。"}' | jq .
```

---

## 命令速查

### `start.sh`（部署 / 更新 / 运行）

| 命令 | 用途 | 耗时 |
|---|---|---|
| `./start.sh` | **生产部署**（默认）：跳过 infra + 跳过模型 | 5~30 分钟 |
| `./start.sh dev` | **完整部署**：本地/测试环境（部署 infra + 上传模型） | 10~40 分钟 |
| `./start.sh online` | 只部署服务（迭代代码/配置用） | 3~60 秒 |
| `./start.sh precheck` | 仅执行网络探针预检 | 10~30 秒 |
| `./start.sh render` | 只渲染 manifests/runtime/ | 秒级 |
| `./start.sh up` | 只部署服务（配置未变不重启） | 3~60 秒 |
| `./start.sh offline` | 跑离线建库 Job | 5~30 分钟 |
| `./start.sh pf` | 起 port-forward | 秒级 |
| `./start.sh pf-stop` | 停 port-forward | 秒级 |
| `./start.sh status` | 查看部署状态 | 秒级 |

### `stop.sh`（停止 / 清理）

| 命令 | 删除内容 | 数据保留情况 |
|---|---|---|
| `./stop.sh` | cca 内 workload（app + config + secret） | ✅ PVC 保留 |
| `./stop.sh --purge` | cca 内 workload + PVC + config | ❌ cca 内数据丢失 |
| `./stop.sh --all` | 整个 cca namespace | ❌ cca 内全部删除 |

**⚠️ 安全保证**：`stop.sh` 的所有模式都**永不触碰** `gv` / `model-service` namespace 及其中的基础设施。

### 智能跳过机制

`start.sh` 内部有幂等优化，稳态下不会做多余工作：

| 操作 | 判断依据 | 已就绪时 |
|---|---|---|
| **模型上传** | `exec deploy/tei-svc -- test -f /data/models/<short>/config.json` | 不停 TEI，跳过上传 |
| **数据上传** | 远端文件数与本地数量比对 | 不停 app，跳过上传 |
| **app 重启** | ConfigMap + Secret 内容 hash 对比 | 配置未变不重启 |

---

## 常见场景

### 场景 1：改了业务配置（LLM、DB 等）
```bash
vim .env
./start.sh online
```
`online` 会检测配置变化，只有内容真变了才 `rollout restart` app。

### 场景 2：换数据批次
```bash
# 修改 .env 中的 DATA_DIR_LOCAL 和 DATA_DIR_BATCH
./start.sh upload offline
```

### 场景 3：改了业务代码，重建镜像
```bash
docker build -f docker/dockerfile -t ...
minikube image load ...
FORCE_RESTART=1 ./start.sh online
```

### 场景 4：临时停止，稍后重启
```bash
./stop.sh
./start.sh online
```

### 场景 5：只重建库
```bash
./start.sh offline
```

---

## 配置修改生效规则

| 修改项 | 生效命令 | 是否重启 app |
|---|---|---|
| `LLM_BASE_URL` / `LLM_MODEL` / `LLM_HOST` | `./start.sh online` | ✅ 自动 |
| `LLM_API_KEY` / DB 密码（Secret） | `./start.sh online` | ✅ 自动 |
| `APP_IMAGE`（镜像地址） | `FORCE_RESTART=1 ./start.sh online` | ✅ 强制 |
| `DATA_DIR_LOCAL` / `DATA_DIR_BATCH` | `./start.sh upload offline` | ✅ 上传时停/启 |

---

## 目录结构

```text
scripts/k8s/
├── .env                    # 环境变量（gitignore）
├── start.sh
├── stop.sh
├── healthcheck.py          # 独立的探针脚本（被自动打包成 ConfigMap）
├── prompts.yaml            # 自动打包挂载
├── .pf.pid                 # 脚本运行时生成（gitignore）
├── .pf.log                 # 脚本运行时生成（gitignore）
└── manifests/
    ├── probe-pod.yaml      # 动态探针 Pod 模板
    ├── template/           # K8s 资源模板（源，git 跟踪）
    │   ├── 00-namespace-app.yaml
    │   ├── ...
    │   └── upload/
    └── runtime/            # start.sh render 生成（gitignore）
```

---

## 排障

### 常见错误

| 现象 | 原因 | 修法 |
|---|---|---|
| `StatefulSet ... immutable` | 之前 sts 的 `serviceName` 不同 | `kubectl delete sts postgres -n gv`，再 `./start.sh dev` |
| `Multi-Attach error` | PVC 被其他 Pod 占用 | `start.sh upload` 会自动停 app/tei，无需手动 |
| `TEI Pod Pending` | 无 GPU 或 device plugin 未就绪 | 见「无 GPU 集群」 |
| `CreateContainerConfigError` | Secret 或 ConfigMap 缺失 | `./start.sh online` |
| `ImagePullBackOff` | 镜像不在集群里 | `minikube image load xxx` |
| `pf 30s 未就绪` | app 还没 Ready | `kubectl -n cca get pods` 查 app |

### 网络探针专项排障

| 现象 | 原因 | 修法 |
|---|---|---|
| `namespaces "cca" not found` | 预检时机过早 | 已在 `start.sh` 中修复（预检前自动建 ns） |
| `HTTP Error 401 Unauthorized` | API Key 错误或过期 | 检查 `.env` 的 `LLM_API_KEY` |
| `❌ TEI FAILED: HTTP Error 404` | TEI 没有 `/health` 接口 | 代码已兼容：404 说明端口通，视为连通 |
| `safe_json_loads 未能解析出合法 JSON` | 模型输出了 Markdown 或截断 | 确保 Prompt 包含 "JSON" 词；代码已支持剥离代码块和截断修复 |

### 诊断命令

```bash
# 三个 namespace 全览
./start.sh status

# Pod 详细事件
kubectl -n cca describe pod <pod-name> | tail -30

# app 实际 env（确认 ConfigMap 生效）
POD=$(kubectl -n cca get pod -l app=cca-app -o jsonpath='{.items[0].metadata.name}')
kubectl -n cca exec $POD -- env | grep -E "DB_|LLM_|EMBEDDING_|DATA_DIR"
```

---

## 无 GPU 集群

修改 `manifests/template/05-tei.yaml`：
1. 删除 `nvidia.com/gpu: 1` 这一行。
2. 把 image 改为 CPU 版本：`ghcr.io/huggingface/text-embeddings-inference:cpu-1.9`。
3. `./start.sh dev` 重新部署。

---

## 一句话速记

| 需求 | 命令 |
|---|---|
| **生产首次部署** | `cp .env.example .env && vim .env && ./start.sh` |
| **本地/测试部署** | `./start.sh dev` |
| **改配置** | `vim .env && ./start.sh online` |
| **改代码** | `docker build ... && minikube image load ... && FORCE_RESTART=1 ./start.sh online` |
| **换数据** | 改 `.env` 的 `DATA_DIR_LOCAL` 和 `DATA_DIR_BATCH` → `./start.sh upload offline` |
| **只重建库** | `./start.sh offline` |
| **临时停** | `./stop.sh` → `./start.sh online` |
| **清 cca 数据重建** | `./stop.sh --purge && ./start.sh` |
| **看状态** | `./start.sh status` |
| **看日志** | `kubectl -n cca logs -f deploy/cca-app` |
