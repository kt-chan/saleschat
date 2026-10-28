# CCA Kubernetes 部署

## 快速开始

```bash
cd scripts/k8s
chmod +x *.sh

# 1. 配置（首次）
./configure.sh --wizard

# 2. 编辑密钥
vim manifests/cca-secret.yaml       # 填 LLM_API_KEY

# 3. 部署
./install.sh                        # 生产：只部署 app
./install.sh --dev                  # 开发：全量
```

## 目录

```
scripts/k8s/
├── config.env.example    # 配置模板（提交 git）
├── config.env            # 实际配置（gitignore，首次自动创建）
├── configure.sh          # 配置向导
├── install.sh            # 部署
├── status.sh             # 状态检查
├── upload.sh             # 上传模型/数据
├── offline.sh            # 离线建库
├── uninstall.sh          # 卸载
├── lib/common.sh         # 公共函数 + 自动加载 config.env
└── manifests/            # 5 个 YAML
```

## 配置管理

### 配置文件

`config.env` 保存所有可配置项。所有脚本 `source lib/common.sh` 后自动加载。

**优先级**（高 → 低）：
```
shell 环境变量 > config.env > 脚本默认值
```

即：
```bash
# 使用 config.env 里的 NS
./install.sh

# 临时覆盖（不影响 config.env）
NS=my-app ./install.sh

# 永久修改
./configure.sh --set NS=my-app
```

### `configure.sh` 用法

```bash
# 显示当前配置
./configure.sh

# 交互式向导（推荐，首次必跑）
./configure.sh --wizard

# 用编辑器打开
./configure.sh --edit

# 修改单项
./configure.sh --set NS=my-app
./configure.sh --set IMAGE=ghcr.io/org/saleschat:v1.2.3
./configure.sh --set PF_PORT=8080

# 修改多项
./configure.sh --set NS=my-app INFRA_PG_NS=prod-pg

# 重置为模板默认
./configure.sh --reset
```

## 配置项说明

### 必改（生产环境）

| 变量 | 示例 | 说明 |
|---|---|---|
| `LLM_API_KEY` | `sk-xxxxx` | **在 `manifests/cca-secret.yaml`** |
| `DB_PASSWORD` | `cca_pass` | **在 `manifests/cca-secret.yaml`** |

### 常用（改 config.env 即可）

| 变量 | 默认 | 说明 |
|---|---|---|
| **App** | | |
| `NS` | `cca` | app namespace |
| `IMAGE` | `ghcr.io/kt-chan/saleschat-amd64:latest` | app 镜像 |
| `PF_PORT` | `8000` | port-forward 端口 |
| **App 连接的 PG** | | |
| `DB_HOST` | `postgres` | 默认走 bridge，改外部地址可直连 |
| `DB_PORT` | `5432` | |
| `DB_NAME` | `cca_db` | |
| `DB_USER` | `cca_user` | |
| **App 连接的 TEI** | | |
| `EMBEDDING_TEI_URL` | `http://tei:80/v1/embeddings` | 默认走 bridge，可改外部 |
| `EMBEDDING_MODEL` | `BAAI/bge-base-zh-v1.5` | Embedding 模型（与 TEI 侧一致） |
| `MODEL_NAME` | 派生 | 从 `EMBEDDING_MODEL` 去前缀，用于本地路径 |
| **LLM** | | |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | LLM 服务地址 |
| `LLM_MODEL` | `gpt-4o-mini` | LLM 模型名 |
| **日志** | | |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |
| **数据** | | |
| `CALLS_FILE_PATTERN` | `*.jsonl` | offline 扫描的文件 glob |
| **基础设施位置** | | |
| `INFRA_PG_NS` | `gv` | PG namespace（bridge 指向这里） |
| `INFRA_PG_SVC` | `postgres` | PG service 名 |
| `INFRA_TEI_NS` | `model-service` | TEI namespace |
| `INFRA_TEI_SVC` | `tei` | TEI service 名 |
| **本地路径** | | |
| `HF_CACHE_DIR` | `$HOME/app/saleschat/models/huggingface` | 本地 HF 缓存 |
| `DATA_DIR_LOCAL` | `$HOME/app/saleschat/data` | 本地数据 |
| `DEFAULT_BATCH` | 空 | 默认批次（空=自动选最新） |
| `PROMPTS_FILE` | `$HOME/app/saleschat/prompts.yaml` | prompts 文件 |

## 5 个脚本

| 脚本 | 何时用 |
|---|---|
| `configure.sh` | **配置**（首次 / 换环境） |
| `install.sh` | **部署**（生产 / `--dev` 开发） |
| `status.sh` | **检查**（随时） |
| `upload.sh` | **换数据/换模型** |
| `offline.sh` | **建库** |
| `uninstall.sh` | **下线** |

## 典型工作流

### 生产：首次部署

```bash
# 1. 配置（如果不在默认 namespace）
./configure.sh --wizard

# 2. 编辑密钥
vim manifests/cca-secret.yaml

# 3. 检查 infra（可选）
./status.sh

# 4. 部署
./install.sh

# 5. 验证
curl -s http://127.0.0.1:8000/health/ready
```

### 生产：升级 app

```bash
# 打新 tag 后
./configure.sh --set IMAGE=ghcr.io/org/saleschat:v1.2.3
./install.sh --skip-build
```

### 开发：完整部署

```bash
./install.sh --dev
./offline.sh
```

### 换数据

```bash
./upload.sh data data/202611
./offline.sh
```

### 换模型

```bash
./configure.sh --set MODEL_NAME=bge-large-zh-v1.5
./upload.sh model bge-large-zh-v1.5
# 编辑 manifests/infra-ms.yaml 的 --model-id
kubectl apply -f manifests/infra-ms.yaml
```

### 下线

```bash
./uninstall.sh              # 只清 app
./uninstall.sh --purge      # 清 app + PVC
./uninstall.sh --all        # 全清（含 infra）
```

## 容错

| 场景 | 行为 |
|---|---|
| `config.env` 不存在 | 首次运行自动从 example 复制 |
| 配置文件格式错误 | source 时报错并指出行号 |
| `install.sh` 生产模式 infra 未就绪 | 明确报错并给出诊断命令，不动 infra |
| 上传失败 | `trap` 自动删临时 Pod |
| 端口被占用 | 自动清理 + 提示换 `PF_PORT` |
| shell 环境变量覆盖 | **优先生效**，无需改 config.env |

## 排障

```bash
# 一切从这里开始
./configure.sh --show         # 检查配置
./status.sh                   # 检查状态
```

### 常见错误

| 错误 | 修复 |
|---|---|
| `infra 未就绪` | 生产：联系平台团队；开发：`./install.sh --dev` |
| `TEI CrashLoop` | `./upload.sh model bge-base-zh-v1.5` |
| `配置错乱` | `./configure.sh --reset` 后重新配置 |
| `namespace 不存在` | `./configure.sh --show` 检查配置 |


# Final Instruciton
cd ~/app/saleschat/scripts/k8s

# 日常停服务（cca 停，数据保留，infra 完全不动）
./undeploy.sh

# 输出示例：
# ==========================================
#   CCA 卸载（仅 cca namespace）
# ==========================================
#   操作范围   : cca
#   保护范围   : gv（PG） / model-service（TEI）— 不触碰
#   模式       : DEFAULT（停服务，保留 cca 的 PVC/config）
# ==========================================
#
# ==> 0. 确认共享基础设施状态（只读）
#     gv（PG）       : namespace 存在，postgres pod 数=1  → 保持运行
#     model-service（TEI）: namespace 存在，tei pod 数=1  → 保持运行
#
# ==> 1. 停止 port-forward
# ==> 2. 停止 cca workload / svc
# ==> 3. 保留 cca 的 PVC / ConfigMap / Secret
#
# 当前状态：
#     cca            pods=0 svc=0 pvc=2
#     gv             pods=1 svc=1 pvc=1  ← 共享 PG，未触碰
#     model-service  pods=1 svc=1 pvc=2  ← 共享 TEI，未触碰
#
#   ⚠️  gv / model-service 保持运行，不会被本脚本删除

# 之后重启只起 app（infra 还在，秒起）
./deploy.sh --skip-infra