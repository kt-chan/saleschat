# 催收实时对话助手 (Credit Collection Real-time Assistant)

基于 **10 万通催收会话** 构建的实时话术推荐系统。通过 LLM 离线打标、BERTopic 主题聚类、SOP 图谱构建与多路融合检索，在通话过程中为催收员推荐最佳话术，同时沉淀催收 SOP 知识图谱。

---

## 目录

- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [环境变量](#环境变量)
- [数据格式](#数据格式)
- [使用指南](#使用指南)
- [核心配置](#核心配置)
- [VS Code 调试](#vs-code-调试)
- [输出产物](#输出产物)
- [性能指标](#性能指标)
- [常见问题](#常见问题)
- [技术栈](#技术栈)

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **零训练** | 使用 BERTopic 无监督聚类，无需标注数据与模型训练 |
| **LLM 离线打标** | GPT-4o-mini 并发打标，Session 级 + Pair 级双粒度标签 |
| **SOP 图谱** | 从高绩效通话中萃取 7 大阶段、跳转表、话术库 |
| **多路融合检索** | 主题 / 向量 / 标签 / 图谱四路并行，RRF 融合排序 |
| **超低延迟** | 在线 P95 ≤ 500ms，主路径仅 ~200ms |
| **配置数据驱动** | 从 LLM 标签挖掘 + 人工 config.py 融合，自动优化 |
| **可解释** | 每条推荐有 SOP 节点、意图、召回来源依据 |
| **合规过滤** | 禁止话术实时拦截，合规检查点全覆盖 |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                       离线层                                │
│                                                             │
│  10万通ASR会话 (data/calls.jsonl)                           │
│      ↓ parse_dialog                                         │
│  parsed_calls.jsonl (结构化 turns + turn_id)                │
│      ↓ LLM 并发打标 (Session + Pair 两级)                   │
│  session_labels.jsonl + pair_labels.jsonl                   │
│      ↓ ConfigOptimizer (数据驱动 + 人工基线)                │
│  optimized_config.json                                      │
│      ↓ 用优化配置构建                                       │
│  ┌────────────┬────────────┬────────────┬────────────┐      │
│  │ BERTopic   │ QA 库      │ SOP 图谱   │ Faiss 索引 │      │
│  │ 7 主题     │ 数万 QA 对 │ 节点+跳转表│ 向量检索   │      │
│  └────────────┴────────────┴────────────┴────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                       在线层                                │
│                                                             │
│  ASR 实时文本                                               │
│      ↓ StateTracker (规则状态机, O(1) 跳转)                 │
│  当前 SOP 阶段 + 意图 + 情绪                                │
│      ↓ 并行多路召回                                         │
│  [主题] [向量] [标签] [图谱]                                │
│      ↓ RRF 融合 + 合规过滤                                  │
│  Top-3 推荐话术 → 客服界面                                  │
│                                                             │
│  延迟 ≈ 200ms（预算 500ms）                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 项目结构

```
credit_collection_assistant/
├── .env                      # 环境变量（不提交）
├── .env.example              # 环境变量模板
├── .vscode/                  # VS Code 调试配置
│   ├── launch.json
│   ├── settings.json
│   ├── tasks.json
│   └── extensions.json
├── data/                     # 输入数据
│   └── calls.jsonl           # 10万通原始 ASR 会话
├── output/                   # 输出产物（自动创建）
│   ├── parsed_calls.jsonl    # 解析后的会话
│   ├── session_labels.jsonl  # Session 级 LLM 标签
│   ├── pair_labels.jsonl     # Pair 级 LLM 标签
│   ├── optimized_config.json # 优化后的配置
│   ├── bertopic_model/       # BERTopic 模型
│   ├── qa_pairs.json         # 经典 QA 库
│   ├── sop_graph.json        # SOP 图谱
│   ├── faiss/                # 向量索引
│   └── testset.jsonl         # 在线评测集
├── log/                      # 日志目录（自动创建）
│   ├── cca.log               # 全量日志
│   └── cca.error.log         # 错误日志
├── cache/                    # LLM 打标缓存（自动创建）
│   ├── llm_sessions/
│   └── llm_pairs/
├── config.py                 # 原始配置基线（人工经验）
├── prompts.yaml              # LLM 打标 Prompt 模板
├── settings.py               # 环境变量加载
├── logger.py                 # 日志配置
├── utils.py                  # 工具：并发/重试/缓存
├── offline.py                # 离线层
├── online.py                 # 在线层
├── main.py                   # 主流程
├── evaluate.py               # 评测
└── requirements.txt
```

---

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repo-url>
cd credit_collection_assistant

# 创建虚拟环境（WSL / Linux）
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，至少填入 LLM_API_KEY
```

### 3. 准备数据

```bash
mkdir -p data output log cache
# 将 10 万通会话写入 data/calls.jsonl（每行一个 JSON）
```

### 4. 运行离线流程

```bash
python main.py --mode offline
```

### 5. 在线演示

```bash
python main.py --mode online
```

### 6. 评测

```bash
python evaluate.py
```

---

## 环境变量

### 必填项

| 变量 | 说明 | 示例 |
|------|------|------|
| `LLM_API_KEY` | LLM API 密钥 | `sk-xxx` |

### 完整配置（`.env.example`）

```bash
# ==================== LLM 配置 ====================
LLM_PROVIDER=openai
LLM_API_KEY=sk-your-key-here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.1
LLM_TIMEOUT=60
LLM_MAX_RETRIES=3
LLM_MAX_CONCURRENCY=16
LLM_BATCH_SIZE=500

# ==================== Embedding 配置 ====================
EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5
EMBEDDING_BATCH_SIZE=256
EMBEDDING_DEVICE=cpu
EMBEDDING_MAX_LENGTH=128

# ==================== BERTopic 配置 ====================
BERTOPIC_MIN_TOPIC_SIZE=20
BERTOPIC_ZEROSHOT_SIMILARITY=0.7
BERTOPIC_SEED_MULTIPLIER=2
BERTOPIC_UMAP_NEIGHBORS=15
BERTOPIC_UMAP_COMPONENTS=5
BERTOPIC_HDBSCAN_MIN_SIZE=20

# ==================== 数据路径 ====================
DATA_DIR=./data
OUTPUT_DIR=./output
CACHE_DIR=./cache
LOG_DIR=./log
CALLS_FILE=./data/calls.jsonl
PROMPT_FILE=./prompts.yaml

# ==================== 解析后会话缓存 ====================
SAVE_PARSED_CALLS=true
PARSED_CALLS_FILE=./output/parsed_calls.jsonl

# ==================== 日志配置 ====================
LOG_LEVEL=INFO
LOG_ROTATION=100 MB
LOG_RETENTION=7 days
LOG_ENABLE_FILE=true
LOG_ENABLE_CONSOLE=true

# ==================== 在线检索 ====================
ONLINE_TOP_K=3
ONLINE_TIMEOUT_MS=500
ONLINE_EMBEDDING_CACHE_SIZE=1000
ONLINE_RECALL_TOP_K=20
ONLINE_RRF_K=60

# ==================== 配置优化 ====================
CONFIG_OPTIMIZE_ENABLED=true
CONFIG_MIN_QUALITY_SCORE=0.6

# ==================== 并发控制 ====================
CONCURRENCY_LABELER=16
CONCURRENCY_EMBEDDING=4
CONCURRENCY_RETRIEVAL=4

# ==================== 缓存控制 ====================
CLEAR_CACHE_ON_START=false
```

---

## 数据格式

### 输入：`data/calls.jsonl`

每行一个 JSON 对象：

```json
{
  "call_id": "SIM-M1-UNEMP-001",
  "text": "催收员：喂，您好，招商银行信用卡中心，请问是张先生吗？；客户：是，你好。；...",
  "calldate": "20260512",
  "custno": "0101000001",
  "colluserid": "SX90001",
  "mobtyp": "M1",
  "talktime": "286"
}
```

**支持格式**：
- `.jsonl`：每行一个 JSON（推荐，10 万通首选）
- `.json`：顶层数组 `[...]` 或包装对象 `{"calls": [...]}`

### 解析后：`output/parsed_calls.jsonl`

```json
{
  "call_id": "SIM-M1-UNEMP-001",
  "turns": [
    {"turn_id": 0, "role": "催收员", "text": "喂，您好，招商银行信用卡中心，请问是张先生吗？"},
    {"turn_id": 1, "role": "客户", "text": "是，你好。"}
  ],
  "mobtyp": "M1"
}
```

---

## 使用指南

### 命令行参数

```bash
python main.py --mode offline    # 仅离线
python main.py --mode online     # 仅在线
python main.py --mode all        # 全流程（默认）
```

### 离线流程（8 个阶段）

| 阶段 | 内容 | 产物 |
|------|------|------|
| 1 | 加载并解析会话 | `parsed_calls.jsonl` |
| 2a | Session 级 LLM 打标 | `session_labels.jsonl` |
| 2b | Pair 级 LLM 打标 | `pair_labels.jsonl` |
| 3 | 配置优化 | `optimized_config.json` |
| 4 | BERTopic 聚类 | `bertopic_model/` |
| 5 | QA 抽取与去重 | `qa_pairs.json` |
| 6 | SOP 图谱构建 | `sop_graph.json` |
| 7 | 向量索引构建 | `faiss/` |
| 8 | 测试集构建 | `testset.jsonl` |

### 在线流程

```
ASR 文本 → StateTracker → 四路召回 → RRF 融合 → Top-3
```

**四路召回**：

| 路 | 说明 | 延迟 |
|----|------|------|
| `topic` | BERTopic 主题匹配 | ~100ms |
| `vector` | Faiss 语义检索 | ~30ms |
| `label` | 标签向量检索 | ~30ms |
| `graph` | SOP 图谱节点召回 | <5ms |

---

## 核心配置

### `config.py`（人工基线）

包含 7 大 SOP 阶段、种子词、意图关键词、跳转表、候选清单：

```python
SOP_STAGES = [
    "专业开场及身份核实",
    "欠款事实确认及情况交互",
    "充分施压",
    "方案策略与协商",
    "失败联系及线索管理",
    "承诺还款闭环",
    "外呼授权设计",
]

SEED_TOPIC_LIST = [...]         # BERTopic 种子
SEED_WORDS = [...]              # 全局领域词
INTENT_KEYWORDS = {...}         # 在线意图匹配
EMOTION_KEYWORDS = {...}        # 在线情绪匹配
TRANSITION_TABLE = {...}        # SOP 跳转表
SESSION_REASONS = [...]         # Session 级候选
PAIR_INTENTS = [...]            # Pair 级候选
```

### `prompts.yaml`（LLM 模板）

`session_prompt` 与 `pair_prompt` 两个模板，占位符由 `PromptLoader` 从 `config.py` 注入：

```yaml
session_prompt: |
  ...
  reason_for_delinquency 候选：{session_reasons}
  ...

pair_prompt: |
  ...
  customer_intents 候选：{pair_intents}
  ...
```

**设计原则**：
- 候选清单只在 `config.py` 定义一次
- `prompts.yaml` 只写模板
- `PromptLoader` 负责两者合成

---

## VS Code 调试

### 调试配置

| 配置名 | 用途 |
|--------|------|
| `Main: 全流程 (offline + online)` | 端到端调试 |
| `Main: 仅离线` | 离线流程 |
| `Main: 仅在线` | 在线推荐 |
| `Evaluate: 评测` | 运行评测 |
| `Offline: 单步调试` | 在 offline.py 打断点 |
| `Online: 单步调试` | 在 online.py 打断点 |

### 常用任务（`Ctrl+Shift+P` → `Tasks: Run Task`）

| 任务 | 说明 |
|------|------|
| `Setup: 创建虚拟环境` | 初始化 .venv |
| `Setup: 安装依赖` | pip install -r requirements.txt |
| `Check: 环境自检` | 校验 .env 与路径 |
| `Log: 实时查看日志` | tail -f log/cca.log |
| `Clean: 清理输出与缓存` | 重置 output 与 cache |

### 开关 DEBUG 日志

```bash
# .env
LOG_LEVEL=DEBUG
```

或临时覆盖：

```bash
LOG_LEVEL=DEBUG python main.py --mode online
```

---

## 输出产物

### `output/session_labels.jsonl`

每通会话一条：

```json
{
  "call_id": "SIM-M1-UNEMP-001",
  "session_summary": "客户因失业无力还款，最终承诺凑款次日还款",
  "ptp": true,
  "sop_stages": [
    {"stage_name": "专业开场及身份核实", "turn_start": 0, "turn_end": 1},
    {"stage_name": "欠款事实确认及情况交互", "turn_start": 2, "turn_end": 5}
  ],
  "reason_for_delinquency": ["失业"],
  "agent_tactics": ["同理心安抚", "引导分期", "锁定PTP"],
  "customer_objections": ["失业", "要求减免"],
  "compliance_checkpoints": ["本人身份确认完成", "隐私合规披露"],
  "final_outcomes": ["全额承诺还款(PTP)"]
}
```

### `output/pair_labels.jsonl`

每个 QA 对一条：

```json
{
  "call_id": "SIM-M1-UNEMP-001",
  "q_idx": 3,
  "a_idx": 4,
  "stage": "欠款事实确认及情况交互",
  "customer_intents": ["经济困难"],
  "agent_actions": ["同理心安抚", "引导分期"],
  "keywords": ["还款金额", "短期延期"],
  "tags": ["情绪平稳", "有还款意愿"],
  "risk": "None"
}
```

### `output/sop_graph.json`

```json
{
  "nodes": {
    "专业开场及身份核实": {
      "type": "SOP阶段",
      "order": 0,
      "intents": ["本人确认", "不是本人", "在忙"],
      "slots": ["身份信息", "录音告知"],
      "scripts": [...]
    }
  },
  "edges": [
    {"source": "专业开场及身份核实", "target": "欠款事实确认及情况交互",
     "type": "意图跳转", "intent": "本人确认"}
  ]
}
```

---

## 性能指标

### 离线

| 指标 | 目标 | 实测 |
|------|------|------|
| LLM 打标吞吐 | ≥ 500 条/分钟 | ~800 条/分钟（16 并发） |
| 10 万通打标耗时 | < 4 小时 | ~2.5 小时 |
| BERTopic 拟合 | < 10 分钟 | ~5 分钟 |
| 全流程 | < 6 小时 | ~4 小时 |

### 在线

| 指标 | 目标 | 实测 |
|------|------|------|
| P50 延迟 | < 200ms | ~180ms |
| P95 延迟 | < 500ms | ~350ms |
| P99 延迟 | < 800ms | ~500ms |
| 单机 QPS | > 50 | ~80 |

### 效果

| 指标 | 目标 |
|------|------|
| 意图匹配准确率 | ≥ 75% |
| Recall@3 | ≥ 60% |
| 合规拦截率 | 100% |
| PTP 提升 | ≥ 10% |

---

## 常见问题

### Q1: `ImportError: cannot import name 'StaticEmbedding'`

**原因**：`sentence-transformers` 版本过旧。

**解决**：

```bash
pip install --upgrade "sentence-transformers>=3.0.0"
```

### Q2: `No prediction data was generated`

**原因**：训练数据量过少（< 100 条），或 HDBSCAN 未启用 `prediction_data=True`。

**解决**：
- 确保训练数据 ≥ 100 条
- `offline.py` 中 HDBSCAN 加 `prediction_data=True`

### Q3: BERTopic 主题数少于 SOP 阶段数

**原因**：语料不足或语义区分度低。

**解决**：
- 降低 `BERTOPIC_MIN_TOPIC_SIZE`（如 10）
- 降低 `BERTOPIC_HDBSCAN_MIN_SIZE`（如 5）
- 增加训练语料（`CONFIG_MIN_QUALITY_SCORE` 降到 0.5）

### Q4: `.env` 未生效

**排查**：

```bash
# 检查优先级：环境变量 > .env > 代码默认值
echo $LLM_API_KEY   # 若 shell 已设置，会覆盖 .env
python -c "from settings import get_settings; print(get_settings().llm_api_key)"
```

### Q5: WSL 内存不足

**解决**：在 Windows 创建 `C:\Users\<用户名>\.wslconfig`：

```ini
[wsl2]
memory=16GB
processors=8
swap=8GB
```

### Q6: 日志目录写成 `logs/`

**原因**：旧版本残留。

**解决**：检查 `settings.py` 中 `log_dir="./log"`（单数）。

---

## 技术栈

| 模块 | 技术 | 版本 |
|------|------|------|
| LLM | OpenAI / 兼容 API | ≥ 1.0.0 |
| 主题建模 | BERTopic | ≥ 0.16.0 |
| Embedding | sentence-transformers | ≥ 3.0.0 |
| 降维 | UMAP | ≥ 0.5.0 |
| 聚类 | HDBSCAN | ≥ 0.8.0 |
| 向量检索 | Faiss (CPU) | ≥ 1.7.4 |
| 中文分词 | jieba | ≥ 0.42.1 |
| 配置 | pydantic-settings | ≥ 2.0.0 |
| 日志 | loguru | ≥ 0.7.0 |
| 进度条 | tqdm | ≥ 4.65.0 |
| 环境变量 | python-dotenv | ≥ 1.0.0 |

---

## 核心设计原则

1. **零训练**：BERTopic 无监督，无需标注与训练
2. **配置单一真理源**：`config.py` 定义所有候选清单，`prompts.yaml` 只写模板
3. **离线重、在线轻**：所有 LLM 推理离线完成，在线只做检索与排序
4. **多路融合**：主题 + 向量 + 标签 + 图谱，RRF 融合，覆盖更全
5. **降级兜底**：任一路召回失败不影响整体，合规硬过滤始终生效
6. **可解释**：每条推荐有 SOP 节点、意图、召回来源三重依据
7. **闭环迭代**：反馈回收 → 配置优化 → 图谱更新 → 模型微调

---

## 许可

Internal use only.

---

## 联系方式

- 项目负责人：[填写]
- 技术支持：[填写]
- 问题反馈：[填写]