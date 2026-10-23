# 催收话术智能推荐系统 (CCA)

> **Collections Conversation Assistant** — 面向催收坐席的实时话术推荐引擎。
> 基于 SOP 图谱的双分支混合检索（pgvector + Golden QA），支持 CUDA / Ascend NPU / CPU 多后端部署。

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
- [硬件后端](#硬件后端)
- [性能调优](#性能调优)
- [故障排查](#故障排查)
- [开发与测试](#开发与测试)

---

## 项目简介

CCA 是一个**离线建库 + 在线检索**的两阶段系统，用于在催收通话过程中，根据当前对话上下文为坐席推荐下一句最合适的应答话术。

- **离线阶段**：解析历史通话 → LLM 双级打标（会话级 + 话对级）→ 向量化入库 → 图谱合成"经典 Q&A"。
- **在线阶段**：解析实时对话 → 双分支混合检索（向量 + 图谱 golden QA）→ 图感知投票 → 返回最佳话术。

系统不依赖在线 LLM 推理，在线延迟稳定在 **毫秒级**（典型 20–80 ms，视检索深度与硬件而定）。

---

## 核心特性

| 特性 | 说明 |
|---|---|
| **SOP 图谱驱动** | 阶段（stage）/ 意图（intent）/ 转移（transition）显式建模，检索时按图跳数衰减，避免跨阶段语义漂移 |
| **双分支混合检索** | Branch A: pgvector 元数据过滤；Branch B: 内存 golden QA 点积；分数融合 + 平方加权投票 |
| **三种检索模式** | `off`（无约束）/ `guide`（图引导，允许跳转）/ `enforce`（锁定阶段，失败即拒绝） |
| **多后端 Embedding** | 本地多卡（CUDA / Ascend NPU）数据并行，或远程 TEI HTTP 服务；同代码切换 |
| **离线-在线语义一致** | 离线入库与在线检索共用同一 `encode(normalize_embeddings=True)` 口径；L2 归一化后点积 ≡ 余弦相似度 |
| **嵌入式数据库** | `pgembed` 启动本地 PostgreSQL + pgvector，零运维；也可指向外部集群 |
| **容错与可观测性** | 单 chunk 重试、手动 HF 推理兜底、设备令牌队列超时、详细的 `debug.trace` |

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
│                                              ▼  PgvectorWriter           │
│                                        qa_pairs (pgvector)              │
│                                              │                           │
│                                              ▼  GoldenScriptGenerator    │
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
│                         pgvector                         golden QA      │
│                         (SQL)                            (内存点积)     │
│                              └───────────────┬───────────────┘          │
│                                              ▼                           │
│                              merge (hop decay) ──► vote (stage+intent)  │
│                                              ▼                           │
│                                    best script + alternates              │
└──────────────────────────────────────────────────────────────────────────┘

```

**关键设计点**：

1. **阶段/意图约束**：pgvector 检索前先按 `(stage, primary_intent) IN (...)` 过滤，候选节点由 SOP 图 BFS 决定（`hybrid_max_hops` 控制跳数）。
2. **跳数衰减**：`score = raw_sim × weight × hop_decay[hop]`，默认 `[1.0, 0.95, 0.85]`。
3. **投票**：对 top-N 结果按 `score²` 加权投票，先定 stage 再定 intent，避免弱证据干扰。
4. **Golden QA 优先**：`golden` 命中时作为主推荐，`pgvector` 作为 alternate，兼顾精度与召回。

---

## 环境要求

| 组件 | 版本 | 说明 |
|---|---|---|
| **Python** | 3.12.x | 已验证与 PyTorch 2.10 / torch_npu 2.10 兼容 |
| **PyTorch** | `2.10.0`（CPU wheel） | `torch.accelerator` 统一设备管理 |
| **torch_npu** | `2.10.0.post4`（对应 `v2.10.0-26.0.0`） | 仅 NPU 节点安装 |
| **CANN** | `9.0.0` | 仅 NPU 节点 |
| **npu-smi** | `26.0` | 仅 NPU 节点 |
| **sentence-transformers** | `>=4.0.0` | NPU `device=` 参数支持 |
| **PostgreSQL** | 内嵌（pgembed）或 ≥15 外部实例 | 需 pgvector 扩展 |

**硬件支持**：

- 🟢 **CPU** — 无加速器时的兜底
- 🟢 **NVIDIA CUDA** — 单卡或多卡数据并行
- 🟢 **Ascend NPU** — 通过 `torch_npu` + CANN 9.0

---

## 快速开始

### 1. 克隆与依赖

```bash
git clone <repo> cca && cd cca

# 通用依赖
pip install -r requirements.txt

# NPU 节点额外安装（CUDA / CPU 节点跳过）
pip install -r requirements-npu.txt

# 环境变量
cp .env.example .env
# 编辑 .env：至少填写 LLM_API_KEY
```

NPU 环境前置：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1   # 限定可见卡（可选）
```

2. 准备数据

将历史通话数据放入 ./data/calls.jsonl，每行一个 JSON：

```jsonl
{"call_id": "C001", "text": "催收员：您好，请问是张先生吗？\n客户：是我。\n催收员：您名下信用卡本期逾期1900元，请问能处理吗？\n客户：我最近被裁员了，暂时没有收入。"}
{"call_id": "C002", "text": "..."}
```

对话文本支持 行模式（每行一条 角色:内容）或 分隔符模式（；/; 分隔），自动检测。

3. 运行离线建库

```bash
python main.py --mode offline
```

输出：

· output/session_labels.jsonl — 会话级标签
· output/pair_labels.jsonl — 话对级标签
· output/sop_graph.json — SOP 图谱 + golden QA
· qa_pairs 表（pgvector 向量库）

4. 运行在线演示

```bash
python main.py --mode online
```

演示会分别以 fast path（给定 stage_hint）和 slow path（自动推断）两条路径检索，并打印匹配结果、耗时与 debug 信息。

5. 编程接口

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

使用方式

CLI

```bash
# 离线建库
python main.py --mode offline

# 在线演示
python main.py --mode online
```

关键配置项（.env）

变量 默认值 说明
EMBEDDING_MODE local local（本地模型） / tei（远程 HTTP）
EMBEDDING_BACKEND auto auto / npu / cuda / cpu；auto 优先级 NPU→CUDA→CPU
EMBEDDING_NUM_GPUS 0 0 = 当前后端内所有可见加速器
EMBEDDING_BATCH_SIZE 32 单卡 forward batch（local）或 HTTP 请求 chunk（tei）
EMBEDDING_QUEUE_TIMEOUT 300 设备令牌获取超时（秒）
STATE_MACHINE_MODE guide off / guide / enforce
HYBRID_MAX_HOPS 2 图 BFS 最大跳数
DB_WRITE_BATCH_SIZE 500 每批写库条数
USE_EMBEDDED_DB true true 用 pgembed；false 用外部 PG
CLEAR_CACHE_ON_START false true 则清空 cache/、output/、qa_pairs

完整配置见 app/settings.py 与 .env.example。

三种检索模式

模式 行为 适用场景
off 无约束 pgvector 检索 冷启动 / 数据探索
guide 图引导，允许跨 stage 跳转，失败时回退 pass 1 生产默认
enforce 锁定 hint 或 pass-1 推断的 stage，任何失败硬拒绝 合规严格场景

---

目录结构

```
.
├── app/
│   ├── settings.py           # Pydantic Settings（.env 加载）
│   ├── logger.py             # 日志
│   ├── config.py             # SOP 阶段/意图/转移静态定义
│   ├── llm.py                # LLMClient（OpenAI 兼容）
│   ├── utils.py              # DB / Embedder / 对话解析 / 并发工具
│   ├── offline.py            # LLMLabeler / PgvectorWriter / GoldenScriptGenerator
│   └── online.py             # HybridRetriever / suggest_script*
├── data/
│   ├── calls.jsonl           # 历史通话（输入）
│   └── demo_text.txt         # 演示对话
├── output/                   # 离线产物（自动创建）
│   ├── session_labels.jsonl
│   ├── pair_labels.jsonl
│   └── sop_graph.json
├── cache/                    # 嵌入缓存 & 嵌入式 PG 数据目录
├── log/
├── prompts.yaml              # LLM prompt 模板
├── main.py                   # CLI 入口
├── requirements.txt          # 通用依赖
├── requirements-npu.txt      # NPU 节点专属依赖
└── .env.example
```

---

硬件后端

Local 模式（多加速器数据并行）

```
每设备 1 份模型副本
       │
       ▼
ThreadPoolExecutor(max_workers=N)
       │  +  device 令牌队列（每卡同一时刻至多 1 个 encode）
       ▼
长度排序切 chunk ──► 并发 encode ──► 按原索引重组 ──► 统一 L2 归一化
```

CUDA：

```dotenv
EMBEDDING_BACKEND=cuda
EMBEDDING_NUM_GPUS=0                    # 用全部可见 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3
```

Ascend NPU：

```dotenv
EMBEDDING_BACKEND=npu
EMBEDDING_NUM_GPUS=0                    # 用全部可见 NPU
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
```

TEI 模式（远程 HTTP）

```bash
# 服务端
text-embeddings-router --model-id BAAI/bge-base-zh-v1.5 --port 8080
```

```dotenv
EMBEDDING_MODE=tei
EMBEDDING_TEI_URL=http://localhost:8080
EMBEDDING_TEI_BATCH_SIZE=32
EMBEDDING_TEI_CONCURRENCY=8
```

同一离线/在线阶段必须使用同一模型，否则向量不可比。

---

性能调优

参数 影响 建议
DB_WRITE_BATCH_SIZE 单次写库条数 建议 ≥ embedding_batch_size × 加速器数 × 4
EMBEDDING_BATCH_SIZE 单卡 forward 大小 显存/内存受限时降低；GPU 上调至 64–128
EMBEDDING_SORT_BY_LENGTH padding 浪费 保持 true
HYBRID_PG_TOP_K 向量检索深度 20 → 50 提升召回，延迟 ×1.5
HYBRID_MAX_HOPS 图 BFS 半径 2 平衡召回与噪声；严格场景降至 1
EMBED_CACHE_SIZE 在线 embedding 缓存 高频相似 query 场景提高至 5000+

参考吞吐（BGE-base，4×A100，batch=32）：

场景 吞吐
单卡 CPU（16 核） ~30 q/s
单卡 A100 ~600 q/s
4×A100 数据并行 ~2200 q/s

---

故障排查

启动类

现象 排查
ImportError: torch_npu 检查 pip list \| grep torch，torch 与 torch_npu 版本必须严格匹配
npu-smi 显示设备但 torch.npu.is_available() 为 False 未 source CANN 环境或驱动不匹配，执行 source /usr/local/Ascend/ascend-toolkit/set_env.sh
pgembed 启动失败 检查 cache/local_pgdata 目录权限；首次启动可能需要 ulimit -n 65535

运行类

现象 排查
SOP 图谱未找到 ... 先执行 python main.py --mode offline
获取设备令牌超时 有 worker 未正常归还设备，检查是否有超长 encode 或进程被 SIGSTOP；临时提高 EMBEDDING_QUEUE_TIMEOUT
手动 HF 推理兜底 频繁触发 sentence-transformers 在 NPU 上的 encode() 异常；升级 ST 至最新版或固定 EMBEDDING_BACKEND=cuda
cosine(normal, manual) < 0.9999 pooling 语义不一致，检查 1_Pooling/config.json 中的 pooling_mode_*
在线结果与离线预期不符 检查离线/在线是否使用同一 embedding 模型和同一后端

调试工具

```bash
# 检查环境
python3 -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())"

# 检查 embedding 一致性（NPU）
python3 -c "
import numpy as np, torch
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('BAAI/bge-base-zh-v1.5', device='npu:0')
a = m.encode(['测试'], normalize_embeddings=True, convert_to_numpy=True)
tok = m.tokenizer
enc = {k: v.to('npu:0') for k, v in tok(['测试'], padding=True, return_tensors='pt').items()}
with torch.no_grad():
    out = m[0].auto_model(**enc)
    p = m.pooling({'token_embeddings': out.last_hidden_state, 'attention_mask': enc['attention_mask']})['sentence_embedding'].cpu().numpy()
p /= np.linalg.norm(p, axis=1, keepdims=True)
print('cosine:', float((a*p).sum()))
"

# 清空全部状态重来
CLEAR_CACHE_ON_START=true python main.py --mode offline
```

---

开发与测试

```bash
# 静态检查
ruff check app/

# 单元测试（embedder / 对话解析）
pytest tests/ -v

# 快速回归：仅验证核心 API
python -c "
from app.online import suggest_script_for_conversation
r = suggest_script_for_conversation('客户：我最近被裁员了。', stage_hint='04 方案协商与签署')
assert r['script'] and not r['reject']
print('OK:', r['latency_ms'], 'ms')
"
```

---

License

内部项目，未公开发布。使用前请联系项目维护者。
