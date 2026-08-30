# llm-lab 整体技术报告

> 报告日期：2026-08-30 | 作者：程宣赫

---

## 目录

1. [项目概述](#1-项目概述)
2. [第一层：API 接口层](#2-第一层api-接口层-api-basics)
3. [第二层：RAG 工作层](#3-第二层rag-工作层)
4. [第三层：Agent 应用层](#4-第三层agent-应用层)
5. [第四层：多 Agent 协作层](#5-第四层多-agent-协作层)
6. [第五层：Agent 扩展层](#6-第五层agent-扩展层-agent-extension)
7. [架构演进全景图](#7-架构演进全景图)
8. [关键设计决策汇总](#8-关键设计决策汇总)

---

## 1. 项目概述

`llm-lab` 是一个大模型（LLM）学习实验项目，按**五层递进架构**组织，从底层 API 调用逐步构建到上层 Agent 扩展系统。整个项目基于 **DeepSeek API**（OpenAI 兼容协议），使用纯 Python 手写实现，不依赖 LangChain、CrewAI、AutoGen 等第三方 Agent 框架。

之所以层层递进而非一次性搭一个「完整系统」，是因为 LLM 应用里每一层的能力都建立在下一层之上：不理解 API 的 messages 结构，就理解不了 Function Calling；不理解向量检索，就理解不了 RAG；不理解单 Agent 主循环，就理解不了多 Agent 协作。逐层构建让每个阶段只聚焦一个新问题，同时复用前一层已经验证过的基础设施。全程手写而非直接引入 LangChain 这类框架，则是为了看清每一层「黑箱」内部到底发生了什么——框架替你省掉的那些细节，恰恰是这个项目想学的东西。

### 1.1 五层架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                    第五层：Agent 扩展层                            │
│   单 Agent + Skill 提示词注入 + MCP 外部工具，ToolRegistry 三层统一 │
│   agent-extension/ 子项目 (✅ 完成)                               │
├──────────────────────────────────────────────────────────────────┤
│                    第四层：多 Agent 协作层                         │
│   Orchestrator + Workers 模式，按需创建、并行执行、上下文隔离       │
│   multi-agent/ 子项目 (✅ 5/5 回合完成)                           │
├──────────────────────────────────────────────────────────────────┤
│                    第三层：Agent 应用层                            │
│   完整单 Agent 框架：记忆系统 + 规划 + 反思 + 工具发现 + RAG       │
│   agent-advanced/ + rag-agent/ 子项目                             │
├──────────────────────────────────────────────────────────────────┤
│                    第二层：RAG 工作层                              │
│   文档分块 → 向量/BM25 混合检索 → RRF 融合 → Cross-Encoder 精排    │
│   rag-basics/ + rag-agent/src/capabilities/rag_infra/             │
├──────────────────────────────────────────────────────────────────┤
│                    第一层：API 接口层                              │
│   LLM API 调用、流式输出、多轮对话、参数调优                        │
│   api-basics/ 子项目                                              │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| LLM | DeepSeek API (`deepseek-v4-flash`) | OpenAI 兼容协议，复用 `openai` SDK |
| Embedding | `BAAI/bge-small-zh-v1.5` | sentence-transformers，512 维向量 |
| Cross-Encoder | `BAAI/bge-reranker-v2-m3` | 精排模型 |
| 向量数据库 | ChromaDB (PersistentClient) | 持久化到 `chroma_data/` |
| 分词 | jieba (中文) + tiktoken (token 计数) | `cl100k_base` 编码 |
| BM25 | rank-bm25 | 稀疏检索 |
| 并发 | `concurrent.futures.ThreadPoolExecutor` | Worker 并行执行（multi-agent，第五层已移除） |
| Skill | SKILL.md（YAML frontmatter + Markdown） | 提示词注入，仿 Claude Code |
| MCP | 纯标准库 JSON-RPC 2.0 over stdio | 子进程通信，无第三方依赖 |

---

## 2. 第一层：API 接口层 (api-basics)

### 2.1 定位

这是整个项目的**地基**。目标是建立对 LLM API 交互模式的基本认知，包括：单轮调用、System Prompt、多轮对话、流式输出、参数调优。所有上层模块最终都调用同一套 API 交互模式。

### 2.2 核心实现

#### 2.2.1 客户端工厂 — `src/client.py`

```python
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 模块导入时自动加载 .env

def get_client():
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
```

**设计要点**：

- **工厂函数模式**：不在模块级别创建全局 `OpenAI` 实例，而是每次调用时新建。这避免模块导入时的副作用（如网络连接），也让调用方可以灵活控制客户端的生命周期。
- **环境变量分离**：API Key 通过 `.env` 文件和 `os.getenv()` 读取，不硬编码在代码中。`load_dotenv()` 在模块顶部执行一次，保证任何导入 `client` 模块的代码都能自动获得环境变量注入。
- **base_url 硬编码**：`https://api.deepseek.com` 写在模块内部，不暴露为参数。这遵循项目的核心约定——"底层配置不暴露为构造函数参数，模块内硬编码"。每个子项目的学习目标是确定的，不是做一个通用多提供商客户端。
- **DeepSeek + OpenAI SDK**：DeepSeek API 兼容 OpenAI 协议，可以直接复用 `openai` Python SDK。这意味着消息格式、流式协议、错误码都与 OpenAI 一致。

#### 2.2.2 流式输出 — `src/streaming.py`

```python
def print_stream(response):
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
    print()
```

**设计要点**：

- **逐 Token 打印**：`end=""` 不换行 + `flush=True` 立即刷新缓冲区，实现"打字机效果"。这与 ChatGPT 等产品的用户体验一致。
- **`flush=True` 的必要性**：Python 的 `print` 默认使用行缓冲（终端模式）或块缓冲（管道模式）。如果不设 `flush=True`，内容会在缓冲区积累，直到换行符或缓冲区满才输出，破坏流式体验。
- **空 delta 保护**：流式响应的某些 chunk 可能 `delta.content` 为 `None`（如首帧包含 `role: "assistant"`、或包含 `tool_calls` 而非 `content`），不做检查会打印 `None` 字符串。
- **循环后 `print()`**：在所有 chunk 消费完毕后输出一个换行，确保后续终端输出不在同一行。
- **单函数模块**：无类、无状态、无依赖，纯副作用函数。调用方只需 `from streaming import print_stream` 即可使用。

#### 2.2.3 流式响应的底层机制

OpenAI SDK 的 `stream=True` 返回一个生成器对象，底层通过 **SSE（Server-Sent Events）** 协议接收数据。每个 `chunk` 是一个 `ChatCompletionChunk` 对象，`delta` 字段包含本次流式传输中新增的一个或几个 token。Python 的 `for` 循环自动调用 `__next__()` 逐个消费事件。

### 2.3 实验路线

| 回合 | 内容 | 学习点 |
|------|------|--------|
| 01 | 单轮调用 | messages 结构：`[{"role": "user", "content": "..."}]` |
| 02 | System Prompt | `{"role": "system"}` 控制 LLM 行为 |
| 03 | 多轮对话 | 完整 messages 数组传递，保留对话上下文 |
| 04 | 流式输出 | SSE 协议 + `stream=True` + `print_stream()` |
| 05 | 参数调优 | temperature、top_p、max_tokens 等参数的影响 |
| 06 | Token 计量 | 跳过（后续在 agent-advanced 中用 tiktoken 实现） |

### 2.4 关键设计决策

- **04 回合引入 `src/` 架构**：前三回合每个 notebook 自包含初始化代码，到 04 时重复已达临界点。抽取 `src/client.py` 和 `src/streaming.py` 为共享模块，但**不回溯修改 01-03**——早期 notebook 保持独立可运行，作为学习过程的完整记录。
- **不使用 LangChain 等高层封装**：从第一层开始就坚持手写底层调用，为后续理解 Agent 内部机制打下基础。

---

## 3. 第二层：RAG 工作层

### 3.1 定位

RAG（Retrieval-Augmented Generation）让 LLM 能基于外部文档回答问题，解决**幻觉**和**知识过期**两个核心问题。这一层分为两个阶段：

- **rag-basics**：RAG 基础概念——Embedding、向量检索、完整 RAG 流程
- **rag-agent 的 rag_infra**：生产级 RAG 管线——混合检索、Query Rewriting、Cross-Encoder 精排

### 3.2 rag-basics：RAG 基础

#### 3.2.1 Embedding 基础

使用 **BAAI/bge-small-zh-v1.5**（BGE 模型系列），这是一个 Encoder-only 的 sentence-transformers 模型：

- **输入**：任意长度文本
- **输出**：512 维归一化向量
- **为什么选 BGE**：BGE（BAAI General Embedding）在 MTEB 中文基准上表现优秀，`bge-small` 版本在质量和速度间取得平衡
- **为什么本地部署而不调 API**：Embedding 模型相对轻量（~100MB），本地运行无延迟、无费用，且便于理解向量化原理

#### 3.2.2 向量检索

首次实现采用**余弦相似度 + 手动索引**（纯 Python 内存），不用向量数据库：

- **为什么先手动实现**：保持概念清晰。向量数据库（ChromaDB/FAISS/Milvus）是对"存向量 + 算相似度"这一核心操作的工程优化，先手写一遍才能理解它们解决了什么问题
- 计算方式：`cos(a, b) = dot(a, b) / (||a|| * ||b||)`，对归一化向量简化为 `dot(a, b)`

#### 3.2.3 RAG 完整流程

```
用户问题 → Embedding(问题) → 向量相似度检索 → 获取 Top-K 文档片段
→ 拼入 prompt：根据以下文档回答...\n[片段1]\n[片段2]\n问题：...
→ LLM 生成回答
```

### 3.3 rag-agent 的 RAG 管线（rag_infra/）

这是 RAG 能力从"理解概念"到"工程可用"的跨越。管线刻意设计成三段（TokenChunker → Retriever → Reranker），每一段在检索质量漏斗中解决一层问题：

- **分块**：决定「检索的基本单位」。文档太长，既塞不进上下文窗口，也会稀释语义密度；切成合适粒度的 chunk，才能让每个检索结果聚焦一个语义主题。
- **混合检索**：解决「单一检索方式的盲区」。语义检索（Dense）和关键词检索（BM25）各有失明区，两路并行召回 + RRF 融合，才能在召回阶段尽量不漏。
- **精排**：解决「召回多但不够准」。粗排为保召回把候选放宽到 Top-20，精度交给 Cross-Encoder 收尾。

三段对应信息检索经典的「**先召回、后精排**」（retrieve-then-rerank）两阶段范式：用便宜的 Bi-Encoder 快速从海量文档里捞出一批候选，再用昂贵的 Cross-Encoder 对少量候选做精细排序。粗排追求「宁可多、不可漏」，精排追求「宁缺毋滥」，两级各司其职，才能在检索质量与计算开销之间取得平衡——只用 Bi-Encoder 精度不够，对全库直接上 Cross-Encoder 又因为每对 (query, doc) 都要重过一遍 Transformer 而成本不可接受。

#### 3.3.1 TokenChunker — 文档分块

```python
_CHUNK_TOKENS = 256   # 每个 chunk 的 token 数
_OVERLAP_TOKENS = 64  # 窗口重叠 token 数（25%）
_MODEL = "cl100k_base"
```

**设计要点**：

- **为什么用 Token 数而非字符数**：LLM 的上下文窗口按 token 计费，用 token 分块可以精确控制送入 LLM 的内容量。中英文的字符-token 比例差异很大（英文 ~4 chars/token，中文 ~1.5 chars/token），用字符数无法精准控制。
- **为什么用 `cl100k_base`**：先明确 token 计数在这里的任务——不是给 API 账单做精确核算，而是在本地判断「对话历史是否逼近上下文上限、何时触发摘要裁剪」。这个任务是相对的：差几个百分点没关系（摘要裁剪本身留了 30% 缓冲吸收误差，见 4.4.3）。既然要的是近似，就选最省事、覆盖最全的通用编码：`cl100k_base` 是 GPT-4 的编码，10 万词表、覆盖中文在内的多语言，BPE 粒度与 DeepSeek 这类 GPT 系模型接近，对中英文混合文本的计数误差在可接受范围。反观 DeepSeek，通过 OpenAI 兼容协议调用时响应里并不回传 token 用量，要为它做精确计数就得额外引入官方 tokenizer——对「近似计数」这个目标来说纯属浪费。
- **滑动窗口 + Overlap**：`step = CHUNK_TOKENS - OVERLAP_TOKENS = 192`，每次滑动 192 token，保留 64 token 重叠。这样做的目的是防止关键信息恰好被切在 chunk 边界上导致丢失。
- **元数据丰富**：每个 chunk 带有 `source`（源文件名）、`chunk_index`（序号）、`token_start`/`token_end`，方便检索结果溯源。

滑动窗口示意：
```
文本: [token 0..1023]
Chunk 0: [0..255]
Chunk 1: [192..447]   ← 192 新 token + 64 重叠
Chunk 2: [384..639]
Chunk 3: [576..831]
Chunk 4: [768..1023]
```

#### 3.3.2 Retriever — 混合检索

**核心架构：Dense + Sparse 双路并行 + RRF 融合**。

```
                     ┌──────────────────┐
用户查询 ──┬────────→│ Query Rewriting  │ (LLM expand/decompose)
           │         └──────────────────┘
           │                  │
           │    ┌─────────────┴─────────────┐
           │    ▼                           ▼
           │ ┌──────────┐            ┌──────────┐
           │ │  Dense   │            │  BM25    │
           │ │ BGE向量  │            │ 稀疏检索  │
           │ │ ChromaDB │            │ jieba分词│
           │ └──────────┘            └──────────┘
           │    │ 粗排20                 │ 粗排20
           │    └──────────┬─────────────┘
           │               ▼
           │        ┌──────────┐
           │        │ RRF 融合  │  RRF_K = 60
           │        └──────────┘
           │               │
           │               ▼
           │        ┌──────────┐
           └───────→│ Reranker │  Cross-Encoder 精排 20→5
                    └──────────┘
```

**Dense 检索（语义匹配）**：
- 使用 ChromaDB 的向量相似度搜索（BGE embedding）
- 优势：理解语义，"汽车"能匹配到"车辆"、"轿车"
- 取粗排 Top-20

**BM25 检索（关键词匹配）**：

实现上先用 `jieba` 中文分词（中文没有空格，必须先切词），再用 `rank-bm25` 库计算。BM25（Best Matching 25）是 TF-IDF 的改进版，稀疏检索的代表。它的打分公式是：

```
score(D, Q) = Σ_{t∈Q} IDF(t) · TF_norm(t, D)

IDF(t)       = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))                       # 词的稀有度
TF_norm(t,D) = f(t,D)·(k1+1) / (f(t,D) + k1·(1 - b + b·|D|/avgdl))           # 词频 + 长度归一
```

相比 TF-IDF，它改进了三点，对应三个直观直觉：

- **IDF 逆文档频率——「越稀有的词越有区分度」**：一个词在越多文档里出现，权重越低。"的、是、一个"这类词几乎篇篇都有，提供不了区分信息；"Transformer"这种只出现在少数文档里的词，一旦命中就是强信号。注意 IDF 只跟「这个词在多少篇文档出现过」有关，与具体某篇文档无关。
- **词频饱和——「出现 10 次不该比 2 次强 5 倍」**：TF-IDF 里词频线性增长，堆砌关键词的文档会虚高。BM25 用饱和函数 `f·(k1+1)/(f+k1)`，词频越大边际贡献越小，最终趋近 `k1+1` 的上限——词出现 2 次基本就说明文档相关了，出现 20 次不会让它更相关多少。
- **文档长度归一化——「长文档不能天然占便宜」**：长文档词多、词频天然更高，会在 TF 上占优。BM25 用 `|D|/avgdl`（文档长度 / 平均长度）惩罚长文档：文档比平均长，分母变大、分数被压低。`b` 控制惩罚强度（0 = 不惩罚，1 = 完全按长度比例）。

`k1`、`b` 是两个经验常数（默认 1.5、0.75），**无需训练**。这正是 BM25 的价值：简单、稳健、零成本，是关键词匹配的默认基线。

它捕捉的是「词面匹配」——专有名词、术语、数字、代码符号，这些恰恰是语义向量（Dense）容易失明的地方，所以 BM25 和 Dense 天然互补。取粗排 Top-20。

**RRF（Reciprocal Rank Fusion）融合**：
```python
RRF_K = 60
# 对每条结果，按其在各检索器中的排名加权：
rrf_score += 1.0 / (K + rank + 1)
```
- **为什么用 RRF 而非加权求和**：Dense score（余弦相似度）和 BM25 score（词频统计）的数值范围完全不同，无法直接加权。RRF 只关注排名，天然解决了分数归一化问题。
- **K=60 的作用**：平滑参数，防止排名第一的结果主导融合分数。K 越大，排名差异的影响越小。

**Query Rewriting（查询改写）**：
- `expand` 策略：将查询扩展为同义词 + 近义词。例如"机器学习" → "机器学习 深度学习 AI 人工智能 监督学习"。适合概念性问题。
- `decompose` 策略：将复杂查询拆解为 2-3 个子问题。例如"比较 Python 和 Java 的性能和生态系统" → "Python 性能特点" + "Java 性能特点" + "Python 生态系统" + "Java 生态系统"。适合多方面比较。
- LLM 通过 strategy 参数选择策略，由用户的 search_docs 调用传入。

#### 3.3.3 Reranker — Cross-Encoder 精排

```python
# 模型：BAAI/bge-reranker-v2-m3
# 单例模式，local_files_only=True
```

**Bi-Encoder vs Cross-Encoder**：

| | Bi-Encoder (BGE Embedding) | Cross-Encoder (Reranker) |
|------|------|------|
| 工作方式 | 分别编码 query 和 doc，再算余弦相似度 | 将 (query, doc) 拼接后一起编码 |
| 交互深度 | 浅（向量点积） | 深（注意力机制全交互） |
| 速度 | 快（doc 向量可预计算） | 慢（每对都要重新编码） |
| 精度 | 较低 | 较高 |
| 用途 | 粗排（海量文档 → Top-20） | 精排（Top-20 → Top-5） |

- **为什么需要精排**：Bi-Encoder 为了速度牺牲了 query-doc 交互。Cross-Encoder 将 query 和 doc 拼接后通过完整的 Transformer 注意力机制，能捕捉到更精细的语义匹配关系。
- **为什么选 `bge-reranker-v2-m3`**：精排模型与召回阶段的 BGE embedding 同属 BAAI 的 BGE 系列，语义空间的训练方式一致，配合使用时「召回判相关、精排又判不相关」的错位更少。m3 版本本身是多语言（multi-lingual）、多粒度（multi-granularity）模型，对中文及中英混合查询的排序质量有保证，与项目的中文语料场景匹配。
- **单例模式**：Cross-Encoder 模型较大（~2GB），在模块级别创建单例 `_singleton_reranker`，多个 Agent 共享一份模型，避免重复加载和 CUDA OOM。
- **`local_files_only=True`**：避免运行时因网络波动导致模型加载失败。

### 3.4 实验路线

| 项目 | 回合 | 内容 |
|------|------|------|
| rag-basics | 01 | Embedding 基础 |
| rag-basics | 02 | 向量检索 |
| rag-basics | 03 | RAG 完整流程 |
| rag-agent | 01 | Token-based chunking |
| rag-agent | 03 | BM25 + Dense 混合检索 + RRF + Query Rewriting |
| rag-agent | 04 | Cross-Encoder 精排 |

---

## 4. 第三层：Agent 应用层

### 4.1 定位

Agent 应用层是项目的核心，构建了一个完整的**工业级单 Agent 框架**。这个 Agent 具备：

- **LLM 调用**（继承自第一层）
- **短期记忆**（对话历史管理 + token 精确计数 + 自动摘要压缩）
- **长期记忆**（JSON 持久化 + ChromaDB 向量检索 + LLM 去重合并 + 时间衰减）
- **计划管理**（Plan-as-Tool + 文件持久化 + 自动归档）
- **知识库**（RAG 管线，继承自第二层）
- **工具系统**（注册、索引、按需发现、执行）
- **自我反思**（Prompt 驱动，不加新机制）

这一层经历了 agent-basics → agent-advanced → rag-agent → multi-agent 的演进，单 Agent 框架的最终形态在 agent-extension（第五层）中沉淀下来。本节引用的代码以 agent-extension 为最新基准。

### 4.2 Agent 主循环 — `core.py`

#### 4.2.1 初始化流程

```python
class Agent:
    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 max_rounds: int = 50):
        self._max_rounds = max_rounds

        # 核心依赖
        self.llm = LLMClient()                          # ① LLM 客户端
        self.store = ChromaDBStore()                    # ② 向量存储

        # 组件
        self.ltm = LongTermMemory(self.store, llm_client=self.llm)   # ③ 长期记忆
        self.pm = PlanManager()                         # ④ 计划管理
        self.kb = KnowledgeBase(self.store, llm_client=self.llm)     # ⑤ 知识库

        # ToolRegistry — 自己负责加载三类工具（本地/Skill/MCP）
        self.tr = ToolRegistry(                         # ⑥ 工具中心
            self.store,
            components=[self.ltm, self.pm, self.kb],
            llm_client=self.llm,
        )

        # 对话记忆（Skills 列表注入 system prompt）
        self._skills = scan_skills()                    # ⑦ 扫描 Skill
        full_prompt = system_prompt
        if self._skills:
            skill_lines = "\n".join(
                f"- {s['name']}: {s['description']}" for s in self._skills
            )
            full_prompt += f"\n\n## 可用 Skills\n{skill_lines}"
        self.cm = ConversationMemory(self.llm, system_prompt=full_prompt)  # ⑧ 短期记忆
```

**相比 multi-agent 时代的三个变化**：

1. **属性名 `es` → `store`**：它不再是单纯的 embedding store，而是统一的向量存储（KB/LTM/工具索引都挂在上面），改名更贴切。
2. **Agent 不再手动收集工具**：旧版要 `builtin_tools() + ltm.get_tools() + pm.get_tools() + ...` 逐个拼装，新版直接把 `components=[ltm, pm, kb]` 丢给 ToolRegistry，由它自己加载本地工具 + Skill + MCP 三类（见 6.3）。`tools` 过滤参数和 `extra_tools` 注入随之消失——那是 multi-agent 用来给不同 Worker 配工具子集的，单 Agent 场景用不上了。
3. **索引自建下沉到组件**：LTM/KB 各自在 `__init__` 里建索引（见 4.5/4.6），Agent 不再显式调 `build_kb_index()`；ToolRegistry 也在 `__init__` 里 `build_tool_index()`。

**初始化顺序**：

1. 先创建无依赖的基础设施（LLM、ChromaDBStore）
2. 再创建依赖基础设施的组件（LTM、KB 需要 store + LLM；PM 独立）
3. ToolRegistry 汇总三类工具并自建工具索引
4. `scan_skills()` 扫出 Skill 列表，注入 system prompt
5. 最后初始化对话记忆（需要完整的 prompt 做摘要压缩）

#### 4.2.2 主循环 `chat()`

```python
def chat(self, user_input: str, verbose: bool = True) -> str:
    self.cm.add_user(user_input)          # ① 将用户输入加入对话历史

    for _ in range(self._max_rounds):     # ② 最多 50 轮迭代
        response = self.llm.chat(
            self.cm.get_messages(),
            tools=self.tr.get_definitions(  # ③ 动态工具发现
                query=user_input,
                always_include=self._compute_always_include(),
            ),
        )
        msg = response.choices[0].message

        if msg.tool_calls:                # ④ LLM 要调工具
            self.cm.add_assistant(msg)
            self._execute_tools(msg.tool_calls, verbose)
        else:                              # ⑤ LLM 直接回复
            self.cm.add_assistant(msg)
            return _sanitize(msg.content or "")

    return "达到最大轮次，停止。"
```

**关键设计决策**：

- **ReAct 循环**：`思考 → 行动 → 观察 → 再思考` 的循环模式。LLM 每次可以选择回复用户或调用工具，调用工具后结果回传，LLM 基于新信息继续决策。最多 50 轮防止无限循环。

- **动态工具发现**（Tool Discovery）：工具总数 ≤ 5（`top_k`）时全量返回；> 5 时按语义相似度筛选最相关的工具。这解决了"工具太多塞不进 prompt"的问题——只把 LLM 当前需要的工具定义发给它。

- **always_include 机制**：`_CRITICAL_TOOLS` 集合（`recall_memory`、`save_memory`、`make_plan`、`complete_step`、`search_docs`、`Skill`、`fetch__fetch` 等）始终包含在工具列表中，无论是否与当前查询相关。因为这些工具是 LLM 需要"想起来用"的，不能因为语义不匹配就被过滤掉。

- **`_sanitize()` 清洗**：`msg.content` 输出前过一遍 `_SURROGATE_RE` 清理，防止 WSL 终端中文输入产生的代理字符碎片导致 print 崩溃。

- **`max_rounds` 默认 50**：安全上限。如果 LLM 在 50 轮内没有给出最终回复（陷入工具调用循环），返回"达到最大轮次，停止。" 实际正常使用中很少超过 10 轮。

#### 4.2.3 工具执行

```python
def _execute_tools(self, tool_calls, verbose: bool):
    for tc in tool_calls:
        result = self.tr.execute(                    # 统一执行，不区分类型
            tc.function.name,
            json.loads(tc.function.arguments),
        )
        self.cm.add_tool_result(tc.id, result)       # 结果回填对话历史
        if verbose:
            print(f"🔧 [{tc.function.name}] → {_sanitize(result)}")
```

**一律串行执行**。agent-extension 是单 Agent 优先，移除了 multi-agent 时代的 `delegate_task` 并行路由（`_execute_parallel` + ThreadPoolExecutor）——那是 Worker 协作的特性，随 Orchestrator/Worker 一起归入第五层（见 5.3.4）。这里工具执行不再区分类型：本地工具、Skill、MCP 工具都通过 `tr.execute()` 统一执行，ToolRegistry 内部负责分发到正确的实现。

### 4.3 LLM 客户端 — `llm.py`

```python
class LLMClient:
    def __init__(self):
        self._client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.model = "deepseek-v4-flash"

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             thinking: bool = True):
        kwargs = dict(model=self.model, messages=messages)
        if tools:
            kwargs["tools"] = tools
        kwargs["extra_body"] = {"thinking": {"type": "enabled" if thinking else "disabled"}}
        return self._client.chat.completions.create(**kwargs)
```

**设计要点**：

- **从函数升级为类**：api-basics 中是工厂函数 `get_client()`，到了 Agent 框架变为 `LLMClient` 类。因为 Agent 有多处需要调 LLM（主循环、摘要压缩、记忆合并、consolidate），用一个稳定的实例更方便。
- **每次调用都是新实例**：与 api-basics 一样，每个 Agent 创建自己的 `LLMClient`。不做全局单例是因为 DeepSeek API 是无状态的 HTTP 调用，没有连接池的概念。
- **`thinking` 参数**：DeepSeek V4 起 `deepseek-chat` / `deepseek-reasoner` 旧别名统一指向 `deepseek-v4-flash`，思考/非思考两模式改用 `thinking` 参数切换（默认思考）。通过 `extra_body` 透传 `{"thinking": {"type": "enabled"}}`，思考模式下响应会额外返回 `reasoning_content` 字段（见 4.4）。
- **model 字段而非参数**：`self.model = "deepseek-v4-flash"` 硬编码在类内部。备选模型 `deepseek-v4-pro` 注释在文件顶部。

### 4.4 短期记忆 — `memory.py`

#### 4.4.1 消息存储

```python
class ConversationMemory:
    def __init__(self, llm_client, system_prompt: str, max_tokens: int = 100000):
        self._messages: list[dict] = []
        self.max_tokens = max_tokens
        self._llm = llm_client
        self._summary = ""   # 当前摘要文本
        self.add_system(system_prompt)
```

消息以 OpenAI 原生格式存储（思考模式下 assistant 消息还带 `reasoning_content`）：
```python
[
    {"role": "system", "content": "你是一个有用的 AI 助手..."},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": None, "reasoning_content": "...",
     "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "call_xxx", "content": "..."},
    ...
]
```

**设计要点**：

- **与 API 格式对齐**：消息格式直接对应 OpenAI Chat Completions API 的 messages 参数，无需转换，避免映射错误。
- **`reasoning_content` 回传**：V4 思考模式下 assistant 消息带 `reasoning_content`（模型的思考过程）。API 规定带 tool_calls 的回合必须把 `reasoning_content` 原样回传，否则 400，所以 `add_assistant` 会把它存进消息里。
- **`_strip_nones` 清理**：`get_messages()` 返回前去掉值为 `None` 的字段（如无思考时的 `reasoning_content`、无工具时的 `tool_calls`），避免 API 拒收 None。
- **System prompt 只设一次**：在 `__init__` 中通过 `add_system()` 设置后，不再重建或修改。所有动态内容（LTM 信息、计划状态、摘要）通过工具调用或 messages 中的 assistant 消息注入，不修改 system prompt。
- **摘要作为 assistant 消息**：裁剪后的摘要以 `{"role": "assistant", "content": "[对话摘要] ..."}` 形式插入 messages 历史中。这样对 LLM 的 prompt cache 友好——system prompt 不变，摘要作为历史的一部分自然流动。

#### 4.4.2 Token 精确计数

```python
_enc = tiktoken.get_encoding("cl100k_base")
_MSG_OVERHEAD = 4  # 每条消息的格式开销约 4 token

def _count_tokens(self, messages):
    total = 0
    for m in messages:
        total += self._MSG_OVERHEAD
        total += len(self._enc.encode(m.get("content") or ""))
        reasoning = m.get("reasoning_content") or ""
        total += len(self._enc.encode(reasoning))
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                total += len(self._enc.encode(fn.get("name", "")))
                total += len(self._enc.encode(fn.get("arguments", "")))
        if m.get("tool_call_id"):
            total += len(self._enc.encode(m["tool_call_id"]))
    return total
```

**设计要点**：

- **tiktoken 精确计数**：替代早期版本的 `len(chars) // 2` 粗略估算。`cl100k_base` 编码对中英文混合内容能做到 ±2% 的精度。
- **`MSG_OVERHEAD = 4`**：OpenAI API 在每条消息前后添加特殊 token（如 `<|im_start|>` 等），大约 3-4 个 token。这个常量是对 API 格式开销的近似补偿。
- **tool_calls 也计入**：工具调用的函数名和参数 JSON 字符串都会送到 LLM，必须计入 token 统计。
- **`reasoning_content` 也计入**：思考模式下模型的思考过程同样占 token，且会随消息回传，必须纳入计数，否则摘要裁剪的触发时机判断会失真。

#### 4.4.3 自动摘要压缩

这是整个记忆系统中**最精妙的设计**。

```python
TRIM_TARGET_RATIO = 0.7  # 裁到 max_tokens * 70%，留 30% 缓冲

def _summarize_and_trim(self):
    """超限时按完整轮次 pop，裁到 70%"""
    target = int(self.max_tokens * self.TRIM_TARGET_RATIO)

    # ① 在临时副本上模拟 pop（peek-before-pop）
    temp = list(self._messages)
    to_remove = []
    while self._count_tokens(temp) > target and len(temp) > start + 1:
        to_remove.append(temp.pop(start))
        while temp[start].get("role") != "user":
            to_remove.append(temp.pop(start))  # 补齐这轮的后续消息

    if not to_remove:
        return

    # ② LLM 摘要（用旧摘要 + 新对话拼接）
    prompt = f"已有摘要：{self._summary}\n\n新对话：\n{text}"
    response = self._llm.chat([...])

    # ③ LLM 成功才真正删除 + 插入摘要
    for _ in range(len(to_remove)):
        self._messages.pop(start)
    self._messages.insert(start, {
        "role": "assistant",
        "content": f"[对话摘要] {new_summary}",
    })
```

**关键设计决策**：

1. **Peek-Before-Pop（先模拟后删除）**：先在临时副本上模拟需要删除的消息，LLM 摘要成功后才真正修改 `self._messages`。如果 LLM 调用失败（网络异常、API 限流等），`self._messages` 保持原样，不丢数据。

2. **按完整轮次裁剪**：不按消息条数裁，而是按"user + 后续非 user 消息"作为一个完整轮次来 pop。防止把 tool result 和对应的 tool call 拆散。

3. **70% 目标缓冲**：不裁到刚好 `max_tokens`，而是 `max_tokens * 70%`。这样新消息进来时有 30% 的空间，避免马上又触发摘要。

4. **摘要合并**：如果已有旧摘要（`self._summary`），新摘要不是替换而是与旧摘要合并。这保证了长期对话中关键信息不丢失。

5. **跳过已摘要消息**：被删消息中如果包含 `[对话摘要]` 开头的旧摘要，不送入 LLM，避免重复摘要。

6. **摘要作为 assistant 消息**：不修改 system prompt，而是作为 assistant 角色插入消息历史。好处：
   - System prompt 保持静态，LLM API 的 prompt cache 可以复用
   - 摘要作为对话历史的一部分，与后续消息有自然的上下文关系

### 4.5 向量存储 — `chroma_store.py`

#### 4.5.1 架构设计

```python
class ChromaDBStore:
    def __init__(self):
        self._client = PersistentClient(path="chroma_data")
        self._ef = BGEEmbedding()
```

**为什么用 ChromaDB**：
- **持久化**：`PersistentClient` 将数据存到磁盘（`chroma_data/` 目录），进程重启后仍在。早期版本用 Python 内存字典（`EmbeddingStore`），每次启动都要重建。
- **零配置**：不需要独立服务器或 Docker，`pip install chromadb` 即可，适合学习项目。
- **自定义 Embedding**：通过 `EmbeddingFunction` 接口注入 BGE 模型，`get_or_create_collection` 自动调用。

#### 4.5.2 BGE Embedding 单例

```python
_singleton_model = None

def _get_model():
    global _singleton_model
    if _singleton_model is None:
        _singleton_model = SentenceTransformer(
            "BAAI/bge-small-zh-v1.5", local_files_only=True,
        )
    return _singleton_model
```

**为什么需要单例**：
- BGE 模型加载到 GPU 约需 500MB 显存。如果不做单例，每个 Agent 实例加载一份，3 个 Agent（Orch + researcher + programmer）就是 1.5GB，容易 CUDA OOM。
- `local_files_only=True`：禁止从 HuggingFace Hub 下载，强制使用本地缓存。避免网络波动导致加载失败。

#### 4.5.3 核心操作

**`add()` — 单条写入**：
```python
def add(self, collection, text, meta=None):
    col = self._get_or_create(collection)
    mid = meta.get("id", str(uuid.uuid4()))
    col.add(documents=[text], metadatas=[meta], ids=[mid])
```
支持通过 `meta["id"]` 指定 UUID，用于 LTM 的精准单条操作。

**`rebuild()` — 批量重建**：
```python
def rebuild(self, collection, items):
    self._client.delete_collection(collection)   # 先删（不存在也不报错）
    col = self._client.create_collection(
        collection,
        embedding_function=self._ef,
        metadata={"hnsw:space": "cosine"},       # 显式指定余弦距离
    )
    col.add(documents=texts, metadatas=metas, ids=ids)  # 批量写
```
用于 KB 和 ToolRegistry 的 `build_xxx_index(force=True)`。显式指定 `hnsw:space = cosine`，让 ChromaDB 的 HNSW 索引用余弦距离构建，与 `search()` 里 `1.0 - distance` 的换算保持一致。

**`search()` — 语义检索**：
```python
def search(self, collection, query, threshold, top_k=5):
    result = col.query(query_texts=[query], n_results=n, ...)
    for text, meta, dist in zip(...):
        score = 1.0 - float(dist)  # 余弦距离 → 相似度
        if score < threshold:
            continue
        results.append({"score": score, "text": text, "meta": meta})
```
ChromaDB 使用余弦距离（cosine distance）作为默认度量，`1.0 - distance` 转换为相似度分数（0~1，越大越相似）。threshold 过滤低分结果。

**`similarity()` / `batch_similarity()`**：两两相似度计算，用于 LTM 的去重判断。

#### 4.5.4 从 EmbeddingStore 到 ChromaDBStore 的迁移

| | EmbeddingStore（旧） | ChromaDBStore（新） |
|------|------|------|
| 存储后端 | Python dict（内存） | SQLite + Parquet（磁盘） |
| 持久化 | ❌ 重启丢失 | ✅ PersistentClient |
| 检索算法 | 全量遍历 + numpy 矩阵乘 | ChromaDB 内置 HNSW 索引 |
| UUID 支持 | 无 | ✅ `meta["id"]` |
| 单条删除 | ❌ 不支持 | ✅ `col.delete(ids=[mid])` |
| 性能 | 小数据量 OK | 大数据量 HNSW 优势 |

迁移发生在 `rag-agent` Round 02，关键改进是 LTM 从"全量重建"变为"单条增删"。旧方案每次写入都要重建整个 LTM 的 numpy 矩阵（O(n)），新方案只需 `add()` 或 `delete()` 一条（O(log n)）。

### 4.6 长期记忆 — `long_term_memory.py`

#### 4.6.1 双存储架构

```
┌─────────────────────────────────────────────────────────┐
│                    LongTermMemory                        │
│                                                         │
│  ┌─────────────────────┐    ┌─────────────────────┐     │
│  │ agent_memory.json   │    │ ChromaDB Collection  │     │
│  │ (权威数据源)         │    │ "memories"           │     │
│  │                     │    │ (向量索引)            │     │
│  │ [                    │    │                     │     │
│  │   {"id": "a1b2c3",  │◄──►│ BGE 向量 + 元数据    │     │
│  │    "content": "...",│    │                     │     │
│  │    "timestamp":".."}│    │                     │     │
│  │ ]                    │    │                     │     │
│  └─────────────────────┘    └─────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

**JSON 是权威数据源，ChromaDB 是向量索引**。这个设计很重要：
- JSON 包含完整文本 + UUID + 时间戳，可读、可手改、可备份
- ChromaDB 只存向量和元数据引用（`id` + `timestamp`），用于快速语义检索
- 合并/删除时，JSON 和 ChromaDB 同步更新

#### 4.6.2 add() — 智能写入

```python
def add(self, content: str) -> bool:
    related = self._find_related(content)   # ① 语义去重检查

    if not related:
        # ② 无重叠 → 直接追加
        m = {"id": uuid.uuid4().hex[:12], ...}
        self._memories.append(m)
        self._es.add(self._COLLECTION, content, {"id": m["id"], ...})
        self._save()
        self._maybe_consolidate()
        return True  # 直接追加

    # ③ 语义重叠 → LLM 合并
    to_merge = [self._memories[i]["content"] for i in related]
    to_merge.append(content)
    merged = self._merge_batch(to_merge)    # LLM 合并

    # ④ 删除旧记忆 + 写入合并结果
    for i in sorted(related, reverse=True):
        self._es.delete(self._COLLECTION, self._memories[i]["id"])
        self._memories.pop(i)
    for text in merged:
        # 追加新记忆（UUID 重新生成）
        ...
    return False  # 已合并
```

**去重流程**：

1. **`_find_related()`**：用 `batch_similarity()` 计算新记忆与所有已有记忆的语义相似度，阈值 0.6 以上的视为"相关"。
2. **无重叠**：直接追加到 JSON 和 ChromaDB。
3. **有重叠**：调用 LLM 合并新旧记忆。LLM 可能输出 1 条（完全重叠合并）或多条（部分重叠，拆分保留独立信息）。
4. **增删同步**：JSON 和 ChromaDB 同步更新。

#### 4.6.3 search() — 语义检索 + 时间衰减

```python
def search(self, query, top_k=3, min_score=0.3):
    results = self._es.search(collection, query, top_k=len(self._memories), threshold=0.0)
    for r in results:
        r["score"] = r["score"] * _time_decay(r["meta"]["timestamp"])
    results.sort(key=lambda x: x["score"], reverse=True)
    # 取 top_k，过滤 min_score
```

**时间衰减**：
```python
def _time_decay(timestamp):
    age_days = (now - t).total_seconds() / 86400
    return 0.5 ** (age_days / 30)  # 30 天半衰期
```

- 30 天前的记忆：分数 × 0.5
- 60 天前的记忆：分数 × 0.25
- 90 天前的记忆：分数 × 0.125

**为什么需要时间衰减**：用户的偏好和计划会随时间变化。一周前的偏好比三个月前的偏好更值得参考。时间衰减让新记忆在语义相似度相近的情况下排序更靠前。

**为什么先全量搜索再加衰减**：如果先限制 top_k 再加衰减，可能漏掉"语义相似度很高但很久远"的记忆——它们虽然旧但可能仍然关键（如用户的名字）。全量搜索保证了不遗漏。

#### 4.6.4 consolidate() — LLM 驱动全量合并

```python
CONSOLIDATE_PROMPT = (
    "你是一个记忆整理助手。以下是关于用户的一组长期记忆。\n"
    "可能出现的问题：\n"
    "1. 重复：不同时间记录了同一个事实\n"
    "2. 矛盾：后记录的信息与之前的相悖（以最新的为准）\n"
    "3. 碎片化：同一主题的多条记忆分散\n"
    "4. 过时：某些信息已不再适用\n"
    "请将这些记忆合并整理..."
)
```

**触发策略**：
- 每新增 10 条记忆自动触发（`_CONSOLIDATE_INTERVAL = 10`）
- 启动时触发一次（处理存量记忆）
- 手动 `/consolidate` 命令

**consolidate 和 merge 的区别**：
- `_merge_batch()`：针对一小组语义重叠记忆（2-5 条），局部合并
- `consolidate()`：针对全部记忆，全局整理（去重、去矛盾、碎片整合）

#### 4.6.5 演进：从 Jaccard 到 Embedding 到 bi-encoder + LLM

| 阶段 | 去重方案 | 问题 |
|------|---------|------|
| agent-advanced 02 | Jaccard 相似度（字符级） | "张三"和"张先生"相似度为 0，完全不认识 |
| agent-advanced 05 | BGE Embedding 替换 Jaccard | 语义匹配好了，但阈值难调 |
| agent-advanced 08 | bi-encoder 批量 + LLM 判断 | cross-encoder 尝试后回退（效果提升不大但显著变慢） |
| multi-agent (当前) | bi-encoder + LLM 批量合并/拆分 | 成熟方案，语义检索 + LLM 决策 |

### 4.7 计划管理 — `plan_manager.py`

#### 4.7.1 架构设计

```python
class PlanManager:
    def __init__(self):
        self._dir = Path("agent_memory/plans")
        self._file = self._dir / "active.json"      # 当前活跃计划
        self._done_dir = self._dir / "done"          # 已完成归档
        self.active: dict | None = self._load()
```

**计划结构**：
```json
{
    "task": "研究机器学习算法并写报告",
    "steps": [
        {"desc": "搜索监督学习相关资料", "status": "✓"},
        {"desc": "搜索无监督学习相关资料", "status": "→"},
        {"desc": "整理对比表格", "status": "○"}
    ],
    "current_step": 1
}
```

状态标记：`○` 未开始、`→` 进行中、`✓` 已完成。`current_step` 是**0-based 索引**指向当前进行中的步骤（新建计划时全为 `○`、`current_step = 0`），而 `complete_step(step)` 接收的是 1-based 编号（内部 `idx = step - 1` 换算），完成一步后指针推进并把下一步标为 `→`。

#### 4.7.2 关键操作

- **`create(task, steps)`**：创建新计划。如果有旧计划，自动 `_archive()` 归档到 `done/` 目录。
- **`complete_step(step)`**：标记步骤完成，自动推进 `current_step` 指针。如果所有步骤完成，自动归档。
- **`modify_step(step, desc, restart=False)`**：
  - `restart=False`：只修改描述，不改变状态
  - `restart=True`：修改描述并重置该步骤及其后续所有步骤为未完成。用于方向性大改。
- **`add_step(desc)`**：追加新步骤，支持动态扩展计划。
- **`clear()`**：归档当前计划到 `done/`，文件名取自 `task` 字段。

**归档文件名去重**：
```python
name = task[:30].replace("/", "_")
dest = self._done_dir / f"{name}.json"
n = 1
while dest.exists():
    dest = self._done_dir / f"{name}_{n}.json"
    n += 1
```

#### 4.7.3 Plan-as-Tool 模式

计划不是一个独立的系统模块，而是**通过工具暴露给 LLM**。LLM 自己决定什么时候需要列计划、什么时候需要修改计划。这个设计来自一个关键洞察：**不要替 LLM 做决定，给它做决定的工具**。

- System prompt 中说"当面对需要多步协调的复杂任务时，先调用 make_plan 制定计划"
- 但实际上 LLM 可以不遵循——它在做简单任务时自然跳过计划步骤
- 这比硬编码"超过 3 步就强制列计划"更灵活

### 4.8 工具系统 — `tool_registry.py`

#### 4.8.1 统一工具格式

所有工具遵循 OpenAI Function Calling 格式：
```python
{
    "name": "search_docs",
    "description": "在知识库中语义检索相关文档片段...",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索关键词"},
            "strategy": {"type": "string", "enum": ["expand", "decompose"]},
        },
        "required": ["query"],
    },
    "fn": self._tool_search_docs,  # Python 可调用对象
}
```

**工具来源**（agent-extension 的三层）：
| 来源 | 工具数 | 工具 |
|------|--------|------|
| 本地工具（`local_tools.py`） | 10 | `calculate` + `save_memory`/`recall_memory`（LTM）+ 6 个计划工具（PM）+ `search_docs`（KB） |
| Skill（入口工具） | 1 | `Skill`（按需加载 SKILL.md） |
| MCP（fs__ / fetch__ / git__） | 若干 | 各 Server `discover()` 发现的远程工具 |

三类工具最终都收敛成 OpenAI Function Calling 格式的一条定义，进入同一个 `_tools` 字典，Agent 不区分来源（详见 6.3）。

#### 4.8.2 工具体系的三次演进

**第一代（agent-advanced / rag-agent）——集中创建，闭包绑定**：
```python
def create_demo_tools(pm, ltm, kb):
    tools = []
    tools.append({"name": "save_memory", ..., "fn": lambda content: ltm.add(content)})
    # ... 所有工具集中定义
    return tools
```
问题：加工具要改三处——工具定义、stub、绑定。紧耦合。

**第二代（multi-agent）——组件自供工具**：
```python
all_tools = builtin_tools()
all_tools.extend(self.ltm.get_tools())
all_tools.extend(self.pm.get_tools())
all_tools.extend(self.kb.get_tools())
```
优势：组件自治，LTM 知道自己的工具需要什么功能；加能力只需在组件里加 `get_tools()` 项。但仍有局限——工具定义分散在各组件，多 Agent 场景还要靠 `tools` 参数过滤 Worker 工具子集。

**第三代（agent-extension，当前）——local_tools 集中 + 三层注册**：
```python
# ToolRegistry 内部：本地工具 + Skill + MCP 统一收口
for t in get_local_tools(components or []):   # 本地：集中在一个文件定义
    self.register_tool(**t)
skill_tool = get_skill_tool()                  # Skill：一个入口工具
if skill_tool:
    self.register_skill(skill_tool)
for mcp in get_mcp_clients():                  # MCP：discover 后逐一注册
    self.register_mcp(mcp)
```
第三代的关键变化：工具定义从「分散在各组件」收敛到「`local_tools.py` 一个文件集中定义」，组件只暴露 `_tool_*` 方法供绑定；同时把 Skill、MCP 纳入同一套注册机制。Agent 不再需要 `tools` 过滤参数——单 Agent 场景没有 Worker 工具子集的需求。

#### 4.8.3 工具向量索引

```python
def build_tool_index(self, force=False):
    items = [
        {"text": t["definition"]["function"]["description"],
         "meta": {"name": t["definition"]["function"]["name"]}}
        for t in self._tools.values()
    ]
    self._es.rebuild(self._collection, items)
```

工具的描述文本被向量化存入 ChromaDB（来源无差别，三类工具混在一个 collection）。agent-extension 单 Agent 只有一个 `tools` collection（本地 10 + Skill 1 + MCP 若干），multi-agent 时代的 `tools_orch`/`tools_researcher`/`tools_programmer` 分 collection 随 Worker 角色一起消失。

每个 Agent 只在工具数 > `top_k`（5）时才做语义搜索，否则全量返回。

### 4.9 知识库 — `knowledge_base.py`

```python
class KnowledgeBase:
    COLLECTION = "documents"

    def __init__(self, es, llm_client=None):
        self._es = es
        self._tc = TokenChunker()
        self._rt = Retriever(es, llm_client, reranker=Reranker())
        self.build_kb_index()   # 索引自建，不再由 Agent 显式调用
```

**组合模式**：KB 本身不实现检索逻辑，而是将 TokenChunker、Retriever、Reranker 组合在一起。这是"组合优于继承"的实践。**索引自建**：`__init__` 里直接 `build_kb_index()`，与 LTM 的 `build_ltm_index()` 对称，core.py 不再插手索引构建。

**build/index 模式**：
```python
def build_kb_index(self, path="data/", force=False):
    if not force and not self.is_empty():
        self._rt.build_bm25(self.COLLECTION)  # 只重建 BM25（内存中的）
        return "知识库已有数据，跳过索引"       # ChromaDB 持久化数据不丢

    docs = load_markdown_files(path)         # 读取所有 .md 文件
    for doc in docs:
        for c in self._tc.chunk(doc["content"], doc["name"]):
            all_chunks.append(...)

    self._es.rebuild(self.COLLECTION, all_chunks)  # 重建 ChromaDB
    self._rt.rebuild_bm25(self.COLLECTION, texts, metas)  # 重建 BM25
```

- 首次调用：读取文件 → 分块 → 写入 ChromaDB → 构建 BM25 索引
- 后续调用（force=False）：检测到 ChromaDB 已有数据，只重建 BM25 内存索引（BM25 存在 Python 内存中，重启后需重建）
- `force=True`：删除 ChromaDB collection 重建，用于数据文件更新后

### 4.10 本地工具 — `local_tools.py`

本地工具集中定义在 `tool_infra/local_tools.py`，核心是一个纯函数工具：

- **calculate**：`eval(expression)`，支持数学表达式

其余本地工具都是组件的绑定方法（`ltm._tool_save`、`pm._tool_make_plan`、`kb._tool_search_docs` 等），`get_local_tools(components)` 接收 `[LTM, PM, KB]` 后统一组装。

对比 multi-agent 时代的 `demo_tools.py`，agent-extension 移除了 `get_weather`、`search_web` 两个占位工具——它们只是演示 Function Calling 流程用的假数据，真实的外部信息获取已经由 MCP 的 `fetch__fetch`、`git__*` 等远程工具承接，不再需要本地硬编码字典。

---

## 5. 第四层：多 Agent 协作层

### 5.1 定位

在成熟的单 Agent 框架之上，构建多 Agent 协作系统。核心学习目标：
- Agent 角色化（不同 system_prompt + 不同工具子集）
- 任务分解与分配
- 并行执行与结果合并
- 上下文传递与隔离

### 5.2 总体架构

```
User
 │
 ▼
┌──────────────────────────────────────────────────────────┐
│              Orchestrator (全能副手 Agent)                 │
│  tools: 12 个基础工具 + delegate_task + workflow           │
│  简单自己做，复杂的 delegate，重复的走 workflow             │
│                                                          │
│  System Prompt:                                          │
│  "你有一个专业团队：researcher（研究）、programmer（编程）"│
│  "简单的事自己做，需要深度研究或编程时派给 Worker"          │
└──────────┬────────────────────────────┬─────────────────┘
           │ delegate_task              │ delegate_task
           │ (ThreadPoolExecutor 并行)   │
           ▼                            ▼
    ┌──────────────┐            ┌──────────────┐
    │  researcher  │            │  programmer  │
    │  深度搜索+检索│            │  代码+计算    │
    │  tools: 3    │            │  tools: 2    │
    └──────────────┘            └──────────────┘
```

### 5.3 Orchestrator — `orchestrator.py`

#### 5.3.1 初始化

```python
class Orchestrator:
    def __init__(self):
        self._last_user_input = ""      # 记录当前用户请求
        self._agent = Agent(
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            tools=None,                  # None = 全量基础工具
            extra_tools=self.get_extra_tools(),  # 注入 delegate_task
            tool_collection="tools_orch",
        )
```

**Orchestrator 本身也是一个 Agent**，拥有全量 13 个工具（12 个基础 + delegate_task）。这来自 Round 03 的关键升级——从"纯分发者"变为"全能副手"：

- **升级前**（Round 01）：Orch 只有 `delegate_task` 一个工具，所有事情都派给 Worker，连简单查询都要走 Worker 流程
- **升级后**（Round 03）：Orch 拥有全量工具，简单的事（查天气、读记忆、文档检索）自己做，只有需要深度研究或编程时才 delegate

#### 5.3.2 System Prompt 设计

```python
ORCHESTRATOR_SYSTEM_PROMPT = (
    "你是一个全能的 AI 副手..."
    # ↑ 基础 Agent 能力（和单 Agent 一样的 prompt）

    "你有一个专业团队可以通过 delegate_task 调派："
    "  - researcher：深度研究、多轮搜索、大量文档检索"
    "  - programmer：代码编写、复杂计算"
    # ↑ Worker 能力和分工

    "工作原则："
    "1. 简单的事自己做 —— 单次查询、日常计算、记忆读写、文档检索。"
    "2. 需要深度研究或专业编程时，派给对应 Worker。"
    "3. 多个独立的子任务一次性派发，它们会并行执行；有依赖的子任务串行。"
    "4. 汇总所有结果回复用户。"
    # ↑ 行为准则
)
```

**Prompt 工程的关键点**：
- 明确"简单/复杂"的边界：单次查询 vs 深度研究
- 明确并行条件："多个独立的子任务"一次性派发
- 明确结果处理：汇总所有结果

#### 5.3.3 delegate_task 工具

```python
def _tool_delegate(self, worker_name: str, task: str) -> str:
    worker = create_worker(worker_name)     # ① 按需创建 Worker
    ctx_task = (
        f"用户原始请求：{self._last_user_input}\n\n"
        f"你的子任务：{task}"
    )                                       # ② 拼上下文
    result = worker.chat(ctx_task, verbose=False)  # ③ 执行
    return json.dumps({
        "worker": worker_name,
        "result": result,
    })
```

**关键设计决策**：

1. **Worker 按需创建，用完 GC**：
   - Round 01 是预创建 Worker 常驻内存，从池子取。问题：同一 role 无法并行，Worker 历史积累。
   - Round 02 改为每次调用 `create_worker(role)` new 新实例，用完 Python 自动 GC。
   - 代价：每次创建新 Agent 需要初始化 LLMClient + ChromaDBStore + ToolRegistry（~200ms）。但这个开销远小于 LLM API 调用的延迟（~2-5s），可忽略。

2. **上下文传递**：
   - `_last_user_input` 记录用户原始请求
   - Worker 收到"用户原始请求 + 子任务"，有背景信息，不需要猜上下文
   - 例如：用户说"研究二分查找和快速排序，各写一份实现"，Worker 看到原始请求知道这是"各写一份实现"的一部分

3. **Worker 创建失败处理**：如果 worker_name 不在 ROLES 中，返回错误 JSON，不影响当前对话。

4. **Worker 执行异常处理**：`try/except` 包裹，异常返回错误 JSON，不影响其他并行 Worker。

#### 5.3.4 多个 delegate_task 的并行执行

当 LLM 一次性调用多个 `delegate_task`（彼此独立的子任务）时，主循环走并行路径：

```python
def _execute_parallel(self, tool_calls, verbose):
    with ThreadPoolExecutor() as pool:
        futures = {}
        for tc in tool_calls:
            args = json.loads(tc.function.arguments)
            futures[pool.submit(self.tr.execute, tc.function.name, args)] = tc

        for f in as_completed(futures):
            tc = futures[f]
            try:
                result = f.result()
            except Exception as e:
                result = json.dumps({"error": f"Worker 执行失败: {e}"})
            self.cm.add_tool_result(tc.id, result)
```

**为什么用 ThreadPoolExecutor 而非 ProcessPoolExecutor**：
- Worker 执行的主要工作是 LLM API 调用（I/O 密集型而非 CPU 密集型），线程切换开销小
- 线程间共享内存，Worker 的 Agent 实例直接用 Python 对象传递
- 无需序列化/反序列化（多进程需要 pickle）

**为什么不需要锁**：每个 Worker 是独立创建的 `Agent` 实例，有自己的 `ConversationMemory`、`LongTermMemory`、`ChromaDBStore`，不存在共享状态竞争。

**`as_completed` 而非 `wait`**：`as_completed` 先完成的先处理，用户能更快看到部分结果，体验更好。

### 5.4 Worker 角色 — `roles.py`

```python
ROLES = {
    "researcher": {
        "system_prompt": "你是一个研究员，擅长信息检索与分析...",
        "tools": ["search_docs", "search_web", "recall_memory"],
        "collection": "tools_researcher",
    },
    "programmer": {
        "system_prompt": "你是一个程序员，擅长编写代码和执行计算...",
        "tools": ["calculate", "search_web"],
        "collection": "tools_programmer",
    },
}
```

**角色设计原则**：

- **工具最小化**：researcher 只有 3 个工具，programmer 只有 2 个。Worker 专注于单一职责，不需要全量工具。
- **独立 ChromaDB collection**：每个角色用专用 collection（`tools_researcher`、`tools_programmer`），工具索引互相隔离。
- **System prompt 角色化**：researcher 被提示"优先搜索知识库和网络"，programmer 被提示"写出完整可运行的代码"。

**create_worker() 工厂**：
```python
def create_worker(role: str) -> Agent:
    config = ROLES[role]
    return Agent(
        system_prompt=config["system_prompt"],
        tools=config["tools"],
        tool_collection=config["collection"],
    )
```

Worker 本质就是一个 `Agent` 实例，只是用了不同的 system_prompt 和工具子集。这体现了框架的统一性——Orch 是 Agent，Worker 也是 Agent，没有"特殊"的 Agent 类型。

### 5.5 三条 Agent 线

| Agent | 入口 | 工具数 | 定位 |
|------|------|--------|------|
| **Full Agent** | `cli.py` | 12 | 单 Agent 模式。拥有全部基础工具，不做多 Agent 协作 |
| **Orchestrator** | `cli_orch.py` | 13 | 全能副手。简单自己做，复杂 delegate。Worker 透明 |
| **researcher** | Orch 内部创建 | 3 | 深度研究专用。search_docs + search_web + recall_memory |
| **programmer** | Orch 内部创建 | 2 | 代码编写专用。calculate + search_web |

### 5.6 执行流程示例

```
用户: "帮我研究二分查找和快速排序，各写一份 Python 实现"

Orch 思考:
  1. 这是一个多方面任务，两个子任务独立
  2. delegate_task("researcher", "研究二分查找的原理和时间复杂度")
  3. delegate_task("researcher", "研究快速排序的原理和时间复杂度")
  4. delegate_task("programmer", "写二分查找的 Python 实现")
  5. delegate_task("programmer", "写快速排序的 Python 实现")

执行（并行）:
  researcher-1 ──┐
  researcher-2 ──┤ 同时执行
  programmer-1 ──┤
  programmer-2 ──┘

Orch 汇总:
  "二分查找：O(log n)，每次将搜索范围减半... [代码]\n
   快速排序：O(n log n)，分治策略... [代码]"
```

### 5.7 CLI — 两个入口

**`cli.py`**（单 Agent 模式）：
```
🤖 Agent CLI — 输入消息开始对话
   可用工具: get_weather, search_web, calculate, save_memory, recall_memory,
            check_plan, make_plan, complete_step, add_plan_step,
            modify_plan_step, clear_plan, search_docs
```

**`cli_orch.py`**（Orchestrator 模式）：
```
🤖 Orchestrator CLI — 多 Agent 协作模式
   Workers: researcher, programmer
```

两个 CLI 的 `/` 命令支持：`/exit`、`/clear`、`/reindex`、`/reindex_memories`、`/reindex_tools`、`/help`。

---

## 6. 第五层：Agent 扩展层 (agent-extension)

### 6.1 定位

前四层完成了从 API 调用到多 Agent 协作的完整演进。agent-extension 在此基础上回答一个新问题：**Agent 的能力如何在不改核心代码的前提下扩展？**

它从 multi-agent 中提取成熟的基础设施（LLMClient、ChromaDBStore、LTM、PM、KB、rag_infra），重构为一个**单 Agent 优先**的框架，并引入两种业界主流的扩展机制：

- **Skill**：仿 Claude Code 的提示词注入模式，把"操作指南"写成 `SKILL.md` 文件，LLM 按需加载执行
- **MCP**（Model Context Protocol）：通过标准协议接入外部工具 Server，用纯 Python 标准库实现

三者（本地工具 / Skill / MCP）统一收敛到一个 `ToolRegistry`，Agent 只对接它，不关心工具来自哪里。

### 6.2 总体架构

```
Agent ──→ ToolRegistry（唯一对接点）
              │
              ├── 本地工具: 纯函数 + LTM/KB/PM 绑定方法   (register_tool)
              ├── Skill:   SKILL.md 提示词注入（独立通道）  (register_skill)
              └── MCP:     远程 MCP Server 工具（子进程）   (register_mcp)
```

**核心设计：三类工具对称注册**。无论工具是本地函数、Skill 还是远程 MCP，最终都变成 OpenAI Function Calling 格式的一条定义，进入同一个 `_tools` 字典，统一走 `get_definitions()`（语义发现）和 `execute()`（执行）。Agent 侧代码零改动。

### 6.3 ToolRegistry — 三层统一注册

```python
class ToolRegistry:
    def __init__(self, es, components, llm_client, collection="tools"):
        self._tools = {}
        # ① 本地工具
        for t in get_local_tools(components or []):
            self.register_tool(**t)
        # ② Skill（只有一个入口工具）
        skill_tool = get_skill_tool()
        if skill_tool:
            self.register_skill(skill_tool)
        # ③ MCP（每个 Server discover 后逐一注册）
        for mcp in get_mcp_clients():
            self.register_mcp(mcp)
        self.build_tool_index()
```

- `register_tool(name, description, parameters, fn)` — 本地工具
- `register_skill(skill_tool)` — 本质是 `register_tool(**skill_tool)`，复用同一套注册
- `register_mcp(mcp)` — `discover()` 发现远程工具，`_make_caller()` 包装成 callable 后 `register_tool`

**关键点**：

- MCP 工具用 namespace 前缀避免冲突（`fs__`、`fetch__`、`git__`），注册进 ToolRegistry 后与本地工具完全同构，`execute()` 不区分类型，统一 `str(fn(**args))`
- 所有工具的 `description` 统一写入 ChromaDB（来源无差别），工具数 > `top_k`（5）时语义搜索
- `_CRITICAL_TOOLS`（`recall_memory`、`search_docs`、`fetch__fetch`、`Skill` 等）始终包含，不因语义不匹配被过滤

### 6.4 Skill — 提示词注入模式

仿 Claude Code 的 Skill 机制，但实现极简：

```
skills/
└── deep-research/
    └── SKILL.md          # YAML frontmatter + Markdown body
```

SKILL.md 结构：

```markdown
---
name: deep-research
description: 当用户要求"深度调研"、"深入研究某个主题"时使用...
---

# 深度调研
## 执行步骤
1. 知识库搜索：使用 search_docs ...
2. 网络搜索：使用 fetch__fetch ...
3. 对比分析、撰写报告 ...
```

**两段式加载**：

1. **启动时** `scan_skills()` 扫描 `skills/` 目录，只解析 frontmatter（name + description），注入 system prompt 的"## 可用 Skills"段。LLM 由此知道"有哪些 Skill、何时该用"。
2. **运行时** ToolRegistry 只注册一个 `Skill` 工具（不是每 Skill 一个）。LLM 调 `Skill(name="deep-research")` → `fn` 从磁盘读 body 返回 → LLM 读指令自己执行。

**设计要点**：

- **hot-reload**：`fn` 每次从磁盘读 SKILL.md，修改后下次调用立即生效，无需重启
- **为什么不是每 Skill 一个工具**：Skill 数量可能很多，且 body 很长，全塞工具列表会撑爆 prompt。一个入口 + 按需加载，body 不进工具定义
- **提示词注入 vs 子 Agent**：早期方案是把 Skill 作为子 Agent 调用，后改为提示词注入——Skill 本质是"一段可复用的指令"，注入后 LLM 直接执行更简单，省去子 Agent 的上下文隔离开销

### 6.5 MCP — 纯标准库 JSON-RPC 客户端

`MCPClient` 不依赖任何第三方 MCP 包，用 `subprocess + json + threading` 手写 MCP over stdio：

```
启动流程:
  1. subprocess.Popen 启动 Server
  2. initialize 请求握手（协商协议版本和能力）
  3. initialized 通知（告知 Server 客户端已就绪）
  4. tools/list 请求发现工具（批量注册到 ToolRegistry）
  5. 对话中收到工具调用 → tools/call 请求 → 返回结果
```

```python
MCP_SERVERS = [
    {"command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "..."],
     "namespace": "fs__"},
    {"command": ["python", "-m", "mcp_server_fetch"], "namespace": "fetch__"},
    {"command": ["python", "-m", "mcp_server_git"], "namespace": "git__"},
]
```

**设计要点**：

- 每行一条 JSON-RPC 2.0 消息，`\n` 分隔；请求带 id，响应回传相同 id
- 串行调用 + 全局锁（`threading.Lock`），避免并发写 stdin 交错
- 错误处理：Server 启动失败只打印警告并跳过，不影响其他 Server 和本地工具
- MCP 返回格式 `{content: [{type: "text", text: ...}]}` 规范化为字符串，非标准格式直接序列化

### 6.6 与 multi-agent 的架构差异

| 维度 | multi-agent | agent-extension |
|------|-------------|-----------------|
| 核心模式 | Orchestrator + Worker 多 Agent | 单 Agent 优先 |
| 工具来源 | 本地（builtin + LTM/PM/KB） | 本地 + Skill + MCP 三层 |
| 能力扩展 | 改代码（加 role、加工具） | 写 SKILL.md / 配 MCP Server，不改核心 |
| 工具定义 | 分散在各组件 `get_tools()` | 集中在 `tool_infra/local_tools.py` + 三层注册 |
| CLI | 两个入口（cli / cli_orch） | 一个入口（cli） |

**核心洞察**：multi-agent 的复杂在于"多个 Agent 怎么协作"，agent-extension 则把焦点收回到"单个 Agent 的能力边界怎么扩展"。Worker 协作是 Agent 内部决策的延伸，而 Skill/MCP 是把外部世界的能力（知识、工具）拉进 Agent。

### 6.7 CLI — 单入口 REPL

```python
# python -m src.cli
Agent Extension — 单 Agent + Skill + MCP 扩展框架
命令: /reset /reindex /stats /tools /skills /help /exit
```

`/reindex` 一次性重建 KB + LTM + Tools 三个索引，`/skills` 列出可用 Skill，`/tools` 列出全部注册工具（本地 + Skill + MCP 混排）。退出时 `tr.close()` 清理所有 MCP 子进程。

---

## 7. 架构演进全景图

### 7.1 七个子项目的接力

```
api-basics ──→ rag-basics ──→ agent-basics ──→ agent-advanced ──→ rag-agent ──→ multi-agent ──→ agent-extension
 (API 调用)     (RAG 基础)    (Function Call)  (Agent 框架)      (RAG 增强)     (多 Agent)      (扩展框架)

第一层          第二层                          第三层                          第四层          第五层
```

每个子项目继承前一个的产出，形成能力累进：

| 子项目 | 继承自 | 新增 | 产出 |
|--------|--------|------|------|
| api-basics | 零 | API 调用、流式输出 | LLM 交互基础 |
| rag-basics | api-basics | Embedding、向量检索 | RAG 基础概念 |
| agent-basics | api-basics | Function Calling、ReAct | Agent 基础 |
| agent-advanced | agent-basics | 记忆系统、规划、反思、工具发现 | 完整单 Agent 框架 |
| rag-agent | agent-advanced | ChromaDB、混合检索、精排 | Agent RAG 增强 |
| multi-agent | rag-agent | Orchestrator+Worker、并行、角色化 | 多 Agent 协作 |
| agent-extension | multi-agent | Skill 提示词注入、MCP 客户端、三层统一注册 | Agent 扩展框架 |

### 7.2 关键架构转折点

1. **api-basics → src/ 架构**（04 回合）：共享代码抽取，结束"每个 notebook 自包含"的阶段
2. **agent-advanced → System prompt 静态化**（07 回合）：所有动态内容走工具或 messages，不再重建 system prompt
3. **agent-advanced → 摘要固化到 messages**（07 回合）：摘要作为 assistant 消息，对 cache 友好
4. **rag-agent → ChromaDB 迁移**（02 回合）：从内存 dict 到持久化向量数据库
5. **multi-agent → 工具体系重构**（01 回合）：从集中创建闭包绑定 → 组件自供工具
6. **multi-agent → Worker 按需创建**（02 回合）：从预创建常驻 → 用完 GC
7. **multi-agent → Orch 全能副手**（03 回合）：从纯分发者 → 自己也能干活
8. **agent-extension → Skill 提示词注入**：从子 Agent 方案回退为提示词注入（Skill 本质是可复用指令，注入后直接执行更简单）
9. **agent-extension → MCP 纯标准库**：不引入第三方 MCP 包，手写 JSON-RPC 2.0 over stdio

### 7.3 代码量演变

| 子项目 | 框架文件数 | 核心能力 |
|------|-----------|---------|
| api-basics | 2 | client + streaming |
| rag-basics | 2 | client + embed |
| agent-basics | 1 | client (Function Calling) |
| agent-advanced | 6 | core + llm + memory + tools + embedding_store + LTM + PM |
| rag-agent | 10+ | + chroma_store + kb + rag_infra/ (3) + tool_registry |
| multi-agent | 14+ | + orchestration/ (2) + cli_orch |
| agent-extension | 15+ | + tool_infra/ (3: local_tools + skill + mcp_client) |

---

## 8. 关键设计决策汇总

### 8.1 架构原则

| 原则 | 体现 |
|------|------|
| **渐进式构建** | 7 个子项目接力，每回合只加一个核心能力 |
| **手写核心** | 不依赖 LangChain/CrewAI/AutoGen，所有底层机制自己实现 |
| **组合优于继承** | KB 组合 Chunker+Retriever+Reranker；Agent 组合 LTM+PM+KB+ToolRegistry |
| **组件自治** | 各能力模块自建索引、暴露 `_tool_*` 方法，工具定义由 local_tools.py 集中组装 |
| **单一职责** | 每个文件/类做一件事：llm.py 只管 API，memory.py 只管对话历史，plan_manager.py 只管计划 |
| **配置硬编码** | 底层参数（模型名、base_url、chunk 大小）不暴露为构造函数参数 |
| **隔离先行** | Worker 按需创建、独立上下文、用完 GC，互不干扰 |
| **扩展不改核心** | 加 Skill = 写 SKILL.md，加 MCP = 配 Server，不改 Agent 核心代码 |

### 8.2 工程实践

| 实践 | 说明 |
|------|------|
| **Peek-Before-Pop** | 摘要裁剪先在副本模拟，成功后真正删除，LLM 失败不丢数据 |
| **单例模型** | BGE Embedding 和 CrossEncoder 全局单例，解决 CUDA OOM |
| **local_files_only** | 禁止从 HuggingFace Hub 下载，强制本地缓存 |
| **build/index 模式** | 首次构建跳过已存在的数据，`force=True` 强制重建 |
| **双存储** | JSON 权威数据源 + ChromaDB 向量索引，各司其职 |
| **时间衰减** | 30 天半衰期，新记忆优于旧记忆 |
| **Unicode 清理** | `_SURROGATE_RE` 移除代理字符，防止 WSL 中文输入导致的 print 崩溃 |

### 8.3 模型选择逻辑

| 场景 | 选型 | 理由 |
|------|------|------|
| LLM | DeepSeek (deepseek-v4-flash) | OpenAI 兼容 + 中文能力强 + 成本低（08-14 由 deepseek-chat 迁移，支持 thinking 模式） |
| Embedding | BGE-small-zh-v1.5 | 本地运行 + 中文优化 + 轻量 |
| Cross-Encoder | bge-reranker-v2-m3 | 中英混合 + 精度高 |
| 向量 DB | ChromaDB | 零配置 + 持久化 + 嵌入式 |
| Token 计数 | tiktoken cl100k_base | 精确 + 中英通用 |
| 中文分词 | jieba | 经典 + 成熟 |

### 8.4 命名体系

| 类型 | 约定 | 示例 |
|------|------|------|
| 类名 | PascalCase | `LLMClient`, `ChromaDBStore`, `ConversationMemory` |
| 文件名 | snake_case | `llm.py`, `chroma_store.py`, `memory.py` |
| 实例名 | 缩写 | `llm`, `es`, `cm`, `ltm`, `pm`, `kb`, `tr`, `tc` |
| 函数名 | snake_case | `get_client()`, `build_kb_index()`, `_find_related()` |
| 私有属性 | `_` 前缀 | `_messages`, `_llm`, `_es` |

---

## 附录：文件索引

### multi-agent/src/agent_framework/ (框架核心)

| 文件 | 类 | 职责 |
|------|-----|------|
| `llm.py` | `LLMClient` | DeepSeek API 调用封装 |
| `memory.py` | `ConversationMemory` | 对话历史 + token 计数 + 自动摘要 |
| `chroma_store.py` | `ChromaDBStore`, `BGEEmbedding` | ChromaDB 向量存储 |
| `core.py` | `Agent` | Agent 主循环 + 工具执行路由 + 并行 |

### multi-agent/src/capabilities/ (能力层)

| 文件 | 类 | 职责 |
|------|-----|------|
| `long_term_memory.py` | `LongTermMemory` | 长期记忆：JSON + ChromaDB + LLM 合并 |
| `plan_manager.py` | `PlanManager` | 计划管理：Plan-as-Tool + 文件持久化 |
| `knowledge_base.py` | `KnowledgeBase` | 知识库：RAG 管线组合 |
| `tool_registry.py` | `ToolRegistry` | 工具注册、索引、查询、执行 |
| `demo_tools.py` | (函数) | 纯工具：weather, search, calculate |

### multi-agent/src/capabilities/rag_infra/ (RAG 管线)

| 文件 | 类 | 职责 |
|------|-----|------|
| `token_chunker.py` | `TokenChunker` | tiktoken 文档分块（256 token, 64 overlap） |
| `retriever.py` | `Retriever` | Dense + BM25 混合检索 + RRF + Query Rewriting |
| `reranker.py` | `Reranker` | Cross-Encoder 精排（20→5） |

### multi-agent/src/orchestration/ (编排层)

| 文件 | 类 | 职责 |
|------|-----|------|
| `orchestrator.py` | `Orchestrator` | 多 Agent 编排主控 + delegate_task |
| `roles.py` | (ROLES + 工厂) | Worker 角色定义 + create_worker() |

### agent-extension/src/agent_framework/ (框架核心)

| 文件 | 类 | 职责 |
|------|-----|------|
| `llm.py` | `LLMClient` | DeepSeek API 调用封装（v4-flash + thinking） |
| `memory.py` | `ConversationMemory` | 对话历史 + token 计数 + 自动摘要 |
| `chroma_store.py` | `ChromaDBStore`, `BGEEmbedding` | ChromaDB 向量存储 |
| `core.py` | `Agent` | Agent 主循环 + 工具执行 |

### agent-extension/src/capabilities/ (能力层)

| 文件 | 类 | 职责 |
|------|-----|------|
| `long_term_memory.py` | `LongTermMemory` | 长期记忆：JSON + ChromaDB + LLM 合并 |
| `plan_manager.py` | `PlanManager` | 计划管理：Plan-as-Tool + 文件持久化 |
| `knowledge_base.py` | `KnowledgeBase` | 知识库：RAG 管线组合 |
| `tool_registry.py` | `ToolRegistry` | 三类工具统一注册、索引、执行 |

### agent-extension/src/capabilities/tool_infra/ (工具底层)

| 文件 | 类 | 职责 |
|------|-----|------|
| `local_tools.py` | (函数) | 本地工具集中定义（纯函数 + 组件绑定方法） |
| `skill.py` | (函数) | Skill 扫描 + SKILL.md 解析 + 入口工具 |
| `mcp_client.py` | `MCPClient` | MCP over stdio 客户端（subprocess + JSON-RPC） |

### agent-extension/src/capabilities/rag_infra/ (RAG 管线)

| 文件 | 类 | 职责 |
|------|-----|------|
| `token_chunker.py` | `TokenChunker` | tiktoken 文档分块 |
| `retriever.py` | `Retriever` | Dense + BM25 混合检索 + RRF + Query Rewriting |
| `reranker.py` | `Reranker` | Cross-Encoder 精排 |

### CLI 入口

| 文件 | 模式 | 说明 |
|------|------|------|
| `multi-agent/cli.py` | 单 Agent | 12 工具，REPL 交互 |
| `multi-agent/cli_orch.py` | Orchestrator | 13 工具 + Worker 调派，REPL 交互 |
| `agent-extension/src/cli.py` | 单 Agent + Skill + MCP | REPL 交互，`python -m src.cli` |
