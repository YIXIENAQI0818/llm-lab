# multi-agent

多 Agent 协作实验。在 rag-agent 成熟单 Agent 框架基础上，构建 Orchestrator + Worker 架构，学习任务分解、角色化 Agent、并行委派与执行。所有核心机制手写实现，不依赖 LangGraph/CrewAI。

## 架构

```
User → Orchestrator（全能副手 Agent）
          ├── 13 个工具：全量基础工具（weather/search/calc/memory/plan/docs）
          │              + delegate_task
          ├── 简单任务 → 自己处理
          ├── 复杂任务 → delegate_task → Worker
          └── Worker 按需创建（researcher / programmer），用完 GC
```

## 实验路线

| 回合 | 主题 | 学习点 |
|------|------|--------|
| 01 | Agent 工厂 + 串行协作 + 工具体系重构 | 组件自供工具、ChromaDB 分 collection、模型单例化 |
| 02 | Worker 按需创建 + 并行委派 + 上下文传递 | ThreadPoolExecutor、独立上下文、Orch 中转 |
| 03 | Orch 升级为全能副手 | 全量基础工具 + delegate_task，简单自己做复杂 delegate |
| 04 | 索引导入体系统一 + CLI + Bug 修复 | build_xxx_index(force) 统一模式、consolidate 同步 |
| 05 | 收尾 + 工业调研 | Worker 通信（主从/消息/黑板）、辩论与反思评估 |

Round 03-04 评价跳过，转为工业调研报告。Round 06（预编排 Workflow）待开始。

## 核心特性

- **角色化 Worker**: 不同 system_prompt + 工具子集定制（researcher: 3, programmer: 2）
- **按需创建/GC**: 每次 delegate 新建 Agent 实例，用完 Python 自动回收
- **并行委派**: 全 delegate_task 时走 ThreadPoolExecutor 并发，Worker 崩溃不影响其他
- **工具体系重构**: 组件自供工具（get_tools()），Agent 统一 tools 参数
- **ChromaDB 分 collection**: 每个 Agent 角色使用独立 collection 存工具描述向量

## 技术栈

- **LLM**: DeepSeek API
- **Embedding**: BAAI/bge-small-zh-v1.5
- **Cross-Encoder**: BAAI/bge-reranker-v2-m3
- **向量数据库**: ChromaDB
- **分块**: tiktoken（cl100k_base）
- **并发**: concurrent.futures（ThreadPoolExecutor）

## 环境

```bash
cd multi-agent
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env: DEEPSEEK_API_KEY=sk-your-key-here
```

## 使用

**Orchestrator 模式**（多 Agent 协作）：

```bash
python cli_orch.py
```

**单 Agent 模式**：

```bash
python cli.py
```

### CLI 命令（orch 模式）

| 命令 | 功能 |
|------|------|
| `/workers` | 列出可用 Worker 角色 |
| `/reindex` | 重建知识库索引 |
| `/reindex_memories` | 重建长期记忆索引 |
| `/reindex_tools` | 重建所有 Agent（含 Worker）的工具索引 |

### CLI 命令（单 Agent 模式）

| 命令 | 功能 |
|------|------|
| `/memories` | 查看长期记忆 |
| `/forget <n>` | 删除记忆 |
| `/plan` | 查看当前计划 |
| `/consolidate` | 手动触发 LTM 合并 |
| `/history` | 查看对话历史 |

## 项目结构

```
multi-agent/
├── cli.py                     # 单 Agent REPL
├── cli_orch.py                # Orchestrator REPL
├── src/
│   ├── agent_framework/       # Agent 核心
│   ├── capabilities/          # LTM / KB / PM / ToolRegistry / RAG
│   └── orchestration/         # 多 Agent 编排层
│       ├── orchestrator.py    # Orchestrator（Agent 子类）
│       └── roles.py           # Worker 角色模板 + 工厂
├── data/                      # 知识库 Markdown 文档
├── agent_memory/              # LTM + Plan（gitignored）
└── chroma_data/               # ChromaDB（gitignored）
```
