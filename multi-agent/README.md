# multi-agent

多 Agent 协作实验。在 rag-agent 成熟单 Agent 框架基础上，构建 Orchestrator + Worker 架构，学习任务分解、委派和并行执行。

## 架构

```
User → Orchestrator（全能副手）
          ├── 12 个基础工具（天气、搜索、计算、记忆、计划、文档检索）
          ├── delegate_task → Worker（按需创建，用完 GC）
          └── Worker 角色：
                ├── researcher（深度搜索 + 文档检索 + 记忆召回）
                └── programmer（代码 + 计算）
```

## 特性

- **角色化 Worker**：不同 system_prompt + 工具子集定制
- **按需创建**：Worker 独立 Agent 实例，用后 GC
- **并行委派**：多个 Worker 通过 ThreadPoolExecutor 并发执行
- **工具体系重构**：组件自供工具，统一注册
- **全量基础工具**：Orchestrator 包含 LTM、KB、PM 全部能力

## 环境

- Python 3.10+
- DeepSeek API Key
- ChromaDB（持久化向量数据库）
- 向量模型首次运行时自动下载

### 安装

```bash
cd multi-agent
pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，填入 API Key：

```
DEEPSEEK_API_KEY=sk-your-key-here
```

### 运行

**Orchestrator 模式**（多 Agent 协作，推荐）：

```bash
python src/cli_orch.py
```

**单 Agent 模式**：

```bash
python src/cli.py
```

### CLI 命令

| 命令 | 功能 |
|------|------|
| `/workers` | 列出可用 Worker 角色（仅 orch 模式） |
| `/reindex` | 重建知识库索引 |
| `/reindex_memories` | 重建长期记忆索引 |
| `/reindex_tools` | 重建工具索引 |

### 实验笔记

`notebooks/` 目录下包含 5 个回合的实验笔记，覆盖工具体系重构到并行委派的完整演进。
