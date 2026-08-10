# agent-extension

单 Agent + Skill + MCP 扩展框架，从 multi-agent 提取重构。本地工具、Skill、MCP 三类工具统一接入 ToolRegistry，Agent 只对接一个组件。

## 核心架构

```
Agent ──→ ToolRegistry（唯一对接点）
              │
              ├── register_tool()  — 本地工具（纯函数 + LTM/KB/PM 绑定方法）
              ├── register_skill() — Skill 入口（SKILL.md 提示词注入，独立通道）
              └── register_mcp()   — MCP 工具（子进程 JSON-RPC over stdio）
```

三类工具对 Agent 透明，LLM 不关心工具来源。Skills 列表注入 system prompt 始终全量可见，不参与工具语义检索。

## 环境

```bash
cd agent-extension
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env: DEEPSEEK_API_KEY=sk-your-key-here
```

### 额外依赖（MCP Server）

三个 MCP Server 需要外部运行时或包：

| Server | 类型 | 安装方式 |
|--------|------|---------|
| filesystem | npm (Node) | `npx` 自动下载，无需手动安装 |
| fetch | PyPI (Python) | `pip install "mcp<2" mcp-server-fetch` |
| git | PyPI (Python) | `pip install "mcp<2" mcp-server-git` |

如果某个 Server 启动失败，启动日志会打印警告并跳过，不影响其他工具。

## 用法

### 启动

```bash
cd agent-extension
python -m src.cli
```

启动过程：加载模型 → 连接 MCP Server → 扫描 Skill → 建向量索引 → 进入 REPL。

```
============================================================
  Agent Extension — 单 Agent + Skill + MCP 扩展框架
  输入 /help 查看命令，输入 /exit 退出
============================================================

正在初始化 Agent...
[MCP] secure-filesystem-server (协议 2024-11-05) 已连接
[MCP] mcp-fetch (协议 2024-11-05) 已连接
[MCP] mcp-git (协议 2024-11-05) 已连接
[OK] Agent 就绪，38 个工具已注册, 1 个 Skill

>>>
```

### 命令

| 命令 | 说明 |
|------|------|
| `/help` | 显示帮助 |
| `/tools` | 列出所有注册工具的名称 |
| `/skills` | 列出可用的 Skill（名称 + 描述） |
| `/stats` | 查看对话统计（消息数、Token 数、工具数、LTM 条数、活跃计划） |
| `/reindex` | 重建所有索引（KB 文档 + LTM 记忆 + Tools 描述） |
| `/reset` | 清空对话上下文，重新开始 |
| `/exit` 或 `/quit` | 退出 |

### 什么时候需要 `/reindex`

- 新增或修改了 `data/` 下的 Markdown 文档
- LTM 记忆向量与实际 JSON 不同步（一般不需要手动触发）
- 新增了 Skill 目录（索引会缓存工具描述）

> **注意**：新增 Skill 需要**重启 Agent**（扫描发生在启动时），新增 MCP Server 需要修改配置后重启。

---

## 工具列表

### 本地工具（7 个）

| 工具 | 来源 | 说明 |
|------|------|------|
| `calculate` | 纯函数 | 数学表达式计算 |
| `save_memory` | LTM | 保存信息到长期记忆 |
| `recall_memory` | LTM | 语义检索长期记忆 |
| `search_docs` | KB | 混合检索知识库（Dense + BM25 + Cross-Encoder） |
| `make_plan` / `check_plan` / `complete_step` / `add_plan_step` / `modify_plan_step` / `clear_plan` | PM | 多步任务计划管理 |

### MCP 工具（27 个）

| Server | 工具数 | 命名空间 | 示例 |
|--------|--------|---------|------|
| filesystem | 14 | `fs__` | `fs__read_file`, `fs__list_directory`, `fs__write_file` |
| fetch | 1 | `fetch__` | `fetch__fetch` |
| git | 12 | `git__` | `git__git_status`, `git__git_log`, `git__git_diff` |

### Skill 工具

| 工具 | 说明 |
|------|------|
| `Skill` | 唯一的 Skill 入口，调用时指定 `name` 参数加载对应 Skill 的操作指南 |

---

## Skill

### 添加 Skill

1. 在 `skills/` 下新建子目录
2. 创建 `SKILL.md`：

```markdown
---
name: my-skill
description: 当用户要求"xxx"时使用。简短描述触发场景。
---

# My Skill

## 执行步骤

1. 使用 xxx 工具 ...
2. 使用 yyy 工具 ...

## 完成标准

...输出后此 Skill 任务即告完成。
```

3. 重启 Agent（`/exit` → `python -m src.cli`）

### 工作原理

- 启动时 `scan_skills()` 扫描 `skills/` 目录，只读 frontmatter（name + description）
- Skills 列表注入 system prompt（始终可见，不走语义检索）
- LLM 决定使用某个 Skill 时调 `Skill(name="xxx")`
- 系统从磁盘读取对应 SKILL.md 的 body 作为 tool_result 返回
- LLM 读指令后自己调工具执行
- 修改 SKILL.md 后下次调用立即生效（hot-reload），**不需要重启**

### 已有 Skill

| Skill | 触发条件 |
|-------|---------|
| `deep-research` | 用户要求"深度调研"、"深入研究某个主题"、"写研究报告"时使用。多来源搜索 + 结构化报告。 |

---

## MCP

### 添加 MCP Server

编辑 `src/capabilities/tool_infra/mcp_client.py` 中的 `MCP_SERVERS` 列表：

```python
MCP_SERVERS = [
    # ...已有配置...
    {
        "command": ["python", "-m", "my_mcp_server"],  # 启动命令
        "namespace": "my__",                            # 工具名前缀（避免冲突）
        "env": {"TOKEN": "xxx"},                        # 可选：环境变量
    },
]
```

重启 Agent 即可。启动失败会被跳过，不影响其他 Server。

### 协议

- JSON-RPC 2.0 over stdio（纯 Python 标准库，无第三方依赖）
- 握手 → `tools/list` 发现工具 → `tools/call` 执行
- 每个 MCP Server 一个子进程，退出时 `close()` 清理

---

## 知识库

### 添加文档

将 Markdown 文件放入 `data/` 目录，然后：

- 首次启动自动建索引（`build_kb_index` 在 `KnowledgeBase.__init__` 中调用）
- 已有索引时，在 REPL 中执行 `/reindex` 重建

### 检索策略

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| `expand`（默认） | 查询扩展（同义词 + 近义词） | 概念性问题 |
| `decompose` | 查询拆解为独立子问题 | 多面比较 |

### 检索流程

BM25 + Dense 混合检索 → RRF 融合 → Cross-Encoder 精排 → top_k 结果

---

## 长期记忆

- 数据文件：`agent_memory/agent_memory.json`（JSON 持久化）
- 向量索引：ChromaDB collection `memories`
- 自动合并：每 10 条新增触发生成式合并去重
- 时间衰减：30 天半衰期

---

## 数据目录

```
agent-extension/
├── data/                   # 知识库 Markdown 文档
├── chroma_data/            # ChromaDB 持久化向量数据
├── agent_memory/
│   ├── agent_memory.json   # 长期记忆 JSON
│   └── plans/              # 计划归档
└── skills/                 # Skill 定义（SKILL.md）
```

---

## 项目结构

```
agent-extension/
├── skills/                     # Skill 定义目录
│   └── <skill-name>/
│       └── SKILL.md            # YAML frontmatter + Markdown body
├── data/                       # 知识库文档（.md）
├── chroma_data/                # ChromaDB 持久化
├── agent_memory/               # 长期记忆 + 计划归档
├── src/
│   ├── cli.py                  # 入口（交互式 REPL）
│   ├── agent_framework/        # 基础框架
│   │   ├── core.py             # Agent 主循环
│   │   ├── llm.py              # LLMClient
│   │   ├── memory.py           # ConversationMemory
│   │   └── chroma_store.py     # ChromaDBStore
│   └── capabilities/
│       ├── tool_registry.py    # 工具中心
│       ├── tool_infra/         # 工具底层
│       │   ├── local_tools.py  # 本地工具
│       │   ├── skill.py        # Skill 扫描 + 加载
│       │   └── mcp_client.py   # MCP 客户端
│       ├── long_term_memory.py # 长期记忆管理
│       ├── knowledge_base.py   # 知识库
│       ├── plan_manager.py     # 计划管理
│       └── rag_infra/          # RAG 基础设施
│           ├── token_chunker.py
│           ├── retriever.py
│           └── reranker.py
├── requirements.txt
├── .env.example
├── CLAUDE.md
├── README.md
└── memory/
```

## 设计原则

1. **单 Agent 优先**：移除 multi-agent 的 Orchestrator/Worker 概念
2. **三类统一**：register_tool / register_skill / register_mcp 三层对称
3. **组件自管索引**：LTM/KB 各自 `__init__` 中建索引，core.py 不插手
4. **扩展不改核心**：加本地工具 = 编辑 `local_tools.py`，加 Skill = 写 `SKILL.md`，加 MCP = 配 `MCP_SERVERS`
