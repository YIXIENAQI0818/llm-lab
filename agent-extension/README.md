# agent-extension

Agent 扩展框架（进行中）。从 multi-agent 实验代码提取成熟基础设施，重构为单 Agent 优先 + 可扩展工具体系。核心目标：本地工具、Skill、MCP 三类工具统一接入 ToolRegistry，Agent 只对接一个组件。

## 核心架构

```
Agent ──→ ToolRegistry（唯一对接点）
              ├── 本地工具: 纯函数 + LTM/KB/PM 绑定方法
              ├── Skill:    SKILL.md 提示词注入（独立通道，不走工具检索）
              └── MCP:      远程 MCP Server 工具
```

## 设计原则

1. **单 Agent 优先**：移除 multi-agent 的 Orchestrator/Worker
2. **三类统一**：ToolRegistry 提供 register_tool / _register_skill_tool / register_mcp
3. **Skill 独立通道**：Skills 列表注入 system prompt，唯一 `Skill` 工具作为入口
4. **扩展不改核心**：加工具 = 编辑对应模块；加 Skill = 在 skills/ 下新建 SKILL.md

## 当前进度

| 组件 | 状态 |
|------|------|
| agent_framework/ | ✅ LLM / Memory / Store / Agent 核心就位 |
| tool_registry.py | ✅ 三类注册 + 统一向量索引 |
| local_tools.py | ✅ 本地工具集中定义（纯函数 + LTM/KB/PM） |
| skill.py | ✅ 提示词注入 + skills/ 目录加载 |
| mcp_client.py | ✅ 纯标准库实现 + 三个 Server 接入 |
| cli.py | ✅ 交互式 REPL |

## 环境

```bash
cd agent-extension
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env: DEEPSEEK_API_KEY=sk-your-key-here
```

## 用法

```bash
cd agent-extension
python -m src.cli
```

命令：`/reset` `/reindex` `/stats` `/tools` `/skills` `/help` `/exit`

## 项目结构

```
agent-extension/
├── skills/                     # Skill 定义目录
│   └── deep-research/
│       └── SKILL.md            # YAML frontmatter + Markdown body
├── src/
│   ├── cli.py
│   ├── agent_framework/        # 基础框架（LLM / Memory / Store / Agent）
│   └── capabilities/
│       ├── tool_registry.py    # 工具中心（高层，类似 KB）
│       ├── tool_infra/         # 工具底层（类似 rag_infra）
│       │   ├── local_tools.py  # 本地工具集中定义
│       │   ├── skill.py        # Skill 扫描 + 加载
│       │   └── mcp_client.py   # MCP 客户端
│       ├── long_term_memory.py
│       ├── knowledge_base.py
│       ├── plan_manager.py
│       └── rag_infra/          # RAG 基础设施
```
