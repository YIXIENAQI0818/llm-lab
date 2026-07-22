# agent-extension

Agent 扩展框架。从 multi-agent 实验中提取成熟基础设施，重构为单 Agent 优先 + 可扩展工具体系。核心目标是将本地工具、Skill、MCP 三类工具统一接入 ToolRegistry，Agent 只对接一个组件。

## 核心架构

```
Agent ──→ ToolRegistry（唯一对接点）
              ├── 本地工具: 纯函数 + LTM/KB/PM 绑定方法
              ├── Skill:    声明式复合工具（内部 LLM 循环）
              └── MCP:      远程 MCP Server 工具
```

## 特性

- **统一工具体系**：三类工具集中注册、统一向量索引、统一执行。不区分来源
- **Skill 扩展**：声明式 Skill 定义（SkillDef），内部可运行独立 LLM 循环
- **MCP 扩展**：连接 MCP Server 自动发现工具（待完成）
- **扩展不改核心**：新增工具只需编辑对应模块（local_tools.py / skill.py / mcp_client.py），Agent 和 ToolRegistry 零改动
- **工具描述语义搜索**：工具过多时自动按 query 语义筛选最相关的 top-k 工具发送给 LLM

## 环境

- Python 3.10+
- DeepSeek API Key
- ChromaDB（持久化向量数据库）
- 向量模型首次运行时自动下载

### 安装

```bash
cd agent-extension
pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，填入 API Key：

```
DEEPSEEK_API_KEY=sk-your-key-here
```

### 运行

CLI 待完成。

### 项目结构

```
agent-extension/
├── src/
│   ├── agent_framework/        # 基础框架（LLM、Memory、Store、Agent）
│   ├── capabilities/
│   │   ├── tool_registry.py    # 工具中心（三类统一注册/索引/执行）
│   │   ├── tool_infra/         # 工具底层
│   │   │   ├── local_tools.py  # 本地工具（集中定义）
│   │   │   ├── skill.py        # Skill 定义 + 执行器
│   │   │   └── mcp_client.py   # MCP 客户端（待完成）
│   │   ├── long_term_memory.py # 长期记忆
│   │   ├── knowledge_base.py   # 知识库
│   │   ├── plan_manager.py     # 计划管理
│   │   └── rag_infra/          # RAG 基础设施
│   └── cli.py                  # 入口（待完成）
```
