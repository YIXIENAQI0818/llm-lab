# agent-advanced

Agent 能力深入实验。在 agent-basics 基础上，从零构建一个工业级单 Agent 框架：对话管理 → 长期记忆 → 计划执行 → 自我反思 → 工具发现 → 分层记忆流转。这是整个 llm-lab 中回合最多、演进最完整的子项目。

## 实验路线

**阶段一：核心能力（01-05）**

| 回合 | 主题 | 学习点 |
|------|------|--------|
| 01 | 框架骨架 | 可复用 Agent 类 + REPL + 短时记忆 |
| 02 | 长期记忆 | JSON 持久化 + Jaccard 去重 + LLM 合并 |
| 03 | 计划管理 | Plan-as-Tool：LLM 自主决定何时制定计划 |
| 04 | 自我反思 | Prompt 驱动：一行提示词改变 LLM 行为 |
| 05 | 工具发现 | 语义搜索匹配工具描述，按需筛选 |

**阶段二：上下文工程（06-07）**

| 回合 | 主题 | 学习点 |
|------|------|--------|
| 06 | Token 计数 + 上下文裁剪 | tiktoken 精确计数、超限自动移除旧消息 |
| 07 | 智能摘要 + Cache 优化 | LLM 摘要写入消息、System Prompt 静态化 |

**阶段三：记忆系统精炼（08-09）**

| 回合 | 主题 | 学习点 |
|------|------|--------|
| 08 | LTM 去重合并 + 工具化 | bi-encoder 批量合并 + LLM 判断；记忆/计划按需拉取 |
| 09 | 分层记忆流转 + 自动 Consolidate | 工作记忆 → 摘要 → LTM 自动降级 + 每 10 条触发合并 |

## 技术栈

- **LLM**: DeepSeek API
- **Embedding**: BAAI/bge-small-zh-v1.5
- **分词**: tiktoken（cl100k_base）

## 环境

```bash
cd agent-advanced
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env: DEEPSEEK_API_KEY=sk-your-key-here
```

## 使用

```bash
python cli.py
```

进入 REPL 交互界面（基于 `src/agent_framework/core.py` 的 Agent 类）。

### CLI 命令

| 命令 | 功能 |
|------|------|
| `/memories` | 查看所有长期记忆 |
| `/forget <n>` | 删除第 n 条记忆 |
| `/remember` | 显示记忆摘要 |
| `/plan` | 查看当前计划 |
| `/consolidate` | 手动触发 LTM 合并去重 |
| `/history` | 查看对话历史 |
| `/clear` | 清空对话 |
| `/exit` | 退出 |

## 项目结构

```
agent-advanced/
├── cli.py                     # REPL 入口
├── src/
│   ├── agent_framework/       # Agent 核心（core / llm / memory / tools）
│   └── capabilities/          # 能力模块（LTM / PlanManager / ToolRegistry）
├── notebooks/                 # 01-07 回合计时笔记（08-09 直接改框架源码）
├── agent_memory/              # LTM + Plan 持久化存储（gitignored）
└── chroma_data/               # 向量索引（gitignored）
```

## 实验笔记

`notebooks/` 目录下包含 7 个回合的实验笔记。Rounds 08-09 没有独立 notebook，内容直接体现在框架源码中（`src/agent_framework/core.py`、`src/capabilities/long_term_memory.py`、`src/capabilities/plan_manager.py`）。
