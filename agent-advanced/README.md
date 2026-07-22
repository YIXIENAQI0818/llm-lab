# agent-advanced

Agent 能力深入实验。在 agent-basics 基础上，为 Agent 添加长期记忆、计划管理、对话管理等高级能力。

## 特性

- **长期记忆（LTM）**：JSON + 向量双存储，语义去重，LLM 合并重复记忆，时间衰减检索
- **计划管理（PM）**：复杂任务分步执行，步骤自动推进，计划归档
- **对话管理（CM）**：tiktoken 精确计数，超 token 自动摘要裁剪，无静默丢失
- **分层记忆流转**：对话 → 短时记忆 → 长时记忆 → 知识库，自动 consolidate

## 环境

- Python 3.10+
- DeepSeek API Key
- ChromaDB（持久化向量数据库）
- 向量模型首次运行时自动下载

### 安装

```bash
cd agent-advanced
pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，填入 API Key：

```
DEEPSEEK_API_KEY=sk-your-key-here
```

### 运行

```bash
python src/cli.py
```

### REPL 命令

| 命令 | 功能 |
|------|------|
| `/memories` | 查看所有长期记忆 |
| `/forget <n>` | 删除第 n 条记忆 |
| `/plan` | 查看当前计划 |
| `/consolidate` | 手动触发记忆合并 |
| `/history` | 查看对话历史 |
| `/clear` | 清空对话 |
| `/exit` | 退出 |

### 实验笔记

`notebooks/` 目录下包含 9 个回合的实验笔记，覆盖从基础 Agent 到分层记忆流转的完整演进。
