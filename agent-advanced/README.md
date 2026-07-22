# agent-advanced

Agent 能力深入实验。涵盖长期记忆（LTM）、计划管理、对话记忆自动摘要、LTM 去重合并、分层记忆流转等高级特性。

## 特性

- **长期记忆**: JSON + 向量双存储，语义去重，LLM 合并，时间衰减检索
- **计划管理**: 多步任务分步执行，自动推进和归档
- **对话管理**: tiktoken 精确计数，自动摘要裁剪

## 运行

```bash
pip install -r requirements.txt
cp .env.example .env  # 填入 DEEPSEEK_API_KEY
python src/cli.py
```
