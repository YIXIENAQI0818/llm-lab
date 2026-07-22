# agent-basics

Agent 与 Function Calling 基础实验。学习 LLM 工具调用机制：如何注册工具、LLM 如何选择工具、如何执行工具调用并返回结果。

## 内容

- OpenAI Function Calling 协议
- 工具注册与描述编写
- 工具调用执行循环（LLM 决策 → 工具执行 → 结果返回 → 继续推理）
- 多工具组合调用

## 环境

- Python 3.10+
- DeepSeek API Key

### 安装

```bash
cd agent-basics
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

进入 REPL 后可以体验 Function Calling。比如问「北京天气怎么样」或「帮我计算 123 * 456」，Agent 会自动选择合适的工具。

详见 `notebooks/` 目录下的实验笔记。
