# agent-basics

Agent 与 Function Calling 基础实验。学习 LLM 如何调用外部工具——工具定义、LLM 自主选择工具、执行结果回传并继续推理（ReAct 循环）。

## 实验内容

| 回合 | 主题 | 学习点 |
|------|------|--------|
| 01 | Function Calling | 工具定义格式、LLM 决策、执行反馈 |
| 02 | 多工具选择 | LLM 在多工具场景中自动匹配合适工具 |
| 03 | Agent Loop | ReAct 模式：思考 → 行动 → 观察 → 继续思考 |

## 技术栈

- **LLM**: DeepSeek API（支持 OpenAI 兼容 Function Calling）
- **工具**: Python 纯函数（mock weather / search / calculator）

## 环境

```bash
cd agent-basics
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env: DEEPSEEK_API_KEY=sk-your-key-here
```

## 使用

```bash
jupyter notebook notebooks/01_function_calling.ipynb
```

`notebooks/` 目录下包含 3 个回合的实验笔记。代码集中在 `src/client.py`，每个 notebook 从中 import 所需函数。
