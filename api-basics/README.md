# api-basics

LLM API 调用基础实验。学习使用 OpenAI 兼容 SDK 与 DeepSeek API 进行大模型交互，覆盖从单轮对话到参数调优的完整 API 基础。

## 实验内容

| 回合 | 主题 | 学习点 |
|------|------|--------|
| 01 | 单轮对话 | OpenAI SDK 基础用法、API 调用流程 |
| 02 | System Prompt | 角色设定、指令遵循 |
| 03 | 多轮对话 | 对话历史管理、上下文窗口 |
| 04 | 流式输出 | Streaming、实时交互 |
| 05 | 参数调优 | temperature、top_p、max_tokens |

## 技术栈

- **LLM**: DeepSeek API（`deepseek-chat`），通过 OpenAI 兼容接口调用
- **SDK**: `openai` Python SDK
- **环境管理**: `python-dotenv`

## 环境

```bash
cd api-basics
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env: DEEPSEEK_API_KEY=sk-your-key-here
```

## 使用

`src/client.py` 和 `src/streaming.py` 是导入辅助模块（提供 `get_client()` 等工厂函数），不直接运行。

实验通过 Jupyter 笔记本进行：

```bash
jupyter notebook notebooks/01_single_turn.ipynb
```

`notebooks/` 目录下包含对应 5 个回合的实验笔记，每个可独立运行。
