# api-basics

LLM API 调用基础实验。这是 llm-lab 系列的第一个子项目，学习如何使用 OpenAI 兼容 SDK 与 DeepSeek API 交互。

## 内容

- 模型参数控制（temperature、top_p、max_tokens 等）
- 流式输出（streaming）
- Token 计数（tiktoken）
- 批量处理与错误重试

## 环境

- Python 3.10+
- DeepSeek API Key

### 安装

```bash
cd api-basics
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

进入 REPL 交互界面，输入问题即可与大模型对话。详见 `notebooks/` 目录下的实验笔记。
