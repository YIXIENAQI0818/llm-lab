import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent.parent / '.env')


class LLMClient:
    """LLM API 客户端，封装 DeepSeek API 调用。"""

    def __init__(self, model: str = "deepseek-chat"):
        self.model = model
        self._client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )

    def chat(self, messages: list[dict], tools: list[dict] | None = None):
        """发送消息到 LLM，返回 ChatCompletion 对象。"""
        kwargs = dict(model=self.model, messages=messages)
        if tools:
            kwargs["tools"] = tools
        return self._client.chat.completions.create(**kwargs)
