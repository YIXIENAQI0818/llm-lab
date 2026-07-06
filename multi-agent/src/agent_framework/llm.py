import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent.parent / '.env')

# DeepSeek 可用模型:
#   deepseek-chat     — 通用对话
#   deepseek-reasoner — 推理增强
# 换用其他厂商（如 OpenAI）需修改 _client 的 base_url + api_key 变量名


class LLMClient:
    """LLM API 客户端，封装 API 调用。"""

    def __init__(self):
        self._client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.model = "deepseek-chat"

    def chat(self, messages: list[dict], tools: list[dict] | None = None):
        """发送消息到 LLM，返回 ChatCompletion 对象。"""
        kwargs = dict(model=self.model, messages=messages)
        if tools:
            kwargs["tools"] = tools
        return self._client.chat.completions.create(**kwargs)
