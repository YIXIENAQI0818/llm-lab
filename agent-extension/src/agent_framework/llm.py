import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent.parent / '.env')

# DeepSeek 可用模型:
#   deepseek-v4-flash — 通用对话（非思考/思考两模式，用 thinking 参数切换）
#   deepseek-v4-pro   — 更高能力推理
# 换用其他厂商（如 OpenAI）需修改 _client 的 base_url + api_key 变量名


class LLMClient:
    """LLM API 客户端，封装 API 调用。"""

    def __init__(self):
        self._client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.model = "deepseek-v4-flash"

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             thinking: bool = True):
        """发送消息到 LLM，返回 ChatCompletion 对象。

        thinking=True（默认）→ 思考模式（返回 reasoning_content）；
        thinking=False → 非思考模式。
        """
        kwargs = dict(model=self.model, messages=messages)
        if tools:
            kwargs["tools"] = tools
        kwargs["extra_body"] = {"thinking": {"type": "enabled" if thinking else "disabled"}}
        return self._client.chat.completions.create(**kwargs)
