import tiktoken


class ConversationMemory:
    """对话历史管理器，提供短期记忆能力。

    消息以 OpenAI 原生格式存储，与 API 对齐。
    使用 tiktoken 精确计数 token，支持超限自动裁剪。
    """

    _enc = tiktoken.get_encoding("cl100k_base")
    # 每条消息的固定结构开销（role + 标点等），粗估 ~4 token
    _MSG_OVERHEAD = 4

    def __init__(self, system_prompt: str | None = None, max_tokens: int | None = None):
        self._messages: list[dict] = []
        self.max_tokens = max_tokens
        if system_prompt:
            self.add_system(system_prompt)

    # ---- 写入 ----

    def add_system(self, content: str):
        if self._messages and self._messages[0]["role"] == "system":
            self._messages[0]["content"] = content
        else:
            self._messages.insert(0, {"role": "system", "content": content})

    def add_user(self, content: str):
        self._messages.append({"role": "user", "content": content})
        self._auto_trim()

    def add_assistant(self, message):
        """添加完整的 assistant message（可能含 tool_calls）。"""
        self._messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in (message.tool_calls or [])
            ] if message.tool_calls else None,
        })
        self._auto_trim()

    def add_tool_result(self, call_id: str, result: str):
        self._messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": result,
        })
        self._auto_trim()

    # ---- 读取 ----

    def get_messages(self) -> list[dict]:
        return _strip_nones(self._messages)

    def clear(self):
        sp = None
        if self._messages and self._messages[0]["role"] == "system":
            sp = self._messages[0]["content"]
        self._messages = []
        if sp:
            self.add_system(sp)

    # ---- Token 计数 ----

    def token_count(self) -> int:
        """精确 token 计数（tiktoken）。"""
        return self._count_tokens(self._messages)

    def stats(self) -> dict:
        return {
            "n_messages": len(self._messages),
            "tokens": self.token_count(),
        }

    # ---- 裁剪 ----

    def trim(self, max_tokens: int):
        """裁剪到 max_tokens 以内，保留 system prompt + 最近消息。"""
        start = 1 if self._messages and self._messages[0]["role"] == "system" else 0
        while self._count_tokens(self._messages) > max_tokens and len(self._messages) > start + 1:
            self._messages.pop(start)

    # ---- 内部 ----

    def _count_tokens(self, messages: list[dict]) -> int:
        """对消息列表做精确 token 计数。"""
        total = 0
        for m in messages:
            total += self._MSG_OVERHEAD
            content = m.get("content") or ""
            total += len(self._enc.encode(content))
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    total += len(self._enc.encode(fn.get("name", "")))
                    total += len(self._enc.encode(fn.get("arguments", "")))
            if m.get("tool_call_id"):
                total += len(self._enc.encode(m["tool_call_id"]))
        return total

    def _auto_trim(self):
        if self.max_tokens is not None:
            self.trim(self.max_tokens)


def _strip_nones(messages: list[dict]) -> list[dict]:
    """去掉消息中值为 None 的字段（API 可能拒收 None）。"""
    result = []
    for m in messages:
        cleaned = {}
        for k, v in m.items():
            if v is not None:
                cleaned[k] = v
        result.append(cleaned)
    return result
