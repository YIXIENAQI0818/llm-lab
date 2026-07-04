class ConversationMemory:
    """对话历史管理器，提供短期记忆能力。

    消息以 OpenAI 原生格式存储，与 API 对齐。
    支持 token 粗估和上下文窗口裁剪。
    """

    def __init__(self, system_prompt: str | None = None):
        self._messages: list[dict] = []
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

    def add_tool_result(self, call_id: str, result: str):
        self._messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": result,
        })

    # ---- 读取 ----

    def get_messages(self) -> list[dict]:
        """返回当前消息列表（清理掉 None 字段）。"""
        return _strip_nones(self._messages)

    def clear(self):
        """清空历史，保留 system prompt。"""
        sp = None
        if self._messages and self._messages[0]["role"] == "system":
            sp = self._messages[0]["content"]
        self._messages = []
        if sp:
            self.add_system(sp)

    # ---- 工具 ----

    def stats(self) -> dict:
        """统计消息数与估算 token 数。"""
        total_chars = sum(
            len(str(m.get("content", "")))
            for m in self._messages
        )
        return {
            "n_messages": len(self._messages),
            "estimated_tokens": total_chars // 2,  # 粗估
        }

    def trim(self, max_tokens: int):
        """裁剪到 max_tokens 以内，保留 system prompt + 最近消息。"""
        start = 1 if self._messages and self._messages[0]["role"] == "system" else 0
        while self._stats_from(start)["estimated_tokens"] > max_tokens and len(self._messages) > start + 1:
            self._messages.pop(start)

    def _stats_from(self, start: int) -> dict:
        total = sum(
            len(str(m.get("content", "")))
            for m in self._messages[start:]
        )
        return {"estimated_tokens": total // 2}


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
