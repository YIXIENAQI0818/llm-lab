import tiktoken


class ConversationMemory:
    """对话历史管理器，提供短期记忆能力。

    消息以 OpenAI 原生格式存储，与 API 对齐。
    使用 tiktoken 精确计数 token，支持超限自动裁剪 + LLM 摘要压缩。
    """

    _enc = tiktoken.get_encoding("cl100k_base")
    _MSG_OVERHEAD = 4

    SUMMARIZE_PROMPT = (
        "请用一段简洁的文字（2-3句话）摘要以下对话历史，"
        "保留关键信息（人物、事件、决策、数字、偏好等），省略寒暄和重复内容。"
        "如果之前已经有摘要，新摘要应和旧摘要合并，不要重复相同信息。"
    )

    def __init__(self, system_prompt: str | None = None, max_tokens: int | None = None,
                 llm_client=None):
        self._messages: list[dict] = []
        self.max_tokens = max_tokens
        self._llm = llm_client
        self._summary = ""
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
        self._summary = ""
        if sp:
            self.add_system(sp)

    @property
    def summary(self) -> str:
        return self._summary

    # ---- Token 计数 ----

    def token_count(self) -> int:
        """精确 token 计数（tiktoken）。"""
        return self._count_tokens(self._messages)

    def stats(self) -> dict:
        return {
            "n_messages": len(self._messages),
            "tokens": self.token_count(),
            "summary_tokens": len(self._enc.encode(self._summary)) if self._summary else 0,
        }

    # ---- 裁剪 ----

    def trim(self, max_tokens: int):
        """纯裁剪（无 LLM 时用），保留 system prompt + 最近消息。"""
        start = 1 if self._messages and self._messages[0]["role"] == "system" else 0
        while self._count_tokens(self._messages) > max_tokens and len(self._messages) > start + 1:
            self._messages.pop(start)
            self._clean_tool_orphans(start)

    # ---- 内部 ----

    def _count_tokens(self, messages: list[dict]) -> int:
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

    def _clean_tool_orphans(self, start: int):
        """删除 start 位置开始的孤立 tool 消息（assistant tool_calls 已被裁）。"""
        while start < len(self._messages) and self._messages[start].get("role") == "tool":
            self._messages.pop(start)

    def _auto_trim(self):
        if self.max_tokens is None:
            return
        if self._llm:
            self._summarize_and_trim()
        else:
            self.trim(self.max_tokens)

    def _summarize_and_trim(self):
        """裁剪前把被删消息发给 LLM 摘要，结果累积到 _summary。"""
        start = 1 if self._messages and self._messages[0]["role"] == "system" else 0

        # 收集需要被删的消息
        to_remove = []
        while self._count_tokens(self._messages) > self.max_tokens and len(self._messages) > start + 1:
            to_remove.append(self._messages.pop(start))
            self._clean_tool_orphans(start)

        if not to_remove:
            return

        # 拼接被删消息
        text = "\n".join(
            f"{m['role']}: {m.get('content', '')[:300]}"
            for m in to_remove
        )

        # LLM 摘要
        prompt = f"已有的摘要：{self._summary}\n\n新增对话：\n{text}" if self._summary else text
        try:
            response = self._llm.chat([
                {"role": "system", "content": self.SUMMARIZE_PROMPT},
                {"role": "user", "content": f"请摘要：\n\n{prompt}"},
            ])
            self._summary = response.choices[0].message.content.strip()
        except Exception:
            pass  # 摘要失败不影响主流程


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
