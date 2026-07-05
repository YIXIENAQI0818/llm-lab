import tiktoken


class ConversationMemory:
    """对话历史管理器，提供短期记忆能力。

    消息以 OpenAI 原生格式存储，与 API 对齐。
    使用 tiktoken 精确计数 token，支持超限自动裁剪 + LLM 摘要压缩。
    """

    _enc = tiktoken.get_encoding("cl100k_base")
    _MSG_OVERHEAD = 4
    TRIM_TARGET_RATIO = 0.7  # 摘要裁剪到 max_tokens * 70%，留 30% 缓冲

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
        # 摘要消息的 token 数从 messages 中实际统计
        summary_tokens = 0
        for m in self._messages:
            if (isinstance(m.get("content"), str)
                    and m["content"].startswith("[对话摘要]")):
                summary_tokens += self._count_tokens([m])
        return {
            "n_messages": len(self._messages),
            "tokens": self.token_count(),
            "summary_tokens": summary_tokens,
        }

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

    def _auto_trim(self):
        if self.max_tokens is None:
            return
        if self._llm:
            self._summarize_and_trim()
        # 无 LLM 则不做裁剪；摘要压缩是本项目的核心能力

    def _summarize_and_trim(self):
        """超限时按完整轮次 pop，一次性裁到 max_tokens * 70%。

        每轮 pop：user + 后续非 user 消息（assistant+tc、tool），保证不拆散。
        先用临时副本模拟 pop 找出要删的消息，LLM 摘要成功后才真正删除。
        如果 LLM 调用失败，self._messages 保持原样，不丢数据。

        裁到 70% 而非刚好不超限，避免下一轮新消息进来马上又触发摘要。
        """
        start = 1 if self._messages and self._messages[0]["role"] == "system" else 0
        if len(self._messages) - start <= 1:
            return

        target = int(self.max_tokens * self.TRIM_TARGET_RATIO)

        # 在临时副本上模拟 pop，找出要删的消息（不修改 self._messages）
        temp = list(self._messages)
        to_remove = []
        while self._count_tokens(temp) > target and len(temp) > start + 1:
            to_remove.append(temp.pop(start))
            while start < len(temp) and temp[start].get("role") != "user":
                to_remove.append(temp.pop(start))

        if not to_remove:
            return

        # 拼接被删消息（跳过旧的摘要消息，避免和新摘要重复）
        text = "\n".join(
            f"{m['role']}: {m.get('content', '')[:300]}"
            for m in to_remove
            if not (isinstance(m.get("content"), str)
                    and m["content"].startswith("[对话摘要]"))
        )

        # 旧摘要 + 新对话 → LLM 合并为一份
        prompt = f"已有摘要：{self._summary}\n\n新对话：\n{text}" if self._summary else text
        try:
            response = self._llm.chat([
                {"role": "system", "content": self.SUMMARIZE_PROMPT},
                {"role": "user", "content": f"请摘要以下对话：\n\n{prompt}"},
            ])
            new_summary = response.choices[0].message.content.strip()
        except Exception:
            return  # LLM 失败，self._messages 未被修改，消息完整保留

        # LLM 成功 → 真正删除 + 插入摘要在原位
        for _ in range(len(to_remove)):
            self._messages.pop(start)

        self._messages.insert(start, {
            "role": "assistant",
            "content": f"[对话摘要] {new_summary}",
        })
        self._summary = new_summary


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
