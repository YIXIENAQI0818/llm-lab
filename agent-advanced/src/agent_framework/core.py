import json
import re

from .llm import LLMClient
from .memory import ConversationMemory
from .tools import ToolRegistry

# Unicode 代理对范围 (U+D800–U+DFFF)，单独出现时不是合法 Unicode
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _sanitize(text: str) -> str:
    """移除不合法的 Unicode 代理字符，防止 print 时崩溃。"""
    return _SURROGATE_RE.sub("", text)


class Agent:
    """Agent 主循环。

    组装 LLM、Memory、Tools，对外暴露简洁的 chat() 接口。
    即支持编程调用，也支持交互式 REPL。
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        tools: list[dict] | None = None,
        system_prompt: str | None = None,
        max_rounds: int = 10,
        long_term_memory=None,
    ):
        self.llm = llm or LLMClient()
        self.memory = ConversationMemory(system_prompt)
        self.tools = ToolRegistry(tools)
        self.max_rounds = max_rounds
        self.ltm = long_term_memory

    def chat(self, user_input: str, verbose: bool = True) -> str:
        """执行一轮对话，返回最终回复。

        Args:
            user_input: 用户输入
            verbose: 是否在工具调用时打印过程

        Returns:
            模型的最终文本回复
        """
        # 长期记忆检索：增强用户输入
        if self.ltm:
            recalled = self.ltm.search(user_input, top_k=3)
            if recalled:
                context = "\n".join(f"- {r['content']}" for r in recalled)
                user_input = f"相关记忆：\n{context}\n\n用户问题：{user_input}"

        self.memory.add_user(user_input)

        for _ in range(self.max_rounds):
            response = self.llm.chat(
                self.memory.get_messages(),
                tools=self.tools.get_definitions(),
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                self.memory.add_assistant(msg)
                for tc in msg.tool_calls:
                    result = self.tools.execute(
                        tc.function.name,
                        json.loads(tc.function.arguments),
                    )
                    self.memory.add_tool_result(tc.id, result)
                    if verbose:
                        print(f"🔧 [{tc.function.name}] → {_sanitize(result)}")
            else:
                self.memory.add_assistant(msg)
                return _sanitize(msg.content or "")

        return "达到最大轮次，停止。"

    def reset(self):
        """清空对话历史，保留 system prompt 和已注册的工具。"""
        self.memory.clear()
