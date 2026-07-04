import json

from .llm import LLMClient
from .memory import ConversationMemory
from .tools import ToolRegistry


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
    ):
        self.llm = llm or LLMClient()
        self.memory = ConversationMemory(system_prompt)
        self.tools = ToolRegistry(tools)
        self.max_rounds = max_rounds

    def chat(self, user_input: str, verbose: bool = True) -> str:
        """执行一轮对话，返回最终回复。

        Args:
            user_input: 用户输入
            verbose: 是否在工具调用时打印过程

        Returns:
            模型的最终文本回复
        """
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
                        print(f"🔧 [{tc.function.name}] → {result}")
            else:
                self.memory.add_assistant(msg)
                return msg.content or ""

        return "达到最大轮次，停止。"

    def reset(self):
        """清空对话历史，保留 system prompt 和已注册的工具。"""
        self.memory.clear()
