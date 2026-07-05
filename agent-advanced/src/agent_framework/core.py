import json
import re

from .embedding_store import EmbeddingStore
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

    组装 LLM、Memory、Tools，以及外部注入的能力（LTM、PlanManager）。
    对外暴露简洁的 chat() 接口。
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        tools: list[dict] | None = None,
        system_prompt: str | None = None,
        max_rounds: int = 50,
        long_term_memory=None,
        plan_mgr=None,
        tool_top_k: int | None = None,
        embedding_store=None,
        max_tokens: int | None = None,
    ):
        self.llm = llm or LLMClient()
        self.memory = ConversationMemory(system_prompt, max_tokens=max_tokens,
                                         llm_client=self.llm)

        es = embedding_store if embedding_store is not None else EmbeddingStore()
        self.tools = ToolRegistry(es, tools)

        self.max_rounds = max_rounds
        self.ltm = long_term_memory
        self.plan_mgr = plan_mgr
        self.tool_top_k = tool_top_k

    def chat(self, user_input: str, verbose: bool = True) -> str:
        """执行一轮对话，返回最终回复。"""
        self.memory.add_user(user_input)

        for _ in range(self.max_rounds):
            response = self.llm.chat(
                self.memory.get_messages(),
                tools=self.tools.get_definitions(
                    query=user_input, top_k=self.tool_top_k,
                    always_include={"recall_memory", "check_plan"},
                ),
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

    # system prompt 在 __init__ 中通过 ConversationMemory 设置后不再变动。
    # LTM、plan 均通过工具由 LLM 按需拉取，不注入 system prompt。
