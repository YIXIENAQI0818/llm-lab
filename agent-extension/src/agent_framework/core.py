import json
import re
from .llm import LLMClient
from .chroma_store import ChromaDBStore
from .memory import ConversationMemory
from ..capabilities.tool_registry import ToolRegistry
from ..capabilities.long_term_memory import LongTermMemory
from ..capabilities.plan_manager import PlanManager
from ..capabilities.knowledge_base import KnowledgeBase

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")

DEFAULT_SYSTEM_PROMPT = (
    "你是一个有用的 AI 助手，可以用中文或用户使用的语言回复。"
    "当用户提及你不知道或不确定的事实信息时，先调用 recall_memory 搜索长期记忆再回答。"
    "当用户告诉你关于自己的重要信息（名字、偏好、计划等）时，主动调用 save_memory 保存。"
    "如果用户的信息是对已有记忆的更新（而非完全新的事实），存入的内容应同时包含新旧信息，不要丢失旧记忆中的重要事实。"
    "当面对需要多步协调的复杂任务时，先调用 make_plan 制定计划，再逐步执行。"
    "当用户的问题需要基于已索引的文档内容回答时，先调用 search_docs 检索相关上下文。"
    "如果检索结果涉及问题的多个方面（如方法A和方法B），应尽量全面回答，不要遗漏。"
    "在给出最终答案前，请先自我检查：数据是否准确？逻辑是否完整？是否有遗漏？如果发现问题，先修正再回答。"
)

_CRITICAL_TOOLS = {
    "recall_memory", "save_memory",
    "make_plan", "check_plan", "complete_step", "add_plan_step",
    "modify_plan_step", "clear_plan",
    "search_docs",
}


def _sanitize(text: str) -> str:
    return _SURROGATE_RE.sub("", text)


class Agent:
    """Agent 主循环 — 单 Agent 优先设计。

    内部创建 LLMClient、ChromaDBStore、LTM、PM、KB，
    通过 ToolRegistry 统一管理三类工具（本地/Skill/MCP）。
    """

    _MAX_ROUNDS = 50

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 max_rounds: int = 50):
        self._max_rounds = max_rounds

        # 核心依赖
        self.llm = LLMClient()
        self.store = ChromaDBStore()

        # 组件
        self.ltm = LongTermMemory(self.store, llm_client=self.llm)
        self.pm = PlanManager()
        self.kb = KnowledgeBase(self.store, llm_client=self.llm)
        self.kb.build_kb_index()

        # ToolRegistry — 自己负责加载三类工具
        self.tr = ToolRegistry(
            self.store,
            components=[self.ltm, self.pm, self.kb],
            llm_client=self.llm,
        )

        # 对话记忆
        self.cm = ConversationMemory(self.llm, system_prompt=system_prompt)

    def _execute_tools(self, tool_calls, verbose: bool):
        for tc in tool_calls:
            result = self.tr.execute(
                tc.function.name,
                json.loads(tc.function.arguments),
            )
            self.cm.add_tool_result(tc.id, result)
            if verbose:
                print(f"🔧 [{tc.function.name}] → {_sanitize(result)}")

    def _compute_always_include(self) -> set[str]:
        return _CRITICAL_TOOLS & set(self.tr.list_tools())

    def chat(self, user_input: str, verbose: bool = True) -> str:
        self.cm.add_user(user_input)

        for _ in range(self._max_rounds):
            response = self.llm.chat(
                self.cm.get_messages(),
                tools=self.tr.get_definitions(
                    query=user_input,
                    always_include=self._compute_always_include(),
                ),
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                self.cm.add_assistant(msg)
                self._execute_tools(msg.tool_calls, verbose)
            else:
                self.cm.add_assistant(msg)
                return _sanitize(msg.content or "")

        return "达到最大轮次，停止。"

    def reset(self):
        self.cm.clear()

    def reindex_kb(self, force: bool = False) -> str:
        return self.kb.build_kb_index(force=force)

    def reindex_memories(self, force: bool = False):
        self.ltm.build_ltm_index(force=force)

    def reindex_tools(self, force: bool = False):
        self.tr.build_tool_index(force=force)
