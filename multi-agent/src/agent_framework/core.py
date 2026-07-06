import json
import re

from .chroma_store import ChromaDBStore
from .llm import LLMClient
from .memory import ConversationMemory
from ..capabilities.tool_registry import ToolRegistry
from ..capabilities.demo_tools import builtin_tools
from ..capabilities.long_term_memory import LongTermMemory
from ..capabilities.plan_manager import PlanManager
from ..capabilities.knowledge_base import KnowledgeBase

# Unicode 代理对范围 (U+D800–U+DFFF)，单独出现时不是合法 Unicode
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


def _sanitize(text: str) -> str:
    """移除不合法的 Unicode 代理字符，防止 print 时崩溃。"""
    return _SURROGATE_RE.sub("", text)


class Agent:
    """Agent 主循环。

    内部自建 LLM、ChromaDBStore、ConversationMemory、ToolRegistry、
    LongTermMemory、PlanManager、KnowledgeBase。
    对外只暴露 chat() / reset()。
    """

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        tools: list[str] | None = None,
        tool_collection: str = "tools",
        extra_tools: list[dict] | None = None,
    ):
        self.llm = LLMClient()
        self.es = ChromaDBStore()

        self.ltm = LongTermMemory(self.es, llm_client=self.llm)
        self.pm = PlanManager()
        self.kb = KnowledgeBase(self.es, llm_client=self.llm)
        self.kb.build_kb_index()

        # 收集工具：组件工具 + 外部注入
        all_tools = builtin_tools()
        all_tools.extend(self.ltm.get_tools())
        all_tools.extend(self.pm.get_tools())
        all_tools.extend(self.kb.get_tools())
        if extra_tools:
            all_tools.extend(extra_tools)

        # tools=None → 全部；tools=[...] → 只保留指定工具
        if tools is not None:
            keep = set(tools)
            all_tools = [t for t in all_tools if t["name"] in keep]

        self.tr = ToolRegistry(self.es, all_tools, collection=tool_collection)
        self.tr.build_tool_index()

        self.cm = ConversationMemory(self.llm, system_prompt=system_prompt)

    def reindex_kb(self, force: bool = False) -> str:
        """重建知识库索引。force=True 强制覆盖已有数据。"""
        return self.kb.build_kb_index(force=force)

    def reindex_memories(self, force: bool = False):
        """重建长期记忆向量索引。force=True 强制覆盖已有数据。"""
        self.ltm.build_ltm_index(force=force)

    def reindex_tools(self, force: bool = False):
        """重建工具向量索引。force=True 强制覆盖已有数据。"""
        self.tr.build_tool_index(force=force)

    _MAX_ROUNDS = 50
    _ALWAYS_INCLUDE = {
        "recall_memory", "check_plan", "make_plan",
        "complete_step", "add_plan_step", "modify_plan_step",
        "save_memory", "search_docs", "clear_plan",
        "delegate_task",
    }

    def _always_include(self) -> set[str]:
        """动态计算 always_include：只包含已注册的工具。"""
        return self._ALWAYS_INCLUDE & set(self.tr.list_tools())

    def chat(self, user_input: str, verbose: bool = True) -> str:
        """执行一轮对话，返回最终回复。"""
        self.cm.add_user(user_input)

        for _ in range(self._MAX_ROUNDS):
            response = self.llm.chat(
                self.cm.get_messages(),
                tools=self.tr.get_definitions(
                    query=user_input,
                    always_include=self._always_include(),
                ),
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                self.cm.add_assistant(msg)
                for tc in msg.tool_calls:
                    result = self.tr.execute(
                        tc.function.name,
                        json.loads(tc.function.arguments),
                    )
                    self.cm.add_tool_result(tc.id, result)
                    if verbose:
                        print(f"🔧 [{tc.function.name}] → {_sanitize(result)}")
            else:
                self.cm.add_assistant(msg)
                return _sanitize(msg.content or "")

        return "达到最大轮次，停止。"

    def reset(self):
        """清空对话历史，保留 system prompt 和已注册的工具。"""
        self.cm.clear()
