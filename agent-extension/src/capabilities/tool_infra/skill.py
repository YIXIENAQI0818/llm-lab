"""Skill — 带内部 LLM 循环的复合工具。

对外：一个普通 tool（name + description + parameters + fn）
对内：独立 Agent 循环（LLM + 工具调用 + 多轮），结果汇总后返回

新增 Skill 只需在 get_skills() 里加一个 SkillDef。
"""

import json
from ...agent_framework.memory import ConversationMemory


class SkillDef:
    """Skill 声明式定义。

    Attributes:
        name: 工具名（暴露给父 Agent）
        description: 工具描述（帮助 LLM 决定何时调用）
        system_prompt: Skill 内部的系统提示词
        tools: 内部可用的工具名列表（从父 ToolRegistry 借用）
        parameters: 输入参数 JSON Schema
        max_rounds: 内部最大轮次
    """

    def __init__(self, *, name: str, description: str, system_prompt: str,
                 tools: list[str] | None = None, parameters: dict | None = None,
                 max_rounds: int = 10):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.parameters = parameters or {"type": "object", "properties": {}}
        self.max_rounds = max_rounds


class SkillRunner:
    """执行 Skill 的内部 Agent 循环。

    complex=True（默认）：多轮 LLM + 工具调用
    complex=False：单次 LLM 调用
    """

    def __init__(self, skill_def: SkillDef, llm_client,
                 parent_registry, complex: bool = True):
        self._def = skill_def
        self._llm = llm_client
        self._registry = parent_registry
        self._complex = complex

    def run(self, **kwargs) -> str:
        """执行 Skill（被父 Agent 作为工具调用时触发）。"""
        user_msg = self._format_args(kwargs)

        if not self._complex:
            resp = self._llm.chat([
                {"role": "system", "content": self._def.system_prompt},
                {"role": "user", "content": user_msg},
            ])
            return resp.choices[0].message.content or ""

        cm = ConversationMemory(self._llm, self._def.system_prompt)
        cm.add_user(user_msg)

        skill_tools = self._registry.get_definitions_for(self._def.tools)

        for _ in range(self._def.max_rounds):
            resp = self._llm.chat(cm.get_messages(), tools=skill_tools or None)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return msg.content or ""

            cm.add_assistant(msg)
            for tc in msg.tool_calls:
                result = self._registry.execute(
                    tc.function.name,
                    json.loads(tc.function.arguments),
                )
                cm.add_tool_result(tc.id, result)

        # 达到最大轮次，强制汇总
        cm.add_user("请汇总以上所有结果，用中文简洁回答。")
        resp = self._llm.chat(cm.get_messages())
        return resp.choices[0].message.content or "已达到最大轮次"

    def _format_args(self, kwargs: dict) -> str:
        if not kwargs:
            return "请开始执行任务。"
        parts = [f"{k}: {v}" for k, v in kwargs.items()]
        return "任务参数：\n" + "\n".join(parts)


def get_skills() -> list[SkillDef]:
    """返回所有已定义的 Skill。新增 Skill 在这里加。"""
    return []
