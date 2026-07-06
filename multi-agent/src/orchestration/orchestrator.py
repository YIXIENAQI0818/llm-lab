"""Orchestrator — 多 Agent 编排主控。

按需创建 Worker Agent，delegate_task 工具将任务分派给指定 Worker。
"""

import json

from ..agent_framework.core import Agent
from .roles import ROLES, create_worker, list_roles

ORCHESTRATOR_SYSTEM_PROMPT = (
    "你是一个全能的 AI 副手，可以用中文或用户使用的语言回复。"
    "当用户提及你不知道或不确定的事实信息时，先调用 recall_memory 搜索长期记忆再回答。"
    "当用户告诉你关于自己的重要信息时，主动调用 save_memory 保存。"
    "当面对需要多步协调的复杂任务时，先调用 make_plan 制定计划，再逐步执行。"
    "当用户的问题需要基于已索引的文档内容回答时，先调用 search_docs 检索。"
    "在给出最终答案前，请先自我检查：数据是否准确？逻辑是否完整？"
    ""
    "你有一个专业团队可以通过 delegate_task 调派："
    f"  - researcher：深度研究、多轮搜索、大量文档检索"
    f"  - programmer：代码编写、复杂计算"
    ""
    "工作原则："
    "1. 简单的事自己做 —— 单次查询、日常计算、记忆读写、文档检索。"
    "2. 需要深度研究或专业编程时，派给对应 Worker。"
    "3. 多个独立的子任务一次性派发，它们会并行执行；有依赖的子任务串行。"
    "4. 汇总所有结果回复用户。"
)


class Orchestrator:
    """多 Agent 编排器。

    按需创建 Worker，暴露 chat() / reset() 接口，与 Agent 对齐。
    """

    def __init__(self):
        self._last_user_input = ""
        self._agent = Agent(
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            tools=None,
            extra_tools=self.get_extra_tools(),
            tool_collection="tools_orch",
        )

    def chat(self, user_input: str, verbose: bool = True) -> str:
        """执行一轮对话。"""
        self._last_user_input = user_input
        return self._agent.chat(user_input, verbose=verbose)

    def reset(self):
        """清空 Orchestrator 对话历史。"""
        self._agent.reset()

    def list_workers(self) -> list[str]:
        return list_roles()

    def reindex_kb(self, force: bool = False) -> str:
        return self._agent.reindex_kb(force=force)

    def reindex_memories(self, force: bool = False):
        self._agent.reindex_memories(force=force)

    def reindex_tools(self):
        """重建所有工具索引（Orch 自身 + 各 Worker 角色）。"""
        self._agent.reindex_tools(force=True)

        for role in ROLES:
            create_worker(role).reindex_tools(force=True)

    # ---- 工具 ----

    def get_extra_tools(self) -> list[dict]:
        """返回 Orchestrator 额外注入的工具（delegate_task）。"""
        roles = list_roles()
        return [
            {
                "name": "delegate_task",
                "description": (
                    "将子任务分派给指定的 Worker 执行。"
                    "当需要搜索研究相关信息时，派给 researcher。"
                    "当需要编写代码或执行计算时，派给 programmer。"
                    "可以多次调用，每次派给一个 Worker，拿到结果后再决定下一步。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "worker_name": {
                            "type": "string",
                            "enum": roles,
                            "description": f"Worker 名称，可选: {', '.join(roles)}",
                        },
                        "task": {
                            "type": "string",
                            "description": "要交给 Worker 执行的任务描述，应清晰完整",
                        },
                    },
                    "required": ["worker_name", "task"],
                },
                "fn": self._tool_delegate,
            },
        ]

    def _tool_delegate(self, worker_name: str, task: str) -> str:
        roles = list_roles()
        if worker_name not in roles:
            return json.dumps(
                {"error": f"未知 Worker: {worker_name}，"
                          f"可用: {roles}"},
                ensure_ascii=False,
            )
        try:
            worker = create_worker(worker_name)
            ctx_task = (f"用户原始请求：{self._last_user_input}\n\n"
                        f"你的子任务：{task}")
            result = worker.chat(ctx_task, verbose=False)
            return json.dumps(
                {"worker": worker_name, "result": result},
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps(
                {"error": f"Worker {worker_name} 执行失败: {e}"},
                ensure_ascii=False,
            )
