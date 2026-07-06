"""Orchestrator — 多 Agent 编排主控。

按需创建 Worker Agent，delegate_task 工具将任务分派给指定 Worker。
"""

import json

from ..agent_framework.core import Agent
from .roles import ROLES, create_worker, list_roles

ORCHESTRATOR_SYSTEM_PROMPT = (
    "你是一个任务分发调度者。你的职责是："
    "1. 理解用户的需求，判断该交给哪个 Worker 处理。"
    f"2. 可用的 Worker：{', '.join(list_roles())}。"
    "3. 当需要多个独立任务时（如同时研究 A 和 B），"
    "   一次性派发多个 delegate_task，它们会并行执行。"
    "4. 当任务有依赖关系（如先研究再写代码），"
    "   先 delegate 第一个，拿到结果后再 delegate 下一个。"
    "5. 所有子任务完成后，汇总结果回复用户。"
)


class Orchestrator:
    """多 Agent 编排器。

    按需创建 Worker，暴露 chat() / reset() 接口，与 Agent 对齐。
    """

    def __init__(self):
        self._last_user_input = ""
        self._agent = Agent(
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            tools=["delegate_task"],
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
