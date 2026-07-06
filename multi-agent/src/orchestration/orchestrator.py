"""Orchestrator — 多 Agent 编排主控。

持有多个 Worker Agent，delegate_task 工具将任务分派给指定 Worker。
Orchestrator 本身也是一个 Agent，system_prompt 强调任务分发角色。
"""

import json

from ..agent_framework.core import Agent
from .roles import create_worker, list_roles

ORCHESTRATOR_SYSTEM_PROMPT = (
    "你是一个任务分发调度者。你的职责是："
    "1. 理解用户的需求，判断该交给哪个 Worker 处理。"
    f"2. 可用的 Worker：{', '.join(list_roles())}。"
    "3. 如果任务需要多个 Worker 协作（如先研究再写代码），"
    "   先 delegate 给第一个 Worker，拿到结果后再 delegate 给下一个。"
    "4. 所有子任务完成后，汇总结果回复用户。"
)


class Orchestrator:
    """多 Agent 编排器。

    内部持有 Worker 池，暴露 chat() / reset() 接口，与 Agent 对齐。
    """

    def __init__(self):
        # 先创建 Worker 池
        self._workers: dict[str, Agent] = {
            role: create_worker(role) for role in list_roles()
        }

        # 通过 extra_tools 注入 delegate_task（和 LTM/PM/KB 的 get_tools 模式一致）
        self._agent = Agent(
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            tools=["delegate_task"],
            extra_tools=self.get_extra_tools(),
            tool_collection="tools_orch",
        )

    def chat(self, user_input: str, verbose: bool = True) -> str:
        """执行一轮对话。"""
        return self._agent.chat(user_input, verbose=verbose)

    def reset(self):
        """清空 Orchestrator 和所有 Worker 的对话历史。"""
        self._agent.reset()
        for w in self._workers.values():
            w.reset()

    def list_workers(self) -> list[str]:
        return list(self._workers.keys())

    # ---- 工具 ----

    def get_extra_tools(self) -> list[dict]:
        """返回 Orchestrator 额外注入的工具（delegate_task）。"""
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
                            "enum": list(self._workers.keys()),
                            "description": f"Worker 名称，"
                                           f"可选: {', '.join(self._workers.keys())}",
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
        if worker_name not in self._workers:
            return json.dumps(
                {"error": f"未知 Worker: {worker_name}，"
                          f"可用: {list(self._workers.keys())}"},
                ensure_ascii=False,
            )
        try:
            result = self._workers[worker_name].chat(task, verbose=False)
            return json.dumps(
                {"worker": worker_name, "result": result},
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps(
                {"error": f"Worker {worker_name} 执行失败: {e}"},
                ensure_ascii=False,
            )
