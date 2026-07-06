"""Worker 角色定义 + 工厂函数。

每个角色包含 system_prompt 和 tools_filter。
create_worker() 返回配置好的 Agent 实例。
"""

from ..agent_framework.core import Agent

ROLES: dict[str, dict] = {
    "researcher": {
        "system_prompt": (
            "你是一个研究员，擅长信息检索与分析。"
            "当用户提出问题时，优先搜索知识库（search_docs）和网络（search_web）获取相关信息，"
            "也可以从长期记忆（recall_memory）中查找相关历史。"
            "将检索结果整理成清晰、有条理的回复，标注信息来源。"
        ),
        "tools": ["search_docs", "search_web", "recall_memory"],
        "collection": "tools_researcher",
    },
    "programmer": {
        "system_prompt": (
            "你是一个程序员，擅长编写代码和执行计算。"
            "当用户要求编写代码时，写出完整可运行的代码并解释关键逻辑。"
            "需要进行数学计算时，使用 calculate 工具。"
            "需要查找信息时，使用 search_web 工具。"
        ),
        "tools": ["calculate", "search_web"],
        "collection": "tools_programmer",
    },
}


def create_worker(role: str) -> Agent:
    """根据角色名创建一个 Worker Agent。"""
    if role not in ROLES:
        raise ValueError(f"未知角色: {role}，可用角色: {list(ROLES.keys())}")

    config = ROLES[role]
    return Agent(
        system_prompt=config["system_prompt"],
        tools=config["tools"],
        tool_collection=config["collection"],
    )


def list_roles() -> list[str]:
    """列出所有可用角色。"""
    return list(ROLES.keys())
