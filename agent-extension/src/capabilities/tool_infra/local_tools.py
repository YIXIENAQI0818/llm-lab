"""本地工具 — 纯函数 + 组件绑定方法。

所有本地工具定义集中管理。新增工具只需在这里加，不用改其他文件。
"""


def _calculate(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception:
        return "计算错误"


def get_local_tools(components: list) -> list[dict]:
    """返回所有本地工具定义（纯函数 + 组件方法）。

    components = [LTM, PM, KB]，顺序固定，三者必须存在。
    """
    ltm, pm, kb = components

    tools = [
        # 纯函数
        {
            "name": "calculate",
            "description": "执行数学计算",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "数学表达式"}},
                "required": ["expression"],
            },
            "fn": _calculate,
        },
    ]

    # LTM 工具
    tools.extend([
        {
            "name": "save_memory",
            "description": (
                "保存重要信息到长期记忆。"
                "当用户告诉你关于自己的事实、偏好、计划等信息时主动调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string",
                                "description": "要记住的内容，一句话概括"},
                },
                "required": ["content"],
            },
            "fn": ltm._tool_save,
        },
        {
            "name": "recall_memory",
            "description": (
                "从长期记忆中检索与查询相关的信息。"
                "当用户提及你不知道、不确定的人名、偏好、经历、计划等事实信息时，"
                "应先调用此工具搜索记忆。返回 top-3 条结果。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "检索关键词或问题"},
                },
                "required": ["query"],
            },
            "fn": ltm._tool_recall,
        },
    ])

    # KB 工具
    tools.append({
        "name": "search_docs",
        "description": (
            "在已索引的知识库中语义检索与查询最相关的文档片段。"
            "当用户问题需要基于已索引文档内容回答时，"
            "应先调用此工具检索上下文。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索查询，应提取用户问题的关键信息",
                },
                "strategy": {
                    "type": "string",
                    "enum": ["expand", "decompose"],
                    "description": (
                        "expand：扩展为同义词+近义词，适合概念性问题。"
                        "decompose：拆解为独立子问题，适合多面比较。"
                        "默认 expand。"
                    ),
                },
            },
            "required": ["query"],
        },
        "fn": kb._tool_search_docs,
    })

    # PM 工具
    tools.extend([
        {
            "name": "check_plan",
            "description": "查看当前活跃计划的所有步骤及完成情况。",
            "parameters": {"type": "object", "properties": {}},
            "fn": pm._tool_check_plan,
        },
        {
            "name": "make_plan",
            "description": "为复杂任务制定分步计划。已有计划时请勿重复调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string",
                             "description": "原始复杂任务描述"},
                    "steps": {"type": "array",
                              "items": {"type": "string"},
                              "description": "步骤列表，每步一句话"},
                },
                "required": ["task", "steps"],
            },
            "fn": pm._tool_make_plan,
        },
        {
            "name": "complete_step",
            "description": "标记当前计划的一个步骤为完成。",
            "parameters": {
                "type": "object",
                "properties": {
                    "step": {"type": "integer",
                             "description": "步骤编号（从1开始）"},
                },
                "required": ["step"],
            },
            "fn": pm._tool_complete_step,
        },
        {
            "name": "add_plan_step",
            "description": "向当前计划追加一个新步骤。",
            "parameters": {
                "type": "object",
                "properties": {
                    "desc": {"type": "string", "description": "新步骤描述"},
                },
                "required": ["desc"],
            },
            "fn": pm._tool_add_step,
        },
        {
            "name": "modify_plan_step",
            "description": (
                "修改当前计划中某个步骤的描述。"
                "小改用 restart=false，方向性大改用 restart=true（会重置后续步骤）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "step": {"type": "integer",
                             "description": "步骤编号（从1开始）"},
                    "desc": {"type": "string",
                             "description": "新的步骤描述"},
                    "restart": {"type": "boolean",
                                "description": "是否重置后续步骤，默认 false"},
                },
                "required": ["step", "desc"],
            },
            "fn": pm._tool_modify_step,
        },
        {
            "name": "clear_plan",
            "description": (
                "清空（归档）当前活跃计划。"
                "当用户明确要求取消计划、放弃当前任务、或计划已完成时调用。"
            ),
            "parameters": {"type": "object", "properties": {}},
            "fn": pm._tool_clear_plan,
        },
    ])

    return tools
