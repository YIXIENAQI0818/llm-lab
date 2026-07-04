"""示例工具集 — 供 CLI 和 notebook 共用。

工具函数 + 定义 + 绑定逻辑全部集中在这里，
CLI 只需一行 create_demo_tools(agent_ref, ltm) 拿到绑好的工具列表。
"""

import json


# ============================================================
# 工具函数（纯函数，不依赖外部状态）
# ============================================================

def get_weather(city: str) -> str:
    w = {
        "北京": {"temp": 25, "desc": "晴朗"},
        "上海": {"temp": 28, "desc": "多云"},
        "东京": {"temp": 22, "desc": "小雨"},
        "纽约": {"temp": 15, "desc": "阴天"},
    }
    for k, v in w.items():
        if k in city:
            return f"温度 {v['temp']}°C, {v['desc']}"
    return f"未找到 {city} 的天气数据"


def search_web(query: str) -> str:
    db = {
        "特斯拉": "特斯拉当前股价 $245，上季度为 $220。",
        "茅台": "茅台当前股价 ¥1650。",
        "图灵奖": "图灵奖是计算机领域最高荣誉，由ACM于1966年设立。",
        "东京人口": "东京都人口约1400万。",
    }
    for k, v in db.items():
        if k in query or query in k:
            return v
    return f"未找到关于 '{query}' 的结果"


def calculate(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception:
        return "计算错误"


# ============================================================
# 工具定义（占位函数，启动时绑定真实实例）
# ============================================================

def _save_memory_stub(content: str) -> str:
    return "记忆已保存"


def _make_plan_stub(task: str, steps: list) -> str:
    return "计划已创建"


def _update_plan_stub(action: str, changes: str = "") -> str:
    return "计划已更新"


# ============================================================
# 绑定逻辑 + 工具列表
# ============================================================

SYSTEM_PROMPT = (
    "你是一个有用的 AI 助手，可以用中文或用户使用的语言回复。"
    "需要时使用工具获取信息。"
    "当用户告诉你关于自己的重要信息（名字、偏好、计划等）时，主动调用 save_memory 保存。"
    "当面对需要多步协调的复杂任务时，先调用 make_plan 制定计划，再逐步执行。"
)


def create_demo_tools(agent_ref: list, ltm=None) -> list[dict]:
    """创建绑定好的示例工具列表。

    Args:
        agent_ref: 单元素列表 [agent]，闭包访问 Agent 实例（PlanManager 等）
        ltm: LongTermMemory 实例，None 时不启用 save_memory
    """

    tools = [
        {
            "name": "get_weather",
            "description": "查询指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名"}},
                "required": ["city"],
            },
            "fn": get_weather,
        },
        {
            "name": "search_web",
            "description": "搜索网页获取知识或信息",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "搜索关键词"}},
                "required": ["query"],
            },
            "fn": search_web,
        },
        {
            "name": "calculate",
            "description": "执行数学计算",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "数学表达式"}},
                "required": ["expression"],
            },
            "fn": calculate,
        },
        {
            "name": "save_memory",
            "description": (
                "保存重要信息到长期记忆，当用户告诉你关于自己的事实或偏好时调用。"
                "如：名字、喜好、计划等。"
            ),
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string", "description": "要记住的内容，一句话概括"}},
                "required": ["content"],
            },
            "fn": _save_memory_stub,
        },
        {
            "name": "make_plan",
            "description": (
                "当任务复杂需要多步协调时，先制定分步计划再执行。"
                "已有计划时请勿重复调用，改用 update_plan 修改。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "原始复杂任务描述"},
                    "steps": {"type": "array", "items": {"type": "string"}, "description": "分步计划，每步一句话"},
                },
                "required": ["task", "steps"],
            },
            "fn": _make_plan_stub,
        },
        {
            "name": "update_plan",
            "description": "更新当前计划：complete_step n | add_step 内容 | modify_step n 新内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "操作类型和参数"},
                    "changes": {"type": "string", "description": "具体变更内容"},
                },
                "required": ["action"],
            },
            "fn": _update_plan_stub,
        },
    ]

    # ---- 绑定需要外部实例的工具 ----

    for t in tools:
        if t["name"] == "save_memory" and ltm is not None:
            def _save(content, _ltm=ltm):
                _ltm.add(content)
                return "记忆已保存"
            t["fn"] = _save

        elif t["name"] == "make_plan":
            def _make_plan(task, steps):
                agent = agent_ref[0]
                if not agent:
                    return "Agent 未初始化"
                if agent.plan_mgr.is_active:
                    return "已有活跃计划。请先完成当前计划或调用 update_plan。"
                cleaned = []
                for s in steps:
                    s = s.strip()
                    while s and s[0] in "0123456789.、 ":
                        s = s[1:]
                    cleaned.append(s.strip())
                agent.plan_mgr.create(task, cleaned)
                steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(cleaned))
                return f"计划已创建（已保存到文件）：\n{steps_text}\n\n请按顺序执行第一步。"
            t["fn"] = _make_plan

        elif t["name"] == "update_plan":
            def _update_plan(action, changes=""):
                agent = agent_ref[0]
                if not agent:
                    return "Agent 未初始化"
                return agent.plan_mgr.update(action, changes)
            t["fn"] = _update_plan

    return tools


def create_demo_tools_no_memory(agent_ref: list) -> list[dict]:
    """创建不带 save_memory 的工具列表。"""
    tools = create_demo_tools(agent_ref, ltm=None)
    return [t for t in tools if t["name"] != "save_memory"]
