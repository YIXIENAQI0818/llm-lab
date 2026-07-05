"""示例工具集 — Agent 内部调用，构建工具列表。

工具函数 + 定义 + 绑定逻辑全部集中在这里。
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


def _recall_memory_stub(query: str) -> str:
    return "未找到相关记忆"


def _make_plan_stub(task: str, steps: list) -> str:
    return "计划已创建"


def _check_plan_stub() -> str:
    return "当前没有活跃计划"


def _complete_step_stub(step: int) -> str:
    return "步骤已完成"


def _add_plan_step_stub(desc: str) -> str:
    return "步骤已追加"


def _modify_plan_step_stub(step: int, desc: str, restart: bool = False) -> str:
    return "步骤已修改"


def _search_docs_stub(query: str, top_k: int = 3) -> str:
    return "未找到相关文档（知识库未启用或未索引）"


# ============================================================
# 绑定逻辑 + 工具列表
# ============================================================

def create_demo_tools(pm, ltm, kb) -> list[dict]:
    """创建绑定好的工具列表。"""

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
            "name": "recall_memory",
            "description": (
                "从长期记忆中检索与查询相关的信息。"
                "当用户提及或问及你不知道、不确定的人名、账号、偏好、经历、计划等事实信息时，"
                "应优先调用此工具搜索记忆。即使觉得可能不知道，也建议先查一下再回答。"
                "返回与查询最相关的 top-3 条记忆。"
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "检索关键词或问题"}},
                "required": ["query"],
            },
            "fn": _recall_memory_stub,
        },
        {
            "name": "check_plan",
            "description": (
                "查看当前活跃计划的完整状态，包括所有步骤及其完成情况。"
                "当不确定当前计划进度或忘记下一步该做什么时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
            "fn": _check_plan_stub,
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
            "name": "complete_step",
            "description": "标记当前计划的一个步骤为完成。完成后自动推进到下一步。",
            "parameters": {
                "type": "object",
                "properties": {"step": {"type": "integer", "description": "要标记完成的步骤编号（从1开始）"}},
                "required": ["step"],
            },
            "fn": _complete_step_stub,
        },
        {
            "name": "add_plan_step",
            "description": "向当前计划追加一个新步骤。",
            "parameters": {
                "type": "object",
                "properties": {"desc": {"type": "string", "description": "新步骤描述"}},
                "required": ["desc"],
            },
            "fn": _add_plan_step_stub,
        },
        {
            "name": "modify_plan_step",
            "description": (
                "修改当前计划中某个步骤的描述。"
                "小改（修正措辞、补充细节）用 restart=false。"
                "方向性大改（换项目、换方案）用 restart=true，会重置后续步骤。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "step": {"type": "integer", "description": "要修改的步骤编号（从1开始）"},
                    "desc": {"type": "string", "description": "新的步骤描述"},
                    "restart": {"type": "boolean", "description": "是否重置该步骤及后续步骤为未完成"},
                },
                "required": ["step", "desc"],
            },
            "fn": _modify_plan_step_stub,
        },
        {
            "name": "search_docs",
            "description": (
                "在已索引的知识库中语义检索与查询最相关的文档片段。"
                "当用户问的问题需要基于已索引的文档内容回答时，"
                "应优先调用此工具检索相关上下文，然后再根据检索结果回答。"
                "返回与查询最相关的 top-k 条文档片段。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索查询，应提取用户问题的关键信息"},
                    "top_k": {"type": "integer", "description": "返回的最相关文档片段数量，默认 3"},
                },
                "required": ["query"],
            },
            "fn": _search_docs_stub,
        },
    ]

    # ---- 绑定需要外部实例的工具 ----

    for t in tools:
        if t["name"] == "save_memory":
            def _save(content, _ltm=ltm):
                _ltm.add(content)
                return "记忆已保存"
            t["fn"] = _save

        elif t["name"] == "recall_memory":
            def _recall(query, _ltm=ltm):
                results = _ltm.search(query, top_k=3)
                if not results:
                    return "未找到相关记忆"
                lines = []
                for r in results:
                    ts = r.get("timestamp", "")[:10]
                    lines.append(f"- [{ts}] {r['content']}")
                return "\n".join(lines)
            t["fn"] = _recall

        elif t["name"] == "check_plan":
            def _check_plan(_pm=pm):
                if not _pm.is_active:
                    return "当前没有活跃计划"
                return _pm.format_context()
            t["fn"] = _check_plan

        elif t["name"] == "make_plan":
            def _make_plan(task, steps, _pm=pm):
                if _pm.is_active:
                    return "已有活跃计划。请先完成当前计划或调用 update_plan。"
                cleaned = []
                for s in steps:
                    s = s.strip()
                    while s and s[0] in "0123456789.、 ":
                        s = s[1:]
                    cleaned.append(s.strip())
                _pm.create(task, cleaned)
                steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(cleaned))
                return f"计划已创建（已保存到文件）：\n{steps_text}\n\n请按顺序执行第一步。"
            t["fn"] = _make_plan

        elif t["name"] == "complete_step":
            def _complete_step(step, _pm=pm):
                return _pm.complete_step(step)
            t["fn"] = _complete_step

        elif t["name"] == "add_plan_step":
            def _add_ps(desc, _pm=pm):
                return _pm.add_step(desc)
            t["fn"] = _add_ps

        elif t["name"] == "modify_plan_step":
            def _modify_ps(step, desc, restart=False, _pm=pm):
                return _pm.modify_step(step, desc, restart=restart)
            t["fn"] = _modify_ps

        elif t["name"] == "search_docs":
            def _search_docs(query, top_k=3, _kb=kb):
                results = _kb.search(query, top_k=top_k)
                if not results:
                    return "未找到相关文档"
                lines = []
                for r in results:
                    src = r["meta"].get("source", "?")
                    idx = r["meta"].get("chunk_index", "?")
                    lines.append(f"--- [{src}#{idx}] (score:{r['score']:.3f}) ---\n{r['text']}")
                return "\n".join(lines)
            t["fn"] = _search_docs

    return tools
