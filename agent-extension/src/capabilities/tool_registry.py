"""工具中心 — Agent 唯一对接的工具组件。

三类工具统一注册:
  register_tool()   — 本地工具（纯函数 / LTM/KB/PM 方法）
  register_skill()  — Skill（内部 LLM 循环,包成 fn）
  register_mcp()    — MCP（远程 Server,批量注册）

对外:
  get_definitions() — 返回给 LLM 的工具列表（> top_k 时语义搜索）
  execute()         — 执行工具（不区分类型）
"""

import json
from ..agent_framework.chroma_store import ChromaDBStore
from .tool_infra.local_tools import get_local_tools
from .tool_infra.skill import scan_skills, load_skill_prompt
from .tool_infra.mcp_client import get_mcp_clients


class ToolRegistry:

    def __init__(self, es: ChromaDBStore, components: list | None = None,
                 llm_client=None, collection: str = "tools"):
        self._tools: dict[str, dict] = {}
        self._mcp_clients: list = []  # 持有 MCPClient 引用（用于 cleanup）
        self._es = es
        self._collection = collection

        # === 三类工具加载 ===

        # 1. 本地工具（纯函数 + LTM/KB/PM 组件方法，统一在 local_tools 内部处理）
        for t in get_local_tools(components or []):
            self.register_tool(**t)

        # 2. Skills（扫描 skills/ 目录，注册单个 Skill 工具）
        skills = scan_skills()
        if skills:
            self._register_skill_tool(skills)

        # 3. MCP
        for mcp in get_mcp_clients():
            self.register_mcp(mcp)

        self.build_tool_index()

    # ================================================================
    # 三类注册
    # ================================================================

    def register_tool(self, name: str, description: str,
                      parameters: dict, fn):
        """注册本地工具。"""
        self._tools[name] = {
            "definition": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            },
            "fn": fn,
        }

    def _register_skill_tool(self, skills):
        """注册唯一的 Skill 入口工具。

        description 拼接所有 Skill 的 name + description（给 LLM 做语义匹配），
        fn 接收 name 参数，调 load_skill_prompt 从磁盘读 body。
        """
        skill_desc_lines = [
            f"- {s.name}: {s.description}" for s in skills
        ]
        skill_names = [s.name for s in skills]

        def fn(name: str) -> str:
            prompt = load_skill_prompt(name)
            if prompt is None:
                return f"Skill '{name}' 不存在。可用 Skills: {', '.join(skill_names)}"
            return prompt

        self.register_tool(
            "Skill",
            "加载一个 Skill 的操作指南。"
            "可用 Skills:\n" + "\n".join(skill_desc_lines),
            {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "要加载的 Skill 名称",
                    },
                },
                "required": ["name"],
            },
            fn,
        )

    def register_mcp(self, mcp_client):
        """注册 MCP — discover 工具 → 逐一 register_tool。"""
        self._mcp_clients.append(mcp_client)
        for tool in mcp_client.discover():
            name = tool["name"]
            if mcp_client.namespace:
                name = f"{mcp_client.namespace}{name}"
            self.register_tool(
                name,
                tool.get("description", ""),
                tool.get("inputSchema",
                         {"type": "object", "properties": {}}),
                mcp_client._make_caller(tool["name"]),
            )

    # ================================================================
    # 索引
    # ================================================================

    def build_tool_index(self, force: bool = False):
        """所有工具的 description 统一写入 ChromaDB。来源无差别。"""
        if not force and self._es.collection_size(self._collection) > 0:
            return
        items = [
            {"text": t["definition"]["function"]["description"],
             "meta": {"name": t["definition"]["function"]["name"]}}
            for t in self._tools.values()
        ]
        self._es.rebuild(self._collection, items)

    # ================================================================
    # 查询
    # ================================================================

    def get_definitions(self, query: str | None = None, top_k: int = 5,
                        always_include: set[str] | None = None) -> list[dict]:
        """返回 OpenAI 格式的工具定义列表。

        工具总数 ≤ top_k 时全量返回，> top_k 时语义搜索。
        """
        if query and top_k and len(self._tools) > top_k:
            results = self._es.search(
                self._collection, query, top_k=top_k, threshold=0.0,
            )
            names = {r["meta"]["name"] for r in results}
            if always_include:
                names.update(always_include)
            return [self._tools[n]["definition"]
                    for n in names if n in self._tools]

        return [t["definition"] for t in self._tools.values()]

    def get_definitions_for(self, tool_names: list[str]) -> list[dict]:
        """返回指定名称的工具定义（给 Skill 内部用，不经过语义搜索）。"""
        return [self._tools[n]["definition"]
                for n in tool_names if n in self._tools]

    # ================================================================
    # 执行
    # ================================================================

    def execute(self, name: str, args: dict) -> str:
        """执行工具。不区分类型，统一 str(fn(**args))。"""
        if name not in self._tools:
            return json.dumps(
                {"error": f"未知工具: {name}"}, ensure_ascii=False,
            )
        try:
            return str(self._tools[name]["fn"](**args))
        except Exception as e:
            return json.dumps(
                {"error": f"执行失败: {e}"}, ensure_ascii=False,
            )

    # ================================================================
    # 工具
    # ================================================================

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def close(self):
        """清理 MCP 子进程。"""
        for mcp in self._mcp_clients:
            try:
                mcp.close()
            except Exception:
                pass
        self._mcp_clients.clear()

    def __len__(self):
        return len(self._tools)
