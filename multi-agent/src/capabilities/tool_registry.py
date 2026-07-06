"""工具注册中心 — 注册、语义查询、执行。

ToolRegistry 管理当前 Agent 的工具子集。
每个 Agent 使用独立的 ChromaDB collection，由 build_tool_index() 写入。
"""

import json
from typing import Callable

from ..agent_framework.chroma_store import ChromaDBStore


def build_tool_index(es: ChromaDBStore, all_tools: list[dict],
                    collection: str):
    """写入工具向量索引到指定 collection。已有数据则跳过。

    all_tools 中每个元素是原始工具 dict：{name, description, parameters, fn}。
    """
    if es.collection_size(collection) > 0:
        return
    items = [
        {"text": t["description"],
         "meta": {"name": t["name"]}}
        for t in all_tools
    ]
    es.rebuild(collection, items)


class ToolRegistry:

    def __init__(self, es: ChromaDBStore, tools: list[dict],
                 collection: str = "tools"):
        self._tools: dict[str, dict] = {}
        self._es = es
        self._collection = collection
        for t in tools:
            self.register(**t)

    # ================================================================
    # 注册
    # ================================================================

    def register(self, name: str, description: str,
                 parameters: dict, fn: Callable):
        """注册一个工具到内存。"""
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

    # ================================================================
    # 查询
    # ================================================================

    def get_definitions(self, query: str | None = None, top_k: int = 5,
                        always_include: set[str] | None = None) -> list[dict]:
        """返回 OpenAI 格式的工具定义列表。

        工具总数 ≤ top_k 时全量返回，> top_k 时语义搜索自己的 collection。
        搜自己的 collection 不需要过滤——搜出来的全是自己的。
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

    # ================================================================
    # 执行
    # ================================================================

    def execute(self, name: str, args: dict) -> str:
        """执行工具并返回字符串结果。"""
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

    def __len__(self):
        return len(self._tools)
