"""工具注册中心 — 注册、索引、查询、执行。"""

import json
from typing import Callable

from ..agent_framework.chroma_store import ChromaDBStore


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
    # 索引
    # ================================================================

    def build_tool_index(self, force: bool = False):
        """写入工具向量索引。已有数据则跳过，force=True 强制重建。"""
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

        工具总数 ≤ top_k 时全量返回，> top_k 时语义搜索自己的 collection。
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
