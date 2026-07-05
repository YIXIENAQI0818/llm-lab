import json
from typing import Callable

from ..agent_framework.chroma_store import ChromaDBStore


class ToolRegistry:
    """工具注册中心：注册、查询、执行。

    支持基于 embedding 的工具过滤：工具多时，只把语义最相关的发给 LLM。
    """

    def __init__(self, es : ChromaDBStore, tools: list[dict] | None = None):
        self._tools: dict[str, dict] = {}
        self._es = es
        if tools:
            for t in tools:
                self.register(**t)

    def register(self, name: str, description: str, parameters: dict, fn: Callable):
        """注册一个工具到内存。描述向量索引由 build() / reindex() 统一管理。"""
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

    def build(self):
        """首次建立工具向量索引。已有数据则跳过。"""
        if self._es.collection_size("tools") > 0:
            return
        self.reindex()

    def reindex(self):
        """强制重建工具向量索引。"""
        items = [
            {"text": t["definition"]["function"]["description"],
             "meta": {"name": name}}
            for name, t in self._tools.items()
        ]
        self._es.rebuild("tools", items)

    def get_definitions(self, query: str | None = None, top_k: int = 5,
                        always_include: set[str] | None = None) -> list[dict]:
        """返回 OpenAI 格式的工具定义列表。

        query + top_k 提供时，只返回语义最相关的 top_k 个工具。
        always_include 中的工具始终包含，不受过滤影响。
        """
        if query and top_k and len(self._tools) > top_k:
            results = self._es.search("tools", query, top_k=top_k, threshold=0.0)
            names = {r["meta"]["name"] for r in results}
            if always_include:
                names.update(always_include)
            return [self._tools[n]["definition"] for n in names if n in self._tools]

        return [t["definition"] for t in self._tools.values()] if self._tools else []

    def execute(self, name: str, args: dict) -> str:
        """执行工具并返回字符串结果。异常会被捕获并返回错误信息。"""
        if name not in self._tools:
            return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
        try:
            result = self._tools[name]["fn"](**args)
            return str(result)
        except Exception as e:
            return json.dumps({"error": f"执行失败: {e}"}, ensure_ascii=False)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self):
        return len(self._tools)
