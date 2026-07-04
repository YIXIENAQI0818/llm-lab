import json
from typing import Callable

from .embedding_store import EmbeddingStore


class ToolRegistry:
    """工具注册中心：注册、查询、执行。

    支持基于 embedding 的工具过滤：工具多时，只把语义最相关的发给 LLM。
    """

    def __init__(self, embedding_store: EmbeddingStore | None = None, tools: list[dict] | None = None):
        self._tools: dict[str, dict] = {}
        self._embedding = embedding_store if embedding_store is not None else EmbeddingStore()
        if tools:
            for t in tools:
                self.register(**t)

    def register(self, name: str, description: str, parameters: dict, fn: Callable):
        """注册一个工具，并自动将其描述加入 embedding 索引。"""
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
        self._embedding.add("tools", description, {"name": name})

    def get_definitions(self, query: str | None = None, top_k: int | None = None) -> list[dict]:
        """返回 OpenAI 格式的工具定义列表。

        query + top_k 提供时，只返回语义最相关的 top_k 个工具。
        不提供 query 时返回全部（向后兼容）。
        """
        if query and top_k and len(self._tools) > top_k:
            results = self._embedding.search("tools", query, top_k=top_k)
            names = {r["meta"]["name"] for r in results}
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
