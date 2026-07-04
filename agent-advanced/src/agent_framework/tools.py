import json
from typing import Callable


class ToolRegistry:
    """工具注册中心：注册、查询、执行。"""

    def __init__(self, tools: list[dict] | None = None):
        self._tools: dict[str, dict] = {}
        if tools:
            for t in tools:
                self.register(**t)

    def register(self, name: str, description: str, parameters: dict, fn: Callable):
        """注册一个工具。

        Args:
            name: 工具名称（LLM 通过它来调用）
            description: 工具用途说明
            parameters: JSON Schema 格式的参数定义
            fn: 实际执行的 Python 函数
        """
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

    def get_definitions(self) -> list[dict]:
        """返回 OpenAI 格式的工具定义列表。"""
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
