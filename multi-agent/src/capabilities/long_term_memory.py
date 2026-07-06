"""长期记忆管理器 — JSON 持久化 + ChromaDB 向量检索。

JSON 是权威数据源（含全文+UUID），ChromaDB 是向量索引。
每条记忆有 UUID，合并/删除时单条操作，无需全量重建。
"""

import json
import re
import uuid
from datetime import datetime
from pathlib import Path

from ..agent_framework.chroma_store import ChromaDBStore


class LongTermMemory:

    _CONSOLIDATE_INTERVAL = 10
    _COLLECTION = "memories"

    def __init__(self, es: ChromaDBStore, llm_client):
        self._dir = Path("agent_memory")
        self._file = self._dir / "agent_memory.json"
        self._es = es
        self._llm = llm_client
        self._memories: list[dict] = []
        self._add_count = 0

        self._dir.mkdir(parents=True, exist_ok=True)
        if self._file.exists():
            self._memories = json.loads(self._file.read_text(encoding="utf-8"))
        else:
            self._save()
        self.build()
        if self._memories and self._CONSOLIDATE_INTERVAL > 0:
            self.consolidate()

    # ================================================================
    # 索引
    # ================================================================

    def build(self):
        """首次建立向量索引。已有数据则跳过。"""
        if self._es.collection_size(self._COLLECTION) > 0:
            return
        self.reindex()

    def reindex(self):
        """强制从 JSON 全量重建向量索引。"""
        items = [
            {"text": m["content"],
             "meta": {"id": m["id"], "timestamp": m["timestamp"]}}
            for m in self._memories
        ]
        self._es.rebuild(self._COLLECTION, items)

    # ================================================================
    # 写入
    # ================================================================

    def add(self, content: str) -> bool:
        """添加一条记忆。返回 True 表示直接追加，False 表示已合并到已有记忆。"""
        content = content.strip()
        now = datetime.now().isoformat(timespec="seconds")
        related = self._find_related(content)

        if not related:
            m = {"id": uuid.uuid4().hex[:12], "content": content, "timestamp": now}
            self._memories.append(m)
            self._es.add(self._COLLECTION, content,
                         {"id": m["id"], "timestamp": now})
            self._save()
            self._maybe_consolidate()
            return True

        # 与已有记忆语义重叠 → LLM 合并
        to_merge = [self._memories[i]["content"] for i in related]
        to_merge.append(content)
        merged = self._merge_batch(to_merge) if len(to_merge) > 1 else [content]

        for i in sorted(related, reverse=True):
            self._es.delete(self._COLLECTION, self._memories[i]["id"])
            self._memories.pop(i)

        for text in merged:
            m = {"id": uuid.uuid4().hex[:12], "content": text, "timestamp": now}
            self._memories.append(m)
            self._es.add(self._COLLECTION, text,
                         {"id": m["id"], "timestamp": now})

        self._save()
        self._maybe_consolidate()
        return False

    def forget(self, index: int):
        """按序号删除一条记忆。"""
        if 0 <= index < len(self._memories):
            self._es.delete(self._COLLECTION, self._memories[index]["id"])
            self._memories.pop(index)
            self._save()

    # ================================================================
    # 读取
    # ================================================================

    def search(self, query: str, top_k: int = 3,
               min_score: float = 0.3) -> list[dict]:
        """语义检索，返回 top_k 条记忆（含时间衰减）。"""
        if not self._memories or not query.strip():
            return []

        results = self._es.search(
            self._COLLECTION, query,
            top_k=len(self._memories), threshold=0.0,
        )
        for r in results:
            ts = r["meta"].get("timestamp", "")
            r["score"] = r["score"] * _time_decay(ts)

        results.sort(key=lambda x: x["score"], reverse=True)

        picked = []
        for r in results[:top_k]:
            if r["score"] < min_score:
                continue
            picked.append({
                "content": r["text"],
                "score": r["score"],
                "timestamp": r["meta"].get("timestamp", ""),
            })
        return picked

    def list_all(self) -> list[dict]:
        """列出所有记忆（带序号）。"""
        return [{"index": i, **m} for i, m in enumerate(self._memories)]

    # ================================================================
    # 合并（LLM 驱动）
    # ================================================================

    CONSOLIDATE_PROMPT = (
        "你是一个记忆整理助手。以下是关于用户的一组长期记忆。\n"
        "\n"
        "可能出现的问题：\n"
        "1. 重复：不同时间记录了同一个事实，只是表述不同\n"
        "2. 矛盾：后记录的信息与之前的相悖（应以最新的为准）\n"
        "3. 碎片化：同一主题的多条记忆分散\n"
        "4. 过时：某些信息已不再适用\n"
        "\n"
        "请将这些记忆合并整理，输出清洗后的 JSON 数组。要求：\n"
        "- 同一主题的碎片记忆合并为一条总结，保留所有关键信息\n"
        "- 重复的事实合并为一条，保留最完整的版本\n"
        "- 冲突的信息只保留最新的\n"
        "- 用户的行为偏好和约束必须原样保留\n"
        "- 不要编造原始记忆里不存在的信息\n"
        "\n"
        "直接输出 JSON 数组，格式：[{\"content\": \"...\"}, ...]"
    )

    def consolidate(self) -> int:
        """LLM 驱动的全量记忆合并去重。返回移除的条数。"""
        if len(self._memories) <= 1:
            return 0

        before = len(self._memories)
        text = "\n".join(
            f"- [{m['timestamp']}] {m['content']}" for m in self._memories
        )
        response = self._llm.chat([
            {"role": "system", "content": self.CONSOLIDATE_PROMPT},
            {"role": "user", "content": f"请整理以下记忆：\n\n{text}"},
        ])
        cleaned = _extract_json_array(response.choices[0].message.content)
        if cleaned is None:
            return 0

        now = datetime.now().isoformat(timespec="seconds")
        self._memories = [
            {"id": uuid.uuid4().hex[:12],
             "content": item["content"].strip(),
             "timestamp": now}
            for item in cleaned
            if isinstance(item, dict) and "content" in item
        ]
        self._save()
        self.reindex()
        self._add_count = 0
        return before - len(self._memories)

    # ================================================================
    # 工具
    # ================================================================

    def get_tools(self) -> list[dict]:
        """返回 LTM 提供的工具（save_memory / recall_memory）。"""
        return [
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
                "fn": self._tool_save,
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
                "fn": self._tool_recall,
            },
        ]

    def _tool_save(self, content: str) -> str:
        self.add(content)
        return "记忆已保存"

    def _tool_recall(self, query: str) -> str:
        results = self.search(query, top_k=3)
        if not results:
            return "未找到相关记忆"
        lines = []
        for r in results:
            ts = r.get("timestamp", "")[:10]
            lines.append(f"- [{ts}] {r['content']}")
        return "\n".join(lines)

    # ================================================================
    # 内部
    # ================================================================

    def _maybe_consolidate(self):
        if self._CONSOLIDATE_INTERVAL <= 0:
            return
        self._add_count += 1
        if self._add_count >= self._CONSOLIDATE_INTERVAL:
            self.consolidate()

    def _find_related(self, content: str,
                      sim_threshold: float = 0.6) -> list[int]:
        """找到与 content 语义相似的已有记忆（返回索引列表）。"""
        scores, items = self._es.batch_similarity(self._COLLECTION, content)
        id_to_idx = {m["id"]: i for i, m in enumerate(self._memories)}
        related = []
        for score, item in zip(scores, items):
            mid = item["meta"].get("id", "")
            if score >= sim_threshold and mid in id_to_idx:
                related.append(id_to_idx[mid])
        return related
    
    _MERGE_BATCH_PROMPT = (
        "你是记忆整理助手。以下是一组语义重叠的记忆。\n"
        "请将它们整理为简洁的记忆列表。\n"
        "\n"
        "规则：\n"
        "- 关于同一事实的多条记忆 → 合并为一条，保留所有独有信息\n"
        "- 关于不同主题的记忆 → 各自独立保留\n"
        "- 每条记忆一句话，中文，20 字以内\n"
        "- 不要编造不存在的信息\n"
        "\n"
        "直接输出 JSON 数组，格式：[{\"content\": \"...\"}, ...]"
    )

    def _merge_batch(self, contents: list[str]) -> list[str]:
        """LLM 合并一组语义重叠的记忆。"""
        items = "\n".join(f"- {c}" for c in contents)
        response = self._llm.chat([
            {"role": "system", "content": self._MERGE_BATCH_PROMPT},
            {"role": "user", "content": f"请整理以下记忆：\n\n{items}"},
        ])
        parsed = _extract_json_array(response.choices[0].message.content)
        if parsed and isinstance(parsed, list):
            return [item["content"].strip() for item in parsed
                    if isinstance(item, dict) and "content" in item]
        return contents

    def _save(self):
        self._file.write_text(
            json.dumps(self._memories, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ================================================================
# 模块级工具
# ================================================================

def _time_decay(timestamp: str) -> float:
    """时间衰减：30 天半衰期。"""
    try:
        t = datetime.fromisoformat(timestamp)
        age_days = (datetime.now() - t).total_seconds() / 86400
        return 0.5 ** (age_days / 30)
    except Exception:
        return 0.1


def _extract_json_array(text: str) -> list | None:
    """从 LLM 返回文本中提取 JSON 数组。"""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or start >= end:
        return None
    try:
        result = json.loads(text[start:end + 1])
        return result if isinstance(result, list) else None
    except json.JSONDecodeError:
        return None
