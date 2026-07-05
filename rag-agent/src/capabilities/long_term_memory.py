import json
import re
import uuid
from datetime import datetime
from pathlib import Path

from ..agent_framework.embedding_store import EmbeddingStore


class LongTermMemory:
    """长期记忆管理器 — JSON 持久化 + embedding 语义检索。

    JSON 是权威源数据（含全文+UUID），ChromaDB 是向量索引。
    每条记忆有 UUID，合并/删除时单条操作，无需全量重建。
    """

    _SIMILARITY_THRESHOLD = 0.6
    _MIN_SCORE_THRESHOLD = 0.3
    _CONSOLIDATE_INTERVAL = 10

    def __init__(self, es: EmbeddingStore, llm_client):
        self._dir = Path("agent_memory")
        self._file = self._dir / "agent_memory.json"
        self._es = es
        self._llm = llm_client
        self._add_since_consolidate = 0
        self._memories: list[dict] = []

        self._dir.mkdir(parents=True, exist_ok=True)
        if self._file.exists():
            self._memories = json.loads(self._file.read_text(encoding="utf-8"))
            self._rebuild_embedding_index()
            if self._CONSOLIDATE_INTERVAL > 0:
                self.consolidate()
        else:
            self._save()

    # ---- 写入 ----

    def add(self, content: str) -> bool:
        content = content.strip()
        now = datetime.now().isoformat(timespec="seconds")

        related = self._find_related(content)

        if not related:
            # 全新记忆
            m = {"id": uuid.uuid4().hex[:12], "content": content, "timestamp": now}
            self._memories.append(m)
            self._es.add("memories", content, {"id": m["id"], "timestamp": now})
            self._save()
            self._maybe_consolidate()
            return True

        # 合并路径：LLM 整理 → 删旧 UUID → 写新
        to_merge = [self._memories[i]["content"] for i in related]
        to_merge.append(content)

        merged_list = self._merge_batch(to_merge) if len(to_merge) > 1 else [content]

        for i in sorted(related, reverse=True):
            self._es.delete("memories", self._memories[i]["id"])
            self._memories.pop(i)

        for m_text in merged_list:
            m = {"id": uuid.uuid4().hex[:12], "content": m_text, "timestamp": now}
            self._memories.append(m)
            self._es.add("memories", m_text, {"id": m["id"], "timestamp": now})

        self._save()
        self._maybe_consolidate()
        return False

    def forget(self, index: int):
        if 0 <= index < len(self._memories):
            self._es.delete("memories", self._memories[index]["id"])
            self._memories.pop(index)
            self._save()

    # ---- 读取 ----

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not self._memories or not query.strip():
            return []

        results = self._es.search(
            "memories", query, top_k=len(self._memories), threshold=0.0,
        )

        for r in results:
            ts = r["meta"].get("timestamp", "")
            r["score"] = r["score"] * _time_decay(ts)

        results.sort(key=lambda x: x["score"], reverse=True)

        picked = []
        for r in results[:top_k]:
            if r["score"] < self._MIN_SCORE_THRESHOLD:
                continue
            picked.append({
                "content": r["text"],
                "score": r["score"],
                "timestamp": r["meta"].get("timestamp", ""),
            })
        return picked

    def list_all(self) -> list[dict]:
        return [{"index": i, **m} for i, m in enumerate(self._memories)]

    # ---- LLM 驱动记忆合并 ----

    CONSOLIDATE_PROMPT = """你是一个记忆整理助手。以下是关于用户的一组长期记忆，每一条记录了用户的事实或偏好。

可能出现的问题：
1. 重复：不同时间记录了同一个事实，只是表述不同
2. 矛盾：后记录的信息与之前的相悖（应以最新的为准）
3. 碎片化：同一主题的多条记忆分散（如'开了A项目''选了B方案''取消了A项目'）
4. 过时：某些信息已不再适用

请将这些记忆合并整理，输出清洗后的 JSON 数组。要求：
- 同一主题的碎片记忆合并为一条总结，合并时保留所有关键信息，不得遗漏
- 重复的事实合并为一条，保留最完整的版本
- 冲突的信息只保留最新的
- 用户的行为偏好和约束（如'不要做X'）必须原样保留
- 保持记忆简洁，无关临时状态应压缩而非保留细节
- 不要编造原始记忆里不存在的信息

直接输出 JSON 数组，格式：[{"content": "..."}, ...]"""

    def consolidate(self) -> int:
        if len(self._memories) <= 1:
            return 0

        before = len(self._memories)

        memory_text = "\n".join(
            f"- [{m['timestamp']}] {m['content']}"
            for m in self._memories
        )
        messages = [
            {"role": "system", "content": self.CONSOLIDATE_PROMPT},
            {"role": "user", "content": f"请整理以下记忆：\n\n{memory_text}"},
        ]

        response = self._llm.chat(messages)
        text = response.choices[0].message.content

        cleaned = _extract_json_array(text)
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
        self._rebuild_embedding_index()
        self._add_since_consolidate = 0
        return before - len(self._memories)

    # ---- 内部 ----

    def _maybe_consolidate(self):
        if self._CONSOLIDATE_INTERVAL <= 0:
            return
        self._add_since_consolidate += 1
        if self._add_since_consolidate >= self._CONSOLIDATE_INTERVAL:
            self.consolidate()

    def _find_related(self, content: str) -> list[int]:
        scores, items = self._es.batch_similarity("memories", content)
        id_to_idx = {m["id"]: i for i, m in enumerate(self._memories)}
        related = []
        for score, item in zip(scores, items):
            mid = item["meta"].get("id", "")
            if score >= self._SIMILARITY_THRESHOLD and mid in id_to_idx:
                related.append(id_to_idx[mid])
        return related

    _MERGE_BATCH_PROMPT = (
        "你是记忆整理助手。以下是一组长期记忆，它们可能在语义上有重叠。\n"
        "请将这些记忆整理为简洁的记忆列表。\n\n"
        "规则：\n"
        "- 确实关于同一事实/实体的多条记忆 → 合并为一条，保留所有独有信息\n"
        "- 关于不同主题的记忆 → 各自独立保留\n"
        "- 每条记忆一句话，中文，20字以内最佳\n"
        "- 不要编造原始记忆里不存在的信息\n\n"
        "直接输出 JSON 数组，格式：[{\"content\": \"...\"}, ...]"
    )

    def _merge_batch(self, contents: list[str]) -> list[str]:
        items = "\n".join(f"- {c}" for c in contents)
        messages = [
            {"role": "system", "content": self._MERGE_BATCH_PROMPT},
            {"role": "user", "content": f"请整理以下记忆：\n\n{items}"},
        ]
        response = self._llm.chat(messages)
        text = response.choices[0].message.content

        parsed = _extract_json_array(text)
        if parsed and isinstance(parsed, list):
            return [item["content"].strip() for item in parsed
                    if isinstance(item, dict) and "content" in item]
        return contents

    def _rebuild_embedding_index(self):
        items = [
            {"text": m["content"], "meta": {"id": m["id"], "timestamp": m["timestamp"]}}
            for m in self._memories
        ]
        self._es.rebuild("memories", items)

    def _save(self):
        self._file.write_text(
            json.dumps(self._memories, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _time_decay(timestamp: str) -> float:
    try:
        t = datetime.fromisoformat(timestamp)
        age_days = (datetime.now() - t).total_seconds() / 86400
        return 0.5 ** (age_days / 30)
    except Exception:
        return 0.1


def _extract_json_array(text: str) -> list | None:
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
