import json
import re
from datetime import datetime
from pathlib import Path

from ..agent_framework.embedding_store import EmbeddingStore


class LongTermMemory:
    """长期记忆管理器 — JSON 持久化 + embedding 语义检索。

    记忆以 JSON 数组形式存储在文件中，支持增删查。
    检索使用 embedding 余弦相似度 + 时间衰减。
    支持 embedding 相似度去重和 LLM 驱动的记忆合并。
    """

    _SIMILARITY_THRESHOLD = 0.75  # embedding 余弦相似度阈值（短中文文本偏高）
    _MIN_SCORE_THRESHOLD = 0.3   # 检索时低于此值的记忆不返回

    def __init__(self, embedding_store: EmbeddingStore, storage_dir: str = "agent_memory",
                 llm_client=None):
        self._dir = Path(storage_dir)
        self._file = self._dir / "agent_memory.json"
        self._embedding = embedding_store
        self._llm = llm_client
        self._memories: list[dict] = []

        self._dir.mkdir(parents=True, exist_ok=True)
        if self._file.exists():
            self._memories = json.loads(self._file.read_text(encoding="utf-8"))
            self._rebuild_embedding_index()
        else:
            self._save()

    # ---- 写入 ----

    def add(self, content: str) -> bool:
        """添加一条记忆。自动去重：如与已有记忆高度相似则覆盖。

        Returns:
            True 表示新增，False 表示覆盖了已有记忆。
        """
        content = content.strip()
        now = datetime.now().isoformat(timespec="seconds")

        # embedding 相似度去重
        for idx, m in enumerate(self._memories):
            if self._content_similarity(content, m["content"]) >= self._SIMILARITY_THRESHOLD:
                if self._llm:
                    merged = self._merge_pair(m["content"], content)
                else:
                    merged = content
                m["content"] = merged
                m["timestamp"] = now
                self._save()
                self._embedding.update("memories", idx, merged, {"timestamp": now})
                return False

        self._memories.append({"content": content, "timestamp": now})
        self._embedding.add("memories", content, {"timestamp": now})
        self._save()
        return True

    def forget(self, index: int):
        """删除指定序号的记忆（序号从 0 开始）。"""
        if 0 <= index < len(self._memories):
            self._memories.pop(index)
            self._save()
            self._embedding.delete("memories", index)

    # ---- 读取 ----

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """检索与 query 语义最相关的 top_k 条记忆。

        算法：embedding 余弦相似度 × 时间衰减，低于阈值的过滤。
        """
        if not self._memories or not query.strip():
            return []

        results = self._embedding.search(
            "memories", query, top_k=len(self._memories), threshold=0.0,
        )

        # 叠加时间衰减
        for r in results:
            ts = r["meta"].get("timestamp", "")
            r["score"] = r["score"] * _time_decay(ts)

        # 按衰减后得分排序，过滤低分
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
        """返回所有记忆（包含序号）。"""
        return [{"index": i, **m} for i, m in enumerate(self._memories)]

    # ---- LLM 驱动记忆合并 ----

    CONSOLIDATE_PROMPT = """你是一个记忆整理助手。以下是关于用户的一组长期记忆，每一条记录了用户的事实或偏好。

可能出现的问题：
1. 重复：不同时间记录了同一个事实，只是表述不同
2. 矛盾：后记录的信息与之前的相悖（应以最新的为准）
3. 过时：某些信息已不再适用

请将这些记忆合并整理，输出清洗后的 JSON 数组。要求：
- 合并重复的记忆，保留信息更完整的那条
- 冲突的记忆只保留最新的
- 保持记忆简洁（每条一句话，20字以内最佳）
- 不要编造原始记忆里不存在的信息

直接输出 JSON 数组，格式：[{"content": "..."}, ...]"""

    def consolidate(self, llm_client) -> int:
        """调用 LLM 对所有记忆做合并去重。返回合并前后数量差。"""
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

        response = llm_client.chat(messages)
        text = response.choices[0].message.content

        cleaned = _extract_json_array(text)
        if cleaned is None:
            return 0

        now = datetime.now().isoformat(timespec="seconds")
        self._memories = [
            {"content": item["content"].strip(), "timestamp": now}
            for item in cleaned
            if isinstance(item, dict) and "content" in item
        ]

        self._save()
        self._rebuild_embedding_index()
        return before - len(self._memories)

    # ---- 内部 ----

    def _content_similarity(self, a: str, b: str) -> float:
        """两条记忆文本的语义相似度（0~1）。"""
        return self._embedding.similarity(a, b)

    _MERGE_PAIR_PROMPT = (
        "你是记忆整理助手。以下是两条关于同一个事实的长期记忆，旧记忆和新记忆有重叠但不完全一样。"
        "请将它们合并成一条简洁的记忆（一句话，中文），保留双方各自独有的重要信息。"
        "直接输出合并后的记忆文本，不要加任何前缀或标点包裹。"
    )

    def _merge_pair(self, old: str, new: str) -> str:
        """用 LLM 将新旧记忆合并为一条。"""
        messages = [
            {"role": "system", "content": self._MERGE_PAIR_PROMPT},
            {"role": "user", "content": f"旧记忆：{old}\n\n新记忆：{new}"},
        ]
        response = self._llm.chat(messages)
        merged = response.choices[0].message.content.strip()
        return merged or new  # LLM 返回空则保留新的

    def _rebuild_embedding_index(self):
        """从当前记忆列表重建 EmbeddingStore 中的 memories collection。"""
        items = [
            {"text": m["content"], "meta": {"timestamp": m["timestamp"]}}
            for m in self._memories
        ]
        self._embedding.rebuild("memories", items)

    def _save(self):
        self._file.write_text(
            json.dumps(self._memories, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _time_decay(timestamp: str) -> float:
    """时间衰减因子：每 30 天减半。"""
    try:
        t = datetime.fromisoformat(timestamp)
        age_days = (datetime.now() - t).total_seconds() / 86400
        return 0.5 ** (age_days / 30)
    except Exception:
        return 0.1


def _extract_json_array(text: str) -> list | None:
    """从 LLM 回复中提取 JSON 数组，容忍 markdown 代码块包裹。"""
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
