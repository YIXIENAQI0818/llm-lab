import json
import re
from datetime import datetime
from pathlib import Path


class LongTermMemory:
    """长期记忆管理器 — 基于 JSON 文件的持久化记忆。

    记忆以 JSON 数组形式存储在文件中，支持增删查。
    检索使用关键词匹配 + 时间衰减，不依赖外部 embedding 服务。
    支持相似度去重（本地）和 LLM 驱动的记忆合并。

    用法:
        ltm = LongTermMemory()
        ltm.add("用户叫小明，住在北京")       # 自动相似度去重
        results = ltm.search("北京")
        ltm.consolidate(llm_client)            # LLM 深度清理
    """

    # 相似度阈值：超过此值视为同一事实，覆盖旧记忆
    _SIMILARITY_THRESHOLD = 0.5

    def __init__(self, storage_dir: str = "agent_memory"):
        self._dir = Path(storage_dir)
        self._file = self._dir / "agent_memory.json"
        self._memories: list[dict] = []

        self._dir.mkdir(parents=True, exist_ok=True)
        if self._file.exists():
            self._memories = json.loads(self._file.read_text(encoding="utf-8"))
        else:
            self._save()

    # ---- 写入 ----

    def add(self, content: str) -> bool:
        """添加一条记忆。自动去重：如与已有记忆高度相似则覆盖，否则追加。

        Returns:
            True 表示新增，False 表示覆盖了已有记忆。
        """
        content = content.strip()
        now = datetime.now().isoformat(timespec="seconds")

        # 相似度去重
        for m in self._memories:
            if _content_similarity(content, m["content"]) >= self._SIMILARITY_THRESHOLD:
                m["content"] = content
                m["timestamp"] = now
                self._save()
                return False  # 覆盖

        # 全新记忆
        self._memories.append({"content": content, "timestamp": now})
        self._save()
        return True  # 新增

    def forget(self, index: int):
        """删除指定序号的记忆（序号从 0 开始）。"""
        if 0 <= index < len(self._memories):
            self._memories.pop(index)
            self._save()

    # ---- 读取 ----

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """检索与 query 最相关的 top_k 条记忆。

        算法：对每条记忆计算关键词重叠得分，再乘以时间衰减因子。
        返回按得分降序排列的记忆列表。
        """
        if not self._memories or not query.strip():
            return []

        query_words = set(query)
        scored = []
        for m in self._memories:
            content = m["content"]
            content_words = set(content)
            overlap = query_words & content_words
            score = len(overlap) / max(len(query_words), 1)
            age_days = _age_days(m["timestamp"])
            decay = 0.5 ** (age_days / 30)
            scored.append((score * decay, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

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
        """调用 LLM 对所有记忆做合并去重。返回合并前后数量差。

        Args:
            llm_client: LLMClient 实例，用于调用 LLM

        Returns:
            被合并掉的记忆数量（合并前 - 合并后）
        """
        if len(self._memories) <= 1:
            return 0

        before = len(self._memories)

        # 构建 prompt
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

        # 从 LLM 回复中提取 JSON 数组
        cleaned = _extract_json_array(text)
        if cleaned is None:
            return 0  # 解析失败，不改动

        # 替换记忆：保留原始 timestamp 的逻辑（新整理的用当前时间）
        now = datetime.now().isoformat(timespec="seconds")
        self._memories = []
        for item in cleaned:
            if isinstance(item, dict) and "content" in item:
                self._memories.append({
                    "content": item["content"].strip(),
                    "timestamp": now,
                })

        self._save()
        return before - len(self._memories)

    # ---- 内部 ----

    def _save(self):
        self._file.write_text(
            json.dumps(self._memories, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _content_similarity(a: str, b: str) -> float:
    """计算两条记忆内容的相似度（0~1）。

    使用字级别 Jaccard 系数：交集字数 / 并集字数。
    """
    set_a = set(a)
    set_b = set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _age_days(timestamp: str) -> float:
    """计算时间戳距现在的天数。"""
    try:
        t = datetime.fromisoformat(timestamp)
        return (datetime.now() - t).total_seconds() / 86400
    except Exception:
        return 365  # 解析失败就当很旧


def _extract_json_array(text: str) -> list | None:
    """从 LLM 回复中提取 JSON 数组，容忍 markdown 代码块包裹。"""
    # 去掉 markdown 代码块
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)

    # 找到第一个 [ 和最后一个 ]
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or start >= end:
        return None

    try:
        result = json.loads(text[start:end + 1])
        return result if isinstance(result, list) else None
    except json.JSONDecodeError:
        return None
