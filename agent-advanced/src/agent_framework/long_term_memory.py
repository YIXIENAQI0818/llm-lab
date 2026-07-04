import json
from datetime import datetime
from pathlib import Path


class LongTermMemory:
    """长期记忆管理器 — 基于 JSON 文件的持久化记忆。

    记忆以 JSON 数组形式存储在文件中，支持增删查。
    检索使用关键词匹配 + 时间衰减，不依赖外部 embedding 服务。

    用法:
        ltm = LongTermMemory()
        ltm.add("用户叫小明，住在北京")
        results = ltm.search("北京")  # → [{"content": "用户叫小明...", ...}]
        ltm.forget(0)
        ltm.list_all()
    """

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

    def add(self, content: str):
        """追加一条记忆。"""
        self._memories.append({
            "content": content.strip(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        self._save()

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
            # 关键词重叠得分：交集 / 查询词数（Jaccard 简化版）
            overlap = query_words & content_words
            score = len(overlap) / max(len(query_words), 1)
            # 时间衰减：越近得分越高，半衰期约 30 天
            age_days = _age_days(m["timestamp"])
            decay = 0.5 ** (age_days / 30)
            scored.append((score * decay, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    def list_all(self) -> list[dict]:
        """返回所有记忆（包含序号）。"""
        return [{"index": i, **m} for i, m in enumerate(self._memories)]

    # ---- 内部 ----

    def _save(self):
        self._file.write_text(
            json.dumps(self._memories, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _age_days(timestamp: str) -> float:
    """计算时间戳距现在的天数。"""
    try:
        t = datetime.fromisoformat(timestamp)
        return (datetime.now() - t).total_seconds() / 86400
    except Exception:
        return 365  # 解析失败就当很旧
