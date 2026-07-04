import json
from pathlib import Path


class PlanManager:
    """计划管理器 — 持久化计划文件，管理计划的增删改查。

    和 LongTermMemory 对称：一个管记忆，一个管计划。
    """

    def __init__(self, storage_dir: str = "agent_memory/plans"):
        self._dir = Path(storage_dir)
        self._file = self._dir / "active.json"
        self._done_dir = self._dir / "done"
        self.active: dict | None = self._load()

    # ---- 属性 ----

    @property
    def is_active(self) -> bool:
        return self.active is not None

    # ---- Plan 生命周期 ----

    def create(self, task: str, steps: list[str]):
        """创建新计划，覆盖旧计划（旧计划自动归档）。"""
        if self.active:
            self._archive()
        self.active = {
            "task": task,
            "steps": [{"desc": s, "status": "○"} for s in steps],
            "current_step": 0,
        }
        self._save()

    def update(self, action: str, changes: str = "") -> str:
        """更新当前计划。返回操作结果描述。"""
        if not self.active:
            return "没有活跃计划。"
        plan = self.active
        parts = action.split(maxsplit=1)
        cmd = parts[0]
        try:
            if cmd == "complete_step":
                idx = int(parts[1]) - 1 if len(parts) > 1 else plan["current_step"]
                plan["steps"][idx]["status"] = "✓"
                if idx + 1 < len(plan["steps"]):
                    plan["current_step"] = idx + 1
                    plan["steps"][idx + 1]["status"] = "→"
            elif cmd == "add_step" and changes:
                plan["steps"].append({"desc": changes, "status": "○"})
            elif cmd == "modify_step" and changes:
                idx = int(parts[1]) - 1 if len(parts) > 1 else plan["current_step"]
                plan["steps"][idx]["desc"] = changes
        except (ValueError, IndexError):
            return "操作失败：参数格式不正确。"
        self._save()

        # 所有步骤完成 → 自动归档
        if all(s["status"] == "✓" for s in plan["steps"]):
            self.clear()
            return "计划已全部完成，已归档。"

        return f"计划已更新。（当前第 {plan['current_step']+1} 步）"

    def inject(self, user_input: str) -> str:
        """将当前计划注入用户输入。"""
        if not self.active:
            return user_input
        steps_text = "\n".join(
            f"  {s['status']} 步骤{i+1}: {s['desc']}"
            for i, s in enumerate(self.active["steps"])
        )
        plan_hint = (
            f"[当前计划]\n{steps_text}\n"
            f"每完成一步请调用 update_plan 标记（complete_step N）。"
            f"所有步骤完成后计划会自动归档。\n"
        )
        return f"{plan_hint}\n{user_input}"

    def clear(self):
        """清空当前计划（归档）。"""
        if self.active:
            self._archive()
            self.active = None

    # ---- 内部 ----

    def _load(self) -> dict | None:
        if self._file.exists():
            try:
                return json.loads(self._file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
        return None

    def _save(self):
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file.write_text(
            json.dumps(self.active, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _archive(self):
        if not self.active:
            return
        self._done_dir.mkdir(parents=True, exist_ok=True)
        name = self.active.get("task", "unnamed")[:30].replace("/", "_")
        dest = self._done_dir / f"{name}.json"
        n = 1
        while dest.exists():
            dest = self._done_dir / f"{name}_{n}.json"
            n += 1
        self._file.rename(dest)
