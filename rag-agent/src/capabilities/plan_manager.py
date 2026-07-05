import json
from pathlib import Path


class PlanManager:
    """计划管理器 — 持久化计划文件，管理计划的增删改查。

    和 LongTermMemory 对称：一个管记忆，一个管计划。
    """

    def __init__(self):
        self._dir = Path("agent_memory/plans")
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

    def complete_step(self, step: int) -> str:
        """标记指定步骤为完成。"""
        if not self.active:
            return "没有活跃计划。"
        plan = self.active
        idx = step - 1
        if not (0 <= idx < len(plan["steps"])):
            return f"步骤 {step} 不存在（共 {len(plan['steps'])} 步）。"
        plan["steps"][idx]["status"] = "✓"
        if idx + 1 < len(plan["steps"]):
            plan["current_step"] = idx + 1
            plan["steps"][idx + 1]["status"] = "→"
        self._save()
        return self._check_done(plan)

    def add_step(self, desc: str) -> str:
        """追加一个步骤。"""
        if not self.active:
            return "没有活跃计划。"
        plan = self.active
        plan["steps"].append({"desc": desc, "status": "○"})
        self._save()
        return f"已追加步骤 {len(plan['steps'])}：{desc}"

    def modify_step(self, step: int, desc: str, restart: bool = False) -> str:
        """修改指定步骤的描述。

        restart=False: 仅修改描述，不影响进度（小修小改）
        restart=True: 重置该步骤及后续步骤为未完成（方向性大改）
        """
        if not self.active:
            return "没有活跃计划。"
        plan = self.active
        idx = step - 1
        if not (0 <= idx < len(plan["steps"])):
            return f"步骤 {step} 不存在（共 {len(plan['steps'])} 步）。"
        plan["steps"][idx]["desc"] = desc
        if restart:
            for i in range(idx, len(plan["steps"])):
                plan["steps"][i]["status"] = "○"
            plan["current_step"] = idx
            plan["steps"][idx]["status"] = "→"
            self._save()
            return f"步骤 {step} 已更新并重启。后续步骤已重置，请从步骤 {step} 重新开始。"
        self._save()
        return f"步骤 {step} 已更新。"

    def _check_done(self, plan: dict) -> str:
        """所有步骤完成 → 自动归档。"""
        if all(s["status"] == "✓" for s in plan["steps"]):
            self.clear()
            return "计划已全部完成，已归档。"
        return f"步骤已完成。（当前第 {plan['current_step']+1} 步）"

    def format_context(self) -> str:
        """返回计划文本，供 Agent 注入 system prompt。"""
        if not self.active:
            return ""
        steps_text = "\n".join(
            f"  {s['status']} 步骤{i+1}: {s['desc']}"
            for i, s in enumerate(self.active["steps"])
        )
        return (
            f"[当前计划]\n{steps_text}\n"
            f"每完成一步请调用 complete_step 标记。如需修改计划请调用 modify_plan_step。"
        )

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
