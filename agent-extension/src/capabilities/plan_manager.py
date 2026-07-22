"""计划管理器 — 持久化多步任务计划。

与 LongTermMemory 对称：一个管记忆，一个管计划。
活动计划存 active.json，完成计划归档到 done/ 目录。
"""

import json
from pathlib import Path


class PlanManager:

    def __init__(self):
        self._dir = Path("agent_memory/plans")
        self._file = self._dir / "active.json"
        self._done_dir = self._dir / "done"
        self.active: dict | None = self._load()

    # ================================================================
    # 属性
    # ================================================================

    @property
    def is_active(self) -> bool:
        return self.active is not None

    # ================================================================
    # 计划操作
    # ================================================================

    def create(self, task: str, steps: list[str]):
        """创建新计划。旧计划自动归档。"""
        if self.active:
            self._archive()
        self.active = {
            "task": task,
            "steps": [{"desc": s, "status": "○"} for s in steps],
            "current_step": 0,
        }
        self._save()

    def complete_step(self, step: int) -> str:
        """标记步骤为完成，自动推进到下一步。"""
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

        if all(s["status"] == "✓" for s in plan["steps"]):
            self.clear()
            return "计划已全部完成，已归档。"
        return f"步骤已完成。（当前第 {plan['current_step'] + 1} 步）"

    def add_step(self, desc: str) -> str:
        """追加一个新步骤。"""
        if not self.active:
            return "没有活跃计划。"
        self.active["steps"].append({"desc": desc, "status": "○"})
        self._save()
        return f"已追加步骤 {len(self.active['steps'])}：{desc}"

    def modify_step(self, step: int, desc: str, restart: bool = False) -> str:
        """修改步骤描述。
        restart=False: 仅改描述
        restart=True:  重置该步及后续步骤为未完成
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
            return f"步骤 {step} 已更新并重启，后续步骤已重置。"
        self._save()
        return f"步骤 {step} 已更新。"

    def clear(self):
        """清空当前计划（归档到 done/）。"""
        if self.active:
            self._archive()
            self.active = None

    def format_context(self) -> str:
        """返回计划文本，供 Agent 查看当前进度。"""
        if not self.active:
            return ""
        steps_text = "\n".join(
            f"  {s['status']} 步骤{i + 1}: {s['desc']}"
            for i, s in enumerate(self.active["steps"])
        )
        return (
            f"[当前计划]\n{steps_text}\n"
            f"每完成一步请调用 complete_step 标记。"
        )

    # ================================================================
    # 工具
    def _tool_check_plan(self) -> str:
        if not self.is_active:
            return "当前没有活跃计划"
        return self.format_context()

    def _tool_make_plan(self, task: str, steps: list[str]) -> str:
        if self.is_active:
            return "已有活跃计划，请先完成或归档。"
        # 去除 LLM 可能加的前缀编号
        cleaned = []
        for s in steps:
            s = s.strip()
            while s and s[0] in "0123456789.、 ":
                s = s[1:]
            cleaned.append(s.strip())
        self.create(task, cleaned)
        lines = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(cleaned))
        return f"计划已创建：\n{lines}\n\n请按顺序执行第一步。"

    def _tool_complete_step(self, step: int) -> str:
        return self.complete_step(step)

    def _tool_add_step(self, desc: str) -> str:
        return self.add_step(desc)

    def _tool_modify_step(self, step: int, desc: str,
                          restart: bool = False) -> str:
        return self.modify_step(step, desc, restart=restart)

    def _tool_clear_plan(self) -> str:
        if not self.is_active:
            return "当前没有活跃计划"
        task = self.active["task"]
        self.clear()
        return f"计划「{task}」已归档。"

    # ================================================================
    # 持久化
    # ================================================================

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
        """将 active.json 移动到 done/ 目录。"""
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
