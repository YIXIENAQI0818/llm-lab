import json
import re
from pathlib import Path

from .llm import LLMClient
from .memory import ConversationMemory
from .tools import ToolRegistry

# Unicode 代理对范围 (U+D800–U+DFFF)，单独出现时不是合法 Unicode
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _sanitize(text: str) -> str:
    """移除不合法的 Unicode 代理字符，防止 print 时崩溃。"""
    return _SURROGATE_RE.sub("", text)


class Agent:
    """Agent 主循环。

    组装 LLM、Memory、Tools，对外暴露简洁的 chat() 接口。
    即支持编程调用，也支持交互式 REPL。
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        tools: list[dict] | None = None,
        system_prompt: str | None = None,
        max_rounds: int = 10,
        long_term_memory=None,
    ):
        self.llm = llm or LLMClient()
        self.memory = ConversationMemory(system_prompt)
        self.tools = ToolRegistry(tools)
        self.max_rounds = max_rounds
        self.ltm = long_term_memory

        # Plan 持久化
        self._plan_dir = Path("agent_memory/plans")
        self._plan_file = self._plan_dir / "active.json"
        self.active_plan: dict | None = self._load_plan()

    def chat(self, user_input: str, verbose: bool = True) -> str:
        """执行一轮对话，返回最终回复。

        Args:
            user_input: 用户输入
            verbose: 是否在工具调用时打印过程

        Returns:
            模型的最终文本回复
        """
        # 长期记忆检索：增强用户输入
        if self.ltm:
            recalled = self.ltm.search(user_input, top_k=3)
            if recalled:
                context = "\n".join(f"- {r['content']}" for r in recalled)
                user_input = f"相关记忆：\n{context}\n\n用户问题：{user_input}"

        # 活跃计划注入
        if self.active_plan:
            steps_text = "\n".join(
                f"  {s['status']} 步骤{i+1}: {s['desc']}"
                for i, s in enumerate(self.active_plan["steps"])
            )
            plan_hint = (
                f"[当前计划]\n{steps_text}\n"
                f"你正在按计划执行。如需修改计划请调用 update_plan。\n"
                f"如需重新制定计划，请先完成当前计划。\n"
            )
            user_input = f"{plan_hint}\n{user_input}"

        self.memory.add_user(user_input)

        for _ in range(self.max_rounds):
            response = self.llm.chat(
                self.memory.get_messages(),
                tools=self.tools.get_definitions(),
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                self.memory.add_assistant(msg)
                for tc in msg.tool_calls:
                    result = self.tools.execute(
                        tc.function.name,
                        json.loads(tc.function.arguments),
                    )
                    self.memory.add_tool_result(tc.id, result)
                    if verbose:
                        print(f"🔧 [{tc.function.name}] → {_sanitize(result)}")
            else:
                self.memory.add_assistant(msg)
                return _sanitize(msg.content or "")

        return "达到最大轮次，停止。"

    # ---- Plan 管理 ----

    def set_plan(self, task: str, steps: list[str]):
        """创建新计划，覆盖旧计划（旧计划自动归档）。"""
        if self.active_plan:
            self._archive_plan()
        self.active_plan = {
            "task": task,
            "steps": [{"desc": s, "status": "○"} for s in steps],
            "current_step": 0,
        }
        self._save_plan()
        return self.active_plan

    def update_plan(self, action: str, changes: str = "") -> str:
        """更新当前计划。返回操作结果描述。"""
        if not self.active_plan:
            return "没有活跃计划。"
        plan = self.active_plan
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
        self._save_plan()
        return f"计划已更新。（当前第 {plan['current_step']+1} 步）"

    def _load_plan(self) -> dict | None:
        """从文件加载活跃计划，不存在返回 None。"""
        if self._plan_file.exists():
            try:
                return json.loads(self._plan_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
        return None

    def _save_plan(self):
        """持久化当前计划到文件。"""
        self._plan_dir.mkdir(parents=True, exist_ok=True)
        self._plan_file.write_text(
            json.dumps(self.active_plan, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _archive_plan(self):
        """旧计划移到 done 目录。"""
        if not self.active_plan:
            return
        done_dir = self._plan_dir / "done"
        done_dir.mkdir(parents=True, exist_ok=True)
        task_name = self.active_plan.get("task", "unnamed")[:30].replace("/", "_")
        dest = done_dir / f"{task_name}.json"
        # 避免同名覆盖：加序号
        n = 1
        while dest.exists():
            dest = done_dir / f"{task_name}_{n}.json"
            n += 1
        self._plan_file.rename(dest)

    def reset(self):
        """清空对话历史，保留 system prompt 和已注册的工具。"""
        self.memory.clear()
