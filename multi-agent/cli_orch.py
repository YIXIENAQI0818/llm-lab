#!/usr/bin/env python3
"""Orchestrator 交互式 REPL — 多 Agent 协作模式。"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.orchestration.orchestrator import Orchestrator

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def main():
    orch = Orchestrator()
    _print_startup(orch)

    while True:
        try:
            user_input = _SURROGATE_RE.sub("", input("👤 You: ").strip())
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            _handle_command(user_input, orch)
            continue

        try:
            reply = orch.chat(user_input)
            print(f"🤖 Orchestrator: {reply}\n")
        except Exception as e:
            print(f"❌ 错误: {e}\n")


def _print_startup(orch: Orchestrator):
    print("🤖 Orchestrator CLI — 多 Agent 协作模式")
    print(f"   Workers: {', '.join(orch.list_workers())}")
    print("   /exit 退出  /clear 清空历史  /workers 查看 Worker")
    print("   /reindex / reindex_memories / reindex_tools  /help 帮助")
    print()


def _handle_command(cmd: str, orch: Orchestrator):
    parts = cmd.split(maxsplit=1)
    action = parts[0].lower()

    if action == "/exit":
        print("再见！")
        sys.exit(0)

    elif action == "/clear":
        orch.reset()
        print("对话历史已清空（含所有 Worker）。\n")

    elif action == "/workers":
        print(f"可用 Worker: {', '.join(orch.list_workers())}\n")

    elif action == "/reindex":
        result = orch.reindex_kb(force=True)
        print(f"{result}\n")

    elif action == "/reindex_memories":
        orch.reindex_memories(force=True)
        print("记忆索引已重建\n")

    elif action == "/reindex_tools":
        orch.reindex_tools()
        print("工具索引已重建\n")

    elif action == "/help":
        print("命令列表:")
        print("  /exit             退出程序")
        print("  /clear            清空对话历史")
        print("  /workers          查看可用 Worker 列表")
        print("  /reindex          重建知识库索引")
        print("  /reindex_memories 重建记忆索引")
        print("  /reindex_tools    重建所有工具索引")
        print("  /help             显示此帮助\n")

    else:
        print(f"未知命令: {action}，输入 /help 查看帮助\n")


if __name__ == "__main__":
    main()
