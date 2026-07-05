#!/usr/bin/env python3
"""Agent 交互式 REPL — 像对话一样使用 Agent。"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.agent_framework.core import Agent


# 清洗终端输入时可能产生的代理字符碎片（WSL 删除中文时的残留）
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def main():
    agent = Agent()
    _print_startup(agent)

    while True:
        try:
            user_input = _SURROGATE_RE.sub("", input("👤 You: ").strip())
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            _handle_command(user_input, agent)
            continue

        try:
            reply = agent.chat(user_input)
            print(f"🤖 Agent: {reply}\n")
        except Exception as e:
            print(f"❌ 错误: {e}\n")


def _print_startup(agent: Agent):
    print("🤖 Agent CLI — 输入消息开始对话")
    print("   /exit 退出  /clear 清空历史  /history 查看历史  /memories 查看记忆  /plan <任务> 手动计划  /help 帮助")
    print(f"   📐 Token 上限: {agent.cm.max_tokens}（tiktoken 精确计数，超限自动裁剪）")
    print(f"   🧠 长期记忆: {len(agent.ltm.list_all())} 条")
    print("   /remember <内容>  手动存储记忆  /forget <序号>  删除记忆")
    if agent.pm.is_active:
        print(f"   📋 活跃计划: {agent.pm.active['task']}")
    tools_list = ", ".join(agent.tr.list_tools())
    print(f"   可用工具: {tools_list}")
    print()


# ============================================================
# 命令处理
# ============================================================

def _handle_command(cmd: str, agent: Agent):
    parts = cmd.split(maxsplit=1)
    action = parts[0].lower()

    if action == "/exit":
        print("再见！")
        sys.exit(0)

    elif action == "/clear":
        agent.reset()
        print("对话历史已清空。\n")

    elif action == "/history":
        _show_history(agent)

    elif action == "/remember":
        if len(parts) < 2:
            print("用法: /remember <要记住的内容>\n")
        else:
            agent.ltm.add(parts[1])
            print("记忆已保存。\n")

    elif action == "/forget":
        try:
            idx = int(parts[1]) if len(parts) > 1 else -1
            agent.ltm.forget(idx)
            print(f"已删除记忆 #{idx}。\n")
        except (ValueError, IndexError):
            print("用法: /forget <序号>，用 /memories 查看序号。\n")

    elif action == "/memories":
        all_mem = agent.ltm.list_all()
        if not all_mem:
            print("(无长期记忆)\n")
        else:
            for m in all_mem:
                print(f"  [{m['index']}] {m['content']}  ({m['timestamp']})")
            print()

    elif action == "/consolidate":
        before = len(agent.ltm.list_all())
        removed = agent.ltm.consolidate()
        after = len(agent.ltm.list_all())
        print(f"记忆合并完成：{before} → {after}（合并了 {removed} 条）\n")

    elif action == "/plan":
        if len(parts) < 2:
            print("用法: /plan <任务描述>\n")
        else:
            task = parts[1]
            if agent.pm.is_active:
                print("(旧计划已归档)")
                agent.pm.clear()
            print(f"开始规划：{task}")
            reply = agent.chat(
                f"请为以下任务制定分步计划，调用 make_plan 工具：{task}"
            )
            print(f"🤖 Agent: {reply}\n")

    elif action == "/reindex":
        result = agent.kb.reindex()
        print(f"{result}\n")

    elif action == "/reindex_memories":
        agent.ltm.reindex()
        print("记忆索引已重建\n")

    elif action == "/reindex_tools":
        agent.tr.reindex()
        print("工具索引已重建\n")

    elif action == "/help":
        print("命令列表:")
        print("  /exit         退出程序")
        print("  /clear        清空对话历史")
        print("  /history      查看历史消息与统计")
        print("  /remember     手动存储长期记忆")
        print("  /forget       删除长期记忆")
        print("  /memories     查看所有长期记忆")
        print("  /consolidate  LLM 合并清理长期记忆")
        print("  /reindex      强制重建知识库索引")
        print("  /reindex_memories  强制重建记忆索引")
        print("  /reindex_tools  强制重建工具索引")
        print("  /plan         手动创建计划（强制覆盖旧计划）")
        print("  /help         显示此帮助\n")

    else:
        print(f"未知命令: {action}，输入 /help 查看帮助\n")


def _show_history(agent: Agent):
    msgs = agent.cm.get_messages()
    if not msgs:
        print("(无历史)\n")
        return
    for i, m in enumerate(msgs):
        role = m["role"]
        content = str(m.get("content", ""))[:120]
        tool_info = ""
        if m.get("tool_calls"):
            tool_info = f" [调用: {', '.join(tc['function']['name'] for tc in m['tool_calls'])}]"
        print(f"  [{i}] {role}{tool_info}: {content}")
    stats = agent.cm.stats()
    print(f"\n  📊 消息数: {stats['n_messages']}, Token: {stats['tokens']} (tiktoken)\n")


if __name__ == "__main__":
    main()
