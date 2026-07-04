#!/usr/bin/env python3
"""Agent 交互式 REPL — 像对话一样使用 Agent。

用法:
    python cli.py                 # 默认：工具 + 长期记忆
    python cli.py --no-tools      # 纯对话模式
    python cli.py --no-memory     # 不启用长期记忆
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.agent_framework import Agent, EmbeddingStore
from src.capabilities import LongTermMemory, PlanManager
from src.capabilities.demo_tools import create_demo_tools

SYSTEM_PROMPT = (
    "你是一个有用的 AI 助手，可以用中文或用户使用的语言回复。"
    "需要时使用工具获取信息。"
    "当用户告诉你关于自己的重要信息（名字、偏好、计划等）时，主动调用 save_memory 保存。"
    "当面对需要多步协调的复杂任务时，先调用 make_plan 制定计划，再逐步执行。"
    "在给出最终答案前，请先自我检查：数据是否准确？逻辑是否完整？是否有遗漏？如果发现问题，先修正再回答。"
)

# 清洗终端输入时可能产生的代理字符碎片（WSL 删除中文时的残留）
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def main():
    parser = argparse.ArgumentParser(description="Agent 交互式 REPL")
    parser.add_argument("--no-tools", action="store_true", help="不带工具，纯对话模式")
    parser.add_argument("--no-memory", action="store_true", help="不启用长期记忆")
    args = parser.parse_args()

    # 创建共享的 EmbeddingStore，同时传给 Agent(ToolRegistry) 和 LTM
    embedding_store = EmbeddingStore()

    ltm = None if args.no_memory else LongTermMemory(embedding_store=embedding_store)
    plan_mgr = PlanManager()

    tools = [] if args.no_tools else create_demo_tools(plan_mgr=plan_mgr, ltm=ltm)

    agent = Agent(
        tools=tools, system_prompt=SYSTEM_PROMPT,
        long_term_memory=ltm, plan_mgr=plan_mgr,
        tool_top_k=5, embedding_store=embedding_store,
    )

    _print_startup(agent, ltm, tools)

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


def _print_startup(agent: Agent, ltm, tools: list):
    print("🤖 Agent CLI — 输入消息开始对话")
    print("   /exit 退出  /clear 清空历史  /history 查看历史  /memories 查看记忆  /plan <任务> 手动计划  /help 帮助")
    print(f"   🔗 Embedding: BAAI/bge-small-zh-v1.5 (collections: {agent.tools._embedding.list_collections()})")
    if ltm:
        print(f"   🧠 长期记忆已启用 (已有 {len(ltm.list_all())} 条记忆)")
        print("   /remember <内容>  手动存储记忆  /forget <序号>  删除记忆")
    if agent.plan_mgr.is_active:
        print(f"   📋 活跃计划: {agent.plan_mgr.active['task']}")
    if tools:
        tools_list = ", ".join(t["name"] for t in tools)
        print(f"   可用工具: {tools_list} (过滤 top_k={agent.tool_top_k})" if agent.tool_top_k else f"   可用工具: {tools_list}")
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
        elif agent.ltm is None:
            print("长期记忆未启用，请不带 --no-memory 启动。\n")
        else:
            agent.ltm.add(parts[1])
            print("记忆已保存。\n")

    elif action == "/forget":
        if agent.ltm is None:
            print("长期记忆未启用。\n")
        else:
            try:
                idx = int(parts[1]) if len(parts) > 1 else -1
                agent.ltm.forget(idx)
                print(f"已删除记忆 #{idx}。\n")
            except (ValueError, IndexError):
                print("用法: /forget <序号>，用 /memories 查看序号。\n")

    elif action == "/memories":
        if agent.ltm is None:
            print("长期记忆未启用。\n")
        else:
            all_mem = agent.ltm.list_all()
            if not all_mem:
                print("(无长期记忆)\n")
            else:
                for m in all_mem:
                    print(f"  [{m['index']}] {m['content']}  ({m['timestamp']})")
                print()

    elif action == "/consolidate":
        if agent.ltm is None:
            print("长期记忆未启用。\n")
        else:
            before = len(agent.ltm.list_all())
            removed = agent.ltm.consolidate(agent.llm)
            after = len(agent.ltm.list_all())
            print(f"记忆合并完成：{before} → {after}（合并了 {removed} 条）\n")

    elif action == "/plan":
        if len(parts) < 2:
            print("用法: /plan <任务描述>\n")
        else:
            task = parts[1]
            if agent.plan_mgr.is_active:
                print("(旧计划已归档)")
                agent.plan_mgr.clear()
            print(f"开始规划：{task}")
            reply = agent.chat(
                f"请为以下任务制定分步计划，调用 make_plan 工具：{task}"
            )
            print(f"🤖 Agent: {reply}\n")

    elif action == "/help":
        print("命令列表:")
        print("  /exit         退出程序")
        print("  /clear        清空对话历史")
        print("  /history      查看历史消息与统计")
        if agent.ltm:
            print("  /remember     手动存储长期记忆")
            print("  /forget       删除长期记忆")
            print("  /memories     查看所有长期记忆")
            print("  /consolidate  LLM 合并清理长期记忆")
        print("  /plan         手动创建计划（强制覆盖旧计划）")
        print("  /help         显示此帮助\n")

    else:
        print(f"未知命令: {action}，输入 /help 查看帮助\n")


def _show_history(agent: Agent):
    msgs = agent.memory.get_messages()
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
    print(f"\n  📊 {agent.memory.stats()}\n")


if __name__ == "__main__":
    main()
