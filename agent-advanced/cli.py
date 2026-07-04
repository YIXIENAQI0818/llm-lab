#!/usr/bin/env python3
"""Agent 交互式 REPL — 像对话一样使用 Agent。

用法:
    python cli.py              # 使用默认工具启动
    python cli.py --no-tools   # 不带工具，纯对话模式
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.agent_framework import Agent, LongTermMemory

# 清洗终端输入时可能产生的代理字符碎片（WSL 删除中文时的残留）
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


# ---- 示例工具 ----
def get_weather(city: str) -> str:
    w = {
        "北京": {"temp": 25, "desc": "晴朗"},
        "上海": {"temp": 28, "desc": "多云"},
        "东京": {"temp": 22, "desc": "小雨"},
        "纽约": {"temp": 15, "desc": "阴天"},
    }
    for k, v in w.items():
        if k in city:
            return f"温度 {v['temp']}°C, {v['desc']}"
    return f"未找到 {city} 的天气数据"


def search_web(query: str) -> str:
    db = {
        "特斯拉": "特斯拉当前股价 $245，上季度为 $220。",
        "茅台": "茅台当前股价 ¥1650。",
        "图灵奖": "图灵奖是计算机领域最高荣誉，由ACM于1966年设立。",
        "东京人口": "东京都人口约1400万。",
    }
    for k, v in db.items():
        if k in query or query in k:
            return v
    return f"未找到关于 '{query}' 的结果"


def calculate(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception:
        return "计算错误"


DEMO_TOOLS = [
    {
        "name": "get_weather",
        "description": "查询指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名"}},
            "required": ["city"],
        },
        "fn": get_weather,
    },
    {
        "name": "search_web",
        "description": "搜索网页获取知识或信息",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "搜索关键词"}},
            "required": ["query"],
        },
        "fn": search_web,
    },
    {
        "name": "calculate",
        "description": "执行数学计算",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "数学表达式"}},
            "required": ["expression"],
        },
        "fn": calculate,
    },
]

def save_memory(content: str) -> str:
    """由工具调用入口转为 CLI 层操作，实际函数在 main() 中被替换。"""
    return "记忆已保存"


DEMO_TOOLS = [
    {
        "name": "get_weather",
        "description": "查询指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名"}},
            "required": ["city"],
        },
        "fn": get_weather,
    },
    {
        "name": "search_web",
        "description": "搜索网页获取知识或信息",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "搜索关键词"}},
            "required": ["query"],
        },
        "fn": search_web,
    },
    {
        "name": "calculate",
        "description": "执行数学计算",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "数学表达式"}},
            "required": ["expression"],
        },
        "fn": calculate,
    },
    {
        "name": "save_memory",
        "description": "保存重要信息到长期记忆，当用户告诉你关于自己的事实或偏好时调用。如：名字、喜好、计划等。",
        "parameters": {
            "type": "object",
            "properties": {"content": {"type": "string", "description": "要记住的内容，一句话概括"}},
            "required": ["content"],
        },
        "fn": save_memory,
    },
]

SYSTEM_PROMPT = "你是一个有用的 AI 助手，可以用中文或用户使用的语言回复。需要时使用工具获取信息。当用户告诉你关于自己的重要信息（名字、偏好、计划等）时，主动调用 save_memory 保存。"


def main():
    parser = argparse.ArgumentParser(description="Agent 交互式 REPL")
    parser.add_argument("--no-tools", action="store_true", help="不带工具，纯对话模式")
    parser.add_argument("--no-memory", action="store_true", help="不启用长期记忆")
    args = parser.parse_args()

    # 长期记忆
    ltm = None
    if not args.no_memory:
        ltm = LongTermMemory()
        # 把 save_memory 工具的函数替换为可操作实际 ltm 的版本
        for t in DEMO_TOOLS:
            if t["name"] == "save_memory":
                t["fn"] = lambda content: (ltm.add(content) or "记忆已保存")

    tools = [] if args.no_tools else DEMO_TOOLS
    agent = Agent(tools=tools, system_prompt=SYSTEM_PROMPT, long_term_memory=ltm)

    print("🤖 Agent CLI — 输入消息开始对话")
    print("   /exit 退出  /clear 清空历史  /history 查看历史  /memories 查看记忆  /help 帮助")
    if ltm:
        print(f"   🧠 长期记忆已启用 (已有 {len(ltm.list_all())} 条记忆)")
        print("   /remember <内容>  手动存储记忆  /forget <序号>  删除记忆")
    if not args.no_tools:
        tools_list = ", ".join(t["name"] for t in DEMO_TOOLS)
        print(f"   可用工具: {tools_list}")
    print()

    while True:
        try:
            user_input = _SURROGATE_RE.sub("", input("👤 You: ").strip())
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # 处理命令
        if user_input.startswith("/"):
            _handle_command(user_input, agent)
            continue

        # 正常对话
        try:
            reply = agent.chat(user_input)
            print(f"🤖 Agent: {reply}\n")
        except Exception as e:
            print(f"❌ 错误: {e}\n")


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
            print(f"记忆已保存。\n")
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
    elif action == "/help":
        print("命令列表:")
        print("  /exit       退出程序")
        print("  /clear      清空对话历史")
        print("  /history    查看历史消息与统计")
        if agent.ltm:
            print("  /remember   手动存储长期记忆")
            print("  /forget     删除长期记忆")
            print("  /memories   查看所有长期记忆")
        print("  /help       显示此帮助\n")
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
