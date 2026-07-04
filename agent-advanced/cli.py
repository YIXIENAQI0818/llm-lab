#!/usr/bin/env python3
"""Agent 交互式 REPL — 像对话一样使用 Agent。

用法:
    python cli.py              # 使用默认工具启动
    python cli.py --no-tools   # 不带工具，纯对话模式
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.agent_framework import Agent


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

SYSTEM_PROMPT = "你是一个有用的 AI 助手，可以用中文或用户使用的语言回复。需要时使用工具获取信息。"


def main():
    parser = argparse.ArgumentParser(description="Agent 交互式 REPL")
    parser.add_argument("--no-tools", action="store_true", help="不带工具，纯对话模式")
    args = parser.parse_args()

    tools = [] if args.no_tools else DEMO_TOOLS
    agent = Agent(tools=tools, system_prompt=SYSTEM_PROMPT)

    print("🤖 Agent CLI — 输入消息开始对话")
    print("   /exit 退出  /clear 清空历史  /history 查看历史  /help 帮助")
    if not args.no_tools:
        tools_list = ", ".join(t["name"] for t in DEMO_TOOLS)
        print(f"   可用工具: {tools_list}")
    print()

    while True:
        try:
            user_input = input("👤 You: ").strip()
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
    elif action == "/help":
        print("命令列表:")
        print("  /exit     退出程序")
        print("  /clear    清空对话历史")
        print("  /history  查看历史消息与统计")
        print("  /help     显示此帮助\n")
    else:
        print(f"未知命令: {action}，输入 /help 查看帮助\n")


if __name__ == "__main__":
    main()
