"""agent-extension 交互式命令行入口。

用法:
    cd agent-extension
    python -m src.cli

命令:
    /reset      重置对话
    /reindex    重建所有索引（KB + LTM + Tools）
    /stats      查看对话统计（消息数、token 数）
    /tools      列出当前注册的所有工具
    /help       显示帮助
    /exit, /quit  退出
"""

import re
import sys
import traceback

from .agent_framework.core import Agent

# 清洗 WSL 终端操作中文输入时产生的代理字符碎片
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _print_banner():
    print("=" * 60)
    print("  Agent Extension — 单 Agent + Skill + MCP 扩展框架")
    print("  输入 /help 查看命令，输入 /exit 退出")
    print("=" * 60)
    print()


def _print_help():
    print()
    print("命令:")
    print("  /reset      重置对话（清空上下文，重新开始）")
    print("  /reindex    重建所有索引（KB 文档 + LTM 记忆 + Tools）")
    print("  /stats      查看当前对话统计")
    print("  /tools      列出已注册工具")
    print("  /help       显示此帮助")
    print("  /exit       退出程序")
    print()


def _handle_command(agent: Agent, cmd: str) -> bool:
    """处理 / 开头的特殊命令。返回 True 表示继续，False 表示退出。"""
    cmd = cmd.strip()

    if cmd in ("/exit", "/quit"):
        print("再见！")
        return False

    if cmd == "/help":
        _print_help()
        return True

    if cmd == "/reset":
        agent.reset()
        print("[OK] 对话已重置")
        return True

    if cmd == "/reindex":
        print("重建索引中...")
        try:
            r1 = agent.reindex_kb(force=True)
            print(f"  KB: {r1}")
            agent.reindex_memories(force=True)
            print("  LTM: 已重建")
            agent.reindex_tools(force=True)
            print("  Tools: 已重建")
        except Exception as e:
            print(f"  [错误] {e}")
        print("[OK] 重建完成")
        return True

    if cmd == "/stats":
        stats = agent.cm.stats()
        n_tools = len(agent.tr)
        print()
        print(f"  消息数:  {stats['n_messages']}")
        print(f"  Token 数: {stats['tokens']}")
        print(f"  摘要 Token: {stats['summary_tokens']}")
        print(f"  注册工具: {n_tools}")
        print(f"  活跃计划: {'是' if agent.pm.is_active else '否'}")
        print(f"  LTM 条数: {len(agent.ltm.list_all())}")
        print()
        return True

    if cmd == "/tools":
        tools = agent.tr.list_tools()
        print()
        if tools:
            for i, name in enumerate(tools, 1):
                print(f"  {i}. {name}")
            print(f"\n  共 {len(tools)} 个工具")
        else:
            print("  (无)")
        print()
        return True

    print(f"未知命令: {cmd}（输入 /help 查看帮助）")
    return True


def main():
    _print_banner()

    print("正在初始化 Agent...")
    try:
        agent = Agent()
    except Exception as e:
        print(f"\n[错误] Agent 初始化失败: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"[OK] Agent 就绪，{len(agent.tr)} 个工具已注册")
    print()

    try:
        while True:
            try:
                user_input = _SURROGATE_RE.sub("", input(">>> ").strip())
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not user_input:
                continue

            # 特殊命令
            if user_input.startswith("/"):
                if not _handle_command(agent, user_input):
                    break
                continue

            # 正常对话
            try:
                print()
                response = agent.chat(user_input, verbose=True)
                print(f"\n🤖 {response}\n")
            except Exception as e:
                print(f"\n[错误] {e}\n")
                traceback.print_exc()

    finally:
        # 清理 MCP 子进程
        try:
            agent.tr.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
