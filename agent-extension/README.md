# agent-extension

Agent 扩展框架。从 multi-agent 实验中提取成熟基础设施，重构为单 Agent 优先 + 可扩展工具体系。

## 核心架构

```
Agent → ToolRegistry（唯一对接点）
           ├── 本地工具: 纯函数 + LTM/KB/PM 方法
           ├── Skill:    内部 LLM 循环的复合工具
           └── MCP:      远程 MCP Server 工具
```

## 特性

- **统一工具体系**: 三类工具集中注册、统一索引、统一执行
- **Skill 扩展**: 声明式 Skill 定义，内部 LLM 循环
- **MCP 扩展**: 接 MCP Server 自动发现工具（待完成）
- **扩展不改核心**: 加工具 = 编辑对应模块，Agent 和 Registry 不用改

## 运行

```bash
pip install -r requirements.txt
cp .env.example .env  # 填入 DEEPSEEK_API_KEY
# cli 待完成
```
