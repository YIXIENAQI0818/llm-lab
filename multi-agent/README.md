# multi-agent

多 Agent 协作实验。学习 Orchestrator + Worker 架构、任务委派与并行执行。

## 架构

```
User → Orchestrator（全能副手）
          ├── 简单任务自己处理
          ├── 复杂任务 delegate_task → Worker
          └── Worker 按需创建（researcher / programmer），用完 GC
```

## 运行

```bash
pip install -r requirements.txt
cp .env.example .env  # 填入 DEEPSEEK_API_KEY
python src/cli_orch.py   # Orchestrator 模式
python src/cli.py         # 单 Agent 模式
```
