# local-inference — 本地模型推理侧实验

在本地部署开源 LLM，跑通单轮推理。这是 llm-lab 第一个「模型侧」子项目——第一次直接接触模型本身，而非通过托管 API 调用。

## 环境

- 部署工具：Ollama（本地服务，默认 `http://localhost:11434`）
- 模型：Qwen2.5-3B-Instruct（Q4_K_M，1.9GB）
- 硬件：RTX 3060 Laptop 6GB 显存，模型 100% GPU 运行

## 使用

```bash
# 交互式对话
ollama run qwen2.5:3b

# 单轮 I/O 测试（OpenAI 兼容接口）
pip install -r requirements.txt
python scripts/test_io.py

# KV cache 显存计算（纯本地，无需服务）
python scripts/kv_cache_calc.py

# KV cache 显存实测（需 Ollama + nvidia-smi）
python scripts/kv_cache_measure.py
```

## 注意

- 本机有 Clash 代理（`all_proxy`），脚本已用 `trust_env=False` 直连本地，勿删该参数。
- 模型空闲 5 分钟自动卸载（`keep_alive` 默认）。
