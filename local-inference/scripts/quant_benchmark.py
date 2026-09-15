"""量化档位对比 benchmark：体积 / 显存 / 速度。

对比同一模型不同量化档位的三个客观指标：
- 体积：直接读 `ollama list`
- 显存：加载后读 nvidia-smi 的显存增量（num_ctx 固定）
- 速度：固定 num_predict 生成，读 eval_duration 算 tok/s，多次取平均

用法:
    python scripts/quant_benchmark.py [模型 tag ...]

默认对比 qwen2.5:3b (Q4_K_M) / qwen2.5:3b-q8_0 / qwen2.5:3b-q2_k。
前置: Ollama 已启动，各模型已 pull/create，需要 nvidia-smi。
"""

import subprocess
import sys
import time

import httpx

BASE = "http://localhost:11434"
# trust_env=False 直连本地，避免 Clash 劫持（同 test_io.py）
CLIENT = httpx.Client(base_url=BASE, timeout=180.0, trust_env=False)

NUM_CTX = 4096        # 固定上下文，保证显存对比公平
NUM_PREDICT = 256     # 固定生成长度，保证速度对比公平
SPEED_RUNS = 3        # 速度测几次取平均，抵消系统抖动
SPEED_PROMPT = "请用中文详细介绍量子计算机的工作原理，分三点说明，每点至少三句话，尽量详细。"


def vram_used_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    )
    return int(out.decode().strip())


def stop_all(models):
    for m in models:
        subprocess.run(["ollama", "stop", m], capture_output=True)
    time.sleep(3)  # 等显存真正释放


def measure_vram(model):
    """加载 model，返回显存增量（相对空载基线，MiB）。"""
    baseline = vram_used_mib()
    CLIENT.post("/api/chat", json={
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "options": {"num_ctx": NUM_CTX},
        "keep_alive": "10m",
    })
    time.sleep(8)  # 等权重 + KV cache 全部载入显存
    return vram_used_mib() - baseline


def measure_speed(model):
    """固定 num_predict 生成，测 decode 速度（tok/s），取 SPEED_RUNS 次平均。"""
    speeds = []
    for _ in range(SPEED_RUNS):
        r = CLIENT.post("/api/chat", json={
            "model": model,
            "messages": [{"role": "user", "content": SPEED_PROMPT}],
            "stream": False,
            "options": {"num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
        })
        d = r.json()
        ec = d.get("eval_count", 0)
        ed = d.get("eval_duration", 1) / 1e9  # 纳秒 → 秒
        if ec:
            speeds.append(ec / ed)
    return sum(speeds) / len(speeds) if speeds else 0.0


def main():
    models = sys.argv[1:] or ["qwen2.5:3b", "qwen2.5:3b-q8_0", "qwen2.5:3b-q2_k"]

    print("=== 体积（ollama list）===")
    subprocess.run(["ollama", "list"])
    print(f"\n=== 显存 + 速度（num_ctx={NUM_CTX}, num_predict={NUM_PREDICT}, "
          f"速度取 {SPEED_RUNS} 次平均）===")
    print(f"{'模型':<20} {'显存':>10} {'速度':>12}")
    print("-" * 46)

    for model in models:
        stop_all(models)
        vram = measure_vram(model)
        speed = measure_speed(model)
        print(f"{model:<20} {vram:>8} MiB {speed:>9.1f} tok/s")

    stop_all(models)  # 测完卸载，释放显存


if __name__ == "__main__":
    main()
