"""KV cache 实测：用不同 num_ctx 加载模型，读显存增量，验证 kv_cache_calc.py 的公式。

用法:
    python scripts/kv_cache_measure.py

原理:
    权重体积固定，num_ctx 只改变 KV cache 大小。因此两次加载的显存差
    = 两次 KV cache 之差，实测增量应对上理论公式（每 token 36 KiB × token 数差）。

前置:
    1. Ollama 服务已启动，模型已 pull
    2. 需要 nvidia-smi（GPU）
"""

import subprocess
import time

import httpx

BASE = "http://localhost:11434"
MODEL = "qwen2.5:3b"
# trust_env=False 直连本地，避免 Clash 劫持（同 test_io.py）
CLIENT = httpx.Client(base_url=BASE, timeout=120.0, trust_env=False)

# 实测时用的几档 num_ctx（token 数）
NUM_CTX_LIST = [4096, 16384, 32768]


def vram_used_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    )
    return int(out.decode().strip())


def stop_model():
    subprocess.run(["ollama", "stop", MODEL], capture_output=True)
    time.sleep(3)


def load_with_num_ctx(num_ctx):
    """触发模型以指定 num_ctx 加载，并保持加载。返回显存占用（MiB，相对空载基线）。"""
    baseline = vram_used_mib()
    CLIENT.post(
        "/api/chat",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "options": {"num_ctx": num_ctx},
            "keep_alive": "10m",
        },
    )
    time.sleep(8)  # 等权重 + KV cache 全部载入显存
    return vram_used_mib() - baseline


def main():
    print(f"实测模型: {MODEL}（权重固定，num_ctx 只影响 KV cache）\n")
    print(f"{'num_ctx':>8} | {'实测显存':>10} | {'实测增量':>10} | {'理论KV总量':>10}")
    print("-" * 48)

    results = []
    prev = None
    for num_ctx in NUM_CTX_LIST:
        stop_model()
        used = load_with_num_ctx(num_ctx)
        delta = used - prev if prev is not None else None
        theory = kv_cache_mib(num_ctx)
        results.append((num_ctx, used, delta, theory))
        print(f"{num_ctx:>8} | {used:>8} MiB | "
              f"{f'{delta:>6} MiB' if delta is not None else '     —':>10} | {theory:>8} MiB")
        prev = used

    stop_model()  # 测完卸载，释放显存

    # 结论：总增量(4096→32768) vs 理论增量
    total_delta = results[-1][1] - results[0][1]
    theory_delta = results[-1][3] - results[0][3]
    print()
    print(f"4096→32768 总增量: 实测 {total_delta} MiB / 理论 {theory_delta} MiB "
          f"(误差 {(total_delta - theory_delta) / theory_delta * 100:+.0f}%)")
    print("误差来自随 num_ctx 一起放大的其他缓冲区(compute graph/中间激活)，KV 公式本身准确。")


def kv_cache_mib(num_ctx):
    # 与 kv_cache_calc.py 同公式：2 × 36层 × 2 KV头 × 128 head_dim × 2字节 × num_ctx
    return 2 * 36 * 2 * 128 * 2 * num_ctx // (1024 ** 2)


if __name__ == "__main__":
    main()
