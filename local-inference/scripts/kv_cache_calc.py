"""KV cache 显存计算：按 Qwen2.5-3B 真实架构参数，算不同上下文长度下的 KV cache 大小。

用法:
    python scripts/kv_cache_calc.py

不依赖任何服务，纯本地计算。架构参数来自 `ollama show qwen2.5:3b`（或 /api/show）。
"""

# Qwen2.5-3B-Instruct 架构参数（qwen2）
N_LAYERS = 36            # block_count
N_HEAD_Q = 16            # attention.head_count（查询头）
N_HEAD_KV = 2            # attention.head_count_kv（KV 头）—— GQA 的关键：KV 头远少于 Q 头
HIDDEN = 2048            # embedding_length
HEAD_DIM = HIDDEN // N_HEAD_Q  # 128，每个头的维度
MAX_CONTEXT = 32768      # context_length（模型上限）

BYTES_FP16 = 2
BYTES_FP8 = 1

# Q4_K_M 权重体积（实测约 1.9GB），用于估算总显存
WEIGHTS_BYTES = 1.9 * 1024 ** 3


def kv_cache_bytes(n_ctx, n_head_kv=N_HEAD_KV, bytes_per_val=BYTES_FP16):
    """KV cache 总字节数 = 2(K,V) × L层 × H_kv头 × head_dim × n_ctx × 每值字节数。"""
    return 2 * N_LAYERS * n_head_kv * HEAD_DIM * n_ctx * bytes_per_val


def mib(n_bytes):
    return n_bytes / 1024 ** 2


def main():
    per_token = kv_cache_bytes(1)                     # 每 token 的 KV cache
    per_token_mha = kv_cache_bytes(1, N_HEAD_Q)       # 假设不用 GQA、KV 头 = Q 头

    print(f"架构: Qwen2.5-3B (L={N_LAYERS}, head_dim={HEAD_DIM}, KV头={N_HEAD_KV})")
    print(f"每 token KV cache (fp16): {per_token} B ≈ {per_token / 1024:.0f} KiB")
    print(f"GQA 相比 MHA({N_HEAD_Q} 个 KV 头): {per_token_mha} B/token，"
          f"缩小 {N_HEAD_Q // N_HEAD_KV}×")
    print()

    # 不同上下文长度 + 精度的 KV cache 表
    print("KV cache 随上下文长度/精度变化：")
    print(f"{'num_ctx':>8} | {'fp16':>10} | {'fp8':>10} | {'权重+KV(fp16)':>14}")
    print("-" * 52)
    for n_ctx in [1024, 2048, 4096, 8192, 16384, 32768]:
        f16 = kv_cache_bytes(n_ctx)
        f8 = kv_cache_bytes(n_ctx, bytes_per_val=BYTES_FP8)
        total = WEIGHTS_BYTES + f16
        print(f"{n_ctx:>8} | {mib(f16):>8.0f} MiB | {mib(f8):>8.0f} MiB | "
              f"{mib(total) / 1024:>11.1f} GiB")

    print()
    print(f"结论: 6GB 显存下，fp16 KV cache 即使拉到 {MAX_CONTEXT} 上限也只占 "
          f"{mib(kv_cache_bytes(MAX_CONTEXT)) / 1024:.1f} GiB；真正省下显存靠的是 GQA"
          f"（KV 头 2 vs 16，省 8×）和 fp8 量化（再省一半）。")


if __name__ == "__main__":
    main()
