"""单轮 I/O 测试：通过 OpenAI 兼容接口调用本地 Ollama 模型，做一次问答。

用法:
    python scripts/test_io.py

前置:
    1. Ollama 服务已启动（systemd 自启，或 ollama serve）
    2. 模型已 pull：ollama pull qwen2.5:3b
"""

import time

import httpx
from openai import OpenAI

# Ollama 的 OpenAI 兼容端点：本地服务 + 占位 key（本地模型不需要真实 key）
# trust_env=False：忽略系统代理(all_proxy 等)，否则 Clash/V2Ray 会把
# localhost 也劫持走，导致连不上本地服务。
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    http_client=httpx.Client(trust_env=False),
)

MODEL = "qwen2.5:3b"
PROMPT = "给我解释一下排序算法的基本原理。"


def main():
    # 注意：第一次调用会触发模型冷加载（把权重读进显存），耗时较长；
    # 模型已加载（5 分钟 keep_alive 内）则很快。
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
    )
    elapsed = time.perf_counter() - start

    answer = resp.choices[0].message.content
    print(f"模型: {MODEL}")
    print(f"问题: {PROMPT}")
    print(f"回答: {answer}")
    print(f"耗时: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
