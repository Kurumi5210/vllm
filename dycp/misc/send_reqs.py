import asyncio, httpx, json, time

URL = "http://0.0.0.0:8400/v1/completions"

# 构造恰好 N token 的 prompt（空格分词≈1 token）
def make_prompt(n: int) -> str:
    # 每个 "hello" 占 1 token，中间空格也占 1 token，因此 (n//2) 个单词即可
    return " ".join(["hello"] * (n // 2))

async def req(payload: dict, name: str):
    async with httpx.AsyncClient(timeout=30) as client:
        t0 = time.perf_counter()
        r = await client.post(URL, json=payload)
        r.raise_for_status()
        cost = time.perf_counter() - t0
        text = r.json()["choices"][0]["text"]
        print(f"[{name}] 耗时 {cost:.2f}s → {text!r}")

async def main():
    # 两条请求参数
    p2k = make_prompt(256)
    p4k = make_prompt(640)

    tasks = [
        req(
            {
                "model": "auto",
                "prompt": p2k,
                "max_tokens": 10,
                "temperature": 0,
                "top_p": 1.0,
            },
            "2k-prompt",
        ),
        req(
            {
                "model": "auto",
                "prompt": p4k,
                "max_tokens": 10,
                "temperature": 0,
                "top_p": 1.0,
            },
            "4k-prompt",
        ),
        req(
            {
                "model": "auto",
                "prompt": p4k,
                "max_tokens": 10,
                "temperature": 0,
                "top_p": 1.0,
            },
            "4k-prompt-2",
        ),
        req(
            {
                "model": "auto",
                "prompt": p4k,
                "max_tokens": 10,
                "temperature": 0,
                "top_p": 1.0,
            },
            "4k-prompt-3",
        ),
        req(
            {
                "model": "auto",
                "prompt": p2k,
                "max_tokens": 10,
                "temperature": 0,
                "top_p": 1.0,
            },
            "2k-prompt-3",
        ),
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
