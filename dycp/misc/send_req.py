#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调用本地 vLLM 服务：
输入 128 token，输出 10 token
"""

import httpx
import json

# 1. 构造一条恰好 128 token 的 prompt
#    这里用“单词 + 空格”简单凑数，实际可按需替换
TOKEN_CNT = 50
prompt = " ".join(["hello"] * TOKEN_CNT)        # 128 个 hello

# 2. 请求参数
url = "http://0.0.0.0:8400/v1/completions"
payload = {
    "model": "auto",          # 若 /v1/models 返回有具体名称，可替换
    "prompt": prompt,
    "max_tokens": 10,            # 固定输出 10 token
    "temperature": 0.0,          # 确定性采样
    "top_p": 1.0
}

# 3. 发送请求
def main():
    with httpx.Client(timeout=30) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        result = resp.json()

    # 4. 打印结果
    text = result["choices"][0]["text"]
    print("=== 输出（10 token） ===")
    print(text)

if __name__ == "__main__":
    main()
