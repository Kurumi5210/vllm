#!/bin/bash
# save as: pack_vllm.sh

# 进入 vllm 项目根目录（假设当前在 vllm/ 的父目录）
alias ossput="bash ~/osstool.sh put"

# 使用 tar + exclude 打包 vllm/vllm/ 中的源码，排除大文件
gtar --exclude='vllm/vllm/vllm_flash_attn' \
    --exclude='vllm/vllm/*.so' \
    --exclude='vllm/vllm/__pycache__' \
    --exclude='vllm/vllm/build' \
    --exclude='vllm/vllm/third_party/flash-attention' \
    --exclude='vllm/vllm/*.abi3.so' \
    -cvf pcp_src.tar vllm/vllm

# 上传
yes | ossput pcp_src.tar

# 清理
# rm -f vllm_src.tar

echo "✅ vLLM source packed and uploaded (without .so or vllm_flash_attn)"
