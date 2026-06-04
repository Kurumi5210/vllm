#!/bin/bash
set -x

# =============================================================================
# PCP32 Ray 模式启动脚本
# =============================================================================
# 拓扑：
#   PP=1, TP=1, PCP=32  → world_size = 32 GPU
#   4 节点 × 8 GPU = 32 GPU，DP=1（无需 data parallel）
#   每个节点承载 8 个 PCP rank，EP 组大小 = 32
#
# 前置条件：
#   1. 所有节点已配置 Ray 集群，确保 head 节点和 worker 节点互通
#      - head 节点:  ray start --head --port=6379
#      - worker 节点: ray start --address=<head_ip>:6379
#   2. 所有节点的 MODEL_PATH 指向同一模型
#   3. 所有节点的 PYTHONPATH 一致
#   4. 在 head 节点（10.11.16.120）上运行本脚本
#
# 对比 mp 模式的关键变化：
#   - 不再需要 NODE_RANK 参数
#   - 不再需要 --headless / --data-parallel-start-rank
#   - 不再需要 --data-parallel-address / --data-parallel-rpc-port
#   - 不再需要 --data-parallel-size / --data-parallel-size-local（32 GPU 全用于 PCP）
#   - 不再需要手动在每个节点启动脚本
#   - 单次启动，Ray 自动管理跨节点的 placement group 和 worker 分布
# =============================================================================

# =============================================================================
# Ray 集群配置
# =============================================================================
# 根据实际环境修改以下地址
RAY_HEAD_IP="${RAY_HEAD_IP:-10.11.16.120}"
RAY_PORT="${RAY_PORT:-6379}"

# 连接到 Ray 集群
export RAY_ADDRESS="${RAY_HEAD_IP}:${RAY_PORT}"

# =============================================================================
# 清理
# =============================================================================
rm -rf /root/.cache/vllm/torch_compile_cache/

# =============================================================================
# 模型路径
# =============================================================================
export MODEL_PATH=/input/chenxiao.cx/model/DeepSeek-R1/model/

# =============================================================================
# vLLM 环境变量
# =============================================================================
export VLLM_USE_V1=1
export VLLM_VERSION=0.13.0

# DeepEP / MoE 配置
export VLLM_DEEPEP_BUFFER_SIZE_MB=0
export VLLM_MOE_DP_CHUNK_SIZE=64
export VLLM_USE_DEEP_GEMM=1
export VLLM_ALL2ALL_BACKEND=deepep_low_latency

# 注意力后端
export VLLM_ATTENTION_BACKEND=FLASHMLA

# 扩展配置
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=380
export VLLM_IGNORE_TENSOR_PLACEHOLDER=1
export VLLM_USE_FORCE_LOAD_BLANCE=1

# PyTorch 内存
export PYTORCH_ALLOC_CONF=expandable_segments:True

# NCCL
export NCCL_DEBUG=WARN

# Python 路径
export PYTHONPATH=/input/chenxiao.cx/codebase/pcp/vllm:$PYTHONPATH
export PYTHONPATH=/input/chenxiao.cx/codebase/temp-code-1-30/tools/ep_kernels/ep_kernels_workspace/DeepEP:$PYTHONPATH

# =============================================================================
# Profiler（可选）
# =============================================================================
export VLLM_TORCH_PROFILER_DIR=${VLLM_TORCH_PROFILER_DIR:-"./profiles"}
rm -rf $VLLM_TORCH_PROFILER_DIR
mkdir -p $VLLM_TORCH_PROFILER_DIR
export VLLM_TORCH_PROFILER_WITH_STACK=0

# =============================================================================
# 系统配置
# =============================================================================
ulimit -n 65536

# =============================================================================
# 通用参数
# =============================================================================
COMMON_ARGS="
    --trust-remote-code
    --served-model-name auto
    --model-loader-extra-config {\"enable_multithread_load\":true,\"num_threads\":8}
    --disable-log-requests
"

# =============================================================================
# 启动 vllm serve（单次启动，Ray 自动管理跨节点分布）
# =============================================================================
# 无 data parallel：world_size=32 已占满 4 节点 × 8 GPU
# Ray 的 PACK placement group 将 32 个 worker 分布到 4 个节点，
# 每个节点 8 个 worker，满足 DeepEP 的 rank [8n, 8n+7] 同节点要求。
# =============================================================================
vllm serve ${MODEL_PATH} \
    --port 8400 \
    --api-server-count 1 \
    $COMMON_ARGS \
    --distributed-executor-backend ray \
    --hf-overrides '{"rope_parameters": {"rope_type":"yarn","factor":8.0,"original_max_position_embeddings":131072}}' \
    --max-model-len 1048576 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.775 \
    --no-enable-prefix-caching \
    --tensor-parallel-size 1 \
    --prefill-context-parallel-size 32 \
    --block-size 64 \
    --cp-kv-cache-interleave-size 64 \
    --enforce-eager \
    --max-num-seqs 128 \
    --enable-expert-parallel \
    &> pcp32_ray.log &
