
#!/bin/bash

# Example usage (This is a 16 H800 GPUs example with 2 nodes and 8 GPUs per node):
# For node 0: bash run_dsr1_dycp.sh 0
# For node 1: bash run_dsr1_dycp.sh 1

set -x
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <node_rank>"
    echo "<node_rank> should start from 0 for both prefill and decode."
    exit 1
fi
NODE_RANK=$1
EXTRA_PARAMS=$([ $NODE_RANK -ne 0 ] && echo "--headless" || echo "--api-server-count 1")

# rm -rf /root/.cache/vllm/torch_compile_cache/
# export VLLM_LOGGING_LEVEL=DEBUG
# export CUDA_LAUNCH_BLOCKING=1

# max graph size should <= max_num_seqs for decode with cudagraph
# In dycp MAX_SEQS_PER_DP represents Seqs per DP.
MAX_SEQS_PER_DP=64
export VLLM_MOE_DP_CHUNK_SIZE=${MAX_SEQS_PER_DP}
export VLLM_DEEPEP_BUFFER_SIZE_MB=0
export VLLM_USE_DEEP_GEMM=1
export VLLM_ALL2ALL_BACKEND=deepep_low_latency
export VLLM_IGNORE_TENSOR_PLACEHOLDER=1

export NCCL_DEBUG=WARN

export MODEL_PATH=<Path to your model>

export VLLM_USE_V1=1
export COMMON_ARGS="
    --trust-remote-code
    --served-model-name auto
    --model-loader-extra-config {\"enable_multithread_load\":true,\"num_threads\":8}
    --disable-log-requests
"

export VLLM_VERSION=0.13.0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=380
export VLLM_ATTENTION_BACKEND=FLASHMLA
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Profiler setttings
# export VLLM_TORCH_PROFILER_DIR=${VLLM_TORCH_PROFILER_DIR:-"./profiles"}
# rm -rf $VLLM_TORCH_PROFILER_DIR
# mkdir -p $VLLM_TORCH_PROFILER_DIR
# export VLLM_TORCH_PROFILER_WITH_STACK=0

export PYTORCH_ALLOC_CONF=expandable_segments:True
export VLLM_USE_FORCE_LOAD_BALANCE=1
# ========== config vllm ==========
args=(
    --port 8400 
    ${EXTRA_PARAMS} 
    $COMMON_ARGS 
    --async-scheduling 
    --distributed-executor-backend dmp 
    --hf-overrides '{"rope_parameters": {"rope_type":"yarn","factor":8.0,"original_max_position_embeddings":131072}}' 
    --max-model-len 524288 
    --max-num-batched-tokens 128 
    --gpu-memory-utilization 0.9 
    --no-enable-prefix-caching 
    --data-parallel-size 16 
    --tensor-parallel-size 1 
    --data-parallel-size-local 8 
    --data-parallel-address=< Master IP Address> 
    --data-parallel-rpc-port 8400 
    --data-parallel-start-rank $((NODE_RANK * 8)) 
    --block-size 64 
    --cp-kv-cache-interleave-size 64 
    --no-enforce-eager 
    --max-num-seqs ${MAX_SEQS_PER_DP} 
    --enable-expert-parallel 
    --dp-per-domain 8 
    --num-cp-seqs 2 
    --compilation-config '{"cudagraph_capture_sizes":[2, 4, 8, 10, 12, 16, 18, 24, 26, 32, 34, 64], "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes_for_cp": 2}' 
    --kv-transfer-config 
    '{
        "kv_connector": "CrossDPExampleConnector",
        "kv_connector_module_path": "vllm.distributed.kv_transfer.kv_connector.v1.cross_dp_example_connector",
        "kv_role": "kv_consumer",
        "kv_parallel_size": 2,
        "kv_port": "20002",
        "engine_id": "decode-'${NODE_RANK}'",
        "kv_rank": 1,
        "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 1,
                    "tp_size": 16
             },
             "decode": {
                    "dp_size": 8,
                    "tp_size": 1
             }
        }
    }'
)

# ========== execute vllm server ==========
vllm serve "${MODEL_PATH}" "${args[@]}" &> "${NODE_RANK}dycp.log" &
