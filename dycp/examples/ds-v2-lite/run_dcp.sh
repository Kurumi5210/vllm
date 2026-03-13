set -x
rm -rf /root/.cache/vllm/torch_compile_cache/
export PYTHONPATH=/input/chenxiao.cx/codebase/vllm-dycp:$PYTHONPATH
export NCCL_DEBUG=WARN
# export MODEL_PATH=/input/model_weights/Qwen3-30B-A3B
export MODEL_PATH=/input/model_weights/DeepSeek-V2-Lite
export VLLM_USE_V1=1
export COMMON_ARGS="
    --trust-remote-code
    --served-model-name auto
    --model-loader-extra-config {\"enable_multithread_load\":true,\"num_threads\":8}
    --disable-log-requests
"
export VLLM_VERSION=0.13.0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=380
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_TORCH_PROFILER_DIR=${VLLM_TORCH_PROFILER_DIR:-"./profiles"}
rm -rf $VLLM_TORCH_PROFILER_DIR
mkdir -p $VLLM_TORCH_PROFILER_DIR
export VLLM_TORCH_PROFILER_WITH_STACK=0
export VLLM_USE_FORCE_LOAD_BALANCE=1
export VLLM_ALL2ALL_BACKEND=allgather_reducescatter

# max graph size should <= max_num_seqs for decode with cudagraph
MAX_SEQS_PER_DP=1024

# ========== config vllm ==========
args=(
    --async-scheduling 
    --port 8400 
    $COMMON_ARGS 
    --distributed-executor-backend mp 
    --hf-overrides '{"rope_parameters": {"rope_type":"yarn","factor":8.0,"original_max_position_embeddings":163840}}' 
    --max-model-len 1048576 
    --max-num-batched-tokens 4096 
    --gpu-memory-utilization 0.8 
    --no-enable-prefix-caching 
    --data-parallel-size 1 
    --tensor-parallel-size 8 
    --decode-context-parallel-size 8 
    --block-size 64 
    --cp-kv-cache-interleave-size 64
    --no-enforce-eager
    --max-num-seqs ${MAX_SEQS_PER_DP} 
    --enable-expert-parallel
    --compilation-config '{"cudagraph_capture_sizes":[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], "cudagraph_mode": "FULL_DECODE_ONLY"}'
    --kv-transfer-config
    '{
        "kv_connector": "ExampleConnector",
        "kv_connector_module_path": "vllm.distributed.kv_transfer.kv_connector.v1.example_connector",
        "kv_role": "kv_consumer",
        "kv_parallel_size": 2,
        "kv_port": "20002",
        "engine_id": "decode-0",
        "kv_rank": 1,
        "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 1,
                    "tp_size": 16
             },
             "decode": {
                    "dp_size": 1,
                    "tp_size": 8
             }
        }
    }'
)

# ========== execute vllm server ==========
vllm serve "${MODEL_PATH}" "${args[@]}" &> "ds_dcp.log" &
