set -x

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
# Profiler setttings
# rm -rf $VLLM_TORCH_PROFILER_DIR
# mkdir -p $VLLM_TORCH_PROFILER_DIR
# export VLLM_TORCH_PROFILER_WITH_STACK=0

export VLLM_USE_FORCE_LOAD_BALANCE=1
export VLLM_ALL2ALL_BACKEND=allgather_reducescatter

# max graph size should <= max_num_seqs for decode with cudagraph
MAX_SEQS_PER_DP=128

# ========== config vllm ==========
args=(
    --port 8400 
    $COMMON_ARGS 
    --distributed-executor-backend dmp 
    --hf-overrides '{"rope_parameters": {"rope_type":"yarn","factor":8.0,"original_max_position_embeddings":40960}}' 
    --max-model-len 327680 
    --max-num-batched-tokens 4096 
    --gpu-memory-utilization 0.80 
    --no-enable-prefix-caching 
    --data-parallel-size 2 
    --tensor-parallel-size 4 
    --dp-per-domain 2 
    --block-size 64 
    --cp-kv-cache-interleave-size 64 
    --no-enforce-eager 
    --max-num-seqs ${MAX_SEQS_PER_DP} 
    --num-cp-seqs 2 
    --enable-expert-parallel 
    --compilation-config '{"cudagraph_capture_sizes":[2, 4, 8, 16, 32, 64, 128], "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes_for_cp": 2}' 
    --kv-transfer-config 
    '{
        "kv_connector": "CrossDPExampleConnector",
        "kv_connector_module_path": "vllm.distributed.kv_transfer.kv_connector.v1.cross_dp_example_connector",
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
                    "dp_size": 2,
                    "tp_size": 4
             }
        }
    }'
)

# ========== execute vllm server ==========
vllm serve "${MODEL_PATH}" "${args[@]}" &> "qwen_dycp.log" &
