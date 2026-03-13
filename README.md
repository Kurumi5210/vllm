# vLLM-DyCP
Support Dynamic Context parellem on decode instance on vLLM.
## System Overview
![alt text](./dycp/images/dycp.png)
## Install vLLM
``` shell
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```
## Launch Server
Currently, only the CrossDPExampleConnector is supported for performance testing.
```shell
set -x

export NCCL_DEBUG=WARN
export MODEL_PATH=/weights/DeepSeek-V2-Lite/
export VLLM_USE_V1=1
export COMMON_ARGS="
    --trust-remote-code
    --served-model-name auto
    --model-loader-extra-config {\"enable_multithread_load\":true,\"num_threads\":8}
    --disable-log-requests
"
export VLLM_VERSION=0.13.0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=380

ulimit -n 65536
vllm serve ${MODEL_PATH} \
    --port 8400 \
    $COMMON_ARGS \
    --distributed-executor-backend dmp \
    --max-model-len 163840 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.80 \
    --no-enable-prefix-caching \
    --data-parallel-size 4 \
    --tensor-parallel-size 1 \
    --dp-per-domain 4 \
    --block-size 64 \
    --cp-kv-cache-interleave-size 64 \
    --no-enforce-eager \
    --max-num-seqs 6 \
    --enable-expert-parallel \
    --num-cp-seqs 2 \
    --compilation-config '{"cudagraph_capture_sizes":[6], "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes_for_cp": 2}' \
    --kv-transfer-config \
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
                    "dp_size": 4,
                    "tp_size": 1
             }
        }
    }'
```