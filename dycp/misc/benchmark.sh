set -x

export VLLM_TORCH_PROFILER_WITH_STACK=0

RESULT_DIR="/tmp/theta_workdir/bench-results"
MODEL_PATH=/weights/DeepSeek-V2-Lite/

mkdir -p ${RESULT_DIR}

# Online - 4k/1.5k/3s/70ms
export INPUT_LEN=4000
export OUTPUT_LEN=1024
export NUM_PROMPTS=50
export MAX_CONCURRENCY=25
export REQUEST_RATE=inf

ulimit -n 65536
vllm bench serve \
    --backend openai-chat \
    --dataset-name random \
    --trust-remote-code \
    --served-model-name auto \
    --model ${MODEL_PATH} \
    --random-input-len ${INPUT_LEN} \
    --random-output-len ${OUTPUT_LEN} \
    --num-prompts ${NUM_PROMPTS} \
    --max-concurrency ${MAX_CONCURRENCY} \
    --request-rate ${REQUEST_RATE} \
    --ignore-eos \
    --metric-percentiles "50,90,99" \
    --host localhost \
    --port 8400 \
    --save-result \
    --result-dir ${RESULT_DIR} \
    --endpoint /v1/chat/completions \
    --temperature 0.6
