#!/usr/bin/env bash
# Sharded-CP (DSA context parallel) test driver for the dsa-cp-v0.26 branch.
#
# Usage (from the vLLM repo root, on a GPU server):
#   bash tools/sharded_cp/run_sharded_cp_tests.sh [stage ...]
#
# Stages (default: setup lint unit dist):
#   setup  - create .venv via uv and install vLLM (precompiled wheel)
#   lint   - ruff undefined-name / syntax checks on the Sharded-CP files
#   unit   - CPU pytest: token-range utils + config validation
#   dist   - torchrun multi-GPU unit tests: collectives round-trip,
#            compact-KV round-trip, shard-linear broadcast/prefetch
#   e2e    - offline generation parity: flag OFF vs ON (needs MODEL_PATH)
#
# Environment:
#   TP_SIZE      number of GPUs / CP degree (default 2)
#   MODEL_PATH   DeepSeek-V3.2 style DSA checkpoint dir (e2e stage only)
#   MAX_TOKENS   generated tokens per prompt in e2e (default 32)
#   MAX_MODEL_LEN  optional max_model_len override for e2e
#   ALLOW_MISMATCH=1  do not fail the e2e stage on token mismatches
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
REPO_ROOT=$(pwd)
TP_SIZE=${TP_SIZE:-2}
PY=.venv/bin/python
STAGES=("$@")
if [ ${#STAGES[@]} -eq 0 ]; then
    STAGES=(setup lint unit dist)
fi

log() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }

has_stage() {
    local s
    for s in "${STAGES[@]}"; do [ "$s" = "$1" ] && return 0; done
    return 1
}

if has_stage setup; then
    log "setup: uv venv + editable install"
    command -v uv > /dev/null || {
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    }
    [ -x "$PY" ] || uv venv --python 3.12
    if ! "$PY" -c 'import vllm' 2> /dev/null; then
        VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
    fi
    "$PY" -c 'import vllm; print("vllm", vllm.__version__)'
fi

if has_stage lint; then
    log "lint: ruff F821/E9 on Sharded-CP files"
    "$PY" -m pip show ruff > /dev/null 2>&1 || uv pip install ruff
    "$PY" -m ruff check --select F821,E9 \
        vllm/distributed/sharded_cp_utils.py \
        vllm/distributed/sharded_cp_compact_kv.py \
        vllm/v1/attention/backends/mla/sharded_cp_metadata.py \
        vllm/model_executor/layers/sharded_cp_shard_linear.py \
        vllm/model_executor/layers/fused_moe/sharded_cp_moe.py \
        vllm/model_executor/layers/mla.py \
        vllm/model_executor/models/deepseek_v2.py \
        vllm/v1/worker/gpu_model_runner.py \
        vllm/config/vllm.py vllm/config/parallel.py
    log "lint: import smoke test"
    "$PY" - << 'EOF'
import vllm.distributed.sharded_cp_utils
import vllm.distributed.sharded_cp_compact_kv
import vllm.v1.attention.backends.mla.sharded_cp_metadata
import vllm.model_executor.layers.sharded_cp_shard_linear
import vllm.model_executor.layers.fused_moe.sharded_cp_moe
import vllm.model_executor.layers.mla
import vllm.model_executor.models.deepseek_v2
import vllm.model_executor.models.deepseek_mtp
import vllm.v1.worker.gpu_model_runner
print("import smoke OK")
EOF
fi

if has_stage unit; then
    log "unit: CPU pytest (token-range utils + config validation)"
    "$PY" -m pip show pytest > /dev/null 2>&1 || uv pip install pytest
    "$PY" -m pytest -q \
        tests/distributed/test_sharded_cp_utils.py \
        tests/config/test_sharded_context_parallel.py
fi

if has_stage dist; then
    log "dist: torchrun x${TP_SIZE} multi-GPU unit tests"
    NGPU=$("$PY" -c 'import torch; print(torch.cuda.device_count())')
    if [ "$NGPU" -lt "$TP_SIZE" ]; then
        echo "SKIP: need $TP_SIZE GPUs, found $NGPU" >&2
    else
        "$PY" -m torch.distributed.run --standalone \
            --nproc-per-node="$TP_SIZE" \
            tools/sharded_cp/test_dist_sharded_cp.py
    fi
fi

if has_stage e2e; then
    log "e2e: generation parity flag OFF vs ON (tp=${TP_SIZE})"
    : "${MODEL_PATH:?e2e stage requires MODEL_PATH=/path/to/DeepSeek-V3.2}"
    OUT_DIR=$(mktemp -d /tmp/sharded_cp_e2e.XXXXXX)
    MAX_TOKENS=${MAX_TOKENS:-32}

    run_gen() {
        # $1: off|on   $2: extra flag
        VLLM_ATTENTION_BACKEND=FLASHMLA_SPARSE "$PY" \
            tools/sharded_cp/e2e_generate.py \
            --model "$MODEL_PATH" --tp "$TP_SIZE" \
            --max-tokens "$MAX_TOKENS" \
            ${MAX_MODEL_LEN:+--max-model-len "$MAX_MODEL_LEN"} \
            --output "$OUT_DIR/$1.json" $2
    }
    run_gen off ""
    run_gen on "--enable-sharded-context-parallel"

    log "e2e: comparing outputs"
    "$PY" - "$OUT_DIR/off.json" "$OUT_DIR/on.json" << 'EOF'
import json
import os
import sys

off = json.load(open(sys.argv[1]))
on = json.load(open(sys.argv[2]))
assert len(off) == len(on)
mismatched = 0
for i, (a, b) in enumerate(zip(off, on)):
    ta, tb = a["token_ids"], b["token_ids"]
    if ta == tb:
        print(f"[prompt {i}] MATCH ({len(ta)} tokens)")
        continue
    mismatched += 1
    div = next(
        (j for j, (x, y) in enumerate(zip(ta, tb)) if x != y),
        min(len(ta), len(tb)),
    )
    print(f"[prompt {i}] MISMATCH at token {div}")
    print(f"  off: ...{ta[max(0, div - 3) : div + 3]}")
    print(f"  on : ...{tb[max(0, div - 3) : div + 3]}")
    print(f"  off text: {a['text'][:120]!r}")
    print(f"  on  text: {b['text'][:120]!r}")
print(f"\n{len(off) - mismatched}/{len(off)} prompts matched exactly")
if mismatched and os.environ.get("ALLOW_MISMATCH") != "1":
    # BF16 sparse attention is not bitwise stable across parallel layouts;
    # a small drift can be acceptable. Inspect the diffs, or set
    # ALLOW_MISMATCH=1 to not fail this stage.
    sys.exit(1)
EOF
    echo "e2e artifacts: $OUT_DIR"
fi

log "all requested stages done"
