# Sharded Context Parallel for DSA Sparse Attention Models

This document describes the vLLM implementation plan for the Sharded Context
Parallelism (Sharded-CP) design from the thesis chapter 5 and the related blog.
The target models are DeepSeek V3.2 / GLM-5 style DSA sparse MLA models. The
main objective is to push context parallelism to one CP rank per device, remove
Indexer duplication inside TP groups, and use Shard Linear to control the memory
cost of full attention projection weights.

This document follows the thesis implementation semantics. It is not an
attention-local MVP that slices tokens only inside attention and immediately
returns to TP-full hidden states after every attention block. When Sharded-CP is
enabled, the Transformer body keeps a CP layout across layers:

```text
Embedding TP
  -> Pad/Reduce-Scatter to CP hidden
  -> repeated Transformer layers on [T_local, hidden_size]
  -> final All-Gather to TP/LMHead
```

## References

- RFC: `vllm-project/vllm#30055`, "Sharded Context Parallelism for DeepSeek
  DSA".
- Blog: `https://zzhx1.github.io/2025/12/04/Sharded-Context-Parallel/`.
- Ascend reference PR: `vllm-project/vllm-ascend#4702`.
- Thesis chapter 5: "Sharded context parallel optimization for sparse attention
  model inference".

The Ascend PR is useful as a proof of shape: it adds a CP context, slices tokens
in attention metadata, replaces MLA up/output projections under CP, and adds
shared-weight broadcast/prefetch. Upstream vLLM should not copy the plugin-style
runtime class replacement from that PR; this design adds explicit hooks in model
construction and forward paths.

## Design Boundary

Target scope:

- Model family: DeepSeek V3.2 style DSA / sparse MLA models, initially detected
  by `hf_config.index_topk`.
- Serving phase: prefill first. Decode can continue to use existing DP/DCP/FGTP
  paths. Mixed decode batches should fail closed until a separate design covers
  them.
- Parallel domain: one GPU/NPU is one CP rank. The first upstream vLLM version
  may reuse the existing tensor-parallel process group as the communication
  group, but the semantic layout is CP.
- Tensor layout: Embedding and LMHead remain TP-style boundaries; attention and
  MoE in the Transformer body run on CP hidden states.
- Optimization target: each device handles only `1 / N` query tokens, reducing
  sparse-attention random KV gathers, Indexer query/scoring/top-k work, and
  per-token preprocessing.

Non-goals:

- Generic CP for all attention backends.
- Reusing or overloading existing vLLM PCP/DCP algorithms.
- Pipeline parallel support.
- Speculative decoding / MTP support.
- Full CUDA graph support before the layout is stable.
- Cross-node communication optimization in the first version.
- Ascend plugin-style layer monkey-patching.

## Current vLLM State

DeepSeek V3.2 sparse MLA support already exists:

- `vllm/model_executor/models/deepseek_v2.py`
  - `DeepseekV2MLAAttention.is_v32 = hasattr(config, "index_topk")`.
  - `q_b_proj` and `kv_b_proj` are `ColumnParallelLinear`.
  - `o_proj` is `RowParallelLinear`.
  - `Indexer` uses `ReplicatedLinear` for `wq_b`, `wk`, and `weights_proj`.
- `vllm/model_executor/layers/mla.py`
  - `MultiHeadLatentAttentionWrapper.forward()` computes `q`, `kv_c_normed`,
    `k_pe`, invokes the V3.2 sparse indexer, then calls `MLAAttention`.
- `vllm/model_executor/layers/attention/mla_attention.py`
  - `MLAAttention.forward()` updates KV cache using
    `get_forward_context().slot_mapping`.
  - Sparse MLA implementations reuse MLA attention metadata.
- `vllm/v1/attention/backends/mla/indexer.py`
  - Builds indexer metadata from the same `CommonAttentionMetadata` used by the
    attention backend.
- `vllm/v1/attention/backends/mla/flashmla_sparse.py`
  - Builds sparse MLA metadata and consumes the top-k buffer written by the
    indexer.

Local API detail: `ColumnParallelLinear` and `RowParallelLinear` already support
`disable_tp=True`. This can serve as the correctness-stage full logical weight
path, but the thesis implementation should eventually reduce persistent memory
for `q_b_proj`/`q_up_proj` and `o_proj` with Shard Linear.

## Problem

DSA's Lightning Indexer selects top-k KV entries for each query token. Traditional
TP shards heads, not token rows, so every TP rank processes the full token
sequence for:

- token-level sparse random KV gathers,
- Indexer `W_qb`, `W_k`, and `W_p` projections,
- Indexer scoring and top-k.

The Indexer is not a standard Megatron-style paired matmul. Its output is
immediately consumed by top-k, so it cannot be efficiently sharded along TP
hidden/head dimensions without introducing extra full-activation communication.
CP is a better fit for DSA: each device owns only local `T / N` query tokens, so
Indexer and sparse-attention work scale down with the CP degree.

Existing CP+TP hybrids still have two limitations:

- A CP rank may contain multiple TP ranks, so Indexer work is still duplicated
  inside the TP group for the same token shard.
- CP degree is constrained by weight memory and is hard to scale to one rank per
  device.

Sharded-CP addresses both with single-device CP and Shard Linear.

## Global Dataflow

Let `T` be the global token count, `N` the device count / CP degree, and `r` the
current rank.

### 1. Embedding TP to CP Hidden

Embedding remains TP/vocab-parallel to reduce vocabulary-weight memory. After
global hidden states are produced, pad and reduce-scatter token rows to CP ranks:

```text
token_ids:       [T]
EmbeddingTP:     [T, hidden_size]
Pad + RS:        [T_local, hidden_size]
```

Use contiguous balanced partitioning:

```text
chunk = ceil(T / N)
start = r * chunk
end = min(start + chunk, T)
padded_end = start + chunk
```

Local tensors are padded to `chunk` rows before communication and trimmed back to
`end - start` rows afterward.

### 2. Transformer Body Stays in CP Layout

Under Sharded-CP, every layer input and output is:

```text
hidden_states_local: [T_local, hidden_size]
positions_local:     [T_local]
metadata_local:      describes local query rows, with global sequence/cache IDs
```

Attention, RMSNorm, gating, and quantization run directly on local token rows.
Layers do not all-gather back to full-token hidden states.

### 3. Restore Global Token Rows Before LMHead

The last layer MoE output is still CP-sharded. Before LMHead, all-gather token
rows:

```text
hidden_states_global = AllGather(hidden_states_local)  # [T, hidden_size]
logits_local_vocab = LMHeadTP(hidden_states_global)    # [T, vocab / tp]
logits = GatherVocab(logits_local_vocab)
```

## Attention Flow

For each DeepSeek V3.2 sparse MLA layer, local input is
`X_local: [T_local, hidden_size]`.

### 1. Local Down-Projections and Indexer Projections

Compute the compact values needed for KV aggregation first:

```text
kv_c_local        = kv_down_proj(X_local)         # [T_local, d_c^kv]
indexer_k_local   = Indexer.wk(X_local)           # [T_local, d_k^I]
indexer_w_local   = Indexer.weights_proj(X_local) # [T_local, n_index_heads]
q_c_local         = q_down_proj(X_local)          # [T_local, d_c^q]
indexer_q_local   = Indexer.wq_b(q_c_local)       # [T_local, d_I]
```

Here `kv_c_local` means the compact KV representation that can be written into
the MLA cache, including `kv_lora_rank` and RoPE K components. `indexer_k_local`
means the Indexer key representation after Indexer K projection, norm, and RoPE,
ready for the Indexer K cache. The implementation may split tensors into
`kv_c_normed`, `k_pe`, and backend-specific views before or after all-gather, but
the communication semantic remains the compact 704-dimensional payload.

Typical DeepSeek V3.2 dimensions:

```text
d_c^kv = 576  # kv_lora_rank 512 + qk_rope_head_dim 64
d_k^I  = 128
d_c^q  = 1536
hidden_size = 7168
H_attn = num_heads * v_head_dim = 16384
```

### 2. One All-Gather for KV / Indexer-K

Concatenate `kv_c_local` and `indexer_k_local`, then issue one async all-gather:

```text
AsyncAllGather([kv_c_local || indexer_k_local])
  [T_local, 704] -> [T, 704]
```

Only latent KV and Indexer K are gathered. `q_c_local`, `indexer_q_local`, and
`indexer_w_local` stay local because they are only needed for local query rows.

### 3. Local Full-Head q_up

Each device owns all attention heads for its local token rows:

```text
Q_local = q_up_proj(q_c_local)  # [T_local, num_heads, qk_head_dim]
```

In vLLM naming, `q_up_proj` corresponds to DeepSeek MLA `q_b_proj`. When
Sharded-CP is enabled, the wrapper uses full `num_heads`, not `num_local_heads`.

### 4. Local Indexer and SparseFlashAttention

After compact KV all-gather completes, split the global view:

```text
kv_c_global, indexer_k_global = split(gather_result)
topk_local = Indexer(indexer_q_local, indexer_k_global, indexer_w_local)
attn_local = SparseFlashAttention(Q_local, kv_c_global, topk_local)
```

The top-k buffer only needs to cover local query rows:

```text
topk_indices_buffer[:T_local]
```

No global tensor-row offset is required for top-k rows.

### 5. Local o_proj Without TP All-Reduce

Because every device has all heads for its local token rows, `o_proj` uses a full
logical weight and produces full hidden locally:

```text
O_local = o_proj(attn_local)  # [T_local, hidden_size]
Y_local = X_local + O_local
```

There is no traditional TP All-Reduce after `o_proj`, and `Y_local` is not
all-gathered back to `[T, hidden_size]`. The next module consumes CP hidden.

## MoE Flow

Attention output remains `Y_local: [T_local, hidden_size]`. MoE preprocessing is
per-token, so it can run locally:

```text
Y_norm_local = RMSNorm(Y_local)
Y_int8_local, scale_local, routing_local = QuantGating(Y_norm_local)
```

To reuse existing EP dispatch / expert execution, gather the lightweight
quantized data:

```text
Y_int8_global, scale_global, routing_global =
    AllGather(Y_int8_local, scale_local, routing_local)
```

Then enter existing MoE dispatch / expert compute. Expert output is
reduce-scattered back to CP layout:

```text
moe_out_local = ReduceScatter(MoE(Y_int8_global, scale_global, routing_global))
next_hidden_local = Y_local + moe_out_local
```

Transformer layers therefore continue to exchange `[T_local, hidden_size]`.

## KV Cache and Metadata Semantics

The thesis Sharded-CP implementation is not a replicated-KV attention-local MVP.
It requires current-token KV and Indexer K compute, cache writes, and metadata row
indexing to align with CP-local rows.

### Persistent Cache Ownership

During prefill:

- Each CP rank computes and writes KV / Indexer K only for local `[start, end)`
  token rows.
- `slot_mapping` values remain global cache slot IDs.
- Local tensor row `i` corresponds to global token row `start + i`.
- Block tables, sequence lengths, and request IDs keep global semantics so sparse
  attention can locate global KV.

If the current vLLM KV manager cannot express CP-local ownership, a CP-aware
cache manager / metadata builder is required. Falling back to replicated cache
updates may be useful for debugging, but it is not the thesis Sharded-CP design
and must not be used for the memory or communication conclusions in this
document.

### Local Metadata

For local token range `[start, end)`, build new metadata instances:

- `num_actual_tokens = end - start`.
- `slot_mapping = full.slot_mapping[start:end]`.
- token-indexed buffers such as `req_id_per_token` are sliced to local rows.
- `query_start_loc` describes only local query rows.
- prefill chunk metadata is rebuilt with local token offsets.
- decode metadata is unsupported in the first version; mixed batches fail
  closed.
- block tables and sequence lengths remain global per request.

Do not mutate global metadata in place because multiple layers/backends may
share the same object.

### Forward Context Override

Both indexer and `MLAAttention.forward()` read `get_forward_context()`. The
Sharded-CP path should run under a scoped override:

```python
forward_context = get_forward_context()
local_context = dataclasses.replace(
    forward_context,
    attn_metadata=local_attn_metadata_by_layer,
    slot_mapping=local_slot_mapping,
)
with override_forward_context(local_context):
    run_local_indexer_and_sparse_mla()
```

The model uses a CP-local forward context, not a replicated full-token context.

## Projection Strategy and Shard Linear

### Correctness Stage: Full Logical Weights

The first stage can instantiate full logical weights with `disable_tp=True` to
prove the CP dataflow and metadata:

- `q_b_proj = ColumnParallelLinear(..., disable_tp=True)`
- `kv_b_proj = ColumnParallelLinear(..., disable_tp=True)` if the backend needs
  explicit full-head KV up-projection
- `q_proj = ColumnParallelLinear(..., disable_tp=True)` when no q LoRA is used
- `o_proj = RowParallelLinear(..., input_is_parallel=True, reduce_results=False,
  disable_tp=True)`

`o_proj` must not reduce across ranks in Sharded-CP. Different ranks own
different token rows, so cross-rank accumulation would add unrelated tokens.

### Thesis Implementation: Layer-Owner Broadcast

The thesis Shard Linear implementation does not tensor-shard one matrix and
all-gather shards. Instead, it stores complete single-layer weights on owner
ranks and broadcasts them on demand. In other words, "sharded" means weights are
distributed by layer across devices, not that one layer's matrix is split by rows
or columns. This matches Ascend NZ preconverted weight constraints and avoids
cross-shard reconstruction.

For Sharded-CP, Shard Linear covers at least:

- `q_b_proj` / `q_up_proj`
- `o_proj`

Policy:

```text
owner(layer_id, weight_kind) = layer_id % cp_world_size
```

Each rank persistently stores complete single-layer weights for layers it owns
and keeps `K` prefetched layers in a cache. The thesis uses `K=2` as a practical
default. Before the current layer runs, the owner rank broadcasts the full
logical weight to the CP group. After the layer finishes, the materialized buffer
is released or reused.

Land a synchronous version first:

```text
Wait(q_up_weight_l)
Q_local = q_up_proj(q_c_local, q_up_weight_l)
Wait(o_proj_weight_l)
O_local = o_proj(attn_local, o_proj_weight_l)
Free(weight_l)
```

Then add async prefetch:

```text
AsyncBroadcast(q_up_weight_{l+K})
AsyncBroadcast(o_proj_weight_{l+K})
```

Typical DeepSeek V3.2 INT8 per-layer broadcast sizes:

- `q_up_proj`: `1536 * 16384 * 1 ~= 24 MiB`
- `o_proj`: `16384 * 7168 * 1 ~= 112 MiB`

These broadcasts should overlap with Indexer scoring, top-k, and
SparseFlashAttention.

### vLLM Naming Map

- Thesis `q_up_proj` maps to vLLM DeepSeek MLA `q_b_proj`.
- Thesis `kv_down_proj` maps to `kv_a_proj_with_mqa` or the KV A branch of
  `fused_qkv_a_proj`.
- The 704-dimensional compact KV is
  `kv_lora_rank + qk_rope_head_dim + index_head_dim`.
- Whether `kv_b_proj` joins Shard Linear depends on whether the backend requires
  explicit full-head KV up-projection. If it is included, its communication and
  memory model must be listed separately and not folded into the thesis
  `q_up + o_proj` 136 MiB number.

## Communication Model

Main per-layer attention communication:

1. Compact KV / Indexer-K all-gather:

```text
C_KV_AG = T * (d_c^kv + d_k^I) * bytes * (N - 1) / N
```

DeepSeek V3.2, `T=16384`, `d_c^kv=576`, `d_k^I=128`, BF16, `N=16`:

```text
C_KV_AG ~= 21.56 MiB
```

2. Shard Linear broadcast:

```text
C_q_up  ~= 24 MiB   # INT8
C_o_proj ~= 112 MiB # INT8
```

Total is about `157.6 MiB / layer`. Compared with pure TP's `o_proj`
All-Reduce plus Q All-Gather, the thesis reports about 66% lower per-layer
attention communication. More importantly, Sharded-CP communication is pipelined
with compute, while TP `o_proj` All-Reduce sits on the critical path.

## User-Facing Configuration

Add a boolean flag:

- Config field: `ParallelConfig.enable_sharded_context_parallel: bool = False`
- CLI flag: `--enable-sharded-context-parallel`
- Short internal name can be `enable_sharded_cp`

Validation must fail closed:

- `tensor_parallel_size > 1`, used as the initial CP world size.
- `pipeline_parallel_size == 1`.
- `prefill_context_parallel_size == 1`, to avoid mixing with existing PCP.
- `decode_context_parallel_size == 1`, to avoid mixing with existing DCP.
- model config has `index_topk`.
- MLA is enabled and the selected backend supports sparse MLA.
- speculative decoding / MTP disabled.
- DBO/ubatching disabled until local metadata is validated.
- full CUDA graph disabled or bypassed for affected paths.
- batches containing decode or mixed prefill/decode rejected until separately
  supported.

Log once when the feature is enabled:

- Sharded-CP uses CP hidden layout across Transformer layers.
- Whether the current run uses replicated full logical weights.
- Whether persistent KV cache is CP-sharded. If only a debug replicated path is
  active, state that it does not represent thesis performance/memory gains.

## Interaction With PCP/DCP

Do not overload:

- `prefill_context_parallel_size`
- `decode_context_parallel_size`
- `get_pcp_group()`
- `get_dcp_group()`

The first upstream implementation may reuse `get_tp_group()` as the process
group, but this is not DCP's head gather / LSE reduce algorithm and not PCP's KV
ownership rule. It is a single-device CP layout specific to DSA sparse MLA.

## Code Change Plan

### Commit 1: Config, CP Context, and Partition Utilities

Changes:

- Add `enable_sharded_context_parallel` config and CLI flag.
- Add fail-closed validation.
- Add CP token range, padding, all-gather/reduce-scatter helpers.
- Define that the initial CP group reuses the TP process group.

Tests:

- TP/PP/PCP/DCP/spec-decode validation.
- Model without `index_topk` rejects the flag.
- MLA / sparse-MLA backend absent rejects the flag.
- token partition helper covers even, uneven, empty, and single-token cases.
- multi-rank partition consistency: `T=10, N=4` → verify `(start, end, padded_end)`
  tuples form a gapless, non-overlapping cover of `[0, T)`.
- all-gather + trim round trip preserves original token order for uneven `T`.

### Commit 2: Embedding TP -> CP Hidden and LMHead CP -> TP

Changes:

- Add embedding output pad/reduce-scatter for Sharded-CP prefill.
- Make Transformer layer inputs/outputs CP-local rows.
- All-gather CP hidden before LMHead, then run vocab/tensor-parallel logits.

Tests:

- fake-layer round trip verifies token order, padding trim, and final logits
  shape.  "Fake-layer" means replacing all Transformer layers with `nn.Identity`
  (or the minimal equivalent in the model graph) so the test isolates the
  Embedding→CP→LMHead boundary path from attention/MoE details.
- Same token_ids with Sharded-CP on/off produce identical hidden states after
  all-gathering CP ranks back to the global view.
- Uneven `T` padding + trim does not drop or reorder tokens.

### Commit 3: CP-Local Metadata and KV/Indexer Cache Ownership

Changes:

- Add local metadata builders for `DeepseekV32IndexerMetadata` and
  `FlashMLASparseMetadata`.
- Slice `slot_mapping`, `req_id_per_token`, and prefill chunk offsets locally.
- Rebuild `query_start_loc` for the local token subset; reject batches where a
  single request's tokens are split across CP ranks (boundary falls inside a
  request) to keep metadata shapes simple.
- Write current-token KV / Indexer K only for local rows and global slot IDs.
- Fail closed for unsupported decode/mixed batches.
- Fail closed for FP8 mixed prefill/decode metadata until the rebuild path is
  implemented and tested.
- Add a forward-context override helper (`dataclasses.replace` + scoped context
  manager) so indexer and `MLAAttention.forward()` see CP-local metadata.

Tests:

- metadata slicing unit tests:
  - `num_actual_tokens == len(slot_mapping) == len(req_id_per_token)`.
  - prefill chunk `token_start/token_end` fall inside the local range.
  - `query_start_loc` describes only local query rows and CSR invariants hold.
  - three unequal requests `[100, 200, 50]` with `CP=2`: rank 0 owns request
    0+1 (300 tokens full), rank 1 owns request 2 (50 tokens full); verify
    metadata correctness on both ranks.
- local cache write rows align with global slot IDs on each CP rank.
  - Distributed test: write local KV, then read back from global slots;
    values round-trip correctly.
- FP8 mixed prefill/decode case is rejected.

### Commit 4: Sharded-CP Attention Correctness Path

Changes:

- Use full `num_heads` in the wrapper for DeepSeek V3.2 sparse MLA.
- Use full logical `q_b_proj`/`q_up_proj` and `o_proj`, initially replicated if
  needed.
- `kv_b_proj` strategy: if the sparse MLA backend uses weight absorption
  (`W_UV` absorbed into `o_proj`, `W_UK` absorbed into `q_up_proj`), then
  `kv_b_proj` does not appear as a separate projection at runtime and needs no
  special handling.  If the backend performs explicit full-head KV up-projection
  (requires a per-rank `kv_b_proj` matmul), then `kv_b_proj` must also use
  `disable_tp=True` in the replicated stage.  Commit 6 determines whether
  `kv_b_proj` joins Shard Linear based on whether it exists at runtime.
- Implement compact `[kv_c || indexer_k]` all-gather.
- Run local indexer/top-k/sparse attention/o_proj and return CP-local rows.
- Fail closed for FP8 cases that combine prefill/decode metadata.

Tests:

- TP=2/4 prefill logits parity, including uneven `T`.
  - Tolerances: BF16 max relative error `< 2e-3`, max absolute error `< 3e-2`
    for logits; FP8 separately calibrated or rejected.
- top-k buffer only writes the local prefix, and the top-k indices mapped back
  to global token positions match baseline selections (within floating-point
  tolerance of Indexer scores).
- intermediate attention-output hidden states parity against flag-off baseline
  (not only final logits).
- multi-layer parity: `≥ 2` layers forward, hidden states at each layer
  boundary agree with baseline.
- `[kv_c || indexer_k]` concat → all-gather → split round-trip: the original
  per-rank `kv_c` and `indexer_k` values are lossless on every rank.
- flag-off behavior unchanged.

### Commit 5: MoE CP Adaptation

Changes:

- Run RMSNorm, Quant, and Gating on local rows.
- All-gather quantized activations, scales, and routing metadata before EP
  dispatch.
- Reduce-scatter expert output back to CP layout.
- For dense MLP layers (e.g. first 3 layers in DeepSeek V3.2): dense MLP
  performs per-token transformations that are independent across tokens, so it
  can run directly on CP-local rows without all-gathering the full sequence.
  Verify equivalence to the global-path result at the output of every dense MLP
  layer.

Tests:

- MoE input/output preserves `[T_local, hidden_size]` shape.
- routing metadata after all-gather is ordered consistently with the original
  global token order: for `N` CP ranks with known local token counts, the
  concatenated routing info matches the baseline pre-CP routing.
- EP dispatch: each token is routed to the same expert as in the baseline (no
  silent routing divergence).
- dense MLP layers on CP-hidden produce outputs bitwise-equivalent to
  global-path results (no communication needed, so exact match expected).
- residual connection `Y_local + moe_out_local` adds the correct global token
  pairs: token `i` in rank `r` on the attention path and on the
  reduce-scattered MoE path correspond to the same global token.

### Commit 6: Synchronous Shard Linear

Changes:

- Add layer-owner full-weight storage for `q_b_proj`/`q_up_proj` and `o_proj`.
- `kv_b_proj` decision: only include `kv_b_proj` in Shard Linear if the sparse
  MLA backend performs explicit per-rank KV up-projection at runtime.  If the
  backend uses weight absorption, `kv_b_proj` does not exist as a separate
  matmul and is omitted.  Follow the commit 4 backend analysis to make this
  determination; if included, list its per-layer size and broadcast overhead
  separately so it does not inflate the thesis `q_up + o_proj` 136 MiB figure.
- Synchronously broadcast materialized full logical weight before layer use.
- Release/reuse buffers after layer completion.

Tests:

- owner distribution follows `layer_id % cp_world_size`.
- materialized weights are identical across ranks after broadcast.
- logits parity with the replicated full-weight path (Commit 4).
- W8A8 quantized weight loading: full logical or layer-owner weights load
  correctly from a TP-sharded quantized checkpoint; at least one common
  quantization format covered.
- multi-layer forward: no memory leak after repeated frees across 61 layers.

### Commit 7: Async Broadcast / Prefetch

Changes:

- Add `K`-layer prefetch, default `K=2`.
- Define communication stream, compute stream, event waits, and cleanup.
- Keep synchronous fallback.

Tests:

- prefetch on/off logits parity.
- overlap verification: CUDA/host events confirm that each broadcast's start
  precedes the next layer's dependent compute, and its end precedes the
  consuming `o_proj` / `q_up_proj` call.
- boundary cases:
  - layer 0: prefetch of layer `K` starts before layer 0 compute; no wait for
    a non-existent previous-layer broadcast.
  - last `K` layers: prefetch targets beyond `num_layers-1` are silently
    skipped.
  - model with few layers (e.g. 3-layer dense MLP-only stack): `K=2` prefetch
    does not OOB or stall.
- repeated multi-layer forward stress test: 61-layer model, 100 forward passes
  with varying token distributions across passes; no memory leak, no CUDA error,
  no NCCL timeout.
- no stale weight buffer use: after `Free(l)` the buffer handle is invalidated
  and a guard or sanitizer catches a use-after-free.
- synchronous fallback: when async path initialization fails (e.g. stream
  creation), the synchronous path produces correct results.
- short-sequence scenario (`T <= 512`): broadcast may not be fully hidden by
  compute but must still produce correct logits.

## Verification Matrix

| Case | CP/TP world | Batch | Expected |
| --- | --- | --- | --- |
| config: no `index_topk` | 2 | n/a | rejects flag |
| config: no sparse MLA backend | 2 | n/a | rejects flag |
| config validation | 1 | n/a | rejects flag |
| CP partition | 2/4/16 | uneven T | ordered gather/scatter, gapless cover of `[0,T)` |
| embedding → LMHead round trip | 2 | prefill | shape and order correct, same hidden after all-gather |
| metadata slicing | 2 | prefill | local rows/global slots aligned, field invariants hold |
| metadata: 3 unequal requests | 2 | [100,200,50] | rank 0 owns req 0+1, rank 1 owns req 2 |
| metadata: FP8 mixed | 2 | mixed prefill/decode | rejected with clear error |
| KV cache write/read | 2 | prefill | written values round-trip from global slot IDs |
| attention hidden parity | 2 | one prompt | attention hidden states match baseline |
| attention logits parity | 2/4 | one prompt | logits within tolerance (BF16: rel<2e-3, abs<3e-2) |
| attention logits parity | 2/4 | uneven T | logits within tolerance |
| multi-layer attention parity | 2 | one prompt | per-layer hidden states match baseline (≥2 layers) |
| top-k correctness | 2 | one prompt | mapped back to global, matches baseline selections |
| KV compact concat/split | 2 | prefill | values lossless after all-gather round trip |
| MoE CP shape invariant | 2 | routed tokens | output remains `[T_local, hidden]` |
| MoE routing order | 2 | routed tokens | all-gathered routing matches baseline pre-CP routing |
| dense MLP CP equivalence | 2 | prefill | exact match with global path |
| MoE residual correctness | 2 | routed tokens | attention output + MoE output add correct global pairs |
| shard linear: ownership | 2/4 | n/a | `layer_id % cp_world_size` distribution |
| shard linear: quantized load | 2 | W8A8 | weight loads and matches replicated path |
| shard linear: logits | 2/4 | prefill | parity with replicated weights (Commit 4) |
| shard linear: no memory leak | 2/4 | 61 layers | no growth after repeated frees |
| prefetch: parity | 2/4 | prefill | parity with sync Shard Linear |
| prefetch: overlap | 2/4 | prefill | broadcast ends before consuming compute |
| prefetch: boundary layers | 2/4 | prefill | no OOB prefetch, no stall at last layers |
| prefetch: stress | 2/4 | 100 passes | no leak, no NCCL timeout |
| prefetch: stale buffer | 2/4 | prefill | use-after-free guard catches reuse |
| prefetch: short sequence | 2/4 | T≤512 | correct logits |
| flag off | 2 | existing tests | unchanged behavior |

Use tolerances appropriate for BF16 / FP8 sparse MLA. Exact floating-point
equality is not required.

Suggested commands once GPU/NPU dependencies are available:

```bash
.venv/bin/python -m pytest tests/distributed/test_sharded_cp_utils.py -v
.venv/bin/python -m pytest tests/distributed/test_sharded_cp_metadata.py -v
.venv/bin/python -m pytest tests/models/test_deepseek_v32_sharded_cp.py -v
.venv/bin/python -m pytest tests/distributed/test_sharded_cp_moe.py -v
.venv/bin/python -m pytest tests/distributed/test_sharded_cp_shard_linear.py -v
```

Follow the repository rule: use `.venv/bin/python` and `uv`, not system
`python3` or bare `pip`.

## Open Risks

- CP hidden layout across the Transformer body touches embedding, attention,
  MoE, LMHead, scheduler, and metadata boundaries. The blast radius is larger
  than an attention-local MVP.
- Sparse metadata slicing is the main correctness risk. Any token-indexed
  metadata left in global row indexing can silently corrupt cache writes or
  top-k rows.
- If the current vLLM KV manager assumes replicated ownership, a CP-aware prefill
  ownership path is required to achieve the thesis KV memory / access gains.
- MoE routing metadata gather order must match EP dispatch assumptions.
- Dense MLP layers in Sharded-CP: the thesis covers only MoE layers, but
  DeepSeek V3.2 has 3 initial dense MLP layers.  Dense MLP runs per-token so it
  is naturally CP-compatible, but this must be explicitly verified.
- Quantized loaders may assume TP-sharded parameter shapes. Full logical or
  layer-owner weights must be validated per quantization mode.
- `kv_b_proj` scope: whether it exists at runtime and whether it joins Shard
  Linear depends on the backend's weight-absorption implementation.  Commit 4
  must determine this; Commit 6 must match the decision.
- Async prefetch changes weight-buffer lifetime and stream ordering; it should
  land after the synchronous version.
- For short sequences, broadcast may not be fully hidden by compute, so gains can
  be weaker than the long-context thesis results.
- CUDA graph capture may bake token counts and metadata shapes. Disable or bypass
  graph capture for ShardedCP until the layout is stable.

## Recommended Implementation Order

First implement the thesis data layout with replicated full logical weights for
correctness:

```text
EmbeddingTP
  -> CP hidden
  -> local DSA attention with full heads + compact KV AG
  -> local MoE preprocess + EP-compatible gather/RS
  -> CP hidden across layers
  -> LMHeadTP
```

After logits parity and metadata/cache ownership are proven, add synchronous
Shard Linear. Finally add async broadcast/prefetch so communication is hidden
under `q_up_proj`, Indexer scoring, top-k, and SparseFlashAttention compute
windows.
