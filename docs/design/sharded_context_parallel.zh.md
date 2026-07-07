# DSA 稀疏注意力模型的 Sharded Context Parallel 设计

本文描述如何在 vLLM 中实现论文第 5 章和博客中的 Sharded Context
Parallelism（Sharded-CP）。该方案面向 DeepSeek V3.2 / GLM-5 风格的 DSA
稀疏 MLA 模型，核心目标是把 CP 粒度推进到单卡级别，消除 TP 组内的
Indexer 冗余计算，并通过 Shard Linear 控制完整 attention 权重带来的显存
开销。

这份文档采用论文里的实现语义，而不是一个只在 attention 内临时切 token、
attention 后马上恢复 TP full hidden 的 MVP。Sharded-CP 启用后，Transformer
主体在层间保持 CP 布局：

```text
Embedding TP
  -> Pad/Reduce-Scatter to CP hidden
  -> repeated Transformer layers on [T_local, hidden_size]
  -> final All-Gather to TP/LMHead
```

## 参考资料

- RFC：`vllm-project/vllm#30055`，"Sharded Context Parallelism for
  DeepSeek DSA"。
- 博客：`https://zzhx1.github.io/2025/12/04/Sharded-Context-Parallel/`。
- Ascend 参考 PR：`vllm-project/vllm-ascend#4702`。
- 论文第 5 章："面向稀疏注意力模型的分片上下文并行优化"。

Ascend PR 可以作为实现形态参考：它增加 CP context，切分 attention
metadata 的 token，替换 CP 下的 MLA up/output projection，并加入
shared-weight broadcast/prefetch。上游 vLLM 不应该照搬插件式运行时替换
class 的方式；应在 core 模型构造和 forward 路径中加入显式 hook。

## 设计边界

目标范围：

- 模型范围：DeepSeek V3.2 风格的 DSA / sparse MLA 模型，先通过
  `hf_config.index_topk` 识别。
- 运行阶段：Prefill 优先。Decode 可以继续由现有 DP/DCP/FGTP 路径处理；在
  Sharded-CP prefill 和 decode 共存前，混合 decode batch 应 fail closed。
- 并行域：每张 GPU/NPU 是一个 CP rank。上游 vLLM 第一版可以复用现有
  tensor-parallel process group 作为通信 group，但语义上它是 CP group。
- 数据布局：Embedding 和 LMHead 保留 TP 形式；Transformer 主体的 attention
  与 MoE 在 CP hidden 上执行。
- 优化目标：单卡只处理 `1 / N` 的 query token，从而降低 sparse attention
  随机访存、Indexer 查询/评分/top-k 计算和前置逐 token 处理。

非目标：

- 不实现通用 attention backend 的 CP。
- 不复用或重载 vLLM 现有 PCP/DCP 算法。
- 不支持 pipeline parallel。
- 不支持 speculative decoding / MTP。
- 稳定前不支持完整 CUDA graph。
- 不在第一版承诺跨节点通信优化。
- 不用 Ascend 插件式 layer monkey-patching。

## 当前 vLLM 状态

DeepSeek V3.2 sparse MLA 支持已经存在：

- `vllm/model_executor/models/deepseek_v2.py`
  - `DeepseekV2MLAAttention.is_v32 = hasattr(config, "index_topk")`。
  - `q_b_proj` 和 `kv_b_proj` 是 `ColumnParallelLinear`。
  - `o_proj` 是 `RowParallelLinear`。
  - `Indexer` 的 `wq_b`、`wk`、`weights_proj` 使用 `ReplicatedLinear`。
- `vllm/model_executor/layers/mla.py`
  - `MultiHeadLatentAttentionWrapper.forward()` 计算 `q`、`kv_c_normed`、
    `k_pe`，调用 V3.2 sparse indexer，再调用 `MLAAttention`。
- `vllm/model_executor/layers/attention/mla_attention.py`
  - `MLAAttention.forward()` 通过 `get_forward_context().slot_mapping` 更新
    KV cache。
  - sparse MLA impl 复用 MLA attention metadata。
- `vllm/v1/attention/backends/mla/indexer.py`
  - 从与 attention backend 相同的 `CommonAttentionMetadata` 构造 indexer
    metadata。
- `vllm/v1/attention/backends/mla/flashmla_sparse.py`
  - 构造 sparse MLA metadata，并消费 indexer 写入的 top-k buffer。

本地 API 细节：`ColumnParallelLinear` 和 `RowParallelLinear` 已支持
`disable_tp=True`。这可以作为 correctness 阶段的 full logical weight 路径，
但最终论文实现应通过 Shard Linear 降低 `q_b_proj`/`q_up_proj` 与 `o_proj`
的持久显存。

## 问题分析

DSA 的 Lightning Indexer 为每个 query token 选择 top-k KV 条目。传统 TP
切 head，不切 token，因此每个 TP rank 都围绕完整 token 序列运行：

- sparse attention 的 token 级随机 KV gather；
- Indexer 的 `W_qb`、`W_k`、`W_p` 投影；
- Indexer scoring 和 top-k。

Indexer 不是标准 Megatron 式成对矩阵乘。它的输出需要立刻进行 top-k，不能在
不引入额外全量激活通信的情况下按 TP hidden/head 维度高效切分。因此 CP 更适合
DSA：每卡只负责本地 `T / N` 个 query token，Indexer 和 sparse attention 的
主要工作量随 CP 度线性下降。

传统 CP+TP 混合架构仍有两个限制：

- 一个 CP rank 内部仍包含多个 TP rank，同一 token 分片上的 Indexer 仍被 TP
  ranks 重复计算。
- CP 度受权重显存限制，难以扩展到单卡级别。

Sharded-CP 通过单卡级 CP 和 Shard Linear 解耦这两个问题。

## 全局数据流

令全局 token 数为 `T`，设备数/CP 度为 `N`，当前 rank 为 `r`。

### 1. Embedding TP 到 CP hidden

Embedding 保持 TP/vocab-parallel 形式以降低词表权重显存。得到全局 hidden 后，
执行 pad + reduce-scatter，把 token rows 分发到 CP ranks：

```text
token_ids:       [T]
EmbeddingTP:     [T, hidden_size]
Pad + RS:        [T_local, hidden_size]
```

`T_local` 使用 contiguous balanced partition：

```text
chunk = ceil(T / N)
start = r * chunk
end = min(start + chunk, T)
padded_end = start + chunk
```

本地 tensor 在通信前 padding 到 `chunk` rows，通信后按真实 `end - start`
裁剪。

### 2. Transformer 主体保持 CP 布局

Sharded-CP 下，每层输入输出都是：

```text
hidden_states_local: [T_local, hidden_size]
positions_local:     [T_local]
metadata_local:      describes local query rows, with global sequence/cache IDs
```

attention、RMSNorm、gating、quantization 这类逐 token 操作直接在本地 rows 上
执行。层间不再 all-gather 回 full token hidden。

### 3. LMHead 前恢复全局 token rows

最后一层 MoE 输出仍是 CP 布局。进入 LMHead 前，先 all-gather token rows：

```text
hidden_states_global = AllGather(hidden_states_local)  # [T, hidden_size]
logits_local_vocab = LMHeadTP(hidden_states_global)    # [T, vocab / tp]
logits = GatherVocab(logits_local_vocab)
```

## Attention 模块流程

对每个 DeepSeek V3.2 sparse MLA layer，本地输入为
`X_local: [T_local, hidden_size]`。

### 1. 本地降维与 Indexer 前置投影

优先计算用于 KV 聚合的本地紧凑表示：

```text
kv_c_local        = kv_down_proj(X_local)        # [T_local, d_c^kv]
indexer_k_local   = Indexer.wk(X_local)          # [T_local, d_k^I]
indexer_w_local   = Indexer.weights_proj(X_local)# [T_local, n_index_heads]
q_c_local         = q_down_proj(X_local)         # [T_local, d_c^q]
indexer_q_local   = Indexer.wq_b(q_c_local)      # [T_local, d_I]
```

`kv_c_local` 在这里表示可写入 MLA cache 的 compact KV 表示，包含
`kv_lora_rank` 和 RoPE K 分量。`indexer_k_local` 表示经过 Indexer K projection、
norm、RoPE 后可写入 Indexer K cache 的 key 表示。实现中可以在 all-gather 前后按
backend 需要拆成 `kv_c_normed`、`k_pe` 等张量，但通信语义保持 compact 704 维。

DeepSeek V3.2 典型维度：

```text
d_c^kv = 576  # kv_lora_rank 512 + qk_rope_head_dim 64
d_k^I  = 128
d_c^q  = 1536
hidden_size = 7168
H_attn = num_heads * v_head_dim = 16384
```

### 2. KV / Indexer-K 一次 All-Gather

将 `kv_c_local` 和 `indexer_k_local` 横向拼接后发起一次异步 all-gather：

```text
AsyncAllGather([kv_c_local || indexer_k_local])
  [T_local, 704] -> [T, 704]
```

只聚合 KV latent 和 Indexer K。`q_c_local`、`indexer_q_local`、
`indexer_w_local` 始终保持本地，因为它们只服务本地 query rows。

### 3. 本地 full-head q_up

每张卡拥有本地 token rows 的全部 attention heads：

```text
Q_local = q_up_proj(q_c_local)  # [T_local, num_heads, qk_head_dim]
```

在 vLLM 命名中，`q_up_proj` 对应 DeepSeek MLA 的 `q_b_proj`。Sharded-CP
启用时，wrapper 内使用 full `num_heads`，不是 `num_local_heads`。

### 4. 本地 Indexer 与 SparseFlashAttention

等待 compact KV all-gather 完成后，拆分全局视图：

```text
kv_c_global, indexer_k_global = split(gather_result)
topk_local = Indexer(indexer_q_local, indexer_k_global, indexer_w_local)
attn_local = SparseFlashAttention(Q_local, kv_c_global, topk_local)
```

top-k buffer 只需要覆盖本地 query rows：

```text
topk_indices_buffer[:T_local]
```

不需要为 top-k row 使用全局 tensor row offset。

### 5. 本地 o_proj，无 TP All-Reduce

因为每张卡已经拥有本地 token 的全部 heads，`o_proj` 使用 full logical weight
直接输出完整 hidden：

```text
O_local = o_proj(attn_local)  # [T_local, hidden_size]
Y_local = X_local + O_local
```

这里没有传统 TP `o_proj` 后的 All-Reduce，也不把 `Y_local` all-gather 回
`[T, hidden_size]`。下一模块继续消费 CP hidden。

## MoE 模块流程

Attention 输出仍是 `Y_local: [T_local, hidden_size]`。MoE 前置的 RMSNorm、
量化、gating 都是逐 token 操作，可以本地执行：

```text
Y_norm_local = RMSNorm(Y_local)
Y_int8_local, scale_local, routing_local = QuantGating(Y_norm_local)
```

为了复用现有 EP dispatch / expert execution，可在量化后聚合轻量数据：

```text
Y_int8_global, scale_global, routing_global =
    AllGather(Y_int8_local, scale_local, routing_local)
```

随后进入现有 MoE dispatch / expert compute。专家计算结果通过 reduce-scatter
回到 CP 布局：

```text
moe_out_local = ReduceScatter(MoE(Y_int8_global, scale_global, routing_global))
next_hidden_local = Y_local + moe_out_local
```

这样 Transformer 层间仍保持 `[T_local, hidden_size]`。

## KV Cache 与 Metadata 语义

论文实现的 Sharded-CP 不是 replicated KV cache 的 attention-local MVP。它要求
current-token KV 和 Indexer K 的计算、cache 写入、metadata row indexing 都按
CP local rows 对齐。

### Persistent Cache Ownership

Prefill 阶段：

- 每个 CP rank 只计算并写入本地 `[start, end)` token rows 的 KV / Indexer K。
- `slot_mapping` 的值仍是全局 cache slot ID。
- local tensor row `i` 对应全局 token row `start + i`。
- block tables、sequence lengths、request IDs 保持全局语义，供 sparse attention
  定位全局 KV。

如果当前 vLLM KV manager 不能表达 CP-local ownership，必须增加 CP-aware cache
manager / metadata builder。退回 replicated cache update 可以作为临时调试路径，
但它不等价于论文里的 Sharded-CP，也不能用于文档里的显存/通信收益结论。

### Local Metadata

对本地 token range `[start, end)`，构造新的 metadata instance：

- `num_actual_tokens = end - start`。
- `slot_mapping = full.slot_mapping[start:end]`。
- `req_id_per_token` 等 token-indexed buffer 切到本地 rows。
- `query_start_loc` 只描述本地 query rows。
- prefill chunk metadata 需要用 local token offsets rebuild。
- decode metadata 第一版不支持；混合 batch fail closed。
- block tables 和 sequence lengths 保持 global per request。

不要 in-place 修改全局 metadata，因为多个 layer/backend 可能共享同一个对象。

### Forward Context Override

indexer 和 `MLAAttention.forward()` 都读取 `get_forward_context()`。Sharded-CP
路径应在 scoped override 下运行：

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

模型其他部分使用 CP-local forward context，而不是 full-token replicated
context。

## Projection 与 Shard Linear

### Correctness 阶段：Full Logical Weights

第一阶段可以使用 `disable_tp=True` 构造 full logical weights，先证明 CP 数据流和
metadata 正确：

- `q_b_proj = ColumnParallelLinear(..., disable_tp=True)`
- `kv_b_proj = ColumnParallelLinear(..., disable_tp=True)`，如果 backend 需要显式
  full-head KV up-projection
- 无 q LoRA 时：`q_proj = ColumnParallelLinear(..., disable_tp=True)`
- `o_proj = RowParallelLinear(..., input_is_parallel=True, reduce_results=False,
  disable_tp=True)`

注意：`o_proj` 在 Sharded-CP 下不能对 ranks 做 TP reduce。每个 rank 的 rows
不同，跨 rank 累加会把不相关 token 相加。

### 论文实现：Layer-Owner Broadcast

论文里的 Shard Linear 不是对单个矩阵做 tensor shard 后 all-gather 拼接，而是
按 layer owner 保存完整单层权重并按需 broadcast。换言之，"sharded" 指权重按
层分布在设备之间，而不是把一个 layer 的矩阵按列/行切开。这是为了适配 Ascend
NZ 等预转换权重格式，也避免跨 shard 重组。

对 Sharded-CP，Shard Linear 至少覆盖：

- `q_b_proj` / `q_up_proj`
- `o_proj`

策略：

```text
owner(layer_id, weight_kind) = layer_id % cp_world_size
```

每个 rank 持久保存自己 owner 的完整单层权重，并额外保留 `K` 层预取缓存
（论文中 `K=2` 是合理默认）。当前 layer 运行前，从 owner rank 向 CP group
broadcast full logical weight。layer 完成后释放或复用 materialized buffer。

同步版本先落地：

```text
Wait(q_up_weight_l)
Q_local = q_up_proj(q_c_local, q_up_weight_l)
Wait(o_proj_weight_l)
O_local = o_proj(attn_local, o_proj_weight_l)
Free(weight_l)
```

异步版本再加入 prefetch：

```text
AsyncBroadcast(q_up_weight_{l+K})
AsyncBroadcast(o_proj_weight_{l+K})
```

论文通信分析中，DeepSeek V3.2 INT8 权重的典型单层 broadcast 量为：

- `q_up_proj`: `1536 * 16384 * 1 ~= 24 MiB`
- `o_proj`: `16384 * 7168 * 1 ~= 112 MiB`

这两次 broadcast 应分别和 Indexer scoring、Top-k/SparseFlashAttention 重叠。

### vLLM 命名映射

- 论文 `q_up_proj` 对应 vLLM DeepSeek MLA 的 `q_b_proj`。
- 论文 `kv_down_proj` 对应 `kv_a_proj_with_mqa` 或 fused
  `fused_qkv_a_proj` 中的 KV A branch。
- 论文 compact KV 的 704 维是 `kv_lora_rank + qk_rope_head_dim + index_head_dim`。
- `kv_b_proj` 是否纳入 Shard Linear 取决于 backend 是否需要显式 full-head
  KV up-projection。若纳入，通信/显存模型必须单独列出，不能混入论文里
  136 MiB 的 `q_up + o_proj` 数字。

## 通信模型

每层 Attention 的主要通信：

1. compact KV / Indexer-K all-gather：

```text
C_KV_AG = T * (d_c^kv + d_k^I) * bytes * (N - 1) / N
```

DeepSeek V3.2, `T=16384`, `d_c^kv=576`, `d_k^I=128`, BF16, `N=16`：

```text
C_KV_AG ~= 21.56 MiB
```

2. Shard Linear broadcast：

```text
C_q_up  ~= 24 MiB   # INT8
C_o_proj ~= 112 MiB # INT8
```

总通信量约 `157.6 MiB / layer`。和纯 TP 的 `o_proj` All-Reduce +
Q All-Gather 相比，论文中约降低 66%。更重要的是，Sharded-CP 的通信可以通过
pipeline 与计算重叠，而 TP `o_proj` All-Reduce 位于关键路径。

## 用户配置与校验

增加 boolean flag：

- Config 字段：`ParallelConfig.enable_sharded_context_parallel: bool = False`
- CLI 参数：`--enable-sharded-context-parallel`
- 内部短名可用 `enable_sharded_cp`

校验必须 fail closed：

- `tensor_parallel_size > 1`，作为初始 CP world size。
- `pipeline_parallel_size == 1`。
- `prefill_context_parallel_size == 1`，避免和现有 PCP 混用。
- `decode_context_parallel_size == 1`，避免和现有 DCP 混用。
- model config 存在 `index_topk`。
- MLA 已启用，并选择 sparse MLA backend。
- speculative decoding / MTP 关闭。
- 第一版禁用 DBO/ubatching，除非完成 local metadata 验证。
- full CUDA graph 对受影响路径禁用或绕过。
- batch 中包含 decode 或 mixed prefill/decode 时拒绝，直到单独支持。

feature 启用时 log once：

- Sharded-CP 使用 CP hidden layout across Transformer layers。
- 第一阶段是否使用 replicated full logical weights。
- persistent KV cache 是否已经 CP-sharded；如果只是调试 replicated path，应明确
  不能代表论文性能/显存收益。

## 与 PCP/DCP 的关系

不要复用或重载：

- `prefill_context_parallel_size`
- `decode_context_parallel_size`
- `get_pcp_group()`
- `get_dcp_group()`

Sharded-CP 可以在上游第一版复用 `get_tp_group()` 的 process group，但它不是
DCP 的 head gather / LSE reduce 算法，也不是 PCP 的 KV ownership 规则。它是
针对 DSA sparse MLA 的单卡级 CP 数据布局。

## 代码变更计划

### Commit 1：Config、CP context 和分片工具

变更：

- 增加 `enable_sharded_context_parallel` config 和 CLI flag。
- 增加 fail-closed validation。
- 增加 CP token range、padding、all-gather/reduce-scatter helper。
- 明确 CP group 初始复用 TP process group。

测试：

- TP/PP/PCP/DCP/spec decode validation。
- 模型无 `index_topk` 时拒绝该 flag。
- MLA / sparse MLA backend 不存在时拒绝该 flag。
- token partition helper 覆盖 even、uneven、empty、single-token cases。
- 多 rank partition 一致性：`T=10, N=4` → 验证 `(start, end, padded_end)`
  三元组构成无间隙、无重叠的 `[0, T)` 覆盖。
- all-gather + trim round trip 在 uneven `T` 下保持原始 token 顺序不变。

### Commit 2：Embedding TP -> CP hidden，LMHead CP -> TP

变更：

- 增加 Sharded-CP prefill 的 embedding output pad/reduce-scatter。
- Transformer layer 输入输出改为 CP local rows。
- LMHead 前 all-gather CP hidden，再走 vocab/tensor parallel logits。

测试：

- fake layer round trip 验证 token 顺序、padding trim、最后 logits shape。
  这里的 "fake layer" 指用 `nn.Identity`（或模型图中等价的最小操作）替换所
  有 Transformer layer，从而使测试聚焦在 Embedding→CP→LMHead 首尾边界路径
  上，排除 attention/MoE 细节的干扰。
- 相同 token_ids 在 Sharded-CP on/off 下，all-gather CP ranks 回到全局视图
  后 hidden states 一致。
- uneven `T` padding + trim 不丢失、不重排 token。

### Commit 3：CP-local metadata 和 KV/indexer cache ownership

变更：

- 为 `DeepseekV32IndexerMetadata` 和 `FlashMLASparseMetadata` 增加 local
  metadata builder。
- `slot_mapping`、`req_id_per_token`、prefill chunk offsets 切到本地。
- 在 local token 子集上重建 `query_start_loc`；拒绝单个 request 的 token
  被拆分到不同 CP rank 的 batch（边界落在 request 内部），以保持 metadata
  shape 简单。
- current-token KV / Indexer K 只写本地 rows 的 global slots。
- 对不支持的 decode/mixed batch fail closed。
- 对 FP8 mixed prefill/decode metadata fail closed，直到 rebuild 路径实现并
  测试完毕。
- 增加 forward-context override helper（`dataclasses.replace` + scoped
  context manager），使 indexer 和 `MLAAttention.forward()` 看到 CP-local
  metadata。

测试：

- metadata slicing unit tests：
  - `num_actual_tokens == len(slot_mapping) == len(req_id_per_token)`。
  - prefill chunk 的 `token_start/token_end` 落在本地范围内。
  - `query_start_loc` 只描述本地 query rows 且 CSR 不变量成立。
  - 三个不等长请求 `[100, 200, 50]`，`CP=2`：rank 0 持有请求 0+1（共
    300 token），rank 1 持有请求 2（50 token）；验证两侧 metadata 正确。
- local cache write rows 与 global slot IDs 对齐。
  - 分布式测试：写入 local KV，再从 global slots 读回；数值正确 round-trip。
- FP8 mixed prefill/decode 被正确拒绝。

### Commit 4：Sharded-CP Attention Correctness Path

变更：

- DeepSeek V3.2 sparse MLA 下 wrapper 使用 full `num_heads`。
- 使用 full logical `q_b_proj`/`q_up_proj` 和 `o_proj`，可先 replicated。
- `kv_b_proj` 策略：若 sparse MLA backend 使用了 weight absorption
  （`W_UV` 吸收到 `o_proj`，`W_UK` 吸收到 `q_up_proj`），则 `kv_b_proj`
  在运行时不会作为独立投影出现，无需特殊处理。若 backend 执行显式的
  full-head KV up-projection（需要 per-rank `kv_b_proj` 矩阵乘），则
  `kv_b_proj` 在 replicated 阶段也必须 `disable_tp=True`。Commit 6 基于
  `kv_b_proj` 在运行时是否存在决定是否将其纳入 Shard Linear。
- 实现 compact `[kv_c || indexer_k]` all-gather。
- 本地 indexer/top-k/sparse attention/o_proj，输出仍为 CP local rows。
- 对 FP8 混合 prefill/decode metadata fail closed。

测试：

- TP=2/4 prefill logits parity，包含 uneven `T`。
  - 容忍度：BF16 max relative error `< 2e-3`，max absolute error `< 3e-2`；
    FP8 单独校准或拒绝。
- top-k buffer 只写本地 prefix，且 top-k 索引映射回全局 token 位置后与
  baseline 选择一致（在 Indexer score 浮点容忍度内）。
- 中间 attention 输出 hidden states parity 对比 flag-off baseline（不仅仅
  看最终 logits）。
- 多层 parity：`≥ 2` 层 forward，每层边界 hidden states 与 baseline 一致。
- `[kv_c || indexer_k]` concat → all-gather → split round-trip：每个 rank
  原始的 `kv_c` 和 `indexer_k` 值无损。
- flag off 行为不变。

### Commit 5：MoE CP 适配

变更：

- RMSNorm、Quant、Gating 在 local rows 上执行。
- all-gather quantized activations、scale、routing metadata 后复用 EP dispatch。
- expert output reduce-scatter 回 CP layout。
- 对稠密 MLP 层（如 DeepSeek V3.2 的前 3 层）：稠密 MLP 执行逐 token 变换，
  各 token 间无依赖，因此可直接在 CP-local rows 上运行而不需要 all-gather
  完整序列。在每层稠密 MLP 输出处验证与全局路径的结果等价。

测试：

- MoE 输入/输出保持 `[T_local, hidden_size]` shape。
- all-gather 后 routing metadata 顺序与原始全局 token 顺序一致：对 `N` 个
  CP rank 已知各自 local token 数，拼接后的 routing info 与 baseline pre-CP
  routing 一致。
- EP dispatch：每个 token 被路由到的专家与 baseline 相同（无静默路由偏移）。
- 稠密 MLP 层在 CP hidden 上计算结果与全局路径结果精确一致（无需通信，故相
  等性可做到 bitwise）。
- 残差连接 `Y_local + moe_out_local` 语义正确：rank `r` 中 attention 路径的
  token `i` 与 reduce-scatter 后 MoE 路径的 token `i` 对应同一个全局 token。

### Commit 6：Synchronous Shard Linear

变更：

- 为 `q_b_proj`/`q_up_proj` 和 `o_proj` 增加 layer-owner full-weight storage。
- `kv_b_proj` 决策：仅当 sparse MLA backend 在运行时执行显式 per-rank KV
  up-projection 时才将 `kv_b_proj` 纳入 Shard Linear。若 backend 使用
  weight absorption，`kv_b_proj` 不作为独立矩阵乘存在，则跳过。该决策依赖
  Commit 4 的 backend 分析；若纳入，须单独列出其单层大小和 broadcast 开销，
  不与论文 `q_up + o_proj` 136 MiB 的数字混在一起。
- layer 使用前同步 broadcast materialized full logical weight。
- layer 完成后释放/reuse buffer。

测试：

- owner 分布按 `layer_id % cp_world_size`。
- broadcast 后所有 ranks materialized weight 一致。
- logits parity 对比 replicated full-weight path（Commit 4）。
- W8A8 量化权重加载：full logical 或 layer-owner weight 能从 TP-sharded
  量化 checkpoint 正确加载；至少覆盖一种常见量化格式。
- 多层 forward：61 层 repeated free 后无内存泄漏。

### Commit 7：Async Broadcast / Prefetch

变更：

- 增加 `K` 层预取，默认 `K=2`。
- 明确通信 stream、compute stream、event wait 和 cleanup。
- 保留 synchronous fallback。

测试：

- prefetch on/off logits parity。
- overlap 验证：CUDA/host events 确认每次 broadcast 的发起早于下一层依赖
  它的计算，且结束早于消费该权重的 `o_proj` / `q_up_proj` 调用。
- 边界情况：
  - 第 0 层：prefetch 第 `K` 层在第 0 层计算前发起；无须等待不存在的上一层
    broadcast。
  - 最后 `K` 层：prefetch 目标超出 `num_layers-1` 时静默跳过。
  - 少层模型（如 3 层稠密 MLP 堆叠）：`K=2` prefetch 不越界、不卡死。
- 多层 repeated forward stress test：61 层模型，连续 forward 100 次，每次
  token 分布不同；无内存泄漏、无 CUDA error、无 NCCL timeout。
- 无 stale weight buffer 使用：`Free(l)` 后 buffer 句柄失效，guard 或
  sanitizer 能捕获 use-after-free。
- synchronous fallback：当 async 路径初始化失败（如 stream 创建失败）时，
  同步路径产生正确结果。
- 短序列场景（`T <= 512`）：broadcast 可能无法被计算完全掩盖，但仍须产生
  正确的 logits。

## 验证矩阵

| Case | CP/TP world | Batch | Expected |
| --- | --- | --- | --- |
| config：无 `index_topk` | 2 | n/a | 拒绝 flag |
| config：无 sparse MLA backend | 2 | n/a | 拒绝 flag |
| config validation | 1 | n/a | 拒绝 flag |
| CP partition | 2/4/16 | uneven T | ordered gather/scatter，无间隙覆盖 `[0,T)` |
| embedding → LMHead round trip | 2 | prefill | shape 和 order 正确，all-gather 后 hidden 一致 |
| metadata slicing | 2 | prefill | local rows/global slots 对齐，字段不变量成立 |
| metadata：3 个不等长请求 | 2 | [100,200,50] | rank 0 持有 req 0+1，rank 1 持有 req 2 |
| metadata：FP8 mixed | 2 | mixed prefill/decode | 拒绝并给出明确错误信息 |
| KV cache write/read | 2 | prefill | 写入值从 global slot ID 正确 round-trip |
| attention hidden parity | 2 | one prompt | attention hidden states 与 baseline 一致 |
| attention logits parity | 2/4 | one prompt | logits 在容忍度内（BF16: rel<2e-3, abs<3e-2) |
| attention logits parity | 2/4 | uneven T | logits 在容忍度内 |
| multi-layer attention parity | 2 | one prompt | 每层 hidden states 与 baseline 一致（≥2 层） |
| top-k correctness | 2 | one prompt | 映射回全局后与 baseline 选择一致 |
| KV compact concat/split | 2 | prefill | all-gather round trip 后值无损 |
| MoE CP shape invariant | 2 | routed tokens | 输出保持 `[T_local, hidden]` |
| MoE routing order | 2 | routed tokens | all-gather 后 routing 与 baseline pre-CP routing 一致 |
| dense MLP CP equivalence | 2 | prefill | 与全局路径精确一致 |
| MoE residual correctness | 2 | routed tokens | attention 输出 + MoE 输出对应正确全局 token 对 |
| shard linear：ownership | 2/4 | n/a | `layer_id % cp_world_size` 分布 |
| shard linear：quantized load | 2 | W8A8 | 权重加载并与 replicated 路径一致 |
| shard linear：logits | 2/4 | prefill | 与 replicated weights（Commit 4）一致 |
| shard linear：no memory leak | 2/4 | 61 layers | repeated free 后无增长 |
| prefetch：parity | 2/4 | prefill | 与 sync Shard Linear 一致 |
| prefetch：overlap | 2/4 | prefill | broadcast 在消费计算前完成 |
| prefetch：boundary layers | 2/4 | prefill | 无越界 prefetch，末层无 stall |
| prefetch：stress | 2/4 | 100 passes | 无泄漏，无 NCCL timeout |
| prefetch：stale buffer | 2/4 | prefill | use-after-free guard 捕获复用 |
| prefetch：short sequence | 2/4 | T≤512 | logits 正确 |
| flag off | 2 | existing tests | 行为不变 |

BF16 / FP8 sparse MLA 使用合适 tolerance，不要求 floating-point exact equality。

GPU/NPU 环境可用后建议命令：

```bash
.venv/bin/python -m pytest tests/distributed/test_sharded_cp_utils.py -v
.venv/bin/python -m pytest tests/distributed/test_sharded_cp_metadata.py -v
.venv/bin/python -m pytest tests/models/test_deepseek_v32_sharded_cp.py -v
.venv/bin/python -m pytest tests/distributed/test_sharded_cp_moe.py -v
.venv/bin/python -m pytest tests/distributed/test_sharded_cp_shard_linear.py -v
```

遵守仓库规则：使用 `.venv/bin/python` 和 `uv`，不要使用 system `python3` 或裸
`pip`。

## 主要风险

- CP hidden layout 贯穿 Transformer 主体，会触碰 embedding、attention、MoE、
  LMHead 和 scheduler/metadata 边界，blast radius 大于 attention-local MVP。
- Sparse metadata slicing 是最大 correctness 风险。任何 token-indexed metadata
  保持 global row indexing 都可能导致 cache write 或 top-k row 静默错位。
- vLLM 当前 KV manager 若默认 replicated ownership，需要新增 CP-aware prefill
  ownership；否则无法达到论文中的 KV 显存/访存收益。
- MoE 路径需要保证 routing metadata 的 all-gather 顺序与 EP dispatch 假设一致。
- 稠密 MLP 层与 Sharded-CP：论文仅覆盖 MoE 层，但 DeepSeek V3.2 有 3 层初
  始稠密 MLP。稠密 MLP 执行逐 token 变换、天然 CP 兼容，但必须显式验证。
- Quantized loaders 可能假设 TP-sharded parameter shapes；full logical 或
  layer-owner weights 必须逐量化模式验证。
- `kv_b_proj` 范围：是否在运行时存在以及是否纳入 Shard Linear 取决于
  backend 的 weight absorption 实现。Commit 4 必须做出判断；Commit 6 必须与
  之一致。
- Async prefetch 改变权重 buffer 生命周期和 stream ordering，应晚于同步版本
  落地。
- 短序列下 broadcast 不一定能完全被计算掩盖，收益会弱于论文长上下文场景。
- CUDA graph capture 可能固化 token counts 和 metadata shapes。ShardedCP 布局
  稳定前应禁用或绕过 graph capture。

## 推荐实现顺序

先实现论文语义的数据布局，但用 replicated full logical weights 做 correctness：

```text
EmbeddingTP
  -> CP hidden
  -> local DSA attention with full heads + compact KV AG
  -> local MoE preprocess + EP compatible gather/RS
  -> CP hidden across layers
  -> LMHeadTP
```

确认 logits parity 和 metadata/cache ownership 后，再引入 synchronous
Shard Linear。最后做 async broadcast/prefetch，把通信隐藏在 `q_up_proj`、
Indexer scoring、Top-k 和 SparseFlashAttention 的计算窗口中。
