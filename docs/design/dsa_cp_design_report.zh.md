# DeepSeek V3.2 Sharded Context Parallel (DSA-CP) 设计报告

## 1. 总体概述

### 1.1 项目背景

Sharded Context Parallel (Sharded-CP) 是为 DeepSeek V3.2 DSA (Dynamic Sparse Attention) 稀疏 MLA 模型设计的一种上下文并行方案。其核心思想是将 prefill 阶段的长序列 token 行按 CP rank 分片，每个 rank 仅处理自己拥有的 token 子集，通过 all-gather / reduce-scatter 通信原语在层间同步。

### 1.2 设计原则

1. **单设备 CP 粒度**：复用 TP (Tensor Parallel) 进程组作为 CP 组，无需额外的 NCCL communicator
2. **CP-local hidden-state 布局**：Transformer 层间保持 CP-local 的 hidden states，减少通信量
3. **零冗余 Indexer**：通过 compact KV all-gather 实现全头稀疏 MLA attention，避免 Indexer-K 的重复计算
4. **Shard Linear 权重分片**：每层的 Q/KV/O 投影权重仅在 owner rank 上持久化，其他 rank 通过 broadcast 按需物化
5. **MoE CP 适配**：MoE 层通过 all-gather hidden + router → 全局路由 → reduce-scatter 输出的方式适配
6. **空 rank 保护**：当 token 数少于 CP world_size 时，部分 rank 持有零行 tensor，所有算子层（Linear、LayerNorm、Activation）均需安全处理空 batch

### 1.3 两个 Commit 的分工

| Commit | 功能 |
|--------|------|
| `422fecc7d` (Add sharded CP layer weight sharding) | 引入 `ShardedCPShardLinearLayer` 权重分片辅助类 |
| `0ffe312be` (Add DeepSeek sharded context parallel) | 完整的 Sharded-CP 推理管线：token 分片、元数据本地化、attention 通信、MoE 适配、模型集成 |

---

## 2. 模块架构

```
                         ┌─────────────────────┐
                         │  vllm/config/        │
                         │  parallel.py / vllm.py│  配置层：开关 & 校验
                         └────────┬────────────┘
                                  │
                         ┌────────▼────────────┐
                         │  engine/arg_utils.py  │  CLI 参数注册
                         └────────┬────────────┘
                                  │
           ┌──────────────────────┼──────────────────────┐
           │                      │                      │
  ┌────────▼──────────┐  ┌───────▼────────────┐  ┌──────▼──────────────┐
  │ distributed/       │  │ model_executor/     │  │ v1/attention/       │
  │ sharded_cp_utils.py│  │ layers/             │  │ backends/mla/       │
  │ (token 分片通信)    │  │ (算子层适配)         │  │ (元数据本地化)       │
  └────────────────────┘  └───────┬────────────┘  └──────┬──────────────┘
                                  │                      │
                         ┌────────▼────────────┐         │
                         │ models/deepseek_v2.py│◄────────┘
                         │ (模型层集成)          │
                         └────────┬────────────┘
                                  │
                         ┌────────▼────────────┐
                         │ v1/worker/           │
                         │ gpu_model_runner.py   │  推理运行时
                         └─────────────────────┘
```

---

## 3. 文件级详细文档

---

### 3.1 `vllm/config/parallel.py` — 并行配置

#### 新增字段

##### `enable_sharded_context_parallel: bool = False`
- **用途**：全局开关，启用 Sharded Context Parallelism
- **约束**：
  - `tensor_parallel_size > 1` (需要复用 TP 组)
  - `pipeline_parallel_size == 1` (不支持 PP)
  - `prefill_context_parallel_size == 1` (与 PCP 互斥)
  - `decode_context_parallel_size == 1` (与 DCP 互斥)
  - 不支持 DBO/ubatching

#### 新增校验逻辑（在 `_validate_parallel_config` 中）

当 `enable_sharded_context_parallel=True` 时，校验上述五条拓扑约束，任一不满足则抛出 `ValueError`。这是纯拓扑级校验，不依赖模型或 attention backend。

---

### 3.2 `vllm/config/vllm.py` — 顶层配置

#### 新增常量

##### `SHARDED_CP_SPARSE_MLA_BACKENDS: frozenset`
- **值**：`{FLASHMLA_SPARSE, FLASHINFER_MLA_SPARSE, ROCM_AITER_MLA_SPARSE, XPU_MLA_SPARSE}`
- **用途**：Sharded-CP 允许的 attention backend 白名单

#### 新增方法

##### `VllmConfig._validate_sharded_context_parallel_config() -> None`
- **作用**：跨配置校验 Sharded-CP 的前置条件
- **校验内容**：
  1. `model_config` 存在
  2. 模型使用 MLA (`use_mla=True`)
  3. HF config 包含 `index_topk`（DSA 稀疏 MLA 模型标志）
  4. 若显式指定了 attention backend，必须在白名单内
  5. 不支持 speculative decoding
  6. 不支持 full CUDA graphs（仅 NONE / PIECEWISE）
- **调用时机**：在 `__post_init__` 的配置验证阶段末尾

---

### 3.3 `vllm/engine/arg_utils.py` — 命令行参数

#### 变更

- 新增 `enable_sharded_context_parallel` 字段，默认 `False`
- 注册 CLI 参数 `--enable-sharded-context-parallel`
- 在构建 `ParallelConfig` 时传入该值

---

### 3.4 `vllm/distributed/sharded_cp_utils.py` — Token 分片与通信

此文件是 Sharded-CP 的核心分布式工具库，包含 token 行分片、padding、all-gather、reduce-scatter 等全部原语。设计为纯布局工具，不依赖 DeepSeek 模型类，可在 CPU 上单元测试。

#### 数据结构

##### `ShardedCPTokenRange` (frozen dataclass)
```
rank: int              — 此 rank 的序号
world_size: int        — CP 组大小
start: int             — 全局 token 行的起始索引 (inclusive)
end: int               — 全局 token 行的结束索引 (exclusive)
padded_end: int        — padding 后的结束索引，保证 all ranks 等宽
total_tokens: int      — 全局 token 总数
padded_num_rows: int | None  — 显式指定的 padded 行数（request-aligned 时使用）
rank_starts: tuple     — 所有 rank 的 start 边界
rank_ends: tuple       — 所有 rank 的 end 边界
local_request_starts: tuple   — 本地请求片段的起始偏移
local_request_ends: tuple     — 本地请求片段的结束偏移
local_request_global_starts: tuple — 本地请求片段对应的全局请求起始位置
local_request_indices: tuple  — 本地请求片段在全局请求列表中的索引
```

**属性**：
- `num_tokens -> int`：实际 token 数 `end - start`
- `padded_num_tokens -> int`：padding 后的 token 数
- `has_explicit_rank_ranges -> bool`：是否有显式的每 rank 范围
- `has_local_request_fragments -> bool`：是否有本地请求片段信息

#### 函数

##### `get_sharded_cp_group() -> GroupCoordinator`
- **作用**：返回 Sharded-CP 使用的进程组
- **实现**：复用 TP 进程组 (`get_tp_group()`)

##### `get_sharded_cp_token_range(num_tokens, rank, world_size) -> ShardedCPTokenRange`
- **作用**：为一个 CP rank 计算均匀连续 token 范围
- **分片策略**：`chunk = ceil(num_tokens / world_size)`，每个 rank 持有 `[rank*chunk, min((rank+1)*chunk, num_tokens))`
- **padding**：所有 rank 的 padded 大小为 `chunk`，保证 all-gather 输入等形

##### `get_request_aligned_sharded_cp_token_ranges(query_start_loc_cpu, world_size) -> tuple[ShardedCPTokenRange, ...]`
- **作用**：返回请求对齐的 token 范围（每个请求的 query 行不会被拆分到不同 rank）
- **算法**：
  1. 计算名义 chunk 大小 `ceil(total / world_size)`
  2. 每个 rank 边界向上对齐到最近的请求边界（`bisect_left`）
  3. padded_rows 取所有 rank 中最大的实际行数
- **用途**：确保 Indexer 的 prefill metadata 在单 rank 内是完整的

##### `get_request_aligned_sharded_cp_token_range(query_start_loc_cpu, rank, world_size) -> ShardedCPTokenRange`
- **作用**：单 rank 版本的请求对齐分片

##### `pad_for_token_all_gather(x, token_range, pad_value=0.0) -> Tensor`
- **作用**：将本地 token 行 padding 到 CP chunk 大小
- **约束**：`x.shape[0]` 必须等于 `token_range.num_tokens`

##### `trim_token_all_gather(x, token_range) -> Tensor`
- **作用**：将 all-gather 后的 padded tensor 截断到全局 token 数

##### `assemble_token_all_gather_chunks(gathered, token_range) -> Tensor`
- **作用**：将 padded all-gather chunks 按全局 token 行顺序拼接
- **逻辑**：若有显式 rank ranges，每个 chunk 只取 `[0, end-start)` 的真实行；否则直接 cat + trim

##### `class TokenRowsAllGather`
- **作用**：all-gather 的延迟结果（deferred result）容器
- **成员**：
  - `wait() -> Tensor`：等待 NCCL 操作完成，组装并返回全局 tensor
  - 内部管理 `_work` (NCCL 句柄)、`_stream` (CUDA stream)、`_gathered` (各 rank chunks)

##### `all_gather_token_rows_async(x, token_range, group, pad_value) -> TokenRowsAllGather`
- **作用**：发起异步 token 行 all-gather
- **实现**：
  1. 先 pad 本地行
  2. 在独立 CUDA stream 上发起 `dist.all_gather(async_op=True)`
  3. 返回 `TokenRowsAllGather` 句柄

##### `all_gather_token_rows(x, token_range, group, pad_value) -> Tensor`
- **作用**：同步版本，等待 all-gather 完成并返回全局 tensor

##### `make_reduce_scatter_token_chunks(x, token_range, pad_value) -> list[Tensor]`
- **作用**：将全局 token 行切分为等大小的 reduce-scatter 输入 chunks
- **逻辑**：按 rank ranges 切分并 padding 每个 chunk

##### `reduce_scatter_token_rows(x, token_range, group, pad_value) -> Tensor`
- **作用**：对全局 token 行执行 reduce-scatter，返回本 rank 的 CP-local 行
- **流程**：切分 chunks → `dist.reduce_scatter` → 截断 padding

##### `shard_global_token_rows(x, rank, world_size, pad_value) -> tuple[Tensor, ShardedCPTokenRange]`
- **作用**：将全局 tensor 按 rank 切片，返回本地 padded chunk + token range

##### `slice_for_token_reduce_scatter(x, token_range, pad_value) -> Tensor`
- **作用**：纯切片版本，不执行分布式操作。从全局 tensor 中取出本 rank 的 padded 子集

---

### 3.5 `vllm/model_executor/layers/sharded_cp_shard_linear.py` — 权重分片

实现 Shard Linear 机制：每层的 Q/KV/O 投影权重仅在 `layer_id % world_size` 的 owner rank 上持久化存储，其他 rank 在层执行前通过 broadcast 物化参数，执行后释放。

#### 函数与类

##### `get_sharded_cp_shard_linear_owner(layer_id, world_size) -> int`
- **作用**：返回一层权重的 owner rank
- **算法**：`layer_id % world_size`（round-robin 分布）

##### `ShardedCPShardLinearParam` (frozen dataclass)
- **字段**：`name, shape, dtype, device, owner_rank`
- **用途**：描述一个 Shard Linear 参数的元信息

##### `class ShardedCPShardLinearPrefetch`
- **作用**：异步 broadcast 的句柄，实现 K=2 的 prefetch 流水线
- **方法**：
  - `layer_id -> int`：关联的层 ID
  - `released -> bool`：是否已释放
  - `wait() -> None`：等待所有 broadcast 句柄完成，同步 CUDA streams
  - `release() -> None`：等待 + 释放非 owner rank 的物化参数
  - `materialized() -> contextmanager`：上下文管理器，wait → yield → release

##### `class ShardedCPShardLinearLayer`
- **作用**：管理一层 decoder layer 的权重物化/释放生命周期
- **构造**：`__init__(layer_id, modules)` — `modules` 是 `(name, nn.Module)` 对的列表
- **方法**：
  - `_iter_params() -> Iterator`：遍历所有注册模块的参数
  - `_record_param_metadata(param)` (static)：首次调用时记录参数的 shape/dtype/device
  - `_param_shape/dtype/device(param)` (static)：从缓存的 metadata 读取原始形状/类型/设备
  - `describe_parameters(group) -> list[ShardedCPShardLinearParam]`：返回参数元信息列表
  - `release_non_owner(group)`：非 owner rank 将参数 data 替换为空 tensor `(0,)`，释放显存
  - `_broadcast(tensor, owner_rank, group)`：同步 broadcast 一个 tensor
  - `_can_async_broadcast(group) -> bool`：检查是否支持异步 broadcast（需要 `device_group` 和 `ranks`）
  - `_broadcast_async(tensor, owner_rank, group, stream) -> Work | None`：在指定 stream 上发起异步 broadcast
  - `_materialize_params(group, async_broadcast) -> (handles, streams)`：核心物化逻辑：
    1. 非 owner rank 分配全尺寸 tensor
    2. owner rank 验证 tensor 形状未变
    3. 同步或异步 broadcast 所有参数
    4. 异常时清理已发起的 handles 和释放已分配的 tensors
  - `materialize(group)`：同步物化（等待 broadcast 完成）
  - `prefetch(group) -> ShardedCPShardLinearPrefetch`：异步物化，返回可等待句柄
  - `materialized(group) -> contextmanager`：同步物化的上下文管理器

---

### 3.6 `vllm/v1/attention/backends/mla/sharded_cp_metadata.py` — 元数据本地化

将全局 attention metadata（CommonAttentionMetadata、DeepseekV32IndexerMetadata、FlashMLASparseMetadata）转换为 CP rank 本地的子集视图。

#### 辅助数据结构

##### `_LocalRequestFragment` (frozen dataclass)
- **用途**：描述一个请求在本地 token range 中的片段
- **字段**：`request_index, local_start, local_end, global_start, global_end, request_global_start, request_global_end`
- **属性**：
  - `num_tokens`：片段 token 数
  - `start_offset_in_request`：在原始请求中的起始偏移
  - `end_offset_in_request`：在原始请求中的结束偏移

#### 内部辅助函数

##### `_request_slice_for_token_range(query_start_loc_cpu, token_range) -> slice`
- **作用**：找到 token range 对应的请求切片 `[req_start, req_end)`
- **约束**：token range 边界必须与请求边界对齐

##### `_request_fragments_for_token_range(query_start_loc_cpu, token_range) -> tuple[_LocalRequestFragment, ...]`
- **作用**：计算 token range 与所有请求的交集片段
- **算法**：从 `bisect_right(starts, token_range.start) - 1` 开始，逐请求计算 overlap

##### `_local_query_start_loc_cpu_from_fragments(fragments) -> Tensor`
- **作用**：从 fragments 构造本地的 `query_start_loc` (CPU tensor)

##### `_chunk_query_start_loc_cpu_from_fragments(fragments) -> Tensor`
- **作用**：从 fragments 构造 chunk 内的相对 `query_start_loc`

##### `_local_query_start_loc_like(query_start_loc, fragments) -> Tensor`
- **作用**：构造与原始 `query_start_loc` 同 device/dtype 的本地版本

##### `_token_range_with_request_fragments(token_range, fragments) -> ShardedCPTokenRange`
- **作用**：将 fragment 信息注入 token_range 的可选字段

##### `_slice_optional_fragments(x, fragments) -> Any`
- **作用**：按 fragment 的 request_index 对 Tensor/list/tuple 做 index_select

##### `_slice_optional_req_tensor(x, request_slice) -> Any`
- **作用**：对可选 tensor 做 slice

##### `_seq_lens_cpu(metadata) -> Tensor`
- **作用**：获取 CommonAttentionMetadata 的 CPU seq_lens

##### `_num_computed_tokens_cpu(metadata) -> Tensor`
- **作用**：计算 `seq_lens - query_lens`

##### `_fragment_seq_lens_cpu(global_seq_lens_cpu, global_num_computed_tokens_cpu, fragments) -> Tensor`
- **作用**：计算每个 fragment 的 effective seq_len = `num_computed_tokens + end_offset_in_request`

##### `_fragment_num_computed_tokens_cpu(global_num_computed_tokens_cpu, fragments) -> Tensor`
- **作用**：计算每个 fragment 的 num_computed_tokens = `global_computed + start_offset_in_request`

##### `_index_select_fragments(x, fragments) -> Tensor`
- **作用**：按 fragment 索引选取 tensor 行

#### 主要公开函数

##### `get_sharded_cp_token_range_from_forward_context(forward_context, rank, world_size) -> ShardedCPTokenRange`
- **作用**：从 forward context 的 attention metadata 中计算当前 rank 的 token range
- **流程**：读取第一个 layer 的 `query_start_loc` → 计算 balanced range → 附加 request fragments

##### `localize_common_attention_metadata(common_attn_metadata, token_range) -> CommonAttentionMetadata`
- **作用**：构造 CP rank 本地的 CommonAttentionMetadata
- **处理内容**：
  - `query_start_loc` → 本地化
  - `seq_lens` → 按 fragment 重算
  - `num_computed_tokens` → 按 fragment 重算
  - `block_table_tensor` → index_select
  - `slot_mapping` → 按 token range 切片
  - `encoder_seq_lens`, `dcp_local_seq_lens` → 按 fragment 切片

##### `localize_deepseek_v32_indexer_metadata(metadata, token_range, use_global_compact_kv) -> DeepseekV32IndexerMetadata`
- **作用**：将全局 Indexer 元数据本地化
- **处理内容**：
  - 拆分 decode/prefill fragments
  - 重算 `num_decodes`, `num_prefills`, `num_decode_tokens`, `num_prefill_tokens`
  - 调用 `_slice_indexer_decode_metadata` 处理 decode 部分
  - 调用 `_localize_indexer_prefill_metadata` 处理 prefill 部分
  - 设置 `k_is_global_compact` 标志

##### `_localize_indexer_prefill_metadata(...) -> DeepseekV32IndexerPrefillMetadata | None`
- **作用**：将全局 prefill metadata 的 chunks 本地化
- **算法**：
  1. 遍历全局 chunks
  2. 过滤出与 token range 有交集的 chunks
  3. 映射 fragment → chunk 内请求
  4. 拆分为连续 fragment 组 (`_split_contiguous_fragments`)
  5. 每组构建一个本地 `DeepseekV32IndexerPrefillChunkMetadata`

##### `_build_indexer_prefill_chunk(...)  -> DeepseekV32IndexerPrefillChunkMetadata`
- **作用**：为一组连续 fragments 构建 Indexer prefill chunk
- **关键逻辑**：
  - 当 `use_global_compact_kv=True` 时，`cu_seqlen_ks` 和 `cu_seqlen_ke` 使用全局偏移
  - 否则使用标准的 `kv_spans_from_batches` 分页地址

##### `_slice_indexer_decode_metadata(metadata, fragments, global_num_computed_tokens_cpu, local_max_seq_len) -> DeepSeekV32IndexerDecodeMetadata | None`
- **作用**：为 decode 请求构建本地 metadata
- **逻辑**：
  - 过滤 decode fragments
  - 逐 token 构建 block_table 和 seq_lens
  - 重算 `schedule_metadata`（DeepGEMM paged MQA logits）

##### `localize_flashmla_sparse_metadata(metadata, token_range, use_global_compact_kv) -> FlashMLASparseMetadata`
- **作用**：本地化 FlashMLA sparse metadata
- **处理**：
  - `query_start_loc` → 本地化
  - `req_id_per_token` → 重映射为本地请求索引
  - `block_table` → index_select
  - `slot_mapping` → 按 token range 切片
  - 设置 `topk_indices_are_global_compact_offsets` 标志

##### `localize_sharded_cp_attention_metadata(attn_metadata, token_range, use_global_compact_kv) -> AttentionMetadata`
- **作用**：统一分发器，根据 metadata 类型调用对应的本地化函数

##### `build_sharded_cp_attention_metadata(attn_metadata, token_range, use_global_compact_kv) -> dict[str, AttentionMetadata]`
- **作用**：对整个 per-layer attention metadata dict 执行本地化

##### `_use_global_compact_kv_for_sharded_cp(attn_metadata) -> bool`
- **作用**：判断当前 batch 是否可以使用 global compact KV
- **条件**：纯 first-prefill batch（无 decode、seq_lens == query_lens）

##### `sharded_cp_forward_context(forward_context, token_range) -> contextmanager`
- **作用**：将全局 ForwardContext 替换为 CP 本地版本
- **流程**：
  1. 计算 request fragments
  2. 判断是否使用 global compact KV
  3. 构建本地 attention metadata dict
  4. 提取本地 slot mapping
  5. 创建 local ForwardContext，注入 `sharded_cp_global_slot_mapping`、`sharded_cp_token_range`、`sharded_cp_use_global_compact_kv`
  6. 通过 `override_forward_context` 覆盖当前上下文

---

### 3.7 `vllm/v1/attention/sharded_cp_attention.py` — Compact KV 通信

实现 MLA compact KV + Indexer-K 的打包 all-gather，是 attention 层通信的核心路径。

#### 数据结构

##### `ShardedCPCompactKVLayout` (frozen dataclass)
- **字段**：`kv_lora_rank, qk_rope_head_dim, indexer_head_dim`
- **属性**：
  - `kv_dim -> int`：`kv_lora_rank + qk_rope_head_dim`
  - `total_dim -> int`：`kv_dim + indexer_head_dim`

#### 函数

##### `pack_sharded_cp_compact_kv(kv_c_normed, k_pe, indexer_k) -> Tensor`
- **作用**：将 MLA compact KV 和 Indexer-K 打包为一行 payload
- **布局**：`[kv_c_normed || k_pe || indexer_k]`（沿最后一维拼接）
- **校验**：所有输入行数相同、device/dtype 一致

##### `split_sharded_cp_compact_kv(compact_kv, layout) -> (kv_c_normed, k_pe, indexer_k)`
- **作用**：将打包的 payload 拆分回三个 tensor
- **k_pe**：拆分后 unsqueeze(1) 恢复 `[N, 1, rope_dim]` 形状

##### `assemble_sharded_cp_compact_kv_chunks(gathered, token_range, layout) -> (kv_c_normed, k_pe, indexer_k)`
- **作用**：从 padded all-gather chunks 组装全局 compact KV 并拆分

##### `class ShardedCPCompactKVAllGather`
- **作用**：异步 compact KV all-gather 的句柄
- **方法**：
  - `wait() -> (kv_c_normed, k_pe, indexer_k)`：等待底层 `TokenRowsAllGather` 完成并拆分
  - `release()`：释放（内部调用 wait）

##### `all_gather_sharded_cp_compact_kv_async(kv_c_normed, k_pe, indexer_k, token_range, group, pad_value) -> ShardedCPCompactKVAllGather`
- **作用**：发起异步 compact KV all-gather
- **流程**：pack → `all_gather_token_rows_async` → 包装为 `ShardedCPCompactKVAllGather`

##### `all_gather_sharded_cp_compact_kv(kv_c_normed, k_pe, indexer_k, token_range, group, pad_value) -> (kv_c_normed, k_pe, indexer_k)`
- **作用**：同步版本

##### `sharded_cp_topk_prefix(topk_indices_buffer, token_range) -> Tensor`
- **作用**：返回 topk_indices_buffer 中本 rank 拥有的 token 行前缀

---

### 3.8 `vllm/model_executor/layers/fused_moe/sharded_cp_moe.py` — MoE 适配

描述 MoE 层在 Sharded-CP 下的 token 行通信模式。

#### 数据结构

##### `ShardedCPMoEInputs` (frozen dataclass)
- **字段**：`hidden_states, router_logits, activation_scales (optional)`
- **用途**：all-gather 后的全局 MoE 输入

##### `ShardedCPMoERoutingMetadata` (frozen dataclass)
- **字段**：`topk_weights, topk_ids`
- **用途**：all-gather 后的全局 top-k 路由信息

#### 函数

##### `_validate_local_token_rows(name, x, token_range)`
- **作用**：验证 tensor 行数等于 `token_range.num_tokens`

##### `_validate_global_token_rows(name, x, token_range)`
- **作用**：验证 tensor 行数等于 `token_range.total_tokens`

##### `_validate_same_local_rows(lhs_name, lhs, rhs_name, rhs)`
- **作用**：验证两个 tensor 行数一致

##### `assemble_sharded_cp_moe_input_chunks(gathered_hidden_states, gathered_router_logits, token_range, gathered_activation_scales) -> ShardedCPMoEInputs`
- **作用**：从 gathered chunks 组装全局 MoE 输入

##### `all_gather_sharded_cp_moe_inputs(hidden_states, router_logits, token_range, activation_scales, group) -> ShardedCPMoEInputs`
- **作用**：all-gather 本地 MoE 激活和 router logits
- **流程**：分别 all-gather hidden_states、router_logits、activation_scales → 组装

##### `assemble_sharded_cp_moe_routing_chunks(gathered_topk_weights, gathered_topk_ids, token_range) -> ShardedCPMoERoutingMetadata`
- **作用**：从 gathered chunks 组装全局路由信息

##### `all_gather_sharded_cp_moe_routing_metadata(topk_weights, topk_ids, token_range, group) -> ShardedCPMoERoutingMetadata`
- **作用**：all-gather top-k 路由信息

##### `reduce_scatter_sharded_cp_moe_output(expert_output, token_range, group) -> Tensor`
- **作用**：reduce-scatter 全局 MoE 输出回到 CP-local 行

##### `slice_sharded_cp_moe_output(expert_output, token_range) -> Tensor`
- **作用**：从全局 tensor 中切出本 rank 的 MoE 输出行（无通信版本，用于测试）

##### `combine_sharded_cp_moe_residual(attention_output_local, moe_output_local, token_range) -> Tensor`
- **作用**：将 attention 和 MoE 的 CP-local 输出相加

---

### 3.9 `vllm/model_executor/layers/linear.py` — Linear 层空 batch 保护

#### 新增方法（`LinearBase`）

##### `_empty_output(input_, output_size) -> Tensor`
- **作用**：当输入 batch 为空时，创建正确形状的空输出 tensor

##### `_has_empty_batch(input_) -> bool`
- **作用**：检查输入 tensor 的 batch 维是否包含 0（即空 rank）

#### 修改的 forward 方法

##### `ReplicatedLinear.forward()`
- **变更**：在 `quant_method.apply` 前检查空 batch，空 batch 直接返回空输出

##### `ColumnParallelLinear.forward()`
- **变更**：同上，空 batch 时跳过 GEMM

##### `RowParallelLinear.forward()`
- **变更**：同上，空 batch 时跳过 GEMM

---

### 3.10 `vllm/model_executor/layers/layernorm.py` — LayerNorm 空 batch 保护

#### 修改

在 `RMSNorm` 的以下三个方法中添加了空 batch 短路：

##### `forward_native(x, residual)`
- 检查 `0 in x.shape[:-1]`，若为空直接返回输入

##### `forward_cuda(x, residual)`
- 同上

##### `forward_xpu(x, residual)` (隐含)
- 同上

---

### 3.11 `vllm/model_executor/layers/activation.py` — Activation 空 batch 保护

#### 修改

##### `SiluAndMul.forward_cuda(x)`
- 检查 `0 in x.shape[:-1]`，若为空直接返回空 output tensor

---

### 3.12 `vllm/model_executor/layers/vocab_parallel_embedding.py` — Embedding 适配

#### 修改

##### 新增 `forward_parallel(input_) -> Tensor`
- **作用**：返回 TP rank 的本地 embedding 贡献（即 masked embedding + zero-fill），**不**执行 all-reduce
- **用途**：Sharded-CP 需要在 reduce-scatter 而非 all-reduce 后得到 CP-local hidden states

##### `forward_native(input_)` 重构
- 原先的 embedding + mask + all-reduce 逻辑拆分为 `forward_parallel` + all-reduce

---

### 3.13 `vllm/model_executor/layers/attention/mla_attention.py` — MLA Attention 适配

#### 新增方法

##### `MLAAttention.update_kv_cache(kv_c_normed, k_pe, layer_slot_mapping)`
- **作用**：独立的 KV cache 更新方法（从 forward 中解耦）
- **用途**：Sharded-CP global compact KV 路径需要先 all-gather KV，再用全局 slot_mapping 写入 cache

#### 修改

##### `MLAAttention.forward(q, kv_c_normed, k_pe, output_shape, use_global_kv)`
- **新增参数**：`use_global_kv: bool = False`
- **global KV 路径**：
  - 跳过 KV cache 更新（已在 MLA wrapper 中完成）
  - 将 `kv_c_normed || k_pe` concat 为 `attn_kv` 直接传给 attention kernel（而非从 paged KV cache 读取）

---

### 3.14 `vllm/model_executor/layers/sparse_attn_indexer.py` — 稀疏 Indexer 适配

#### 新增辅助函数（模块级）

##### `_query_chunk_size_for_bytes(q, bytes_per_query_row, max_bytes) -> int`
- **作用**：根据内存限制计算 query chunk 大小

##### `_mqa_logits_torch_query_chunk_size(q, seq_len_kv) -> int`
- **作用**：为 torch fallback 路径计算 chunk 大小（限制 score tensor 不超过 64MB）

##### `_mqa_logits_deep_gemm_query_chunk_size(q, seq_len_kv) -> int`
- **作用**：为 DeepGEMM 路径计算 chunk 大小（限制 logits tensor 不超过 512MB）

##### `_topk_per_row_prefill(logits, cu_seqlen_ks, cu_seqlen_ke, topk_indices, topk_tokens)`
- **作用**：封装平台相关的 prefill top-k 调用 (CUDA/XPU)

##### `_fill_prefill_topk_from_logits(logits, cu_seqlen_ks, cu_seqlen_ke, topk_indices, topk_tokens, add_cu_seqlen_ks)`
- **作用**：从 logits 填充 prefill topk indices
- **关键参数**：`add_cu_seqlen_ks`—当使用 global compact KV 时，topk indices 需要加上 `cu_seqlen_ks` 偏移

##### `_fill_prefill_topk_from_mqa_logits_torch(q_fp8, kv, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices, topk_tokens, add_cu_seqlen_ks)`
- **作用**：torch fallback 路径的 chunked MQA logits → topk

##### `_validate_prefill_topk_rows(q_fp8, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices)`
- **作用**：验证所有 prefill topk 输入的行数一致

##### `_fill_prefill_topk_from_mqa_logits(q_fp8, kv, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices, topk_tokens, add_cu_seqlen_ks, use_deep_gemm)`
- **作用**：统一的 chunked prefill topk 填充，根据 `use_deep_gemm` 选择 kernel

##### `_fill_topk_from_global_compact_k(q_fp8, k, weights, attn_metadata, quant_block_size, scale_fmt, topk_tokens, topk_indices_buffer) -> Tensor`
- **作用**：Sharded-CP 核心路径—从 all-gathered 的 global compact Indexer-K 计算 top-k
- **流程**：
  1. 对全局 K 执行 `per_token_group_quant_fp8`
  2. 遍历每个 prefill chunk
  3. 调用 `_fill_prefill_topk_from_mqa_logits` 填充 topk（add_cu_seqlen_ks=True）

#### 修改

##### `sparse_attn_indexer(...)` — 主调度函数
- 新增 `k_is_global_compact` 分支：当为 True 时，直接调用 `_fill_topk_from_global_compact_k`
- 原有的 paged prefill 路径重构为使用 `_fill_prefill_topk_from_mqa_logits` 通用函数

#### `Indexer` 类新增方法

##### `project(hidden_states, qr, positions, rotary_emb) -> (q_fp8, k, weights)`
- **作用**：将原 `forward` 拆分为独立的 Q/K/weights 投影步骤

##### `project_q(qr, positions, rotary_emb) -> (q_fp8, q_scale)`
- **作用**：投影 Q 并量化为 FP8

##### `project_k(hidden_states, positions, rotary_emb, q_pe_for_rope=None) -> Tensor`
- **作用**：投影 K 并应用 RoPE

##### `project_weights(hidden_states, q_scale) -> Tensor`
- **作用**：计算 Indexer weights

##### `forward_global_compact(hidden_states, q_fp8, indexer_k_global, weights) -> Tensor`
- **作用**：使用全局 compact K 执行 Indexer forward（调用 `indexer_op.forward_global_compact`）

##### `update_local_k_cache(indexer_k, layer_slot_mapping=None)`
- **作用**：将 Indexer-K 写入 paged KV cache（使用 `indexer_k_quant_and_cache` 内核）
- **slot_mapping 来源**：优先使用显式传入的，否则从 forward context 获取

##### `forward_local_paged(hidden_states, q_fp8, indexer_k, weights) -> Tensor`
- **作用**：在 global compact KV 模式下，临时切换 metadata 为非 global-compact 模式执行 decode 的 paged Indexer

##### `SparseAttnIndexer.forward_global_compact(hidden_states, q_fp8, k, weights)`
- **作用**：CustomOp 级别的 global compact forward，委托给 `forward_native`

---

### 3.15 `vllm/model_executor/layers/mla.py` — MLA Wrapper 适配

#### `MLAModules` 新增字段

##### `enable_sharded_context_parallel: bool = False`
- 传递到 MLA wrapper

#### `MultiHeadLatentAttentionWrapper` 新增方法

##### `_sharded_cp_token_range() -> ShardedCPTokenRange | None`
- **作用**：从 forward context 获取当前 token range（仅当启用 sharded CP 且是 sparse attention 时）

##### `_use_global_compact_kv_for_sharded_cp() -> bool`
- **作用**：检查 forward context 中的 `sharded_cp_use_global_compact_kv` 标志

##### `_sharded_cp_indexer_metadata()`
- **作用**：获取 Indexer 层的 attention metadata

##### `_sharded_cp_global_slot_mapping(layer_name) -> Tensor | None`
- **作用**：获取全局 slot mapping（Sharded-CP 用全局 slot mapping 写 cache，而非本地的）

##### `_update_local_kv_cache_for_global_compact(kv_c_normed, k_pe, layer_slot_mapping)`
- **作用**：使用全局 token 的 KV 数据更新本地 KV cache

##### `_update_local_indexer_k_cache_for_global_compact(indexer_k)`
- **作用**：使用全局 Indexer-K 更新本地 Indexer K cache

##### `_forward_empty_sharded_cp(hidden_states, token_range, use_global_compact_kv) -> Tensor`
- **作用**：空 rank 路径—即使无 token，仍参与 all-gather 通信

#### `forward()` 重大修改

原始流程：
```
kv_a → kv_c + k_pe → RoPE(q, k_pe) → q_proj → Indexer → MLA Attention
```

Sharded-CP 流程：
```
kv_a → kv_c + k_pe → RoPE(k_pe only) →
  ├─ Indexer.project_k → pack(kv_c, k_pe, indexer_k) → all_gather_async (通信)
  ├─ q_proj → RoPE(q only) → Indexer.project_q + project_weights (计算，与通信重叠)
  └─ wait all_gather →
     ├─ update_kv_cache (全局 slot mapping)
     ├─ update_indexer_k_cache (全局 slot mapping)
     ├─ Indexer.forward_global_compact (全局 compact K → topk)
     └─ MLA Attention(use_global_kv=True)
```

关键优化：compact KV all-gather 与 Q 投影 + Indexer Q/weights 投影重叠执行。

---

### 3.16 `vllm/model_executor/layers/fused_moe/router/gate_linear.py` — Gate Linear 空 batch 保护

#### 修改

##### `GateLinear.forward(x)`
- 添加空 batch 检查：`self._has_empty_batch(x)` 时直接返回空输出

---

### 3.17 `vllm/model_executor/model_loader/utils.py` — 模型加载后处理

#### 修改

##### `process_weights_after_loading(model, ...)`
- 新增：调用模型的 `post_process_weights_after_loading()` 方法（如果存在）
- **用途**：DeepSeek 模型在加载权重后调用 `release_sharded_cp_non_owner_weights()` 释放非 owner 的 Shard Linear 参数

---

### 3.18 `vllm/model_executor/models/deepseek_v2.py` — 模型层集成

这是最大的文件变更，将所有 Sharded-CP 组件集成到 DeepSeek V2/V3.2 模型中。

#### 新增模块级函数

##### `_layer_id_from_prefix(prefix) -> int`
- **作用**：从模块前缀字符串（如 `model.layers.5.self_attn`）中提取层号

#### `DeepseekV2MoE` 修改

##### 新增字段
- `use_sharded_cp_token_parallel: bool` — 是否使用 Sharded-CP token 并行

##### `_get_sharded_cp_token_range() -> ShardedCPTokenRange`
- **作用**：从 forward context 获取 token range（强制要求存在）

##### `_maybe_get_sharded_cp_token_range() -> ShardedCPTokenRange | None`
- **作用**：尝试获取 token range（允许不存在）

##### `_apply_moe_output_epilogue(hidden_states, shared_output, final_hidden_states) -> Tensor`
- **作用**：提取原 forward 中的 MoE 后处理逻辑：routed_scaling_factor、shared_experts 输出合并
- **重构动机**：Sharded-CP 和非 Sharded-CP 路径共享此逻辑

##### `_forward_sharded_cp(hidden_states, token_range) -> Tensor`
- **作用**：Sharded-CP 下的 MoE forward
- **流程**：
  1. 本地 gate routing（CP-local hidden → router_logits）
  2. all-gather hidden_states + router_logits → 全局 MoE 输入
  3. 全局 expert dispatch + compute
  4. epilogue（scaling、shared experts）
  5. reduce-scatter 输出回 CP-local

##### `forward()` 修改
- 在 forward 入口检查 `use_sharded_cp_token_parallel`
- 若有 token range，走 `_forward_sharded_cp` 路径

#### `DeepseekV2MLAAttention` 修改

##### 新增字段
- `is_v32: bool` — 是否为 V3.2 模型
- `enable_sharded_context_parallel: bool`
- `use_sharded_cp_full_attention: bool` — 两者的交集
- `sharded_cp_shard_linear: ShardedCPShardLinearLayer | None`

##### 构造函数变更
- 当 `use_sharded_cp_full_attention=True` 时：
  - `q_b_proj` / `q_proj` / `kv_b_proj` 设置 `disable_tp=True`（不做 TP 切分）
  - `o_proj` 设置 `reduce_results=False, disable_tp=True`
  - `num_local_heads` 改为使用全量 `num_heads`
  - 创建 `ShardedCPShardLinearLayer` 管理权重生命周期
  - `MLAModules.enable_sharded_context_parallel = True`

#### `DeepseekV2DecoderLayer` 修改

##### 新增方法

##### `prefetch_sharded_cp_attention_weights(group) -> ShardedCPShardLinearPrefetch | None`
- **作用**：发起下一层（或后续层）的异步权重 broadcast

##### `_materialized_sharded_cp_attention_weights(prefetch) -> contextmanager`
- **作用**：返回上下文管理器，确保 attention 执行时权重已物化

##### `forward()` 修改
- 新增 `sharded_cp_prefetch` 参数
- 在 `self.self_attn(...)` 外包裹 `_materialized_sharded_cp_attention_weights`

#### `DeepseekV2Model` 修改

##### 新增方法

##### `_maybe_scatter_to_sharded_cp(hidden_states, positions, reduce_hidden_states) -> (Tensor, Tensor)`
- **作用**：将全局 hidden states 和 positions 分散到 CP-local
- **两种模式**：
  - `reduce_hidden_states=True`：embedding 是 TP-parallel 的，需要 reduce-scatter
  - `reduce_hidden_states=False`：直接切片（用于 inputs_embeds 已 all-reduced 的情况）

##### `_sharded_cp_forward_context() -> contextmanager`
- **作用**：在 Transformer 层循环外包裹 Sharded-CP forward context

##### `_release_pending_sharded_cp_prefetches(prefetches) -> None`
- **作用**：释放所有未消费的 prefetch 句柄

##### `release_sharded_cp_non_owner_weights() -> None`
- **作用**：遍历所有层，调用 `shard_linear.release_non_owner` 释放非 owner 权重

##### `forward()` 重大修改
- **Embedding 阶段**：
  - 使用 `embed_tokens.forward_parallel` 获取 TP-local embedding
  - 调用 `_maybe_scatter_to_sharded_cp` 执行 reduce-scatter
- **Transformer 循环**：
  - 包裹在 `_sharded_cp_forward_context()` 中
  - 实现 K=2 的 prefetch 流水线：
    1. 预启动前 2 层的 broadcast
    2. 每层执行时消费 prefetch + 启动后续层的 broadcast
  - finally 块确保异常时释放所有 pending prefetch

#### `DeepseekV2ForCausalLM` 修改

##### `prepare_hidden_states_for_logits(hidden_states) -> Tensor`
- **作用**：在计算 logits 前，将 CP-local hidden states all-gather 回全局

##### `compute_logits(hidden_states) -> Tensor | None`
- **修改**：在调用 `logits_processor` 前执行 `prepare_hidden_states_for_logits`

##### `post_process_weights_after_loading()`
- **作用**：加载权重后释放非 owner 的 Shard Linear 参数

---

### 3.19 Attention Backend 修改

以下四个 backend 文件均添加了 `topk_indices_are_global_compact_offsets: bool = False` 字段，并在 topk indices 转换逻辑中添加了分支：

#### `flashmla_sparse.py`
- `FlashMLASparseMetadata` 新增字段
- 三个 forward 方法（`_forward_bf16_kv`, `_forward_fp8_kv_separate`, `_forward_bf16_mixed_batch`）中：
  - 当 `topk_indices_are_global_compact_offsets=True` 时，跳过 `triton_convert_req_index_to_global_index`
- FP8 cache 路径：当 global compact 时禁用 FP8（尚不支持）

#### `flashinfer_mla_sparse.py`
- `FlashInferMLASparseMetadata` 新增字段
- `forward` 中同上分支

#### `rocm_aiter_mla_sparse.py`
- `ROCMAiterMLASparseMetadata` 新增字段
- `forward` 中同上分支

#### `xpu_mla_sparse.py`
- `XPUMLASparseMetadata` 新增字段
- `forward` 中同上分支

#### `indexer.py`
- `DeepSeekV32IndexerDecodeMetadata` 新增 `block_size: int = 64` 字段
- `DeepseekV32IndexerMetadata` 新增 `k_is_global_compact: bool = False` 字段
- Builder 传递 `block_size` 到 decode metadata

---

### 3.20 `vllm/v1/worker/gpu_model_runner.py` — 运行时适配

#### 新增方法

##### `_prepare_hidden_states_for_logits(hidden_states) -> Tensor`
- **作用**：调用模型的 `prepare_hidden_states_for_logits` hook（如果存在）

##### `_select_hidden_states_for_logits(hidden_states, logits_indices) -> (Tensor, Tensor)`
- **作用**：先 prepare（可能 all-gather），再 index-select

##### `_temporary_sharded_cp_kv_cache(should_init) -> contextmanager`
- **作用**：在 profiling/warmup 阶段临时创建最小 KV cache，以便 `_build_attention_metadata` 能生成 per-layer metadata
- **逻辑**：`_init_minimal_kv_cache_for_profiling(num_blocks=1)` → yield → `_cleanup_profiling_kv_cache()`

#### 修改

##### `_dummy_run()`
- 当 `enable_sharded_context_parallel=True` 时强制 `force_attention=True`
- 整个 dummy run 包裹在 `_temporary_sharded_cp_kv_cache` 中

##### `execute_model()` 相关
- `hidden_states[logits_indices]` 替换为 `_select_hidden_states_for_logits(hidden_states, logits_indices)`
- 确保 PP last rank 和非 last rank 路径都正确处理

##### `_init_minimal_kv_cache_for_profiling(num_blocks=None)`
- 新增 `num_blocks` 参数，允许指定最小 block 数

##### `_cleanup_profiling_kv_cache()`
- 添加 `torch.cuda.synchronize()` 确保异步操作完成

---

## 4. 通信模式总结

### 4.1 Embedding → 第一层
```
Embedding (TP-parallel, 全局行)
    → reduce-scatter (TP group = CP group)
    → CP-local hidden states
```

### 4.2 Attention 层
```
CP-local hidden → kv_a_proj → kv_c + k_pe
CP-local hidden → indexer.project_k → indexer_k
    ↓
pack(kv_c_normed, k_pe, indexer_k) → all-gather → 全局 compact KV
    ↓                                               ↓
    ├─ q_b_proj (重叠计算)                          ├─ update_kv_cache (全局 slots)
    ├─ indexer.project_q (重叠计算)                  ├─ update_indexer_k_cache
    └─ indexer.project_weights (重叠计算)            └─ indexer.forward_global_compact
                                                    ↓
                                          MLA Attention (全局 KV, CP-local Q)
                                                    ↓
                                          o_proj (无 TP reduce) → CP-local output
```

### 4.3 MoE 层
```
CP-local hidden + router_logits
    → all-gather → 全局 MoE 输入
    → gate routing + expert dispatch (全局)
    → reduce-scatter → CP-local MoE 输出
```

### 4.4 最后一层 → Logits
```
CP-local hidden
    → all-gather → 全局 hidden states
    → lm_head → logits
```

### 4.5 权重管线
```
Layer i 执行:     prefetch(i) 消费 → materialize → execute → release
Layer i+2 异步:   prefetch(i+2) 发起 broadcast ↗
```

---

## 5. 关键设计决策

### 5.1 复用 TP 组作为 CP 组
- **优势**：无需额外 NCCL communicator，降低初始化开销
- **限制**：CP world_size == TP world_size，Q/KV/O proj 必须 disable_tp

### 5.2 Global Compact KV vs Paged KV
- 纯 first-prefill batch 使用 global compact KV：all-gather 后直接用 concat KV 做 attention，避免 paged KV 寻址
- 混合 batch（含 decode）使用 paged KV：all-gather 后写入全局 paged cache，decode 走标准 paged 路径

### 5.3 Shard Linear 的 Round-Robin 分布
- `owner = layer_id % world_size`：均匀分布权重，避免单 rank 显存热点
- K=2 prefetch：前瞻 2 层的 broadcast 与当前层计算重叠

### 5.4 空 Rank 保护
- 当 `num_tokens < world_size` 时，部分 rank 持有零行 tensor
- 所有算子层（Linear、LayerNorm、SiluAndMul、GateLinear）添加了空 batch 短路
- 空 rank 仍参与 all-gather/reduce-scatter 通信（发送零行 padding）
