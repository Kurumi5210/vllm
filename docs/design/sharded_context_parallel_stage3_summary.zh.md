# Sharded Context Parallel Stage 3 总结

## 阶段目标

Stage 3 完成 Sharded-CP 的 CP-local attention metadata 和 forward context 边界：

1. 从全局 `query_start_loc` 生成 request-aligned CP token ranges。
2. 在 request 边界对齐时构造本地 `CommonAttentionMetadata`。
3. 构造本地 `DeepseekV32IndexerMetadata` 和 `FlashMLASparseMetadata`。
4. 提供 scoped forward-context override，让 attention/indexer 后续能看到本地 metadata 和 slot mapping。
5. 对 decode、FP8 sparse MLA metadata、非 request-aligned range fail closed。

本阶段不打开真实 sparse MLA attention correctness path。真实 decoder layer 仍会 fail closed；后续阶段需要先补 compact KV all-gather、full-head sparse attention 基础，再补 MLP/MoE CP path 后才能解除完整 transformer layer guard。

## 代码改动

### `vllm/v1/worker/sharded_cp_utils.py`

`ShardedCPTokenRange` 增加：

1. `padded_num_rows`：支持 request-aligned variable range 时显式记录 collective chunk size。
2. `rank_starts/rank_ends`：记录每个 CP rank 的真实 token range，用于 gather trim 和 reduce-scatter chunk 构造。

新增 `get_request_aligned_sharded_cp_token_ranges(query_start_loc_cpu, world_size)`：

1. 使用全局 request query offsets。
2. 先计算 nominal chunk，再把边界向后对齐到 request boundary。
3. 示例 `[100, 200, 50]`、`CP=2`：
   - rank 0: `[0, 300)`，包含 request 0 和 request 1。
   - rank 1: `[300, 350)`，包含 request 2。
4. 所有 rank 使用最大本地 token 数作为 padded chunk size，保证 collective shape 一致。

新增：

1. `make_reduce_scatter_token_chunks()`：为 request-aligned ranges 构造等长 reduce-scatter chunks。
2. `assemble_token_all_gather_chunks()`：按 `rank_starts/rank_ends` 裁掉 padding 后恢复全局 token 顺序。

固定等长 range 的旧路径仍保留，用于无 forward context 的边界测试和 fallback。

### `vllm/v1/worker/sharded_cp_metadata.py`

新增 Sharded-CP metadata localizer。

`get_sharded_cp_token_range_from_forward_context()`：

1. 从 `ForwardContext.attn_metadata` 的任意一层 metadata 中读取 `query_start_loc`。
2. 基于 request boundaries 计算当前 rank 的 `ShardedCPTokenRange`。
3. 要求 forward context 使用 per-layer metadata dict。

`localize_common_attention_metadata()`：

1. 验证 `token_range.start/end` 都落在 request boundary。
2. 切分 `query_start_loc/query_start_loc_cpu`，并减去 `token_range.start` 转成本地 offsets。
3. 切分 `seq_lens`、`block_table_tensor`、`slot_mapping`。
4. 重新计算 `num_reqs`、`num_actual_tokens`、`max_query_len`、`max_seq_len`。

`localize_deepseek_v32_indexer_metadata()`：

1. 只支持 prefill，decode 显式拒绝。
2. 更新本地 `seq_lens/query_start_loc/slot_mapping`。
3. 根据本地 request 子集重建 `DeepseekV32IndexerPrefillChunkMetadata`。
4. 原始 prefill chunk 可以跨多个 CP ranks，只要切点是 request boundary，本地 chunk 会重新生成。

`localize_flashmla_sparse_metadata()`：

1. 只支持非 FP8 metadata，FP8 mixed/separate metadata 先 fail closed。
2. 切分 `query_start_loc/slot_mapping/block_table/req_id_per_token`。
3. 本地 `req_id_per_token` 从 0 重新编号，避免 rank 1 仍看到全局 request id。

`sharded_cp_forward_context()`：

1. 使用 `dataclasses.replace` 构造新的 `ForwardContext`。
2. 替换其中的 `attn_metadata` 为本地 per-layer metadata dict。
3. 替换 `slot_mapping` 为本地 per-layer slot mapping dict。
4. 通过现有 `override_forward_context()` 做 scoped override，退出 context 后恢复原始 context。

### `vllm/model_executor/models/deepseek_v2.py`

`_maybe_scatter_to_sharded_cp()` 现在优先从当前 `ForwardContext` 读取 request-aligned token range：

1. 正常 runner forward：根据 metadata 的 request boundaries 切 CP rows。
2. 无 forward context 的本地边界 UT：回退到 Commit 2 的固定等长 range。

真实 decoder layer guard 从“metadata 未实现”更新为“attention/MLP/MoE transformer path 未实现”。原因是 Stage 3 已经实现 metadata localizer，但还没有实现后续的 attention 计算和 MLP/MoE CP 路径。

layer loop 外新增 `sharded_cp_forward_context()` 接线准备。当前因为真实 layer guard 仍在，真实 sparse MLA layer 不会执行；后续解除 guard 后会消费这个 scoped local context。

## UT 覆盖

新增 `tests/v1/worker/test_sharded_cp_metadata.py`：

1. request-aligned ranges：
   - `[100, 200, 50]`、`CP=2` 得到 `[0,300)` 和 `[300,350)`。
   - padded chunk size 为 300。

2. request-aligned reduce-scatter/all-gather round trip：
   - rank 1 的 50 rows padding 到 300 rows。
   - gather 后恢复原始 350 rows 顺序。

3. `CommonAttentionMetadata` 本地化：
   - rank 0 持有两个 request，`query_start_loc_cpu=[0,100,300]`。
   - rank 1 持有一个 request，`query_start_loc_cpu=[0,50]`。
   - `slot_mapping` 保持 global slot IDs。

4. request split fail closed：
   - token range 边界落在 request 内部时抛出 `RuntimeError`。

5. `DeepseekV32IndexerMetadata` 本地化：
   - rank 1 的 prefill chunk 重建为 `[0,50)`。
   - block table 切到 request 2。
   - decode metadata 被拒绝。

6. `FlashMLASparseMetadata` 本地化：
   - `req_id_per_token` 切分后从 0 重新编号。
   - FP8 metadata 被拒绝。

7. per-layer dict 和 forward context：
   - 本地 metadata 保留原 layer keys。
   - scoped override 内看到 local metadata/local slot mapping。
   - scoped override 退出后恢复原始 `ForwardContext`。

扩展 `tests/v1/worker/test_sharded_cp_boundaries.py`：

1. `_maybe_scatter_to_sharded_cp()` 使用 forward context 中的 request-aligned range。
2. rank 1 对 `[100, 200, 50]` 得到 `[300,350)`，hidden/positions 同步切分。
3. boundary-only 路径在没有 forward context 时不会误触发 scoped override。
4. 真实 decoder layer 仍 fail closed，等待后续 attention 与 MLP/MoE 路径完整。

## 测试结果

通过的 Stage 1+2+3 focused 回归：

```bash
.venv/bin/python -m pytest tests/v1/worker/test_sharded_cp_utils.py tests/v1/worker/test_sharded_cp_boundaries.py tests/v1/worker/test_sharded_cp_metadata.py tests/test_sharded_context_parallel_config.py tests/engine/test_arg_utils.py::test_enable_sharded_context_parallel_cli_arg tests/engine/test_arg_utils.py::test_enable_sharded_context_parallel_flows_to_parallel_config tests/engine/test_arg_utils.py::test_enable_sharded_context_parallel_rejects_incompatible_cli_topology -q
```

结果：

```text
65 passed, 16 warnings in 0.95s
```

通过的语法检查：

```bash
.venv/bin/python -m py_compile vllm/v1/worker/sharded_cp_utils.py vllm/v1/worker/sharded_cp_metadata.py vllm/model_executor/models/deepseek_v2.py tests/v1/worker/test_sharded_cp_metadata.py tests/v1/worker/test_sharded_cp_boundaries.py
```

该命令无输出，表示编译通过。

通过的 whitespace 检查：

```bash
git diff --check
```

该命令无输出。

未运行 ruff：当前 `.venv` 中没有安装 `ruff` 模块。

## 本阶段未实现内容

Stage 3 仍未实现：

1. compact `[kv_c || indexer_k]` all-gather。
2. full-head `q_b_proj/q_up_proj` 和 `o_proj` correctness path。
3. sparse MLA hidden/logits parity。
4. FP8 sparse MLA metadata local rebuild。
5. decode 或 mixed prefill/decode。
6. MoE CP 适配。

这些属于 Commit 4 及之后阶段；其中完整 transformer layer guard 需要等 MLP/MoE CP path 一起完成后再解除。

---

## 代码审查（2026-06-30）

### 设计文档要求对照

**变更：**

| 设计要求 | 状态 | 证据 |
|----------|------|------|
| request-aligned CP token ranges | ✅ | `sharded_cp_utils.py:91-157` `get_request_aligned_sharded_cp_token_ranges()`，bisect 对齐到 request boundary |
| 构造本地 `CommonAttentionMetadata` | ✅ | `sharded_cp_metadata.py:92-157`，切分所有 token-indexed 字段，重建 `num_reqs`/`max_query_len` |
| 构造本地 `DeepseekV32IndexerMetadata` | ✅ | `sharded_cp_metadata.py:260-307`，直接从 Indexer metadata 切分 request/token 字段并重建 prefill chunk |
| 构造本地 `FlashMLASparseMetadata` | ✅ | `sharded_cp_metadata.py:310-348`，`req_id_per_token` 从 0 重新编号 |
| scoped forward-context override | ✅ | `sharded_cp_metadata.py:382-413` `sharded_cp_forward_context()`，`dataclasses.replace` + `override_forward_context` |
| decode fail closed | ✅ | `sharded_cp_metadata.py:265-266` |
| FP8 fail closed | ✅ | `sharded_cp_metadata.py:315-318` |
| `_maybe_scatter_to_sharded_cp` 优先用 forward context | ✅ | `deepseek_v2.py:1226-1238`，`is_forward_context_available()` 明确分支，避免 `AssertionError` flow control |
| layer loop 包在 cp_context 内 | ✅ | `deepseek_v2.py:1324-1334` SCP 开 → `sharded_cp_forward_context()`，关 → `nullcontext()` |

**测试：**

| 设计要求 | 状态 | 证据 |
|----------|------|------|
| request-aligned ranges `[100,200,50]` + CP=2 | ✅ | `test_sharded_cp_metadata.py:52-63` |
| reduce-scatter/all-gather round trip | ✅ | `test_sharded_cp_metadata.py:66-79` |
| CommonAttentionMetadata rank 0 | ✅ | `test_sharded_cp_metadata.py:82-97` |
| CommonAttentionMetadata rank 1 | ✅ | `test_sharded_cp_metadata.py:100-115` |
| request split fail closed | ✅ | `test_sharded_cp_metadata.py:118-127` |
| IndexerMetadata prefill chunk 重建 rank 1 | ✅ | `test_sharded_cp_metadata.py:172-197`，chunk `token_start=0, token_end=50` |
| Indexer decode reject | ✅ | `test_sharded_cp_metadata.py:199-219` |
| FlashMLASparse `req_id_per_token` 重编号 | ✅ | `test_sharded_cp_metadata.py:242-258` |
| FP8 reject | ✅ | `test_sharded_cp_metadata.py:261-272` |
| per-layer dict key 保留 | ✅ | `test_sharded_cp_metadata.py:275-298` |
| forward context override/restore | ✅ | `test_sharded_cp_metadata.py:301-319`，scoped 内外不同 context |
| boundary test 走 request-aligned range | ✅ | `test_sharded_cp_boundaries.py` 覆盖 `_maybe_scatter_to_sharded_cp()` forward context 分支 |

### 逐文件审查

#### `vllm/v1/worker/sharded_cp_utils.py` ✅

1. **`ShardedCPTokenRange` 新增字段（L25-30）**：`padded_num_rows`、`rank_starts`、
   `rank_ends` 支持 request-aligned variable range ✅。

2. **`get_request_aligned_sharded_cp_token_ranges`（L91-157）**：使用
   `bisect_left` 在 `query_start_loc` 中找到最近的 request boundary 做切分点。
   处理了边界退化为 0 行的情况，`rank_starts/rank_ends` 记录所有 rank 真实范围 ✅。

3. **`make_reduce_scatter_token_chunks`**：有 `has_explicit_rank_ranges` 时按每个
   rank 的真实范围切分+padding；否则走固定等长 split 兼容旧路径 ✅。

4. **`assemble_token_all_gather_chunks`（L211-235）**：对称的 gather 端，
   按 `rank_starts/rank_ends` 裁掉每个 chunk 的 padding 后拼接 ✅。

#### `vllm/v1/worker/sharded_cp_metadata.py` ✅

1. **`get_sharded_cp_token_range_from_forward_context`（L66-89）**：从
   `ForwardContext.attn_metadata` 提取 `query_start_loc`，计算 request-aligned
   range。要求 per-layer dict 格式 ✅。

2. **`localize_common_attention_metadata`（L92-157）**：
   - `_request_slice_for_token_range` 用 `torch.nonzero` 验证 range 边界落在
     request boundary ✅。
   - `query_start_loc` 减去 `token_range.start` 转为 local offsets ✅。
   - `slot_mapping` 值保持全局 slot ID，只是 token 维切到 local ✅。
   - `logits_indices_padded`、`causal` 等字段透传 ✅。

3. **`localize_deepseek_v32_indexer_metadata`（L260-307）**：
   - decode → `RuntimeError` ✅。
   - 不再通过半初始化的 `CommonAttentionMetadata` 做中转，而是直接从
     `DeepseekV32IndexerMetadata` 切 `query_start_loc/seq_lens/slot_mapping` 并重算
     `num_reqs/max_query_len/max_seq_len` ✅。
   - Prefill chunk 重建：原 chunk 跨 rank 时取交集，`block_table` 按
     `local_req_start - chunk_req_start` 对齐 ✅。

4. **`localize_flashmla_sparse_metadata`（L310-348）**：
   - FP8 → `RuntimeError` ✅。
   - `req_id_per_token` 减去第一个元素的值（从 rank 1 的 2 变成 0）✅。

5. **`sharded_cp_forward_context`（L382-413）**：
   - 用 `dataclasses.replace` 创建新 `ForwardContext`，不修改原始 ✅。
   - `override_forward_context` 做 scoped override，退出恢复原始 ✅。

#### `vllm/model_executor/models/deepseek_v2.py` ✅

1. **`_maybe_scatter_to_sharded_cp` fallback 链（L1216-1252）**：
   - 正常 runner forward：`is_forward_context_available()` 为真时，从
     `ForwardContext` 中读取 request boundaries，计算 request-aligned CP range ✅。
   - 边界 UT 或没有 forward context 的路径：显式走固定等长 range fallback ✅。
   - 已修复 Claude review 指出的 `except AssertionError` flow control 问题 ✅。

2. **guard → cp_context 时序（L1308→L1324）**：guard 在 `with` 外，context 在
   `with` 内。Commit 3 中 guard 抛异常不进 layer loop，后续解除 guard 后
   `cp_context` 生效 ✅。

#### `tests/v1/worker/test_sharded_cp_metadata.py` ✅

1. **request-aligned ranges**：`[0,100,300,350]` + `world_size=2` →
   `[(0,300,300), (300,350,300)]` ✅。

2. **round trip**：`make_reduce_scatter_token_chunks` → `assemble_token_all_gather_chunks`，
   rank 1 padding 到 300 rows → 恢复原始 350 rows ✅。

3. **metadata 本地化**：rank 0/1 各覆盖 `query_start_loc`/`seq_lens`/
   `slot_mapping`/`block_table` ✅。

4. **Indexer chunk 重建**：rank 1 chunk `token_start=0, token_end=50` ✅。

5. **FP8/decode/request split 拒绝**：每种各一个 test ✅。

6. **forward context override/restore**：scoped 内看到 local metadata，
   scoped 外恢复原始 context ✅。

### 审查结论

| 类别 | 评估 |
|------|------|
| 代码质量 | ✅ Claude review 中两个代码味道问题已修复 |
| 设计文档对齐 | ✅ 全部 9 项变更 + 12 项测试已实现 |
| 测试覆盖 | ✅ 13 个 metadata test + boundary/config focused 回归通过 |
| 安全性 | ✅ decode/FP8/request split 全部 fail-closed |

### Claude review 修复记录

**问题 1** ✅ `_maybe_scatter_to_sharded_cp` 不再用 `except AssertionError` 做
flow control。当前实现用 `is_forward_context_available()` 做显式分支：有
forward context 时走 request-aligned metadata range，没有 forward context 时才走
固定等长 range fallback。

**问题 2** ✅ `_common_from_indexer_metadata` 已删除。
`localize_deepseek_v32_indexer_metadata()` 直接处理 Indexer metadata 自身字段，
避免构造字段不完整的临时 `CommonAttentionMetadata`。

**问题 3** ✅ guard → cp_context 时序安全：`_raise_if_sharded_cp_transformer_path_not_ready()`
在 `with cp_context` 之外，Commit 3 抛异常不进 layer loop，后续解除 guard
调用即可，无需调整缩进。

**结论**：Claude review 的前两条评价是合理的，当前 commit3 已按该意见修复；
第三条是对当前时序的确认，不需要改代码。
