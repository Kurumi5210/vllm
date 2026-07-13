# Sharded Context Parallel Stage 2 总结

## 阶段目标

Stage 2 完成 Sharded-CP 的首尾 hidden layout 边界：

1. Embedding 输出从 TP vocab-parallel contribution 转成 CP-local token rows。
2. Transformer layer 的输入输出保持 CP-local token rows。
3. LMHead 前把 CP-local hidden all-gather 回全局 token rows，再复用现有 vocab/tensor-parallel logits 路径。

本阶段只处理 Embedding -> Transformer -> LMHead 的 token-row layout 边界，不实现 CP-local attention metadata、KV/indexer cache ownership、MoE CP 适配或 Shard Linear。真实 DeepSeek sparse MLA layer 路径在本阶段显式 fail-closed，避免在 metadata 仍是全局 token 视图时静默执行错误 attention/cache 写入。

## 代码改动

### `vllm/v1/worker/sharded_cp_utils.py`

`ShardedCPTokenRange` 新增 `rank` 和 `world_size` 字段。这样 collective helper 不需要从 `start == 0` 推断是否单 rank，而是明确以 `world_size == 1` 判断本地 fast path。这个修正很重要：多 rank 且 rank 0 的 `start == 0` 时不能跳过通信，否则 embedding contribution 不会跨 TP rank reduce。

新增 `reduce_scatter_token_rows(x, token_range, group=None, pad_value=0.0)`：

1. 输入 `x` 是当前 TP rank 对全局 token hidden 的局部贡献，例如 vocab-parallel embedding 在 all-reduce 前的输出。
2. 多 rank 时先按 `token_range.padded_num_tokens * world_size` 把 token 维补齐。
3. 沿 token 维切成每个 CP rank 一份的等长 chunks。
4. 执行 `dist.reduce_scatter`，把所有 TP rank 的 embedding contribution 相加，同时只保留当前 CP rank 的 token rows。
5. 返回前裁掉本 rank 的 padding rows，只暴露真实 `[T_local, hidden_size]`。

新增 `shard_global_token_rows(x, rank, world_size, pad_value=0.0)`，用于 CPU UT 里模拟某个 CP rank 的 token rows 切片，避免本阶段必须启动真实多进程 distributed 测试。

`all_gather_token_rows()` 也改成只在 `world_size == 1` 时走本地 fast path，并且 fast path 仍校验本地行数。多 rank 未初始化 distributed 会显式报错，避免测试或调用方误以为 rank 0 的完整行数等价于全局 gather。

### `vllm/model_executor/layers/vocab_parallel_embedding.py`

`VocabParallelEmbedding.forward_native()` 被拆成两步：

1. `forward_parallel(input_)`：只计算当前 TP rank 的 vocab shard embedding contribution，并把不属于本 shard 的 token 置零。
2. `forward_native(input_)`：调用 `forward_parallel()` 后再执行原有 `tensor_model_parallel_all_reduce()`。

普通模型路径仍然通过 `forward_native()` 得到全局 embedding，行为不变。Sharded-CP 路径需要的是 all-reduce 前的局部 contribution，因为后续 `reduce_scatter_token_rows()` 同时承担跨 rank reduction 和 token-row scatter。

### `vllm/model_executor/models/deepseek_v2.py`

`DeepseekV2Model` 保存两个 Sharded-CP 状态：

1. `self.enable_sharded_context_parallel`：来自 `ParallelConfig` 的总开关。
2. `self.sharded_cp_token_range`：本次 forward 中当前 CP rank 的 token-row 范围，供 logits 阶段 all-gather 使用。

新增 `_maybe_scatter_to_sharded_cp(hidden_states, positions, reduce_hidden_states=...)`：

1. flag 关闭时直接返回原始 hidden 和 positions。
2. flag 打开时根据 `get_sharded_cp_group()` 和全局 token 数计算 `ShardedCPTokenRange`。
3. 如果 hidden 来自 `input_ids` 的 vocab-parallel embedding contribution，则执行 `reduce_scatter_token_rows()`。
4. 如果 hidden 来自 `inputs_embeds`，说明每个 TP rank 已经持有完整 hidden，不应再跨 rank 求和，只做 token-row slicing。
5. `positions` 同步切到 `[token_range.start, token_range.end)`，保证后续 Transformer layer 看到的是 CP-local rows。

`DeepseekV2Model.forward()` 在第一 PP rank 上改为：

1. Sharded-CP + `input_ids`：调用 `self.embed_tokens.forward_parallel(input_ids)`，得到 TP-local embedding contribution。
2. 非 Sharded-CP：继续走 `self.embed_input_ids(input_ids)`。
3. `inputs_embeds`：保持 full hidden 输入语义，只做 CP-local slicing。
4. 进入 decoder layers 前统一调用 `_maybe_scatter_to_sharded_cp()`，因此本阶段的首尾边界会形成 `[T_local, hidden_size]` layout。
5. 如果真实 decoder layers 非空，则 `_raise_if_sharded_cp_transformer_path_not_ready()` 抛出 `RuntimeError`。后续阶段完成 CP-local metadata、attention、MLP/MoE correctness path 前，不允许真实 transformer layer 消费 CP-local hidden。

`DeepseekV2ForCausalLM.compute_logits()` 在 Sharded-CP 打开时：

1. 读取 forward 阶段保存的 `self.model.sharded_cp_token_range`。
2. 调用 `all_gather_token_rows()` 把 CP-local hidden 拼回全局 token 顺序。
3. 再调用原有 `LogitsProcessor(self.lm_head, hidden_states)`。
4. all-gather 后清空 `self.model.sharded_cp_token_range`，避免后续 logits 调用误用上一轮 forward 的 token range。

如果用户在没有先执行 model forward 的情况下直接调用 `compute_logits()`，会抛出 `RuntimeError`，避免在缺少 token range 的情况下静默输出错误 layout。

## UT 覆盖

新增 `tests/v1/worker/test_sharded_cp_boundaries.py`：

1. `test_fake_layer_round_trip_preserves_token_order_and_logits_shape`
   - 用 `nn.Identity` 模拟 Transformer layer。
   - 验证 global hidden -> CP-local rows -> all-gather 后 token 顺序不变。
   - 验证 LMHead 前 hidden 恢复全局 token 维后 logits shape 正确。

2. `test_sharded_cp_hidden_matches_unsharded_after_gather_for_uneven_tokens`
   - 覆盖 uneven `T=11, CP=4`。
   - 验证 padding + trim 不丢 token、不重排 token。

3. `test_embedding_contributions_reduce_then_gather_matches_full_hidden`
   - 模拟多个 TP rank 的 embedding contribution。
   - 先按 CP rank 做 token-row reduce，再 gather 回全局视图。
   - 验证结果等于传统 TP all-reduce 后的 full hidden。

4. `test_inputs_embeds_are_sliced_without_embedding_reduction`
   - 验证 `inputs_embeds` 入口只切 token rows，不做跨 rank reduction。
   - 防止 full hidden 被错误放大 `world_size` 倍。

5. `test_lm_head_all_gather_single_rank_fast_path_before_logits`
   - 覆盖单 rank 下 all-gather fast path。

6. `test_sharded_cp_real_transformer_layers_fail_closed_until_metadata_ready`
   - 覆盖真实 decoder layer 非空时显式 fail-closed。
   - 防止 Stage 2 在 CP-local attention metadata 尚未实现时静默进入 sparse MLA。

7. `test_sharded_cp_boundary_only_model_can_skip_metadata_guard`
   - 覆盖 fake-layer / boundary-only 场景可以绕过 metadata guard。

8. `test_compute_logits_clears_sharded_cp_token_range_after_gather`
   - 覆盖 logits all-gather 后清理 `sharded_cp_token_range`。

扩展 `tests/v1/worker/test_sharded_cp_utils.py`：

1. 验证 `ShardedCPTokenRange` 记录 `rank/world_size`。
2. 验证多 rank 且 rank 0 行数看似完整时，`all_gather_token_rows()` 仍要求 distributed 初始化。
3. 验证 `reduce_scatter_token_rows()` 的单 rank fast path。
4. 验证多 rank reduce-scatter 未初始化 distributed 时显式报错。
5. 验证多 rank rank 0 不会错误跳过 reduce-scatter。
6. 验证单 rank all-gather fast path 仍拒绝错误 local rows。
7. 验证单 rank reduce-scatter fast path 仍拒绝错误 global rows。

## 测试结果

通过的边界测试：

```bash
.venv/bin/python -m pytest tests/v1/worker/test_sharded_cp_utils.py tests/v1/worker/test_sharded_cp_boundaries.py -q
```

结果：

```text
31 passed, 16 warnings in 0.76s
```

通过的 Stage 1+2 focused 回归：

```bash
.venv/bin/python -m pytest tests/v1/worker/test_sharded_cp_utils.py tests/v1/worker/test_sharded_cp_boundaries.py tests/test_sharded_context_parallel_config.py tests/engine/test_arg_utils.py::test_enable_sharded_context_parallel_cli_arg tests/engine/test_arg_utils.py::test_enable_sharded_context_parallel_flows_to_parallel_config tests/engine/test_arg_utils.py::test_enable_sharded_context_parallel_rejects_incompatible_cli_topology -q
```

结果：

```text
51 passed, 16 warnings in 1.02s
```

通过的语法检查：

```bash
.venv/bin/python -m py_compile vllm/model_executor/models/deepseek_v2.py vllm/model_executor/layers/vocab_parallel_embedding.py vllm/v1/worker/sharded_cp_utils.py
```

该命令无输出，表示编译通过。

## 本阶段未实现内容

Stage 2 仍未实现：

1. CP-local `DeepseekV32IndexerMetadata` 和 `FlashMLASparseMetadata` 重建。
2. `slot_mapping`、`req_id_per_token`、prefill chunk offsets 的本地化。
3. 当前 token KV cache 和 indexer K cache 的 CP-local ownership。
4. sparse MLA attention 的 full logical weight correctness path。
5. MoE 的 CP-local routing、EP dispatch 前 gather、expert output reduce-scatter。
6. 多进程 distributed 集成测试。
7. decode 或 mixed prefill/decode 支持。

因为第 1 到第 3 项未实现，真实 DeepSeek transformer layers 在本阶段会 fail-closed；只有 embedding/reduce-scatter、fake layer layout round trip 和 logits all-gather 这些首尾边界可执行并由本地 UT 验证。

这些内容属于后续 Commit 3 及之后的阶段。

---

## 代码审查（2026-06-30）

### 设计文档要求对照

**变更：**

| 设计要求 | 状态 | 证据 |
|----------|------|------|
| embedding output pad/reduce-scatter | ✅ | `vocab_parallel_embedding.py:464-480` `forward_parallel()`；`deepseek_v2.py:1258` `input_ids` 路径调用 `reduce_scatter_token_rows()` |
| Transformer layer 输入输出 CP-local rows | ⚠️ | 首尾边界形成 CP-local hidden，但真实 decoder layer 在 transformer path 完整前 fail-closed |
| LMHead 前 all-gather → TP logits | ✅ | `deepseek_v2.py:1462-1473` `compute_logits()` 中 `all_gather_token_rows()` |
| `inputs_embeds` 入口不做跨 rank reduction | ✅ | `reduce_hidden_states=False` → `slice_for_token_reduce_scatter` only |
| 真实 transformer layer 不误跑 | ✅ | `_raise_if_sharded_cp_transformer_path_not_ready()` 在 layer 非空时抛错 |

**测试：**

| 设计要求 | 状态 | 证据 |
|----------|------|------|
| fake-layer round trip | ✅ | `test_sharded_cp_boundaries.py:29-55` |
| uneven T padding + trim | ✅ | `test_sharded_cp_boundaries.py:58-73` T=11, CP=4 |
| embedding contribution reduce→gather 一致性 | ✅ | `test_sharded_cp_boundaries.py:76-98` |
| `inputs_embeds` 切片不复用 reduce 路径 | ✅ | `test_sharded_cp_boundaries.py:101-114` |
| 单 rank all-gather fast path | ✅ | `test_sharded_cp_boundaries.py` |
| transformer path 完整前 fail-closed | ✅ | `test_sharded_cp_real_transformer_layers_fail_closed_until_mlp_moe_ready` |
| logits 后清理 token range | ✅ | `test_compute_logits_clears_sharded_cp_token_range_after_gather` |

### 逐文件审查

#### `vllm/v1/worker/sharded_cp_utils.py` ✅⚠️

1. **`ShardedCPTokenRange` 新增 `rank`/`world_size`（L18-19）**：明确语义，
   消除了之前靠 `start == 0` 推断单 rank 的 hack ✅。

2. **`all_gather_token_rows` 快路径修正（L129-130）**：改为 `world_size == 1`
   判断，多 rank 且 rank 0 时不再错误跳过通信 ✅。

3. **`reduce_scatter_token_rows`（L139-177）**：正确实现了 token-row 维度
   reduce-scatter。单 rank 时保持透传，多 rank 时补齐 token 维、split、
   `dist.reduce_scatter`、trim padding ✅。

4. **单 rank fast path 已保留行数校验**：`all_gather_token_rows` 在
   `world_size == 1` 时先校验 `x.shape[0] == token_range.num_tokens` 再返回，
   不再提前做无用 padding ✅。

5. **`reduce_scatter_token_rows` 去掉冗余切片**：现在直接校验全局 token 行数，
   基于 `token_range.padded_num_tokens` 构造通信 chunks 和输出 tensor，不再
   调用 `slice_for_token_reduce_scatter` 只为推断 shape ✅。

#### `vllm/model_executor/layers/vocab_parallel_embedding.py` ✅

1. **`forward_parallel` 拆分（L464-480）**：只做 vocab shard embedding + mask
   清零，不再调用 `tensor_model_parallel_all_reduce`。Sharded-CP 路径拿到
   TP-local contribution 后由 `reduce_scatter_token_rows` 统一 reduction ✅。

2. **`forward_native` 回退（L482-487）**：调用 `forward_parallel` +
   `tensor_model_parallel_all_reduce`，普通路径行为完全不变 ✅。

#### `vllm/model_executor/models/deepseek_v2.py` ✅⚠️

1. **`_maybe_scatter_to_sharded_cp`**：
   - flag 关闭 → 透传 ✅
   - `input_ids` + Sharded-CP → `reduce_scatter_token_rows` ✅
   - `inputs_embeds` + Sharded-CP → `slice_for_token_reduce_scatter` only ✅
   - `positions` 同步切分 ✅

2. **真实 decoder layer fail-closed**：
   - `start_layer != end_layer` 且 Sharded-CP 打开时抛出 `RuntimeError` ✅
   - 明确阻止 Commit 3 之前使用全局 attention metadata 消费 CP-local rows ✅

3. **`compute_logits` CP 适配**：
   - Sharded-CP 打开 → `all_gather_token_rows` 恢复全局 token 顺序
   - `token_range is None` → `RuntimeError`，fail-closed ✅
   - all-gather 后清理 `self.model.sharded_cp_token_range` ✅

#### `tests/v1/worker/test_sharded_cp_boundaries.py` ✅

1. **fake-layer round trip**：Identity 层 → gather → exact match ✅。
2. **uneven T 覆盖**：`T=11, CP=4` ✅。
3. **embedding contribution reduce→gather**：模拟多 rank 求和等价于单次
   all-reduce ✅。
4. **`inputs_embeds` 不被放大**：切片 only，不做跨 rank sum ✅。
5. **metadata guard**：真实 layer 非空时抛错，boundary-only 时允许继续 ✅。
6. **logits range cleanup**：`compute_logits` 消费 range 后置空 ✅。

### 审查结论

| 类别 | 评估 |
|------|------|
| 代码质量 | ✅ 风格一致、错误处理清晰 |
| 设计文档对齐 | ✅ Stage 2 首尾 hidden layout 已实现；真实 layer 路径按阶段边界 fail-closed |
| 测试覆盖 | ✅ 8 个 boundary test + 8 个 utils 扩展 test，31 passed |
| 后续风险 | ⚠️ Commit 3 必须实现 CP-local metadata/cache ownership 后才能打开真实 sparse MLA layer |

### 发现的问题

**已修复问题 1**：`all_gather_token_rows` 的单 rank fast path 移到 padding
之前，并保留行数校验。

**已修复问题 2**：`reduce_scatter_token_rows` 不再调用
`slice_for_token_reduce_scatter` 只为推断 shape。

**已修复问题 3**：`compute_logits` all-gather 后清理
`self.model.sharded_cp_token_range`。

**补充问题 4**：原审查把“全部设计要求已实现”说得过满。Stage 2 只实现
首尾 hidden layout 边界；由于 attention metadata、slot mapping、KV/indexer
cache ownership、attention 和 MLP/MoE 还未完整本地化，真实 transformer layer
在本阶段必须 fail-closed。当前代码已加 guard 和 UT，后续阶段再逐步解除该限制。
