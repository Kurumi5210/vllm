# Sharded Context Parallel Stage 4 总结

## 阶段目标

Stage 4 开始落地 Sharded-CP attention correctness path 的本地可验证基础：

1. 明确 compact `[kv_c || k_pe || indexer_k]` all-gather payload layout。
2. 为 compact KV / Indexer-K 增加 pack、all-gather、split helper。
3. DeepSeek V3.2 sparse MLA 在 Sharded-CP 下使用 full logical attention heads。
4. Sharded-CP 下 `q_b_proj/q_proj`、`kv_b_proj`、`o_proj` 先走 replicated full logical weight 路径。
5. `o_proj` 不做 TP all-reduce，避免把不同 CP token rows 相加。

本阶段仍不解除真实 transformer layer guard。attention wrapper 内已接入 compact KV all-gather / full-head sparse attention 的本地可验证数据流，但 attention 后 dense MLP/MoE 的 CP-local 适配，以及真实 GPU sparse kernel parity 仍属于后续验收。当前 guard 文案保持为完整 CP-local attention and MLP/MoE transformer path 未完成。

## 代码改动

### `vllm/v1/worker/sharded_cp_attention.py`

新增 Sharded-CP attention helper。

`ShardedCPCompactKVLayout`：

1. 记录 `kv_lora_rank`、`qk_rope_head_dim`、`indexer_head_dim`。
2. `kv_dim = kv_lora_rank + qk_rope_head_dim`。
3. `total_dim = kv_dim + indexer_head_dim`。

`pack_sharded_cp_compact_kv()`：

1. 输入 `kv_c_normed: [T_local, kv_lora_rank]`。
2. 输入 `k_pe: [T_local, rope_dim]` 或 `[T_local, 1, rope_dim]`。
3. 输入 `indexer_k: [T_local, indexer_head_dim]`。
4. 输出 `[T_local, kv_lora_rank + rope_dim + indexer_head_dim]`。
5. 检查三者 row count、device、dtype 一致。

`split_sharded_cp_compact_kv()`：

1. 按 layout 把 compact payload 拆回 `kv_c_normed`、`k_pe`、`indexer_k`。
2. `k_pe` 统一恢复为 `[T, 1, rope_dim]`，匹配 MLA attention cache update 语义。

`assemble_sharded_cp_compact_kv_chunks()`：

1. 复用 Stage 3 的 `assemble_token_all_gather_chunks()`。
2. 对 request-aligned variable ranges 裁掉 padding。
3. 恢复全局 token 顺序后再 split。

`all_gather_sharded_cp_compact_kv()`：

1. 单 rank 走 `all_gather_token_rows()` fast path。
2. 多 rank 路径使用现有 token-row all-gather helper。
3. all-gather 后返回全局 `kv_c_normed/k_pe/indexer_k`。

`sharded_cp_topk_prefix()`：

1. 返回 top-k buffer 的 local token prefix。
2. 明确 Sharded-CP 下 top-k rows 不使用 global token offset，而是只写 `[0, T_local)`。

### `vllm/model_executor/models/deepseek_v2.py`

`DeepseekV2MLAAttention` 新增：

1. `self.enable_sharded_context_parallel`。
2. `self.use_sharded_cp_full_attention = enable_sharded_context_parallel and is_v32`。

当 `use_sharded_cp_full_attention` 为真：

1. `q_b_proj` 使用 `disable_tp=True`。
2. 无 q LoRA 时 `q_proj` 使用 `disable_tp=True`。
3. `kv_b_proj` 使用 `disable_tp=True`。
4. `o_proj` 显式使用 `input_is_parallel=True`，并设置 `disable_tp=True` 且 `reduce_results=False`。
5. `MultiHeadLatentAttentionWrapper` 的 `num_heads` 从 TP-local heads 改成 full `num_heads`。

flag off 或非 V3.2 MLA 时保持原行为：

1. `q_b_proj/q_proj/kv_b_proj/o_proj` 仍按 TP shard 构造。
2. wrapper 仍使用 `num_local_heads`。
3. `o_proj` 仍按 parallel input 语义在 TP 下 reduce。

`DeepseekV2Model` guard 改名为 `_raise_if_sharded_cp_transformer_path_not_ready()`：

1. Stage 4 attention 基础已经补齐，因此旧的 “full-head sparse attention 未实现” 文案不准确。
2. 真实 decoder layer 仍 fail closed，原因是完整 CP-local attention 调用链和 MLP/MoE CP path 尚未完成。

`Indexer` 与 `MultiHeadLatentAttentionWrapper` 新增 Sharded-CP attention wiring：

1. `Indexer.project()` 暴露 `q_fp8/indexer_k/weights`，保留原 `forward()` 的 paged-cache Indexer 路径。
2. `Indexer.sharded_cp_topk()` 提供本地可验证的 request-local top-k reference 路径，输出 global compact KV row offsets，并只写本地 `[0, T_local)` top-k prefix。
3. `MLAModules.enable_sharded_context_parallel` 将 DeepSeek V3.2 Sharded-CP 标志传入 wrapper。
4. wrapper 从 `ForwardContext.additional_kwargs["sharded_cp_token_range"]` 读取当前 CP token range。
5. wrapper 调用 `all_gather_sharded_cp_compact_kv()` 聚合 `[kv_c_normed || k_pe || indexer_k]`，再把 global `kv_c_normed/k_pe` 传给 `MLAAttention(..., use_global_kv=True)`。
6. `MLAAttention` 在 `use_global_kv=True` 时跳过本地 KV cache update，直接把 global compact KV 传给 sparse backend；opaque custom-op 路径 fail closed。
7. sparse MLA metadata 增加 `topk_indices_are_global_compact_offsets` 标记，Sharded-CP localized FlashMLA metadata 设置为 `True`，backend 在该模式下把 top-k 解释为 all-gather 后的 global compact KV row offset，并跳过 paged-cache request-local index 转换。

## UT 覆盖

新增 `tests/v1/worker/test_sharded_cp_attention.py`：

1. compact KV pack/split round trip：
   - `kv_c`、`k_pe`、`indexer_k` 拼接后能无损拆回。
   - `k_pe` 恢复为 `[T, 1, rope_dim]`。

2. request-aligned compact KV assemble：
   - `[100, 200, 50]`、`CP=2`。
   - rank 0 chunk 为 300 rows，rank 1 的 50 rows padding 到 300。
   - assemble 后恢复原始 350 rows 的 `kv_c/k_pe/indexer_k`。

3. 单 rank compact KV all-gather fast path：
   - 不依赖 `torch.distributed`。
   - 输入即输出，形状语义保持一致。

4. compact KV 输入校验：
   - row count 不一致时报 `ValueError`。

5. top-k prefix：
   - rank 1 虽然持有全局 `[300,350)` token range，但 top-k buffer 只使用本地前 50 rows。

6. DeepSeek V3.2 Sharded-CP full-head 构造：
   - `q_b_proj/kv_b_proj/o_proj.disable_tp == True`。
   - `o_proj.input_is_parallel == True`。
   - `o_proj.reduce_results == False`。
   - wrapper `num_heads == full num_heads`。

7. flag off 行为不变：
   - projection 不 `disable_tp`。
   - wrapper `num_heads == num_local_heads`。
   - `o_proj.input_is_parallel == True`。
   - `o_proj.reduce_results == True`。

8. 非 V3.2 MLA 行为不变：
   - 即使 Sharded-CP flag 打开，也不启用 sparse MLA full-head projection 策略。
   - `o_proj.input_is_parallel == True` 保持现有 TP 路径语义。

9. Sharded-CP request-local top-k reference：
   - 多请求场景中，后一个请求的 top-k 不会跨请求选中前一个请求的高分 token。
   - top-k 写入本地 prefix，索引值为 global compact KV row offset。

10. wrapper global compact KV wiring：
    - 单 rank fast path 下，wrapper 调用 compact KV all-gather 后传给 attention。
    - attention 收到 global `kv_c_normed/k_pe`，`use_global_kv=True`。
    - Indexer 收到 global `indexer_k` 和本地 request-local `query_start_loc`。

11. `MLAAttention.use_global_kv`：
    - direct-call 路径跳过本地 KV cache update，并把 `[kv_c_normed || k_pe]` 传给 backend。
    - opaque custom-op 路径 fail closed。

扩展 `tests/v1/worker/test_sharded_cp_boundaries.py`：

1. guard 名称更新为 `_raise_if_sharded_cp_transformer_path_not_ready()`。
2. 真实 transformer layer 仍 fail closed，错误原因收敛为完整 CP-local attention and MLP/MoE transformer path 未完成。

扩展 `tests/v1/worker/test_sharded_cp_metadata.py`：

1. Sharded-CP forward context 会在 `additional_kwargs` 中携带 `sharded_cp_token_range`。
2. localized FlashMLA sparse metadata 标记 `topk_indices_are_global_compact_offsets=True`。

## 测试结果

通过的 Stage 1+2+3+4 focused 回归：

```bash
.venv/bin/python -m pytest tests/v1/worker/test_sharded_cp_utils.py tests/v1/worker/test_sharded_cp_boundaries.py tests/v1/worker/test_sharded_cp_metadata.py tests/v1/worker/test_sharded_cp_attention.py tests/test_sharded_context_parallel_config.py tests/engine/test_arg_utils.py::test_enable_sharded_context_parallel_cli_arg tests/engine/test_arg_utils.py::test_enable_sharded_context_parallel_flows_to_parallel_config tests/engine/test_arg_utils.py::test_enable_sharded_context_parallel_rejects_incompatible_cli_topology -q
```

结果：

```text
77 passed, 16 warnings in 0.85s
```

## 本阶段未实现内容

Stage 4 仍未实现：

1. sparse MLA hidden/logits parity，需要真实 GPU sparse kernel 验证。
2. FP8 sparse MLA metadata rebuild。
3. decode 或 mixed prefill/decode。
4. MLP/MoE CP path。
5. 完整 transformer layer guard 解除。
6. 非 FlashMLA sparse backend 的 Sharded-CP metadata localization 仍需真实 backend 验证。

这些需要后续 commit 和 GPU 环境继续推进。当前 commit 的价值是把 full-head projection 构造、compact KV payload、wrapper global-KV 数据流和 request-local top-k 语义固定下来，并用本地 UT 防止后续接入真实 kernel 时发生 token/head/padding/索引语义偏移。

---

## 代码审查（2026-07-01）

### 设计文档要求对照

**变更：**

| 设计要求 | 状态 | 证据 |
|----------|------|------|
| compact KV payload layout `kv_c \|\| k_pe \|\| indexer_k` | ✅ | `sharded_cp_attention.py:19-31` `ShardedCPCompactKVLayout`，`total_dim = kv_lora_rank + qk_rope_head_dim + indexer_head_dim` |
| pack helper（拼接三路 tensor） | ✅ | `sharded_cp_attention.py:67-90` `pack_sharded_cp_compact_kv()`，row count/device/dtype 统一校验 |
| split helper（无损拆回 + k_pe reshape） | ✅ | `sharded_cp_attention.py:93-112`，`k_pe` 恢复为 `[T, 1, rope_dim]` |
| assemble chunks（复用 Stage 3 assemble + layout split） | ✅ | `sharded_cp_attention.py:115-122` |
| all-gather compact KV（单 rank fast path + token-row AG） | ✅ | `sharded_cp_attention.py:125-147` `all_gather_sharded_cp_compact_kv()` |
| top-k prefix | ✅ | `sharded_cp_attention.py:150-163` `sharded_cp_topk_prefix()` |
| wrapper 调用 compact KV all-gather | ✅ | `MultiHeadLatentAttentionWrapper.forward()` Sharded-CP 分支 |
| sparse backend 消费 global compact KV | ✅ | `MLAAttention.forward(..., use_global_kv=True)` |
| Sharded-CP top-k 使用 global compact row offsets | ✅ | `Indexer.sharded_cp_topk()` + `topk_indices_are_global_compact_offsets=True` |
| `q_b_proj` `disable_tp=True` | ✅ | `deepseek_v2.py:913` |
| `kv_b_proj` `disable_tp=True` | ✅ | `deepseek_v2.py:931` |
| `o_proj` `input_is_parallel=True` | ✅ | `deepseek_v2.py` `RowParallelLinear(...)` |
| `o_proj` `disable_tp=True` + `reduce_results=False` | ✅ | `deepseek_v2.py` `RowParallelLinear(...)` |
| wrapper `num_heads` → full `num_heads` | ✅ | `deepseek_v2.py:1010-1012`，SCP+V3.2 时用 `self.num_heads`，否则 `self.num_local_heads` |
| guard 名称/文案更新 | ✅ | `deepseek_v2.py:1266-1276`，`_raise_if_sharded_cp_transformer_path_not_ready()` |
| flag off 行为不变 | ✅ | `test 7` |
| 非 V3.2 不启用 | ✅ | `test 8`，即使 flag 打开也不走 full-head 路径 |

**测试：**

| 测试 | 证据 |
|------|------|
| pack/split round trip（逐元素无损） | `test_sharded_cp_attention.py:29-49` |
| request-aligned assemble + split round trip | `test_sharded_cp_attention.py:52-80`，rank 1 padding 50→300，assemble 后恢复 350 rows |
| 单 rank fast path 不依赖 distributed | `test_sharded_cp_attention.py:83-102` |
| row count 不一致 → `ValueError` | `test_sharded_cp_attention.py:105-111` |
| top-k prefix 只取本地行 | `test_sharded_cp_attention.py:114-124`，rank 1 全局 [300,350)，top-k 只取前 50 行 |
| V3.2+ShardedCP 全头构造 | `test_sharded_cp_attention.py`，`disable_tp=True`、`input_is_parallel=True`、`reduce_results=False`、`num_heads=8` |
| flag off 保持 TP 头 | `test_sharded_cp_attention.py`，`disable_tp=False`、`input_is_parallel=True`、`reduce_results=True`、`num_local_heads=4` |
| 非 V3.2 不变 | `test_sharded_cp_attention.py:190-198` |
| request-local top-k global offset | `test_sharded_cp_attention.py`，多请求下不跨请求选 token |
| wrapper global compact KV wiring | `test_sharded_cp_attention.py`，`use_global_kv=True` 且 attention 输入为 global `kv_c/k_pe` |
| `MLAAttention.use_global_kv` | `test_sharded_cp_attention.py`，跳过 cache update，opaque 路径 fail closed |
| localized metadata 标记 global compact top-k | `test_sharded_cp_metadata.py`，`topk_indices_are_global_compact_offsets=True` |
| guard 名称/匹配更新 | `test_sharded_cp_boundaries.py:142-152` |

### 逐文件审查

#### `vllm/v1/worker/sharded_cp_attention.py` ✅

1. **`ShardedCPCompactKVLayout`（L19-31）**：frozen dataclass，3 个维度字段 + 2 个
   计算属性。`total_dim` 恰好等于论文中的 704 (= 512 + 64 + 128) ✅。

2. **`pack_sharded_cp_compact_kv`（L67-90）**：先 flatten `k_pe`（支持 `[T, 1, rope_dim]`
   和 `[T, rope_dim]` 两种 shape），再校验 row count + device/dtype 一致性，
   最后 `torch.cat` 沿最后一维拼接 ✅。

3. **`split_sharded_cp_compact_kv`（L93-112）**：用 `layout.total_dim` 校验
   payload 最后一维，再用 `torch.split` 按 `[kv_lora_rank, rope_dim, indexer_k_dim]`
   拆分。`k_pe.unsqueeze(1)` 恢复为 `[T, 1, rope_dim]` ✅。

4. **`assemble_sharded_cp_compact_kv_chunks`（L115-122）**：先走 Stage 3 的
   `assemble_token_all_gather_chunks`（request-aligned 裁 padding），再 split ✅。

5. **`all_gather_sharded_cp_compact_kv`（L125-147）**：在函数内部从输入 tensor
   shape 反向推导 `ShardedCPCompactKVLayout`，然后 `pack → all_gather_token_rows → split`。
   单 rank 走 `all_gather_token_rows` 的 `world_size==1` fast path ✅。

6. **`sharded_cp_topk_prefix`（L150-163）**：校验 buffer 有足够行数后返回
   `[:num_tokens]` ✅。

#### `vllm/model_executor/models/deepseek_v2.py` ✅⚠️

1. **`is_v32` 上移（L875）**：从原来在 `max_position_embeddings` 块后面移到
   constructor 靠前位置，确保 `q_b_proj`/`kv_b_proj`/`o_proj` 构造时
   `self.is_v32` 已可用 ✅。

2. **`use_sharded_cp_full_attention`（L879-881）**：逻辑清晰，两个条件
   同时满足才启用 ✅。

3. **Projection 构造**：
   - `q_b_proj.disable_tp = use_sharded_cp_full_attention` ✅
   - `q_proj.disable_tp = use_sharded_cp_full_attention`（无 q LoRA 时）✅
   - `kv_b_proj.disable_tp = use_sharded_cp_full_attention` ✅⚠️ 见问题 2
   - `o_proj.input_is_parallel = True` ✅
   - `o_proj.reduce_results = not use_sharded_cp_full_attention` ✅
   - `o_proj.disable_tp = use_sharded_cp_full_attention` ✅

4. **Wrapper head count（L1010-1012）**：SCP+V3.2 → `self.num_heads`，
   否则 `self.num_local_heads` ✅。

5. **`o_proj` 显式传 `input_is_parallel=True`** ✅：`RowParallelLinear`
   现有默认值也是 `True`，这里显式传入是为了把 TP 路径的 parallel input
   语义固定下来。注意没有采用 `input_is_parallel=self.use_sharded_cp_full_attention`，
   因为 flag off 或非 V3.2 时把该参数改成 `False` 会改变现有 TP 行为。

6. **`kv_b_proj` 无条件 `disable_tp=True`** ⚠️：如果 backend 走 weight
   absorption 路径（`W_UK` 吸收到 `q_up_proj`，`W_UV` 吸收到 `o_proj`），
   `kv_b_proj` 在 forward 中不会被调用。此时 `disable_tp=True` 只增加了一
   份 full logical weight 的加载开销而无运行时收益。当前 replicated 阶段
   无害，Commit 6 Shard Linear 时需要确认 backend 实际行为再决策。

7. **guard → cp_context 时序（L1317→L1324）**：guard 在 `with` 之外，
   Commit 6+ 解除 guard 后 `cp_context` 生效 ✅。

8. **guard 错误信息已补准** ✅：guard 现在明确说明真实 layer 需要完整
   CP-local attention and MLP/MoE transformer path，覆盖 wrapper 内 compact KV
   all-gather / full-head sparse attention 调用链，以及后续 MLP/MoE CP path。

9. **`Indexer.project()` 与 `sharded_cp_topk()`** ✅：`project()` 复用原
   `Indexer.forward()` 的 q/k/weight 前置投影；`sharded_cp_topk()` 使用 global
   `indexer_k` 和本地 `query_start_loc` 计算 request-local top-k，并把结果写入
   本地 top-k prefix。输出是 global compact KV row offset，供 Sharded-CP global
   KV 模式直接消费。

#### `vllm/model_executor/layers/mla.py` ✅

1. **`MLAModules.enable_sharded_context_parallel`**：只由 DeepSeek V3.2
   Sharded-CP full-head path 置为 true，其他模型默认 false ✅。

2. **wrapper Sharded-CP 分支**：从 forward context 读取 `sharded_cp_token_range`，
   调用 `Indexer.project()` 得到本地 `indexer_k`，执行 compact KV all-gather，
   再把 global `kv_c_normed/k_pe` 传入 attention ✅。

3. **flag off 不变**：没有 `sharded_cp_token_range` 时仍走原 `Indexer.forward()`
   和本地 KV cache attention 路径 ✅。

#### sparse MLA backend wiring ✅⚠️

1. **`MLAAttention.use_global_kv`**：direct-call 路径跳过本地 KV cache update，
   将 global compact `[kv_c || k_pe]` 传给 backend；opaque custom-op 路径
   fail closed ✅。

2. **`topk_indices_are_global_compact_offsets`**：FlashMLA/FlashInfer/ROCm/XPU
   sparse metadata 增加标记；为 true 时 backend 把 top-k 解释为 all-gather 后的
   global compact KV row offset，跳过 request-local 到 paged-cache physical index
   的转换 ✅。

3. **真实 kernel parity**：本地 UT 只验证数据流和索引语义，尚未验证 GPU sparse
   kernel 数值 parity ⚠️。

#### `tests/v1/worker/test_sharded_cp_attention.py` ✅

1. **pack/split round trip**（L29-49）：`kv_c=[3,4]`、`k_pe=[3,1,2]`、
   `indexer_k=[3,3]` → pack 得到 `[3,9]` → split 后三路逐元素一致 ✅。

2. **request-aligned assemble**（L52-80）：`T=350, CP=2`，rank 1 的 50
   rows padding 到 300，assemble+split 后三路都与原始输入一致 ✅。

3. **单 rank fast path**（L83-102）：不依赖 `torch.distributed`，输入即输出，
   同时验证了 `k_pe` 从 2D 被 unsqueeze 到 3D ✅。

4. **row count 校验**（L105-111）：`k_pe` 行数不匹配 → `ValueError` ✅。

5. **top-k prefix**（L114-124）：rank 1 拿到 `[300,350)` token range，
   `sharded_cp_topk_prefix` 返回 `topk[:50]` ✅。

6. **full-head 构造 3 个 test**（L165-198）：用 monkeypatch 替换
   `ColumnParallelLinear`、`RowParallelLinear`、`MultiHeadLatentAttentionWrapper`
   为 `_Fake*` 类，验证三场景的 projection 参数正确。`_FakeLinear`
   记录 `o_proj.input_is_parallel=True`，`_FakeWrapper` 记录传入的 `num_heads`，
   直接验证 SCP+V3.2 时为 `8`（full），否则为 `4`（local） ✅。

### 审查结论

| 类别 | 评估 |
|------|------|
| 代码质量 | ✅ 风格一致、校验完整、frozen dataclass |
| 设计文档对齐 | ✅ projection/head/layout/guard 语义均已对齐 |
| 测试覆盖 | ✅ focused 回归 77 passed |
| 安全性 | ✅ guard 未解除，真实 layer loop 不可达 |

### 发现的问题

**问题 1** ✅ 已修补：`o_proj` 显式传 `input_is_parallel=True`。这里保留
`True` 而不是 review 建议的 `self.use_sharded_cp_full_attention`，因为
`RowParallelLinear` 默认 parallel input；flag off 时传 `False` 会破坏现有
TP 行为。

**问题 2** ⚠️ 保留并延后：`kv_b_proj.disable_tp=True` 在当前 replicated
correctness 阶段是有意为之。vLLM MLA 的 `process_weights_after_loading()`
会读取 `kv_b_proj` 权重来 materialize full-head `W_UK/W_UV`，因此 full-head
路径需要完整 logical `kv_b_proj`。等后续 Shard Linear / 真实 backend 阶段
再根据实际 kernel 路径优化显存。

**问题 3** ✅ 已修补：guard 文案改为完整 CP-local attention and MLP/MoE
transformer path 未完成，覆盖真实 attention 调用链和 MLP/MoE CP path。

**问题 4** ✅ compact KV payload 维度与论文一致：论文 `kv_lora_rank(512) +
qk_rope_head_dim(64) + index_head_dim(128) = 704`。代码拆分 `kv_c_normed`
（仅 kv_lora_rank）和 `k_pe`（rope 分量），拼装后总维度一致。

处理结果：问题 1 和问题 3 已修补并补 UT；问题 2 是当前 replicated
correctness 策略的显存 tradeoff，保留到后续 Shard Linear / GPU backend
阶段再优化；问题 4 确认为实现正确。

---

## 审查（2026-07-01）：complete attention wiring commit

前一 commit 补齐了 full-head MLA projection 构造和 compact KV helper，但 wrapper
和 backend 尚未接入。本 commit 完成了完整的 Sharded-CP attention 数据流 wiring。

### 变更覆盖（11 文件，+541/-80）

| 变更类别 | 文件 | 关键改动 |
|----------|------|----------|
| Indexer 拆分 | `deepseek_v2.py` | `forward()` 拆为 `project()`（返回 `q_fp8, k, weights`）+ `forward()`（调用前者 + `indexer_op`）；新增 `sharded_cp_topk()` 用全局 indexer K 和 request-local `query_start_loc` 直接产生 global-token-position 索引 |
| Wrapper CP 路径 | `mla.py` | `forward()` 通过 `additional_kwargs["sharded_cp_token_range"]` 检测 SCP；CP 路径：`indexer.project()` → `all_gather_sharded_cp_compact_kv()` → `indexer.sharded_cp_topk()` → `MLAAttention.forward(use_global_kv=True)` |
| MLAAttention | `mla_attention.py` | 新增 `use_global_kv` 参数：跳过 KV cache update，构造 `attn_kv = cat(kv_c, k_pe)` 直传 backend；opaque custom-op 路径拒绝 |
| topk_indices_are_global_compact_offsets | `flashmla_sparse.py`, `flashinfer_mla_sparse.py`, `rocm_aiter_mla_sparse.py`, `xpu_mla_sparse.py` | 4 个 sparse backend metadata 新增 flag；为 true 时把 top-k 解释为 all-gather 后的 global compact KV row offset，并跳过 `triton_convert_req_index_to_global_index` |
| Metadata localizer | `sharded_cp_metadata.py` | `localize_flashmla_sparse_metadata()` 设 `topk_indices_are_global_compact_offsets=True`；`sharded_cp_forward_context()` 将 `token_range` 存入 `additional_kwargs` |
| MLAModules | `mla.py` | 新增 `enable_sharded_context_parallel` 字段，从 `DeepseekV2MLAAttention` 构造传入 |
| 测试 | `test_sharded_cp_attention.py`, `test_sharded_cp_metadata.py` | +6 个新 test：top-k 全局索引语义、wrapper 全流程集成、MLAAttention global KV、metadata flag 透传 |

### 逐模块审查

#### 1. `Indexer.project()` / `sharded_cp_topk()` — `deepseek_v2.py` ⚠️

**拆分动机正确**：`project()` 返回 `(q_fp8, k, weights)` 三元组后，CP 路径可
对 `k` 做 all-gather，传统路径调用 `forward()` 走 `indexer_op` 原地计算 top-k ✅。

**`sharded_cp_topk` 全局索引语义**：逐 request 逐 local row 迭代，每个 row 在
`[global_start, global_start + local_row)` 范围内做 `einsum` 得分 + top-k 选择，
结果加 `global_start` 得到全局 token 位置。例如 token_range 为 `[300, 350)`，
request 的 `req_start=0`，则 top-k 索引落在 `[300, ...]` 区间。

⚠️ **性能**：Python 双层 for-loop + per-row `torch.einsum`。当前 correctness 阶段
可接受，但进入 GPU 验证阶段需替换为 compiled kernel 或向量化实现。

⚠️ **`topk_indices_are_global_compact_offsets` 与 block-table mapping**：`sharded_cp_topk`
产生的是**全局 compact KV row offset**（如 `token_range.start + local_offset`），
不是 KV cache 物理 slot。`topk_indices_are_global_compact_offsets=True` 在 backend
中完全跳过了 `triton_convert_req_index_to_global_index`（该函数通常做 per-request
index → block-table-mapped physical address 的转换）。这意味着此优化仅在 block
table 为 identity 映射（`arange_block_indices`）时正确——而这恰好是 prefill 的
常见测试配置。在真实 PagedAttention 部署中，block table 非 identity 时仍需一次
block-table 映射步骤。

**建议**：要么在 `sharded_cp_topk` 产出索引后加一次 block-table 查找，要么在
backend 侧重构为 "跳过 per-request 转换，但保留 block-table 映射"。当前阶段
（guard 未解除、测试用 arange block indices）可接受。

**复核结论**：该 review 抓到了命名风险，但 “block table 非 identity 时不正确”
这一结论不适用于当前 `use_global_kv=True` 路径。该路径传给 backend 的
`kv_c_and_k_pe_cache` 不是 persistent paged KV cache，而是 all-gather 后按 token
顺序连续排列的 global compact KV tensor，因此 `sharded_cp_topk()` 产出的
global compact row offset 正好可以直接索引这块 tensor。为避免误解，代码 flag
已改名为 `topk_indices_are_global_compact_offsets`。真实 GPU parity 仍需验证各
backend kernel 是否都按连续 compact KV row offset 解释该索引。

#### 2. `MultiHeadLatentAttentionWrapper.forward()` CP 路径 — `mla.py` ✅

**token_range 获取**（L120-128）：从 `forward_context.additional_kwargs` 中读取，
而非绕过 context 系统或存储在 wrapper 实例上。这保证了 scoped override 内外的
context 一致性 ✅。

**CP 路径分支**（L178-199）：
```
indexer.project() → all_gather_sharded_cp_compact_kv() → indexer.sharded_cp_topk()
→ MLAAttention.forward(use_global_kv=True) → o_proj
```
完整的 compact KV AG + global indexer top-k + global KV attention 链路 ✅。

**`MLAAttention.forward` 的 `attn_metadata` 来源于 forward context**：通过
`get_forward_context().attn_metadata[f"{self.prefix}.attn"]` 获取当前层的
local metadata。这是在 `sharded_cp_forward_context` scoped override 内的，
所以拿到的是 CP-local metadata ✅。

**flag off 兼容性**：`_sharded_cp_token_range()` 返回 `None` 时走原有
`Indexer.forward()` + local KV cache attention，行为完全不变 ✅。

#### 3. `MLAAttention.forward(use_global_kv=...)` — `mla_attention.py` ✅

**direct-call 路径**（L490-517）：这是 `use_direct_call=True` 的分支——即
sparse MLA 的正常路径：
- `use_global_kv=True` → 跳过 `do_kv_cache_update` ✅（当前 token KV 已在
  wrapper all-gather 后写入，由 wrapper 管理写入时机）
- `use_global_kv=True` → `attn_kv = cat(kv_c_normed, k_pe.squeeze(1))` 直传
  backend，而非从 persistent cache 读取 ✅

**opaque custom-op 路径**（L518-522）：`use_global_kv=True` → `RuntimeError`。
因为这条路径的 `kv_cache_dummy_dep` tensor 无法表达 "直接使用 all-gather 后
的 KV" 语义 ✅。

**KV cache update 跳过**：纯 prefill（无历史 KV）时不需要写 cache。满足当前
阶段要求 ✅。后续支持 incremental prefill 或 mixed batch 时需补回。

#### 4. `topk_indices_are_global_compact_offsets` — 4 个 backend + metadata localizer ✅⚠️

**4 个 backend**：FLASHMLA_SPARSE、FLASHINFER_MLA_SPARSE、ROCM_AITER_MLA_SPARSE、
XPU_MLA_SPARSE 的 metadata 各加 `topk_indices_are_global_compact_offsets: bool = False`，
默认行为不变。为 true 时把 top-k 解释为 all-gather 后的 global compact KV row
offset，并跳过 `triton_convert_req_index_to_global_index` ✅。

⚠️ 这里必须配合 `MLAAttention(use_global_kv=True)` 使用；如果 backend 输入仍是
paged KV cache，则不能跳过 block-table mapping。

**Metadata localizer**：`localize_flashmla_sparse_metadata()` 在构造 local
metadata 时设 `topk_indices_are_global_compact_offsets=True`。语义正确：
Sharded-CP top-k 索引已在 global compact KV row offset 空间 ✅。

#### 5. `additional_kwargs["sharded_cp_token_range"]` — `sharded_cp_metadata.py` ✅

`sharded_cp_forward_context()` 将 `token_range` 注入 `additional_kwargs`：

```python
additional_kwargs={
    **forward_context.additional_kwargs,
    "sharded_cp_token_range": token_range,
},
```

**设计优点**：token_range 通过 forward context 传递，wrapper 不持有任何
ShardedCP 状态。scoped override 退出后自动恢复，不污染原始 context ✅。

**测试**：`test_sharded_cp_forward_context_overrides_and_restores` 验证了
scoped 内能读到 `additional_kwargs["sharded_cp_token_range"]`，scoped 外恢复
原始 context ✅。

#### 6. `MLAModules.enable_sharded_context_parallel` — `mla.py` + `deepseek_v2.py` ✅

新增字段 `MLAModules.enable_sharded_context_parallel: bool = False`，由
`DeepseekV2MLAAttention` 在构造 `MLAModules` 时传入
`self.use_sharded_cp_full_attention`。链路清晰 ✅。

#### 7. 测试

| 测试 | 验证点 | 状态 |
|------|--------|:--:|
| `test_sharded_cp_topk_keeps_request_local_global_indices` | top-k 产生 global token 位置 + `-1` padding | ✅ |
| `test_wrapper_sharded_cp_uses_global_compact_kv` | wrapper 全流程：project→AG→topk→attention→oproj，`use_global_kv=True` 传递、indexer 看到 global `k`、`query_start_loc` 是 local | ✅ |
| `test_mla_attention_global_kv_skips_cache_update` | `use_global_kv=True` 不写 KV cache，attn_kv 从 `cat(kv_c, k_pe)` 构造 | ✅ |
| `test_mla_attention_global_kv_rejects_opaque_path` | indirect-call 路径拒绝 `use_global_kv` | ✅ |
| `localize` 验证 `topk_indices_are_global_compact_offsets` | local metadata 设 flag | ✅ |
| `test_sharded_cp_forward_context` 验证 `additional_kwargs` | scoped 内 token_range 可读，scoped 外恢复 | ✅ |

**测试质量**：`test_wrapper_sharded_cp_uses_global_compact_kv` 覆盖了完整的
wrapper 内部 6 步流程（project → AG → topk → attention → o_proj），用
`_Recording*` / `_ProjectingIndexer` 类捕获中间调用参数，对关键语义逐项 assert
（`use_global_kv=True`、`q` shape、`kv_c`/`k_pe` 拆分、indexer 看到 global `k`、
`query_start_loc` 正确）。是高质量的集成测试 ✅。

### 审查结论

| 类别 | 评估 |
|------|------|
| 代码质量 | ✅ 数据流清晰、错误处理完整、flag off 兼容 |
| 设计文档对齐 | ✅ wrapper CP 路径、global KV attention、backend flag 全部实现 |
| 测试覆盖 | ✅ 6 个新 test 覆盖全链路 + 回归 77 passed |
| 安全性 | ✅ guard 未解除；opaque custom-op 路径 fail-closed |

### 新发现问题

**问题 5** ⚠️ `sharded_cp_topk` Python 双层循环 per-row einsum：当前
correctness 阶段可接受，但 GPU 验证阶段有性能瓶颈。后续需替换为 compiled
kernel。

**问题 6** ✅ 已收口：原 review 将 `global` 理解为 paged-cache physical slot，
因此担心 block-table mapping 被跳过。当前实现的 `use_global_kv=True` 实际传入
的是 all-gather 后连续 global compact KV tensor，不是 paged KV cache；所以
top-k offset 不需要 block-table mapping。代码已把 flag 改名为
`topk_indices_are_global_compact_offsets`，明确它只能和 global compact KV 输入配套使用。

**问题 7** ✅ KV cache update 跳过是纯 prefill 的正确做法。后续支持
incremental prefill 或 decode 时需补回。当前阶段语义正确。

问题 5 和问题 7 仍是后续阶段限制；问题 6 的命名风险已在本阶段修正，不阻塞
CPU UT 后交给 GPU 端到端验证。
