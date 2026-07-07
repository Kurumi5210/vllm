# Sharded Context Parallel Stage 5 总结

## 阶段目标

Stage 5 开始落地 Sharded-CP 下 MLP/MoE token-row layout：

1. 稠密 MLP 是逐 token 计算，CP-local rows 可直接运行，不需要 all-gather
   完整 token 序列。
2. MoE gate / routing 在本地 token rows 上执行。
3. EP dispatch 前把 MoE hidden/router logits 按 CP token-row 顺序 all-gather。
4. expert 输出按 CP token-row reduce-scatter 回本地 rows。
5. 残差连接前确认 attention output 与 MoE output 的 local row 对应同一个全
   局 token。

本阶段仍不解除完整 transformer layer guard。Stage 5 固定了 MoE 边界的数据
布局和 DeepSeek MoE 分支的 Sharded-CP 调用点，但真实 GPU EP kernel、完整
layer logits parity、Shard Linear 和 async prefetch 仍属于后续阶段。

## 代码改动

### `vllm/v1/worker/sharded_cp_moe.py`

新增 Sharded-CP MoE helper。

`ShardedCPMoEInputs`：

1. 保存 all-gather 后的 global `hidden_states`。
2. 保存 all-gather 后的 global `router_logits`。
3. 可选保存 all-gather 后的 global `activation_scales`，为后续量化 MoE
   payload 预留边界。

`ShardedCPMoERoutingMetadata`：

1. 保存 all-gather 后的 global `topk_weights`。
2. 保存 all-gather 后的 global `topk_ids`。
3. token 维度顺序与 baseline pre-CP routing 保持一致。

`assemble_sharded_cp_moe_input_chunks()`：

1. 复用 `assemble_token_all_gather_chunks()`。
2. 从 padded CP chunks 恢复全局 token 顺序。
3. 同时处理 hidden、router logits 和可选 activation scales。
4. 检查 assembled hidden/router/scale 的 row count 一致。

`all_gather_sharded_cp_moe_inputs()`：

1. 校验 local hidden/router logits rows 等于 `token_range.num_tokens`。
2. 校验 hidden/router logits 位于同一 device。
3. 分别 all-gather hidden 和 router logits，返回 global MoE input。
4. 单 rank 走 `all_gather_token_rows()` fast path，不依赖 distributed。

`assemble_sharded_cp_moe_routing_chunks()`：

1. 复用 token-row assemble 逻辑恢复 global top-k routing。
2. 检查 `topk_weights` 与 `topk_ids` shape 一致。
3. 用于 CPU UT 验证 routing metadata gather 顺序和 EP dispatch expert id
   不偏移。

`all_gather_sharded_cp_moe_routing_metadata()`：

1. all-gather local top-k metadata。
2. `topk_ids` padding 使用 `-1`，避免 padding row 看起来像合法 expert id。
3. 单 rank fast path 覆盖无需 distributed 的本地验证。

`reduce_scatter_sharded_cp_moe_output()`：

1. 复用 Stage 1 的 `reduce_scatter_token_rows()`。
2. 将 global expert output contribution reduce-scatter 回 CP-local rows。

`slice_sharded_cp_moe_output()`：

1. 提供 CPU reference path。
2. 从 global expert output 直接切出当前 rank 的 rows。

`combine_sharded_cp_moe_residual()`：

1. 检查 attention output local rows 和 MoE output local rows 都匹配当前
   `token_range`。
2. 检查两者 shape 完全一致。
3. 返回本地残差相加结果。

### `vllm/model_executor/models/deepseek_v2.py`

`DeepseekV2MoE` 新增 Sharded-CP 分支：

1. 初始化时记录 `self.use_sharded_cp_token_parallel`。
2. `_get_sharded_cp_token_range()` 从 `ForwardContext.additional_kwargs` 读取
   `sharded_cp_token_range`。
3. `_forward_sharded_cp()`：
   - 输入必须是 CP-local hidden rows。
   - gate/router 在本地 rows 上执行。
   - `all_gather_sharded_cp_moe_inputs()` 聚合 global hidden/router logits。
   - 复用现有 `self.experts(...)` 做 expert dispatch / compute。
   - `_apply_moe_output_epilogue()` 保留 routed scaling、shared expert 合并和
     FP16 overflow 处理。
   - `reduce_scatter_sharded_cp_moe_output()` 将 global expert output 回到
     CP-local rows。
4. 默认非 Sharded-CP forward 逻辑保持不变，只把原有 MoE epilogue 提取成
   helper，避免重复实现。

`DeepseekV2DecoderLayer` 稠密 MLP 构造更新：

1. 当 `parallel_config.enable_sharded_context_parallel=True` 时，稠密
   `DeepseekV2MLP` 使用 `is_sequence_parallel=True`。
2. 这会让 `gate_up_proj/down_proj` 走 `disable_tp=True` 的 replicated logical
   weight 路径。
3. 稠密 MLP 因为逐 token 无跨 token 依赖，可直接消费 CP-local hidden rows。

## UT 覆盖

新增 `tests/v1/worker/test_sharded_cp_moe.py`：

1. MoE input 单 rank all-gather fast path：
   - hidden/router logits/activation scales 输入即输出。
   - shape 保持 `[T_local, hidden]` / `[T_local, experts]`。

2. request-aligned MoE input assemble：
   - `[100, 200, 50]`、`CP=2`。
   - rank 1 的 50 rows padding 到 300。
   - assemble 后恢复原始 350 rows hidden/router/scale。

3. routing metadata all-gather order：
   - 从 global router logits 计算 baseline top-k。
   - 模拟 CP chunks 后 assemble。
   - assembled `topk_ids/topk_weights` 与 baseline 完全一致。

4. routing metadata 单 rank fast path：
   - 不依赖 `torch.distributed`。
   - `topk_weights/topk_ids` 输入即输出。

5. fake EP dispatch expert 选择：
   - 使用 assembled `topk_ids` 选择专家 bias。
   - 输出与 baseline pre-CP expert id 选择完全一致。
   - 验证没有 silent routing offset。

6. routing metadata shape 校验：
   - `topk_weights/topk_ids` shape 不一致时报 `ValueError`。

7. 稠密 MLP CP-local 等价：
   - deterministic `gate_up -> silu_and_mul -> down` reference。
   - 分 rank 本地计算后拼接，与 global path bitwise 相等。

8. MoE output / residual row 对齐：
   - rank 2 只持有 global token `[6,7)`。
   - local MoE output 与 local attention output 相加后等于 global slice。

9. MoE output reduce-scatter 单 rank fast path：
   - 输入即输出。

10. DeepSeek MoE Sharded-CP 分支：
    - fake gate 在 local rows 上执行。
    - fake experts 收到 all-gather 后的 global hidden/router logits。
    - 返回值 reduce-scatter 回 local rows。

11. DeepSeek MoE forward context：
    - `DeepseekV2MoE.forward()` 从 `ForwardContext.additional_kwargs` 读取
      `sharded_cp_token_range`。
    - Sharded-CP 分支只在 flag 打开时执行。

12. 稠密 decoder MLP 构造：
    - Sharded-CP 打开时 `is_sequence_parallel=True`。
    - flag off 时保持 `is_sequence_parallel=False`。

## 测试结果

新增 MoE UT：

```bash
.venv/bin/python -m pytest tests/v1/worker/test_sharded_cp_moe.py -q
```

结果：

```text
13 passed, 16 warnings in 0.77s
```

Stage 1-5 focused 回归：

```bash
.venv/bin/python -m pytest tests/v1/worker/test_sharded_cp_utils.py tests/v1/worker/test_sharded_cp_boundaries.py tests/v1/worker/test_sharded_cp_metadata.py tests/v1/worker/test_sharded_cp_attention.py tests/v1/worker/test_sharded_cp_moe.py tests/test_sharded_context_parallel_config.py tests/engine/test_arg_utils.py::test_enable_sharded_context_parallel_cli_arg tests/engine/test_arg_utils.py::test_enable_sharded_context_parallel_flows_to_parallel_config tests/engine/test_arg_utils.py::test_enable_sharded_context_parallel_rejects_incompatible_cli_topology -q
```

结果：

```text
90 passed, 16 warnings in 0.88s
```

语法检查：

```bash
.venv/bin/python -m py_compile vllm/v1/worker/sharded_cp_moe.py vllm/model_executor/models/deepseek_v2.py tests/v1/worker/test_sharded_cp_moe.py
```

结果：通过。

## 本阶段未实现内容

Stage 5 仍未实现：

1. 完整 transformer layer guard 解除。
2. 真实 GPU EP all2all / expert kernel 下的 end-to-end logits parity。
3. 量化 activation payload 的真实 packed dtype / scale kernel 接入。
4. Shard Linear synchronous full-weight owner/broadcast。
5. async broadcast / prefetch。
6. decode 或 mixed prefill/decode。

当前 commit 的价值是把 MLP/MoE 的 CP token-row 边界固定下来：稠密 MLP 可
直接在 local rows 上运行，MoE local gate 后按 global token order gather，
expert 输出再回到 CP-local rows。后续解除 guard 前，还需要在 GPU 上跑真实
attention + MoE + Shard Linear 的完整 parity。

---

## 代码审查（2026-07-01）

### 设计文档 vs 代码对照

**变更：**

| 设计要求 | 状态 | 证据 |
|----------|:--:|------|
| 稠密 MLP `is_sequence_parallel=True` | ✅ | `deepseek_v2.py:1258` |
| MoE gate/router 在本地 rows 执行 | ✅ | `_forward_sharded_cp` L427-430 |
| EP dispatch 前 all-gather hidden + router_logits | ✅ | `_forward_sharded_cp` L432-437 |
| Expert 输出 reduce-scatter 回 CP-local | ✅ | `_forward_sharded_cp` L456-459 |
| `_apply_moe_output_epilogue` helper 提取 | ✅ | L389-406，原路径和 CP 路径复用 |
| Stage 5 不解除 guard | ✅ | `_raise_if_sharded_cp_transformer_path_not_ready()` 仍在 |

**测试：**

| 测试 | 证据 |
|------|------|
| MoE input 单 rank fast path | `test_sharded_cp_moe.py:38-54` |
| request-aligned assemble round trip | `:57-94` |
| routing AG → baseline top-k 一致 | `:97-138` |
| routing 单 rank fast path | `:141-152` |
| fake EP dispatch expert 选择 | `:155-182` |
| routing shape 校验 | `:185-194` |
| 稠密 MLP CP-local bitwise | `:197-232` |
| MoE output/residual row 对齐 | `:235-252` |
| MoE RS 单 rank fast path | `:255-260` |
| DeepSeek MoE CP 全链路（fake） | `:263-300` |
| `forward()` 读 forward context | `:303-326` |
| 稠密 decoder MLP SP flag 参数化 | `:361-415` |

### 逐文件审查

#### `vllm/v1/worker/sharded_cp_moe.py` ✅

1. **`ShardedCPMoEInputs` / `ShardedCPMoERoutingMetadata`**：frozen dataclass，
   语义清晰，字段校验完备。

2. **`all_gather_sharded_cp_moe_inputs`（L113-160）**：localhost 校验
   row count/device 一致性后分别 all-gather hidden 和 router_logits，
   可选 activation_scales。单 rank 走 `all_gather_token_rows` fast path ✅。

3. **`assemble_sharded_cp_moe_input_chunks`（L81-111）**：复用 Stage 3
   `assemble_token_all_gather_chunks`，额外校验 assembled 后 hidden 与
   router 行数一致 ✅。

4. **`all_gather_sharded_cp_moe_routing_metadata`（L178-214）**：
   `topk_ids` padding 用 `-1`，避免 padding row 被误识别为 expert 0 ✅。

5. **`reduce_scatter_sharded_cp_moe_output`（L217-228）**：直接复用
   `reduce_scatter_token_rows`，一行代码，无冗余 ✅。

6. **`combine_sharded_cp_moe_residual`（L242-263）**：校验 attention output
   和 MoE output 的 local rows 都匹配 token_range 且 shape 一致。仅在 UT
   中使用，不在 `_forward_sharded_cp` 热路径上 ✅。

#### `vllm/model_executor/models/deepseek_v2.py` ✅⚠️

1. **`_apply_moe_output_epilogue`（L389-406）**：从原 `forward()` 中提取
   routed scaling + shared expert merge + FP16 overflow 处理。原路径和
   `_forward_sharded_cp` 都调用，消除了重复代码 ✅。

2. **`_forward_sharded_cp`（L408-460）**：
   - gate 在 local rows 上执行 ✅
   - `all_gather_sharded_cp_moe_inputs()` 聚合全局 hidden/router_logits ✅
   - `self.experts(...)` 接收全局视图 → EP dispatch 正确 ✅
   - `reduce_scatter_sharded_cp_moe_output()` 回到 CP-local ✅

3. **`is_internal_router` 路径 router_logits 传参** ⚠️：
   ```python
   if self.experts.is_internal_router:
       fused_moe_out = self.experts(
           hidden_states=moe_inputs.hidden_states,
           router_logits=moe_inputs.hidden_states,  # ← 传给 kernel 的是 hidden
       )
   ```
   `is_internal_router=True` 时 gate 在 kernel 内部执行，`router_logits` 参数
   通常被忽略。但传 `moe_inputs.hidden_states` 而非 `moe_inputs.router_logits`
   语义上不准确。当前无害，但建议加注释说明或统一为 `router_logits`。

4. **`_forward_sharded_cp` 的 shape 校验** ✅：入口校验
   `hidden_states.shape[0] == token_range.num_tokens`，防止错配。

5. **稠密 MLP `is_sequence_parallel=True`** ✅：`deepseek_v2.py:1258`。
   稠密 MLP 逐 token 无跨 token 依赖，CP-local rows 直接运行。flag 使得
   `gate_up_proj/down_proj` 走 `disable_tp=True`。

6. **Guard 仍在** ⚠️：`_raise_if_sharded_cp_transformer_path_not_ready()`
   未被移除。MoE 和稠密 MLP 的 CP 路径已就位，但 layer loop 仍然不可达。
   端到端 GPU 验证被阻塞。建议在此 commit 或紧接着的 commit 中移除 guard。

#### `tests/v1/worker/test_sharded_cp_moe.py` ✅

1. **MoE input round trip**：单 rank fast path + request-aligned assemble ✅。
2. **Routing metadata AG order**：与 baseline `torch.topk` 完全一致 ✅。
3. **Fake EP dispatch**：assembled `topk_ids` → expert 选择与 baseline 一致 ✅。
4. **稠密 MLP CP-local bitwise**：float64 精度下 exact match ✅。
5. **MoE output/residual**：rank 2 只持 [6,7)，local attention + local MoE
   = global slice ✅。
6. **DeepSeek MoE CP 全链路**：`_RecordingGate` + `_RecordingExperts` 验证
   gate 在 local 执行、experts 收到 global hidden ✅。
7. **Dense decoder MLP 构造**：参数化 True/False ✅。

### 审查结论

| 类别 | 评估 |
|------|------|
| 代码质量 | ✅ 校验充分、helper 拆分合理、epilogue 消除重复 |
| 设计文档对齐 | ✅ 全部 6 项变更 + 12 项测试已实现 |
| 测试覆盖 | ✅ 13 MoE test + 回归 90 passed |
| 端到端 | ⚠️ guard 未移，GPU 完整 parity 不可达 |

### 发现的问题

**问题 1** ⚠️ `is_internal_router` 时 `router_logits=moe_inputs.hidden_states`：
当前 `is_internal_router=True` 时 kernel 忽略该参数，无害。但语义不准确，
建议改为 `router_logits=moe_inputs.router_logits` 或加注释。

**问题 2** ⚠️ **guard 仍在，端到端无法验证**。MoE 和稠密 MLP 的 CP 适配已
完成，但 `_raise_if_sharded_cp_transformer_path_not_ready()` 仍在 layer loop
入口抛异常。至此 attention（Commit 4）和 MLP/MoE（Commit 5）的 CP token-row
路径均已就位，guard 的移除条件已经满足。建议在此 commit 删除 guard 调用，
使 GPU 端到端 logits parity 测试成为可能（logits 会不正确因为 weights 仍是
replicated full logical，但至少能跑通完整 transformer layer）。

**问题 3** ✅ `combine_sharded_cp_moe_residual` 仅在 UT 中使用。残差连接在
`DeepseekV2DecoderLayer.forward()` 中处理，不在 MoE 内部。helper 设计合理，
提供了 CPU 可验证的 row 对齐校验。
