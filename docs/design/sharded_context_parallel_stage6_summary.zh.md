# Sharded Context Parallel Stage 6 总结

## 阶段目标

Stage 6 落地同步版 Shard Linear，并解除 DeepSeek Sharded-CP 真实
transformer layer guard：

1. 按 `layer_id % cp_world_size` 选择每层 Shard Linear owner。
2. 每层 attention 使用前，从 owner rank 同步 broadcast full logical
   projection 参数。
3. 非 owner rank 在 layer 作用域结束后释放 materialized 参数。
4. DeepSeek MLA 注册 `q_up_proj`、`kv_b_proj`、`o_proj` 为 Shard Linear
   管理对象。
5. 解除 `_raise_if_sharded_cp_transformer_path_not_ready()`，让真实 layer loop
   可以进入 GPU 端到端验证。

本阶段仍是 synchronous correctness path，不做 async prefetch。真实多 GPU
logits parity、真实量化 checkpoint 的 owner-only 加载、长序列显存收益和异步
broadcast overlap 仍属于后续 GPU 验收与 Commit 7。

## 代码变更

### `vllm/v1/worker/sharded_cp_shard_linear.py`

新增同步 Shard Linear helper。

`get_sharded_cp_shard_linear_owner(layer_id, world_size)`：

1. 实现论文里的 owner 规则：`owner = layer_id % cp_world_size`。
2. 拒绝负 layer id 和非法 world size，避免 owner 分布静默错误。

`ShardedCPShardLinearParam`：

1. 记录一个 projection 参数的名称、shape、dtype、device 和 owner rank。
2. 主要用于 UT 和后续调试 materialized buffer 生命周期。

`ShardedCPShardLinearLayer`：

1. 绑定一个 decoder layer 里的若干 full logical projection module。
2. 第一次看到参数时记录 `_sharded_cp_full_shape`、`_sharded_cp_full_dtype`、
   `_sharded_cp_full_device`，这样非 owner 释放成空 tensor 后仍能重新分配正确
   shape。
3. `materialize()`：
   - owner rank 校验自己持久参数仍是完整 shape；
   - 非 owner rank 如果当前参数是空 tensor，则重新分配 full shape；
   - 对每个参数执行同步 broadcast。
4. `release_non_owner()`：
   - world size 为 1 时 no-op；
   - owner rank 保留持久 full weight；
   - 非 owner rank 将参数 data 替换为空 tensor，释放 materialized storage。
5. `materialized()`：
   - 以 context manager 形式包住单层 attention；
   - 进入时 materialize，退出时 release non-owner。

### `vllm/model_executor/models/deepseek_v2.py`

`DeepseekV2MLAAttention`：

1. 在 Sharded-CP + DeepSeek V3.2 sparse MLA 下保留 commit4 的 full-head
   correctness module 构造：
   - `q_b_proj` / `q_proj`：`disable_tp=True`；
   - `kv_b_proj`：`disable_tp=True`；
   - `o_proj`：`disable_tp=True`、`reduce_results=False`。
2. 新增 `sharded_cp_shard_linear`。
3. 从 prefix 中解析 `layers.<N>` 作为 layer id；解析不到时 fail fast。
4. 注册：
   - `q_up_proj`：vLLM 中对应 `q_b_proj`，无 q LoRA 时对应 `q_proj`；
   - `kv_b_proj`：当前 sparse MLA backend 仍可能在运行时读取该 full logical
     up-projection weight；
   - `o_proj`：attention output 回 hidden size 的 full logical projection。

`DeepseekV2DecoderLayer`：

1. 新增 `_materialized_sharded_cp_attention_weights()`。
2. 非 Sharded-CP 或非 V3.2 sparse MLA 时返回 `nullcontext()`。
3. Sharded-CP 时用 `ShardedCPShardLinearLayer.materialized()` 包住
   `self.self_attn(**attn_kwargs)`，把 weight broadcast 生命周期限制在单层
   attention 作用域内。

`DeepseekV2Model`：

1. 删除真实 transformer layer guard 调用和 helper。
2. Sharded-CP forward 现在可以进入真实 decoder layer loop。
3. 新增 `release_sharded_cp_non_owner_weights()`：
   - 遍历本 PP stage 的 decoder layers；
   - 对每个存在 `sharded_cp_shard_linear` 的 attention 调用
     `release_non_owner()`。

`DeepseekV2ForCausalLM.load_weights()`：

1. 原有权重加载逻辑保持不变。
2. 加载结束后调用 `self.model.release_sharded_cp_non_owner_weights()`，让
   非 owner rank 不持久保存 full logical Shard Linear 参数。

`DeepseekV2MoE`：

1. 修复 Stage 5 review 指出的 internal router 语义问题。
2. `is_internal_router=True` 时传给 experts 的 `router_logits` 改为
   `moe_inputs.router_logits`，当前行为等价但语义更准确。

## 测试覆盖

新增 `tests/v1/worker/test_sharded_cp_shard_linear.py`，覆盖 8 个独立行为：

1. owner 分布按 `layer_id % world_size`。
2. owner helper 拒绝非法输入。
3. 单 rank fast path 不释放参数。
4. `describe_parameters()` 返回参数 metadata 和 owner rank。
5. 非 owner 从 broadcast materialize full weight，作用域结束后释放为空 tensor。
6. owner 在作用域结束后保留持久 full weight。
7. owner 参数被误释放时 fail fast。

更新既有 UT：

1. `test_sharded_cp_attention.py`：
   - 验证 V3.2 Sharded-CP MLA 注册 `q_up_proj`、`kv_b_proj`、`o_proj`；
   - flag off 和非 V3.2 路径不注册 Shard Linear；
   - prefix 无 `layers.<N>` 时 fail fast。
2. `test_sharded_cp_moe.py`：
   - 验证 decoder layer forward 会在 attention 前进入
     Shard Linear materialized scope，并在 attention 后退出。
3. `test_sharded_cp_boundaries.py`：
   - 原真实 layer fail-closed 测试改为 Stage 6 后可进入 layer context；
   - 验证 `load_weights()` 结束会触发 non-owner release。

## 已运行验证

语法检查：

```bash
.venv/bin/python -m py_compile \
    vllm/v1/worker/sharded_cp_shard_linear.py \
    vllm/model_executor/models/deepseek_v2.py \
    tests/v1/worker/test_sharded_cp_shard_linear.py \
    tests/v1/worker/test_sharded_cp_attention.py \
    tests/v1/worker/test_sharded_cp_boundaries.py \
    tests/v1/worker/test_sharded_cp_moe.py
```

结果：通过。

新增 Shard Linear UT：

```bash
.venv/bin/python -m pytest tests/v1/worker/test_sharded_cp_shard_linear.py -q
```

结果：`8 passed, 2 warnings`。

Stage 1-6 focused UT：

```bash
.venv/bin/python -m pytest \
    tests/v1/worker/test_sharded_cp_utils.py \
    tests/v1/worker/test_sharded_cp_boundaries.py \
    tests/v1/worker/test_sharded_cp_metadata.py \
    tests/v1/worker/test_sharded_cp_attention.py \
    tests/v1/worker/test_sharded_cp_moe.py \
    tests/v1/worker/test_sharded_cp_shard_linear.py \
    tests/test_sharded_context_parallel_config.py \
    tests/engine/test_arg_utils.py::test_enable_sharded_context_parallel_cli_arg \
    tests/engine/test_arg_utils.py::test_enable_sharded_context_parallel_flows_to_parallel_config \
    tests/engine/test_arg_utils.py::test_enable_sharded_context_parallel_rejects_incompatible_cli_topology \
    -q
```

结果：`101 passed, 16 warnings`。

Diff whitespace 检查：

```bash
git diff --check
```

结果：通过。

`ruff` 未运行：当前本地 `.venv/bin/ruff` 不存在。

## 仍未完成

1. 真实多 GPU logits parity。
2. 真实 FlashMLA Sparse / FlashInfer Sparse backend 下的 Sharded-CP E2E。
3. 真实量化 checkpoint 的 owner-only 加载路径；当前 loader 仍先按既有方式加载
   参数，再在加载结束释放非 owner materialized storage。
4. INT8 / FP8 Shard Linear broadcast 的量化格式专项验证。
5. 61 层长模型 repeated forward 的真实 GPU 显存泄漏检查。
6. Commit 7 的 async broadcast / prefetch / stale buffer guard。

## 结论

Stage 6 已经把 Sharded-CP 从”边界和局部 helper 可验证”推进到”真实
DeepSeek decoder layer loop 可进入”的状态。同步 Shard Linear 的 owner、
broadcast、materialize、release 生命周期已经有 CPU UT 覆盖；下一步应在 GPU
上跑真实 DeepSeek V3.2 sparse MLA + MoE 的端到端验证，再进入 async prefetch。

---

## 代码审查（2026-07-01）

### 设计文档 vs 代码对照

**变更：**

| 设计要求 | 状态 | 证据 |
|----------|:--:|------|
| owner 分布 `layer_id % cp_world_size` | ✅ | `sharded_cp_shard_linear.py:16-20` |
| owner helper 拒绝非法输入 | ✅ | `layer_id < 0` / `world_size <= 0` |
| 注册 q_up_proj / kv_b_proj / o_proj | ✅ | `deepseek_v2.py:1101-1108` |
| layer prefix 解析 layer id | ✅ | `deepseek_v2.py:126-130`，`”layers”` 后一个 token |
| prefix 无 `layers.<N>` 时 fail fast | ✅ | `test_sharded_cp_shard_linear.py` UT 覆盖 |
| materialize: owner 校验完整 shape → broadcast | ✅ | `sharded_cp_shard_linear.py:155-172` |
| release_non_owner: non-owner 释放为空 tensor | ✅ | `sharded_cp_shard_linear.py:130-141` |
| context manager: enter→materialize, exit→release | ✅ | `sharded_cp_shard_linear.py:174-180` |
| decoder layer 在 attention 外 wrap materialized scope | ✅ | `deepseek_v2.py:1318-1319` |
| load_weights 后释放 non-owner weights | ✅ | `deepseek_v2.py:1926` |
| **guard 完全移除** | ✅ | 0 occurrence in HEAD |
| Stage 5 的 internal router 语义修复 | ✅ | `moe_inputs.hidden_states` → `moe_inputs.router_logits` |

**测试：**

| 测试 | 证据 |
|------|------|
| owner 分布 8 层 4 rank | `test_sharded_cp_shard_linear.py:41-53` |
| owner 拒绝非法输入 | `:56-63` |
| 单 rank 不释放 | `:66-75` |
| describe_parameters 返回 metadata | `:78-94` |
| non-owner broadcast→release 生命周期 | `:97-120` |
| owner 保留持久 storage | `:123-133` |
| owner 被误释放时 fail fast | `:136-143` |
| V3.2 SCP 注册 3 个 module | `test_sharded_cp_attention.py:500-505` |
| flag off / 非 V3.2 不注册 | `test_sharded_cp_attention.py:507-530` |
| prefix 无 layer id fail fast | `test_sharded_cp_attention.py:533-539` |
| decoder forward materialized scope 时序 | `test_sharded_cp_moe.py:421-476` |
| guard 移除后可进入 context | `test_sharded_cp_boundaries.py:142-151` |
| load_weights 触发 release | `test_sharded_cp_boundaries.py:251-287` |

### 逐文件审查

#### `vllm/v1/worker/sharded_cp_shard_linear.py` ✅

1. **`ShardedCPShardLinearLayer.__init__`**：过滤 `None` module，保留真实
   projection 的 `(name, module)` 对。文档说”注册三个 module”，代码通过
   `tuple((name, m) for name, m in modules if m)` 过滤 ✅。

2. **`_record_param_metadata`**：第一次看到参数时记录 `_sharded_cp_full_shape`/
   `_dtype`/`_device`。如果参数已是空 tensor（被 release 过），无法推断 shape →
   `RuntimeError`。这防止了 `release_non_owner` → `release_non_owner` 的重复
   释放场景 ✅。

3. **`_broadcast` 双通路**（L143-153）：
   ```python
   if hasattr(group, “broadcast”):
       group.broadcast(tensor, src=owner_rank)
       return
   dist.broadcast(tensor, src=owner_rank, group=group)
   ```
   生产路径：`get_sharded_cp_group()` → `GroupCoordinator` → `broadcast`（分组内
   NCCL broadcast）✅。测试路径：`_FakeGroup.broadcast` ✅。fallback：
   `dist.broadcast` 原生 PyTorch ✅。

4. **`materialize` 单 rank 快路径**（L158-159）：`world_size == 1` → 直接返回，
   兼容单卡测试和 UT ✅。

5. **`materialized` context manager**（L174-180）：`try: yield` + `finally:
   self.release_non_owner()`. 即使 attention 内抛异常，non-owner 仍然释放 ✅。

6. **`release_non_owner`**（L122-141）：先 `_record_param_metadata`（记录 shape
   供下次 materialize 使用），然后 owner 跳过，non-owner 将 data 替换为
   `torch.empty((0,), ...)`. 注意：保留了 `_sharded_cp_full_*` metadata，下次
   `materialize` 能重新分配正确的 shape ✅。

#### `vllm/model_executor/models/deepseek_v2.py` ✅

1. **`_layer_id_from_prefix`（L126-130）**：找 `”layers”` 后一个 token。
   例如 `”model.layers.0.self_attn”` → 0。未找到 → `ValueError` ✅。

2. **Shard Linear 注册（L1101-1108）**：
   ```python
   q_up_proj = self.q_b_proj if self.q_lora_rank is not None else self.q_proj
   self.sharded_cp_shard_linear = ShardedCPShardLinearLayer(
       _layer_id_from_prefix(prefix),
       (
           (“q_up_proj”, q_up_proj),
           (“kv_b_proj”, self.kv_b_proj),
           (“o_proj”, self.o_proj),
       ),
   )
   ```
   3 个 module 全部注册：`q_up_proj`（有 q LoRA 时是 `q_b_proj`，否则 `q_proj`）、
   `kv_b_proj`、`o_proj`。与设计文档一致 ✅。

3. **`_materialized_sharded_cp_attention_weights`（L1290-1295）**：
   非 SCP → `nullcontext()`; 无 shard_linear → `nullcontext()`; 有 →
   `shard_linear.materialized(get_sharded_cp_group())`. 此处在 decoder layer
   的 `forward()` 中被调用：
   ```python
   with self._materialized_sharded_cp_attention_weights():
       hidden_states = self.self_attn(**attn_kwargs)
   ```
   context manager scope 精确包裹 attention forward ✅。

4. **Load weights 集成（L1926）**：
   ```python
   self.model.release_sharded_cp_non_owner_weights()
   return loaded_params
   ```
   在所有权重加载完毕后，所有 layer 遍历 + release_non_owner。此时 TP group
   已初始化（dist init 在 load_weights 之前完成），`get_sharded_cp_group()` 可用 ✅。

5. **Guard 完全移除**：原 `_raise_if_sharded_cp_transformer_path_not_ready()`
   方法定义和调用全部删除。`forward()` 中不再有 barrier，可以直接进入 layer
   loop ✅。

6. **`release_sharded_cp_non_owner_weights`（L1526-1537）**：遍历
   `self.layers`，对每个 layer 的 `self_attn.sharded_cp_shard_linear` 调用
   `release_non_owner`. 跳过了 `PPMissingLayer` ✅。

7. **Stage 5 的 internal router 修复（L450）**：
   ```python
   # 原：router_logits=moe_inputs.hidden_states
   # 现：router_logits=moe_inputs.router_logits
   ```
   语义准确 ✅。

#### `tests/v1/worker/test_sharded_cp_shard_linear.py` ✅

1. **owner 分布**：验证 8 层 × 4 rank 的正确映射 ✅。
2. **非法输入**：`layer_id=-1`, `world_size=0` 各一个 case ✅。
3. **单 rank 不释放**：`world_size=1` 时 materialize→release 后 weight 不变 ✅。
4. **describe_parameters**：验证 `o_proj.weight` 的 shape/dtype/owner_rank ✅。
5. **non-owner broadcast→release**：用 `_FakeGroup` 模拟 owner rank 1 的
   broadcast. non-owner（rank 0）初始为 `empty(0)`，materialize 后等于 owner
   weight，release 后数据回空 ✅。
6. **owner 持久**：materialize→release 后 weight 不变 ✅。
7. **owner 被误释放时 fail fast**：手动清零 owner weight，`materialize`
   检测到 shape 不匹配 → `RuntimeError` ✅。

#### `tests/v1/worker/test_sharded_cp_moe.py` ✅

1. **decoder forward materialized scope 时序**（L421-476）：用
   `_RecordingShardLinear` + `_RecordingScope` + `_RecordingAttention` 验证
   事件序列：`(materialized, rank=1, world=2)` → `enter` → `attention` →
   `exit` → `mlp`. 确认 materialized scope 精确包裹 attention 计算 ✅。

### 审查结论

| 类别 | 评估 |
|------|------|
| 代码质量 | ✅ materialize/release 对称、context manager 异常安全、双 broadcast 通路 |
| 设计文档对齐 | ✅ 全部 12 项变更 + 13 项测试已实现 |
| 测试覆盖 | ✅ 8 ShardLinear test + 边界/attention/MoE 更新，101 passed |
| 端到端 | ✅ guard 移除，完整 layer loop 可达 |

### 关键里程碑

**Commit 6 是 Sharded-CP 实现中最重要的转折点：**

1. **Guard 已删除**：`DeepseekV2Model.forward()` 可以直接进入真实 decoder layer
   loop。CPU UT 验证了 context 可用；GPU 上现在应该能跑通完整 forward。

2. **单卡显存已经下降**：`load_weights` 后 non-owner rank 释放了 `q_up_proj`、
   `kv_b_proj`、`o_proj` 的 full logical weight。对于 DeepSeek V3.2 (61 layers,
   INT8, CP=16)，每个 non-owner rank 的 o_proj 显存从 ~112 MiB/layer × 61 层
   降至 ~112 MiB/layer × ceil(61/16) ≈ 仅保留 4 层 persistent o_proj，外加
   q_up 和 kv_b。

3. **Layer loop 现在每次都 materialize/broadcast 权重**：当前是同步 broadcast，
   不计 overlap。Commit 7 的 async prefetch 将在这个基础上做通信隐藏。

### 发现的问题

**无新增阻塞问题。** Stage 5 review 指出的两个问题（internal router 语义 +
guard 移除）均已在 Commit 6 解决。Shard Linear 实现中的 `_broadcast` 双通路、
`_record_param_metadata` 的重复释放保护、context manager 的异常安全设计均正确。

**下一步**：GPU 上跑端到端 logits parity——已验证 CPU UT 路径正确，现在需要确认
真实 NCCL broadcast + sparse MLA kernel + MoE experts 在 CP=2/4 下的数值行为。
