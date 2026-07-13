# Sharded Context Parallel Debug 记录

本文记录 DSACP / Sharded-CP 在 GPU 端到端启动时的调试过程、修复理由和预期效果。当前日志来源是 `prefill.txt`，启动参数里已经打开：

- `--enable-sharded-context-parallel`
- `--tensor-parallel-size 8`
- `--enable-expert-parallel`
- `--cp-kv-cache-interleave-size 64`

## 1. 加载阶段的 Shard Linear 权重释放过早

### 现象

第一轮 GPU 启动在 `load_model()` 阶段失败。调用链集中在 MLA attention 的加载后处理：

```text
MLAAttention.process_weights_after_loading
  -> get_and_maybe_dequant_weights(self.kv_b_proj)
  -> prep_scale_for_group_broadcast
  -> IndexError: tuple index out of range
```

这说明 `kv_b_proj` 在 MLA 后处理读取权重和 scale 时已经不是正常形状。结合 commit6 的实现看，Sharded-CP 为了节省非 owner rank 的显存，会把 Shard Linear 的非本 rank 权重释放为空 tensor。问题是释放发生在 `DeepseekV2ForCausalLM.load_weights()` 的末尾，而 vLLM 通用的 `process_weights_after_loading()` 还没有执行。

### 根因

MLA 后处理需要完整或至少仍然 materialized 的 `kv_b_proj` 权重来构造运行时需要的投影权重，例如 `W_UK/W_UV`。如果在模型自己的 `load_weights()` 里立即释放非 owner shard，那么后续 attention module 的 `process_weights_after_loading()` 会读到空权重或空 scale，最终在 shape 访问上失败。

这个 bug 不是 kernel 正确性问题，而是加载生命周期顺序错误：

1. `load_weights()` 加载参数。
2. vLLM 通用路径调用各模块的 `process_weights_after_loading()`。
3. Sharded-CP 才能释放非 owner 的 Shard Linear 权重。

原实现把第 3 步提前到了第 1 步末尾。

### 修复

修复分成两处：

- `DeepseekV2ForCausalLM.load_weights()` 不再直接调用 `release_sharded_cp_non_owner_weights()`。
- 增加 `DeepseekV2ForCausalLM.post_process_weights_after_loading()`，在 vLLM 通用加载后处理结束后再释放 Sharded-CP 非 owner 权重。
- 在 `vllm/model_executor/model_loader/utils.py` 的 `process_weights_after_loading()` 末尾增加可选模型级 hook 调用：

```python
post_process = getattr(model, "post_process_weights_after_loading", None)
if post_process is not None:
    post_process()
```

### 为什么这样修

`process_weights_after_loading()` 是 vLLM 已有的统一后处理入口，量化、Marlin、MoE、attention 的加载后转换都在这个阶段完成。把 Sharded-CP 的释放动作放在该入口的末尾，可以保证：

- MLA 后处理仍然能读到 `kv_b_proj` 的真实权重和 scale。
- Sharded-CP 仍然能在服务正式 profile / run 前释放非 owner 权重，保留 commit6 的显存收益。
- 该 hook 是可选的，不改变其他模型的加载行为。

### 预期效果

端到端启动应当不再在 `MLAAttention.process_weights_after_loading()` 中因为 Shard Linear 空权重报错。日志里应能看到模型权重加载完成，并继续进入 `determine_available_memory()` / `profile_run()` 阶段。

当前 `prefill.txt` 已经体现了这一点：日志显示 `Loading weights took ...` 和 `Model loading took ...`，后续错误转移到了 profile dummy forward。

## 2. Profile dummy forward 缺少 per-layer attention metadata

### 现象

第二轮 GPU 启动已经通过模型加载，但在初始化可用显存时失败：

```text
gpu_worker.determine_available_memory
  -> gpu_model_runner.profile_run
  -> gpu_model_runner._dummy_run
  -> DeepseekV2Model._maybe_scatter_to_sharded_cp
  -> get_sharded_cp_token_range_from_forward_context
  -> RuntimeError: Sharded-CP requires per-layer attention metadata in the forward context.
```

`prefill.txt` 中多个 TP worker 都在同一处失败，说明这是统一的 profile 路径问题，不是单卡偶发。

### 根因

Sharded-CP 的真实请求路径需要 request-aligned token range，因为 attention metadata、indexer metadata、FlashMLA sparse metadata 都要按完整 request 边界切分。commit3/commit4 因此要求从 per-layer attention metadata 里读取 `query_start_loc`，再计算每个 CP rank 的 token range。

但 `profile_run()` 调用 `_dummy_run(self.max_num_tokens, is_profile=True)` 时，`_dummy_run` 默认不会构建 attention metadata：

```python
attn_metadata = None
if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
    attn_metadata, _ = self._build_attention_metadata(...)
```

profile 阶段的目标是测显存和预分配通信 buffer，不是跑真实 request-local attention。此时 forward context 存在，但 `attn_metadata` 是 `None`。原 Sharded-CP 逻辑只判断“有 forward context 就必须有 per-layer metadata”，因此在 dummy profile 上误 fail closed。

### 已否决的临时修复

曾经尝试过让 `attn_metadata is None` 返回 `None`，再由模型侧回退到按 token 数均分的 range；同时让 `sharded_cp_forward_context()` 在 dummy metadata 下只注入 `sharded_cp_token_range`。

这个方案已经回退。原因是它把 Sharded-CP 的核心不变量变成了“有 metadata 时 request-aligned，没有 metadata 时 balanced split”。这会让 profile、indexer、FlashMLA sparse metadata 和真实请求路径产生两套语义；一旦真实启动路径里某处因为初始化顺序漏掉 metadata，代码会继续跑下去，而不是 fail closed。这个行为本身就是掩盖错误。

### 修复

根因修复放在 `GPUModelRunner._dummy_run()`：

```python
if self.parallel_config.enable_sharded_context_parallel:
    force_attention = True
init_temp_kv = (
    self.parallel_config.enable_sharded_context_parallel
    and not hasattr(self, "kv_cache_config")
)

with self._temporary_sharded_cp_kv_cache(init_temp_kv):
    slot_mappings_by_group, slot_mappings = self._get_slot_mappings(...)
    attn_metadata, _ = self._build_attention_metadata(...)
    ...
```

也就是：只要打开 Sharded-CP，所有 dummy forward 都必须构建 per-layer attention metadata。若此时正式 KV cache 还没有初始化，`_dummy_run()` 只在函数作用域内临时初始化一个 profiling KV cache，让 slot mapping、block table 和每层 metadata 能正常创建；dummy forward 结束后清理这份临时 cache。

### 为什么这样修

Sharded-CP 的 token range 必须从 `query_start_loc` 计算，且必须 request-aligned。这个要求不应该因为当前是 profile run 就放松，因为 profile 阶段同样会进入模型边界、sparse MLA wrapper、MoE 和 Shard Linear 通信路径。

正确的边界是：

- `attn_metadata` 是 per-layer dict：继续使用 request-aligned range，并 localize 每层 metadata。
- `attn_metadata is None` 或其他非 dict metadata：继续 fail closed，说明上游没有为 Sharded-CP 准备必要 metadata。
- dummy run 自己负责构建 metadata，而不是让模型层绕过 metadata。

### 预期效果

端到端启动应当不再在 `profile_run()` 的 `_dummy_run()` 中因为缺少 per-layer metadata 报错。profile 阶段会真实覆盖 Sharded-CP 的 request-aligned hidden-state scatter、metadata localization、sparse MLA top-k、Shard Linear/MoE 通信 buffer 预热和显存测算。

通过 profile 后，下一步 GPU 端到端验证应关注真实请求路径：

- 服务能完成初始化并监听端口。
- 小 batch / 单请求 prefill 能正常返回。
- 打开 `--enable-sharded-context-parallel` 后，对长 prefill 的每 rank activation、Indexer/MLA sparse 前缀计算、Shard Linear 非 owner 权重显存占用符合预期。

## 3. Profile dummy forward 进入 sparse MLA top-k

### 现象

第三轮 GPU 启动继续推进，已经越过了 “Sharded-CP requires per-layer attention metadata” 的显式检查，但在 sparse MLA wrapper 里失败：

```text
gpu_worker.determine_available_memory
  -> gpu_model_runner.profile_run
  -> gpu_model_runner._dummy_run
  -> DeepseekV2Model.forward
  -> MultiHeadLatentAttentionWrapper.forward
  -> attn_metadata.query_start_loc
  -> AttributeError: 'NoneType' object has no attribute 'query_start_loc'
```

这说明前一轮“metadata=None 时继续跑”的修复只是把错误从模型边界推迟到了 sparse MLA wrapper。wrapper 需要 `query_start_loc` 做 request-local top-k；如果 profile 阶段没有 metadata，它仍然无法正确运行。

### 根因

底层 `MLAAttention.forward_impl()` 已经有 `attn_metadata is None` 的 profile fallback：它会分配 worst-case workspace 并返回 zero output，用于显存测算。

但 DSACP 在 `MultiHeadLatentAttentionWrapper` 外层增加了 sparse indexer 的 Sharded-CP top-k 更新。这个更新需要 request-local `query_start_loc`，因为它要按 request 边界保证 causal prefix top-k 不跨 request。profile dummy forward 缺 metadata 时，外层 wrapper 无法安全产生 top-k。

因此，当前失败不是底层 MLA 不支持 profile，而是外层 Sharded-CP sparse indexer 在 profile fallback 前抢先读取了不存在的 `query_start_loc`。

### 修复

曾经尝试过在 Sharded-CP sparse MLA wrapper 中判断 `attn_metadata is None`，然后跳过 `indexer.sharded_cp_topk(...)`。这个方案也已经回退。

当前修复是让 profile run 构建真实 per-layer attention metadata。因此 wrapper 不需要也不应该跳过 top-k：

- `attn_metadata` 是 per-layer dict 时，取 `f"{prefix}.attn"` 对应的 metadata。
- 使用 request-local `query_start_loc` 更新 sparse top-k。
- compact KV all-gather 和 `MLAAttention(..., use_global_kv=True)` 继续执行。
- 如果 metadata 仍然缺失，代码在 `attn_metadata.query_start_loc` 处 fail closed，说明 runner profile metadata 构建没有生效。

### 为什么这样修

只跳过 sparse top-k 看似能让 profile 继续，但会漏测一段 Sharded-CP 真实热路径，而且会允许 metadata 初始化错误静默通过。更稳妥的方案是在 `_dummy_run()` 层补齐依赖：Sharded-CP 统一强制 attention metadata，并在真实 KV cache 尚未初始化时使用函数内临时 KV cache。

这样 profile 阶段和真实请求阶段使用同一类 per-layer metadata，只是 profile 的 request/token 形态由 `_dummy_run()` 合成。

### 预期效果

端到端启动应当不再在 `attn_metadata.query_start_loc` 处失败。若仍失败，优先检查 `_dummy_run()` 是否在 Sharded-CP 下实际构造 per-layer metadata，以及临时 profiling KV cache 是否成功初始化出 metadata builders 并在退出时清理。

## 4. Opaque MLA op 不支持 global compact KV

### 现象

第四轮 GPU 启动继续推进，已经越过了 `query_start_loc` 的 profile fallback 问题，但在底层 MLA attention 入口失败：

```text
MultiHeadLatentAttentionWrapper.forward
  -> self.mla_attn(..., use_global_kv=True)
  -> MLAAttention.forward
  -> RuntimeError: Sharded-CP global compact KV requires direct MLA calls.
```

### 根因

Sharded-CP sparse MLA 在每个 CP rank 上先计算本地 `kv_c_normed/k_pe/indexer_k`，再通过 compact KV all-gather 拼成 global compact KV，然后调用底层 MLA：

```python
self.mla_attn(..., use_global_kv=True)
```

`use_global_kv=True` 的含义是：本次 attention 不走 layer 内部 KV cache update，而是直接使用 wrapper 已经准备好的 compact KV tensor。这个路径只能在 `MLAAttention.use_direct_call=True` 时工作，因为 direct call 分支可以把 `attn_kv = torch.cat((kv_c_normed, k_pe.squeeze(1)), dim=-1)` 直接传给 backend。

服务器日志里的平台默认让 MLA 走 opaque/unified op 路径。该路径只知道从 layer name 找上下文里的 KV cache，不接受外部传入的 global compact KV，所以代码里主动 fail closed。

### 修复

在 `MultiHeadLatentAttentionWrapper` 初始化完成底层 `MLAAttention` 后，如果开启的是 Sharded-CP sparse MLA，则强制：

```python
self.mla_attn.use_direct_call = True
```

### 为什么这样修

这是 Sharded-CP sparse MLA 的功能约束，不只是 profile 阶段的特殊情况。只要打开 Sharded-CP sparse MLA，wrapper 就会在 forward 中传 `use_global_kv=True`；因此底层 MLA 必须使用 direct call。把开关放在 wrapper 初始化阶段，能保证 profile、warmup、真实请求都走同一条可支持 global compact KV 的路径。

这不会影响非 Sharded-CP 或非 sparse MLA：

- Sharded-CP 未开启时不改底层 MLA 的平台默认路径。
- 非 sparse MLA 没有 Sharded-CP compact KV all-gather，不需要强制 direct call。

### 预期效果

端到端启动应当不再因为 opaque MLA op 不支持 `use_global_kv=True` 失败。下一轮 GPU 日志如果继续失败，应当已经进入 direct MLA backend 的 profile fallback、正式 KV cache 初始化或后续真实请求路径。

## 5. FP8 sparse MLA metadata 本地化

### 现象

第五轮 GPU 启动已经进入 Sharded-CP 的真实 per-layer metadata localization，但在 FlashMLA sparse metadata 本地化时失败：

```text
sharded_cp_forward_context
  -> build_sharded_cp_attention_metadata
  -> localize_flashmla_sparse_metadata
  -> RuntimeError: Sharded-CP metadata does not support FP8 sparse MLA metadata yet.
```

日志里 engine config 显示 `quantization=fp8`，因此 FlashMLA sparse metadata 带有 `fp8_extra_metadata` 或 `fp8_use_mixed_batch=True`。

### 根因

原来的 Sharded-CP localizer 只支持 BF16 sparse MLA metadata，遇到 FP8 metadata 直接 fail closed。这个检查在早期是合理的，因为普通 FP8 sparse MLA 使用的是 paged FP8 KV cache metadata，包括 scheduler metadata、dummy block table、cache lens、prefill workspace 信息等。

但 Sharded-CP sparse MLA 的执行语义不同：wrapper 已经通过 compact KV all-gather 构造了 global compact `kv_c_normed || k_pe`。此时传给底层 attention 的不再是原始 FP8 paged KV cache，而是可直接按 top-k global compact row offset 读取的 BF16 compact KV。

因此，继续保留原始 FP8 paged-cache metadata 反而是错的；Sharded-CP local metadata 应该转成 global compact KV 语义。

### 修复

`localize_flashmla_sparse_metadata()` 不再拒绝 FP8 metadata。它仍然按 request 边界切分基础 metadata：

- `query_start_loc`
- `slot_mapping`
- `block_table`
- `req_id_per_token`
- `num_reqs`
- `num_actual_tokens`

但在返回 local metadata 时清掉 FP8 paged-cache 专用状态：

```python
fp8_extra_metadata=None
fp8_use_mixed_batch=False
topk_indices_are_global_compact_offsets=True
```

同时，`FlashMLASparseImpl.forward_mqa()` 在看到 `topk_indices_are_global_compact_offsets=True` 时，即使 `kv_cache_dtype == "fp8_ds_mla"`，也走 BF16 compact KV path，而不是 FP8 paged-cache path：

```python
use_fp8_cache = (
    self.kv_cache_dtype == "fp8_ds_mla"
    and not attn_metadata.topk_indices_are_global_compact_offsets
)
```

### 为什么这样修

这不是跳过 FP8 metadata，而是把 FP8 模型配置下的 Sharded-CP attention 输入语义纠正为 global compact KV。否则底层 FP8 path 会假设输入是 paged FP8 cache，并读取 `fp8_extra_metadata` 中的 scheduler / block table 信息；而 Sharded-CP wrapper 实际传入的是 all-gather 后的 dense compact KV tensor。

### 预期效果

端到端启动应当不再在 `Sharded-CP metadata does not support FP8 sparse MLA metadata yet` 处失败。下一轮如果仍失败，应当已经进入 sparse FlashMLA compact KV kernel、后续 profile memory、正式 KV cache 初始化或真实请求路径。

## 6. Profile logits selection 使用全局索引访问局部 hidden

### 现象

第六轮 GPU 启动已经越过 FP8 sparse metadata localize，继续进入 profile dummy forward。最新 `prefill.txt` 的第一个真实错误是：

```text
/pytorch/aten/src/ATen/native/cuda/IndexKernelUtils.cu:16:
vectorized_gather_kernel ... Assertion `ind >=0 && ind < ind_dim_size
&& "vectorized gather kernel index out of bounds"` failed.
```

随后 Python 栈显示错误浮现在：

```text
GPUModelRunner.profile_run
  -> _dummy_sampler_run
  -> torch.rand_like(hidden_states)
  -> CUDA error: device-side assert triggered
```

这里的 `_dummy_sampler_run` 不是根因。CUDA kernel 是异步执行的，真正的越界发生在更早的 profile forward 尾部，后续 `rand_like()` 只是第一个触发同步报错的位置。

### 根因

打开 Sharded-CP 后，DeepSeek forward 末尾返回的是当前 CP rank 的局部 token rows。以 profile 为例：

```python
hidden_states, positions = self._maybe_scatter_to_sharded_cp(...)
...
return hidden_states
```

但 `GPUModelRunner._dummy_run()` 在模型 forward 返回后仍按全局 request 末尾位置计算：

```python
logit_indices = np.cumsum(num_scheduled_tokens) - 1
return hidden_states, hidden_states[logit_indices_device]
```

`logit_indices` 是全局 batch token offset；`hidden_states` 已经是局部 token rows。TP/CP rank 不为 0 或 request-aligned 分片较小时，`logit_indices` 很容易大于本地 `hidden_states.shape[0]`，于是 PyTorch 自己的 gather kernel 触发 `index out of bounds`。

真实请求路径也有同类边界：

```python
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

这里 `logits_indices` 同样来自全局 `query_start_loc`，不能直接用于 DSACP 局部 hidden。

### 修复

在 `DeepseekV2ForCausalLM` 增加 `prepare_hidden_states_for_logits()`：

```python
hidden_states = all_gather_token_rows(hidden_states, token_range, group=...)
self.model.sharded_cp_token_range = None
self._sharded_cp_logits_hidden_states_prepared = True
return hidden_states
```

含义是：在 runner 使用全局 logits/sample indices 之前，先把 DSACP 局部 token rows all-gather 回全局 token rows。之后再按全局 `logits_indices` 或 profile `logit_indices` 取样。

同时在 `GPUModelRunner` 增加统一入口：

```python
hidden_states, sample_hidden_states = self._select_hidden_states_for_logits(
    hidden_states, logits_indices
)
```

该 helper 会先调用模型可选的 `prepare_hidden_states_for_logits()`，再执行索引。非 Sharded-CP 模型没有这个方法，继续保持原来的直接索引行为。

`compute_logits()` 仍保留直接接收 DSACP 局部 hidden 的兼容能力：如果调用者没有提前 prepare，它会自己 all-gather；如果 runner 已经 prepare 过，则不会二次 all-gather。

### 为什么这样修

这不是跳过 profile，也不是跳过 sampler。根因是 runner 和模型对 hidden-state token 维度的语义不一致：

- runner 的 `logits_indices` / `logit_indices` 是全局 batch token offset。
- DSACP 模型 forward 后的 hidden 是 CP-local token rows。
- logits 计算前的正确边界是“先恢复全局 token rows，再按全局 index 取样”。

把修复放在 logits-selection 边界上，可以覆盖 profile dummy run 和真实 execute path，并保留 `compute_logits()` 的 fail-closed 行为：如果 DSACP forward 后完全没有 token_range，也没有经过 prepare，仍然报 `Sharded-CP logits require a token range from model forward.`。

### 预期效果

端到端启动应当不再在 profile 阶段触发 PyTorch `vectorized_gather_kernel index out of bounds`。如果后续仍有 CUDA assert，需要重新看第一个 kernel 报错位置；但这次日志中的 profile logits selection 越界已经由 CPU UT 直接覆盖。

## 7. 已补充的 CPU UT

本次修复对应补充了以下 CPU 可验证点：

- `load_weights()` 不再提前释放 Sharded-CP 非 owner 权重。
- 通用 `process_weights_after_loading()` 会调用模型级 `post_process_weights_after_loading()` hook。
- `get_sharded_cp_token_range_from_forward_context()` 对缺失 metadata 和非 per-layer dict metadata 都 fail closed。
- `sharded_cp_forward_context()` 对缺失 metadata fail closed，不再注入一个只有 `sharded_cp_token_range` 的 dummy context。
- `GPUModelRunner.profile_run()` 将 Sharded-CP dummy setup 统一委托给 `_dummy_run()`，避免 profile 和 warmup 走不同语义。
- `GPUModelRunner._dummy_run()` 在 Sharded-CP 开启时统一强制构造 attention metadata，并在缺少真实 KV cache 时只创建函数内临时 profiling KV cache。
- `_temporary_sharded_cp_kv_cache()` 在成功和异常路径都会清理临时 KV cache、attention backend 和 layer KV 绑定，避免污染后续真实 `initialize_kv_cache()`。
- sparse MLA wrapper 不再对 `attn_metadata=None` 跳过 `sharded_cp_topk`。
- Sharded-CP sparse MLA 初始化时强制底层 MLA 使用 direct call，以支持 global compact KV。
- FP8 FlashMLA sparse metadata localize 后会清掉 FP8 paged-cache extra metadata，并标记为 global compact top-k offsets。
- FP8 cache 配置下，只要 metadata 标记为 global compact offsets，FlashMLA sparse backend 就走 BF16 compact KV path。
- `GPUModelRunner._select_hidden_states_for_logits()` 会先调用模型的 logits hidden prepare，再按全局 logits indices 索引。
- 非 Sharded-CP 模型没有 logits hidden prepare 时，runner 仍保持原来的直接索引行为。
- `DeepseekV2ForCausalLM.prepare_hidden_states_for_logits()` 会 all-gather DSACP 局部 hidden，并清掉 token range。
- runner 已经 prepare 过的 DSACP hidden 传给 `compute_logits()` 时不会二次 all-gather。
- 未 prepare 且缺 token range 的 DSACP logits 路径继续 fail closed。

建议验证命令：

```bash
.venv/bin/python -m pytest \
  tests/v1/worker/test_sharded_cp_metadata.py \
  tests/v1/worker/test_sharded_cp_boundaries.py \
  tests/v1/worker/test_gpu_model_runner.py::test_profile_run_delegates_dummy_setup_for_sharded_cp \
  tests/v1/worker/test_gpu_model_runner.py::test_profile_run_keeps_default_dummy_path_without_sharded_cp \
  tests/v1/worker/test_gpu_model_runner.py::test_temporary_sharded_cp_kv_cache_cleans_after_success \
  tests/v1/worker/test_gpu_model_runner.py::test_temporary_sharded_cp_kv_cache_cleans_after_exception \
  tests/v1/worker/test_gpu_model_runner.py::test_select_hidden_states_for_logits_prepares_before_indexing \
  tests/v1/worker/test_gpu_model_runner.py::test_select_hidden_states_for_logits_keeps_default_indexing \
  tests/v1/worker/test_sharded_cp_shard_linear.py -q
```

当前聚焦验证已跑：

```text
.venv/bin/python -m pytest \
  tests/v1/worker/test_gpu_model_runner.py::test_profile_run_delegates_dummy_setup_for_sharded_cp \
  tests/v1/worker/test_gpu_model_runner.py::test_profile_run_keeps_default_dummy_path_without_sharded_cp \
  tests/v1/worker/test_gpu_model_runner.py::test_temporary_sharded_cp_kv_cache_cleans_after_success \
  tests/v1/worker/test_gpu_model_runner.py::test_temporary_sharded_cp_kv_cache_cleans_after_exception \
  tests/v1/worker/test_gpu_model_runner.py::test_select_hidden_states_for_logits_prepares_before_indexing \
  tests/v1/worker/test_gpu_model_runner.py::test_select_hidden_states_for_logits_keeps_default_indexing \
  tests/v1/worker/test_sharded_cp_boundaries.py::test_prepare_hidden_states_for_logits_clears_sharded_cp_token_range \
  tests/v1/worker/test_sharded_cp_boundaries.py::test_compute_logits_uses_prepared_sharded_cp_hidden_without_second_gather \
  tests/v1/worker/test_sharded_cp_boundaries.py::test_compute_logits_rejects_missing_unprepared_sharded_cp_token_range \
  -q

9 passed, 16 warnings
```

```text
.venv/bin/python -m pytest \
  tests/v1/worker/test_sharded_cp_metadata.py \
  tests/v1/worker/test_sharded_cp_attention.py::test_flashmla_sparse_global_compact_offsets_ignore_fp8_cache_path \
  tests/v1/worker/test_sharded_cp_attention.py::test_wrapper_sharded_cp_uses_global_compact_kv -q

17 passed, 16 warnings
```

完整 Sharded-CP 聚焦回归：

```bash
.venv/bin/python -m pytest \
  tests/v1/worker/test_gpu_model_runner.py::test_profile_run_delegates_dummy_setup_for_sharded_cp \
  tests/v1/worker/test_gpu_model_runner.py::test_profile_run_keeps_default_dummy_path_without_sharded_cp \
  tests/v1/worker/test_gpu_model_runner.py::test_temporary_sharded_cp_kv_cache_cleans_after_success \
  tests/v1/worker/test_gpu_model_runner.py::test_temporary_sharded_cp_kv_cache_cleans_after_exception \
  tests/v1/worker/test_gpu_model_runner.py::test_select_hidden_states_for_logits_prepares_before_indexing \
  tests/v1/worker/test_gpu_model_runner.py::test_select_hidden_states_for_logits_keeps_default_indexing \
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

本次本地结果：

```text
115 passed, 16 warnings
```

## 8. 后续 GPU 验证建议

同步修复后，先复跑相同 `vllm serve` 命令。判断顺序：

1. 如果仍在 `load_model()` 失败，优先检查 `post_process_weights_after_loading()` 是否已经同步到服务器代码。
2. 如果仍在 dummy forward 报 per-layer metadata，优先检查 `GPUModelRunner._dummy_run()` 是否在 Sharded-CP 下统一强制构造 attention metadata，以及临时 profiling KV cache 是否只在函数作用域内初始化并清理。
3. 如果仍在 FP8 sparse metadata localize 失败，优先检查 `localize_flashmla_sparse_metadata()` 是否已经同步到服务器代码。
4. 如果仍出现 `vectorized_gather_kernel index out of bounds`，优先确认 `GPUModelRunner._select_hidden_states_for_logits()` 和 `DeepseekV2ForCausalLM.prepare_hidden_states_for_logits()` 是否已经同步到服务器代码。
5. 如果服务启动成功，再用单请求短 prompt 验证功能返回，然后逐步提高 prefill 长度和 batch。
6. 若后续出现 OOM，再看日志中的 `Model loading took ... GiB memory`、`max_num_batched_tokens`、`gpu_memory_utilization` 和 KV cache block 数，区分是权重加载峰值、profile 峰值还是真实 prefill 峰值。

## 9. `Attention backends are already initialized`

### 日志现象

同步 `_dummy_run()` 统一构造 attention metadata 之后，服务启动进入真实 KV cache 初始化阶段时报：

```text
File ".../gpu_worker.py", line 556, in initialize_from_config
  self.model_runner.initialize_kv_cache(kv_cache_config)
File ".../gpu_model_runner.py", line 6534, in initialize_kv_cache
  self.initialize_attn_backend(kv_cache_config)
File ".../gpu_model_runner.py", line 5895, in initialize_attn_backend
  assert len(self.attn_groups) == 0, "Attention backends are already initialized"
AssertionError: Attention backends are already initialized
```

### 根因

Sharded-CP 在 profile / warmup / cudagraph dummy forward 中都需要真实 per-layer attention metadata。`_dummy_run()` 在真实 KV cache 尚未初始化时会临时创建一个最小 profiling KV cache，以便 `_build_attention_metadata()` 能正常工作。

前一版修复把临时 KV cache 初始化下沉到 `_dummy_run()`，但没有把清理动作绑定到同一个作用域。结果 profile 阶段创建的 `kv_cache_config`、`attn_groups`、metadata builders 和 layer KV 绑定会残留到后续真实 `initialize_kv_cache()`，真实初始化再次调用 `initialize_attn_backend()` 时看到 `attn_groups` 非空，于是 fail closed。

### 修复方式

新增 `_temporary_sharded_cp_kv_cache()`：

- 只有在 Sharded-CP 开启且 runner 尚无真实 `kv_cache_config` 时创建临时 profiling KV cache。
- 将 slot mapping、attention metadata 构造、model forward 和 drafter dummy run 都放在同一个临时 KV 作用域内。
- `finally` 中调用 `_cleanup_profiling_kv_cache()`，成功和异常路径都会清理临时 `kv_caches`、`cross_layers_kv_cache`、`attn_groups`、`kv_cache_config` 和静态 attention layer 上的 `kv_cache` 绑定。

这不是跳过 profile，也不是跳过 metadata；它只是把临时资源的所有权收回到 `_dummy_run()` 内部，避免 profile 阶段状态泄漏到真实初始化阶段。

### 预期效果

端到端启动不应再在 `initialize_from_config -> initialize_kv_cache -> initialize_attn_backend` 阶段触发 `Attention backends are already initialized`。如果后续还有失败，应当继续看第一个新的异常栈；该栈之前的临时 attention backend 泄漏已经由 CPU UT 覆盖。

## 10. DeepSeek V3.2 indexer decode metadata 本地化

### 日志现象

修复临时 KV cache 生命周期后，服务启动进入真实 KV 初始化后的 warmup。最新失败发生在 sampler/logits warmup 的 `_dummy_run(num_tokens=max_num_reqs, ...)`：

```text
gpu_worker.compile_or_warm_up_model
  -> gpu_model_runner._dummy_run
  -> DeepseekV2Model.forward
  -> sharded_cp_forward_context
  -> localize_deepseek_v32_indexer_metadata
  -> RuntimeError: Sharded-CP metadata does not support decode batches yet.
```

这次 `_dummy_run()` 合成的是每个请求 1 个 query token 的 batch。DeepSeek V3.2 indexer metadata builder 会通过 `split_decodes_and_prefills()` 把它判定为纯 decode batch，因此旧 localizer 的 fail-closed 分支被真实触发。

### 根因

早期 Sharded-CP 只实现了 DeepSeek V3.2 indexer 的 prefill metadata 本地化，所以 `localize_deepseek_v32_indexer_metadata()` 对任何 decode batch 都直接报错。这对 commit4/5 的边界验证是合理的，但 commit6 已经让真实 layer loop、sparse indexer 和 logits path 都进入 Sharded-CP 热路径后，decode metadata 不再是可选项。

这不是 profile run 的特殊问题，也不能通过把 warmup 改成 prefill 形状解决。真实在线解码同样会产生 decode metadata；如果只改 dummy batch shape，服务启动可能绕过去，但第一轮 decode 请求仍会失败。

DeepSeek V3.2 indexer decode metadata 有两种行布局：

- 普通或 native speculative decode：`decode_lens/seq_lens/block_table` 按 decode request 计数。
- flatten speculative decode：builder 会把多 token decode 展平成每 token 一行，此时 metadata 行数等于 `num_decode_tokens`。

Sharded-CP 的 request-aligned token range 保证不会切开一个 request，但 localizer 仍必须同时支持这两种 decode 行布局。

### 修复方式

本次修复把 DeepSeek V3.2 indexer localizer 从 “遇到 decode 直接报错” 改成真实本地化：

- 按 `query_start_loc` 找到当前 CP rank 的 request slice。
- 计算本地 decode request 数、本地 decode token 数、本地 prefill request/token 数。
- 对普通 decode metadata 按 request slice 切 `decode.block_table/seq_lens/decode_lens`。
- 对 flatten decode metadata 按 decode token slice 切 metadata 行。
- mixed decode/prefill batch 中，decode 保持本地 token 前缀，prefill chunk 继续用已有 request-aligned chunk rebuild 逻辑，并把 `token_start/token_end` 调整成本地 token offset。
- CUDA + DeepGEMM decode path 下，基于本地 `seq_lens` 和 indexer block size 重建 paged MQA schedule metadata，避免本地 batch 仍复用全局调度元数据。
- `DeepSeekV32IndexerDecodeMetadata` 增加 `block_size` 字段，由 builder 写入真实 `kv_cache_spec.block_size`，localizer 用它重建 schedule metadata。

### 为什么这样修

`sparse_attn_indexer()` 消费的是 forward context 中本地化后的 metadata，并且它用的是本地 token row：

```python
q_fp8[:num_decode_tokens]
weights[:num_padded_tokens]
decode_metadata.block_table
decode_metadata.seq_lens
```

因此 localizer 必须把 decode 相关计数和 metadata 都变成本地视图。保留全局 `num_decode_tokens` 或全局 decode metadata 会让本地 rank 用全局 token 前缀索引本地 `q_fp8/weights`，后续很容易变成越界或错误 top-k。

这次修复不是跳过 warmup，也不是把 decode 改成 prefill；它让 Sharded-CP 的 indexer metadata 在 profile、warmup 和真实 decode 请求中保持同一套 request-aligned 语义。

### 预期效果

端到端启动不应再在 `Sharded-CP metadata does not support decode batches yet` 处失败。同步后重新跑同一个 `vllm serve`，如果仍失败，应当继续看新的第一个异常栈；这条 decode metadata fail-closed 路径已经由 CPU UT 覆盖。

新增 CPU 覆盖：

- 纯 1-token decode batch 本地化，覆盖当前 warmup 失败形态。
- mixed decode + prefill batch 本地化，覆盖 decode prefix 和 prefill chunk 同时存在的形态。
- flatten multi-token decode metadata 本地化，覆盖 speculative decode 被 builder 展平成 token 行的形态。

本次本地验证：

```text
.venv/bin/python -m pytest tests/v1/worker/test_sharded_cp_metadata.py -q

17 passed, 16 warnings
```

完整 Sharded-CP 聚焦回归：

```text
.venv/bin/python -m pytest \
  tests/v1/worker/test_gpu_model_runner.py::test_profile_run_delegates_dummy_setup_for_sharded_cp \
  tests/v1/worker/test_gpu_model_runner.py::test_profile_run_keeps_default_dummy_path_without_sharded_cp \
  tests/v1/worker/test_gpu_model_runner.py::test_temporary_sharded_cp_kv_cache_cleans_after_success \
  tests/v1/worker/test_gpu_model_runner.py::test_temporary_sharded_cp_kv_cache_cleans_after_exception \
  tests/v1/worker/test_gpu_model_runner.py::test_select_hidden_states_for_logits_prepares_before_indexing \
  tests/v1/worker/test_gpu_model_runner.py::test_select_hidden_states_for_logits_keeps_default_indexing \
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

117 passed, 16 warnings
```

## 11. 单请求/少请求 batch 导致 CP-local token 为 0

> 更新：Stage 7 最终实现已经从 request-aligned partition 改为 balanced token
> split，并补齐 split request metadata/top-k 语义。下面这一节保留当时 debug
> 结论，用来解释为什么早期实现会出现大量 0-token rank；它不再代表最终
> Stage 7 的 token partition 策略。

### 日志现象

服务启动后，单条请求进入真实 forward 时，Sharded-CP token range 日志显示：

```text
rank=0/8 start=0 end=4 tokens=4 total=4
rank=1/8 start=4 end=4 tokens=0 total=4
...
rank=7/8 start=4 end=4 tokens=0 total=4
```

随后空 rank 仍进入 sparse MLA 的本地 FP8 projection：

```text
MultiHeadLatentAttentionWrapper.forward
  -> self.fused_qkv_a_proj(hidden_states)[0]
  -> cutlass_scaled_mm
  -> RuntimeError: Error Internal
```

### 为什么会有 0-token rank

Sharded-CP 当前使用 request-aligned token partition。也就是说，CP 边界只能落在
request boundary 上，不能把同一条 request 的 query rows 强行切到多个 rank。

例如单条请求 `query_start_loc=[0, 4]`，`cp_world_size=8` 时，合法 request boundary
只有 `0` 和 `4`。为了保持 request 完整，range 只能是：

```text
rank0: [0, 4)
rank1: [4, 4)
rank2: [4, 4)
...
rank7: [4, 4)
```

因此，batch 请求数少于 CP rank 数、或者某些请求很长但不能跨 rank 切分时，
0-token rank 是合法且必然会出现的状态。

### 根因

0-token rank 仍然要参与 collective，否则其他 rank 会在 compact KV all-gather、
MoE all-gather / reduce-scatter、logits all-gather 等通信点死锁。

但 0-token rank 不应该执行本地 CUDA compute kernel。当前失败点是 FP8 CUTLASS
linear 不支持 `M=0` 的输入矩阵。即使绕过第一个 `fused_qkv_a_proj`，后面仍可能在：

- `RMSNorm` custom op
- `SiluAndMul` custom op
- `GateLinear` router GEMM
- sparse MLA indexer quant/top-k
- MLA attention backend
- `o_proj`

继续触发同类 0 行 kernel 问题。

### 已否决的修法

不能在 `DeepseekV2DecoderLayer.forward()` 顶层看到 `hidden_states.shape[0] == 0`
就直接返回。这样空 rank 会跳过 attention / MoE 内部 collective，而非空 rank
仍会进入这些 collective，最终造成通信顺序不一致或 hang。

正确边界是：**空 rank 仍进入 layer loop 和所有必要 collective，只跳过纯本地
0 行 compute kernel。**

### 修复方式

本次修复分四层：

1. `LinearBase` 增加空 batch fast path。`ReplicatedLinear`、`ColumnParallelLinear`、
   `RowParallelLinear` 在本地 batch 维为 0 时直接构造正确 shape 的空输出，
   不进入 quant/GEMM kernel。

2. `RMSNorm` 和 `SiluAndMul` 对空 batch 直接返回空同形状输出，避免进入 custom op。

3. `GateLinear` 在空 batch 时构造 `[0, n_experts]` router logits，保留
   `out_dtype` 和 bias 语义，不进入 DSV3 router GEMM / cuBLAS / fallback GEMM。

4. `MultiHeadLatentAttentionWrapper` 在 Sharded-CP sparse MLA 且本地 token 为 0 时：
   - 构造空 `kv_c_normed/k_pe/indexer_k`。
   - 仍调用 `all_gather_sharded_cp_compact_kv()`，保证 compact KV collective 顺序一致。
   - 不调用本地 q/k/v projection、indexer project/top-k、MLA attention backend 和
     `o_proj`。
   - 返回 `[0, hidden_size]` 空 hidden 给后续 layer。

MoE 侧也补了空 token range 分支：空 rank 不调用本地 gate，而是创建
`[0, n_routed_experts]` 的空 router logits，再进入已有的
`all_gather_sharded_cp_moe_inputs()`。

### 预期效果

单请求或少请求 batch 下，非 owner/空 token rank 不应再因为 FP8 CUTLASS
`M=0` 报 `RuntimeError: Error Internal`。所有 rank 仍会以相同顺序执行必要
collective，因此不会引入新的 hang。

GPU 复测时，建议继续用单条短请求验证：

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","prompt":"hello","max_tokens":4,"temperature":0}'
```

如果还有新错误，应优先看第一个异常栈。如果栈已经越过
`fused_qkv_a_proj(hidden_states)`，说明这次 0-token FP8 projection 问题已经修复；
后续再按新的 kernel 或 metadata 栈继续定位。

本次本地验证：

```text
env PYTHONPYCACHEPREFIX=/private/tmp/vllm_pycache .venv/bin/python -m py_compile \
  vllm/model_executor/layers/linear.py \
  vllm/model_executor/layers/layernorm.py \
  vllm/model_executor/layers/activation.py \
  vllm/model_executor/layers/mla.py \
  vllm/model_executor/layers/fused_moe/router/gate_linear.py \
  vllm/model_executor/models/deepseek_v2.py

.venv/bin/python -m pytest \
  tests/model_executor/test_enabled_custom_ops.py::test_empty_batch_rms_norm_and_silu_and_mul \
  tests/v1/worker/test_sharded_cp_utils.py \
  tests/v1/worker/test_sharded_cp_boundaries.py \
  tests/v1/worker/test_sharded_cp_metadata.py \
  tests/v1/worker/test_sharded_cp_attention.py \
  tests/v1/worker/test_sharded_cp_moe.py \
  -q

87 passed, 16 warnings
```

## Stage 7 top-k 路径纠偏：复用原 SparseAttnIndexer kernel

### 触发背景

Claude review 指出 Ascend PR #4702 并没有重写 `npu_lightning_indexer`，而是：

1. all-gather 本地 Indexer K。
2. 调整 CP-local 的 query/key seq lengths。
3. 继续调用同一个 indexer kernel。

之前的 Stage 7 代码在拿到 global compact `indexer_k` 后，走了
`Indexer.sharded_cp_topk()` 的 PyTorch/einsum chunked 实现。这个实现虽然比逐
token Python loop 快，但仍然会构造 `[chunk, heads, prefix]` 临时 logits，且没有复用
已有 `SparseAttnIndexer` 的 native logits/top-k kernel，所以长 prefill 性能不可能接近
论文里的 Indexer 收益。

### 根因

GPU 这边不能简单把 all-gather 后的 global compact K 直接塞进原
`SparseAttnIndexer` 调用并结束，因为原 op 的前半段会做：

```python
slot_mapping = attn_metadata.slot_mapping
k = k[:slot_mapping.shape[0]]
ops.indexer_k_quant_and_cache(k, kv_cache, slot_mapping, ...)
```

Sharded-CP 下 `slot_mapping` 是本 rank 的本地 token rows。若传入 global compact K，
这里会把 K 截断成本地长度，并只写本地 slot；rank1/rank2 处理同一长请求的后半段时，
原 paged indexer cache 仍然看不到前面 rank 的 K。也就是说，**直接调旧 op 会保留
本地 paged-cache 语义，不等价于 Ascend PR 的 global K 输入。**

之前的 chunked einsum 绕开了这个问题，但代价是放弃原 kernel，性能路径错误。

### 修复方式

新增 `DeepseekV32IndexerMetadata.k_is_global_compact`：

- 普通路径默认为 `False`，保持原 paged indexer cache 语义。
- Sharded-CP metadata localizer 设置为 `True`。

在 `SparseAttnIndexer` custom op 内新增 global compact K 分支：

1. 不再用本地 `slot_mapping` 截断 K，也不再写 paged indexer cache。
2. 对 all-gather 后的 global compact K 执行原有 FP8 per-token group quant。
3. 继续复用原 `fp8_mqa_logits` / `fp8_mqa_logits_torch`。
4. 继续复用原 `_C.top_k_per_row_prefill`。
5. `top_k_per_row_prefill` 输出的是 request-local offset，因此再加回每行
   `cu_seqlen_ks`，得到 FlashMLA/FlashInfer sparse backend 需要的 global compact
   KV row offset。

同时，CP-local Indexer prefill metadata 的 row spans 改为 global compact K 的绝对范围：

```text
cu_seqlen_ks[row] = request_global_start
cu_seqlen_ke[row] = token_global_offset + 1
```

这样 split request 的 rank1/rank2 可以合法选择前面 rank 的 K，且 causal prefix 仍然
由同一个 top-k kernel 的 row start/end 控制。

`MultiHeadLatentAttentionWrapper` 侧改为调用：

```python
self.indexer.forward_global_compact(
    hidden_states,
    q_fp8,
    indexer_k_global,
    indexer_weights,
)
```

生产路径上的 `Indexer.sharded_cp_topk()` 已移除，避免后续误回退到 PyTorch/einsum。

### 预期效果

长 prefill 的 top-k 计算不再受 Python/einsum chunk 临时张量和调度开销限制。
单请求 8K prompt 在 CP=TP=8 下，Q rows 应为本 rank 的约 `T/8`，K 为 all-gather 后的
global compact rows，top-k logits/top-k kernel 复用原 `SparseAttnIndexer` 性能路径。

这一步仍需要 GPU 侧验证：

1. logits parity：flag off/on 对同一 prompt 的 prefill logits 接近。
2. profiler：确认没有 `Indexer.sharded_cp_topk`/einsum 热点，看到 `fp8_mqa_logits`
   和 `top_k_per_row_prefill`。
3. benchmark：对比 flag off/on 的 TTFT，确认 Indexer 计算不再是 44s 级别的瓶颈。

本次本地验证：

```text
PYTHONPYCACHEPREFIX=/private/tmp/vllm_pycache python3 -m py_compile \
  vllm/model_executor/layers/sparse_attn_indexer.py \
  vllm/model_executor/layers/mla.py \
  vllm/model_executor/models/deepseek_v2.py \
  vllm/v1/attention/backends/mla/indexer.py \
  vllm/v1/worker/sharded_cp_metadata.py \
  tests/v1/worker/test_sharded_cp_attention.py \
  tests/v1/worker/test_sharded_cp_metadata.py

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
  tests/model_executor/test_enabled_custom_ops.py::test_empty_batch_rms_norm_and_silu_and_mul \
  -q

131 passed, 16 warnings
```

## 12. Warmup decode/mixed batch 误进 global compact KV

### 日志现象

GPU 启动在 `compile_or_warm_up_model -> _dummy_run` 阶段失败：

```text
MultiHeadLatentAttentionWrapper.forward
  -> self.indexer.forward_global_compact(...)
  -> sparse_attn_indexer._fill_topk_from_global_compact_k
  -> RuntimeError: Sharded-CP sparse indexer only supports prefill with global compact Indexer-K.
```

这个 batch 来自 warmup/dummy run，不是用户真实请求。但它构造的是合法的
decode 或 mixed decode/prefill metadata，因此不能通过跳过 warmup 或改 dummy
shape 解决；真实在线 decode 也会走同类 metadata。

### 根因

Stage 7 把 Sharded-CP sparse MLA 的快速路径设置成 global compact KV：

1. 每个 rank 计算本地 `kv_c_normed/k_pe/indexer_k`。
2. all-gather 得到本次 batch 的 global compact KV。
3. top-k 复用 `SparseAttnIndexer` 的 logits/top-k kernel。
4. attention backend 直接读取 compact KV row offset。

这个路径只对“首轮纯 prefill”成立。原因是 global compact KV 只包含当前 batch
token rows：

- decode 需要读历史 paged KV cache。
- mixed decode/prefill 中 decode 部分需要 paged KV cache。
- chunked/extend prefill 虽然 `num_decodes == 0`，但已有 context tokens 不在当前
  compact KV payload 中，也必须读 paged KV cache。

旧判定只看 `num_decodes == 0`，所以 warmup 的 mixed/extend 形状可能误入 global
compact 分支。更深一层的问题是：global compact attention 如果完全跳过 KV cache
update，首 token 可以算出来，但后续 decode 会读不到 prompt KV；Indexer K cache
也有同样问题。

### 修复

这次修复没有跳过 profile/warmup，也没有让 dummy run 走特殊分支，而是把模式边界
收紧并补齐 cache 写入：

1. `sharded_cp_forward_context()` 只在满足以下条件时启用
   `sharded_cp_use_global_compact_kv=True`：
   - DeepSeek V3.2 indexer metadata 中 `num_decodes == 0`。
   - 每个 request 的 `seq_len == query_len`，也就是没有历史 computed/context
     tokens。
2. decode、mixed、chunked/extend prefill 都走 paged KV/indexer 路径：
   - `k_is_global_compact=False`
   - `topk_indices_are_global_compact_offsets=False`
   - Indexer prefill chunk 使用 paged workspace-local `cu_seqlen_ks/ke`
3. global compact 首轮 prefill 仍然会写本地 paged cache：
   - MLA wrapper 在 compact KV all-gather 前，用本地 `kv_c_normed/k_pe` 和本地
     `slot_mapping` 调 `do_kv_cache_update()`。
   - Indexer 在 compact K all-gather 前，用本地 `indexer_k` 和本地
     `slot_mapping` 调 `indexer_k_quant_and_cache()`。
   - 本次 attention/top-k 继续使用 global compact KV，以保留 Stage 7 的性能路径。
4. `SparseAttnIndexer` global compact 分支只负责用 global compact K 计算 prefill
   top-k，不再尝试用 global K 写 paged cache，避免高 rank 把 rank0 的 K 写到本地
   slot。
5. 非首轮 paged FP8 FlashMLA sparse metadata 目前 fail closed。默认 `auto/bfloat16`
   KV cache 可走 paged path；如果显式 `fp8_ds_mla` 且 batch 不是首轮纯 prefill，
   需要补完整本地 FP8 scheduler/workspace metadata 后再放开。

### 预期效果

启动 warmup 不应再因为 decode/mixed metadata 进入 global compact indexer 而失败。
真实首轮纯 prefill 仍走：

```text
local projection -> async compact KV AG -> native SparseAttnIndexer top-k
-> FlashMLA sparse global compact attention
```

后续 decode 或 extend 则走 paged KV/indexer，保证能读到前面写入的 prompt KV。
如果服务端日志出现 “paged FlashMLA sparse metadata does not support fp8_ds_mla”
一类错误，说明当前运行实际启用了 FP8 KV cache 且进入非首轮 paged batch；那不是
同一个 global compact warmup bug，而是需要继续实现 paged FP8 metadata localizer。

本次本地验证：

```text
PYTHONPYCACHEPREFIX=/private/tmp/vllm_pycache python3 -m py_compile \
  vllm/model_executor/layers/mla.py \
  vllm/model_executor/layers/sparse_attn_indexer.py \
  vllm/model_executor/models/deepseek_v2.py \
  vllm/v1/worker/sharded_cp_metadata.py \
  tests/v1/worker/test_sharded_cp_attention.py \
  tests/v1/worker/test_sharded_cp_metadata.py

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
  tests/model_executor/test_enabled_custom_ops.py::test_empty_batch_rms_norm_and_silu_and_mul \
  -q

136 passed, 16 warnings
```

## 13. DeepGEMM 缺失时 Indexer PyTorch fallback 分配全量 logits OOM

### 日志现象

打开 DSACP 后，服务启动已经进入真实 profile/warmup forward，但在
`SparseAttnIndexer` prefill top-k 处 OOM：

```text
deepseek_v2.py:965
  return self.indexer_op(hidden_states, q_fp8, k, weights)
sparse_attn_indexer.py:229
  logits = fp8_mqa_logits_torch(...)
deep_gemm.py:479
  logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
torch.OutOfMemoryError: Tried to allocate 25.00 GiB
```

同一份日志里还能看到：

```text
DeepGEMM is not available. Disabling CUDA graph support for sparse attention indexer.
```

### 根因

这不是 Sharded Linear 权重多 load 4G 直接导致的 OOM。日志里的
`Model loading took 84.11 GiB memory` 是加载后 CUDA 占用统计，DSACP 打开后
可能包含 full-head sparse MLA/Indexer buffer、ShardLinear prefetch 相关结构等
额外常驻或初始化 buffer；但真正触发异常的是后续 `fp8_mqa_logits_torch` 的
25GiB 临时 logits。

`fp8_mqa_logits_torch()` 是 DeepGEMM 不可用时的 reference fallback。它一次性
materialize：

```text
score: [num_heads, num_query_rows, seq_len_kv] float32
logits: [num_query_rows, seq_len_kv] float32
```

在 DeepSeek V3.2 的长 prefill/profile 形状下，`num_heads` 和 `seq_len_kv`
都很大，哪怕 DSACP 已经把 query rows 切到本 rank，本 fallback 仍会产生十几到
几十 GiB 的临时张量。不开 DSACP 时 profile/warmup 不一定真实打满这条 Indexer
prefill 路径；打开 DSACP 后我们不再跳过 profile/warmup，并强制构造真实
per-layer metadata，因此把 DeepGEMM 缺失的问题暴露出来。

### 修复

这次修复没有跳过 Indexer，也没有跳过 profile/warmup，而是把 fallback 的内存峰值
限制住：

1. 新增 `_fill_prefill_topk_from_mqa_logits_torch()`。
2. DeepGEMM 不可用时，不再对整个 chunk 一次性调用
   `fp8_mqa_logits_torch()`。
3. 按 query row chunk 计算 logits，每个小 chunk 立刻调用原
   `top_k_per_row_prefill`，写入 `topk_indices_buffer`。
4. 单个 fallback score tensor 默认限制在约 64MiB：

```text
chunk_rows = 64MiB / (num_heads * seq_len_kv * sizeof(float32))
```

5. DeepGEMM 可用时仍走原 `fp8_mqa_logits` + `top_k_per_row_prefill` 路径，不影响
   目标性能路径。
6. paged prefill 和 global compact prefill 两条路径都接入了分块 fallback。
   global compact 路径仍在 top-k 后加回 `cu_seqlen_ks`，保持输出为 global
   compact KV offset；paged 路径保持 request-local offset 语义。

### 预期效果

在没有 DeepGEMM 的 L20X 环境上，DSACP profile/warmup 不应再因为
`fp8_mqa_logits_torch()` 一次性分配 25GiB logits 而 OOM。代价是 fallback 会变慢，
所以这不是论文收益路径；真正性能验证仍应安装/启用 DeepGEMM，让 Indexer 走
原生 `fp8_mqa_logits` kernel。

如果下一轮日志仍然 OOM，需要区分：

1. 是否仍在 `fp8_mqa_logits_torch` 内分配大 tensor。如果是，检查分块 fallback 是否
   已同步到 GPU 环境。
2. 是否在 decode fallback `fp8_paged_mqa_logits_torch` 分配
   `[B * next_n, max_model_len]` logits。如果是，需要对 decode fallback 做同样的
   query/request 分块。
3. 是否在模型加载后已经只剩很少 free memory。那是常驻显存预算问题，需要继续看
   DSACP 权重/buffer 生命周期。

本次本地验证：

```text
PYTHONPYCACHEPREFIX=/private/tmp/vllm_pycache python3 -m py_compile \
  vllm/model_executor/layers/sparse_attn_indexer.py \
  tests/v1/worker/test_sharded_cp_attention.py

.venv/bin/python -m pytest tests/v1/worker/test_sharded_cp_attention.py -q
21 passed, 16 warnings

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
  tests/model_executor/test_enabled_custom_ops.py::test_empty_batch_rms_norm_and_silu_and_mul \
  -q
138 passed, 16 warnings
```

## 15. profile run paged prefill Indexer 触发 DeepGEMM cu_seqlen 断言

### 日志现象

服务启动 profile run 阶段失败：

```text
gpu_worker.py:388
  self.model_runner.profile_run()
gpu_model_runner.py:5577
  hidden_states, last_hidden_states = self._dummy_run(...)
mla.py:323
  _topk_indices = self.indexer(...)
sparse_attn_indexer.py:322
  logits = fp8_mqa_logits(...)
deep_gemm.py:288
  return _fp8_mqa_logits_impl(...)
RuntimeError: Assertion error (csrc/apis/attention.hpp:91):
  cu_seq_len_k_start.size(0) == seq_len
```

这次失败发生在 `k_is_global_compact=False` 的 paged prefill Indexer 分支，不是
global compact KV all-gather 分支。

### 根因

DeepGEMM `fp8_mqa_logits()` 的输入约束是：

```text
q.shape[0] == weights.shape[0] == cu_seqlen_ks.shape[0] == cu_seqlen_ke.shape[0]
```

DSACP profile run 的 dummy batch 不是单请求：它会把 `max_num_batched_tokens`
拆到多个 dummy request 上，但每个 request 的 `seq_lens` 可能被设置成很大的
profile context length。paged prefill Indexer 会先把这些 request 的 paged
Indexer-K gather 成一个 flattened workspace，再调用 `fp8_mqa_logits()`。

这里有两个问题：

1. localized prefill chunk 如果包含不连续的 CP-local fragments，用
   `fragments[-1].local_end - fragments[0].local_start` 作为 chunk token span 会比
   实际 cu rows 更大，最终 q/weights slice 行数和 cu rows 不一致。
2. mixed decode/prefill 或 paged extend prefill 下，prefill chunk 可能不是从
   CP-local token 0 开始。例如本地 token rows 中前 2 行是 decode，prefill chunk 是
   `q[2:6]`。旧代码用 `_local_query_start_loc_cpu_from_fragments()` 生成
   `[0, 6]`，导致 `cu_seqlen_ks/ke` 有 6 行；但实际传给 DeepGEMM 的 q/weights
   只有 4 行，因此触发 `cu_seq_len_k_start.size(0) == seq_len`。
3. 即使行数一致，profile/extend paged prefill 的 flattened KV rows 可能非常大。
   原 DeepGEMM 路径一次性 materialize `[num_query_rows, total_seq_lens]` float32
   logits，和之前 PyTorch fallback OOM 是同一类内存峰值问题，只是换成了
   DeepGEMM 原生 kernel。

### 修复

1. `SparseAttnIndexer` 新增统一的 prefill top-k helper：
   - 先校验 `q/weights/cu/topk` 行数一致，不再让 DeepGEMM C++ assert 才暴露。
   - DeepGEMM 可用时也按 query rows 分块调用 `fp8_mqa_logits()`。
   - 每个分块立即调用原 `top_k_per_row_prefill` 写回 top-k，避免保留全量 logits。
   - DeepGEMM 路径按 materialized logits 大小限流，PyTorch fallback 仍按
     `[heads, rows, kv]` score 大小限流。
2. paged prefill 和 global compact prefill 都走同一个 helper：
   - paged path 保持 request-local top-k offset。
   - global compact path 继续在 top-k 后加回 `cu_seqlen_ks`，保持 global compact
     KV offset 语义。
3. Sharded-CP Indexer metadata localizer：
   - 构造 prefill chunk 前先把 CP-local fragments 按本地 token 连续性分组。
   - `_build_indexer_prefill_chunk()` 强制检查
     `token_end - token_start == cu_seqlen_ks.numel()`。
   - chunk 内 `query_start_loc` 改为相对 `chunk.token_start` 计数，保证
     `q[chunk.token_start:chunk.token_end]` 与 `cu_seqlen_ks/ke` 行数一致。

### 预期效果

profile run 不应再在 paged prefill Indexer 的 DeepGEMM `cu_seq_len_k_start`
断言处失败；即使 dummy profile 形态产生很大的 flattened KV workspace，也只会按
query-row 分块 materialize logits。

这不是跳过 profile/warmup，也不是关闭 Indexer：paged prefill 仍会更新 paged
Indexer-K cache、gather workspace、调用 DeepGEMM/top-k kernel，只是把一次巨型
logits 改成多个有界 logits 分块。

如果下一轮仍失败：

1. 若 Python 报 `SparseAttnIndexer prefill metadata row mismatch`，说明还有某条
   metadata localize 路径没有保持 q/weights/cu/topk 同一行空间。
2. 若变成 OOM，检查日志里分配位置是否仍在 `fp8_mqa_logits` 输出 logits；需要按
   GPU 实际 free memory 调小 `_DEEP_GEMM_MAX_LOGITS_BYTES`。
3. 若启动通过但 TTFT 仍慢，profile 里应该区分 pure first-prefill global compact
   路径和 extend/paged profile 路径；这次修的是启动/profile paged path 的正确性和
   内存峰值，不代表最终 real prefill 收益已经验证。

本次本地验证：

```text
PYTHONPYCACHEPREFIX=/private/tmp/vllm_pycache .venv/bin/python -m py_compile \
  vllm/model_executor/layers/sparse_attn_indexer.py \
  vllm/v1/worker/sharded_cp_metadata.py \
  tests/v1/worker/test_sharded_cp_attention.py \
  tests/v1/worker/test_sharded_cp_metadata.py

.venv/bin/python -m pytest \
  tests/v1/worker/test_sharded_cp_attention.py::test_sparse_indexer_deep_gemm_chunks_prefill_topk \
  tests/v1/worker/test_sharded_cp_attention.py::test_sparse_indexer_torch_fallback_chunks_prefill_topk \
  tests/v1/worker/test_sharded_cp_metadata.py::test_sharded_cp_forward_context_keeps_extend_prefill_on_paged_kv \
  tests/v1/worker/test_sharded_cp_metadata.py::test_localize_indexer_paged_profile_chunks_keep_row_counts_aligned \
  -q

4 passed, 16 warnings
```

## 16. PD 不分离时 decode 精度错误

### 现象

`--enable-sharded-context-parallel` 下，服务能启动，prefill 能跑通，但 PD 不分离
部署时后续 decode 精度不对。

### 当前 decode 路径

DSACP 当前只在 pure first-prefill 使用 global compact KV：

```text
num_decodes == 0 且 seq_lens == query_lens
```

只要 batch 里有 decode，或者是 extend/chunked prefill，就会关闭
`sharded_cp_use_global_compact_kv`：

```text
decode / mixed / extend -> paged KV + paged Indexer-K
```

decode 本身仍然会按 token range 做本地计算，最后 logits 前 all-gather 回完整
token rows。因此 decode 正确性的前提是：负责该 decode token 的 rank 上，paged
MLA KV cache 和 paged Indexer-K cache 已经包含完整历史 prompt。

### 根因

Stage 7 的 pure first-prefill 为了性能走了：

```text
local KV/Indexer-K -> compact KV all-gather -> global compact attention/top-k
```

这能保证当前 prefill attention 用到完整 K/V。但旧代码在 all-gather 前只把本 rank
的 local KV/Indexer-K 写入本地 paged cache：

```text
rank0 cache: prompt shard 0
rank1 cache: prompt shard 1
...
```

PD 分离时，decode worker 可能通过外部 KV 传输/重建拿到完整 KV，因此这个问题不一定
暴露。PD 不分离时，同一个 engine 后续 decode 会直接读本地 paged cache；如果负责
decode token 的 rank 只保存了 prompt 的 1/CP shard，decode attention/top-k 读到的
历史 KV 就不完整，精度必然不对。

### 修复

1. `sharded_cp_forward_context()` 在 localize metadata 前，把原始全局
   `forward_context.slot_mapping` 保存到
   `additional_kwargs["sharded_cp_global_slot_mapping"]`。
2. `MultiHeadLatentAttentionWrapper` 不再在 compact KV all-gather 前写 local cache。
3. compact KV all-gather 完成后：
   - 用全量 `kv_c_normed/k_pe` 和全局 `layer.attn` slot mapping 更新 MLA paged KV
     cache。
   - 用全量 `indexer_k_global` 和全局 `layer.indexer.k_cache` slot mapping 更新
     Indexer-K paged cache。
4. 对 cudagraph/padded 形态，global slot mapping 按实际 gathered KV rows 截断，
   避免 padded slot 进入 cache update。

### 预期效果

pure first-prefill 仍然走 global compact attention/top-k 性能路径；但 prefill 结束后，
每个 rank 的 paged MLA KV cache 和 Indexer-K cache 都有完整 prompt。PD 不分离时，
后续 decode 继续走 paged KV/indexer，应能读到完整历史，从而恢复 decode 精度。

如果 GPU 侧仍有 decode 精度问题，优先检查：

1. 首轮 prefill 后每层是否有一次 full-row KV cache update，slot mapping 行数是否等于
   prompt token 数。
2. Indexer-K cache 是否同样写入完整 prompt；只修 MLA KV 不够，top-k 会错。
3. decode forward context 中 `sharded_cp_use_global_compact_kv` 是否为 `False`。
4. FlashMLA sparse 的 `topk_indices_are_global_compact_offsets` 是否为 `False`，
   即 decode/paged path 使用 paged slot/workspace 语义。

本次本地验证：

```text
PYTHONPYCACHEPREFIX=/private/tmp/vllm_pycache .venv/bin/python -m py_compile \
  vllm/model_executor/layers/mla.py \
  vllm/model_executor/models/deepseek_v2.py \
  vllm/v1/worker/sharded_cp_metadata.py \
  tests/v1/worker/test_sharded_cp_attention.py

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
  tests/model_executor/test_enabled_custom_ops.py::test_empty_batch_rms_norm_and_silu_and_mul \
  -q

140 passed, 16 warnings
```

## 14. `CUDA_LAUNCH_BLOCKING=1` 下消失的 CUDA illegal memory

### 日志现象

关闭 `CUDA_LAUNCH_BLOCKING` 时偶发 CUDA illegal memory；打开
`CUDA_LAUNCH_BLOCKING=1` 后问题消失。

这种现象通常不是单纯 shape 检查能解释的。`CUDA_LAUNCH_BLOCKING=1` 会强制 kernel
和通信同步，常见会掩盖：

1. tensor 在非默认 CUDA stream 上仍被 NCCL/custom kernel 使用，但 allocator 不知道。
2. async collective 输入或输出 buffer 被提前释放/复用。
3. compute stream 和 communication stream 缺少明确依赖。

### 根因

DSACP Stage 7 引入了两类异步路径：

1. compact KV / Indexer-K token rows 的 async all-gather。
2. ShardLinear full weight 的 async broadcast prefetch。

这些路径都在独立 CUDA stream 上发起通信。之前代码虽然用 Python handle 持有了
input/output tensor，并在 `wait()` 里同步了 stream，但没有对这些 tensor 调用
`record_stream()`。这会让 PyTorch CUDA caching allocator 只知道 tensor 在创建
stream 上的生命周期，不一定知道它还在通信 stream 上被 NCCL 使用。默认异步执行
时，如果后续分配复用了相关 block，就可能表现成 illegal memory；开启
`CUDA_LAUNCH_BLOCKING=1` 后通信和 kernel 被同步，复用窗口消失，因此问题被掩盖。

同时这轮回归还发现 `_use_global_compact_kv_for_sharded_cp()` 又退化成只判断
`num_decodes == 0`。这会把 extend prefill 误判为首轮 pure prefill，导致需要历史
paged KV 的请求走 global compact KV。这类错误更容易表现成 decode 精度问题，也会
增加非法访问风险。

### 修复

1. `all_gather_token_rows_async()`：
   - 通信 stream 显式等待当前 compute stream。
   - 对 padded input 和 gathered output 调用 `record_stream(stream)`。
   - handle 仍持有 input/output，`wait()` 后再 assemble。
2. `ShardedCPShardLinearLayer._broadcast_async()`：
   - broadcast stream 显式等待当前 stream。
   - 对被 broadcast 的 parameter tensor 调用 `record_stream(stream)`。
   - materialized scope 退出前仍会 wait/sync，再释放 non-owner full weight。
3. `_use_global_compact_kv_for_sharded_cp()`：
   - 只有 `num_decodes == 0` 且 `seq_lens == query_lens` 的纯首轮 prefill 才启用
     global compact KV。
   - decode、mixed、extend/chunked prefill 都回到 paged KV/indexer 路径。

### 预期效果

关闭 `CUDA_LAUNCH_BLOCKING` 时，async all-gather 和 ShardLinear prefetch 的 buffer
不会再被 allocator 提前复用；extend prefill 也不会误入 global compact KV。若后续
仍出现 illegal memory，应优先看第一条同步栈：

1. 如果落在 compact KV all-gather 或 ShardLinear broadcast，继续检查对应 tensor
   是否还有没 `record_stream()` 的新通信路径。
2. 如果落在 sparse attention backend，需要看 top-k indices 是否对应该 backend
   期望的 global compact offset 或 paged global slot。
3. 如果只在 decode 出现精度问题，优先检查该 batch 的
   `sharded_cp_use_global_compact_kv` 是否为 False，以及 indexer/FlashMLA metadata
   是否都处于 paged 语义。

本次本地验证：

```text
PYTHONPYCACHEPREFIX=/private/tmp/vllm_pycache python3 -m py_compile \
  vllm/v1/worker/sharded_cp_utils.py \
  vllm/v1/worker/sharded_cp_shard_linear.py \
  vllm/v1/worker/sharded_cp_metadata.py

.venv/bin/python -m pytest \
  tests/v1/worker/test_sharded_cp_metadata.py::test_sharded_cp_forward_context_keeps_extend_prefill_on_paged_kv \
  tests/v1/worker/test_sharded_cp_metadata.py::test_sharded_cp_forward_context_marks_pure_prefill_global_compact \
  tests/v1/worker/test_sharded_cp_metadata.py::test_sharded_cp_forward_context_keeps_decode_on_paged_kv \
  -q
3 passed, 16 warnings

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
  tests/model_executor/test_enabled_custom_ops.py::test_empty_batch_rms_norm_and_silu_and_mul \
  -q
138 passed, 16 warnings
```
