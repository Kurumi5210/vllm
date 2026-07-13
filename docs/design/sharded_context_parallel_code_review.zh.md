# Sharded Context Parallel 代码审查报告

本报告面向后续代码修改执行者。审查范围为 Sharded-CP（DSACP）特性的全部源码改动，
**不含 `test_` 开头的单元测试文件**。审查方式为静态逐文件走查 + 跨文件数据流对照，
本地无 GPU，因此数值 parity 结论一律标注为「需 GPU 验证」。

## 审查范围

新增源文件：

- `vllm/distributed/sharded_cp_utils.py`
- `vllm/v1/attention/sharded_cp_attention.py`
- `vllm/v1/attention/backends/mla/sharded_cp_metadata.py`
- `vllm/model_executor/layers/sharded_cp_shard_linear.py`
- `vllm/model_executor/layers/fused_moe/sharded_cp_moe.py`

修改的已跟踪文件：

- `vllm/config/parallel.py`、`vllm/config/vllm.py`、`vllm/engine/arg_utils.py`
- `vllm/model_executor/layers/mla.py`
- `vllm/model_executor/layers/attention/mla_attention.py`
- `vllm/model_executor/layers/sparse_attn_indexer.py`
- `vllm/model_executor/layers/linear.py`、`layernorm.py`、`activation.py`
- `vllm/model_executor/layers/fused_moe/router/gate_linear.py`
- `vllm/model_executor/layers/vocab_parallel_embedding.py`
- `vllm/model_executor/models/deepseek_v2.py`
- `vllm/model_executor/model_loader/utils.py`
- `vllm/v1/attention/backends/mla/{flashmla_sparse,flashinfer_mla_sparse,rocm_aiter_mla_sparse,xpu_mla_sparse,indexer}.py`
- `vllm/v1/worker/gpu_model_runner.py`

## 严重度约定

- **P0 严重**：数值正确性缺陷，会产生错误输出。必须修。
- **P1 重要**：特定配置下 fail-open / 数值错误风险，或文档与实现矛盾。应修或明确确认边界。
- **P2 清理**：死代码、命名误导、冗余，无正确性影响。择机清理。

修改时请用**符号名 + 相对路径**定位（行号仅作辅助，可能已漂移）。

---

## P0-1：Sharded-CP 非首轮 prefill / decode 下 MLA `k_pe` 被 RoPE 两次

**文件**：`vllm/model_executor/layers/mla.py`
**符号**：`MultiHeadLatentAttentionWrapper.forward`

### 现象与根因

`forward` 内对 MLA 的 `k_pe` 存在两处 RoPE 调用：

1. 「Step 2」（在 `try` 块之前）：

   ```python
   if sharded_cp_token_range is not None and self.rotary_emb is not None:
       dummy_q_pe = k_pe.new_zeros((k_pe.shape[0], 1, self.qk_rope_head_dim))
       _, k_pe = self.rotary_emb(positions, dummy_q_pe, k_pe)   # RoPE #1
   ```

   条件是 `sharded_cp_token_range is not None`，即**所有** Sharded-CP batch 都会执行。

2. 「Step 6」（在 `try` 块内，计算完 `q` 之后）：

   ```python
   if self.rotary_emb is not None:
       if use_global_compact_kv:
           dummy_k_pe = q.new_zeros(...)
           q[..., self.qk_nope_head_dim:], _ = self.rotary_emb(positions, q[...], dummy_k_pe)
       else:
           q[..., self.qk_nope_head_dim:], k_pe = self.rotary_emb(positions, q[...], k_pe)  # RoPE #2
   ```

   `else` 分支再次对同一个 `k_pe` 做 RoPE。

三种运行态的实际 RoPE 次数：

| 运行态 | `sharded_cp_token_range` | `use_global_compact_kv` | k_pe RoPE 次数 | 结果 |
|---|---|---|---|---|
| 非 Sharded-CP | None | False | 1 | 正确 |
| SCP 纯首轮 prefill | 非 None | True | 1（Step2；Step6 走 dummy） | 正确 |
| **SCP decode / extend / chunked / mixed** | 非 None | **False** | **2** | **错误** |

### 可达性（已确认）

- `vllm/v1/attention/backends/mla/sharded_cp_metadata.py` 的 `sharded_cp_forward_context`
  对**每个** batch 都写入 `additional_kwargs["sharded_cp_token_range"]`，因此 decode/extend
  batch 的 `sharded_cp_token_range` 非 None。
- 同文件 `_use_global_compact_kv_for_sharded_cp`：当 `num_decodes != 0` 或
  `seq_lens != query_lens`（存在历史 context）时返回 `False`。

因此以下真实路径都会触发双重 RoPE：

- **chunked / extend prefill 的第 2 段及以后**（即便在纯 prefill worker 上也会出现，
  只要 KV cache 为 `auto/bfloat16`）；
- **PD 不分离部署下的 decode**（见 `sharded_context_parallel_debug.zh.md` 第 16 节）。

双重 RoPE 会破坏 K 的位置编码 → attention 数值错误。**这很可能就是 debug 第 16 节
「PD 不分离时 decode 精度错误」未被根治的真正原因**：该节只补了 KV cache 完整写入，
未发现此处的双重 RoPE。

### 修复方向

`k_pe` 的 RoPE 应当在整个 Sharded-CP 路径中只发生一次（Step 2 已完成）。Step 6 的分支
判据不应是 `use_global_compact_kv`，而应是「本次是否为 Sharded-CP」——只要是 SCP，
Step 6 就只对 `q` 做 RoPE、K 侧传 dummy：

```python
if self.rotary_emb is not None:
    if sharded_cp_token_range is not None:
        # k_pe 已在 Step 2 完成 RoPE，这里只对 q 做 RoPE
        dummy_k_pe = q.new_zeros((q.shape[0], 1, self.qk_rope_head_dim))
        q[..., self.qk_nope_head_dim:], _ = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim:], dummy_k_pe
        )
    else:
        q[..., self.qk_nope_head_dim:], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim:], k_pe
        )
```

修改后请一并检查：Step 2 与 Step 6 的 dummy 张量都使用 `new_zeros`（当前已是），
避免读到未初始化数据。

### 验证

- **CPU 层可加**：一个断言 RoPE 调用次数的测试——对 SCP decode/extend 形态的 batch，
  统计 wrapper 内 `rotary_emb` 对 k_pe 分量的调用次数为 1。
- **GPU 必需**：flag off vs Sharded-CP 在 **chunked prefill** 与 **decode** 两种形态下
  的 logits parity（BF16 容忍度）。修复前后对比应能观测到 decode 精度恢复。

---

## P1-1：Sharded-CP config 校验早于平台 `check_and_update_config`

**文件**：`vllm/config/vllm.py`
**符号**：`VllmConfig.__post_init__` 中对 `_validate_sharded_context_parallel_config()` 的调用点

### 现象

`_validate_sharded_context_parallel_config()` 在 `__post_init__` 中先于
`current_platform.check_and_update_config(self)` 执行。后者仍可能改写
`attention_config.backend` 与 `cudagraph_mode`。

- CUDA / DeepSeek：平台钩子不改这两个值，校验看到的是终态，**无影响**。
- XPU：平台钩子会把 `backend=None`（auto）解析为非 sparse 的 `FLASH_ATTN`，并可能降级
  cudagraph。此时存在 **fail-open**：用户用 auto backend 通过 sparse 白名单校验后，
  实际运行在非 sparse backend 上却未被拒绝。

### 修复方向

将 `_validate_sharded_context_parallel_config()` 的调用移动到
`current_platform.check_and_update_config(self)` **之后**，使其读到 backend 与
cudagraph_mode 的终态。移动后请确认原有的 cudagraph 校验语义不变。

### 验证

- CUDA 主路径行为不变（既有 config 测试应仍通过）。
- 增补：XPU + auto backend + Sharded-CP 应被拒绝（若无 XPU 环境，可用 mock 平台钩子的
  CPU 测试覆盖调用顺序）。

---

## P1-2：flashinfer sparse backend 的 FP8 + global compact 路径未做门控

**文件**：`vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py`
**符号**：`FlashInferMLASparseImpl` 的 forward（`trtllm_batch_decode_with_kv_cache_mla` 调用附近）

### 现象

`flashmla_sparse.py` 对 FP8 有明确门控：
`use_fp8_cache = (kv_cache_dtype == "fp8_ds_mla") and not topk_indices_are_global_compact_offsets`，
当 top-k 为 global compact offset 时回退到 BF16 compact 路径。

`flashinfer_mla_sparse.py` **没有等价门控**：它支持 fp8 kv dtype 且
`supports_quant_query_input = True`，会用 fp8 query + 单一 kernel 处理传入的 KV。
若 Sharded-CP 的 global compact（dense BF16 compact KV）与 fp8 flashinfer 组合，会出现
fp8 query 配 BF16 dense KV 的混用 → 数值错误或 kernel 报错。

这很可能是「flag 已铺、fp8 kernel 未接」的半成品状态。

### 修复方向（二选一）

- **首选（保守）**：在 config 校验或 backend 选择处，明确禁止 Sharded-CP 与
  flashinfer sparse + fp8 组合（fail closed，给出清晰错误）。
- **或者**：为 flashinfer 补上与 flashmla 等价的门控，使 `topk_indices_are_global_compact_offsets`
  为真时走 BF16 compact 路径。

### 验证

- 确认当前主验证 backend（flashmla sparse）不受影响。
- 新增：Sharded-CP + flashinfer + fp8 组合被拒绝，或走 BF16 compact 分支的 parity。

---

## P1-3：文档「KV 显存收益」结论与实现相反（需回写文档）

**实现文件**：`vllm/model_executor/layers/mla.py`
**符号**：`MultiHeadLatentAttentionWrapper._update_local_kv_cache_for_global_compact`、
`_update_local_indexer_k_cache_for_global_compact`

### 现象

为修复 PD 不分离 decode 精度（debug 第 16 节），当前实现在 compact KV all-gather 完成后，
用**全量** `kv_c_normed/k_pe`、`indexer_k_global` 和**全局** slot mapping 写入本 rank 的
paged MLA KV cache 与 Indexer-K cache。即**每个 CP rank 都保存完整 prompt 的 KV**。

这与设计文档矛盾：

- `sharded_context_parallel.md` / `.zh.md` 的 "Persistent Cache Ownership" 段声称
  「每个 CP rank 只计算并写入本地 `[start, end)` token rows 的 KV / Indexer K」。
- 文档由此推出的「KV 显存 / 访存收益」在当前实现下**不成立**。

这是刻意的正确性取舍（换取 PD 非分离 decode 正确），但文档未同步。

### 修复方向

这不是代码 bug，**不要改代码逻辑**。请更新设计文档，说明：在 PD 非分离 / 需本地
decode 的部署下，每个 rank 会持有完整 prompt KV，KV 显存收益不适用；仅在 PD 分离
（decode 由独立路径承担）时才可能恢复该收益。相关命名 `_update_local_kv_cache_*`
实际写入的是全局 KV，可在文档或注释中澄清「local 指目的地是本 rank cache，非数据范围」。

---

## P2 清理项（无正确性影响）

- **P2-1 死代码**：`vllm/model_executor/layers/mla.py` 中
  `self.indexer.forward_local_paged(...)` 只在 `if use_global_compact_kv:` 块内、且
  `indexer_metadata.num_decodes > 0` 时调用；而 `use_global_compact_kv=True` 恒含
  `num_decodes == 0`（见 `_use_global_compact_kv_for_sharded_cp`），故该调用不可达。
  是设计从「global compact 支持 mixed」收紧到「仅纯首轮 prefill」后残留的分支。
  建议删除该 `forward_local_paged` 调用（`Indexer.forward_local_paged` 方法本体如无其他
  调用方亦可一并移除）。

- **P2-2 冗余分支**：
  - `vllm/model_executor/models/deepseek_v2.py` `DeepseekV2MoE._forward_sharded_cp`：
    `if self.experts.is_internal_router: ... else: ...` 两个分支体完全相同，可合并。
  - 同文件 `DeepseekV2ForCausalLM.prepare_hidden_states_for_logits`：`token_range is None`
    时的 `if getattr(..., "_sharded_cp_logits_hidden_states_prepared", False): return ... ; return ...`
    两条路径都 `return hidden_states`，`if` 无意义。
  - `vllm/v1/attention/backends/mla/sharded_cp_metadata.py`
    `_use_global_compact_kv_for_sharded_cp` 结尾 `if saw_global_compact_signal: return True; return True`
    两支相同。

- **P2-3 命名误导**：`vllm/model_executor/layers/mla.py`
  `_update_local_kv_cache_for_global_compact` 实际写入全局 KV（见 P1-3）。建议改名或加注释。

- **P2-4 疑似死代码**：`vllm/distributed/sharded_cp_utils.py`
  `get_request_aligned_sharded_cp_token_ranges` / `get_request_aligned_sharded_cp_token_range`。
  Stage 7 已改用 balanced split（`get_sharded_cp_token_range` + fragment），request-aligned
  版本疑似不再被生产路径调用。请确认无调用方后删除（注意：`ShardedCPTokenRange` 的
  `rank_starts/rank_ends` 与 `has_explicit_rank_ranges` 仍被 assemble/reduce-scatter 使用，
  这些**不能**删）。

- **P2-5 性能（非正确性）**：`vllm/model_executor/layers/sharded_cp_shard_linear.py`
  `ShardedCPShardLinearPrefetch.wait` 用 `stream.synchronize()`（host 全同步），正确但削弱
  async prefetch 的 overlap。可评估改为让当前 compute stream `wait_stream(prefetch_stream)`。

- **P2-6 遗留风险（文档已记录，需 GPU 确认）**：`kv_b_proj` 在使用 weight absorption 的
  sparse MLA backend 下，运行时可能不作为独立矩阵乘存在，却仍被 `disable_tp=True` 加载并
  注册进 Shard Linear（见 `DeepseekV2MLAAttention` 构造与 `ShardedCPShardLinearLayer`）。
  纯属显存浪费，需在真实 backend 上确认后决定是否移除注册。

---

## 已确认正确 —— 请勿在修 P0/P1 时误改

以下点已逐项走查确认正确，修改上述问题时不要改动其语义：

1. **MoE reduce-scatter 非双重归约**：`DeepseekV2MoE._forward_sharded_cp` 强制
   `is_sequence_parallel=False`；此时 `self.experts`（`reduce_results=False`）返回每 rank
   partial 贡献，标准路径靠 `maybe_all_reduce_tensor_model_parallel` 合并。SCP 用
   `reduce_scatter_sharded_cp_moe_output`（对 TP=CP 组求和 + scatter）等价替换 all-reduce，
   正确。（唯一前提：`maybe_all_reduce_tensor_model_parallel` 确实执行归约——保持不变即可。）

2. **空 rank collective 一致性**：`MultiHeadLatentAttentionWrapper._forward_empty_sharded_cp`
   仅在 `use_global_compact_kv=True` 时参与 compact KV all-gather（与非空 rank 对齐），
   非 global 路径 wrapper 内无 collective；`DeepseekV2MoE._forward_sharded_cp` 空 rank 仍进入
   MoE all-gather。空 rank 不会导致 NCCL 死锁。

3. **async buffer 生命周期**：`sharded_cp_utils.all_gather_token_rows_async` 与
   `sharded_cp_shard_linear` 的 async broadcast 均已 `record_stream` + `wait_stream`
   （对应 debug 第 14 节），handle 持有输入/输出 buffer。

4. **sparse indexer 内存与行校验**：`sparse_attn_indexer._fill_prefill_topk_from_mqa_logits`
   对 DeepGEMM 与 torch fallback 均按 query-row 分块，`_validate_prefill_topk_rows` 提前校验
   行数一致（对应 debug 第 13 / 15 节）。global compact 分支不写 paged cache（由 wrapper 单独
   写一次），无双写。

5. **runner profile 生命周期**：`_temporary_sharded_cp_kv_cache` + `_cleanup_profiling_kv_cache`
   正确清理 `attn_groups / kv_cache_config / layer.kv_cache` 绑定（对应 debug 第 9 节）；
   `_select_hidden_states_for_logits` 先 all-gather 回全局再按全局 index 取样（对应 debug 第 6 节）。

6. **空 batch 短路**：`linear.py / layernorm.py / activation.py / gate_linear.py` 的 0 行
   fast-path 返回形状 / dtype / device 均正确，且不跳过任何 collective；
   `vocab_parallel_embedding.py` 的 `forward_parallel` 拆分对非 Sharded-CP 路径行为不变。

7. **loader hook**：`model_loader/utils.py` 在 `process_weights_after_loading` 末尾调用
   可选的 `post_process_weights_after_loading`，`DeepseekV2ForCausalLM` 据此在通用后处理
   之后再释放非 owner Shard Linear 权重（对应 debug 第 1 节），顺序正确。

8. **Shard Linear owner / broadcast / materialize / release 生命周期** 对称且异常安全；
   `_record_param_metadata` 捕获的是 `process_weights_after_loading` 之后的运行时 shape。

---

## 建议修复顺序

1. **P0-1**（双重 RoPE）——唯一确定的数值正确性 bug，很可能是 decode 精度问题根因。
2. **P1-1 / P1-2**（fail-open 门控）。
3. **P1-3**（回写文档，不改代码）。
4. **P2-\***（择机清理，可与上面打包，不要单独发无意义 PR）。

## 修复后必须的验证

- **CPU**：现有 Sharded-CP 聚焦回归全绿；为 P0-1 增补 RoPE 次数断言测试。
- **GPU（本地无法完成，需执行方补）**：
  - flag off vs Sharded-CP 的 **纯首轮 prefill** logits parity（回归确认 P0-1 未破坏首轮路径）。
  - flag off vs Sharded-CP 的 **chunked prefill** 与 **decode** logits parity（确认 P0-1 修复生效）。
  - profiler timeline 确认 async all-gather / Shard Linear broadcast 与计算 overlap。
