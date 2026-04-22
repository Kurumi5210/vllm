# dev-dycp-rebase-0.18 Rebase 记录

## 概览

| 项目 | 详情 |
|------|------|
| 源分支 | `dev-dycp-rebase-0.18` |
| 目标基底 | `releases/v0.18.0` (upstream) |
| 原始提交数 | 55 |
| Rebase 后提交数 | 35 |
| 自动跳过 | 14 个（已在上游的 cherry-pick） |
| 手动跳过 | 2 个（不再需要的 revert/CI 提交） |
| Merge commit 展平 | 4 个 |
| 需手动解决冲突的提交 | 8 个 |

## 跳过的提交

### 自动跳过（14 个）

Git 检测到这些提交已存在于 `releases/v0.18.0` 中，自动跳过：

- `5bd63387c` [XPU][6/N] add xpu scaled_mm kernel
- `e1e984163` [torch.compile][Fusion] Fix attention fusion pass
- `55a1baebc` [Bugfix][ROCm] Use old triton_kernels implementation
- `b2e1fc358` [Bugfix][Core] Fix CPU memory leak
- `83db96d8c` [XPU][9/N] clean up existing ipex code/doc
- `c44d0c6d6` Patch protobuf for CVE-2026-0994
- `b3ee90f96` [Model] GLM adaptation
- `9be1ff2d3` [Bugfix] fix default is_neox_style
- `5e8adb0c4` [Misc] Bump fastsafetensors version
- `946b2f106` [Bugfix] send None sentinel on final commit
- `7a06e5b05` [Bugfix] Fix MTP accuracy for GLM-5
- `89a77b108` [ROCm][CI] Pin TorchCodec to v0.10.0
- `d3c1513f5` [ci] Use the right tag for CPU arm64 image
- `2d5be1dd5` release script

### 手动跳过（2 个）

这两个提交是针对旧基底的版本调整，在 v0.18.0 上已无意义：

1. **`c86cdcbcd`** — Revert "[Release 2.10] Update to Torch 2.10"
   - 该提交试图将 torch 从 2.10 降级到 2.9.1，但 v0.18.0 已自带正确的 torch 版本
   - 涉及文件：`requirements/cuda.txt`, `requirements/rocm-build.txt`, `requirements/test.txt`, `tests/compile/test_aot_compile.py`, `vllm/envs.py`, `gpt_oss_triton_kernels_moe.py`

2. **`5dbfbc967`** — [CI/Build] Fix gRPC version mismatch
   - 上游 CI 修复，v0.18.0 已包含对应修复
   - 涉及文件：`requirements/rocm.txt`, `requirements/test.in`, `requirements/test.txt`

## 冲突解决详情

### 1. `cda20f533` — [feat] rebase npu model runner & cuda dispatch

冲突文件：`forward_context.py`, `cudagraph_dispatcher.py`, `gpu_model_runner.py`

**冲突根因**：v0.18.0 对 cudagraph 调度做了大幅重构（引入 `valid_modes`/`invalid_modes` 替代 `disable_full`，移除 `relax_for_mixed_batch_cudagraphs` 改用内联 `replace()`，新增 `_warmup_and_capture` 方法），而 DYCP 在旧架构基础上添加了 `num_dycp_reqs` 参数。

**解决策略**：以 v0.18.0 的新架构为骨架，将 DYCP 的 `num_dycp_reqs` 集成进去。

具体处理：

- **`forward_context.py`**：`num_dycp_reqs` 字段已在两侧都存在（无冲突）。DYCP 侧新增的 `relax_for_mixed_batch_cudagraphs()` 方法不保留——v0.18.0 已移除该方法，改用 `replace(batch_desc, num_reqs=None, uniform=False)` 内联调用。

- **`cudagraph_dispatcher.py`**（6 处冲突）：
  - `initialize_cudagraph_keys`：保留 HEAD 的 assertion 和 PIECEWISE relaxation 逻辑，同时将 `dycp_reqs` 加入 `product()` 迭代
  - `dispatch` 方法签名：保留 HEAD 的 `valid_modes`/`invalid_modes` 参数，追加 `num_dycp_reqs` 参数
  - 早退条件：同时保留 `allowed_modes <= {CUDAGraphMode.NONE}` 和 `num_dycp_reqs > self.max_dycp_reqs`
  - `_create_padded_batch_descriptor` 调用：使用 HEAD 的 `normalized_uniform` + DYCP 的 `num_dycp_reqs`
  - 返回值：保留 HEAD 的 `assert NONE in allowed_modes`，返回的 `BatchDescriptor` 包含 `num_dycp_reqs`
  - 排序 key：合并为 `(num_tokens, num_active_loras, num_dycp_reqs)`

- **`gpu_model_runner.py`**（4 处冲突）：
  - `_reorder_batch` 后追加 DYCP 的 CP reorder 逻辑，然后保留 HEAD 的 `_init_kv_zero_meta`/`_zero_block_ids`
  - `dispatch_cudagraph` 调用：保留 HEAD 的 `valid_modes`/`invalid_modes`，追加 `num_dycp_reqs`
  - `_dummy_run` 签名：同时保留 `profile_seq_lens` 和 `num_dycp_reqs`
  - cudagraph capture 循环：保留 HEAD 的 `_warmup_and_capture` 方法，并更新该方法内部的 `_dummy_run` 调用以传递 `desc.num_dycp_reqs`

### 2. `8612b3c1e` — Add: engine/core & engine/util

冲突文件：`vllm/v1/engine/core.py`

**冲突根因**：HEAD 有 `_eep_scale_up_before_kv_init` 和 `EngineShutdownState`，DYCP 添加了 domain 相关方法（`log_domain_error_detail`, `step_domain`, `run_domain_engine_core` 等）。

**解决策略**：全部保留。在 `EngineCore` 类中按顺序放置：EEP 方法 → domain 方法。`EngineShutdownState` 枚举保持在 `EngineCore` 类之后、`EngineCoreProc` 类之前。

### 3. `5d6d1c743` — [feat] add some dycp config

冲突文件：`vllm/config/parallel.py`, `vllm/distributed/kv_transfer/kv_connector/factory.py`

- **`parallel.py`**：HEAD 新增了 elastic EP 的端口分配方法和带 `return_store` 重载的 `stateless_init_dp_group`。DYCP 新增了 `stateless_init_domain_group`。解决方式：保留 elastic EP 方法和 overload 签名，在 overload 声明和实现之间插入 `stateless_init_domain_group`。

- **`factory.py`**：HEAD 注册 `FlexKVConnectorV1`，DYCP 注册 `CrossDPExampleConnector`。解决方式：两个注册都保留，各自独立。

### 4. `c4a997d40` — Add: core & engine & executor

冲突文件：`output.py`, `coordinator.py`, `core.py`, `core_client.py`, `multiproc_executor.py`

- **`output.py`**：HEAD 新增 `new_block_ids_to_zero`，DYCP 新增 `cp_rank`/`num_cp_request`/`none_tokens_in_peer_sched`。全部保留。

- **`coordinator.py`**：`local_only_eng` 的计算方式不同。保留 HEAD 的写法（使用 `dp_size == data_parallel_size_local` 和 elastic EP 检查），因为语义更完整。

- **`core.py`**：之前的 rebase 步骤已将 domain 方法合入 HEAD 侧，此处 DYCP 侧为空。直接保留 HEAD，删除冲突标记。

- **`core_client.py`**：HEAD 在 `launch_core_engines` 前先创建 zmq socket 并传入 `addresses`。DYCP 增加了 domain 引擎条件分支。合并方式：保留 HEAD 的 socket 创建逻辑，将 launch 调用改为条件分支（`launch_domain_core_engines` if `dp_per_domain > 1` else `launch_core_engines`），两者都传入 `addresses` 参数。

- **`multiproc_executor.py`**：HEAD 导入 `model_parallel_is_initialized`，DYCP 导入 `get_dycp_group`。两个都保留。

### 5. `476d4cef6` — Add: arg_utils

冲突文件：`vllm/engine/arg_utils.py`

HEAD 新增 `fail_on_environ_validation` 和 `gdn_prefill_backend`，DYCP 新增 `dp_per_domain` 和 `num_cp_seqs`。全部保留。

### 6. `d642b8c96` — Add: entrypoint/cli/serve

冲突文件：`vllm/entrypoints/cli/serve.py`

- 导入：HEAD 只导入 `launch_core_engines`，DYCP 额外导入 `EngineCoreProc` 和 `DomainCoreEngineProcManager`。合并为全部导入。
- `run_headless`：HEAD 使用新签名的 `CoreEngineProcManager`（无 `target_fn`），DYCP 增加了 domain 条件分支（带 `target_fn`）。保留 DYCP 的条件分支，非 domain 路径使用 HEAD 的新签名。

### 7. `99becf647` — [fix] fix dp_domain_engine_core error

冲突文件：`vllm/entrypoints/cli/serve.py`

HEAD 在 `launch_core_engines` 前新增了 `get_engine_zmq_addresses` 调用并传入 `addresses`。DYCP 增加了 domain 条件分支但未传 `addresses`。合并方式：保留 HEAD 的 addresses 预创建，条件分支中两个 launch 函数都传入 `addresses`。

### 8. `ef6418b03` — [feat] add worker fix

冲突文件：`vllm/v1/worker/gpu_worker.py`

HEAD 的 `execute_model` 接收单个 `SchedulerOutput` 并新增了 PP send work 清理逻辑。DYCP 改为接收 `list[SchedulerOutput | None]` 并增加 CP rank 调度。合并方式：签名改为 `SchedulerOutput | list[SchedulerOutput | None]`，先执行 PP send 清理，再做 list/单值分发。

## 注意事项

- Rebase 后所有提交已重写 hash，如需推送到远程需要 force push
- `launch_domain_core_engines` 函数签名已调整为接受 `addresses` 参数，需确认该函数实现已同步更新
- `_warmup_and_capture` 方法已更新为传递 `num_dycp_reqs`，确保 dummy run 时 CP 状态一致
