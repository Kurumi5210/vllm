# Sharded Context Parallel Stage 1 总结

## 阶段目标

Stage 1 只完成 Sharded Context Parallel 的入口、配置 fail-closed 校验，以及后续模型路径会复用的 token-row 分片/collective 辅助函数。这个阶段不改 DeepSeek/DSA sparse MLA forward，不改 scheduler，不接入实际分布式执行路径。

本阶段的可独立验证功能是：

1. 用户可以通过 CLI/EngineArgs 打开 `enable_sharded_context_parallel`。
2. 不支持的并行拓扑、模型类型和运行模式会在配置阶段失败，而不是进入运行期后产生隐式错误。
3. token 维度的 CP 分片、padding、gather 后 trim、reduce-scatter 前切片有独立 CPU UT 覆盖。

## 代码改动

### `vllm/config/parallel.py`

新增 `ParallelConfig.enable_sharded_context_parallel: bool = False`。

这个字段是 Sharded-CP 的总开关，默认关闭，不影响现有 TP/PP/PCP/DCP/DBO 路径。打开后，`ParallelConfig._validate_parallel_config()` 会做只依赖并行拓扑的校验：

1. `tensor_parallel_size > 1`：第一版按设计复用 TP group 作为 CP group，单 TP rank 没有 sharded CP 的意义。
2. `pipeline_parallel_size == 1`：Stage 1 不声明支持 PP，避免后续隐藏状态 layout 和跨 stage 传递语义不清。
3. `prefill_context_parallel_size == 1`、`decode_context_parallel_size == 1`：Sharded-CP 与现有 PCP/DCP 是不同 layout，第一版不允许叠加。
4. `use_ubatching == False`：DBO/ubatching 会改变 batch/token 调度语义，当前阶段先 fail-closed。

这些检查放在 `ParallelConfig` 中，是因为它们只依赖并行配置本身。

### `vllm/engine/arg_utils.py`

新增 EngineArgs 字段和 CLI 参数：

```text
--enable-sharded-context-parallel
```

CLI 解析后会写入 `EngineArgs.enable_sharded_context_parallel`，并在 `create_engine_config()` 构造 `ParallelConfig` 时传入同名字段。这样用户入口、EngineArgs 中间态和最终 ParallelConfig 三层保持一致。

### `vllm/config/vllm.py`

新增 `VllmConfig._validate_sharded_context_parallel_config()`，处理跨配置约束：

1. 必须有 `model_config`。
2. `model_config.use_mla` 必须为 true。
3. `model_config.hf_config` 必须包含 `index_topk`，用于限定当前入口只面向 DSA sparse MLA 模型配置。
4. 如果用户显式指定 attention backend，则必须是 sparse MLA backend；auto 模式保留给后续 backend selector。
5. 暂不支持 `speculative_config`。
6. 暂不支持 full CUDA graph，允许 `NONE` 或 `PIECEWISE`。

该方法在 `__post_init__()` 中 platform defaults 和 cudagraph mode 默认值处理之后调用。这样校验看到的是最终 `cudagraph_mode`，不会因为用户未显式指定 compilation config 而误判。

### `vllm/v1/worker/sharded_cp_utils.py`

新增一组 layout-only 的 token-row 工具，不依赖具体模型类，可以在 CPU 上独立测试。

`ShardedCPTokenRange` 表示一个 CP rank 拥有的 token 行区间：

1. `start`/`end`：真实 token 行范围，半开区间 `[start, end)`。
2. `padded_end`：为了 collective 等长输入保留的 padded 结束位置。
3. `total_tokens`：全局真实 token 数。
4. `num_tokens`：本 rank 的真实 token 行数。
5. `padded_num_tokens`：本 rank collective 输入需要的行数。

`get_sharded_cp_token_range(num_tokens, rank, world_size)` 使用固定 chunk：

```text
chunk = ceil(num_tokens / world_size)
```

每个 rank 分到连续 token 区间；最后一个 rank 或超出真实 token 的 rank 可能真实行数少于 padded 行数，但 collective shape 仍保持一致。

`pad_for_token_all_gather(x, token_range, pad_value=0.0)` 要求 `x.shape[0] == token_range.num_tokens`，然后补齐到 `padded_num_tokens`。这会用于后续 CP-local hidden states 汇总回全 token layout。

`trim_token_all_gather(x, token_range)` 从 gather 后的 padded 全局结果中裁掉末尾 padding，只保留 `total_tokens` 行。

`all_gather_token_rows(x, token_range, group=None, pad_value=0.0)` 先 pad 本地 token 行，再执行 `dist.all_gather`，最后 trim。单 rank 且未初始化 distributed 时直接返回输入；多 rank 未初始化 distributed 时显式报错。

`slice_for_token_reduce_scatter(x, token_range, pad_value=0.0)` 从全局 token layout 中取出当前 rank 的真实 token 行，并 padding 到固定 chunk。这个函数给后续 EmbeddingTP 到 CP hidden layout 的转换做纯切片验证。

`get_sharded_cp_group()` 返回当前第一版 Sharded-CP 通信组。实现上复用 `get_tp_group()` 返回的 `GroupCoordinator`，但通过独立 helper 保留 CP 语义入口，后续如果 CP group 从 TP group 中解耦，不需要修改调用方。

## UT 覆盖

新增/调整的测试：

1. `tests/v1/worker/test_sharded_cp_utils.py`
   - 覆盖 token range 计算。
   - 覆盖非法 rank/world_size/num_tokens。
   - 覆盖 all-gather 前 padding、gather 后 trim。
   - 覆盖 reduce-scatter 前本地 chunk 切片和 padding。
   - 覆盖未初始化 distributed 时的单 rank 快路径和多 rank 报错。
   - 覆盖空 token 单 rank 快路径直接返回原始输入。
   - 覆盖 `get_sharded_cp_group()` 当前复用 TP group。

2. `tests/test_sharded_context_parallel_config.py`
   - 覆盖 `ParallelConfig` 最小可接受拓扑。
   - 覆盖 TP=1、PP、PCP、DCP、DBO/ubatching 等不兼容拓扑。
   - 覆盖 `VllmConfig` 的模型、speculative decoding、full CUDA graph 限制。
   - 覆盖模型配置缺少 `use_mla` 或 `hf_config.index_topk` 时统一 fail-closed。
   - 覆盖显式 sparse MLA backend 允许、显式 dense MLA backend 拒绝。

3. `tests/engine/test_arg_utils.py`
   - 新增 `test_enable_sharded_context_parallel_cli_arg`，验证 CLI flag 能进入 `EngineArgs`。
   - 覆盖默认值为 false。
   - 覆盖 `EngineArgs.create_engine_config()` 里 flag 进入最终 `ParallelConfig`。
   - 覆盖 CLI/EngineArgs 链路中的不兼容拓扑拒绝路径。

这些测试是纯配置/CPU 测试，已标记 `skip_global_cleanup`，避免 macOS 本地 PyTorch 在全局 cleanup 调 `torch.accelerator.empty_cache()` 时触发 MPS allocator 内部错误。

## 测试结果

通过的命令：

```bash
.venv/bin/python -m pytest tests/v1/worker/test_sharded_cp_utils.py tests/test_sharded_context_parallel_config.py tests/engine/test_arg_utils.py::test_enable_sharded_context_parallel_cli_arg tests/engine/test_arg_utils.py::test_enable_sharded_context_parallel_flows_to_parallel_config tests/engine/test_arg_utils.py::test_enable_sharded_context_parallel_rejects_incompatible_cli_topology -q
```

结果：

```text
37 passed, 3 warnings in 0.13s
```

说明：`.venv` 是按仓库要求用 `uv venv --python 3.12` 创建的，并安装了运行本阶段 UT 所需依赖。

## 本阶段未实现内容

Stage 1 没有实现以下内容：

1. DSA sparse MLA forward 中的 CP-local Q/K/V 或 attention 输出 layout。
2. EmbeddingTP 输出到 CP hidden layout 的真实 reduce-scatter 接入。
3. Transformer 层间保持 CP-local hidden states 的模型路径改造。
4. final logits 前从 CP-local hidden states 回全 token layout 的实际 all-gather 接入。
5. 多机/多进程 distributed 集成测试。
6. 与 PP、PCP、DCP、DBO、speculative decoding、full CUDA graph 的兼容。

这些都应进入后续阶段，每个阶段继续保持 fail-closed 和独立 UT。

---

## 代码审查（2026-06-30）

以下审查基于当前 git diff（4 个 tracked 文件，+266 行）和 3 个 untracked
新增文件，对比设计文档 `sharded_context_parallel.md` Commit 1 的要求。

### 审查范围

**已修改的 tracked 文件（`git diff`）：**

| 文件 | 行数 | 说明 |
|------|------|------|
| `vllm/config/parallel.py` | +35 | 字段定义 + topology 校验 |
| `vllm/config/vllm.py` | +59 | 跨配置校验 + backend 白名单 |
| `vllm/engine/arg_utils.py` | +8 | CLI flag + EngineArgs 透传 |
| `tests/engine/test_arg_utils.py` | +164 | CLI 测试（默认值/透传/拒绝） |

**untracked 新增文件：**

| 文件 | 行数 | 说明 |
|------|------|------|
| `vllm/v1/worker/sharded_cp_utils.py` | 166 | token-row 分片/collective 工具 + CP group helper |
| `tests/v1/worker/test_sharded_cp_utils.py` | 132 | 工具函数 UT |
| `tests/test_sharded_context_parallel_config.py` | 147 | config 校验 UT |

### 设计文档要求对照

| 设计要求 | 状态 | 说明 |
|----------|------|------|
| `enable_sharded_context_parallel` config 字段 | ✅ | `parallel.py:107` |
| CLI flag `--enable-sharded-context-parallel` | ✅ | `arg_utils.py:852-855` |
| topology 校验（TP/PP/PCP/DCP/ubatching） | ✅ | `parallel.py:435-460`，6 项全部覆盖 |
| model/speculative/CUDA graph 校验 | ✅ | `vllm.py:668-692`，5 项全部覆盖 |
| sparse MLA backend 显式校验 | ✅ | `vllm.py:243-249` 常量 + `vllm.py:688-698` 校验 |
| CP token range/partition/padding helpers | ✅ | `sharded_cp_utils.py` |
| CP group 复用 TP group 的定义 | ✅ | `sharded_cp_utils.py:35-41` `get_sharded_cp_group()` |
| 校验拒绝路径测试 | ✅ | 全部覆盖，见逐文件详情 |

### 逐文件审查

#### `vllm/config/parallel.py` ✅

1. **字段定义（L107-114）**：`enable_sharded_context_parallel: bool = False`，
   docstring 准确描述了复用 TP group 作为 CP group 以及互斥性。

2. **校验位置（L435-460）**：放在 `_validate_parallel_config()` 末尾，
   `return self` 之前。附近已有 DCP、all2all、EPLB 等校验，风格一致。

3. **校验覆盖**：
   - `tensor_parallel_size > 1` ✅
   - `pipeline_parallel_size == 1` ✅
   - `prefill_context_parallel_size == 1` ✅
   - `decode_context_parallel_size == 1` ✅
   - `!use_ubatching` ✅

4. **可以改进**：校验逻辑自成一块，但没有抽取独立方法。当后续校验增多时建议
   抽成 `_validate_sharded_cp_topology()`。当前规模 OK。

#### `vllm/engine/arg_utils.py` ✅

1. **字段声明**：`enable_sharded_context_parallel: bool =
   ParallelConfig.enable_sharded_context_parallel`。外层括号多余但因为
   `False` 是 immutable 无害。

2. **CLI 注册**：放在 `--prefill-context-parallel-size` 和
   `--data-parallel-size` 之间，位置合理。

3. **透传**：`ParallelConfig(..., enable_sharded_context_parallel=
   self.enable_sharded_context_parallel, ...)` ✅。

#### `vllm/config/vllm.py` ✅

1. **`SHARDED_CP_SPARSE_MLA_BACKENDS` 常量（L243-249）**：模块级
   `frozenset`，包含 4 个 sparse MLA backend enum 值。放在类外是正确的——
   避免每次 `__post_init__` 重新创建集合 ✅。

2. **校验方法（L665-709）**：`_validate_sharded_context_parallel_config()`，
   独立方法，先 early-return 非 ShardedCP 场景。校验项含：
   - `model_config is None` → reject ✅
   - `model_config.use_mla` → reject ✅
   - `hf_config.index_topk` → reject ✅
   - **attention backend 白名单（L688-698）**：
     `backend is None`（auto）→ 允许；显式非 sparse → reject ✅
   - `speculative_config is not None` → reject ✅
   - `cudagraph_mode.has_full_cudagraphs()` → reject ✅

3. **调用时机（L1054）**：在 `_set_cudagraph_sizes()` 之后，确保看到的
   `cudagraph_mode` 是最终值 ✅。

4. **`use_mla` 检查用 `getattr`**：`getattr(self.model_config, “use_mla”,
   False)`，行为正确。

#### `vllm/v1/worker/sharded_cp_utils.py` ✅

1. **`get_sharded_cp_group()`（L35-41）**：返回 `get_tp_group()` 的
   `GroupCoordinator`。通过独立 helper 保留了 CP 语义入口，后续若 CP group
   从 TP group 解耦，调用方无需改动 ✅。

2. **`all_gather_token_rows` 快路径修正（L125-126）**：将快路径条件从
   `world_size=1` 简化为了 `start == 0 and total_tokens == x.shape[0]`。
   满足该条件（单 rank、rank 0、无 padding）时直接返回原始输入，不经过
   pad/collective/trim 流程。此时 `T=0` 也会命中快路径返回原始 `x`，不受
   padding 干扰 ✅。

#### `tests/v1/worker/test_sharded_cp_utils.py` ✅

1. **`test_get_sharded_cp_token_range`**：参数化覆盖 4 种 partition ✅。
2. **非法输入**：覆盖 4 种 ✅。
3. **`all_gather_token_rows` 单 rank 快路径**：`T=2` 和 `T=0` 各一个 case ✅。
4. **`test_get_sharded_cp_group_reuses_tp_group`**：验证返回 `get_tp_group()` ✅。

#### `tests/test_sharded_context_parallel_config.py` ✅

1. **`FakeModelConfig`**：良好的测试替身 ✅。
2. **`ParallelConfig` 参数化拒绝**：覆盖 6 种不兼容拓扑 ✅。
3. **Backend 接受/拒绝**：
   - `FLASHMLA_SPARSE` 显式指定 → 接受 ✅
   - `FLASHMLA`（dense MLA）显式指定 → 拒绝 ✅
4. **speculative / CUDA graph / 模型拒绝**：有独立测试 ✅。

#### `tests/engine/test_arg_utils.py` ✅

1. **`test_enable_sharded_context_parallel_cli_arg`**：
   - 验证默认值为 `False` ✅
   - 验证 `--enable-sharded-context-parallel` 解析为 `True` ✅

2. **`test_enable_sharded_context_parallel_flows_to_parallel_config`**：
   - 通过 monkeypatch 模拟完整 `create_engine_config()` 调用链
   - 验证 flag 最终进入 `ParallelConfig` ✅

3. **`test_enable_sharded_context_parallel_rejects_incompatible_cli_topology`**：
   - CLI 层面模拟 `TP=1` + ShardedCP → 验证 `create_engine_config()` 抛出
     `ValueError(“tensor_parallel_size > 1”)` ✅

### 审查结论

| 类别 | 评估 |
|------|------|
| 代码质量 | ✅ 风格一致、错误处理清晰、docstring 规范 |
| 设计文档对齐 | ✅ 全部 8 项设计要求已实现 |
| 测试覆盖 | ✅ 工具函数、config 校验、CLI 链路均有覆盖 |
| 安全性 | ✅ 全部 fail-closed，不支持的路径显式拒绝 |

### 后续阶段建议

1. 把 untracked 的 `sharded_cp_utils.py`、`test_sharded_cp_utils.py` 和
   `test_sharded_context_parallel_config.py` 加入 git。
2. Commit 2 实现时，`get_sharded_cp_group()` 已经可用，如需扩展 group 语义
   （如正式创建独立 CP process group），只需修改这一个函数的内部实现。
