# BUGFIX 交接：Grad norm 未做 DP 去重导致拓扑相关裁剪

> 这是一份给另一台电脑上的 AI 使用的修复交接文档。目标不是重新调查现象，而是让接手者能够在干净或不同版本的 Aceso 工作区中独立确认根因、实现最小修复，并完成回归验证。

## 1. 任务摘要

- 仓库：`Aceso`
- 相关分支：`dev-cx`
- Bug：普通非 DP-sharded optimizer 计算全局梯度范数时，把 DP 副本重复计入。
- 直接后果：`grad norm` 随 DP 拓扑变化，`--clip-grad 1.0` 在不同拓扑下于不同 iteration 开始裁剪，导致恢复后的 loss 轨迹分叉。
- 影响范围：baseline、LivePipe DP、LivePipe PP、LivePipe TP 都经过同一公共 optimizer 路径，因此四种模式都应视为受影响；TP 只是现象最明显。
- 当前工作区已有一份候选修复和回归测试，但尚未通过 GitHub 推送；接手者应独立检查并只移植本 bug 所需改动，不要覆盖其他未提交工作。

给接手 AI 的任务可以直接使用：

> 请先读本仓库适用的 AGENTS.md，再按本文检查 Aceso 的 grad norm DP 重复计数问题。先确认修复是否已存在；如果缺失，按函数名定位实现，不要依赖固定行号。仅修改本 bug 所需代码并补回归测试，保留其他修复和用户未提交改动。若本文引用的原始 GPU 数据不可用，请明确区分已复核证据和既有记录，不要自行宣布验收通过。完成后列出改动文件、测试结果和剩余验证事项，不要自行 commit 或 push。

## 2. 必须达到的结果

修复后必须满足：

1. 普通 optimizer 的范数只统计每个逻辑参数一次：DP 副本只取 DP group rank 0，TP shard 按 shard 全部保留，TP replicated 参数只取 TP group rank 0。
2. DP、PP、TP 混合拓扑不能使用一个全局 `sqrt(DP)` 或 GPU 数量系数修正；每个参数必须根据自己的 logical op 查询对应的 TP/DP group。
3. 范数筛选只影响“参与 norm 计算的梯度集合”，不能导致非 norm-owner rank 不再缩放自己的本地梯度。计算出全局 clip coefficient 后，所有 rank 的所有本地梯度都要缩放。
4. Distributed optimizer 的 DP 参数通常是不同 shard，不是同一份 replica；它必须保留所有 shard 参与 norm 计算。
5. 空的本地 norm contributor 列表仍必须参加 all-reduce，并正确处理 L2、其他有限阶以及 infinity norm。
6. 修复不能破坏 tied embedding、PP/TP reconfigure、FP16/BF16 FP32 master 参数或现有恢复流程。

## 3. 现象与可复核数据

旧 correctness 实验固定使用 `--clip-grad 1.0`。旧 baseline 和旧 TP 的日志显示：

| iteration | 旧 baseline norm | 旧 TP norm |
| --- | ---: | ---: |
| 31 | 0.653 | 0.465 |
| 48 | 1.017 | 0.723 |
| 100 | 1.433 | 1.039 |

因此旧 baseline 从约第 48 步开始裁剪，而旧 TP 到约第 83 步才开始裁剪。两边从第 48 步开始不再使用同一种参数更新规则，旧 TP 的 loss 差值在后段逐渐扩大：

| iteration | 旧 TP loss - 旧 baseline loss |
| --- | ---: |
| 31--40 | 最大绝对值 0.000110 |
| 51 | -0.000750 |
| 55 | -0.001320 |
| 60 | -0.003410 |
| 100 | -0.029939 |

第 31--40 步很接近并不能排除 bug；真正的分叉发生在错误裁剪开始后。

旧结果目录：

```text
results/correctness-baseline-seed1234-20260912-200624
results/correctness-livepipe-seed1234-20260912-233847
results/correctness-livepipe-pp-seed1234-20260913-190243
results/correctness-livepipe-tp-seed1234-20260919-200148
```

工作电脑上本次 TP 的目录为 `correctness-livepipe-tp-seed1234-20261011-185429`。用户提供的新 baseline 日志与本次 TP 日志可以直接核对以下步数：

| iteration | 新 baseline loss | 本次 TP loss | TP - baseline |
| --- | ---: | ---: | ---: |
| 54 | 10.184040 | 10.184070 | +0.000030 |
| 55 | 10.170990 | 10.170870 | -0.000120 |
| 56 | 10.158370 | 10.158650 | +0.000280 |
| 58 | 10.134450 | 10.134110 | -0.000340 |
| 59 | 10.122480 | 10.122540 | +0.000060 |

这些选定步数的最大绝对差为 `0.000340`，并未呈现旧对照在这一窗口的持续负向分离。这是支持修复有效的初步 GPU 证据，不能单独证明完整 optimizer state 或恢复梯度逐位正确。

验证记录的来源需要区分：

- 本交接整理时已独立复核本地 CPU/Gloo 测试以及上述用户提供的 GPU 日志片段。
- 既有 BUG016 文档的 2026-10-12 结果登记进一步报告：新 baseline `correctness-baseline-seed1234-20261011-224014` 与本次 TP 完成 100 步，对照差值约不超过 `4e-4`、无持续漂移；并报告新 DP、PP 的尾部差值分别为 `5.6e-05`、`4.3e-04`。
- 后一组完整 GPU 结果属于既有文档记录，本次没有获取全部原始新日志、精确 DP/PP run ID 和审计输出进行独立复核。接手者应取得这些产物后验证，不能把记录中的结论当成自己已经完成的测试。
- 小幅 loss 差异与浮点归约顺序变化相容，但仅凭误差量级不能证明残差一定是正常噪声。

正式验收必须使用同一修复版本的新 baseline，不能混入旧 baseline。若接手者修改了代码版本或缺少对应原始数据，需要在目标环境重跑验证。

## 4. 四种 correctness 拓扑

配置位于 `experiments/correctness/configs/`：

| 模式 | stage/op 的 DP 度 | stage/op 的 TP 度 | 说明 |
| --- | --- | --- | --- |
| baseline | `[2, 2, 2, 2]` | `[1, 1, 1, 1]` | 4 个 PP stage，op 数 `[40,39,39,41]` |
| LivePipe DP | `[2, 2, 1, 2]` | `[1, 1, 1, 1]` | PP4，减少一个中间 stage 的 DP 副本 |
| LivePipe PP | `[2, 1, 1, 1, 2]` | `[1, 1, 1, 1, 1]` | PP4 到 PP5，op 数 `[40,13,26,39,41]` |
| LivePipe TP | `[2, 2, 1, 1]` | `[1,1,2,1]` | 第三个 stage 使用 TP2，同时改变 DP 分布 |

注意：DP、PP 旧曲线看起来接近旧 baseline，不代表它们没受影响。它们同样使用错误 norm，只是在旧日志中范数、裁剪时机与旧 baseline 接近。这与被减少副本的 op 梯度平方和较小相容，但各 op 的真实贡献尚需测量，不能据总 norm 直接下结论。DP、PP 保留末 stage 的 DP2，而 TP 还将其改为 DP1。

## 5. 根因

旧代码链如下：

1. `megatron/optimizer/optimizer.py` 的 `get_main_grads_for_grad_norm()` 只排除了 shared 参数和 TP duplicate 参数，没有排除 DP replica。
2. `megatron/mpu/initialize.py::get_model_parallel_group()` 在本项目弹性拓扑实现中返回 `None`。
3. `megatron/optimizer/clip_grads.py::clip_grad_norm_fp32()` 对 `total_norm` 调用 `all_reduce(..., group=None)`，实际变成 WORLD 归约。
4. local DDP 梯度已经在 DP 副本间归约；同一 logical op 的 DP 副本拥有相同梯度，WORLD 归约再次把它们全部相加。

对普通非分片 optimizer，旧公式等价于：

```text
错误 norm² = Σ_op DP_size(op) × ||grad_op||²
正确 norm² = Σ_op                 ||grad_op||²
```

这是实际裁剪行为的 bug，不只是日志显示错误，因为 `clip_grad_norm_fp32()` 返回的 norm 直接决定：

```text
clip_coeff = max_norm / (total_norm + 1e-6)
```

baseline 全部为 DP2 时，旧 norm 约为正确 norm 的 `sqrt(2)`；它会在正确 norm 约 `0.707` 时就开始按阈值 1.0 裁剪。混合 DP 拓扑则对不同 op 施加不同权重，不能用一个全局系数修正。

不要把根因归结为 tied embedding。`flex_ops.py` 中标为 shared 的输出 embedding 已被 norm 筛选排除；tied embedding 是另一个独立 bug，不能用来解释本问题的 TP-specific norm 缺口。

## 6. 推荐的最小修复

### 6.1 给参数保留 logical op index

在 `megatron/model/distributed.py` 的 LocalDDP 包装初始化阶段，遍历 `language_model.ops`，给参数设置：

```python
param._livepipe_op_index = int(op.op_index)
```

当前工作区已有此标记逻辑，迁移前先检查；若已有，不要重复加入。标记代表全局 logical op，不是 stage 编号或 DP rank。应保留 `hasattr` 检查，避免同一参数被遍历多次时覆盖已有归属。

这个属性必须随着 FP16/BF16 参数复制到 FP32 master 参数。最合适的入口是：

```text
megatron/mpu/layers.py::copy_tensor_model_parallel_attributes()
```

在已有 tensor-model-parallel 属性复制后额外复制 `_livepipe_op_index`。

### 6.2 普通 optimizer 按 logical op 去重

修改：

```text
megatron/optimizer/optimizer.py::MegatronOptimizer.get_main_grads_for_grad_norm()
```

建议签名：

```python
def get_main_grads_for_grad_norm(self, data_parallel_sharded=False):
```

每个参数的筛选顺序应为：

1. `param.grad is not None`；
2. `param_is_not_shared(param)`；
3. 如果参数带 `_livepipe_op_index`：
   - TP-sharded 参数保留每个 shard；
   - TP-replicated 参数只保留当前 op 的 TP group rank 0；
   - ordinary optimizer 且 `data_parallel_sharded=False` 时，再只保留当前 op 的 DP group rank 0；
4. 如果参数没有 `_livepipe_op_index`，保持旧的 TP duplicate fallback；DP 选择使用 `mpu.get_data_parallel_group(None)` 的默认组。不要编造 op index 或强行查询一个当前 rank 不属于的 op group；
5. 通过筛选的梯度加入 `grads_for_norm`。

下面是现有筛选函数内部的关键逻辑，`grad_not_none`、`is_not_shared` 和 `grads_for_norm` 使用该函数已有定义：

```python
op_index = getattr(param, '_livepipe_op_index', None)
if op_index is None:
    is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
else:
    is_not_tp_duplicate = (
        getattr(param, 'tensor_model_parallel', False) or
        torch.distributed.get_rank(
            group=mpu.get_tensor_model_parallel_group(op_index)) == 0
    )

if grad_not_none and is_not_shared and is_not_tp_duplicate:
    if not data_parallel_sharded:
        dp_group = mpu.get_data_parallel_group(op_index)
        if torch.distributed.get_rank(group=dp_group) != 0:
            continue
    grads_for_norm.append(grad)
```

### 6.3 Distributed optimizer 必须保留所有 DP shard

在：

```text
megatron/optimizer/distrib_optimizer.py::DistributedOptimizer
```

覆盖或调用父类方法时传入：

```python
def get_main_grads_for_grad_norm(self):
    return super().get_main_grads_for_grad_norm(data_parallel_sharded=True)
```

理由：普通 optimizer 的 DP rank 持有同一份逻辑梯度副本；distributed optimizer 的 DP rank 持有不同 shard，不能按 DP rank 0 去重。

### 6.4 处理空的本地 norm contributor

修改：

```text
megatron/optimizer/clip_grads.py::clip_grad_norm_fp32()
```

要求：

- infinity norm：`max(..., default=0.0)`；
- 非 L2 有限阶：先创建 device tensor 0，再累加；
- L2 已有空列表的 0 tensor 分支；
- 不要因为本 rank 没有 norm contributor 就提前 return；所有 rank 仍需参与 norm all-reduce，并用同一个 `clip_coeff` 缩放自己 `parameters` 中的所有本地梯度。

### 6.5 不要做的修改

- 不要把 `get_model_parallel_group()` 简单改回旧的固定 `_MODEL_PARALLEL_GROUP`；弹性拓扑中该 group 设计为 `None`，问题应在参数筛选和 logical op group 上解决。
- 不要全局把 `param_is_not_tensor_parallel_duplicate()` 改成查询 op-specific group；未标记的 replica/CPU model 可能不属于这些动态 group。
- 不要用全局 DP 倍数、总 GPU 数或 `sqrt(2)` 修正 mixed topology。
- 不要只缩放 `grads_for_norm`；裁剪必须作用于所有本地 `parameters` 的梯度。
- 不要修改 tied embedding 梯度同步逻辑来“顺便”解决本 bug。
- 不要回退已有的 `sequence_parallel` guard 改动；那不是本 bug 的修复内容。
- 不要缓存恢复前的 DP owner/rank；每次筛选应查询当前 op group，PP 重切分或 rank 重映射后需要使用新组。
- 不要整份复制另一个工作区的 `optimizer.py`；该文件可能同时包含其他 bugfix。只移植本节对应的函数改动。

## 7. 最小复现

最小数学复现只需要一个标量梯度：

```text
逻辑梯度 g = 1，clip_grad = 1.1
DP1：正确 norm = 1.0，不裁剪
DP2 旧路径：错误 norm = sqrt(2)，会裁剪
```

因此该问题不需要故障恢复、TP 或大模型即可复现；恢复只是改变 DP 分布，使错误 norm 在两条训练路径之间不同。

下面可以用标准库验证数学机制；它不是对仓库真实 optimizer 的集成测试：

```python
import math

gradient = 1.0
threshold = 1.1
for replicas in (1, 2):
    wrong_norm = math.sqrt(replicas * gradient ** 2)
    true_norm = abs(gradient)
    wrong_scale = min(1.0, threshold / (wrong_norm + 1e-6))
    true_scale = min(1.0, threshold / (true_norm + 1e-6))
    print(replicas, wrong_norm, true_norm, wrong_scale, true_scale)
```

DP2 输出的错误缩放系数约 `0.777817`，而正确系数为 1。

代码级回归测试位于：

```text
tests/test_gradient_norm_contract.py
```

如果另一台电脑尚无该测试文件，应新增独立回归测试；不要仅验证上述数学公式，必须调用实际修改后的函数。由于完整 optimizer 模块依赖 Apex/CUDA，当前 CPU 测试用 AST 提取真实函数和必要方法、以 CPU adapter 替代 CUDA 类型与 Apex kernel，并使用真实 CPU tensors；Gloo 测试中的 group rank 查询和 all-reduce 是真实双进程操作。adapter 不等价于真实 Apex/NCCL 验证。

该测试应覆盖：

- DP1/DP2 标量 norm 不变；
- DP 到 TP 的最小拓扑转换；
- heterogeneous op layout；
- shared 参数和 TP duplicate 参数；
- FP32 master 参数继承 `_livepipe_op_index`；
- reconfiguration 后重新查询 op-specific group；
- distributed optimizer shard；
- norm contributor 为空时的 L1/L2/infinity；
- 两进程真实 Gloo group rank/all-reduce。

## 8. 验证命令

先做静态检查：

```bash
git diff --check
python3 -m py_compile \
  megatron/model/distributed.py \
  megatron/mpu/layers.py \
  megatron/optimizer/optimizer.py \
  megatron/optimizer/distrib_optimizer.py \
  megatron/optimizer/clip_grads.py
```

运行专门测试及完整 CPU 回归：

```bash
python3 -m unittest discover -s tests -p 'test_gradient_norm_contract.py' -v
bash experiments/correctness/run_cpu_tests.sh
```

当前专门测试 **12 tests OK**；完整 CPU suite 已复跑为 **95 tests OK**，另包含 mock recovery、DP/PP/TP 配置校验和 shell 语法检查。测试总数会随接手分支变化，应检查实际通过数、失败和跳过原因，不要只匹配数字。如果环境没有 PyTorch，不能把测试跳过视为通过，应使用目标环境兼容的 CPU PyTorch 或生产环境直接验证。

另已完成一次不写入仓库的合成检查：按四份实际配置的全部 159 个 logical op 构造同一组真实 CPU 梯度，调用真实 norm 筛选/裁剪方法的 CPU adapter。baseline、DP、PP、TP 均得到 norm `60.214201`，裁剪和不裁剪两种阈值均正确缩放所有本地梯度。该检查验证拓扑计数规则，不是完整 PP 通信或模型恢复测试。

确认四份配置仍有效：

```bash
python3 experiments/correctness/validate_configs.py \
  experiments/correctness/configs/gpt_1_3B_12layers_pp4_dp2_8gpu.json \
  experiments/correctness/configs/gpt_1_3B_12layers_pp4_mixed_dp_7gpu.json \
  --require-change dp

python3 experiments/correctness/validate_configs.py \
  experiments/correctness/configs/gpt_1_3B_12layers_pp4_dp2_8gpu.json \
  experiments/correctness/configs/gpt_1_3B_12layers_pp5_mixed_dp_7gpu.json \
  --require-change pp

python3 experiments/correctness/validate_configs.py \
  experiments/correctness/configs/gpt_1_3B_12layers_pp4_dp2_8gpu.json \
  experiments/correctness/configs/gpt_1_3B_12layers_pp4_mixed_tp_dp_7gpu.json \
  --require-change tp
```

## 9. GPU 验收顺序

正式 GPU 对照不要混用旧 baseline。使用同一修复版本、同一 seed、同一数据顺序和同一 NCCL transport，依次运行：

```bash
bash experiments/correctness/run_single_node_8gpu.sh baseline
bash experiments/correctness/run_single_node_8gpu.sh livepipe-tp
bash experiments/correctness/run_single_node_8gpu.sh livepipe
bash experiments/correctness/run_single_node_8gpu.sh livepipe-pp
```

不要并发启动这些实验。默认正式配置为 100 steps、iteration 30 注入故障、`--clip-grad 1.0`。先验收 baseline 与 TP 的后半段，再补 DP/PP；但正式结果矩阵四种都应使用新 baseline 和新代码重跑。

一条同版本的新 baseline 可以供三种恢复模式共用。旧训练状态已经受到错误裁剪影响，正式验证应从相同初始化条件重新运行，不能只修改旧日志中的 norm。为尽快看初步结果，可以在 55--60 步时读取已有日志比较，但不要把正式实验的 `TRAIN_ITERS` 改成 60：启动脚本同时据总步数设定学习率衰减，会改变轨迹。

比较工具：

```bash
python3 experiments/correctness/compare_and_plot_loss_curves.py \
  --baseline-dir "$BASELINE_DIR" \
  --livepipe-dir "$LIVEPIPE_DIR" \
  --failure-iteration 30 \
  --output-dir "$LIVEPIPE_DIR/against-fixed-baseline"
```

重点看：

1. 恢复链路和 7 个 survivor 是否完成 100 steps；
2. grad norm 是否不再因 DP 拓扑出现系统性缺口；
3. 约第 48--60 步及第 84--100 步的 loss delta 是否持续扩大；
4. 新 baseline 与新 TP 是否保持小幅、无趋势的浮点误差，而不是旧 TP 的单调尾部漂移；
5. DP、PP 的曲线不能只看第 31--40 步，必须至少检查进入裁剪后的后半段。

日志的 loss 是本轮 optimizer 更新前的前向结果。第 48 步首次错误裁剪对 loss 的影响最早出现在第 49 步，且不保证在该步就明显超过日志精度。第 83 步 norm 显示 `1.000` 也有三位小数舍入歧义，不能据此精确判定首次裁剪步。

应使用 `experiments/correctness/audit_progress.py` 检查样本/进度记录，但审计通过不能代替梯度和 optimizer 数值校验。若修复后仍出现持续漂移，下一步从相同 V30 参数、完整 Adam/scaler 状态和相同逻辑 batch 做单步拓扑对照；7 卡从头训练与 8 卡训练历史不同，不是决定性的恢复正确性证据。

注意：日志中的 `grad norm` 是裁剪前计算出的 norm；看到 `norm > 1` 并不表示裁剪失效，而是表示该 step 会使用小于 1 的 clip coefficient。

## 10. 交接时请返回的内容

另一台 AI 完成后应返回：

- 实际修改的文件列表；
- 每个修改为何必要，特别是 ordinary optimizer 与 distributed optimizer 的区别；
- 专门测试和完整 CPU suite 的结果；
- 是否跑过真实 CUDA/Apex/NCCL；
- 新 baseline、DP、PP、TP 的 run directory 和 loss comparison 摘要；
- 若仍有漂移，必须区分 norm clipping 问题与同一 V30 参数/完整 optimizer state 下的 TP 计算图数值差异，不要直接把旧 baseline 当正确答案。

## 11. 相关文件

- Bug 记录：`docs/bugs/2026-10-11-grad-norm-dp-duplicate-scaling.md`
- 参数筛选：`megatron/optimizer/optimizer.py`
- DP-sharded 特例：`megatron/optimizer/distrib_optimizer.py`
- norm 归约和缩放：`megatron/optimizer/clip_grads.py`
- logical op metadata：`megatron/model/distributed.py`
- metadata 复制：`megatron/mpu/layers.py`
- 回归测试：`tests/test_gradient_norm_contract.py`
- correctness launcher：`experiments/correctness/run_single_node_8gpu.sh`
