# LivePipe 复现踩坑实录（现象 / 根因 / 修复）

> 2026-09-07/08 调试记录。按"谁再遇到能直接对号入座"的标准写。
> 环境类问题见 ENV_RECOVERY.md，网络类见 RDMA_NOTES.md，此处聚焦代码 bug。
>
> **⚠️ 总定性（2026-09-10 补，见 FAULT_MODEL.md）**：本文件 Bug 3 的十四轮
> 战争，对抗的是 supplementary 套件（555c631）引入的**迭代中途故障注入**
> （`--livepipe-fail-after-backwards`）——一个论文主实验从未使用、恢复
> 机制从未对它承诺过的故障模型（论文版 ab976d9 只有 iteration 边界注入，
> 通信空闲、无 jam）。所有修复依然有效且留作保险，但"论文场景本身有
> 这族 bug"是误读——回到论文故障模型只需去掉该 flag。

## 一、恢复链路上的代码 bug（核心）

### Bug 1：启动即 RuntimeError: interleaved pipeline schedule

- **现象**：首次跑 baseline，全部 worker 启动即挂，报
  `adjacent-replica correctness does not support the interleaved pipeline schedule`
- **根因**：`arguments.py` 的 `--interleave-factor` 默认 1，且无条件赋值
  `args.virtual_pipeline_model_parallel_size = args.interleave_factor`；
  `training.py` 用 truthy 判断 `if getattr(args, 'virtual_pipeline_model_parallel_size', None):`
  ——值为 1（其实没有交错）也被误判为开启了交错调度。
- **修复**：`megatron/training.py` ~1819 行改为 `_vpm > 1` 才报错。

### Bug 2：故障注入后整体挂死（p2p 老路径）★

- **现象**：iteration 30 杀 rank 4 后，agent 无限轮询 `ready_for_rendezvous_rank*`
  （一小时以上无 ready 标记）。py-spy 显示 6 个 worker 卡在
  `p2p_communication.py:692` 的 `torch.cuda.synchronize()`；
  rank 1 的 ALLREDUCE 在 **30 分钟**后 watchdog 超时报错，但也不解救任何人。
- **根因**：`_communicate_flexpipe` 不带 `--elaceso-no-batch-isend-irecv` 时走
  `_communicate_flexpipe_old` 老路径：等待 p2p 完成时的**故障检测循环被作者注释掉了**
  （注释写着 "Temporarily remove for performance"），函数结尾还有全设备
  `torch.cuda.synchronize()`——只要有任何指向死卡的挂起 NCCL kernel，它永不返回。
  带 `failed_rank` 轮询逃逸的实现（`_communicate_flexpipe_async`）默认不启用。
- **修复**：启动参数加 `--elaceso-no-batch-isend-irecv`。
  （`run_single_node_8gpu.sh` 的 livepipe* 块、livepipe-exp 的 03/03b/06 故障脚本均已加）
- **验证**：修复后 5/7 rank 在故障**同一秒**打出 `catch failed rank: 4` 并逃逸。

### Bug 3：冗余副本通信卡死（剩余 rank 不逃逸）★★

- **现象**：Bug 2 修复后，每轮冒烟都卡在 5/7 或 6/7——缺席的永远是**死卡的
  冗余伙伴 rank**（rank 1/7）。py-spy 三轮取证：卡点为 `elaceso.py`
  `send_with_bucket` 里建 header tensor（`torch.tensor(..., device=cuda)`）或
  `livepipe_progress.py` `record_forward_loss` 的 `.item()`——都是**非 NCCL 的
  普通 CUDA 操作**。
- **根因（系统性必现，不是运气）**：故障注入瞬间，死卡的冗余伙伴必然正在与它
  做副本同步（LivePipe 冗余设计使然，微批次 ~2s 一轮）。而 `failed_rank` 由
  agent 在**数秒后**才写入 etcd——这个窗口里伙伴发起的指向死卡的 isend
  kernel 永不完成 → stream 永久 jam → 伙伴后续**任何** CUDA 操作（建 tensor、
  `.item()`）永久排队。入口检查是纯 CPU 的本可逃逸，但人已经卡在 CUDA 里。
  作者在注入代码注释里写的期望是 "peers leave their pipeline calls via NCCL
  error handling"——本版 torch 对非 NCCL 操作做不到。
- **修复（三轮递进，前两轮各自不够）**：
  1. `send_with_bucket`/`recv_with_bucket` 入口对端级 failed_rank 检查
     （elaceso.py `_redundancy_failed_rank()`）——保护"公布后新发起的调用"；
  2. `TORCH_NCCL_ASYNC_ERROR_HANDLING=2` + 子进程组 120s 超时（Bug 4）——
     watchdog 实测开火（120s），但只把 communicator 置错误态，**解不了非 NCCL
     操作的排队**，救不了已 jam 的 rank；
  3. **零延迟公布（最终修复）**：被注入的 worker 在睡觉前**自己**把
     `failed_rank` 写进 etcd（schedules.py 的 fail-after-backwards 注入点 +
     elaceso.py `need_reconfigure` 的普通注入点）——窗口从秒级压到毫秒级，
     伙伴的下一次入口检查（≤2s 后）命中逃逸，**jam 根本不会形成**。
- **核心教训**：故障逃逸检查必须 (1) 纯 CPU/etcd，(2) 在任何 CUDA 操作之前，
  (3) 覆盖所有通信路径，(4) **故障消息的公布延迟要小于一次通信周期**——
  否则伙伴必然已经身处阻塞操作中，任何事后检查都救不了。
- **后记（入口检查自身的 bug，Codex 评审发现）**：初版入口检查写了
  `send_flatten_tensor_all.get(...)`，但这两个变量是 **list**（`[{},{}]` / `[]`），
  `.get()` 抛 AttributeError 且 `except NameError` 接不住——后果：精心写的
  "redundant send aborted" 永不触发（grep 关键词为空本应引起警觉）、对端级
  判断失效变成无差别爆炸（恢复阶段若再调用会在恢复中途炸）、部分 rank 靠
  AttributeError 意外逃生但留下误导性日志。修复：按下标取元素并做边界保护
  （`0 <= replica_index < len(...)`）。**教训：验证补丁时要核对异常类型，
  不能只看"恢复走通了"；grep 关键词零命中 = 补丁没跑过指定路径，要查为什么。**
- **线程路径（Codex 二审发现）**：`send_in_node()` 为真时 send_with_bucket 在
  `threading.Thread` 里执行，线程内抛的异常到不了 training.py:1646 的 except。
  修复：入口检查按 `threading.current_thread() is main_thread()` 分流——主线程
  抛异常进恢复流程；传输线程静默 `return None` 跳过传输，恢复由主线程的
  need_reconfigure（确定性 return True）/p2p 检查触发。
- **写入顺序（Codex 建议）**：两个注入点都把 `failed_rank` 提到最先写——
  它是伙伴入口检查轮询的键，先写可免疫"两次 etcd 写之间被杀"的间隙。
- **已知限制**：`failed_rank` 是单值键。整节点多卡故障（03b，num-gpus-per-node=4）
  时同节点多个死卡互相覆盖，入口检查只认最后写入的那个——跑 03b 前需改成
  列表键或多键写入（当前 1 GPU/agent 拓扑无影响）。
- **第五轮（拆组 barrier，7/7 逃逸后仍卡）**：全员逃逸、4 个 rank 完成拆组后，
  剩 3 个（恰为设备上有残留 jam 的 rank）卡死在 `torch.distributed.barrier`
  内部——**barrier 的 CUDA 级同步被 jam 设备堵死**，它们的 ready 标记永远
  缺席 → agent 凑不齐 → 新 rendezvous（v_2）永不组建 → 先行完成的 4 个在
  `wait_agent_to_rendezvous` 干等。修复：`check_for_preemption` 里的 NCCL
  barrier 换成 **etcd 计数 barrier**（各 rank 写 `destroy_barrier_rank{N}`，
  轮询到齐；纯 CPU，不受设备 jam 影响，语义等价）。
- **通用结论（三轮 jam 战役后）**：故障恢复路径上**任何**涉及 CUDA 的同步
  （synchronize / barrier / 建 tensor / .item()）都是潜在永久阻塞点——只要
  死卡通信对象在故障瞬间有在飞 kernel，相关设备就会被 jam 到进程组重建。
  恢复路径必须全部走 CPU（etcd）做同步与决策。
- **第六轮（destroy 挂死，最后一块骨牌）**：etcd barrier 修复生效（7/7 过
  barrier），但日志计数 7→7→4→4 证明 3 个 jam 设备的 rank 卡死在
  `destroy_all_groups` 内部——ncclCommDestroy 等待在飞操作完成，jam kernel
  永不完成。修复：destroy 放**后台线程限时（60s）执行**，超时则放弃等待
  （旧通信子泄漏，进程组即将重建）、`force_reset_groups()` 清记账、置
  `destroy_give_up` 让主线程跳出 `is_initialized` 自旋。判据更新：验收时
  `destroyed the process group` + `destroy 60s timeout` 合计应 = 7。
- **第七轮（恢复后重 init 超时，级联塌方）**：v_2 组建、agent 重启新一代
  worker 后，重新 `init_process_group`（training.py:1397）的 store barrier 沿用
  `--process-group-timeout 120`——NAS 冷启动的 worker 导入+初始化 >2min，
  迟到的把先到的拖超时（`worker_count=4/7`）→ 先到的崩 → agent 判死 →
  `failed_rank=5` 第二轮重配置 → agent 逐个耗尽退出（8→3）。修复：恢复期
  重 init 的 timeout 取 `max(600s, process_group_timeout)`。**教训：同一个
  超时参数被两个阶段复用时，要按最慢阶段（冷启动）定值；训练期 120s 够用
  不代表恢复期够用。**
- **第十轮（C 级真相：析构挂死，gdb 实锤）**：**现象**——key 分叉修复后
  6/7 rank 在正确的 `store_based_barrier_key:1` 会合，唯 rank 5 缺席；
  py-spy 三线程僵局（主线程原生空转无栈帧、daemon 卡在 force_reset 的
  纯赋值行、bg destroy 卡在 destroy_all_groups）；gdb 原生栈定罪：
  `cudaIpcCloseMemHandle ← p2pSendFree ← ncclCommAbort ←
  ProcessGroupNCCL::~ProcessGroupNCCL ← _Py_Dealloc`——**NCCL 通信子的
  C++ 析构**要释放 IPC 缓冲，而缓冲被挂起 kernel 占用 → 驱动层永久自旋；
  析构由 Python 引用计数触发，**任意线程**（含主线程，伴随 GIL 冻结）
  碰掉旧 PG 引用都会当场内联执行它。**修复（反直觉但有效）：泄漏防析构**
  ——destroy 前把全部旧 PG 对象转入模块级 `_PG_LEAK_LIST` 永久持有，
  引用永不清零 → 析构永不发生 → destroy 退化为纯记账秒回。**教训：
  Python 的"清理"不是免费的——C++ 扩展对象的析构可能挂死，且会在任意
  释放引用的线程内联执行；故障恢复场景下，宁可泄漏也不要触发不可控的
  析构链。**
- **第十九轮（三层连锁：comm 计划层的存活规则副本 + 退出路径析构 + 多故障重入）**：
  **现象**——带十七/十八轮修复的 172008 轮：重 init 7/7 通过，但
  reconfigure 再次 0/7；agent 日志出现 `failed_rank=2`（**第二次故障**）
  且 v_3 组不起来。**取证链**——① gdb 原生栈实锤崩溃 worker 的退出路径：
  `_PyModule_ClearDict → list_dealloc → ~ProcessGroupNCCL → ncclCommAbort →
  cudaFreeHost` 挂死——**第十轮的泄漏清单在解释器清理时自身被销毁，保护
  失效**（泄漏防析构只覆盖恢复路径，没覆盖进程退出路径）；② 第一张骨牌：
  rank 2 在 `get_reconfigure_comm`（elaceso_utils.py:798）抛
  `AssertionError: failed to find an alive dp group for recovery`——
  **传输执行层的 `groups_are_alive` 和 planner 的 `_alive` 是同一"全组
  存活"规则的第二个副本**：planner 放行部分存活组后，comm 计划层拒绝了它。
  **修复（两件套）**——① `get_reconfigure_comm` 加"自给自足跳过"：新持有者
  本身是选中源组的存活成员且 TP 布局未变时（数据就在手上，梯度按 plan
  prefix 重放），跳过通信计划构建；② `pretrain_gpt.py` 主入口 try/finally
  走 `os._exit()`——绕过解释器清理，让 OS 直接回收 CUDA 上下文（进程死亡
  = 驱动强制销毁，挂起 kernel 一起消失）。**验证**——`repro_reconfigure_comm.py`
  纯 CPU 双用例：op 81 场景跳过不炸 + 含新 rank 的正常组仍建计划（不过度
  跳过）。**教训：① 同一不变量（全组存活）散落在 planner 和传输执行层两处，
  改一处必须全局搜同源副本——`grep -rn "are_alive"` 应是改完的第一反应；
  ② 泄漏/防析构类修复必须覆盖对象的全部生命周期终点，包括进程退出；
  ③ 恢复后的二次故障（v_3）触发了从未设计过的多故障重入路径——单故障
  恢复没稳之前，恢复期再死一个 rank 就是自由落体。遗留**：多故障重入
  （恢复中再死 rank）当前无保护，属设计外场景。
- **第十八轮（load_recv_buckets 全设备同步挂死——通用教训又一漏网点）**：
  **现象**——第十七轮修复后重跑：重 init 7/7 照常通过，但
  `Model reconfigure finished` 仍 0/7、12 分钟无进展。py-spy 实锤 3 个
  worker 主线程全部卡在 `torch.cuda.synchronize()`
  （elaceso.py:480，`load_recv_buckets` 内，reconfigure 调用链上）。
  **根因**——该函数把 CPU 上的副本 weights/optimizer `.cuda()` 搬回 GPU
  后来了个**全设备**同步：它要等 jam 设备上永不完成的旧 NCCL kernel，
  永久挂死。Bug 3 通用教训（恢复路径上任何 CUDA 同步都是潜在阻塞点）的
  又一漏网点：tmp_timer 修了（11c）、send/recv 的 sync_p2p 分支当前不走、
  debug 分支不触发——这处漏了。**修复**——换成流级同步
  `torch.cuda.current_stream().synchronize()`：H2D 拷贝全部提交在当前流，
  语义等价，不被旧 kernel 所在流堵。**教训：恢复路径的函数清单要随调用链
  演进维护——审计扫描的函数列表里没有 load_recv_buckets，它就漏网了；
  每修一处新卡点，先回头把它的调用方加进扫描清单。**
- **第十七轮（op 81 恢复计划 invariant：副本水位 + 全组存活规则过严）**：
  **现象**——四轮连炸同一点：重 init 7/7 成功、reconfigure 0/7，
  `build_adjacent_recovery_plan` 抛
  `RecoveryInvariantError: no version-complete normal or adjacent source for op 81`。
  **根因（RECOVERY-DIAG dump 实锤）**——op 81 本尊组含死卡（[-1, 4]），
  `_alive()` 全组规则直接作废；相邻副本组 [2,3] 的 `replica_records`
  **全空**——副本传输按序推进，故障点（backward 64）时水位线停在
  op 80/81 之间，op 81 的副本**从未传输**；而活着的 DP 伙伴
  （旧 rank 5→新 rank 4）`normal_has_op=true, prefix=64, gradient_valid=true`
  数据全须全尾却被闲置。**修复**——`_normal_candidate` 接受部分存活组：
  权重/优化器按 DP 复制语义从存活成员取（数据原地可用），梯度因缺死卡的
  DP 贡献回退 prefix=0 重放。**验证**——`experiments/correctness/repro_op81.py`
  纯 CPU 最小复现矩阵 7 用例：D（副本全空，=op 81 真实场景）从 RAISED→PASS；
  C（attempt 全局偏移）保持正确拒绝（该炸还得炸）；A/B/F 顺带兜住。
  **工具沉淀**——① repro_op81.py：importlib 按路径加载绕过 torch 依赖，
  秒级遍历五个判定维度；② RECOVERY-DIAG dump：invariant 抛出前把 op 的
  全维度证据（各 rank 版本/attempt/records）打进日志，一轮定案。
  **教训：① 整套中途恢复链（fail-after-backwards 注入、adjacent-replica
  后端、preserve-prefix 策略、版本追踪、invariant 检查）都是 555c631
  一个 commit 引入的未检验代码——已有 13 个单测全过，但 fixture 全是
  "版本齐全"的理想数据，真实传输竞态（滞后/中途失效/水位）一个用例没
  覆盖；② 纯数据结构的计划逻辑就该 CPU 单测遍历维度组合，比 20 分钟
  一轮的集群联调便宜一个数量级；③ 排查 dump 输出时 json.tool 要剥日志
  前缀——`char 0`（空）与 `char 1`（前缀挡道）是两种完全不同的状态，
  别把"有输出"误判成"没打出来"。**
- **第十六轮（agent 世代重启 + 日志自毁；遗留）**：第十五轮 barrier 修复
  生效（拆组 1 秒）后，恢复链推进到重 init 7/7，随后进入未知领域：agent
  重启了 worker 世代（新进程 exitcode 1 秒崩）、且日志文件被 truncate +
  NUL 空洞毁坏（新旧 fd 交错写），观测性归零，旧 worker 为何退出/新 worker
  为何崩/重启是否属设计（--max-restarts=0 语义下不应重启）三个问题待查。
  **明日路径**：A. 先修观测性（worker 日志按世代分文件）再冒烟定位；
  B. checkpoint 模式先出 correctness 主体结果。
- **第十五轮（论文故障模型下的首跑 + 我方 barrier 的竞态）**：按
  FAULT_MODEL.md 行动 1 去掉 `--livepipe-fail-after-backwards` 后首跑：
  **论文模型验证成功**——无 jam（逃逸 0 次、destroy 1.03s、无 give-up），
  恢复走作者设计路径。新卡点：6/7 worker 完成拆组等待 v_2，唯 rank 2
  卡在第五轮的 etcd barrier 里。**根因（自伤）**——barrier 入口只读一次
  failed_rank，瞬时 etcd 抖动（except 兜成 -1）会把死卡排除名单算错，
  该 rank 永远等一个不存在的标记；6 幸运 1 倒霉的分布与此完全吻合。
  **修复**——每轮重读 failed_rank（名单自愈）+ 60s 超时放行（barrier 只是
  destroy 前排序，超时无害）。**教训：自己加的兜底逻辑和原始代码一样要
  按"每一次外部读都可能失败"标准写——except 兜默认值进长期循环 = 把瞬时
  故障固化成永久状态；且带循环的修复必须有超时。**
- **第十四轮（etcd 状态残留假说；FAULT_MODEL.md 证伪，修复保留为保险）**：
  **现象**——重 init 7/7 且训练真的续上了（心跳里 iteration 3 出现两次，
  第二次是恢复后重放！此前被误读为"重建未完成"），随后秒级再崩：部分
  rank 死、幸存者卡原生层。**根因**——etcd 故障键全代码库无人重置：
  `failed_rank=4`、`should_reconfigure="1"`、`ready_for_rendezvous_rank*=1`
  残留 → 恢复后第一个 p2p 轮询（无世界门）集体读到 4 再次逃逸 + agent
  监控三条件秒齐发起 v_3 → 刚恢复的训练被打死。**修复（双保险）**——
  ① 恢复完成后清键（`Model reconfigure finished` 后写 failed_rank=-1
  （int！）、should_reconfigure="0"，幂等）；② p2p 逃逸循环 4 处加世界
  尺寸门（与 elaceso 入口检查同款），即使键残留也不误判。**教训：状态机
  要有"恢复完成"的清理或版本化（带 rendezvous 版本号的键天然免疫残留）；
  以及读日志要读细节——iteration 出现两次本可以早两轮指向"恢复成功过"。**
- **第十三轮（雷 2 应验：新通信子在 jam 设备上首连挂死 = 架构墙）**：
  **现象**——11c 生效（全部 worker 进入 reconfigure、sync skipped 正常），
  但 jam rank 的 MainThread 再次原生层 active 无 Python 帧（排除 .item()，
  那会显示 `_as_python_scalar` 栈）；其余 6 个 worker 等它的重建数据超时
  阵亡。**结论**——外部评审预警的雷 2 坐实：泄漏拆弹解决了"关旧 IPC"，
  但"开新 IPC"（新 NCCL 通信子首次 kernel/连接）在同一设备上同样挂死。
  **jam rank 的设备在进程内不可恢复**——这是进程内自愈路线的天花板
  （评审第一因判断成立）。**候选出路**（按成本排序）：① 换
  `TORCH_NCCL_ASYNC_ERROR_HANDLING=3/1`（若该模式超时后 abort 通信子、
  杀掉挂起 kernel，设备即自愈——纯环境变量实验，20 分钟一轮）；
  ② jam rank 重建走 CPU 中转（绕开其设备的 NCCL，手术量大）；
  ③ jam rank 进程级重加入（fresh CUDA context，动 elastic 层）；
  ④ checkpoint 基线先出 correctness 结果，恢复链作为独立工程继续。
- **第十二轮（预防性静态审计——停止"实跑当编译器"）**：外部评审（Claude）
  指出修复组织方式的结构性问题：按症状点打补丁 + 人肉枚举分支 + py-spy 级
  归因，在面状故障域面前只能轮轮还利息。核实与采纳：① `ready_to_destroy`
  埋点实测 **20 处**分散在 p2p_communication/training（横切关注点未收敛，
  未来应统一到通信原语入口）；② `init_process_group` 实测 **5 处调用只修了
  1 处**——已把 max(600s) 统一到全部 3 个重 init 分支；③ 静态审计恢复
  路径危险调用，发现两处残余：
  - `elaceso.py _as_python_scalar` 的 `value.item()`（~1475）——重建路径
    活代码，GPU 张量 .item() = D2H 同步，jam 设备上会挂（与 synchronize
    同类，无法限时）；若下一轮卡在 .item() 即此雷；
  - training.py 恢复区 3 处 `torch.distributed.barrier()`（~1533/1595/1626，
    adapt_dnn/scale-out 分支）——第五轮教训未覆盖的平行分支，correctness
    实验不执行，跑 03b/论文实验前需换 etcd barrier。
  **预警（评审点名的下一颗雷）：泄漏拆弹只解决了"关旧 IPC"
  （cudaIpcCloseMemHandle），新 NCCL 通信子在 jam 设备上的首次连接
  （开新 IPC）未验证——若下轮表现为"重 init 成功但首次集合通信卡死"，
  即此处。**
- **第十一轮（重建期的 synchronize，老敌人换阵地）**：**现象**——泄漏拆弹
  大捷：`PRE: 6 pgs leaked (destructors defused)` ×6、**重 init 7/7 一次通过**
  （十轮战争的最厚城墙倒了）。但恢复停在模型重建：40 分钟无新迭代，
  只剩 1 个 worker 活着。py-spy：`tmp_timer (elaceso.py:1460)` →
  `torch.cuda.synchronize()`——第二轮的老敌人（设备级同步 × jam 设备 =
  永久挂死）在重建路径原地复现：泄漏策略保住了 destroy/init，但 jam 设备
  上的残留 kernel 还在，重建计时器一脚踩上。**修复**——tmp_timer 本是纯
  计时函数（synchronize 只为日志精确，零功能语义）：改为限时 5 秒线程
  执行，超时打 `timing skipped` 继续走。**教训：计时/日志类辅助代码必须
  和功能代码同一标准审查——它一样能把恢复链挂死；jam 设备的影响是全程
  的，恢复路径上每一个 CUDA 同步点都要有超时或跳过策略。**
  - **11b（线程限时方案翻车）**：限时线程版在病态进程上引入新死点——
    py-spy 显示主线程卡死在 `Thread.start()`（新线程永远无法 bootstrap）。
    改为**确定性方案**：不搞线程，give-up 路径置 `elaceso._device_jammed`
    标志，tmp_timer 见标志直接跳过 synchronize。**教训：病态环境下的修复
    要做减法（跳过）而不是加法（再引入并发原语）——新机制本身就是新风险面。**
  - **11c（标志方案的逻辑漏洞）**：`_device_jammed` 只在 give-up 路径置位，
    但第十轮的泄漏拆弹让 destroy 秒回——rank 不再走 give-up → 标志永远
    不置位 → jam 设备上的 synchronize 重新裸奔（jam 的存在与是否走 give-up
    无必然关联）。修复：tmp_timer **无条件**跳过 synchronize（纯计时函数，
    代价只是恢复期计时日志变成 CPU 侧近似值）。**教训：条件防护的触发
    条件要和风险来源直接挂钩——经过另一层修复后，原来的代理指标
    （give-up 路径）会失效。**
- **第八轮（"initialize the default process group twice!"）**：第六轮的限时
  放弃路径副作用——give-up 的 rank 默认进程组没销毁成功（ncclCommDestroy
  挂死），`is_initialized()` 仍为 True，v_2 重配置的 `init_process_group`
  直接抛 twice! 秒崩；它一缺席，其余干净 rank 在 store barrier 等到超时
  团灭（本轮 1 个 twice + 6 个超时，jam 的 rank 数量随故障瞬间在飞通信
  而定，上轮是 3+4 分布）。修复：give-up 时手动清空 c10d 的默认组记账
  （`_world.pg = None`，模拟 destroy 的记账效果、跳过挂死的通信子销毁）。
  **教训：绕过一个挂死的清理操作时，要把它对"状态机"的记账影响补齐，
  否则下游（这里的 re-init）会按旧状态拒绝合法操作。**
  - **8b（私有 API 布局坑）**：初版 poke `_world.pg` 是 torch 2.x 的布局，
    1.13 是模块级 `_default_pg`——try/except 吞掉 AttributeError 后优雅失败，
    twice! 依旧。改为两个布局都尝试（`_default_pg` / `_world.pg`）。
    **教训：碰 torch 私有 API 前先用 hasattr 确认布局；"优雅吞异常"的兜底
    必须打日志，否则补丁静默失效和没打补丁一样。**
  - **8c（第一层根因：包导出缺失）**：日志实锤 `module 'megatron.mpu' has no
    attribute 'force_reset_groups'`——函数写在了 `mpu/initialize.py` 但没加进
    `mpu/__init__.py` 的导出清单，`from megatron import mpu` 看不到它。守护
    线程死在这一行 → poke 从未执行 → 多轮 twice! 由此而来。修复：`__init__.py`
    加一行导出；闸门脚本增加"包导出"检查项。**教训：给包内模块加公共函数，
    要同步改 __init__ 的导出；自检要查"调用方视角"（import 后有没有），
    不是"文件里有没有"。**
  - **8d（静默失灵 + 面包屑实证法）**：**现象**——8c 部署后（mtime 证实
    `__init__.py` 14:16 写入、早于 14:52 的 run 启动；服务器代码逐行 grep
    核实与本地一致）twice! 依然出现，且 poke 呈"三无"：无成功日志、无失败
    日志、无任何 Traceback。静态分析穷尽，排除了导出缺失、导入竞速、GIL
    死锁、logging 丢失、异常吞噬五种可能。**修复（实证手段）**——放弃推理，
    give-up 路径重构为面包屑埋点版：① poke 提到最前（重 init 成败的关键
    步骤，后续步骤炸了不连累它）；② 每步 `print(..., flush=True)` 打
    A/B/C/D 标记（绕开 logging 全链路）；③ poke 与 force_reset_groups 各自
    try/except 隔离，后者失败降级为日志。**判读**：A→B→C 齐全 + twice=0
    = 修好；轨迹断在哪两个字母之间，死点就在那里；连 A 都没有 = 线程在
    force-continue 日志后直接消失，需 py-spy 活捉。**教训：当代码"看起来
    对"但行为不对且零报错时，别继续推理——埋面包屑让下一轮 run 自己招供；
    关键动作放前面，次要动作失败降级不中断。**
  - **8d 实证结果（面包屑招供）**：三个 give-up rank 的轨迹齐断在 **B→C
    之间、无异常**——`hasattr(_c10d, "_default_pg")` 为 False，poke 整段是
    静默空操作：这个 torch 1.13 里 `_default_pg` 和 `_world` **两个候选属性
    都不存在**，靶子本身找错了。回顾性解释了此前所有"三无"失灵。修正：
    用 `inspect.getsource(c10d.is_initialized)` 找到真实看守变量名再戳。
    **教训：私有 API 的属性名要用 inspect 从运行时源码里读出来，不能凭
    版本记忆猜——猜错时 hasattr 守卫会让补丁变成无声的空操作。**
  - **8e（终局：真靶子 = group.WORLD）**：查 torch 1.13.1 官方源码实锤——
    默认进程组存在 **类属性 `group.WORLD`**（`GroupMember.WORLD` 是独立绑定
    的别名），`is_initialized()` 返回 `GroupMember.WORLD is not None`，
    twice! 守卫同源，`destroy_process_group` 的记账效果 = 两者置 None。
    `_default_pg` / `_world.pg` 是其他版本的布局，与 1.13 无关。修复：
    poke 同时置 `group.WORLD` 与 `GroupMember.WORLD`（保留其他布局兜底）。
  - **8f（barrier key 分叉，_group_count 未重置）**：**现象**——twice! 已灭
    （twice=0），但 7 个 rank 全部在重 init 的 store barrier 超时：干净
    rank 等在 `store_based_barrier_key:1`（worker_count=4），give-up rank
    等在 `key:17`（worker_count=3）——两本账本永不会合。**根因**——barrier
    key 由模块级 `_group_count` 生成；`destroy_process_group` 只在默认组
    分支重置它（torch 源码注释明说重置目的就是"故障恢复后重建进程组时
    各 rank 生成一致 key"）；force-continue 路径绕过 destroy 漏了这个
    重置。修复：poke 补 `_group_count = 0`（面包屑 C++）。**教训：绕过
    库的清理函数时，要把它的全部记账副作用抄全——读一遍该函数源码
    列出副作用清单，逐项复刻。**
- **验证（第十八轮后终版判据）**：① 逃逸标记非空；② etcd barrier 7/7；
  ③ 拆组合计 = 7；④ `rdzv/v_2` 组建；⑤ `PRE: ... pgs leaked` 出现（析构
  拆弹生效）；⑥ `Re-initializing finished` = 7；⑦ 重建期无
  `timing skipped` 卡死（tmp_timer 限时生效）；⑧ **`Model reconfigure
  finished` = 7**（reconfigure 全通——含 op 81 恢复计划构建与
  load_recv_buckets 流级同步两个已修卡点；失败则 `RECOVERY-DIAG` dump
  直接给出维度，注意 json.tool 解析前用 `sed 's/.*\[RECOVERY-DIAG\] //'`
  剥前缀）；⑨ **`starts iteration <故障点+1>` 出现（恢复完成续训）+
  `failed_agents ≤ 1`**；⑩ 正式 100/30 复跑 + audit `valid:true` +
  loss 对比通过；⑪ 连续 ≥2 轮复现。
- **第四轮（str/int 类型坑，实跑 failed_agents=8 揪出）**：零延迟公布初版写了
  `str(rank)`，而 project_pactum 的 etcd 封装是 **JSON 类型化存储**
  （write 遇 str 先 json.dumps，get 用 json.loads 还原）——写 str 读回 str，
  而 agent（api.py:343）和 p2p 逃逸循环（p2p_communication.py:905）都对
  failed_rank 做**裸的 `< 0` 数值比较**，直接 TypeError：p2p 逃逸退化成
  靠异常兜住、agent 监控线程崩溃 → 8 个 agent 全灭（顺带暴露其错误路径的
  陈年 bug：`get_agent_status_event` 属性不存在）。修复：写 **int**（去掉
  str()）。**教训：往共享存储写键前，先看所有读者的期望类型；"格式约定"
  是分布式代码的隐形接口。**

### Bug 4：子进程组 watchdog 默认 30 分钟（已验证修复）

- **现象**：rank 1 超时报错里写着 `Timeout(ms)=1800000`（30 分钟），而脚本明明
  传了 `--process-group-timeout 120`。
- **根因**：`--process-group-timeout` 只作用于 `init_process_group` 建的**主**进程组；
  `mpu/initialize.py` 里 13 处 `torch.distributed.new_group`（DP/embedding/alive 组）
  没传 timeout，吃到 NCCL 默认 1800s。卡集合通信的 rank 即使最终能被 watchdog
  解救，也要先等满 30 分钟——恢复耗时完全失真。
- **修复**：`pretrain_gpt.py` 入口 monkeypatch `torch.distributed.new_group`，
  默认 timeout=120s（环境变量 `LIVEPIPE_PG_TIMEOUT` 可调）。
- **验证**：过夜 run 的日志中同一 ALLREDUCE 超时从 `Timeout(ms)=1800000` 变为
  `Timeout(ms)=120000`，实测 127s 开火。跑 Fig14（恢复耗时）类实验时可把
  `LIVEPIPE_PG_TIMEOUT` 再压到 30s。

## 二、环境与工具链坑

### Bug 5：容器重建丢 pip 包（AIS 平台特性）

- **现象**：workflow 号一变（61950227→62410126→62710085），就出现
  `ModuleNotFoundError: No module named 'etcd'` / `'jsbeautifier'`、etcd 进程消失、
  apex 失效。
- **根因**：平台回收容器，容器层（/opt/conda 的 pip 包、手动 nohup 的进程）归零；
  NAS（/input、/ossfs）不受影响。
- **修复**：`livepipe-exp/init_env.sh` 一键恢复（幂等）：PYTORCH2=0（写入
  ~/.bashrc）→ 装 python-etcd/jsbeautifier/pyyaml → apex 源码重编 → etcd 自启 →
  四项终检。每次重连先看 workflow 号变没变。

### Bug 6：apex 安装三连坑

1. plain `pip install ./` 装出**纯 Python 版**（没有 amp_C）——apex 设计如此，
   扩展必须显式 `--cpp_ext --cuda_ext`；
2. pip 的 `--config-settings "--build-option=--cpp_ext"` 在老 pip 上被**静默忽略**
   （退出码还是 0）——最稳的是 `python setup.py install --cpp_ext --cuda_ext`；
3. **验证姿势**：单独 `import amp_C` 报 `libc10.so cannot open` 是正常的——amp_C
   链接 torch 的 libc10.so，必须**先 import torch**。正确检查：
   `python -c "import torch, amp_C, fused_layer_norm_cuda"`。
   另：源码在 `/input/paper/elaceso/bamboo/external/apex_aceso`（藏得深，
   find 的 maxdepth 要 ≥6）。

### Bug 7：路径硬编码被代码同步覆盖

- **现象**：同步代码后秒败 `failed_agents=8`，logs 指向 `/workspace/Aceso`。
- **根因**：tar 同步是**覆盖合并**，服务器上 sed 改好的路径被本地原始版盖回。
- **修复**：脚本头部改自动探测
  `ACESO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)"`。
- **教训**：一切修改改本地再同步，不要直接改服务器上的文件。

### Bug 8：start_etcd.sh 健康检查竞态

- **现象**：init_env 报"etcd 启动失败"，但随后手动 curl /version 又是通的。
- **根因**：etcd 完全就绪需要几秒，脚本 sleep 2 后单次 curl /health 就判死刑。
- **修复**：重试 15 次（每秒一次）探 /version，任一次成功即算启动。

## 三、小坑速记

| 现象 | 原因 | 处置 |
|---|---|---|
| 补丁打了但 run 还是老症状（六次事故） | 先启动后同步，跑的是旧代码 | `check_patches.sh` 闸门：14 项补丁核对，缺一拒绝启动；启动命令用 `闸门 && nohup ...` 串联 |
| 补丁后 import 报 `SyntaxError: name 'X' is used prior to global declaration` | 用到了函数体深处才 `global` 声明的名字；且**同一名字第二次 global 声明之前有过使用也算**（把声明提前后，深处那条要去掉重复的名字） | 改完必须 `python -m py_compile` 校验——`ast.parse` 抓不到符号表错误 |
| 启动实验后终端"不动" | 输出全在 `$RUN_DIR/agent-gpu*.log`，前台本就静默 | tail 日志文件 |
| 跑完还有大量 DEBUG etcd 日志 | agent 监控线程轮询 + 等 FIN 收尾，无害 | 无视 |
| 同步脚本卡在下载后 | NAS 写几千小文件慢（`syscw` 是写系统调用数不是文件数） | 等；已排除大目录 |
| 冒烟参数没生效 | sed 忘跑/跑错目录 | 每次启动前 `grep ^train_iters/^failure_iteration` 确认 |
| rank≠物理 GPU | rendezvous 按加入顺序分配 rank | 别盯着 4 号卡等它挂 |

## 四、取证工具箱

```bash
# worker 卡点（进程活着才能取证，先别杀）
for p in $(pgrep -f pretrain_gpt.py); do py-spy dump --pid "$p"; done

# 恢复进度判读（按序应凑齐 7 个 rank）
grep -h "before destroy barrier" "$RUN_DIR"/agent-gpu*.log | grep -o "rank [0-9]" | sort | uniq -c
grep -h "destroyed the process group" "$RUN_DIR"/agent-gpu*.log | wc -l

# 逃逸/超时关键词
grep -n "catch failed rank\|redundant send aborted\|Watchdog\|Timed out" "$RUN_DIR"/agent-gpu*.log
```
