# ms-swift 上游借鉴报告：production training 与相关推理基础设施

日期：2026-09-12。性质：源码与提交历史比较；建议候选，不是性能验收或实现计划。

## 结论与顺序

**当前最值得先做的小改进，是限制训练缓存预处理的在途任务数量。** 当前实现一次提交全部样本，问题有明确代码位置；可以先局部改善资源占用，同时保留样本、编码、packing 和 loss 语义。实际 RSS、耗时收益仍需测量。

ms-swift 更值得借鉴的是样本与学习 batch 分离、执行阶段的资源管理、分布式兼容边界及具体回归案例。CoordExp 已有的几何监督、计划步分母、缓存身份和执行模型校验应继续由本地模块负责。

| 顺序 | 候选 | 预期价值 | 成本 | 当前判断 |
| --- | --- | --- | --- | --- |
| 1 | 限制预处理在途任务 | 控制大量 Future 和结果管理的额外占用 | 小—中 | 有具体修改位置；可进入小范围设计 |
| 2 | 借鉴上游 loss 权重/位置/分母反例 | 防止后续优化悄悄改变训练目标 | 小 | 先对照已有覆盖，仅补实际缺口 |
| 3 | 数据选择/混合在准备阶段显式化 | 减少手写混合脚本，保持可复现 cohort | 小—中 | 有新数据组合需求时采用 |
| 4 | rank 内按需读取缓存 | 降低启动时常驻 CPU payload | 中—大 | 先测 RSS 和首步耗时 |
| 5 | rollout → learning 的精确 token 接口 | 支持将来在线学习，减少采样与监督错位 | 中 | 可进入接口决策；不等于已有 RL 授权 |
| 6 | vLLM 并发扩展、同步权重交接 | 扩展推理吞吐及迭代训练能力 | 中—大 | 先完成指定配置的小规模验证 |
| 7 | FSDP2 | 解决复制模型/优化器状态的容量限制 | 大 | 出现容量瓶颈后考虑 |
| 8 | CE kernel、异步 checkpoint、adapter 精度选项 | 针对已测出的局部瓶颈 | 各异 | 条件性候选，不批量开启 |

## 1. 比较基线与更新记录

| 对象 | 路径 | 核对版本 |
| --- | --- | --- |
| 上游 | `/data/ms-swift` | `0673cf75dca7d0b9b608b4a76632fb508ead5076`，2026-09-11，`4.6.0.dev0` |
| production infras | `/data/CoordExp/.worktrees/coordexp-infras` | `70e576f9606c48d322b6c67df9baac78c22fc3f6`，2026-09-11 |

按用户要求获取官方默认分支 `main`。原目录停在 detached `v4.2.2`，发布线与 main 分叉，因此执行 fetch 后切换到 detached `origin/main`，没有生成合并提交或重写历史。当前源码不是已发布稳定版。未执行依赖升级、训练、模型 forward 或 GPU benchmark。

原 `.gitignore` 带 `skip-worktree` 且有本地修改；已备份、三方合并并恢复该标记。备份在 `/tmp/ms-swift-upstream-20260912-2n9rpom0/`，保留 stash `codex-ms-swift-upstream-preserve-gitignore-20260912`。切换后曾核实 `.codegraph/` 存在；报告前复核该目录已不存在。本任务没有执行其删除操作，原因未调查，不将其当前状态描述为已保留。

以下判断依据当前代码。历史记忆仅用于定位和保护原有语义，没有用历史测试结果替代本轮验证。

## 2. 当前最具体的资源优化：限制预处理在途任务

**观察。** [cache_workflow.py:1816](/data/CoordExp/.worktrees/coordexp-infras/src/training/cache_workflow.py:1816) 为全部样本创建 Future，等待全部结果，再恢复原始顺序。已有多进程；缺口是提交规模与结果持有方式。ms-swift 的 [IterablePackingDataset:241](/data/ms-swift/swift/dataset/packing.py:241) 分窗口提交，并按原始索引回收结果；2026-07-20 的 `e7a5a7cb4` 进一步修复了 split 编码组的保留。

**建议。** 在现有 cache workflow 内限制 outstanding futures，及时回收，继续按原始 index 排列结果。第一步保留最终 tuple 与缓存格式；这只限制调度开销，**不会把整个流水线变成常数内存 streaming**。全程有界还需要增量存储编码结果。

**责任与边界。** 由 `src/training/cache_workflow.py` 负责进程、窗口和失败传播；数据顺序、样本成员、编码结果及缓存身份保持原契约。不要复制上游跳过错误样本的行为。

**最小验证。** 同一组真实输入在一个/多个 worker 下得到相同编码、pack 成员和监督位置；刻意延迟或失败一行，确认不换样本、不丢失败；测量在途任务上限、代表性规模 RSS 与总准备耗时。只有这些测量支持资源收益后，才能宣称优化有效。

## 3. SFT：借鉴反例，保留本地目标函数

近期上游修复集中体现了一个风险：同一监督信号经过权重、padding-free、并行切分及累积后，loss 和梯度可能不再代表同一目标。

- 2026-09-10 `80cc4f4ed`：custom CE 丢失 `loss_scale`。其 [回归测试](/data/ms-swift/tests/train/test_cross_entropy_loss.py:12) 同时比较非均匀/零权重下的 loss 与梯度。
- 2026-09-04 `32f17c248`、2026-09-07 `1ac8cc047`：SP 权重/评估缩放及 channel label 对齐。
- 2026-09-11 [0673cf75d](https://github.com/modelscope/ms-swift/commit/0673cf75dca7d0b9b608b4a76632fb508ead5076)：修复 Megatron GRPO 的部分目标在 optimizer-step 窗口中的 token 分母及额外平均，并修复 GKD 相关对齐。它不是“所有 GRPO 共用的一个补丁”，也不证明本地存在相同 bug。

本地 [LossRunner:781](/data/CoordExp/.worktrees/coordexp-infras/src/losses/runner.py:781) 已统一计划步分母，并在 [runner.py:605](/data/CoordExp/.worktrees/coordexp-infras/src/losses/runner.py:605) 分开语义 loss 与 DDP 梯度补偿。

**可吸收内容。** 对照现有测试，补上真正缺失的非均匀长度、零权重、错位位置、不同 rank/micro-batch 划分反例。以后增加 RL loss，也应通过这一分母责任边界。不能把当前 segment-balanced 目标直接换为上游 global token mean。

**最小验证。** 在声明的同一全局目标下比较单批参考与累积/分布式执行的参数梯度；不只比较日志中的 scalar loss。责任在 `src/losses`、`src/supervision`，成本小，科学语义保持不变。

## 4. Data/template 管理：借鉴阶段分离，保留显式语义

ms-swift 的 [DatasetMeta](/data/ms-swift/swift/dataset/dataset_meta.py:173) 区分数据来源、revision、subset、preprocess；[TemplateInputs](/data/ms-swift/swift/template/template_inputs.py:52) 和 [TemplateMeta](/data/ms-swift/swift/template/template_meta.py:33) 区分输入与格式；[ConcatLossScale](/data/ms-swift/swift/loss_scale/base.py:202) 自 2026-06-11 `85d98d605` 起支持策略组合。

本地并非缺少分层：[RawExample 入口](/data/CoordExp/.worktrees/coordexp-infras/src/data/examples.py:227) → [render_example](/data/CoordExp/.worktrees/coordexp-infras/src/templates/renderer.py:115) → Qwen encoding → TokenAtom 已有明确责任。当前 [DataConfig](/data/CoordExp/.worktrees/coordexp-infras/src/config/models.py:187) 是单一 train 路径与可选 eval；template 明确保留几何/描述顺序和对象排序。

**可优化方向。** 有多来源训练需求时，先在数据准备阶段产出显式选择/合并 manifest：来源与 revision/hash、保留的 row IDs、顺序、采样种子、重复次数及过滤原因。仍让训练消费一个确定的输入，避免先引入运行时 dataset registry 或随机 interleave。

**template 的借鉴方式。** 保留“格式渲染”和“哪些 token 参与哪种监督”的分离。将来出现多个真实 mask 消费者时，可在现有 typed spans/TokenAtom 上组合选择规则；不要以字符串正则重新识别坐标或对象位置。`ConcatLossScale` 是组合思路，不是本地 loss 分母或对象权重的替代品。

**限制。** 上游 `strict=False`、自动列映射、重复采样、数据与 dataloader 两层 shuffle 都服务于通用训练便利性；直接继承会改变 cohort 或顺序。新混合比例及 stop/repeat 规则由研究目标决定。最小验证为 manifest 成员/顺序与实际读取一致，且 renderer → 编码 → 监督位置对齐。

## 5. Packing/cache：已有能力与值得测量的下一步

2026-07-06 [003b8c3e1](https://github.com/modelscope/ms-swift/commit/003b8c3e1) 新增 sequential packing；上游默认仍为 binpack，且全局 sequential 边界要求单 packing worker。本地已经默认 [source_order_next_fit](/data/CoordExp/.worktrees/coordexp-infras/src/config/models.py:244)，并有 window/online binpack 和显式 [FA2 segment 参数](/data/CoordExp/.worktrees/coordexp-infras/src/qwen/fa2.py:762)。无需重新吸收这些功能。

第二个资源候选是**按需读取已验证的 rank 缓存**。本地 [pack_cache.py:609](/data/CoordExp/.worktrees/coordexp-infras/src/training/pack_cache.py:609) 已按 rank 跳过无关 chunk，但仍把需要的 micro-steps 保留为完整 tuple；[session.py:1850](/data/CoordExp/.worktrees/coordexp-infras/src/training/session.py:1850) 后续继续消费完整集合。上游 [PackingDataset:176](/data/ms-swift/swift/dataset/packing.py:176) 保存成员索引，取样时才读取对应记录。

只有 RSS 或首步等待确实成为问题时，才考虑在现有消费者后使用有界 chunk cache。中—大成本来自现有 session/provider 的 materialized-sequence 假设。验证必须包括计划顺序、重复/恢复游标、chunk 损坏拒绝、loss 分母和梯度，以及额外 I/O 的代价。上游 `IndexedDataset` 未发现当前 packing 调用方，不把其 mmap 实现当作已使用的主路径。

**进程启动方式另列条件项。** 上游 2026-08-26 `af807e3c2` 在 CUDA/distributed 已初始化时选择 spawn；本地 [cache_workflow.py:1894](/data/CoordExp/.worktrees/coordexp-infras/src/training/cache_workflow.py:1894) 使用 fork，但常规准备是 model-free，分布式训练消费预备缓存。尚未证明常规路径存在死锁。先证明可达的危险调用，再决定早拒绝或支持 spawn；不能为了跟随上游就重做进程框架。

## 6. Rollout → learning：最值得为新能力借鉴的接口

当前 production learner 是 SFT；[GenerationPolicy](/data/CoordExp/.worktrees/coordexp-infras/src/inference/backend.py:35) 也是确定性推理约束。在线 RL 是新增能力，不应被描述为“补齐已有 GRPO”。

上游 [OnPolicySample](/data/ms-swift/swift/rl_core/data.py:27) 保存 request、实际 token、mask、采样 logprob、结束原因；[GRPOBatch](/data/ms-swift/swift/rl_core/data.py:312) 再组织 learner 所需信号。2026-08-31 [30e75beb3](https://github.com/modelscope/ms-swift/commit/30e75beb3) 专门修复多轮 token-in/token-out 一致性。

**推荐接口方向。** 采样事实与学习选择分开：inference 交付精确 prompt/continuation IDs、执行策略和权重身份；supervision/packing 将其中明确选择的位置映射到现有 [TokenAtom](/data/CoordExp/.worktrees/coordexp-infras/src/supervision/tokens.py:22)。reward、advantage、credit 范围和归一化由实验所有者选择，不能藏进通用模板。

最小验收是一个真实图文 trajectory，覆盖不满足 decode/re-tokenize 往返一致性的 token、屏蔽前缀、EOS/长度停止；packing 后逐位核对 token、mask、logprob，错一位必须失败。中等成本，状态为可进入接口决策。

**后续同步权重交接。** 上游 [rollout_mixin.py:643](/data/ms-swift/swift/rlhf_trainers/rollout_mixin.py:643) 提供按步更新权重与 prefix/encoder cache 失效；2026-08-26 `43e3b4886` 修复失败时先释放 vLLM 再恢复 learner 的顺序。可以借鉴这些责任，但必须验证本地 DoRA magnitude 与新增 embedding/LM-head 行。最小真实切片为两个 optimizer update，每次交接后与动态 HF 对照，重用旧图文 prompt 检出陈旧 cache，并注入一次生成失败。

异步/Ray 暂不建议引入。2026-09-10 [912eb67d8](https://github.com/modelscope/ms-swift/commit/912eb67d8) 的跨 rank 失败传播修复值得保留为未来验收案例；当前尚无同步 rollout 瓶颈或可接受权重陈旧度的证据。

## 7. Distributed training、vLLM 与 DoRA

### FSDP2：容量驱动，不能只加配置开关

本地 [runtime admission](/data/CoordExp/.worktrees/coordexp-infras/src/runtime/train_runtime.py:45) 只接受 `NO/MULTI_GPU`。上游 [FSDP2 初始化](/data/ms-swift/swift/arguments/sft_args.py:285) 涉及模型加载前环境、device map、CPU RAM、checkpoint 兼容；2026-08-28 `b092b8f69` 修复了初始化时机。

若复制的参数/优化器状态成为实际瓶颈，FSDP2 是真实能力候选。责任跨 `runtime`、Qwen loading、optimizer 和 checkpoint，成本大。至少用真实两 rank 做 update → save → resume，并核对下一步状态和声明的全局目标；不把拓扑更换当作已证明的加速。

### vLLM：已有调度 batching，下一步是限定配置扩展

上游 [VllmEngine](/data/ms-swift/swift/infer_engine/vllm_engine.py:130) 暴露更多并发、TP/PP 和异步参数。本地 [backend 配置](/data/CoordExp/.worktrees/coordexp-infras/src/inference/vllm_backend.py:807) 固定每 worker 的 TP=DP=1，[qualification](/data/CoordExp/.worktrees/coordexp-infras/src/inference/vllm_qualification.py:24) 限定 batch 档位 `{1,4}`。

本地 [pipeline.py:614](/data/CoordExp/.worktrees/coordexp-infras/src/inference/pipeline.py:614) 构造 rank 的请求集合并一起交给引擎；不能据数据分块就声称缺少 continuous batching。更直接的候选是：在当前已验证配置上选一个更高并发档，与 4 比较相同请求的 token/output 一致性、吞吐、wall time 和峰值显存。重新验证该配置；不要直接套用上游默认 256，也不必先建服务层。

### DoRA：学习精度/目标发现经验，保留本地组合模型契约

上游 2026-08-03 `05b5685a7` 在 [PEFT 适配层](/data/ms-swift/swift/tuners/peft.py:147) 防止显式低精度 adapter 被自动提升。若新增 adapter dtype 选项或升级 PEFT，应记录实际 A/B/magnitude dtype，并验证 train → save → HF reload → compose；当前没有证据证明默认 dtype 是 bug。

本地 [DoRA setup](/data/CoordExp/.worktrees/coordexp-infras/src/adapters/dora.py:807) 已发现实际 tower targets、拒绝空集合并核对 A/B/magnitude 训练面。上游普通 `LoRARequest` 或通用 merge-and-unload 不证明支持本地坐标 embedding delta。继续以 [动态 HF 组合契约](/data/CoordExp/.worktrees/coordexp-infras/openspec/specs/infra-base/spec.md:66) 和 [composition 对照](/data/CoordExp/.worktrees/coordexp-infras/src/inference/execution_model_composition.py:23) 判断正确性；无需移入通用 tuner registry。

## 8. 代码分层与管理方式

可以效仿上游把 dataset、template、tuner、infer engine、rollout sample 和 learner 分开命名与归属。对本地而言，重点是继续保持消费者清楚、策略显式，而非重排目录。

本地 [training pipeline](/data/CoordExp/.worktrees/coordexp-infras/src/training/pipeline.py:1) 已缩为 156 行 facade，委托 execution plan、control plane、cache workflow 和 session。上游当前 `Template` 为 2383 行、`GRPOTrainer` 为 2598 行，且存在多层 mixin；这些规模是源码观察，不是质量定论，也不是值得照搬的组织目标。

建议沿用以下管理办法：

1. 一个概念保留一个责任模块；新增 RL 样本边界不要改变 Qwen/geometry/loss 的原责任。
2. 相同功能先核对现有实现及真实调用者，再决定提取共享函数。当前单模型家族无需引入数百模型注册体系。
3. 在依赖升级或新后端决策中，把上游 commit、影响路径和可重现反例写进现有变更记录；不建立独立兼容管理平台。
4. CPU schema/fixture 检查与真实多卡、保存恢复、HF/vLLM 对照各自回答不同问题。源码支持不等于运行验收完成。

## 9. 暂缓的性能选项与停止条件

- **Liger/fused CE：先 profile。** 上游 [seq2seq_trainer.py:137](/data/ms-swift/swift/trainers/seq2seq_trainer.py:137) 明确提示自定义 loss、权重、channel 或 SP 会使 Liger CE 不生效。本地已选择需要的 lm-head 位置，并让坐标辅助项共享 logits；不能宣称一个开关即可加速。候选 kernel 需逐项 loss/梯度等价，再测实际一步显存与耗时。
- **异步 checkpoint：先测保存停顿。** 本地 [checkpoints.py:75](/data/CoordExp/.worktrees/coordexp-infras/src/artifacts/checkpoints.py:75) 已有分 rank、分阶段发布。只有保存时间显著影响训练时才值得新增异步状态；未完成保存不能成为 `last`。
- **AdamW 优化：先用已有配置面。** 当前 optimizer 透传 kwargs；核对现有 `fused=True` 可用性即可，不需要新的 optimizer registry。

本报告在“当前代码支持候选排序”处停止。没有把源码阅读、已有测试文件或上游性能描述当作本地性能实测。下一项具体工作只需围绕所选候选完成最小反例和测量，无需重新做整轮架构审计。

配套报告：[research probing 与手术实验](/data/CoordExp/.worktrees/coordexp-infras/research/investigations/ms-swift-upstream-comparison-2026-09-12/research-probing-report.md)。
