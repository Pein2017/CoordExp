# ms-swift 上游借鉴报告：research probing 与手术实验

日期：2026-09-12。性质：当前源码比较与候选排序；没有启动新实验、改动科学目标或验证模型效果。

## 结论与推荐顺序

**最先值得改善的是新实验的输入准备便利性：指定两三行样本，就能检查 prompt、token、图像 grid、监督位置，再把相同输入交给现有 native 执行。** 手术实验的另一项高价值候选是精确选择很小的可训练参数集合，并在执行前显示实际命中的参数。

research-probes 已有比 ms-swift 通用训练接口更适合精确干预的基础。应学习上游的少量输入/样本/参数选择接口，让现有普通 Python 模块更方便复用；无需把小 probe 接入完整 trainer 或重新建设实验运行框架。

| 顺序 | 候选 | 用户能更容易做什么 | 建议归属 | 成本/状态 |
| --- | --- | --- | --- | --- |
| 1 | 准备并检查指定样本的薄函数 | 两张图比较 prompt/顺序/token，不先加载权重 | shared probes，复用现有 src | 小；可进入接口决策 |
| 2 | 明确的 tiny-cohort 选择 | 指定 IDs 或固定种子抽取少量样本，两臂共用同一选择 | 首个实验内 | 很小；有新 cohort 时采用 |
| 3 | 小范围可训练参数选择 | 只更新指定层的 A/B、magnitude 或其他明确参数 | 实验内；第二个调用者后共享 | 小；训练面属于新的实验选择 |
| 4 | 现有 trajectory 的统一读取视图 | 同一分支直接检查、replay、计算目标，减少字段转接 | dora_owner_learning | 小—中；保留原 artifact 格式 |
| 5 | 离线分支诊断表 | 一眼看到 stop、reward、owner gains/losses、错误与可用分数 | 实验 reducer | 小；不增加模型调用 |
| 6 | Source → candidate → Source 的局部切换 | 在同一 loaded model 上比较少量参数变化 | 首个需要它的实验 | 小—中；先证明现有恢复方式有重复负担 |
| 7 | sampler/replay/learning 分数分离 | 区分执行不一致和真正的更新效果 | 实验诊断消费者 | 小—中；只能使用实际保存的数据 |
| 8 | adapter 低精度或 ReFT 式 learned residual | 新的成本/干预实验 | 实验内 | 条件性；ReFT 适配暂 HOLD |

## 1. 范围与当前状态

- 上游：`/data/ms-swift`，`0673cf75dca7d0b9b608b4a76632fb508ead5076`，2026-09-11，`4.6.0.dev0`。
- research-probes：`/data/CoordExp/.worktrees/research-probes`，`ef6d44d11`，2026-09-12。
- 同时参考 production infras 的责任边界；本报告不建议让研究实验自动继承生产推理的确定性、trace 或配置目录限制。

research-probes 有正在进行的本地工作：dense-enumeration 的 compass/index 修改，未跟踪的 `probes/parallel_owner_research/`、对应实验目录及 `tests/test_training.py`。本轮只读这些内容，没有改写、移动或接管。

特别是 [shared-execution/interface.md](/data/CoordExp/.worktrees/research-probes/research/investigations/qwen3-vl-dense-enumeration/experiments/2026-09-12-parallel-owner-research/shared-execution/interface.md:8) 与 [prepare_packet](/data/CoordExp/.worktrees/research-probes/probes/parallel_owner_research/training.py:154) 已提供 literal records、steps、weights、denominators 的执行接口。该记录报告了 qualification，但本轮没有重跑，也不据此授予运行验收。**不能把“缺少通用小训练执行器”列为新发现。**

活动文件快照（2026-09-12 07:16:53 UTC；这些文件不由上述 HEAD 完整标识）：

| 文件 | SHA256 |
| --- | --- |
| `probes/parallel_owner_research/training.py` | `aee2a58768376fb732e4abf837a1a976bf0765d5fb4d8c5fb1c60d338b17e429` |
| 对应实验 `shared-execution/interface.md` | `d96d683a00d6518fe4346de6affb44d31be00ca84e67e436ab3e2efd44fa2fd3` |

## 2. 已有能力：应继续使用的入口

| 需求 | 当前模块 | 已有关键性质 |
| --- | --- | --- |
| 原始图文准备、精确历史 replay | [src/qwen/native.py](/data/CoordExp/.worktrees/research-probes/src/qwen/native.py:240) | literal token IDs、历史位置重建、causal logits 对齐 |
| 带前缀的自然 continuation | [src/qwen/generation.py](/data/CoordExp/.worktrees/research-probes/src/qwen/generation.py:180) | 显式 token 扩展、剩余预算、EOS/seed；raw 与 policy trace 分开 |
| 捕获输入或指定 hidden rows | [src/qwen/inspection.py](/data/CoordExp/.worktrees/research-probes/src/qwen/inspection.py:54) | 独立张量快照；输入可用于后续 functional replay/autograd，原图梯度不被保留 |
| 局部 residual 替换 | [ResidualPatch](/data/CoordExp/.worktrees/research-probes/probes/logit_lens/causal.py:239) | 指定位置、recipient 校验、仅触发一次、非目标位置不变、hook 清理 |
| Qwen DeepStack 干预解释 | [logit_lens/base.py](/data/CoordExp/.worktrees/research-probes/probes/logit_lens/base.py:287) | 明确边界时机，避免把注入前后 hidden 混为一谈 |
| DoRA 采样、小更新、owner credit | [dora_owner_learning](/data/CoordExp/.worktrees/research-probes/probes/dora_owner_learning/README.md:1) | 独立 CE/RLOO/coordinate/full-action 目标与 native replay |
| 固定前缀/行交换与自然后缀 | [branch_bridge](/data/CoordExp/.worktrees/research-probes/probes/dora_owner_learning/branch_bridge.py:137)、[source_rweak_row_cross](/data/CoordExp/.worktrees/research-probes/probes/source_rweak_row_cross/README.md:1) | 明确 forced 部分和 free suffix；执行与离线解释分离 |

这些是当前源码提供的能力；本轮没有重新确认各实验的全模型数值、科学结果或 held-out 效果。

## 3. 最优先的便利性改进：准备并检查几行输入

**上游可借鉴。** [Template.encode](/data/ms-swift/swift/template/base.py:660) 提供单样本编码与长度/输入信息；[cached_dataset export](/data/ms-swift/swift/pipelines/export/cached_dataset.py:17) 分开无权重准备与训练。值得借鉴的是可单独调用的输入准备阶段。

**本地实际摩擦。** 已有 [build_request](/data/CoordExp/.worktrees/research-probes/probes/dora_owner_learning/runtime.py:35)、[renderer](/data/CoordExp/.worktrees/research-probes/src/templates/renderer.py:124) 和 [CPU preflight](/data/CoordExp/.worktrees/research-probes/probes/dora_owner_learning/preflight.py:17)，但便捷入口分别绑定 Source256 的固定 hash/数量，或 Logit Lens 的历史阶段与前序证据。一个新的两图 prompt 实验不应被迫采用旧实验的整套配置。

**最小接口方向。** 用一个普通函数组合现有调用，输入 resolved config 与明确选中的 rows，返回 rendered text、native model inputs、token 长度、spans、图像/grid 与输入身份。不创建新的 template schema，不把所有实验参数塞进泛化 context，不加载模型权重。历史 preflight 继续保留自己的完整约束。

可先减少 [DORA runtime:16](/data/CoordExp/.worktrees/research-probes/probes/dora_owner_learning/runtime.py:16) 与 [Logit Lens runtime:7](/data/CoordExp/.worktrees/research-probes/probes/logit_lens/runtime.py:7) 重复的 config→processor/template 转接。这已有两个真实消费者，足以支持一个窄的 shared-probes 函数。

**收益。** 新实验先对两条真实样本检查 token 和视觉输入，再把这些相同输入送入现有 native forward/replay，减少复制历史脚本和浪费模型加载。

**最小验证。** 两条保存的真实数据与原 helper 得到相同 prompt IDs、grid、媒体身份和 spans；assert 未加载模型；修改 prompt 或排序后相应身份必须改变。验收这些行为后停止，不扩建新的 CLI/plugin 系统。

## 4. Tiny cohort：方便选择，同时保留成对比较

上游 [sample_dataset](/data/ms-swift/swift/dataset/utils.py:18) 与 [DatasetSyntax](/data/ms-swift/swift/dataset/dataset_syntax.py:14) 把来源与子集数量显式化。本地 [JSONL reader](/data/CoordExp/.worktrees/research-probes/src/data/jsonl.py:15) 的通用限制主要是前若干行；[Human13 panel](/data/CoordExp/.worktrees/research-probes/probes/human13/panel.py:53) 则刻意绑定固定 cohort。

为新实验可先写实验内 `select_examples`：接受显式 IDs，或者 count+seed，返回实际有序 IDs 与来源绑定；两臂共享这一次选择。第二个实际消费者出现后再共享，不改动已有 Human13/Source256。

**不复制上游默认语义。** `sample_dataset` 在数量大于总体时会重复；`LazyLLMDataset(strict=False)` 可用另一行替换编码失败样本。对两图/八图实验，这直接改变分母和处理对象。本地选择应拒绝未知/重复 ID 与未经声明的超额抽样，坏行作为明确失败。

最小验证用三行数据覆盖顺序、固定种子、未知/重复 ID、超额数量，并确认 A/B 两臂实际成员完全相同。成本很小，不需要模型。

## 5. 手术训练：显式指定很小的可训练面

ms-swift 的 [tuner_args.py:116](/data/ms-swift/swift/arguments/tuner_args.py:116) 提供 regex、parameter 与冻结/训练选择器。本地通用 [DoRA setup](/data/CoordExp/.worktrees/research-probes/src/adapters/dora.py:824) 主要是 all-linear 加 tower 选择，而实际模块名已经进入 PEFT 并被验证。

**可吸收的接口。** 为某个具体实验选择一个 decoder block 的指定 q/v/o 模块，或该层现有 DoRA magnitude；执行前输出展开后的参数名、数量、dtype、总标量数。优先使用现有参数和公开 PEFT 接口。

已有 all-layer Source adapter 时，应该冻结不选中的既有参数，保留原 Source policy；不能换成一个不同结构的新 adapter 后仍称“同一个 Source”。本地 [bind_magnitude_surface](/data/CoordExp/.worktrees/research-probes/probes/human13/magnitude_qp.py:169) 已示范冻结全模型、绑定语言 magnitude、核对数量与身份，是更合适的起点。

**责任。** 第一版实验内；第二个相同需求后才提取 shared-probes selector。只有成为正式支持的训练面时才扩展 infras 配置。

**收益与风险。** 显著缩小需要检查的改动集合，可能减少 optimizer state；但不保证 transformer forward/backward 成本按可训练参数比例下降。选择训练面本身是实验干预，不能当透明性能优化。

**最小验证。** 一个真实图文前缀，先确认零更新 Source 一致，再更新一步：只有选中张量改变，冻结张量 hash 不变；空/错误命中拒绝。到此能回答接口正确性，不表示 owner recovery 或泛化变好。

## 6. Trajectory 读取视图：减少 generation → replay → learning 的字段转接

上游 [OnPolicySample](/data/ms-swift/swift/rl_core/data.py:27) 把实际生成的 token、mask、finish reason 和 sampler logprob 放在单样本记录中，之后再 collate。2026-08-31 [30e75beb3](https://github.com/modelscope/ms-swift/commit/30e75beb3) 修复多轮精确 token 传递，是这一边界的实际反例。

本地 [sample artifact](/data/CoordExp/.worktrees/research-probes/probes/dora_owner_learning/sample.py:185)、[plan action 构造](/data/CoordExp/.worktrees/research-probes/probes/dora_owner_learning/prepare.py:452)、[OwnerOutcomeScorer](/data/CoordExp/.worktrees/research-probes/probes/dora_owner_learning/owner_outcome.py:39) 已保存/消费丰富信息，但各消费者仍需转接不同字段。

**最小建议。** 在 `dora_owner_learning` 内给现有记录提供一个只读投影或小 dataclass，让同一分支暴露：prompt/action token IDs、prefix/action split、EOS 是否参与、loss positions、坐标位置、stop reason、执行/model/media 身份以及实际存在的 score。保留原 bank/plan 文件与原 objective；不把历史 artifact 重写为新的统一格式，也不引入完整 GRPOBatch。

**验收。** 保存记录的 CPU round-trip 精确保留 token、EOS、位置和来源 hash；改一个 token、mask 或媒体身份必须被真实 replay/消费边界拒绝。收益是更快检查一个分支并复用现有 scorer，不是增加新的 rollout 策略。

**不能照搬的兼容行为。** 上游 [build_rollout_logps](/data/ms-swift/swift/rlhf_trainers/utils.py:2017) 对缺失/不匹配可能返回 None，并在多出一个 logprob 时截掉最后一项。这不适合作为手术实验的 token/EOS 身份证明；应保留本地精确对齐检查。

## 7. 两类轻量离线诊断

### 7.1 将执行差异与学习变化分开

上游 [GRPOBatch](/data/ms-swift/swift/rl_core/data.py:312) 分开 rollout、old learner、reference logprob；[loss diagnostics](/data/ms-swift/swift/rlhf_trainers/grpo_trainer.py:984) 比较 old learner 与 rollout，而 current/old 更新比率是另一个量。

本地 [ContinuationTrace](/data/CoordExp/.worktrees/research-probes/src/qwen/generation.py:63) 已区分 policy 与 raw。可为未来明确保存 trace 的采样增加一个小 reducer，分别报告：

1. sampler → 同一 checkpoint replay 的逐位置差异，用于定位执行/采样变换不一致；
2. 同一 token 与前缀下，更新前 learner → 更新后 learner 的差异，用于观察更新效果。

当前 [sample_one](/data/CoordExp/.worktrees/research-probes/probes/dora_owner_learning/sample.py:124) 使用 `trace="none"`；历史 bank 没有的数据不能补写为观察值，也不应为所有小实验强制开启昂贵全量 trace。保留缺失值、采样变换、checkpoint 与 mask 身份。

先用已知相同数组、单位置变化和 mask 反例验证 reducer。真实轨迹来自 forced/greedy 分支时，称作条件分数差异，不能直接命名为 on-policy KL 或因果学习效果。不自动添加 importance-sampling/clipping 或改变目标。

### 7.2 一张能解释分支失败的表

上游 [to_reward_row](/data/ms-swift/swift/rl_core/data.py:135) 与 [compute_rewards_per_func](/data/ms-swift/swift/rl_core/grpo_algorithm.py:17) 提供轻量样本视图与独立 reward 列。本地 [_reward_for_rollout](/data/CoordExp/.worktrees/research-probes/probes/dora_owner_learning/prepare.py:340) 已有更适合检测研究的 owner refs、invalid/parser-dropped 数量和匹配证据。

建议实验 reducer 直接读取保存 artifact，投影 image/seed/arm、token 数、stop、现有 reward、owner gains/losses、invalid/dropped，以及可用的分数诊断。沿用当前匹配规则、分母与几何逻辑；不新增 reward、不重跑生成，也不建 dashboard 服务。

最小验证是在已有小样本 artifact 上复算并与冻结的 reward/匹配计数相同，missing 保持 missing。强制前缀成功与自然 owner retrieval 必须保留不同列及不同解释。

## 8. 复用一个 frozen base 做小分支：选择与恢复必须精确

上游 [Part](/data/ms-swift/swift/tuners/part.py:35) 体现“只复制指定模块”的思路，[named adapter activation](/data/ms-swift/swift/tuners/base.py:624) 支持切换。其价值是把基座与候选参数分开管理。

本地 [magnitude_finite.py:334](/data/CoordExp/.worktrees/research-probes/probes/human13/magnitude_finite.py:334) 已有 snapshot/restore 和张量 hash，也有不加载模型就物化 adapter delta 的路径。没有证据说明现有实验普遍重复加载模型。

只有下一实验确实重复写恢复逻辑时，才增加实验内 context：Source → candidate → Source，共享一个 loaded model，保存实际 active adapter 集合与参数值，并在异常后精确恢复。adapter-only 切换优先用 PEFT 公开 API；局部功能性对照可复用 captured inputs/functional replay。各分支的 optimizer state 分离，权重改变后不能复用旧 KV cache。

不能直接复制上游 [disable_adapter](/data/ms-swift/swift/tuners/base.py:616)：它最终激活所有已注册 adapters，而非先前集合。`Part` 复制完整模块也可能比少量 DoRA delta 更昂贵。

验收为真实精确前缀上的 Source/candidate/Source logits、冻结张量 hash、adapter 集合及异常恢复。成本小—中；它是便利性候选，不是已测出的节省。

## 9. 两项需要独立实验问题的选项

### Adapter dtype：成本旋钮，不修改现有 FP32 实验

2026-08-03 `05b5685a7` 在上游 [PEFT 层](/data/ms-swift/swift/tuners/peft.py:147) 保留显式选择的 adapter dtype。本地 [合并路径](/data/CoordExp/.worktrees/research-probes/src/adapters/dora.py:280) 已防止自动 dtype 提升改变执行权重；Source256 学习则有意固定 FP32/SDPA。

若新实验需要低成本迭代，可另设低精度 pilot，核对 A/B/**magnitude** 实际 dtype、loss/梯度/更新与 save/reload。上游 dtype 转换并没有显式覆盖全部 DoRA magnitude，不能原样套用。只有实测 memory/time 才能说明收益，精度容差和实验对照必须明确。

### ReFT：借鉴干预描述，当前直接接入 HOLD

上游确有 [ReFT](/data/ms-swift/swift/tuners/reft.py:105)，用 layer/component/low-rank/locations 描述可训练干预；这是已有模块，不是九月新功能。

直接适配有具体障碍：[reft.py:144](/data/ms-swift/swift/tuners/reft.py:144) 重建 forward 的 `base` 时仅传 input_ids/attention_mask，不能保留本地所需的完整视觉与 MRoPE 输入；它还依赖顶层 hidden_size、pyreft 和全局 patch。因此这不是即插即用的 Qwen3-VL 手术入口。

若未来问题确实需要 learned residual transform，可在单个实验内借鉴“真实模块边界、层、token 位置、干预类型/秩”的声明方式，复用本地 ResidualPatch/inspection。最小真实验证是一个图像、一个边界、一个位置：identity 干预一致，非目标 residual 不变，一步可训练干预有效，prefill/cached decode 行为清楚。没有这个研究需求时停止，不先建 shared intervention framework。

## 10. 模块与实验管理建议

| 内容 | 责任位置 | 保留可见的信息 |
| --- | --- | --- |
| Qwen 位置、原生编码、replay、结构捕获 | 现有 `src/qwen` | 真实执行语义、device/dtype、token/grid 身份 |
| 两个以上实验共享的输入组合等机械操作 | shared probes 的普通函数 | 输入/输出及实际调用，不附带旧实验规则 |
| cohort、干预面、reward/credit、归一化、预算、stop | `probes/<方向>` 与其研究记录 | 实验决定全部显式 |
| 当前小训练 packet 执行 | 已有 `parallel_owner_research` 责任面 | 保留活动所有者；不另建平行 runner |
| 稳定生产支持的 runtime/adapter 扩展 | 经选择后由 infras 负责 | 支持范围、兼容性与真实验收 |

上游 2026-09-04 `37617f106` 修复 external-plugin 在 spawn/forkserver 的加载，说明 import side effects 跨进程会增加成本。本地 `python -m probes.<方向>` 加普通导入已经更适合小实验。只有实际需要多进程注册时才验证 worker 初始化，不为单进程 probe 加 plugin manager。

便捷工作流应是：**明确选择几行 → CPU 检查实际输入 → 一个模型上的指定 forward/replay/小更新 → 离线读取与成对解释**。这描述现有函数的组合，不要求新目录体系、额外 runner 或新的强制检查阶段。

本轮对上述关键判断进行了原始源码复核；未运行测试或模型，未承诺性能或科学提升。每项候选只需其最近消费者的反例与必要的真实切片；其余非阻塞改善不延后小实验。

配套报告：[production training](/data/CoordExp/.worktrees/coordexp-infras/research/investigations/ms-swift-upstream-comparison-2026-09-12/production-training-report.md)。
