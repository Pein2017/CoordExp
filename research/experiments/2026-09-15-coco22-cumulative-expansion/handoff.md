# 接手：11→22 图累积扩展（压缩上下文用）

## 先读这四处

1. 当前用户指令及工作区 AGENTS.md。
2. [本阶段协议 unit.md](unit.md)：研究问题、用户裁决、成功定义、JSONL 写回、Source 触发条件和停止点。
3. [当前状态 state.json](state.json)：唯一维护的生命周期入口。
4. [上一轮已验收结果](../2026-09-15-coco227-ce-normalization/results.md)。需要更早起点信息时才读 [A/B 结果](../2026-09-14-training-set-completion-curriculum/dual-start-results.md)。

本文件是接手导航，不是第二套授权或状态账本。用户要求先产出 handoff，随后压缩上下文；本次仅核对事实和更新研究文档，没有选新图、改标注、启动训练或读回。

## 当前决定：已无待用户拍板的方向问题

- 主线从11图扩为22图，旧11图全量 replay，Sample 等权，从**最新 CE 实验的 S 臂最终 +256**接着训练，fresh optimizer。
- 用户明确允许已核实、范围内的额外真实对象；必须逐对象视觉审核。新 valid/verified unlabeled owner 必须写回对应图像 JSONL 的 `unlabeled`，不能只写报告。
- 用户取消按小时设置的实验时长上限，并要求尽可能8卡、大吞吐。2h、5h、80GPU-hour 都不是现授权边界。256 full-cohort updates 仍是固定实验剂量。
- 主臂充分完成剂量后失败，且排除教师/执行/截断问题，才启动相同22图银行上的 Source joint-fit；用户已预授权此条件分支，无需再问。
-19个历史类别未决对象继续单列，不挡扩图。不要把历史旧 mask 机制问题、held-out 或第三个 CE 变体重新变成前置门槛。
-新教师身份、22张图完整清单、实际执行配置、annotation 输出版本都还没有建立。它们是接手工作，不是当前 blocker；按协议自主准备并由 lead 把关。

## 最短接手动作

1. 核对本 checkout、现存 workers 和 exact owned runtime；保护 dirty/untracked 工作。选图前检查专项拟合历史，不能只因不在旧11 ID中就断言没训练过。
2. 从同来源 COCO train2017 做候选清单和粗难度配对，冻结新增11图身份。排除近重复；不用训练成败筛图。
3. 为新图建立可信对象银行：GT + 探索候选 + 原图/overlay/crop 视觉审核 + lead 裁决。按 `unlabeled` 原规范保存补标、版本及 provenance。原227 teacher token/mask/box/order/EOS 不变。
4. 实现本阶段22图/8卡入口及允许 verified extras 的双层验收；不要直接运行硬编码11图/4卡/7200s的旧入口。
5. 用最小真实切片验证 global22 reduction、checkpoint冷载、readback调度/落盘/恢复，并测吞吐。随后冷读22图 step0，再冻结并执行主臂256 updates；所有22图每步参与梯度。
6. 按0/8/16/32/64/128/256回读、逐图匹配和视觉审核。运行中新增 owner 更新 annotation 版本，不刷新当前冻结教师/评估分母。
7. 主臂成功即结束本单元；满足条件才运行 Source。完成后由 lead 重放关键评分/身份/计数再关闭；不要擅自继续44/64图。

## 精确模型与数据身份

工作区：`/data/CoordExp/.worktrees/research-probes`，分支 `research-probes`。
本次核对 HEAD：`44ed05199c7b5257814c88753f32e8aec517ed59`。
这是 dirty checkout，HEAD 不能代表全部有效实现；关键 probe 与测试还含 untracked 文件。用户要求忽略无关 dirty changes，意为保留它们，不是 reset/clean/stage。

### 新主臂起点，务必避免混淆

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco227-ce-normalization/trial-v1/S/training/checkpoints/step-00256/adapter`

Adapter fingerprint：`0a2d96f2f4a8c89c8e3db5cc743ebf4e4a22e7656d5a70437e68c66b59441a87`。
它是第二轮 Sample 臂最终参数；不要误用该轮训练 manifest 中的 `source_adapter`，后者指向上一轮 A-final256。

参考训练配置：
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco227-ce-normalization/trial-v1/S/training-manifest.json`。
继承配方时显式替换 source adapter，并重建 optimizer；不恢复旧 AdamW。

Conditional Source：
`/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444/adapter`。

两者共用原始2444配套冻结 delta：
`/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444/special_token_embeddings`。

底座：`/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`。
当前记录显示 S 与原 B 的 model、embedding_delta、template、generation 配置相同；差异是 adapter/run。新入口仍须核验实际冷加载身份。

### 冻结旧教师

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco227-ce-normalization/data-v1/bank.json`

File SHA256：`d65126ec827a9cd070b6a06c4557f44e2bb6c5a37b3fb3fc1b1fb6e94d172aac`。
11图、227 owners、2176 active tokens。图像 ID：
`25274,59571,99937,210457,219546,323322,351017,388795,417044,477415,528944`。

原图目录：`/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images/train2017/`。
旧 config 的 input_jsonl 为
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-sft256-dev128-baseline/inputs-v3/train.jsonl`。
这只是已知来源线索，不是已经选定的新11图清单，也不能替代历史曝光排查。

## 已核对的 `unlabeled` 规范

最新已验收原格式补标 JSONL：
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/annotations-with-unlabeled-v4/annotations.jsonl`

SHA256：`c1485dc0bf2e6617c80da6761d5695f6d9fd219ef845288cebec52f40aebdbca`。
11行，原始 `objects` 共169个GT；`unlabeled` 共79个已确认物理 owner，其中60个类别 verified、19个类别 unknown。60个类别明确者含2个非COCO对象，因此可信COCO80总计169+58=227，历史总计248。

每条原记录字段保持原结构：`file_name,height,image_id,images,metadata,objects,unlabeled,width`。
`unlabeled` 是原GT之外的已确认实例，名称不意味着它仍未审核。

具体字段、unknown/null规则、双层成功定义由 [unit.md](unit.md) 维护。实际示例：

```json
{
  "stable_owner_id": "first-fit:new:25274:black-coated-person",
  "bbox_2d": ["<|coord_439|>", "<|coord_658|>", "<|coord_472|>", "<|coord_796|>"],
  "bbox_2d_bins_1000": [439, 658, 472, 796],
  "category_id": null,
  "category_name": "person",
  "desc": "person",
  "class_status": "verified",
  "physical_status": "valid_unlabeled",
  "geometry_status": "reasonable"
}
```

上例只展示核心字段，真正写入必须带完整 reference 与 provenance。
`review-source-index.json` 把每个 stable owner 绑定到实际原图、overlay、crop 和裁决文件；`root-acceptance.json` 是已验收回执。同目录 manifest/README 说明版本和来源。
全历史 catalog：
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-14-training-set-completion-curriculum/target-owners-complete-v6.json`。

代码入口：[unlabeled_annotations.py](/data/CoordExp/.worktrees/research-probes/probes/training_set_completion/unlabeled_annotations.py:390)，
现有测试：[test_training_set_completion_unlabeled_annotations.py](/data/CoordExp/.worktrees/research-probes/tests/research/test_training_set_completion_unlabeled_annotations.py)。
**不要直接运行旧默认 build 当作22图更新器**：它仍硬编码11图、232历史描述、13历史mask，默认ROOT是v3、catalog是v4；当前实际最新导出已是v4/catalog-v6。复用schema和校验，针对新22图路径做最小适配，并保留原已绑定版本。

用户要求是写回实际使用的 JSONL 相应记录；按照历史方式发布下一 annotation 版本、更新当前指针，同时留存旧版本。不能仅增加 detached review JSON，不改变可消费的 annotations.jsonl。没有新裁决时，不要凭空制造新补标。

## 已完成的两轮：只保留决定下一步的事实

- A/B初始化对照：同218对象教师；A已有完整拟合step256，B原2444。
  两臂最终都218/218。A保存点+64已干净，B到+256干净。旧137/218的所谓B基线曾误指N16，已被历史supersession排除，不得复用。
- CE对照：同一A-final256起点、同227教师；Sample/Token均在保存点+16达到干净227/227，后续32/64/128/256保持；旧218全保留、新9全学会。
  +8存在覆盖/输出异常取舍，不能声称精确学习速度相同或统计等价。
- Shared geometry 已共享化，普通/streaming路径及probe调用都有既有验收；本阶段继续用同一image-balanced hinge，不重新设计目标。

上一轮完整 artifacts：
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco227-ce-normalization`。
关键接手证据：`lead-admission-v1/final-acceptance.json`、
`scientific-score-acceptance.json`、`full-update-accounting.json`、
`all-checkpoint-finiteness.json`、`full-readback-validation.json`。
512更新归约重算误差0；12个checkpoint参数/optimizer有限；132条新读回完整且无重复，全部EOS。

## 执行风险与吞吐：保留真实经验

- 旧训练入口 `coco227_training.py` / `coco227_trial.py` 固定11图、4rank/臂和7200s。
  原全局梯度用SUM形成目标；22图/8卡必须验证每图1/22权重，不能平均局部均值。
- 旧microbatch2/3稍快但没过当时冻结的严苛parity gate。mb2归一化CE误差很小、梯度relative-L2过关，失败主要涉及未归一化NLL-sum和norm的共同绝对阈值。
  这不等于已定位 batching bug。可按本轮吞吐要求重新资格验证，提前区分各量的容差；不要事后为某配置放宽门槛。
- 旧readback batch3通过真实token/owner/stop一致性，生成时间比串行减少11.16%。新入口优先复用
  `coco227_readback.py` 的 native batch、media身份、raw batch/per-row回执和missing-only恢复。
- 原controller在训练完成后因 Python无`os.pidfd_open`失败；S8已启动的worker被保留。
  `coco227_recover_readback.py` 用阻塞Popen.wait线程 + completion queue修复，复用11条、补121条后成功。
  **下轮从一开始使用已验证的可移植等待方式，验收实际controller→worker→collection链。**
- 不直接修改旧manifest/hash绑定的 producer 文件。通过小型新probe入口或明确新版本实现22图；不要为此建设通用调度平台。
- 使用bare `python`、named tmux、持久日志/身份/退出回执。随机8卡stress是共享环境常态，直接启动所需GPU工作，只有实际冲突才处理。

推荐复用路径（阅读相关函数即可，不需全仓扫描）：
`probes/training_set_completion/{coco227_data,coco227_evaluation,coco227_training,coco227_trial,coco227_readback,coco227_recover_readback}.py`；
`src/losses/raw_axis_validity_hinge.py`；`src/qwen/native.py`；`src/qwen/generation.py`。
旧evaluator的clean判据要求exact227个valid row，**不能原封不动用作新一轮允许verified extra的成功判据**；保留其冻结匹配/F1读数，再实施已批准双层判定。

## Agent、等待和剩余权限

原package owners：`/root/distributed_training`（训练/调度）、
`/root/teacher_evaluation`（教师/评分）、`/root/batched_readback`（批量读回）。
此前均已交付；恢复前先reconcile可用状态，按上下文可靠性复用或重建，不让已交付worker空等。Lead拥有最终裁决；可以使用native agents分工，不用另开用户侧task。优先Codegraph探索已知符号，图遗漏或stale时用定向原源码。

两个历史monitor `ab43242c-5e67-4782-ab06-b4734eadd492` 与
`88d5cb3b-e6e6-42cf-a56d-09d2c73591c6` 已触发且已按decision读取，**不要再次消费或当作本轮live monitor**。
本次核对原controller PID945496及recovery PID1045226均已不存在。
压缩后仍需核对新进程，不能把这个快照当永久无任务证明。

长GPU任务无独立工作时按wake-me-up规范绑定真实producer，确认armed后再交接；新的任务使用新key。Monitor expiry不会赋予杀进程、重训或算法失败含义。普通native worker不要用monitor代替交付管理。

本阶段尚无新运行输出根；可建立
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-15-coco22-cumulative-expansion`
作为独立新根，别把新结果写进前两轮冻结目录。

未授权事项：推送/提交/发布Notion、修改Codex memory、清理无关dirty文件、自动增加第三臂/新seed/更多updates或继续下一档图像数。用户已授权的当前22图主线、条件Source、审核补标与吞吐优化不需要再重复确认。

## 文档检查与持久性

```sh
python -B scripts/research/check_research_knowledge.py check
```

本轮只新增/更新研究文档及catalog/state入口，未改训练代码或标注内容。
文档与前轮untracked实现均在共享文件系统中；没有提交或写入Codex memory。
压缩后从本文件→unit→state接着未完成准备工作，不要重复已完成的Pro讨论、CE消融或旧轮训练。
