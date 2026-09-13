# Mature geo_sorted_xy：动态 HF 与合并 BF16 一致性诊断

结论：多数可配对检测框只有少量 bins 的变化，但也有未配对预测和明显尾部变化；这 32 张图上的诊断 mAP 下降 1.61 个 AP 点。现有证据不足以把合并 BF16 视为对成熟动态 HF 无损，也不足以判断完整验证集上的稳定退化幅度。

## 实验身份与范围

- 使用成熟 `four-coordinate-xy/step-2444` DoRA r16/a32 与其配套 special-token delta，base 为 `Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`。没有训练，没有使用先前的四步 checkpoint。
- 对照两边都是 HF：动态 base+DoRA+delta，对比相同来源的认证 BF16 合并模型。尚未比较 vLLM 引擎，也未完成生产 vLLM admission。
- 采用 checkpoint 训练/原 val200 完全一致的原生完整 prompt；BF16、FlashAttention2、batch4、greedy、repetition_penalty1.1、max_new_tokens512、相同评分规则。
- 固定 32 张 COCO 验证图，共 296 个 GT：COCO139 是预先指定的诊断锚点，其余31张按 seed20260912 随机抽取，没有按模型结果过滤样本。
- 原始 x1→y1 排序逐行验证并保留。当前 checkout 不支持 geo_sorted_xy 枚举，显式使用 source_order 读取已排序记录；没有重排 GT，没有改变提示词。真实执行的 prompt traces 完全相同，image plans 字节相同。
- 两组均32行完成、0 parser failures、0 score failures，32行可用于检测评分。动态/合并分别保留293/304个预测，丢弃2/3个无效预测跨度。两组均31自然EOS、1行在512token处截断，截断图均为COCO76468；所有32行都保留在比较中。
- 当前流水线要求至少200行才能标为 benchmark；本轮 `benchmark_eligible=false`、`benchmark_metric=false`，以下数值仅是小样本诊断 AP。

## 直接 evaluator 的诊断 AP

所有值按百分制 AP 点显示；差值为合并减动态。COCO GT 转换文件两边字节完全一致。

| 指标 | 动态 HF | 合并 BF16 HF | 差值 |
| --- | ---: | ---: | ---: |
| mAP@[.50:.95] | 38.10 | 36.48 | -1.61 |
| AP50 | 48.52 | 45.87 | -2.65 |
| AP75 | 41.28 | 38.69 | -2.58 |
| AP small | 5.34 | 5.73 | +0.39 |
| AP medium | 28.10 | 25.82 | -2.28 |
| AP large | 59.92 | 59.64 | -0.28 |

## 框的配对与坐标变化

使用同一归一化 COCO 类别且像素 IoU≥0.50 的候选，按 IoU 从高到低贪心一对一匹配；不按生成位置直接 zip，不把同类别但不重叠的框强行视为同一物体。该规则建立的是预测间的空间对应，不是证明预测匹配了 GT。

- 配对 242 组；动态未配对 51 个、合并未配对 62 个。未配对可能包括对象增减、类别改变或低重叠框，不能一律叫新增/丢失 owner。
- 配对框 IoU：中位数 0.9881，均值 0.9274。这些统计有 IoU≥0.5 的条件，未配对数量必须一起阅读。
- 每个配对框取四个坐标差的最大值：中位数 1 bin，P90 9 bins，P95 15 bins，最大 157 bins。
- 242 对中，最大差≤1 bin：122对；≤3 bins：165对（68.2%）；≤5 bins：193对（79.8%）。百分比分母为配对框，不是全部预测。
- 若逐坐标统计，绝对差中位数为0、P90为4、P95为7 bins；完整文本仅3/32张逐字相同。文本不完全相同本身不能代替IoU/AP结论。
- 最大157-bin配对出现在COCO25386的 dining table，配对IoU仍为0.731；该图没有此类GT，因此不能把这个配对解释为已确认GT owner上的定位改善或退化。
- AP包含类别内评分排序和不同IoU阈值的综合影响，本轮没有单独隔离坐标、召回/误报与score排序的因果贡献。初版分析的跨类别图内score inversion计数不用于解释COCO AP。

## COCO139 花瓶：一个具体 owner

固定 GT annotation1667817，类别vase，GT bins为[858,726,915,937]。按同类最高IoU匹配，两组都对应右下角花瓶。

| | 坐标 bins | 对 GT 的 IoU |
| --- | --- | ---: |
| dynamic-hf-bf16 | [857, 720, 913, 941] | 0.9050 |
| materialized-hf-bf16 | [857, 720, 915, 942] | 0.9381 |

两预测之间IoU=0.9532，最大坐标差2bins。该例支持“少量bins变化仍可对应高度重合的有效检测框”，不能代表全部32图。

![花瓶 owner 对比](/data/CoordExp/outputs/infra_base/optimization-20260912/verification/mature-geo-sorted-xy-consistency/visualization/coco139_vase_owner_comparison.png)

## 验证、修正和后续边界

- Sol负责成熟模型真实推理，Luna负责派生配对分析；没有启动Astra子代理。Lead核对真实配置/模型/图像/GT身份，并独立重跑两次直接evaluator，均exit0。
- 输入初版绝对图像路径被真实loader拒绝，失败输入已保留；在任何GPU生成前改成解析到同一图像的相对路径，32行真实loader全部通过。
- 初版配对分析没有最低IoU门槛，会把无重叠的同类框误报为数百bin漂移；Lead用一对不重叠person框复现并拒绝该统计，本文采用修正后的IoU≥.5口径。原始推理与直接AP结果不受此派生修正影响。
- 本轮不修改资格门槛，不安装vLLM admission，不将未完成OpenSpec任务归档。已完成的八GPU训练/精确resume证据仍成立；本轮新增的是成熟模型推理差异诊断。
- 推荐后续按成熟模型的任务指标与明确容差做准入判断。当前32图出现-1.61AP点变化，尚不宜直接认定为无影响；也不应使用硬性的逐token一格要求来替代定位指标。若需要生产质量准入结论，应先约定可接受AP变化，再做更有代表性的验证，而非扩大本轮诊断的结论。

## 可复现证据

- [冻结实验与停止规则](/data/CoordExp/outputs/infra_base/optimization-20260912/verification/mature-geo-sorted-xy-consistency/contract.md)；[32张样本身份](/data/CoordExp/outputs/infra_base/optimization-20260912/verification/mature-geo-sorted-xy-consistency/cohort.json)。
- [实际加载/执行记录](/data/CoordExp/outputs/infra_base/optimization-20260912/verification/mature-geo-sorted-xy-consistency/runtime/launch-receipt.json)；[配对身份验证](/data/CoordExp/outputs/infra_base/optimization-20260912/verification/mature-geo-sorted-xy-consistency/runtime/pair-identity.json)。
- [Lead配置核对](/data/CoordExp/outputs/infra_base/optimization-20260912/verification/mature-geo-sorted-xy-consistency/lead-config-verification.json)；[直接evaluator比较与哈希](/data/CoordExp/outputs/infra_base/optimization-20260912/verification/mature-geo-sorted-xy-consistency/lead-verification/evaluation/comparison.json)。
- [Lead独立核算的配对/坐标统计](/data/CoordExp/outputs/infra_base/optimization-20260912/verification/mature-geo-sorted-xy-consistency/lead-verification/conditional-box-agreement.json)；[错误配对的反例](/data/CoordExp/outputs/infra_base/optimization-20260912/verification/mature-geo-sorted-xy-consistency/lead-verification/pair-matcher-red.json)。
- [修正后的完整分析](/data/CoordExp/outputs/infra_base/optimization-20260912/verification/mature-geo-sorted-xy-consistency/analysis/corrected-iou50/paired_consistency.json)；[Lead反例、阈值边界与32行配对复核](/data/CoordExp/outputs/infra_base/optimization-20260912/verification/mature-geo-sorted-xy-consistency/lead-verification/pair-matcher-green.json)。
- [花瓶详细身份](/data/CoordExp/outputs/infra_base/optimization-20260912/verification/mature-geo-sorted-xy-consistency/visualization/coco139_vase_owner_comparison.json)；[COCO139全图对比](/data/CoordExp/outputs/infra_base/optimization-20260912/verification/mature-geo-sorted-xy-consistency/visualization/anchor139/manifest.json)；[较大变化样本](/data/CoordExp/outputs/infra_base/optimization-20260912/verification/mature-geo-sorted-xy-consistency/visualization/changed-examples/manifest.json)。
