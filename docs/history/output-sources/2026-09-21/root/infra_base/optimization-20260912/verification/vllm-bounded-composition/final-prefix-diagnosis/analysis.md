# 733 → 858：同前缀诊断与原图定位

这是最终四步 smoke checkpoint 的动态 HF 与合并 BF16 HF 对照；差异在 vLLM 引擎运行之前发生。此记录解释观测，不改变已接受的一格验收约束。

## 结论

125 是两个离散坐标 token 的数值距离。同一动态前缀下，两个候选在动态 HF 中恰好并列第一；合并后 0.125 的 logit 优势使 argmax 选择了另一个候选。该结果证明先前 token 的偏移不是此处翻转的必要原因，但不证明合并在所有输入上无害。

| 固定前缀 | 动态 HF 选择 | 合并 HF 选择 |
| --- | ---: | ---: |
| 动态自然生成的前 12 个 token | 733 | 858 |
| 合并自然生成的前 12 个 token | 858 | 858 |

在同一个动态前缀下：

| 候选 | 动态 logit | 合并 logit | 动态 softmax | 合并 softmax |
| --- | ---: | ---: | ---: | ---: |
| coord_733 | 26.5 | 26.25 | 2.1365% | 1.9815% |
| coord_858 | 26.5 | 26.375 | 2.1365% | 2.2453% |

动态模型的两个最大值完全相等，argmax 选 token ID 较小的 733。合并后 858 的相对优势为 exp(0.125)=1.133。两侧约 99.4% 的概率质量都在坐标词表上，但最高单项仅约 2%；这是分散且存在相距较远候选峰值的坐标分布。小分数扰动不保证离散坐标差也小。这里的 softmax 是从保存的 logits 计算的诊断量，不是采样频率。

保持同一动态前缀时，全词表 logit 最大绝对差为 0.25；保持合并前缀时为 0.140625。实验保留原有 BF16 加载和缓存生成设置，未把模型/输入提升到另一种精度。Transformers 返回的生成 logits/scores 张量为 FP32，这不改变前面的模型算子精度。结构、提示、选定行及全部 196 个合并目标的权重身份检查通过；现有证据支持动态 DoRA/delta 与折叠矩阵之间运算顺序和舍入引起的选择敏感性，未定位成某一个单独层的唯一原因。

## Exact image 与 owner 边界

图片为 `coco2017_val_000000000139`，1248×832。原文件、输入 JSONL 和执行图像哈希均与 composition receipt 相同，无图像变换。GT 共 20 个，来自同一输入样本。可视化使用现有 `src.vis` 读取、归一化和 GT 渲染路径。

差异位于生成 token 的零基索引 12，即下面 `y1` 属性的第一个 token：

```text
dynamic: <points x1="<|coord_733|><|coord_858|><|coord_858|>" y1="<|coord_733|><|coord_858|><|coord_858|>" alt="clearly visible object instance">clearly visible object instance</points><|im_end|>
merged:  <points x1="<|coord_732|><|coord_858|><|coord_857|>" y1="<|coord_858|><|coord_858|><|coord_858|>" alt="clearly visible object instance">clearly visible object instance</points><|im_end|>
```

两个输出都不是约定的 object-box 格式，x1/y1 各包含三个坐标 token，描述没有具体类别。当前真实 parser 对二者均返回 `all_spans_dropped`、0 个合法预测、`unmatched_text`。因此没有可以可靠绑定的 exact object owner，不能将 125 格描述为一个已识别物体的合法预测框移动。

图中青色/洋红色水平线仅将两个候选假设为 y 坐标。项目坐标转换为 `round(bin * extent / 1000)`，因此 y733→610px、y858→714px，相差 104px。虚线和圆点额外假设第一个 x1 token 为 x733，固定同一 x 得到 (915,610) 与 (915,714)，两点均不属于任何一个 GT 框。**这些点是假设示意，不是解析出的预测或 owner 指认。**

右下角花瓶的 GT 是 `[858,726,915,937]`，annotation ID 1667817；其中 858 是 x1，与生成 y1 候选恰好同值不能建立 owner 关系。一个合法词表坐标不等于合法或正确的物体预测。现有证据既不能判定同一 owner 下哪个候选更好，也不支持“两个都 valid”的定位结论。

## 证据

- [2×2 同前缀诊断](diagnosis-summary.json)
- [候选分数、概率与完整文本](probability-and-context.json)
- [原图坐标示意](visualization/coco139_coord733_vs858.png)
- [可视化身份与假设记录](visualization/manifest.json)
- [原始 GT/预测渲染记录](visualization/gt-vs-pred/manifest.json)；该 HF batch run 仅用于同图 GT，不将其生成文本冒充 composition 单行对照文本。
- [fresh parser 与图像验证](visualization/verification.json)

此四步 checkpoint 可提供加载/生成/解析链路证据，不能提供成熟模型的定位质量结论。没有新增训练，没有修改生产代码或资格门槛，也未将未完成的 vLLM 资格与评估任务归档。

## 用户追问后的加载身份核对

[加载核对记录](load-identity-audit.json) 确认本次不是成熟 `geo_sorted_xy` adapter。训练配置为 `adapter.seed_mode: initialize_new`、`adapter.path: null`、`max_steps: 4`，ordering 为 `geo_sorted`。HF 与 composition 均加载该新运行的 step-4 DoRA 和配套 delta。HF 记录 196 个启用 adapter 层、588 个保存/运行时状态键检查通过，delta 明确 `loaded: true`；不是遗漏 adapter 后误跑裸 base。

输入 user prompt 是 `Locate each clearly visible object instance in the image. Use the CoordExp object-box format and COCO class text.`，原始 XML `alt` 与其中短语相同。此前将该 `alt` 简称为 desc 不够准确：当前 parser 并没有产出合法 prediction description。没有做 prompt 干预实验，不能将逐字相同升级为唯一因果解释。

已定位用户所指成熟 `geo_sorted_xy` 路径的现有候选：`/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444`。它未参与本次 125 格对照。此四步 smoke 的离散翻转不能代替该成熟 checkpoint 在其匹配 prompt/template 下的推理比较，也不足以据此推进成熟模型的误差门槛放宽。

后续对齐成熟模型时还确认：成熟 checkpoint 的训练/原验证 prompt 包含完整 object-box 示例和 COCO80 类别清单，与四步 smoke 采用的精简 production prompt 不同。因此，四步 smoke 相对于成熟模型同时存在 checkpoint 和 prompt 差异，不能将格式或泛化文本单独因果归因于训练步数。用户已授权的新对照在 `../../mature-geo-sorted-xy-consistency/contract.md` 冻结成熟 checkpoint、原生完整 prompt 和相同 decode，仅改变动态与合并执行方式。
