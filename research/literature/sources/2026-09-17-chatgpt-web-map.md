可以。下面我把这段时间筛出来、我认为**值得真正记录并在后续按需精读**的论文，整理成一张研究地图。不是按发布时间排，而是按它们回答的问题排。

先给一个总纲。最近这些工作共同把 VLM 的视觉计算拆出了几个越来越清晰的层次：

$$
\boxed{
\text{semantic evidence}
\neq
\text{instance individuation}
\neq
\text{spatial grounding}
\neq
\text{readout/routing}
\neq
\text{causal utilization}
\neq
\text{autoregressive selection}
}
$$

另一个反复出现的结论是：

$$
\boxed{
\text{“hidden state 里能 decode 出来”}
\not\Rightarrow
\text{native model 会真正使用它}
}
$$

而 detection 又额外引入了：

$$
\boxed{
\text{object-level credit}
+
\text{set coverage}
+
\text{duplicate suppression}
+
\text{termination}
}
$$

这几条基本可以作为下面所有论文的“索引”。

---

# 一、视觉表征、Binding 与 Localization

| 论文                                                                                    | 中心思想、创新点、主要发现                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Foveated Probes Recover Localized Binding Information in Vision Foundation Models** | 冻结 Vision Encoder，只更换 final patch-token readout。Global pooling 在 clutter / same-category instances 下会把 localized binding signal 稀释掉，而 learned/question-conditioned foveated pooling 能恢复大量 oracle-accessible 信息。最重要的结论是：**global readout 失败不能推出 Vision Tower 没有 localized/binding information**。还提出 counterfactual nuisance-to-signal ratio，把真正 target-changing signal 与 distractor variation 分开。([arXiv][1]) |
| **Object Concepts Emerge from Motion**                                                | 用视频中的 motion boundaries 产生 pseudo-instance supervision，再训练静态图像 encoder 学习 object-centric representation。核心不是“识别类别”，而是保持一个 physical instance 的内部 coherence，并拉开不同 instances 的 representation。是“semantic category representation ≠ instance representation”的很强外部证据。([arXiv][2])                                                                                                                                 |
| **Semantic-Spatial Discriminability Enhancement for Generalized Visual Grounding**    | 针对 multi-target / visually similar objects，分别提高 fine-grained semantic discriminability 和 spatial instance separation；后者通过 instance-center density map 显式建立不同实例的空间 decision boundary。核心启示是：**知道这一片是 person，与能够把 person A/person B 分开，是两个能力。** ([arXiv][3])                                                                                                                                                    |
| **Mechanisms of Object Localization in Vision-Language Models**                       | 当前最值得精读的 localization mechanism paper。通过 token ablation、controlled perturbation、attention knockout、causal mediation 等分析 LLaVA / InternVL。最重要发现是 **containerization**：bbox 很大程度由“一组被归属于 object 的 spatial tokens 的 extent”决定，而不是内部必须显式保存 \((x_1,y_1,x_2,y_2)\)。Classification 与 localization 共享部分早期 processing，但最终依赖部分不同的 sparse circuits。([Open Access CVF][4])                                               |
| **Grounding Isn't Knowing: Do VLMs Need Object Localization for Spatial Reasoning?**  | 分析 grounding 与 spatial reasoning 是否为同一能力。发现 object-aligned tokens 先形成 coarse target/reference anchors，position information 比 relation decision 更早变得 decodable；精确 bbox boundary 并非关系推理的必要条件。Grounding 与 reasoning 共享部分早期机制，但后面分叉成不同 specialized pathways。([arXiv][5])                                                                                                                                         |
| **Canonical Color as a Lens into Concept Decodability in Vision Encoders and VLMs**   | 用“灰度香蕉仍应关联 yellow”这种 canonical-color task 测 conceptual semantics，而不是直接可见颜色。即使颜色被去掉，object identity 与 canonical color 仍可从视觉表征中解码；VLM post-training 又能显著改变 Vision Encoder 的 decodability。提醒我们：**representation geometry / basis 会随 multimodal training 发生大幅重构。** ([arXiv][6])                                                                                                                                  |
| **PANORAMA: Panoptic Grounded Captioning via Mask Proposal Selection**                | 把 dense captioning 与 panoptic grounding 联合起来。VLM 产生 contextualized phrase representation，再条件化 segmenter 得到 proposals，并选择与 phrase 对应的一/多个 masks。很重要的建模观点是：**“我现在指的是谁”与“它的精确 geometry/mask 是什么”可以拆成 query representation 与 proposal selection 两步。** 同时发布 PanoCaps。([arXiv][7])                                                                                                                                 |

---

# 二、Visual Readout、Attention Heads、Residual Stream 与信息路由

这一组是我认为对理解现代 VLM 内部最核心的一批。

| 论文                                                                                                                                      | 中心思想、创新点、主要发现                                                                                                                                                                                                                                                                                                                                                    |
| --------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Gaze Heads: How VLMs Look at What They Describe**                                                                                     | 在 Qwen3-VL 等模型中找到一小组 attention heads，它们的视觉 attention 会随着当前生成文本动态移动到“现在正在描述的区域”。更关键的是 causal intervention：人为重定向这些 heads 的 attention，模型真的会转去描述指定 region。非常强地支持存在一种 **dynamic spatial pointer / gaze mechanism**。([arXiv][8])                                                                                                                                       |
| **Retrieval Heads Meet Vision: Uncovering How VLMs Locate and Extract Visual Information**                                              | 提出 Visual Retrieval Heads。只有约 1.7–2.6% heads，却对 text→visual grounding 有巨大 causal effect；mask top heads 可以让 grounding accuracy 暴跌，而 random heads 影响很小。这些 heads 还能跨 attribute、spatial、counting、visual math 等任务发挥作用。说明视觉 readout 很可能依赖 **稀疏 retrieval circuitry**。([arXiv][9])                                                                                      |
| **Can Retrieval Heads See Images? Multimodal Retrieval Heads in Long-Context Vision-Language Models**                                   | 把 retrieval-head 概念扩展到长上下文 multimodal evidence retrieval。发现 multimodal retrieval heads 同样 sparse、causally important，而且 visual retrieval heads 会随 context length / modality 发生变化。Qwen3-VL-8B 上也验证了这类 heads 可直接用于 visual document retrieval。([arXiv][10])                                                                                                          |
| **Using OCR Heads to Verbalize Image Semantics**                                                                                        | 近期我最推荐的一篇。最初定位的是 OCR-critical heads，但发现它们其实是 **general-purpose verbalization heads**：给它们看 `"bike"` 会写出 bike，指向 bird wing 又会写入 feathers-like semantic direction。把这些 heads 的 OV transformation 合并，可构造 verbalization lens；在 Qwen3-VL 中，从 LLM layer 0 开始 visual tokens 就已经含有可解释的语言语义。强烈支持“视觉 semantics 早已存在，后面少量 heads 负责 fetch + write into residual”。([arXiv][11]) |
| **Causal Tracing of Object Representations in Large Vision Language Models: Mechanistic Interpretability and Hallucination Mitigation** | 提出 FCCT，在 **token role × layer × MHSA/FFN/hidden-state** 三个轴上 causal trace object information。发现 middle-layer last-token MHSA 是重要 cross-modal aggregation point；FFN 表现出 visual-centered → cross-modal → generation-oriented 的三阶段 representation evolution。还提出 IRI，把中间层有用 visual representation 注入后层以改善感知/幻觉。([arXiv][12])                                        |
| **When Vision Becomes Text: Visual Token Pruning via Cross-Modal Residual Guidance in VLMs**                                            | 发现随着 LLM 深度增加，text tokens 会不断通过 self-attention 吸收 visual information。提出 Cross-Modal Absorption / Cross-Modal Residual，用几何投影量化“哪些 visual information 已经可以被 text subspace 解释”。核心机制启示是：**视觉 evidence 后期未必还“住在 image token 上”，可能已经迁移进 text/history residual。** ([arXiv][13])                                                                                         |
| **Decodable but Misrouted: Sparse Features Uncover a Readout Gap in Vision-Language Models for Harmful Meme Detection**                 | 明确把三个问题拆开：**decodability、routing、recoverability**。Sparse probe 可以读出远强于 native prediction 的 task signal，但 native model 并不自然消费这些 feature；进一步干预 routing 可以追回大量 readout gap。虽然任务不是 detection，但它是“**信息存在 ≠ native decision 使用**”最直接的实验证据之一。([arXiv][14])                                                                                                              |
| **Through the LENS: Local Geometric Decomposition of Vision-Language Model Representations**                                            | 反对只寻找全局 linear direction。用 Mixture of Factor Analyzers 把 residual activation 分成局部低秩 neighborhoods，发现 Qwen3-VL 的 vision/text representation 会经历 early mixing → partial re-separation → late re-fusion。沿 local neighborhood centroid steering 可以因果改变输出。适合思考 representation 可能是 **local manifold，而不是一个全局 owner direction**。([arXiv][15])                            |
| **Multimodal Model Diffing for Feature Discovery and Control**                                                                          | 对 base LM 与 multimodal-adapted LM 的 SAE features 做 diff，寻找真正因 multimodal training 产生/旋转的 features；再通过 contrastive firing、ablation 和 steering 验证 causal specificity。发现 spatial、OCR 等能力确实可以关联到稀疏的、可干预 multimodal features。是 residual-stream feature-level mechanistic analysis 很好的工具路线。([arXiv][16])                                                               |

---

# 三、Failure Diagnosis：模型到底在哪一步坏掉？

| 论文                                                                                                                                 | 中心思想、创新点、主要发现                                                                                                                                                                                                                                                                                                                                                                                                       |
| ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **How Do VLMs Fail? Vision-Operation Misalignment in Compositional VQA**                                                           | 把 VLM failure 按 operation 分成四类：**grounding failure、reasoning failure、attribute extraction failure、language-prior dominance**；再用 mean ablation / attention knockout / MLP intervention 定位内部 dependency。论文报告不同 operation 的 failure 具有不同 Attention/FFN causal profile。最有价值的是方法论：不要把所有错误统称“视觉差”，而应问 **哪一种 operation 坏了，以及它依赖哪种 computation**。对部分 “某组件导致错误”的叙事应比论文更谨慎，因为强 causal dependence 不等于该组件本身就是错误来源。([arXiv][17]) |
| **Visual Attention Faithfulness in Vision-Language Models is Heterogeneous**                                                       | 用 causal perturbation 检验 attention-ranked visual tokens，而不是相信 attention heatmap。发现三种模式：**Faithful-Sufficient、Faithful-Distributed、Non-Focal**。也就是说，有时少数局部 token 足够，有时还需要广泛 context，有时根本不存在单一 focal region，但视觉整体仍然 causal。([arXiv][18])                                                                                                                                                                              |
| **The Visual Insensitivity Gap: Diagnosing When Vision-Language Models Fail to Use Visual Evidence**                               | 对 question-relevant region 做视觉扰动并观察 next-token distribution。大量样本即使关键区域被改变，模型输出也几乎不变；但 Vision Tower probe 又能很好地区分 clean vs perturbed。非常漂亮地展示了 **encoder 明明感知到了变化，LLM 决策却没有响应**。([arXiv][19])                                                                                                                                                                                                                         |
| **Ask Twice, Look Twice: Prompt Echoing Resolves the Question-First Paradox in Vision-Language Models**                            | 发现 question-first 确实会 steer image representation，让视觉 patch 更 question-conditioned；但 answer token 后期却难以访问被大量 image tokens 隔开的 question。通过 causal attention knockout 验证“representation steering”和“answer-time access”是两个阶段。Prompt echoing 把问题分别放在 image 两侧，一个负责 perception steering，一个负责 final readout。([arXiv][20])                                                                                                  |
| **What Do Hallucinations Reveal About Multimodal Reasoning? Diagnosing Visual Grounding Failures via Contrastive Decoding Probes** | SAFE 同时运行 normal-vision 与 vision-ablated decoding，比较每个 token 的 logit gap，得到 token-level visual-dependency proxy。最重要发现：**视觉依赖会随 autoregressive generation 整体衰减，hallucination 往往出现 temporal clustering**。这是少数真正把 failure 放到 autoregressive time axis 上看的论文。([arXiv][21])                                                                                                                                              |
| **Who Drives the Probability Game of VLMs? A Temporal Causal Drive Evaluation Framework**                                          | 把 visual、question、generated-prefix 三种 source 的 causal influence 随 decoding step 分别量化成 VCD/QCD/PCD。Qwen3-VL-8B 等模型中观察到一个稳定趋势：早期更依赖视觉/问题，后期 generated prefix influence 上升。比单纯 attention-decay 更直接地研究了 **autoregressive history 逐渐接管决策**。([Academus][22])                                                                                                                                                              |

---

# 四、Instance State、Grounding Interface 与显式 Spatial Readout

| 论文                                                                                                     | 中心思想、创新点、主要发现                                                                                                                                                                                                                                                                                                   |
| ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SCVIB: Editable State-Conditioned Visual Instance Binding for Multi-Turn Personalized Localization** | 构造一个很干净的 multi-turn physical-instance binding setting：多个 support-defined instances 被赋予临时 identity，随后 bind/switch/rollback 等 state events 决定最终要找谁。关键发现：**正确解析最终 target identity，并不保证模型真正使用对应 visual evidence 完成 localization。** 因此将 state transition、evidence routing、same-instance grounding 显式拆开。([arXiv][23]) |
| **Pointing-VLA: Typed Spatial Grounding Interfaces for Vision-Language-Action Manipulation**           | 不再把 spatial information 强迫序列化成 text coordinates，而从 multimodal hidden state 接 typed heads，分别预测 point、heatmap、trajectory。证明 **poor coordinate-token generation 不等价于 poor spatial representation**，不同 geometry 甚至适合不同 readout interface。虽然是 VLA，但非常适合作为 spatial latent / hidden-state readout 的正对照。([arXiv][24])   |
| **PANORAMA: Panoptic Grounded Captioning via Mask Proposal Selection**                                 | 同样值得放在这里再强调一次：contextualized phrase hidden state 更像一个 **referent query**，具体 geometry 再由 segmenter proposal selection 完成。Identity/referent representation 与 geometry decoding 没有必要是同一变量。([arXiv][7])                                                                                                             |

---

# 五、Autoregressive Detection / Grounding Training

这一组与 detection task 本身最直接。

| 论文                                                                                                                             | 中心思想、创新点、主要发现                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Detect Anything in Graphic Design: Element-Level Rewards for Autoregressive Detection**                                      | 把 graphic design detection 建模为 **compositional deconstruction**：按原设计的 back-to-front layer order 自回归输出 elements，并进行 amodal bbox detection。最大创新是 **Element Relative Policy Optimization (EleRPO)**：不是给整条 completion 一个 GRPO advantage，而计算每个 element 加入后对 detection F1 的 marginal contribution，并在相似 prefix-quality rollout 内做 relative credit。强烈说明 structured autoregressive detection 的 RL credit unit 应更靠近 **object/element，而非 sequence**。([ArXivSignals][25]) |
| **PointRL: Learning Point-Level Vision-Language Grounding from Verifiable Annotation Evidence**                                | 把 multi-instance point grounding 明确写成 **set problem**。Verifier 同时考虑 point validity、instance coverage、cardinality、duplicate 与 missing targets，再通过 RL 优化。核心贡献不是 point interface，而是证明 **localization quality + set coverage + count + duplicate suppression** 必须联合进入 reward。([arXiv][26])                                                                                                                                                                        |
| **Your Model Already Knows Don't Teach It, Learn to Ask It: Soft Prompting for Few-Shot Adaptation of Vision-Language Models** | 近期极值得记的一篇。冻结 Qwen3-VL-8B 全部 backbone，只学习 1–3 个 continuous tokens；将它们放在 **visual tokens 与 text tokens 的 cross-modal boundary** 效果最好。平均仅 7,168 trainable parameters，在 Roboflow20-VL 10-shot detection 上达到与最佳 LoRA 相同的 14.2 mAP，却不造成 LoRA 的 catastrophic forgetting。它提供了非常强的证据：**有些 specialized detection failure 更像 access/control/readout 问题，而不是必须新增大量 representational capacity。** 但 medical 等 domain 的失败也提醒：并非所有能力都预先存在。([arXiv][27])                        |

---

# 六、Coverage / Selection：很值得保留，但不用全部第一优先级精读

这几篇不是 detection mechanism 主线，却有一个非常一致的发现：

$$
\boxed{
\text{importance / relevance}
\neq
\text{coverage}
}
$$

### **Who Speaks for the Pruned? Visual Token Pruning as Coverage Optimization**

把 visual-token pruning 改写成 **Representational Coverage Maximization**：

$$
f(S)
=
\sum_i w_i
\max_{j\in S}
\operatorname{sim}(v_i,v_j).
$$

重点不再是“哪些 token 分数最大”，而是“被删掉的每个 evidence 是否还有 survivor 能代表它”。而且 similarity 在 projector / LLM input space 中计算。非常值得作为 set-coverage 的数学参照。([alphaXiv][28])

### **CoVeR: Coverage-Based Token Pruning for Multi-View 3D Reasoning in VLMs**

完全不使用 attention importance，只利用空间 coverage 保证整个 3D scene 都有 representative。证明 learned importance 很容易反复选择显著区域附近的近重复 tokens，而让其他区域完全没有 representation。([arXiv][29])

### **StackTok: Accelerating VLMs Inference with Budget-Adaptive Visual Token Selection**

提出一个很有意思的 distinction：

> **relevance 是 objective，coverage 是 support constraint。**

根据当前 query 与 budget，selector 动态在“补 coverage”和“最大 relevance”两个模式之间切换，而不是固定线性权重。([arXiv][30])

这三篇可以作为一个小专题一起看，不一定全部深读。

---

# 七、另外两篇我建议留档

### **Semantic-Spatial Discriminability Enhancement for Generalized Visual Grounding**

如果后面专门研究“same-category adjacent instances 怎么分开”，这篇值得拿出来。它的 center-density auxiliary supervision 是一个很直接的 instance-separation positive control。([arXiv][3])

### **Object Concepts Emerge from Motion**

如果后面问题进一步前移到 Vision Tower pretraining：“为什么 CLIP/DINO 语义很好，却不一定有 physical-instance identity？”这篇非常值得完整读。它直接把 motion boundary 当作 instance grouping supervision。([arXiv][2])

---

# 八、如果只保留一个“核心精读清单”

如果以后时间有限，我建议真正**完整精读**这 12 篇，其余作为专题补充：

| 优先主题                                       | 论文                                                                                                                                      |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Localization mechanism**                 | **Mechanisms of Object Localization in Vision-Language Models**                                                                         |
| **Localized binding / readout**            | **Foveated Probes Recover Localized Binding Information in Vision Foundation Models**                                                   |
| **Visual→language semantic write**         | **Using OCR Heads to Verbalize Image Semantics**                                                                                        |
| **Dynamic spatial pointer**                | **Gaze Heads: How VLMs Look at What They Describe**                                                                                     |
| **Sparse visual retrieval circuits**       | **Retrieval Heads Meet Vision: Uncovering How VLMs Locate and Extract Visual Information**                                              |
| **Object causal flow through layers**      | **Causal Tracing of Object Representations in Large Vision Language Models: Mechanistic Interpretability and Hallucination Mitigation** |
| **Failure taxonomy**                       | **How Do VLMs Fail? Vision-Operation Misalignment in Compositional VQA**                                                                |
| **Representation ≠ utilization**           | **The Visual Insensitivity Gap: Diagnosing When Vision-Language Models Fail to Use Visual Evidence**                                    |
| **Autoregressive visual-dependence decay** | **What Do Hallucinations Reveal About Multimodal Reasoning? Diagnosing Visual Grounding Failures via Contrastive Decoding Probes**      |
| **AR detection + object credit**           | **Detect Anything in Graphic Design: Element-Level Rewards for Autoregressive Detection**                                               |
| **Few-shot detection / control state**     | **Your Model Already Knows Don't Teach It, Learn to Ask It: Soft Prompting for Few-Shot Adaptation of Vision-Language Models**          |
| **Set-level grounding reward**             | **PointRL: Learning Point-Level Vision-Language Grounding from Verifiable Annotation Evidence**                                         |

---

# 九、我目前从整个文献群得到的“共同画像”

这些论文虽然彼此任务不同，但正在逐渐拼出一个相当一致的 VLM computation picture：

$$
\boxed{
\text{Pixels}
}
$$

↓

### 1. Vision representation

先形成：

* semantic features；
* local appearance；
* spatial layout；
* 某种程度的 instance/object grouping。

但：

$$
\text{category semantics}
\neq
\text{physical-instance individuation}.
$$

↓

### 2. Vision→Language interface / representation reformatting

projector / merger / early LLM 把 visual state 改写成 language backbone 更容易消费的 basis。

因此：

$$
\text{某层 probe 不出来}
\not\Rightarrow
\text{信息不存在}.
$$

↓

### 3. State-conditioned spatial retrieval

current token/query 通过少量：

* gaze heads；
* retrieval heads；

决定：

$$
\text{现在去哪看}.
$$

↓

### 4. Semantic fetch / residual write

verbalization-like heads 把选中的 visual states 经过：

$$
W_V\rightarrow W_O
$$

写入当前 residual stream。

所以：

$$
\text{attention 到 B}
$$

仍然不等于：

$$
\text{B 对 residual 的 write 有效}.
$$

↓

### 5. Cross-modal / text-side transformation

Visual information 会逐渐进入 text/history residual。

中间的 MHSA / MLP 把它从：

$$
\text{visual-centered}
$$

改造成：

$$
\text{cross-modal}
$$

再改造成：

$$
\text{generation-oriented}.
$$

↓

### 6. Native causal utilization

即使 information 可以被 probe 解出：

$$
\boxed{
\text{decodable}
\not\Rightarrow
\text{routed}
\not\Rightarrow
\text{causally used}.
}
$$

这是 Visual Insensitivity、Decodable but Misrouted、Foveated Probes 等共同强调的。

↓

### 7. Autoregressive dynamics

随着 completion 变长：

$$
\text{visual influence}\downarrow
$$

而：

$$
\text{generated-prefix influence}\uparrow
$$

可能出现。

SAFE、Temporal Causal Drive 等工作都开始捕捉这个现象。

↓

### 8. Structured-output / detection layer

到了 detection，又额外需要：

$$
\text{instance discovery}
$$

$$
\text{geometry decoding}
$$

$$
\text{set coverage}
$$

$$
\text{duplicate control}
$$

$$
\text{object-level credit}
$$

$$
\text{termination}.
$$

PointRL 与 DAD/EleRPO 是目前这一层最值得保存的工作。

---

如果把整批论文最后压成一句最值得记在笔记首页的话，我会写：

$$
\boxed{
\textbf{现代 VLM 的主要科学问题已经不只是“它有没有视觉信息”，
而是“这些信息以什么 representation 存在、怎样被当前 state 寻址、
怎样写入 residual、是否真正进入 native decision，以及如何随 autoregressive history 演化”。}
}
$$

而 detection 把这个问题再推进一步，因为它要求上述过程不是成功一次，而是**持续、逐实例、集合一致地重复成功**。

[1]: https://arxiv.org/abs/2608.00726?utm_source=chatgpt.com "Foveated Probes Recover Localized Binding Information in Vision Foundation Models"
[2]: https://arxiv.org/abs/2609.04348?utm_source=chatgpt.com "Object Concepts Emerge from Motion"
[3]: https://arxiv.org/abs/2608.30233?utm_source=chatgpt.com "Semantic-Spatial Discriminability Enhancement for Generalized Visual Grounding"
[4]: https://openaccess.thecvf.com/content/CVPR2026/papers/Schaumloffel_Mechanisms_of_Object_Localization_in_Vision-Language_Models_CVPR_2026_paper.pdf?utm_source=chatgpt.com "Mechanisms of Object Localization in Vision-Language Models"
[5]: https://arxiv.org/abs/2608.23074?utm_source=chatgpt.com "Grounding Isn't Knowing: Do VLMs Need Object Localization for Spatial Reasoning?"
[6]: https://arxiv.org/abs/2609.09124?utm_source=chatgpt.com "Canonical Color as a Lens into Concept Decodability in Vision Encoders and VLMs"
[7]: https://arxiv.org/abs/2609.19143?utm_source=chatgpt.com "PANORAMA: Panoptic Grounded Captioning via Mask Proposal Selection"
[8]: https://arxiv.org/abs/2606.14703?utm_source=chatgpt.com "Gaze Heads: How VLMs Look at What They Describe"
[9]: https://arxiv.org/abs/2608.27417?utm_source=chatgpt.com "Retrieval Heads Meet Vision: Uncovering How VLMs Locate and Extract Visual Information"
[10]: https://arxiv.org/abs/2605.27243?utm_source=chatgpt.com "Can Retrieval Heads See Images? Multimodal Retrieval Heads in Long-Context Vision-Language Models"
[11]: https://arxiv.org/abs/2609.18823?utm_source=chatgpt.com "Using OCR Heads to Verbalize Image Semantics"
[12]: https://arxiv.org/abs/2511.05923?utm_source=chatgpt.com "Causal Tracing of Object Representations in Large Vision Language Models: Mechanistic Interpretability and Hallucination Mitigation"
[13]: https://arxiv.org/abs/2608.10489?utm_source=chatgpt.com "When Vision Becomes Text: Visual Token Pruning via Cross-Modal Residual Guidance in VLMs"
[14]: https://arxiv.org/abs/2609.18860?utm_source=chatgpt.com "Decodable but Misrouted: Sparse Features Uncover a Readout Gap in Vision-Language Models for Harmful Meme Detection"
[15]: https://arxiv.org/abs/2608.00561?utm_source=chatgpt.com "Through the LENS: Local Geometric Decomposition of Vision-Language Model Representations"
[16]: https://arxiv.org/abs/2608.09928?utm_source=chatgpt.com "Multimodal Model Diffing for Feature Discovery and Control"
[17]: https://arxiv.org/abs/2607.16094?utm_source=chatgpt.com "How Do VLMs Fail? Vision-Operation Misalignment in Compositional VQA"
[18]: https://arxiv.org/abs/2609.00830?utm_source=chatgpt.com "Visual Attention Faithfulness in Vision-Language Models is Heterogeneous"
[19]: https://arxiv.org/abs/2609.00868?utm_source=chatgpt.com "The Visual Insensitivity Gap: Diagnosing When Vision-Language Models Fail to Use Visual Evidence"
[20]: https://arxiv.org/abs/2607.15565?utm_source=chatgpt.com "Ask Twice, Look Twice: Prompt Echoing Resolves the Question-First Paradox in Vision-Language Models"
[21]: https://arxiv.org/abs/2609.16646?utm_source=chatgpt.com "What Do Hallucinations Reveal About Multimodal Reasoning? Diagnosing Visual Grounding Failures via Contrastive Decoding Probes"
[22]: https://academ.us/article/2609.02000/?utm_source=chatgpt.com "[2609.02000] Who Drives the Probability Game of VLMs? A Temporal Causal Drive Evaluation Framework - Academus scientific article reader"
[23]: https://arxiv.org/abs/2608.14148?utm_source=chatgpt.com "SCVIB: Editable State-Conditioned Visual Instance Binding forMulti-Turn Personalized Localization"
[24]: https://arxiv.org/abs/2608.23138?utm_source=chatgpt.com "Pointing-VLA: Typed Spatial Grounding Interfaces for Vision-Language-Action Manipulation"
[25]: https://arxivsignals.io/papers/2609.07072?utm_source=chatgpt.com "Detect Anything in Graphic Design: Element-Level Rewards for Autoregressive Detection · ArXivSignals"
[26]: https://arxiv.org/abs/2608.25299?utm_source=chatgpt.com "PointRL: Learning Point-Level Vision-Language Grounding from Verifiable Annotation Evidence"
[27]: https://arxiv.org/abs/2609.11310?utm_source=chatgpt.com "Your Model Already Knows Don't Teach It, Learn to Ask It: Soft Prompting for Few-Shot Adaptation of Vision-Language Models"
[28]: https://www.alphaxiv.org/abs/2609.03158?utm_source=chatgpt.com "Who Speaks for the Pruned? Visual Token Pruning as Coverage Optimization | alphaXiv"
[29]: https://arxiv.org/abs/2609.08345?utm_source=chatgpt.com "CoVeR: Coverage-Based Token Pruning for Multi-View 3D Reasoning in VLMs"
[30]: https://arxiv.org/abs/2609.16841?utm_source=chatgpt.com "StackTok: Accelerating VLMs Inference with Budget-Adaptive Visual Token Selection"
