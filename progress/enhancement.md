下面给你一套**可以直接写进计划书**的“综合修改建议”，核心目标是：

* **保持架构奥卡姆**：不加 DETR head、不引入可微 rasterizer 这类重组件；
* **Channel-A 做热路径、稳定收敛**：主要解决“坐标分布可用、格式可解析、几何一致”；
* **Channel-B 做冷路径、纠错提效**：主要解决“rollout 偏差 + set 对齐 + poly 的非唯一表达”，并把 **mask IoU** 用作 reward/筛样，而把 **OT 边界损失**用作可微 credit assignment。

> 说明：我这次没法在当前 workspace 里打开你那份 `full_idea.md` 原文，但建议完全沿用我们之前讨论的骨架（Stage-1/Stage-2、Channel-A N×forward、Channel-B rollout+matching+reorder-GT/纠错、CoordExp 等）。你可以把下面的段落按模块插回你的文档对应位置。

---

## 0. 一句“统一论述”（建议写在方案总览/贡献点）

**统一几何建模**：bbox 与 poly 都是“几何 token 序列”；训练不再假设“唯一 token 序列监督”，而是用

* **分布级几何损失**（Channel-A）确保 token 分布尖锐且几何结构正确；
* **reward( mask IoU ) + OT(可微形状对齐)**（Channel-B）把“等价表达”纳入纠错闭环，提高采样命中率与鲁棒性。

---

# A. Channel-A（热路径）建议：让它成为“稳定几何学习器”

你原本 Channel-A 的哲学是 **不 rollout、N×并行 forward 做 soft self-context，然后用 CoordExp 回传几何**。这块我建议做三件“只改 loss / stop-grad / damping”的增强。

---

## A1) 把几何从 “loss on expectation” 升级为 “expected loss”（强烈建议）

### bbox / poly 每个坐标 token 都用 expected loss（替换或并行于 L1）

对某个 coord slot 的分布 (p_k=\text{softmax}(s_k/\tau))，bin 值 (v_k=k/999)，GT 连续坐标 (c^{gt}\in[0,1])：

* **expected L1 / Huber**（更鲁棒，推荐）
  [
  L_{\text{E-L1}}=\sum_k p_k|v_k-c^{gt}|
  ]
* 或 **expected L2**（更“逼尖峰”，会隐式惩罚方差）
  [
  L_{\text{E-L2}}=\sum_k p_k(v_k-c^{gt})^2
  ]

> 计划书里建议强调：这一步解决“分布很飘但均值对了”的假好解，显著提升离散 token 生成可解析性。

---

## A2) 加“分布层几何一致性约束”（bbox 的 A1/A2；poly 的轻量先验）

### bbox：失序概率惩罚（强烈建议，成本 O(K)）

对 (X_1\sim p^{x1}, X_2\sim p^{x2})：

[
P_{\text{inv-x}}=\Pr(X_1\ge X_2)=\sum_i p^{x1}*i\underbrace{\sum*{j\le i}p^{x2}*j}*{F^{x2}*i}
]
同理 (P*{\text{inv-y}})。
[
L_{\text{inv}}=\lambda_{\text{inv}}(P_{\text{inv-x}}+P_{\text{inv-y}})
]

可选再加“塌缩惩罚”：用 (\mathbb E[(X_2-X_1)*+]) / (\mathbb E[(Y_2-Y_1)*+]) 约束最小宽高。

### poly：保持奥卡姆的 3 个 shape 正则（不要求等长）

对连续顶点 (\hat u_i)（由 CoordExp 得到）：

* 最小边长（去重/防抖）：(\sum_i \max(0,\ell_{\min}-|\hat u_{i+1}-\hat u_i|))
* 平滑（二阶差分）：(\sum_i |\hat u_{i+1}-2\hat u_i+\hat u_{i-1}|)
* 方向一致（有向面积符号固定）

> 这些是“便宜但有效”的几何先验，写在 Channel-A 能显著减少 poly rollout 时的自交/乱跳/重复点。

---

## A3) N×forward 的稳定性：建议明确“阻尼 + stop-grad 策略”

* **阻尼**：
  [
  e^{(m+1)}=(1-\alpha)e^{(m)}+\alpha,\bar e^{(m)}
  ]
  建议在文档里把 (\alpha) 作为稳定超参（早期小，后期大）。

* **stop-grad**：
  早期：对前 (m < N-1) 的 self-context 期望 embedding **detach**，只对最后一次 forward 回传（truncated）。
  后期再逐步打开更多 unroll（可选）。

> 计划书里建议明确：Channel-A 是“稳定收敛器”，因此优先选择**可控的梯度路径**。

---

# B. Channel-B（冷路径）建议：mask IoU 做 reward，OT 做可微 credit assignment

你现在 Channel-B 的价值是：rollout → parse → matching → reorder-GT / 纠错。
我建议把 poly 的“回传困难”用一个非常统一的闭环解决：

> **reward 不需要可微；可微的是一个与 reward 强相关的 surrogate（OT 边界损失）。**

---

## B1) rollout 后的两条信号：reward 与 gradient 分离

* **reward（stop-grad）**：
  对 pred poly rasterize 得到 mask，算
  [
  R=\text{maskIoU}(\hat P, P)
  ]
  用于：

1. 选择 best-of-N 样本（group sampling / pass@N 思路）
2. gating：只对 (R\ge r_0) 的轨迹更新（避免学坏）
3. 权重：(\alpha(R)) 决定几何梯度强度

* **gradient（可微）**：用 **Sinkhorn-OT（点↔线段）** 提供 credit assignment（推荐方案二）

---

## B2) poly 的可微损失：Sinkhorn-OT（点到线段）——无需对齐长度

### 构造 target：GT segments（边界是连续对象，不是顶点集合）

GT 线段 (s_j=[v_j,v_{j+1}])。

对 pred 顶点 (\hat u_i) 到线段 (s_j) 的距离平方：
[
C_{ij}=d(\hat u_i, s_j)^2
]
（投影到线段，解析可微）

### 设置质量（关键：防止“点数作弊”）

建议文档里写成 “density-invariant mass”：

* GT 边质量 (b_j \propto |v_{j+1}-v_j|)（按边长）
* pred 点质量 (a_i \propto |\hat u_i-\hat u_{i-1}|+|\hat u_{i+1}-\hat u_i|)（按局部周长）

### Sinkhorn 求 transport plan（soft matching）

[
\Pi^*=\arg\min_{\Pi\in U(a,b)} \langle \Pi, C\rangle + \varepsilon H(\Pi)
]
loss：
[
L_{\text{OT}}=\langle \Pi^*, C\rangle
]

**工程稳定建议：detach (\Pi^*)**（把它当 soft E-step），只回传到 (C\rightarrow \hat u\rightarrow) CoordExp logits：
[
\frac{\partial L_{\text{OT}}}{\partial \hat u_i}=\sum_j \Pi^**{ij}\frac{\partial C*{ij}}{\partial \hat u_i}
]

> 计划书里你可以把它描述为：
> “用 OT 将全局 mask IoU 的差异分解为边界上的软对齐与局部代价，从而对每个几何 token 给出低方差梯度。”

---

## B3) Channel-B 的总 loss 建议写成可控组合（和你原本 reorder-GT 不冲突）

对每条 rollout 轨迹（或 best-of-N 选中的那条）：

[
L_{\text{B}}=
\underbrace{\alpha(R),L_{\text{shape}}}*{\text{OT边界/或bbox几何}}
+
\underbrace{\beta(R),L*{\text{SFT-on-rollout}}}_{\text{把采样分布拉回可解析轨道}}
]

其中：

* (L_{\text{shape}} =)

  * bbox：expected loss + 分布一致性（同 Channel-A）
  * poly：(L_{\text{OT}} +)（可选）poly shape 正则
* (L_{\text{SFT-on-rollout}})：保持你已有的“格式/结构严格 CE + FN append”等规则（用于纠错与语法稳态）

权重策略（建议文档写成简单 piecewise）：

* (R<r_0)：不更新或只更新格式 CE（避免把几何学坏）
* (R\ge r_0)：逐步增大 (\alpha(R))

---

## B4) matching（对象级）层面：bbox 可 Hungarian，poly 建议 OT/IoU 驱动

* 对象级匹配仍保留你原本 Hungarian（或加 dustbin 处理 unmatched）。
* poly 情况下，对象级成本可直接用 (1-\text{maskIoU})（stop-grad）或加一个 cheap 的 bbox IoU 作为快速近似。
* 若 early 容易错配：可以用 “软 OT 对齐矩阵（对象级）” 做权重稳定器，但**不一定必须**（优先把形状 OT 做好）。

---

# C. 训练配方与预算：把 Channel-B 明确写成“少量样本、少量步”的纠错器

你之前已经倾向：**热路径不 rollout，只少量走 Channel-B**。我建议计划书把它写得更“可执行”：

* Stage-2 主体：Channel-A 占 90%+ step（稳定几何/协议）
* Channel-B：只对

  1. 高不确定样本
  2. 解析失败/灾难态
  3. 或周期性抽样
     做 rollout + best-of-N + OT 纠错
     （并把它称作“budgeted correction loop”）

---

# D. 你可以在文档里加一张“统一训练流程”伪代码（便于读者理解）

**Channel-A：**

1. teacher-forcing（GT prefix）
2. N×forward 做 soft self-context（damping + truncated stop-grad）
3. 取最后一轮 logits → CoordExp → 连续几何
4. loss：expected loss（坐标） + 分布一致性（bbox inv）/shape 正则（poly）

**Channel-B：**

1. rollout 生成 pred（可 group sampling）
2. parse 成对象集合；rasterize poly 得 mask
3. reward：mask IoU（stop-grad）→ 选轨迹/定权重
4. teacher-force 复跑拿 logits → CoordExp → 顶点
5. loss：shape OT（点↔线段 Sinkhorn，detach plan） + 格式/结构 CE

---

# E. 建议你在计划书里明确的 4 个消融（写上去很加分）

1. **expected loss vs loss-on-expectation**（坐标分布是否更尖）
2. bbox 的 **inv penalty** 是否提升解析稳定与 mAP
3. poly：**OT 边界损失 vs 仅点级 L1/Huber**（不定长鲁棒性）
4. Channel-B 的 **reward gating/weight** 是否降低学坏与提升 pass@1（同样报告 best-of-N）

---

## 最后一句“改文档的抓手”

如果你只想在计划书里写一个“最小但强”的升级路径，我建议你把 Channel-A / B 的新增内容浓缩成两条：

* **Channel-A：expected loss + 分布一致性**（bbox inv；poly 轻量 shape 正则），强调“稳定、可解析、无 rollout”。
* **Channel-B：reward(mask IoU) + OT(Sinkhorn 边界) 可微回传**，强调“少量预算纠错，不需要长度对齐，不需要可微 rasterizer”。

这两条加进去，你的“统一架构”叙事会非常清晰：**SFT 打地基，EM-ish 纠错提效；bbox/poly 共用同一套坐标 token 与训练范式，只在损失/匹配层做最小扩展。**
