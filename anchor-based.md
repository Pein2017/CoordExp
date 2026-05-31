```md
可以。`anchor-base` 缓解多实例的核心不是“多了一个 token”，而是：

> **在 desc 之前，先给当前 object segment 一个粗空间身份。**

原格式：

```text
<object_ref_start>cat<box_start>x1 y1 x2 y2
```

遇到多个 cat 时，`cat` 之后仍然是多实例叠加态：

```text
C = {cat_1, cat_2, cat_3}
```

anchor-base：

```text
<object_ref_start><anchor_12_07>cat<box_start>x1 y1 x2 y2
```

在 `cat` 之前就先选了一个空间锚点：

```text
anchor_12_07 ≈ 当前 object 的中心点 / coarse location
```

于是 active candidate set 立刻从：

```text
R = {cat_1, cat_2, cat_3, dog_1, cup_1, ...}
```

缩小成：

```text
C(anchor_12_07) = {落在 anchor_12_07 附近的 objects}
```

如果这个 anchor cell 里只有 `cat_2`，那么后面的：

```text
cat<box_start>x1 y1 x2 y2
```

就已经绑定到 `cat_2` 了。

这正好对应你笔记里提到的 AR detection 隐含过程：模型先决定 object description，再决定 visual instance，再生成 bbox；而 `x1` 之后可能已经锁定实例。anchor-base 是把这个“锁定实例”的时刻前移到 desc 之前。 

---

# 1. desc-first 的多实例问题

假设图里有：

```text
cat_1: center=(120, 80),  box=[100, 60, 150, 110]
cat_2: center=(420, 90),  box=[390, 65, 460, 125]
cat_3: center=(430, 300), box=[400, 270, 470, 350]
```

desc-first 生成到：

```text
<object_ref_start>cat<box_start>
```

此时 `cat` 只是类别，不是实例。
active candidates 仍然是：

```text
C = {cat_1, cat_2, cat_3}
```

所以后续 bbox token 要承担两个任务：

```text
1. 精确定位 bbox
2. 从多个 cat 中完成 instance binding
```

这就是 binding 混乱的来源：

```text
cat token 语义上指向“猫这个类”
bbox token 才开始决定“哪一只猫”
```

如果坐标 token 局部 multiple-positive 做得不好，就容易出现：

```text
x1 倾向 cat_2
y1 倾向 cat_1
x2 倾向 cat_2
y2 倾向 cat_3
```

也就是你一直担心的“坐标拼盘”。

---

# 2. anchor-base 怎么缓解？

anchor-base 把序列改成：

```text
<object_ref_start><anchor>cat<box_start>x1 y1 x2 y2
```

其中 `<anchor>` 可以是：

```text
<grid_12_07>
```

或：

```text
<anchor_x_12><anchor_y_07>
```

表示 object center 落在某个 coarse spatial cell。

于是生成过程变成：

```text
Step 1:
  从 remaining objects 中选择一个粗空间位置

Step 2:
  在这个局部空间位置附近判断类别

Step 3:
  在这个 anchor 约束下生成精确 bbox
```

也就是因子分解从：

[
P(desc, box \mid I,p)
]

变成：

[
P(anchor, desc, box \mid I,p)
=============================

P(anchor \mid I,p)
P(desc \mid I,p,anchor)
P(box \mid I,p,anchor,desc)
]

这有一个很关键的变化：

> `desc` 不再负责区分多个同类实例；`anchor` 先负责把实例空间上分开。

---

# 3. 用 active candidate set 看最清楚

定义当前 prefix 下的 remaining GT set：

[
R(p)=G-\text{matched}(p,G)
]

在 object segment 开始时：

```text
prefix = ... <object_ref_start>
```

valid anchors 是：

[
A(p)={anchor(o): o\in R(p)}
]

模型输出一个 anchor 后：

```text
<object_ref_start><anchor_12_07>
```

active candidate set 变成：

[
C =
{o\in R(p): anchor(o)=anchor_12_07}
]

如果：

```text
C = {cat_2}
```

那后续就是 pure CE：

```text
cat<box_start>x1 y1 x2 y2
```

如果：

```text
C = {cat_2, cup_1}
```

那 desc 位置继续做 multiple-positive：

```text
valid desc = {cat, cup}
```

如果：

```text
C = {cat_2, cat_3}
```

说明 anchor 太粗，两个 cat 仍在同一 cell。
这时后面的 bbox tokens 继续消歧，但至少候选已经从全图所有 cat 缩小到了局部区域里的 cat。

所以 anchor 的作用不是保证一步唯一绑定，而是：

```text
global ambiguity
  ↓
local ambiguity
  ↓
bbox refinement
```

---

# 4. 它缓解多实例的四个机制

## 4.1 提前 spatial commitment

desc-first：

```text
cat
```

这个 token 对所有 cat 都成立。

anchor-first：

```text
<anchor_12_07>
```

这个 token 只对某个空间区域成立。

所以 anchor 把模型从“类别空间”先拉进“实例空间”。

对同类多实例来说，这是非常大的差异：

```text
cat_1, cat_2, cat_3 都共享 desc = cat
但它们通常不共享 anchor
```

因此 anchor 比 desc 更适合作为 instance identity 的初始分叉点。

---

## 4.2 降低 bbox token 的消歧压力

desc-first 下，bbox 既要定位又要消歧：

```text
P(x1,y1,x2,y2 | cat)
```

anchor-base 下，bbox 更多是在做 refinement：

```text
P(x1,y1,x2,y2 | anchor, cat)
```

这相当于把任务从：

```text
在全图所有 cat 中找一只，并给出 bbox
```

变成：

```text
在 anchor 附近找 cat，并精修 bbox
```

后者容易很多。

---

## 4.3 让 desc 变成局部分类，而不是全局检索

原来的 `cat` 更像：

```text
我要说一个 cat，但不知道哪只
```

anchor 后的 `cat` 更像：

```text
anchor_12_07 这个区域的 object 是 cat
```

这对多实例有利，因为类别预测变成了 region-conditioned classification：

[
P(desc \mid I,p,anchor)
]

而不是全局 object selection：

[
P(desc \mid I,p)
]

如果一个区域里只有一只猫，那么 `cat` 这个 token天然就和那只猫绑定了。

---

## 4.4 提供显式 coverage memory

已输出过：

```text
<anchor_12_07>cat<box>...
```

后续再生成 object 时，可以让模型或 decoder 看到：

```text
anchor_12_07 已经被覆盖
```

这能降低 duplicate。

当然，anchor 不是完美 coverage memory，因为一个 anchor cell 可能有多个 objects。但它比纯 desc 好很多：

```text
已输出 cat
```

这句话无法说明哪只 cat 已经输出。

而：

```text
已输出 anchor_12_07 + cat
```

至少说明某个空间区域里的 cat 已经被覆盖。

这与你笔记中提到的“模型知道物体存在但没有稳定枚举、缺少显式 coverage memory”的问题直接相关。

---

# 5. anchor-base 和 residual set / multiple-positive 的结合方式

它们不是冲突关系，反而很适配。

## 5.1 在 anchor 位置做 residual multiple-positive

当前 remaining objects：

```text
R(p) = {cat_1, cat_2, dog_1, cup_1}
```

它们的 anchor：

```text
cat_1 -> anchor_03_04
cat_2 -> anchor_12_07
dog_1 -> anchor_05_10
cup_1 -> anchor_12_07
```

那么在：

```text
<object_ref_start>
```

之后，valid anchor tokens 是：

```text
anchor_03_04
anchor_12_07
anchor_05_10
```

注意 `anchor_12_07` 下面有两个 objects，所以它的 target mass 应该更大。

定义：

[
q(a)
====

\frac{
\sum_{o\in R(p), anchor(o)=a} w_o
}{
\sum_{o\in R(p)} w_o
}
]

其中 (w_o) 可以是：

```text
普通 object: 1
hard FN: >1
small object: >1
rare class: >1
```

然后对 anchor token 做 soft CE：

[
L_{anchor}
==========

-\sum_a q(a)\log P_\theta(a\mid I,p,<object_ref_start>)
]

这样不会错误惩罚其它 remaining object 的 anchor。

这就是 anchor 版本的 residual-set prediction。

---

## 5.2 anchor 后继续维护 active candidates

输出 anchor 后：

```text
<object_ref_start><anchor_12_07>
```

候选集合变成：

```text
C = {cat_2, cup_1}
```

desc 位置的 target 是：

```text
valid desc = {cat, cup}
```

如果模型选了：

```text
cat
```

候选集合继续缩小：

```text
C = {cat_2}
```

后面 bbox 就可以 pure CE：

```text
<box_start>x1 y1 x2 y2
```

如果 anchor cell 内有两个 cat：

```text
C = {cat_2, cat_3}
```

那么选 `cat` 后仍然不唯一：

```text
C = {cat_2, cat_3}
```

这时 bbox tokens 继续做你的“叠加态 multiple-positive”，直到某个坐标 token 把候选坍缩到单个 instance。

所以完整规则是：

```text
每一步都维护 C = compatible remaining instances

如果 |C| > 1:
  multiple-positive / trie target

如果 |C| == 1:
  pure CE
```

anchor 只是让 `|C|` 更早变小。

---

# 6. 为什么它比 bbox-first 更温和？

你之前试过 bbox-first，mAP 低几个点。这个现象很合理。

bbox-first：

```text
<object_ref_start><box_start>x1 y1 x2 y2 cat
```

第一步就要求模型预测精确坐标 token，例如 `x1`。
这很难，因为：

```text
没有 desc query
没有 anchor coarse prior
直接从全图所有坐标 bin 中选一个
```

anchor-base：

```text
<object_ref_start><anchor>cat<box_start>x1 y1 x2 y2
```

第一步只预测 coarse anchor，不要求完整 bbox。

它的角色是：

```text
先粗绑定
再语义分类
再精细定位
```

所以它比 full bbox-first 更友好。

可以理解为：

```text
bbox-first:
  让模型第一个 token 就报精确地址

anchor-base:
  让模型先报大概街区，再说目标是什么，最后报门牌号
```

---

# 7. anchor 的粒度怎么选？

这是关键。

如果 anchor 太粗：

```text
很多 objects 落在同一 anchor
```

则消歧能力弱。

如果 anchor 太细：

```text
anchor token 太多
数据稀疏
first-token prediction 难
轻微 bbox 标注扰动会改变 anchor
```

建议第一版用 center grid：

```text
14 × 14 = 196 anchors
```

正好接近你说的 200 个 token。

anchor 定义：

[
cx=\frac{x_1+x_2}{2}, \quad cy=\frac{y_1+y_2}{2}
]

[
anchor = grid(cx,cy)
]

如果觉得 196 个新 token 太多，可以 factorize：

```text
<anchor_x_12><anchor_y_07>
```

这样只需要：

```text
14 + 14 = 28 tokens
```

但缺点是：

```text
x/y 分开后，第一个 token只绑定一半空间信息
```

我会优先试：

```text
完整 grid anchor token: <anchor_12_07>
```

因为它更像一个 object handle。

---

# 8. anchor-base 不能解决什么？

它不是银弹。它主要解决：

```text
同类多实例的早期绑定
desc-box mismatch
重复输出同一显著实例
坐标 token 承担过多消歧压力
```

但它不能完全解决：

```text
模型根本没看见小目标
anchor cell 内仍有多个相似实例
过早 EOS
类别体系不完整
prefix rollout 造成的 attention drift
```

尤其是 FN 问题，anchor 只是让“选择哪个实例”更清楚，不保证模型一定会继续枚举所有 remaining anchors。

所以还需要：

```text
remaining 非空时压 EOS
rollout-prefix training
hard FN anchor upweight
coverage-aware decoding
```

这和你笔记中对漏检原因的拆分是一致的：漏检可能不是不会定位，而是不会稳定枚举、stop/continue 校准不稳、prefix 改变后续注意力。

---

# 9. 一个推荐的训练目标

格式：

```text
<object_ref_start><anchor_ij>desc<box_start>x1 y1 x2 y2<object_ref_end>
```

训练时，对每个 object segment 内部维护 active candidates。

伪代码：

```python
def build_target(segment_prefix, remaining_objects):
    C = [
        obj for obj in remaining_objects
        if serialize_anchor_desc_box(obj).startswith(segment_prefix)
    ]

    if len(C) == 0:
        return invalid_or_ignore()

    if len(C) == 1:
        return pure_ce(next_token(C[0], segment_prefix))

    child_mass = defaultdict(float)

    for obj in C:
        tok = next_token(obj, segment_prefix)
        child_mass[tok] += weight[obj]

    target_dist = normalize(child_mass)

    return multi_positive_ce(target_dist)
```

这会自然得到：

```text
<object_ref_start> 后:
  多个 valid anchors

<anchor> 后:
  多个 valid desc

<anchor>cat 后:
  多个 valid box tokens, 如果 cell 里还有多个 cat

唯一实例后:
  pure CE
```

---

# 10. 推理时怎么用？

最简单是普通 greedy / beam decode，但在特定位置加约束：

```text
<object_ref_start> 后:
  只允许 anchor tokens 或 EOS

<anchor> 后:
  只允许 desc tokens

<box_start> 后:
  只允许 coordinate tokens
```

更强一点可以加 coverage penalty：

```text
如果 anchor 已经输出过:
  降低该 anchor 的概率
```

但要小心，一个 anchor cell 可能有多个 objects。更稳的是：

```text
已输出 anchor + desc + box 和新候选 IoU 高时，才判 duplicate
```

也可以做 segment beam：

```text
每次生成完整 object segment
按 segment score + coverage score 重排
```

这会让输出更像 object-level，而不是纯 token-level 抖动。

---

# 11. 推荐 ablation

你可以做这几组：

```text
A. desc-first baseline
<object_ref_start>cat<box_start>x1 y1 x2 y2

B. anchor-desc-box
<object_ref_start><anchor>cat<box_start>x1 y1 x2 y2

C. desc-anchor-box
<object_ref_start>cat<anchor><box_start>x1 y1 x2 y2

D. anchor-desc-box + residual multiple-positive
anchor / desc / bbox 都用 active candidate set

E. anchor-desc-box + hard-FN anchor upweight
对 rollout 漏掉的 object anchor 加权

F. anchor-desc-box + EOS continue loss
remaining 非空时压低 stop
```

我最想看的指标不是只看 mAP，而是：

```text
same-class multi-instance recall
desc-box mismatch rate
duplicate rate
FN recall by object size
premature EOS rate
anchor accuracy
anchor entropy
anchor collision rate
```

其中最关键的诊断是：

```text
anchor collision rate:
  一个 anchor cell 里平均有多少 GT objects
```

如果 collision 很低，anchor-base 会非常有效。
如果 collision 很高，说明 grid 太粗，或者场景太拥挤，需要更细 anchor / two-level anchor / desc 后再 anchor。

---

# 12. 最后的判断

`anchor-base` 缓解多实例的本质是：

> **把 instance binding 从 `cat<box_start>` 之后，提前到 `<object_ref_start><anchor>` 这一刻。**

它让模型先选空间身份，再选类别，再精修 bbox。

所以多实例情况下，原来：

```text
cat = 所有猫的共享语义入口
```

变成：

```text
anchor + cat = 某个空间区域里的猫
```

这就把一个全局多实例歧义，压缩成一个局部、通常更容易消解的歧义。

一句话压缩：

> **anchor-base 不是让模型更会说 `cat`，而是让每个 object segment 一开始就带上“我来自图上的哪一块区域”。这会显著减少同类多实例下 desc 和 bbox 的错绑，同时天然适配 residual set / multiple-positive：先对 remaining objects 的 anchors 做多正例，选中 anchor 后缩小 active set，唯一后再 fallback pure CE。**
```

```md
可以。`anchor-base` 缓解多实例的核心不是“多了一个 token”，而是：

> **在 desc 之前，先给当前 object segment 一个粗空间身份。**

原格式：

```text
<object_ref_start>cat<box_start>x1 y1 x2 y2
```

遇到多个 cat 时，`cat` 之后仍然是多实例叠加态：

```text
C = {cat_1, cat_2, cat_3}
```

anchor-base：

```text
<object_ref_start><anchor_12_07>cat<box_start>x1 y1 x2 y2
```

在 `cat` 之前就先选了一个空间锚点：

```text
anchor_12_07 ≈ 当前 object 的中心点 / coarse location
```

于是 active candidate set 立刻从：

```text
R = {cat_1, cat_2, cat_3, dog_1, cup_1, ...}
```

缩小成：

```text
C(anchor_12_07) = {落在 anchor_12_07 附近的 objects}
```

如果这个 anchor cell 里只有 `cat_2`，那么后面的：

```text
cat<box_start>x1 y1 x2 y2
```

就已经绑定到 `cat_2` 了。

这正好对应你笔记里提到的 AR detection 隐含过程：模型先决定 object description，再决定 visual instance，再生成 bbox；而 `x1` 之后可能已经锁定实例。anchor-base 是把这个“锁定实例”的时刻前移到 desc 之前。 

---

# 1. desc-first 的多实例问题

假设图里有：

```text
cat_1: center=(120, 80),  box=[100, 60, 150, 110]
cat_2: center=(420, 90),  box=[390, 65, 460, 125]
cat_3: center=(430, 300), box=[400, 270, 470, 350]
```

desc-first 生成到：

```text
<object_ref_start>cat<box_start>
```

此时 `cat` 只是类别，不是实例。
active candidates 仍然是：

```text
C = {cat_1, cat_2, cat_3}
```

所以后续 bbox token 要承担两个任务：

```text
1. 精确定位 bbox
2. 从多个 cat 中完成 instance binding
```

这就是 binding 混乱的来源：

```text
cat token 语义上指向“猫这个类”
bbox token 才开始决定“哪一只猫”
```

如果坐标 token 局部 multiple-positive 做得不好，就容易出现：

```text
x1 倾向 cat_2
y1 倾向 cat_1
x2 倾向 cat_2
y2 倾向 cat_3
```

也就是你一直担心的“坐标拼盘”。

---

# 2. anchor-base 怎么缓解？

anchor-base 把序列改成：

```text
<object_ref_start><anchor>cat<box_start>x1 y1 x2 y2
```

其中 `<anchor>` 可以是：

```text
<grid_12_07>
```

或：

```text
<anchor_x_12><anchor_y_07>
```

表示 object center 落在某个 coarse spatial cell。

于是生成过程变成：

```text
Step 1:
  从 remaining objects 中选择一个粗空间位置

Step 2:
  在这个局部空间位置附近判断类别

Step 3:
  在这个 anchor 约束下生成精确 bbox
```

也就是因子分解从：

[
P(desc, box \mid I,p)
]

变成：

[
P(anchor, desc, box \mid I,p)
=============================

P(anchor \mid I,p)
P(desc \mid I,p,anchor)
P(box \mid I,p,anchor,desc)
]

这有一个很关键的变化：

> `desc` 不再负责区分多个同类实例；`anchor` 先负责把实例空间上分开。

---

# 3. 用 active candidate set 看最清楚

定义当前 prefix 下的 remaining GT set：

[
R(p)=G-\text{matched}(p,G)
]

在 object segment 开始时：

```text
prefix = ... <object_ref_start>
```

valid anchors 是：

[
A(p)={anchor(o): o\in R(p)}
]

模型输出一个 anchor 后：

```text
<object_ref_start><anchor_12_07>
```

active candidate set 变成：

[
C =
{o\in R(p): anchor(o)=anchor_12_07}
]

如果：

```text
C = {cat_2}
```

那后续就是 pure CE：

```text
cat<box_start>x1 y1 x2 y2
```

如果：

```text
C = {cat_2, cup_1}
```

那 desc 位置继续做 multiple-positive：

```text
valid desc = {cat, cup}
```

如果：

```text
C = {cat_2, cat_3}
```

说明 anchor 太粗，两个 cat 仍在同一 cell。
这时后面的 bbox tokens 继续消歧，但至少候选已经从全图所有 cat 缩小到了局部区域里的 cat。

所以 anchor 的作用不是保证一步唯一绑定，而是：

```text
global ambiguity
  ↓
local ambiguity
  ↓
bbox refinement
```

---

# 4. 它缓解多实例的四个机制

## 4.1 提前 spatial commitment

desc-first：

```text
cat
```

这个 token 对所有 cat 都成立。

anchor-first：

```text
<anchor_12_07>
```

这个 token 只对某个空间区域成立。

所以 anchor 把模型从“类别空间”先拉进“实例空间”。

对同类多实例来说，这是非常大的差异：

```text
cat_1, cat_2, cat_3 都共享 desc = cat
但它们通常不共享 anchor
```

因此 anchor 比 desc 更适合作为 instance identity 的初始分叉点。

---

## 4.2 降低 bbox token 的消歧压力

desc-first 下，bbox 既要定位又要消歧：

```text
P(x1,y1,x2,y2 | cat)
```

anchor-base 下，bbox 更多是在做 refinement：

```text
P(x1,y1,x2,y2 | anchor, cat)
```

这相当于把任务从：

```text
在全图所有 cat 中找一只，并给出 bbox
```

变成：

```text
在 anchor 附近找 cat，并精修 bbox
```

后者容易很多。

---

## 4.3 让 desc 变成局部分类，而不是全局检索

原来的 `cat` 更像：

```text
我要说一个 cat，但不知道哪只
```

anchor 后的 `cat` 更像：

```text
anchor_12_07 这个区域的 object 是 cat
```

这对多实例有利，因为类别预测变成了 region-conditioned classification：

[
P(desc \mid I,p,anchor)
]

而不是全局 object selection：

[
P(desc \mid I,p)
]

如果一个区域里只有一只猫，那么 `cat` 这个 token天然就和那只猫绑定了。

---

## 4.4 提供显式 coverage memory

已输出过：

```text
<anchor_12_07>cat<box>...
```

后续再生成 object 时，可以让模型或 decoder 看到：

```text
anchor_12_07 已经被覆盖
```

这能降低 duplicate。

当然，anchor 不是完美 coverage memory，因为一个 anchor cell 可能有多个 objects。但它比纯 desc 好很多：

```text
已输出 cat
```

这句话无法说明哪只 cat 已经输出。

而：

```text
已输出 anchor_12_07 + cat
```

至少说明某个空间区域里的 cat 已经被覆盖。

这与你笔记中提到的“模型知道物体存在但没有稳定枚举、缺少显式 coverage memory”的问题直接相关。

---

# 5. anchor-base 和 residual set / multiple-positive 的结合方式

它们不是冲突关系，反而很适配。

## 5.1 在 anchor 位置做 residual multiple-positive

当前 remaining objects：

```text
R(p) = {cat_1, cat_2, dog_1, cup_1}
```

它们的 anchor：

```text
cat_1 -> anchor_03_04
cat_2 -> anchor_12_07
dog_1 -> anchor_05_10
cup_1 -> anchor_12_07
```

那么在：

```text
<object_ref_start>
```

之后，valid anchor tokens 是：

```text
anchor_03_04
anchor_12_07
anchor_05_10
```

注意 `anchor_12_07` 下面有两个 objects，所以它的 target mass 应该更大。

定义：

[
q(a)
====

\frac{
\sum_{o\in R(p), anchor(o)=a} w_o
}{
\sum_{o\in R(p)} w_o
}
]

其中 (w_o) 可以是：

```text
普通 object: 1
hard FN: >1
small object: >1
rare class: >1
```

然后对 anchor token 做 soft CE：

[
L_{anchor}
==========

-\sum_a q(a)\log P_\theta(a\mid I,p,<object_ref_start>)
]

这样不会错误惩罚其它 remaining object 的 anchor。

这就是 anchor 版本的 residual-set prediction。

---

## 5.2 anchor 后继续维护 active candidates

输出 anchor 后：

```text
<object_ref_start><anchor_12_07>
```

候选集合变成：

```text
C = {cat_2, cup_1}
```

desc 位置的 target 是：

```text
valid desc = {cat, cup}
```

如果模型选了：

```text
cat
```

候选集合继续缩小：

```text
C = {cat_2}
```

后面 bbox 就可以 pure CE：

```text
<box_start>x1 y1 x2 y2
```

如果 anchor cell 内有两个 cat：

```text
C = {cat_2, cat_3}
```

那么选 `cat` 后仍然不唯一：

```text
C = {cat_2, cat_3}
```

这时 bbox tokens 继续做你的“叠加态 multiple-positive”，直到某个坐标 token 把候选坍缩到单个 instance。

所以完整规则是：

```text
每一步都维护 C = compatible remaining instances

如果 |C| > 1:
  multiple-positive / trie target

如果 |C| == 1:
  pure CE
```

anchor 只是让 `|C|` 更早变小。

---

# 6. 为什么它比 bbox-first 更温和？

你之前试过 bbox-first，mAP 低几个点。这个现象很合理。

bbox-first：

```text
<object_ref_start><box_start>x1 y1 x2 y2 cat
```

第一步就要求模型预测精确坐标 token，例如 `x1`。
这很难，因为：

```text
没有 desc query
没有 anchor coarse prior
直接从全图所有坐标 bin 中选一个
```

anchor-base：

```text
<object_ref_start><anchor>cat<box_start>x1 y1 x2 y2
```

第一步只预测 coarse anchor，不要求完整 bbox。

它的角色是：

```text
先粗绑定
再语义分类
再精细定位
```

所以它比 full bbox-first 更友好。

可以理解为：

```text
bbox-first:
  让模型第一个 token 就报精确地址

anchor-base:
  让模型先报大概街区，再说目标是什么，最后报门牌号
```

---

# 7. anchor 的粒度怎么选？

这是关键。

如果 anchor 太粗：

```text
很多 objects 落在同一 anchor
```

则消歧能力弱。

如果 anchor 太细：

```text
anchor token 太多
数据稀疏
first-token prediction 难
轻微 bbox 标注扰动会改变 anchor
```

建议第一版用 center grid：

```text
14 × 14 = 196 anchors
```

正好接近你说的 200 个 token。

anchor 定义：

[
cx=\frac{x_1+x_2}{2}, \quad cy=\frac{y_1+y_2}{2}
]

[
anchor = grid(cx,cy)
]

如果觉得 196 个新 token 太多，可以 factorize：

```text
<anchor_x_12><anchor_y_07>
```

这样只需要：

```text
14 + 14 = 28 tokens
```

但缺点是：

```text
x/y 分开后，第一个 token只绑定一半空间信息
```

我会优先试：

```text
完整 grid anchor token: <anchor_12_07>
```

因为它更像一个 object handle。

---

# 8. anchor-base 不能解决什么？

它不是银弹。它主要解决：

```text
同类多实例的早期绑定
desc-box mismatch
重复输出同一显著实例
坐标 token 承担过多消歧压力
```

但它不能完全解决：

```text
模型根本没看见小目标
anchor cell 内仍有多个相似实例
过早 EOS
类别体系不完整
prefix rollout 造成的 attention drift
```

尤其是 FN 问题，anchor 只是让“选择哪个实例”更清楚，不保证模型一定会继续枚举所有 remaining anchors。

所以还需要：

```text
remaining 非空时压 EOS
rollout-prefix training
hard FN anchor upweight
coverage-aware decoding
```

这和你笔记中对漏检原因的拆分是一致的：漏检可能不是不会定位，而是不会稳定枚举、stop/continue 校准不稳、prefix 改变后续注意力。

---

# 9. 一个推荐的训练目标

格式：

```text
<object_ref_start><anchor_ij>desc<box_start>x1 y1 x2 y2<object_ref_end>
```

训练时，对每个 object segment 内部维护 active candidates。

伪代码：

```python
def build_target(segment_prefix, remaining_objects):
    C = [
        obj for obj in remaining_objects
        if serialize_anchor_desc_box(obj).startswith(segment_prefix)
    ]

    if len(C) == 0:
        return invalid_or_ignore()

    if len(C) == 1:
        return pure_ce(next_token(C[0], segment_prefix))

    child_mass = defaultdict(float)

    for obj in C:
        tok = next_token(obj, segment_prefix)
        child_mass[tok] += weight[obj]

    target_dist = normalize(child_mass)

    return multi_positive_ce(target_dist)
```

这会自然得到：

```text
<object_ref_start> 后:
  多个 valid anchors

<anchor> 后:
  多个 valid desc

<anchor>cat 后:
  多个 valid box tokens, 如果 cell 里还有多个 cat

唯一实例后:
  pure CE
```

---

# 10. 推理时怎么用？

最简单是普通 greedy / beam decode，但在特定位置加约束：

```text
<object_ref_start> 后:
  只允许 anchor tokens 或 EOS

<anchor> 后:
  只允许 desc tokens

<box_start> 后:
  只允许 coordinate tokens
```

更强一点可以加 coverage penalty：

```text
如果 anchor 已经输出过:
  降低该 anchor 的概率
```

但要小心，一个 anchor cell 可能有多个 objects。更稳的是：

```text
已输出 anchor + desc + box 和新候选 IoU 高时，才判 duplicate
```

也可以做 segment beam：

```text
每次生成完整 object segment
按 segment score + coverage score 重排
```

这会让输出更像 object-level，而不是纯 token-level 抖动。

---

# 11. 推荐 ablation

你可以做这几组：

```text
A. desc-first baseline
<object_ref_start>cat<box_start>x1 y1 x2 y2

B. anchor-desc-box
<object_ref_start><anchor>cat<box_start>x1 y1 x2 y2

C. desc-anchor-box
<object_ref_start>cat<anchor><box_start>x1 y1 x2 y2

D. anchor-desc-box + residual multiple-positive
anchor / desc / bbox 都用 active candidate set

E. anchor-desc-box + hard-FN anchor upweight
对 rollout 漏掉的 object anchor 加权

F. anchor-desc-box + EOS continue loss
remaining 非空时压低 stop
```

我最想看的指标不是只看 mAP，而是：

```text
same-class multi-instance recall
desc-box mismatch rate
duplicate rate
FN recall by object size
premature EOS rate
anchor accuracy
anchor entropy
anchor collision rate
```

其中最关键的诊断是：

```text
anchor collision rate:
  一个 anchor cell 里平均有多少 GT objects
```

如果 collision 很低，anchor-base 会非常有效。
如果 collision 很高，说明 grid 太粗，或者场景太拥挤，需要更细 anchor / two-level anchor / desc 后再 anchor。

---

# 12. 最后的判断

`anchor-base` 缓解多实例的本质是：

> **把 instance binding 从 `cat<box_start>` 之后，提前到 `<object_ref_start><anchor>` 这一刻。**

它让模型先选空间身份，再选类别，再精修 bbox。

所以多实例情况下，原来：

```text
cat = 所有猫的共享语义入口
```

变成：

```text
anchor + cat = 某个空间区域里的猫
```

这就把一个全局多实例歧义，压缩成一个局部、通常更容易消解的歧义。

一句话压缩：

> **anchor-base 不是让模型更会说 `cat`，而是让每个 object segment 一开始就带上“我来自图上的哪一块区域”。这会显著减少同类多实例下 desc 和 bbox 的错绑，同时天然适配 residual set / multiple-positive：先对 remaining objects 的 anchors 做多正例，选中 anchor 后缩小 active set，唯一后再 fallback pure CE。**
```

``md
是的，`<object_ref_start><anchor>cat<box_start>x1y1x2y2` 的 anchor-base 方式可以非常自然地缓解多实例问题。它的核心作用不是“提高格式稳定性”，而是把每一行 object row 的第一步从：

[
\text{我要生成什么类别？}
]

改成：

[
\text{我要读出图像中的哪个空间 basin / coarse instance region？}
]

这相当于在纯文本序列里插入了一个**离散化的 early binding variable**。

---

## 1. desc-first 的多实例歧义在哪里

原始格式：

```text
<object_ref_start> cat <box_start> x1 y1 x2 y2
```

如果图里有多个 cat：

```text
cat_A: [10,20,30,40]
cat_B: [50,60,70,80]
cat_C: [52,75,71,91]
```

当模型生成：

```text
<object_ref_start> cat
```

时，`cat` 不是 instance identity。它只把模型带进了“cat 这个类别流型”：

[
A(\text{cat})={cat_A,cat_B,cat_C}
]

接下来 (x_1) 才开始分叉。如果模型已经在某个 cat_A 的局部 bbox basin 里概率很高，它就可能反复走：

```text
cat → cat_A bbox
cat → cat_A jitter bbox
cat → cat_A jitter bbox
```

也就是 same-desc duplication burst。

---

## 2. anchor-first 如何改变因果结构

anchor-base 格式：

```text
<object_ref_start> <anchor_07_12> cat <box_start> x1 y1 x2 y2
```

概率分解变成：

[
p(r\mid c)
==========

p(a\mid c)
p(d\mid c,a)
p(b\mid c,a,d)
]

其中 (a) 是 anchor token。这个 (a) 可以是图像上的 coarse grid cell、multi-scale anchor、中心点区域，或者更复杂的 learned spatial bin。

这样一来，same-desc 多实例不再首先靠 `cat` 区分，而是先靠空间 anchor 区分：

```text
cat_A → <anchor_02_03>
cat_B → <anchor_07_12>
cat_C → <anchor_08_12>
```

于是：

```text
<object_ref_start><anchor_07_12>
```

之后 active set 已经从全体 objects 收缩到：

[
A(a)={j: a\in \operatorname{AnchorSet}(b_j)}
]

再生成 `cat` 时，它已经不是“所有 cat 里的 cat”，而是“anchor_07_12 附近的 cat”。

这就是它缓解多实例的第一性原理：**先空间绑定，再语义描述，再精确定位**。

这和 DAB-DETR 的直觉有相通处：DAB-DETR 直接使用 box coordinates 作为 DETR decoder queries，认为显式位置先验可以改善 query-to-feature similarity，并用 box width/height 调制 positional attention map。你的 anchor token 是一个更轻量的、离散化的空间 query。([arXiv][1])

---

## 3. 它为什么比 bbox-first 温和

bbox-first 是：

```text
<object_ref_start> x1 y1 x2 y2 cat
```

问题是第一步就要预测精确坐标，熵太高，而且 desc 的语义帮助来得太晚。

anchor-first 是：

```text
<object_ref_start> <anchor> cat <box_start> x1 y1 x2 y2
```

它只要求第一步选择粗位置：

[
a \approx \operatorname{coarse}(c_x,c_y)
]

然后仍然让 desc 帮助定位：

[
p(b\mid a,d,c)
]

所以它像 bbox-first 的低熵版本。它保留了 bbox-first 的 early spatial binding，又保留了 desc-first 的 semantic conditioning。

一个直觉图：

```text
desc-first:
    cat → 在所有 cat 里找一个 box
    容易 same-desc unbinding

bbox-first:
    exact box → 再分类
    binding 强，但 first token 太难

anchor-first:
    coarse location → cat → exact box
    binding 较强，定位难度较低
```

---

## 4. anchor 如何和 residual / multiple-positive 结合

这是关键。anchor token 不能只是普通 SFT token，否则它只是换了一个输出格式。它应该接入你的 residual set 机制。

设当前 noisy/self prefix 匹配后得到：

[
C_t=\text{covered instances}
]

[
U_t=G\setminus C_t
]

对每个 GT object (j)，定义它允许的 anchor 集合：

[
A_j=\operatorname{AnchorSet}(b_j)
]

例如取 bbox center 最近的 top-k grid anchors：

[
A_j=\operatorname{TopKNearestAnchors}(c_j)
]

那么在 `<object_ref_start>` 后的 anchor 位置：

正例 anchor：

[
A^+*t=\bigcup*{j\in U_t} A_j
]

covered-only 负例 anchor：

[
A^-*t=
\left(
\bigcup*{k\in C_t} A_k
\right)
\setminus
\left(
\bigcup_{j\in U_t} A_j
\right)
]

loss 可以写成：

[
\mathcal L_{\text{anchor}}
==========================

-\log
\sum_{a\in A^+*t}
p*\theta(a\mid c_t)
-------------------

\lambda
\log
\left(
1-\sum_{a\in A^-*t}
p*\theta(a\mid c_t)
\right)
]

这里最重要的是第二项。它告诉模型：

> 已经 covered 的区域对应的 anchor，不应该继续拿走概率质量。

但如果某个 anchor 同时覆盖了已生成实例和未生成实例，它不会被强惩罚，因为它可能是真实 dense ambiguity。这能避免误伤密集相邻物体。

---

## 5. 选择 anchor 后，active set 立即缩小

假设模型选择了：

```text
<anchor_07_12>
```

那么后续 active uncovered set 变成：

[
U_t(a)={j\in U_t: a\in A_j}
]

后面 desc 和 bbox 都只在这个局部 residual set 上做 multiple-positive / trie CE。

也就是：

```text
<object_ref_start>
    anchor: 在 remaining object regions 中选一个 basin

<anchor_07_12>
    desc: 在这个 basin 内选择类别/描述

cat
    bbox: 在 anchor + desc 约束下精确定位
```

如果 (U_t(a)) 只有一个实例，后续可以 pure CE：

[
|U_t(a)|=1 \Rightarrow \text{pure SFT for desc + bbox}
]

如果 (U_t(a)) 里仍有多个实例，比如密集小物体共享一个 anchor，则继续 multiple-positive：

[
|U_t(a)|>1 \Rightarrow \text{residual trie CE}
]

所以 anchor 不替代你的 residual set，而是把 residual set 的分叉提前到了第一步。

---

## 6. 一个具体例子

图里有：

```text
covered:
cat_A: anchor_02_03, box=[10,20,30,40]

uncovered:
cat_B: anchor_07_12, box=[50,60,70,80]
cat_C: anchor_08_12, box=[52,75,71,91]
dog_D: anchor_11_05, box=[80,30,95,48]
```

当前 prefix 后，模型来到：

```text
<object_ref_start>
```

anchor 位置的正例：

```text
anchor_07_12, anchor_08_12, anchor_11_05
```

covered-only 负例：

```text
anchor_02_03
```

所以模型被训练成：

```text
P(anchor_07_12 or anchor_08_12 or anchor_11_05) ↑
P(anchor_02_03) ↓
```

如果它选了：

```text
<anchor_07_12>
```

active set 变成：

```text
cat_B
```

然后：

```text
cat <box_start> 50 60 70 80
```

可以 pure CE。

如果它选了：

```text
<anchor_08_12>
```

active set 是：

```text
cat_C
```

如果两个 cat 很近，都允许 `anchor_07_12`，那么 active set 是：

```text
cat_B, cat_C
```

后续 `cat` 还不能唯一绑定，继续到 bbox slot 才逐渐坍缩。

---

## 7. anchor-base 为什么能减少 duplication burst

duplication burst 的动力学通常是：

```text
row t:
cat_A

row t+1:
模型仍然觉得 cat_A basin 概率最高
→ cat_A jitter

row t+2:
cat_A basin 又被前文强化
→ cat_A jitter again
```

anchor-base 给这个循环加了一个更早的阀门。

原来重复路径在 `cat` 之后才暴露：

```text
<object_ref_start> cat <box_start> x1...
```

现在重复路径在 anchor token 就暴露：

```text
<object_ref_start> <anchor_of_cat_A> ...
```

如果 cat_A 已经 covered，那么 `<anchor_of_cat_A>` 可以在第一步被压低。

这意味着模型不必等到 (x_1) 才发现自己回到了旧 basin，而是在 row 的第一个内容 token 就被引导离开：

[
p(a\in A_C) \downarrow
]

[
p(a\in A_U) \uparrow
]

它把“逃离旧 basin”的时间点提前了。

---

## 8. 它也能缓解边角 basin

如果有边角错误模式：

```text
person <box_start> 0 0 42 51
person <box_start> 0 1 45 53
```

desc-first 下，模型可能先生成 `person`，然后 (x_1=0) 进入边界坐标 basin。

anchor-first 下，边角 basin 对应某些 anchor：

```text
anchor_00_00
anchor_00_01
anchor_01_00
```

如果这些区域没有 uncovered GT 支持，它们就不是正例 anchor，甚至可以作为 background/boundary negative：

[
A_{\text{bg}}={\text{boundary anchors without GT support}}
]

加入：

[
-\lambda_B\log\left(1-\sum_{a\in A_{\text{bg}}}p(a)\right)
]

这样 `[0,0,x,y]` 的错误趋势在第一个 anchor token 就被压下去，而不是等坐标全生成完再处理。

---

## 9. 但 anchor-base 不能单独根治 coverage

需要特别强调：

> anchor-base 缓解 instance binding，但不自动保证 without-replacement sampling。

如果没有 residual / covered negative，模型仍然可能：

```text
<object_ref_start><anchor_07_12> cat box_B
<object_ref_start><anchor_07_12> cat box_B_jitter
<object_ref_start><anchor_07_12> cat box_B_jitter
```

所以 anchor token 本身不是 magic slot。它需要训练目标告诉它：

[
\text{covered anchor mass should be depleted}
]

DETR 之所以不只是“有 query”，而是能抑制重复，关键还包括 set prediction 和 bipartite matching。DETR 原论文把检测建模成 direct set prediction，并用 bipartite matching loss 做唯一分配。([arXiv][2]) DN-DETR 进一步说明，给 DETR decoder 喂带噪 GT boxes 并训练重建原始 boxes，可以缓解 matching 不稳定并加速收敛。([arXiv][3]) 这对你这里的启发是：anchor token 要配合 residual matching / noisy-prefix denoising，而不是裸用。

---

## 10. anchor 设计：不要太粗，也不要太细

如果 anchor 太粗：

```text
8×8 = 64 anchors
```

多个 dense objects 会落在同一个 anchor，binding 作用弱。

如果 anchor 太细：

```text
32×32 = 1024 anchors
```

第一个 token 分类空间太大，训练难，且小物体标注 jitter 会让 anchor label 不稳定。

一个比较实用的起点：

```text
14×14 = 196 anchors
```

刚好接近你说的 200 个 token。

对每个 object 不要只给一个 anchor，建议 top-k：

```text
center 最近的 4 个 anchors
```

这样标注边界和量化误差不会太硬。

可以进一步做 multi-scale anchor：

```text
<anchor_s0_iy_ix> 小物体
<anchor_s1_iy_ix> 中物体
<anchor_s2_iy_ix> 大物体
```

但第一版先别复杂化。先做 14×14 center anchor + top-k 正例就很好。

---

## 11. 和 desc / bbox trie 的完整训练流程

一个推荐训练状态：

```text
prefix = noisy or clean previous rows
matching(prefix, GT) → covered C, uncovered U
next row starts at <object_ref_start>
```

### Step 1：anchor residual CE

[
A^+=\bigcup_{j\in U}A_j
]

[
A^-=\left(\bigcup_{k\in C}A_k\right)\setminus A^+
]

训练：

```text
anchor positives = uncovered anchors
anchor negatives = covered-only anchors
```

### Step 2：anchor-conditioned active set

选一个 training anchor (a\in A_j)，其中 (j\in U)。

[
U(a)={j\in U:a\in A_j}
]

### Step 3：desc residual CE

desc 的 positive set 不是全图 remaining desc，而是：

[
D^+(a)={d_j:j\in U(a)}
]

如果 anchor 里只有一个 cat，那么 desc pure CE。
如果 anchor 内有 cat/person 等多个对象，则 multiple-positive。

### Step 4：bbox residual trie

bbox slot 只在 (U(a,d)) 上做：

[
U(a,d)={j\in U(a):d_j\approx d}
]

然后：

```text
x1: multiple-positive over U(a,d)
y1: active set further shrinks
x2/y2: until unique binding
```

这会比原始 desc-first 更稳定，因为 active set 一开始就小得多。

---

## 12. 你可以加一个 segment-level anchor loss

更强的版本是把：

```text
<anchor> + desc + bbox
```

当作一个 row segment action。

对每个未覆盖实例 (j) 和其 anchor (a\in A_j)：

[
s(a,j)
======

\log p(a\mid c)
+
\log p(d_j\mid c,a)
+
\log p(b_j\mid c,a,d_j)
]

正例：

[
j\in U
]

covered duplicate 负例：

[
j\in C
]

loss：

[
\mathcal L_{\text{anchor-seg}}
==============================

-\log
\frac{
\sum_{j\in U}\sum_{a\in A_j}\exp(s(a,j)/\tau)
}{
\sum_{j\in U}\sum_{a\in A_j}\exp(s(a,j)/\tau)
+
\lambda
\sum_{k\in C}\sum_{a\in A_k}\exp(s(a,k)/\tau)
}
]

这会直接优化：

```text
未覆盖 anchor + 未覆盖 row
```

高于：

```text
已覆盖 anchor + duplicate row
```

它是纯 token loss 层面的 implicit binding，不需要 hidden head。

---

## 13. 最简伪代码

```python
def anchor_base_loss(
    logits_anchor,
    gt_objects,
    covered_ids,
    uncovered_ids,
    obj_to_anchors,
):
    probs = logits_anchor.softmax(dim=-1)

    pos = set()
    for j in uncovered_ids:
        pos |= obj_to_anchors[j]

    cov = set()
    for j in covered_ids:
        cov |= obj_to_anchors[j]

    # anchors that only explain covered objects
    neg = cov - pos

    loss = -torch.log(probs[list(pos)].sum().clamp_min(1e-8))

    if len(neg) > 0:
        neg_mass = probs[list(neg)].sum().clamp(max=1 - 1e-6)
        loss += -lambda_cov * torch.log1p(-neg_mass)

    return loss
```

选择训练 anchor 后：

```python
chosen_obj = sample_from(uncovered_ids)
chosen_anchor = sample_from(obj_to_anchors[chosen_obj])

active = [
    j for j in uncovered_ids
    if chosen_anchor in obj_to_anchors[j]
]

loss_desc_bbox = residual_trie_ce(
    prefix=f"<object_ref_start>{chosen_anchor}",
    active_objects=active,
)
```

---

## 14. 总结一句话

anchor-base 缓解多实例的方式是：

> 把每一行 detection 的第一个选择，从“类别 token”改成“空间 anchor token”，让 same-desc 实例在生成 desc 之前就按空间 basin 分叉；再用 residual set 把未覆盖 anchor 当正例、已覆盖 anchor 当负例，从而把 duplicate basin 在最早阶段压下去。

它不是 NMS，也不是纯 bbox-first。它更像一个轻量、离散、可训练的 early instance binding：

```text
<object_ref_start>
    ↓
<anchor>       先选未覆盖空间区域
    ↓
cat            再说语义
    ↓
bbox           最后精修坐标
```

这样模型不再是“先喊 cat，再在一群 cat 里迷路”，而是“先站到一个区域，再给那个区域里的东西命名”。
```