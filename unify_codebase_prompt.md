你现在在一个基于 Qwen3-VL / Qwen-VL 类 autoregressive VLM 的 detection / dense captioning 微调项目中工作。当前 codebase 已经支持某种 `multi-positive` / `multiple-positive` / `ET-RMP-CE` 训练逻辑，也似乎同时支持 full-sequence teacher forcing 和 prefix sampling / roll-in，但目前命名、概念边界、配置结构和文档都还不统一。

这次任务的目标不是重新发明算法，也不是大改模型架构，而是：

1. 先调研当前 code infrastructure；
2. 从文档开始，把当前 `multi-positive` 架构的整体算法设计、数学定义、训练流程和配置概念统一下来；
3. 再检查 code implementation 是否和统一后的概念一致；
4. 最后让 full-sequence policy 和 prefix-rollin policy 成为可以混合的、可调控的设置。

请优先做“文档与概念统一”，再考虑最小代码整理。不要直接重构训练框架，不要新增 detection head，不要改变 Qwen3-VL 原始 autoregressive decode 机制。

---

# 一、核心背景

任务是让 Qwen3-VL 通过 autoregressive text generation 输出一组 detection objects。每张图片的标注是一个 object set：

    Y = {y_1, ..., y_N}

每个 object 包含类别、bbox，以及可能的额外属性：

    y_i = (class_i, bbox_i, ...)

每个 object 会被序列化成 token span：

    z_i = Tok(Serialize(y_i))

例如：

    {"label": "scratch", "bbox": [123, 456, 170, 500]}

或者项目内已有的其他格式。

问题是：detection label 本质上是 unordered set，但 Qwen3-VL decoder 是 left-to-right autoregressive LM。因此，传统 fixed-order SFT 会人为引入 object ordering bias。我们希望通过 `multi-positive / trie-based CE` 和可混合的 roll-in policy，让模型在任意合法 prefix 下学习 stable continuation，同时尽量保持 Qwen3-VL 原本的训练-解码一致性。

---

# 二、需要统一的核心概念

请在文档中明确区分两个正交概念：

## 1. Roll-in / State Sampling Policy

它决定训练时模型看到哪些 prefix state。

当前需要统一两类 policy：

### A. Full Sequence Teacher-Forcing Policy

从空 assistant response 开始，随机打乱 object order，构造完整 sequence：

    z_{π1}, z_{π2}, ..., z_{πN}

然后对整条 assistant completion 做 teacher forcing。

它的作用：

- 保持完整生成能力；
- 保持普通 SFT 的稳定性；
- 训练从空 prompt 开始完整输出 object set；
- 提供高 token efficiency；
- 作为 autoregressive detection 训练的主干分布。

### B. Prefix Sampling / Roll-in Policy

先采样一个已经 emitted 的 object subset：

    S ⊆ Y

将 S 随机排序并序列化为 assistant prefix：

    h = Serialize(perm(S))

这个 prefix 作为 roll-in context 出现在 input 中，但 prefix token 的 loss 应该 mask 掉。然后只训练 remaining objects：

    R = Y \ S

它的作用：

- 直接训练 permutation tree 上的中间 prefix states；
- 让模型学习“任意 object subset 已经输出后，如何继续输出 remaining objects”；
- 降低对某个 canonical order 的依赖；
- 支持后续 prefix jitter / model-generated prefix recovery；
- 改善 decode-time prefix mismatch、duplication、omission、early stop 等问题。

重要：prefix sampling 不是一个新的 loss，它主要改变训练时的 state distribution / occupancy measure。

---

## 2. Local Target Policy

它决定在一个具体 token position 上如何构造 next-token supervision。

### A. Hard CE

如果当前位置只有唯一合法 next token，则退化为普通 one-hot CE。

### B. Trie-based Multi-Positive CE

在某个 prefix state 下，可能存在多个合法 next tokens。此时不要把其他合法 token 当负例，而是构造 multi-positive / soft target。

设当前已经 emitted 的 object set 为：

    S

remaining objects：

    R = Y \ S

当前正在生成下一个 object，object 内部已有 partial token prefix：

    u

仍然可能匹配的 remaining objects：

    C(u) = { i in R : u is prefix of z_i }

如果下一个 token 是 a：

    C(u + a) = { i in C(u) : z_i[|u| + 1] == a }

那么 trie target 定义为：

    q_trie(a | S, u)
      = sum_{i in C(u+a)} w_i / sum_{i in C(u)} w_i

最简单情况下：

    w_i = 1

此时 target 表示 uniform over remaining objects，而不是 uniform over unique next tokens。

loss：

    L_trie
      = - sum_a q_trie(a | S, u) * log p_theta(a | image, prefix)

如果 legal next token 唯一，则 q_trie 自动退化为 one-hot hard CE。

请在文档中强调：

- multi-positive CE 是 local next-token CE 的推广；
- 它保持 autoregressive 训练和 decode 的一致性；
- 它不同于 segment-level score loss；
- 它也不同于 valid-mass / logsum objective。

### C. Optional Mixed Target

可以支持 hard CE 与 trie CE 的混合：

    q_mix = (1 - λ) * one_hot(hard_token) + λ * q_trie

    L_mix = - sum_a q_mix(a) log p_theta(a)

λ 可以作为配置项控制 multi-positive 强度。

---

# 三、统一数学框架

请在文档中把 full sequence training 和 prefix roll-in 统一到同一个概率建模框架下。

定义 prefix state：

    s = (S, u)

其中：

- S 是已经 emitted 的 object set；
- u 是当前 object 内部的 partial token prefix；
- q_s(a) 是该 state 下的 trie/multi-positive target；
- p_theta(a | x, s) 是模型 next-token distribution。

局部 loss：

    ell_theta(s)
      = CE(q_s, p_theta)
      = - sum_a q_s(a) log p_theta(a | x, s)

总目标：

    J(theta)
      = E_{s ~ μ_rollin} [ ell_theta(s) ]

其中 μ_rollin 是训练时 prefix state 的分布。

不同训练策略的区别只在于 μ_rollin：

- full sequence teacher forcing：从空 prefix 出发，沿一条随机 permutation trajectory 训练整条 suffix；
- prefix sampling / roll-in：直接采样中间 prefix state，再训练 remaining suffix；
- 二者不是两种互斥算法，而是同一个目标下的两种 state sampling policy；
- 它们可以通过配置混合。

推荐统一描述：

    multi-positive architecture
      = roll-in policy + local target policy

其中：

    roll-in policy:
        controls which prefix states are trained.

    local target policy:
        controls what next-token distribution is supervised at each trained state.

---

# 四、full sequence 和 prefix roll-in 的关系

请在文档里明确解释：

1. `full sequence + random object order + trie CE` 已经是一个强 baseline。
   它沿一条随机 object permutation trajectory 训练所有 visited prefixes。

2. `prefix sampling / roll-in` 不是完全不同的目标。
   它主要提供 state reweighting：直接采样中间 prefix state，而不是只依赖完整随机 sequence 经过这些 state。

3. 二者在理想无限采样情况下可能优化相近的目标，但实际中不完全等价，因为：
   - full sequence 每次只访问一条 trajectory；
   - prefix sampling 可以直接 oversample middle / late / hard prefixes；
   - prefix sampling 可以支持 jittered / model-generated prefix recovery；
   - prefix sampling 可以更直接训练 “任意 emitted subset 后继续生成 remaining objects”。

4. 因此二者应该作为可混合 policy，而不是互斥开关。

推荐文档中的一句核心概括：

    Full sequence training preserves complete autoregressive generation behavior;
    prefix roll-in improves coverage and robustness over intermediate prefix states;
    trie-based multi-positive CE defines the local target at each visited state.

中文解释：

    full sequence 负责完整生成稳定性；
    prefix roll-in 负责中间状态覆盖和鲁棒性；
    trie CE 负责当前位置多个合法 continuation 的监督。

---

# 五、统一实现概念

请调研当前代码，并看看能否把两种模式统一成一个概念：

    sample a permutation π over objects
    sample K = number of roll-in objects
    prefix = first K objects in π
    supervised suffix = remaining objects in π

当：

    K = 0

就是 full sequence teacher forcing。

当：

    K > 0

就是 prefix roll-in。

prefix 部分在 input_ids 中存在，但 labels / loss mask 应该为 ignore；suffix 部分参与 loss，并在每个 token position 使用 hard CE 或 multi-positive CE。

这样 full sequence 和 prefix roll-in 可以通过 K 的分布统一。

文档中可以把它描述为：

    K-distribution controlled roll-in policy

而不是两个互相割裂的训练路径。

---

# 六、配置概念需要支持 mixing

请调研当前 config，并提出或添加统一字段，使 full sequence 和 prefix roll-in 可以混合控制。

不要拘泥字段名，但需要表达以下概念：

## Roll-in policy config

需要能表达：

- 是否启用 roll-in mixing；
- full sequence policy 的比例；
- prefix roll-in policy 的比例；
- K 的采样分布；
- 是否允许 K=0；
- 是否允许 K=N；
- 是否偏向 early / middle / late prefix；
- 是否支持 clean prefix；
- 是否未来支持 jittered prefix / model-generated prefix；
- prefix tokens 是否 mask loss。

示意概念：

    rollin:
      enabled: true
      policy_mix:
        full_sequence: 0.8
        prefix_rollin: 0.2
      k_sampling:
        type: uniform | depth_weighted | remaining_weighted | fixed | custom
      prefix_loss_mask: true

## Local target config

需要能表达：

- 是否启用 multi-positive / trie CE；
- loss 类型是 soft CE 还是 valid-mass；
- hard CE 与 trie CE 的 mixing λ；
- q_trie 的 weighting 方式；
- 是否按 object mass 聚合；
- 是否对不同 token type 使用不同 λ；
- EOS / closing token 是否单独处理。

示意概念：

    target:
      type: hard_ce | trie_soft_ce | mixed
      trie:
        weighting: object_uniform | token_uniform | custom
        mix_lambda: 1.0
        loss_mode: soft_ce
      eos:
        censored: true
        completeness_rho: 0.7

注意：这里的字段名只是概念示意。请结合当前 codebase 风格命名，不要强行照抄。

---

# 七、文档更新目标

请优先更新文档。文档应该让后来的人清楚知道当前算法已经从探索阶段确定下来。

请至少覆盖以下内容：

## 1. Algorithm overview

说明整个 multi-positive architecture 的目的：

- set-like generation；
- permutation-aware training；
- local next-token consistency；
- train/decode consistency；
- 不改变 Qwen3-VL 架构。

## 2. Terminology / Glossary

统一以下术语：

- object set
- object serialization
- full sequence teacher forcing
- prefix sampling / roll-in
- emitted subset
- remaining objects
- prefix state
- trie candidate set
- multi-positive target
- hard CE
- trie soft CE
- mixed target
- roll-in policy
- local target policy
- censored EOS

## 3. Mathematical formulation

写出：

    Y, z_i, S, R, u, C(u), q_trie, L_trie, q_mix, J(theta)

并说明 full sequence 和 prefix roll-in 只是不同的 μ_rollin。

## 4. Training flow

用高层流程描述：

    image + object set
      -> serialize objects
      -> choose roll-in policy
      -> sample permutation and K
      -> build assistant prefix and supervised suffix
      -> apply chat template
      -> tokenize
      -> collate input_ids / labels / soft targets
      -> forward Qwen3-VL
      -> compute shifted autoregressive loss
      -> hard CE or trie soft CE per token
      -> optional EOS handling

## 5. Policy mixing

解释 full sequence 和 prefix roll-in 如何混合。

明确：

    K=0 = full sequence
    K>0 = prefix roll-in

二者比例由 config 控制。

## 6. Implementation notes

记录当前代码中对应模块，例如：

- dataset / preprocessing
- collator
- chat template builder
- loss function
- multi-positive target builder
- config schema
- training script
- eval / decode script

先调研实际路径，再更新文档，不要凭空编文件名。

## 7. Invariants / sanity checks

文档中列出必须满足的 invariant：

- hard target token 应该在 positive token set 内；
- q_trie 概率和应为 1；
- prefix roll-in 部分 loss 应该 mask；
- emitted objects 不应再作为 positive candidate；
- remaining objects 应该与 prefix state 一致；
- causal LM shift 必须正确；
- user prompt / image tokens / assistant prefix tokens 不应参与 suffix loss；
- padding 不应参与 loss；
- EOS 在 remaining 非空时不应是 valid next token；
- 如果使用 incomplete annotations，应避免过强 EOS。

---

# 八、代码调研目标

在更新文档后，请检查当前 code implementation，并输出 infrastructure map。

重点查找：

- 当前 full sequence random order 在哪里实现；
- 当前 prefix sampling / roll-in 在哪里实现；
- 当前 multi-positive / ET-RMP-CE 在哪里实现；
- 当前 soft target / positive token set 是如何构造的；
- 当前 loss 是 soft CE 还是 valid-mass；
- 当前 hard CE 和 multi-positive 是否可混合；
- 当前 labels / loss mask / shift 是否正确；
- 当前 chat template 是否影响 span alignment；
- 当前 config 是否能表达 policy mixing；
- 当前 eval / decode 是否默认保持 autoregressive generation；
- 当前是否有 EOS / closing token 的特殊处理。

请不要在调研阶段大改代码。先输出当前行为和统一概念之间的差距。

---

# 九、实现统一时的原则

如果需要修改代码，请遵循这些原则：

1. 最小侵入。
2. 不改变 Qwen3-VL 模型结构。
3. 不改变主 decode 方式。
4. 不把 segment-level score 作为主 loss。
5. 保持 local next-token CE 形式。
6. 让 full sequence 和 prefix roll-in 通过 config 混合。
7. 让 hard CE、trie soft CE、mixed target 通过 config 明确控制。
8. 保持 backward compatibility：旧配置应尽量能复现旧行为。
9. 文档先行，代码跟随文档概念。
10. 对命名做统一，但避免无意义大规模重命名。

---

# 十、需要重点判断的实现风险

请特别检查以下风险点：

## 1. random shuffle 被误认为 prefix roll-in

如果当前只是每次打乱完整 object order，然后训练完整 sequence，这属于 full sequence random order，不等于 explicit prefix roll-in。

## 2. valid-mass loss 被误认为 soft CE

如果当前 loss 类似：

    -log sum_{a in positives} p(a)

它更像 valid-mass，不是 mode-covering soft CE。

理想 trie soft CE 是：

    - sum_a q(a) log p(a)

二者梯度行为不同。

## 3. token span / shift 错位

Qwen causal LM 的 logits[t] 通常预测 token[t+1]。  
multi-positive target 的 position 必须和 shift 后的 logits 对齐。

## 4. chat template 引入隐式 token

assistant header、BOS/EOS、image placeholder、generation prompt 可能影响 span index。

## 5. prefix loss 没有 mask

prefix roll-in 的前 K 个 objects 是 context，不应该作为当前样本的 supervised suffix 参与 loss。

## 6. emitted object 没有从 candidates 移除

否则会鼓励 duplicate generation。

## 7. q_trie 权重错误

如果 object-level uniform 被误实现成 unique-token uniform，会改变 target 分布。

例如：

    [cat_blue, cat_yellow, dog, people]

root target 应该是：

    cat: 2/4
    dog: 1/4
    people: 1/4

而不是：

    cat: 1/3
    dog: 1/3
    people: 1/3

除非文档明确选择 token-uniform。

## 8. EOS 过强

如果数据存在漏标，不应过度强化“标注输出完 = 必须停止”。

---

# 十一、期望输出

请按阶段输出。

## Stage 1: Infrastructure survey

输出：

- 相关文件和模块地图；
- 当前 full sequence / prefix roll-in / multi-positive 的实现位置；
- 当前配置项；
- 当前行为总结；
- 与统一概念的差距。

## Stage 2: Documentation patch

更新或新增文档，内容包括：

- algorithm overview；
- glossary；
- math formulation；
- training flow；
- policy mixing；
- config semantics；
- invariants；
- known risks。

## Stage 3: Config unification proposal / patch

根据当前 codebase 风格，新增或整理 config 字段，使以下概念可以表达：

- full sequence ratio；
- prefix roll-in ratio；
- K sampling；
- multi-positive target type；
- hard/trie mixing lambda；
- q weighting；
- EOS handling；
- backward compatibility。

## Stage 4: Implementation review / minimal cleanup

检查当前 implementation 是否和文档一致。  
如需要，只做最小修改。  
避免大重构。

## Stage 5: Diagnostics / tests proposal

建议或添加最小 sanity tests：

- hard token in positives；
- q sums to 1；
- positive set size distribution；
- K=0 behavior equals full sequence；
- K>0 prefix loss masked；
- emitted objects removed from remaining candidates；
- causal shift alignment；
- chat template span alignment；
- soft CE vs valid-mass behavior；
- EOS with remaining objects invalid。

---

# 十二、最终希望形成的统一描述

请把整个算法最终统一成类似下面的概念：

    Multi-Positive Autoregressive Set Generation for Qwen3-VL

    The method keeps the original autoregressive decoder and generation path.
    It modifies only the training data distribution and the local token target.

    A training sample is generated by:
      1. sampling an object permutation;
      2. sampling a roll-in depth K;
      3. using the first K objects as masked assistant prefix;
      4. supervising the remaining suffix;
      5. using trie-based multi-positive CE at each suffix token position.

    Full sequence training is the special case K = 0.
    Prefix roll-in is the case K > 0.
    Both are controlled by the roll-in policy mix.

    At each visited prefix state, the local target is:
      hard CE if there is only one valid next token;
      trie soft CE if multiple valid next tokens exist;
      optionally a mixture of hard CE and trie soft CE.

中文核心表述：

    我们不是训练一条固定顺序的 detection sequence，
    而是在合法 prefix-state 分布上训练一个 local next-token policy。
    roll-in policy 决定训练哪些 prefix states；
    multi-positive target policy 决定每个 state 下哪些 next tokens 是合法的。

请以这个作为文档和代码配置统一的中心思想。