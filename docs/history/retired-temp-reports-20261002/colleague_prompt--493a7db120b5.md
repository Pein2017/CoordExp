````md
Please refactor the current `ET_RMP_CE` / entry recursive multi-positive CE training objective into a `trie-marginal` objective.

## Background

The current `ET_RMP_CE` objective was designed to reduce false-negative gradients in autoregressive detection / grounding.

In standard teacher-forcing SFT, object detection is serialized into one arbitrary object order. However, the ground-truth target is a set of objects, not an ordered sequence. If an image contains multiple valid objects, then several next entries may be equally legal at the same prefix. Traditional hard CE wrongly treats all non-selected valid objects as negatives.

`ET_RMP_CE` partially fixes this by using a trie and applying multi-positive CE over valid next tokens. However, this still optimizes a distribution-matching objective:

\[
L_{\text{MP-CE}}(v)
=
-\sum_{a\in C(v)} q(a|v)\log p_\theta(a|v)
\]

This objective encourages all valid child tokens to receive probability according to some target distribution `q`. That is not exactly what we want.

The desired decoding semantics are:

> At an ambiguous trie node, the model may choose any valid continuation. It does not need to distribute probability evenly across all valid continuations. It only needs to put probability mass into the valid subtree.

Therefore, the target objective should be changed from **multi-positive distribution matching** to **valid-set marginal likelihood**.

## Core Algorithmic Change

Replace `ET_RMP_CE` at ambiguous trie nodes with:

\[
L_{\text{trie-marginal}}(v)
=
-\log
\sum_{a\in C(v)}
p_\theta(a|v)
\]

where:

- `v` is the current trie node / prefix state;
- `C(v)` is the set of valid next tokens from the trie;
- \(p_\theta(a|v)\) is the model probability of token `a` under the current prefix.

This means:

```text
Current ET_RMP_CE:
"All valid children should be lifted according to q."

New trie-marginal:
"The total probability mass over valid children should be high."
````

This is closer to autoregressive decoding, because decoding chooses one path, not a calibrated distribution over all valid paths.

## Sequence-Level Interpretation

The deeper target is latent-order set likelihood:

[
P_\theta(\mathcal{O}|x)
=======================

\sum_{\pi \in \text{Perm}(\mathcal{O})}
P_\theta(y^\pi|x)
]

where:

* (\mathcal{O}) is the unordered object set;
* (\pi) is one valid object order;
* (y^\pi) is the serialized detection sequence under that order.

The ideal loss is:

[
L_{\text{set}}
==============

-\log
\sum_{\pi}
P_\theta(y^\pi|x)
]

The local trie-marginal loss is a practical approximation to this set-valued autoregressive likelihood.

## Important Distinction

Do not implement this as soft CE or multi-positive CE.

Soft / MP CE says:

[
\frac{\partial L}{\partial z_i}
===============================

p_i-q_i
]

and pulls the hidden state toward the weighted average of all valid children.

Trie-marginal says:

[
L=-\log \sum_{a\in C(v)}p(a|v)
]

and only requires that the valid set receives high total probability. Inside the valid set, the model is free to prefer one legal branch.

This is important because in inference, greedy / beam decoding will select one continuation path. We want to strengthen legal path selection, not force all legal branches to be equally represented.

## Keep Hard Path Commitment

The model should still use ordinary hard CE after a specific branch has been selected by the sampled teacher-forced path.

The desired behavior is:

```text
Before branch choice:
    use trie-marginal over all valid children.

After branch choice:
    fallback to hard CE along the selected object entry.

At the next object boundary:
    again expose all remaining valid objects through trie-marginal.
```

This preserves branch commitment while removing false-negative gradients at ambiguous prefix states.

## Expected Behavioral Difference

Compared with `ET_RMP_CE`, trie-marginal should:

1. avoid penalizing the model for preferring one valid object over another;
2. increase valid-continuation mass at ambiguous object-entry states;
3. reduce conflict between training and greedy / beam decoding;
4. avoid flattening valid branches too much;
5. preserve sharp path commitment after a branch is chosen;
6. potentially improve recall by making valid continuation more competitive against EOS / stop tokens.

## Risks to Audit

Please explicitly reason about and test for these risks:

1. **Mode collapse**

   * Trie-marginal may over-reinforce the easiest valid branch.
   * Random shuffle / sampled teacher paths should still provide coverage across objects.

2. **EOS competition**

   * If the model is conservative, valid continuation mass must be compared against EOS probability.
   * Consider whether an additional continuation/EOS margin objective is needed later, but do not over-design it in this refactor unless the current architecture already supports it cleanly.

3. **Branch commitment**

   * The model should not remain in an ambiguous set state after selecting a branch.
   * Hard CE along the selected entry should remain the commitment mechanism.

4. **Object coherence**

   * The generated bbox coordinates should belong to the same selected object, not mix coordinates from different objects.

5. **Permutation sensitivity**

   * The new objective should reduce likelihood variance across valid object permutations.

## Suggested Diagnostics

After implementation, compare against the current `ET_RMP_CE` using:

* valid-set probability mass at ambiguous trie nodes;
* probability of selected teacher child;
* probability mass of other valid children;
* illegal token mass;
* EOS / stop probability at object boundary;
* branch top-1 margin;
* x1 onset locality under teacher-forced and self-prefix settings;
* recall / AR;
* duplicate rate;
* object coherence;
* permutation NLL variance across sampled object orders.

## Desired Outcome

The final training objective should no longer be described as `entry recursive multi-positive CE`.

It should be described as:

> a sampled-path, trie-marginal objective for latent-order autoregressive set prediction.

The core principle is:

```text
At ambiguous prefixes, optimize valid-set likelihood.
After choosing a branch, optimize hard path likelihood.
```

Please inspect the current training objective and refactor the algorithm accordingly, keeping the conceptual contract clean and avoiding legacy compatibility unless it is absolutely necessary.

```

