---
title: Full Idea v3
status: proposal
scope: stage2-channel-b
topics: [stage2, channel-b, k2-rollout, recoverability, unlabeled, triage-posterior]
supersedes: progress/full_idea_v2.md
references:
  - docs/PROJECT_CONTEXT.md
  - docs/training/STAGE2_RUNBOOK.md
  - openspec/specs/stage2-ab-training/spec.md
  - progress/full_idea_v2.md
  - progress/full_idea_v2_draft.md
  - progress/diagnostics/stage2_ab_near_duplication_diagnosis_2026-03-05.md
  - progress/diagnostics/stage2_ul_capture_highres1024_2026-03-09.md
  - progress/benchmarks/stage2_oracle_k_first200_2026-03-11.md
---

# CoordExp v3: K=2 Triage-Posterior Training for V-LLM Detection

> Goal: keep the clean-prefix / CoordExp / no-extra-head advantages of v2, but replace the narrow “same-desc duplicate UL” view with a more general and more unified Stage-2 Channel-B objective.
>
> v3 treats Stage-2/B as **small-sample posterior triage over candidate object hypotheses**:
>
> - **GT-backed** hypotheses are learned as positives,
> - **unlabeled-but-consistent** hypotheses are protected / kept neutral,
> - **dead / redundant** hypotheses are suppressed.
>
> The intended default is **K=2 rollout**, not large-K RL.
>
> The scientific claim is narrower than “solve open-world detection”:
> v3 aims to fix the current underconstrained **cardinality / self-collision / partial-annotation** regime while staying within standard SFT + teacher-forced LM infrastructure.

---

## 0. Why v3 Exists

v2 solved an important and real problem:

- keep generic unmatched accepted objects neutral,
- detect obvious self-collision duplicates,
- fold those duplicates back into clean-boundary unlikelihood,
- keep the rest of the pipeline compatible with ordinary LM training.

That design was the right minimal step.

However, the latest diagnostics changed the problem statement.
The main pathology is no longer “same-desc duplicate UL capture failed.”
The current residual failure is broader:

- train-side rollout still badly over-generates,
- overlap is now much more visible in **any-desc** region overlap than in exact same-desc duplicates,
- many apparent eval FPs are plausibly **unlabeled real objects** or **class-ambiguous real regions**,
- and crowded-scene failures are often **coarse / redundant / non-atomic proposals**, not obvious hallucinations.

So v3 changes the center of gravity:

> We no longer ask only:
> “is this object an exact-desc duplicate?”
>
> We ask instead:
> “for this region/object hypothesis, should it be treated as GT-backed, unlabeled-consistent, or dead?”

---

## 1. What v3 Keeps From v2

v3 is an evolution of v2, not a reset.
The following remain unchanged:

1. **Base model**: pretrained V-LLM (Qwen3-VL family).
2. **Output protocol**: structured JSON with `desc` plus one geometry field.
3. **CoordExp**: continuous geometry gradients through coord-token logits remain central.
4. **No extra detection head**.
5. **No separate objectness classifier**.
6. **Teacher-forced LM training remains the core implementation substrate**.
7. **Stage-1 stays the language-of-boxes foundation**.
8. **Channel-A / Channel-B split stays**:
   - Channel-A = cheap geometry / self-context stabilizer
   - Channel-B = rollout-time set correction
9. **Generic unmatched predictions are not blanket negatives**.
10. **A-hot / B-cold scheduling still applies**.

So v3 does **not** throw away the clean-prefix philosophy.
It generalizes it.

---

## 2. The Central v3 Thesis

### 2.1 v2 thesis

v2 can be summarized as:

> “Generic unmatched stay neutral; duplicate-certified continuations are folded into clean-boundary UL.”

### 2.2 v3 thesis

v3 upgrades this to:

> “After a small number of on-policy rollouts, cluster region-level object hypotheses and perform posterior triage:
> GT-backed -> positive,
> unlabeled-consistent -> shield / neutral,
> dead -> suppress.”

This is the new unifying idea.

The point is to avoid a large zoo of narrowly-scoped losses:

- one loss for same-desc duplicate,
- another for semantic alias,
- another for long-strip boxes,
- another for hot-tail phrases,
- another for pseudo-label consistency,
- etc.

Those symptoms are different surfaces of the same underlying issue:

> the objective currently lacks a sufficiently explicit mechanism for deciding whether a newly emitted object hypothesis is:
> 1. useful,
> 2. plausibly real-but-unlabeled,
> or 3. dead redundancy.

---

## 3. High-Level Design Contract

### 3.1 Hard constraints

- No extra objectness head.
- No DETR queries.
- No separate detector branch.
- No mandatory RL implementation.
- No dependence on large-K sampling for training.
- Must remain implementable inside the existing rollout + parse + teacher-forced loss pipeline.

### 3.2 Default sample budget

The canonical training configuration for v3 is:

- **K = 2** rollouts in Channel-B
- not K=8, not full RL sweeps

Reason:

- K=2 already provides genuine path-level contrast,
- avoids the prefix-selection ambiguity of single-rollout counterfactual rescoring,
- and is cheap enough to be realistic.

Larger K remains a **diagnostic / audit** tool, not the default training substrate.

### 3.3 Default rollout roles

The most practical default is:

- **anchor rollout** = greedy / conservative decode
- **explore rollout** = one stochastic decode

For the first implementation, keep this **config-first** rather than hard-coded:

- anchor decode = deterministic / greedy (`T = 0`)
- explore decode = stochastic (`T = 0.7` by default)

Current evidence suggests that a model in the `ul_res_1024-v2` family remains a strong explore miner, but the first v3 contract should not bake that lineage into the algorithm definition.

This does **not** mean the hot rollout should be imitated wholesale.
It is a mining path, not a behavior teacher.

---

## 4. Core Objects And Notation

For a training sample `(x, G)`:

- `x` = image + prompt/context
- `G` = GT object set

Run two Channel-B rollouts:

- `y_a` = anchor rollout
- `y_e` = explore rollout

Each rollout is parsed and cleaned separately.
For rollout `r in {a, e}`:

- `O_raw^(r)` = parsed raw objects
- `A^(r)` = accepted objects after within-run sequential dedup
- `D^(r)` = duplicate-certified objects removed from the positive prefix
- `M^(r), U^(r), FN^(r)` = Hungarian result on `A^(r)` vs `G`

So v3 keeps the useful within-run structure of v2.
The new ingredient is what happens **across** rollouts.

---

## 5. New v3 Primitive: Cross-Rollout Region Clusters

### 5.1 Why clustering is the new primitive

The natural object of learning in v3 is **not** a single raw emitted object.
It is a **region-level hypothesis cluster**.

This is the key step that allows us to stop thinking in terms of exact-desc duplicates.

A cluster can absorb things like:

- `person`
- `coach`
- `sister`

when they are all emitted on essentially the same region across rollouts.

### 5.2 Cluster construction

A cluster `c` is built from accepted objects across the K rollouts.
The simplest robust starting point is:

1. only cluster **accepted** objects, not already-within-run duplicates,
2. use a high-IoU geometric criterion as the primary linkage,
3. optionally add a weak semantic-nearness condition when needed.

Recommended starting thresholds:

- `tau_cluster ~= 0.85 ~ 0.90` for cross-rollout “same latent object” clustering
- `tau_redundant_hi ~= 0.95` for high-precision same-region redundancy certification

The philosophy is:

- **high precision first** for dead-redundancy certification,
- only moderate breadth for cross-rollout same-object clustering.

For canonical `K=2`, the first implementation does **not** need a general K-way clustering engine.
It can be implemented as:

- deterministic **one-to-one bipartite max-IoU association** between anchor and explorer accepted objects,
- one record per matched pair or leftover singleton,
- stable tie-breaks when IoU ties occur.

So in v1, “cluster” can be read operationally as **pair-or-singleton triage record**.

### 5.3 Cluster members

Each cluster stores:

- member objects from anchor/explore runs
- each member’s rollout prefix / token span
- each member’s normalized `desc`
- each member’s geometry
- whether any member full-matches a GT object
- whether any member corresponds to a recovered FN

---

## 6. The Triage Heuristic

This is the real center of v3.

For each cluster `c`, define three heuristic scores:

- `s_gt(c)`   : GT-backed positive score
- `s_u(c)`    : unlabeled-consistent latent-positive score
- `s_dead(c)` : dead / redundant score

with:

```text
s_dead(c) = 1 - max(s_gt(c), s_u(c))
```

These are **not** intended to be calibrated posterior probabilities or a normalized partition.
The point is to give the training system a **small, explicit triage**.

### 6.1 GT-backed score

A cluster is GT-backed if it is directly supported by GT.
The strongest cases are:

1. any member is matched to a GT object under the **existing Stage-2 Channel-B Hungarian + gating contract**,
2. or the cluster corresponds to a **recovered FN**:
   - miss in the anchor path,
   - hit in the explore path.

This is the most important positive signal in v3.

However, in v1 this **cannot** be used directly as an anchor-retention rule.
The training action must stay side-specific:

- if the **anchor** member is matched, keep that anchor object as `ANCHOR_GT_BACKED`,
- if the anchor member misses but the **explorer** member matches, treat that GT as `RECOVERED_FN`,
- do **not** keep a bad anchor object positive just because the explorer side matched.

### 6.2 Unlabeled-consistent score

A cluster should receive unlabeled-consistent credit if:

- it does **not** already claim GT-backed status,
- but it appears sufficiently stable / repeated under sampling,
- and it does not look like obvious redundancy around an already-better explanation.

The minimal feature set is:

- `support(c)`   : how often the cluster appears across rollouts
- `stability(c)` : how geometrically consistent the members are
- `like(c)`      : object continuation likelihood / normalized logprob under the local prefix
- `dup_risk(c)`  : how much it looks like redundant re-explanation of an already-better cluster

The broader posterior language above is useful for theory, but it is **not** the canonical v1 implementation contract.
For v1:

- geometry-first cross-rollout stability is the main acceptance rule,
- semantic nearness is at most a weak veto / tie-breaker,
- local likelihood terms are not required for the first implementation.

A simple soft scoring form is:

```text
s_u(c) = sigmoid(a * support(c)
               + b * stability(c)
               + d * like(c)
               - e * dup_risk(c))
```

But the canonical starting implementation should be much simpler:

### 6.3 Canonical first-pass rule-based triage

Use high-precision hard rules first.

#### Rule A — GT-backed

```text
if cluster has any GT-backed evidence under the existing Channel-B matching contract:
    class(c) = GT_BACKED
```

#### Rule B — unlabeled-consistent

```text
else if support(c) == K
    and stability(c) >= tau_stable
    and dup_risk(c) < tau_dup_risk:
    class(c) = UNLABELED_CONSISTENT
```

For canonical `K=2`, this simply means:

> “it appears in both rollouts, lands on the same region, and does not look like an obvious re-description of an already-better object.”

In v1, this is intentionally **high-precision geometry-first**:

- an object must appear in **both** rollouts to qualify as shielded,
- explorer-only, non-GT-backed objects are **not** promoted to shielded context by default.

#### Rule C — dead

```text
else:
    class(c) = DEAD
```

This deliberately avoids premature probabilistic sophistication.

---

## 7. Why v3 Does Not Need A Separate “Complexity” Term

v2-to-v3 transition naturally tempts one to define three separate penalties:

- redundancy,
- complexity,
- cardinality.

That looks principled, but it quickly turns into a taxonomy swamp.

v3 avoids this by collapsing “complexity” into **deadness**.

An object hypothesis is “too complex” only insofar as:

- it adds no GT-backed gain,
- it earns no consistency support,
- and it therefore behaves like dead continuation mass.

So in v3:

- semantic alias on the same region -> dead unless GT-backed or consistent-unlabeled
- long-strip group-box -> dead unless GT-backed or consistent-unlabeled
- hot-tail phrase -> dead almost automatically

This is much cleaner than separately hand-designing:

- alias loss,
- long-box penalty,
- open-phrase penalty,
- extra-count penalty,
- etc.

---

## 8. Channel-B v3 Pipeline

v3 keeps the spirit of `B0/B1/B2/B3/B4`, but changes what is built after rollout.

### B0 — Two rollouts

Generate:

```text
y_a = Rollout(anchor policy)
y_e = Rollout(explore policy)
```

### B1 — Strict parse per rollout

```text
O_raw^(a), O_raw^(e)
```

### B2 — Within-run clean-prefix dedup per rollout

For each rollout independently, run the v2-style sequential dedup.
This yields:

```text
A^(r), D^(r)   for r in {a, e}
```

Important:

- v3 keeps v2’s within-run duplicate removal,
- but same-desc duplicate UL is no longer the whole story.
- it becomes a **high-precision local cleaner**, not the sole semantic of redundancy.

### B3 — Hungarian on accepted sets

Run matching on each accepted set:

```text
M^(r), U^(r), FN^(r)
```

### B4 — Cross-rollout clustering

Build region-level pair-or-singleton records over accepted objects from the two runs using deterministic one-to-one max-IoU association with stable tie-breaks.
This produces:

```text
C = {c_1, ..., c_L}
```

### B5 — Posterior triage

Assign each record enough evidence to drive one of the following **training actions**:

- `ANCHOR_GT_BACKED`
- `RECOVERED_FN`
- `UNLABELED_CONSISTENT`
- `DEAD`

using the rules in Section 6.

### B6 — Build the posterior-clean training view

This is the key construction.

Start from the **anchor clean sequence** `A^(a)`.
Then modify it using the action states above:

#### Keep
- anchor objects that are `ANCHOR_GT_BACKED`
- anchor objects that belong to unlabeled-consistent clusters

#### Remove from positive prefix
- anchor objects that belong to dead clusters

#### Inject as positives
- GT objects that are `RECOVERED_FN`

This produces a new teacher-forced target sequence:

```text
y_v3
```

This sequence is:

- cleaner than the raw anchor path,
- not as aggressive as “GT-purify everything,”
- and informed by actual on-policy evidence from a second rollout.

Important v1 contract:

- this is an **anchor-edited** target, not a rebuilt union over anchor + explore,
- anchor order remains the positive-prefix order,
- explore provides correction evidence only; it does not become a second teacher trajectory.

### B7 — Local negative signal for dead clusters

For dead clusters that appeared in the anchor accepted path, apply a mild local negative signal.

The recommended default remains the v2-style **LCP-divergence unlikelihood** idea:

- identify the clean boundary where the dead continuation was emitted,
- compare dead continuation vs canonical positive continuation,
- apply UL at the first divergence token.

The difference from v2 is conceptual:

- the negative target is no longer only “same-desc duplicate,”
- it is “cluster triage says this continuation is dead.”

### B8 — Geometry losses

Geometry loss stays structurally simple:

- anchor GT-backed accepted objects: supervised
- recovered GT injections: supervised with higher per-object weight in v1
- unlabeled-consistent objects: no geometry loss by default
- dead objects: no geometry regression; handled via local suppression or prefix removal

---

## 9. Canonical v3 Loss

The canonical v3 Channel-B loss should stay small.

Conceptually, this is:

> `L(clean_anchor) + L(explore-derived corrections)`

Operationally, v1 should still run **one merged teacher-forced forward** on the single anchor-edited target `y_v3`.
It should **not** introduce a second teacher-forced payload or a separate explore-prefix forward in the first implementation.

```text
L_B_v3 = L_struct_clean
       + L_desc_tail
       + lambda_rec  * L_rec
       + lambda_dead * L_dead
       + lambda_geo  * L_geo
```

Where:

- `L_struct_clean` = structure CE on the posterior-clean teacher-forced sequence
- `L_desc_tail`    = desc CE on the FN tail under the current Channel-B masking policy
- `L_rec`          = extra per-object weight for `RECOVERED_FN` desc/geo/coord terms
- `L_dead`         = mild local UL on dead anchor-side continuations
- `L_geo`          = CoordExp losses on anchor GT-backed accepted objects + GT injections

All of these terms are read from the **same forward** over `y_v3`.

Notably absent:

- no explicit blanket FP penalty,
- no objectness head,
- no mandatory complexity term,
- no whole-sequence scalar reward RL,
- no need to imitate full hot rollout behavior.

This is deliberate.

---

## 10. Recovered-GT Training Is The Main Positive Driver

The single most important new positive signal in v3 is:

> **baseline-missed but explore-recovered GT objects**.

Define:

```text
R = { g in G : g missed by anchor, hit by explore }
```

These recovered GT objects should receive special treatment.
Here, “hit” and “miss” inherit the **existing Stage-2 Channel-B matching contract** rather than a stricter new notion of full-match.

### 10.1 Minimal version

The minimal, lowest-risk version is:

- keep the current FN injection mechanism,
- but increase the weight for `g in R` across the **desc + geo + coord** terms attached to that injected GT object.

That already makes training recoverability-aware.

### 10.2 Stronger version

The stronger version uses the **actual explore prefix** where `g` succeeded.

For each recovered GT `g`:

- extract the local prefix `s_e(g)` from the explore path,
- teacher-force the **GT object** at that prefix,
- optimize desc/struct/CoordExp there.

This is the natural v3 “recovered-prefix distillation” extension.

It is stronger than simple FN reweighting, but it is also more implementation-heavy.
So v3 should treat it as a **Phase-2** upgrade, not the first landed version.

---

## 11. Treatment Of Unlabeled-Consistent Clusters

This is the most important place where v3 departs from naive count-penalty thinking.

### 11.1 First principle

Do **not** immediately turn unlabeled-consistent clusters into hard pseudo-labels.

That is too risky under partial annotation.

### 11.2 Canonical Phase-1 behavior

For clusters with high `r_u` / class `UNLABELED_CONSISTENT`:

- keep them as context if they appear in the anchor clean sequence,
- do not penalize them,
- do not give them geometry loss,
- do not force them as hard positives.
- do not promote explorer-only objects into shielded context by default.

This is best thought of as **shielding**.

### 11.3 Phase-2 optional upgrade

Once the triage is stable, a small subset of very high-confidence unlabeled-consistent clusters may be used for:

- soft-positive self-distillation,
- or low-weight auxiliary CE / geo targets.

This is not canonical v3 core.
It is an optional follow-up.

---

## 12. Treatment Of Dead Clusters

A dead cluster is a hypothesis that:

- is not GT-backed,
- is not unlabeled-consistent,
- and therefore behaves like pure extra continuation mass.

### 12.1 What dead includes

This absorbs all the annoying symptom types:

- exact-desc duplicates
- same-region semantic alias (`person / coach / sister`)
- long-strip group-box attempts
- hot-tail open phrases
- other zero-gain regional re-descriptions

### 12.2 What to do with dead clusters

The canonical first action is mild and local:

1. remove them from the positive teacher-forced prefix,
2. apply dead-continuation UL at the clean boundary,
3. monitor whether their rate and their late-tail concentration go down.

In v1, this suppression path is intentionally **anchor-side only**:

- if explorer evidence marks an anchor continuation as dead, suppress that **anchor-side** continuation,
- do not create a separate explorer-side UL branch in the first implementation.

Do **not** use large hard negative span penalties as the default.

---

## 13. Why v3 Is More Unified Than “Redundancy + Complexity” Loss Design

A tempting design would define:

- `Coverage`
- `Redundancy`
- `Complexity`

and then optimize a hand-crafted weighted sum.

v3 instead says:

> “Do not try to perfectly define every kind of bad continuation.
> Define a triage posterior over object hypotheses, then let deadness absorb what is neither useful nor consistent.”

This gives a much cleaner implementation and a much cleaner research story.

The center of the method is not:

- lexical duplicate detection,
- hand-tuned complexity penalties,
- or blanket count regularization.

It is:

> **small-sample posterior triage over object hypotheses.**

---

## 14. Minimal Canonical Implementation Order

v3 should land in phases.

### Phase 0 — instrumentation only

Add the objects and metrics needed to inspect the triage problem:

- per-run accepted sets
- cross-rollout clusters
- recovered GT counts
- unlabeled-consistent cluster counts
- dead cluster counts
- dead-late-tail rate
- utility-per-pred style summaries

### Phase 1 — minimal training change

Land:

- K=2 rollouts in B
- GT-backed recovered-FN mining
- recoverability-weighted FN injection
- unlabeled-consistent shield-only behavior
- dead-cluster mild UL only on anchor-side objects
- one merged teacher-forced forward on the anchor-edited target
- no recovered-prefix distillation yet
- explorer-only non-GT-backed objects treated as dead by default

This should be the **canonical first v3 implementation**.

### Phase 2 — stronger positive use of exploration

Add:

- recovered-prefix distillation from explore prefixes
- optional low-weight soft positive for very high-confidence unlabeled-consistent clusters

### Phase 3 — optional refinements

Only after Phase 1/2 prove useful should we explore:

- soft posterior weighting instead of hard triage,
- more semantic cluster features,
- more advanced exploration schedules.

---

## 15. Recommended Default Configuration

The following is the recommended first v3 default:

### Sampling

- for Channel-B steps
- `K = 2`
- anchor = greedy
- explore = stochastic

### Explore regime

Current best starting point:

- explorer decoding kept under a typed config block,
- anchor decode fixed to greedy / deterministic,
- explorer temperature defaulting to `0.7` for the first implementation.

Broader temperature sweeps remain useful for diagnosis, but they are not the canonical v1 training contract.

### Schedule

- keep A-hot / B-cold
- do not raise B aggressively until:
  - truncation is not exploding,
  - dead-cluster rate is measurable,
  - and recovered-GT signal is stable.

### Loss aggressiveness

- recovered-GT positive weighting: moderate
- unlabeled-consistent promotion: off by default
- dead suppression: mild

This is not the place for theatrical oversteering.

---

## 16. Diagnostics And Acceptance Criteria

v3 should be judged by a sharper scorecard than v2.

### 16.1 Core metrics

- `eval_rollout/mAP`
- `eval_rollout/precision`
- `eval_rollout/recall`
- `eval_rollout/matched_maskiou_mean`
- `rollout/pred_per_sample`
- `rollout/rollout_len_mean`
- `rollout/parse_truncated_rate`

### 16.2 New triage metrics

- `triage/recovered_gt_count`
- `triage/unlabeled_consistent_count`
- `triage/dead_count`
- `triage/dead_late_tail_rate`
- `triage/recovered_gt_num`
- `triage/recovered_gt_den`
- `triage/dead_anchor_num`
- `triage/dead_anchor_den`

### 16.3 Legacy overlap metrics that still matter

- `dup/near_iou90_pairs_same_desc`
- `dup/near_iou90_pairs_any_desc`

But in v3 they become **supporting diagnostics**, not the whole worldview.

### 16.4 Acceptance criteria

A good v3 move should:

1. preserve or improve recoverable-GT learning,
2. reduce dead-anchor continuation mass,
3. avoid punishing stable unlabeled-consistent hypotheses,
4. keep healthy crowded same-class scenes intact,
5. improve train-side length/cardinality stability,
6. not regress deterministic eval quality.

---

## 17. What v3 Explicitly Does Not Claim

- It does **not** claim solved open-world detection.
- It does **not** claim a clean global optimum guarantee.
- It does **not** claim that K=2 is statistically sufficient for all purposes.
- It does **not** claim all unmatched objects can be classified perfectly.
- It does **not** require large-K RL or a learned reward model.

Its claim is narrower:

> A clean-prefix V-LLM detector can be upgraded from duplicate-only correction to cross-rollout object-hypothesis triage using just K=2 on-policy samples, while staying inside standard LM training and preserving space for unlabeled true positives.

---

## 18. One-Sentence Summary

v2 can be remembered as:

> “generic unmatched neutral, duplicate continuations negative.”

v3 should be remembered as:

> **“GT-backed learn, unlabeled-consistent shield, dead suppress.”**

That is the whole idea.
