---
title: Ranking Quality Versus Usable Rejection Results
description: Area under the ROC curve is anti-predictive of usable rejection here — support ranks best on every checkpoint and rejects nothing — because it is scored on a population whose retention is free and because a discrete atom at support 1 blocks any admissible threshold.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-07-29-ranking-quality-versus-usable-rejection
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-29
---

# Ranking Quality Versus Usable Rejection Results

Protocol is owned by [unit.md](unit.md). Parent result is
[2026-07-29 Sampled Object-Span Likelihood and Consensus Filtering](../2026-07-29-sampled-span-likelihood-and-consensus-union-filtering/results.md).

## Decision

Area under the ROC curve, as computed in the parent unit's Stage 1, is **not a
valid ranking of filter candidates** for this decision class. On this evidence
it is anti-predictive: the feature with the highest area under the curve on
every checkpoint is the feature that rejects nothing.

| Checkpoint | Feature | Area under curve, all true positives | Area under curve, recovered only | Rejection at 95% retention |
|---|---|---:|---:|---:|
| Sorted | support | **0.894** | 0.797 | **0.000** |
| Sorted | coordinate mean | 0.799 | 0.722 | **0.305** |
| Random | support | **0.821** | 0.632 | **0.000** |
| Random | coordinate mean | 0.758 | 0.662 | **0.214** |
| Permutation | support | **0.829** | 0.686 | **0.000** |
| Permutation | coordinate mean | 0.794 | 0.715 | **0.152** |

Support wins the ranking on all three checkpoints and loses the decision on all
three. The parent unit reached the right verdict; it reached it despite its
headline metric, not because of it.

## Two independent causes, both measured

**1. The metric is scored on a population whose retention is free.**

Splitting the true positive population:

| Checkpoint | Feature | Recovered owners vs catastrophic | Other true positives vs catastrophic | Gap |
|---|---|---:|---:|---:|
| Sorted | coordinate mean | 0.722 | 0.865 | 0.143 |
| Sorted | support | 0.797 | **0.977** | 0.180 |
| Random | coordinate mean | 0.662 | 0.794 | 0.132 |
| Random | support | 0.632 | **0.892** | 0.260 |
| Permutation | coordinate mean | 0.715 | 0.838 | 0.123 |
| Permutation | support | 0.686 | **0.907** | 0.221 |

Every feature is markedly better at separating the true positives greedy already
found than at separating the ones only sampling recovers — and a filter over the
sampled union is never required to retain the former. Support's inflation from
this effect is roughly twice coordinate likelihood's (0.18–0.26 against
0.12–0.14), which is precisely why support looks best on the pooled metric and
performs worst at the operating point.

**2. A discrete atom blocks any admissible threshold.**

| Checkpoint | Recovered owners at support 1 | Catastrophic clusters at support 1 |
|---|---:|---:|
| Sorted | 13.6% (12/88) | 55.3% (109/197) |
| Random | 31.8% (14/44) | 51.9% (68/131) |
| Permutation | 30.0% (18/60) | 54.3% (75/138) |

Over half the catastrophic mass sits at support exactly 1 on every checkpoint,
together with 14–32% of the owners that must be retained. A 95% retention
constraint cannot cut into that atom, and no ordering *within* the atom exists,
because every member carries the identical value. Ranking quality is irrelevant
to a threshold that cannot be placed. This is a property of the signal's
distribution, not of its informativeness.

The control holds: coordinate likelihood is continuous, has no atom, and retains
usable rejection under the identical decomposition. So the failure is specific to
support's distribution, not an artifact of the analysis — the alternative in
[unit.md](unit.md) is ruled out.

## Observed

1. Support has the highest pooled area under the curve on all three checkpoints
   and zero rejection on all three.
2. Both features separate the free-retention population substantially better
   than the retention-critical one; support's gap is about twice as large.
3. Over half of every checkpoint's catastrophic tail sits at support 1, along
   with 14–32% of the recovered owners.
4. Sorted has the smallest recovered-owner atom (13.6%) and the largest usable
   rejection (0.305); Random and Permutation have atoms above 30% and rejections
   of 0.214 and 0.152.

## Supported

The parent unit's decision to make greedy-missed union-recovered owners the
primary retention denominator is load-bearing in a second, independent way. It
does not merely change which family wins — it is the only reason the pooled
ranking's failure was visible at all. Under an all-true-positive denominator,
support would have been selected on both the ranking and the operating point.

## Ruled out

- **Area under the ROC curve over all true positives** as a valid selection
  metric for this filter class.
- **Noise or clustering artifact** as the explanation for support's behaviour;
  the effect decomposes cleanly and coordinate likelihood survives the same
  decomposition.

## Unresolved

- Observation 4 is suggestive of a relationship between atom size and usable
  rejection, but three checkpoints cannot establish it, and the two quantities
  are not independent — a smaller atom mechanically permits a lower threshold.
  It is stated as a co-occurrence, not a mechanism.
- **Why Permutation fails** remains open. This unit explains why *support*
  fails everywhere; it does not explain why *coordinate likelihood* clears the
  bar on Sorted and Random but not Permutation.

## Not claimed

No filter, threshold, policy, or training change is promoted. All numbers here
are in sample by design, because they explain an already-decided out-of-sample
result; they are not a new performance estimate and must not be quoted as one.

## Next discriminator

For the successor design: select confidence candidates by rejection at the
retention constraint directly, not by a pooled ranking metric. Report area under
the curve, if at all, over the retention-critical population only.

For Permutation: compare its catastrophic clusters against Sorted's at matched
category and box scale, restricted to support greater than 1 so the atom does
not dominate the comparison.
