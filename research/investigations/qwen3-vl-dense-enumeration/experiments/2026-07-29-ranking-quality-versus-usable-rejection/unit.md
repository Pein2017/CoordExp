---
title: Ranking Quality Versus Usable Rejection at a Retention Constraint
description: Explains why the best-ranking confidence signal rejects nothing once a retention constraint is applied, by separating the population a signal is scored on from the population a filter must retain, and by measuring the discrete atom that blocks any threshold.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-07-29-ranking-quality-versus-usable-rejection
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-29
---

# Ranking Quality Versus Usable Rejection at a Retention Constraint

Execution is complete. The decision-bearing interpretation is owned by
[results.md](results.md).

Reanalysis only: no generation, no replay, no GPU. Inputs are the frozen
cluster artifact of
[2026-07-29 Sampled Object-Span Likelihood and Consensus Filtering](../2026-07-29-sampled-span-likelihood-and-consensus-union-filtering/results.md).

## Decision and Question

The parent unit produced an apparent contradiction that it recorded but did not
explain. Trajectory support had the **highest** area under the ROC curve against
the catastrophic tail on every checkpoint — 0.894 on Sorted, above every
likelihood feature — and yet rejected **nothing** at the pre-registered
operating point. The parent unit attributed this to a confound between support
and the recovered owners. That is correct but incomplete, and the gap matters:
if area under the curve can be high while usable rejection is zero, then area
under the curve is the wrong instrument for this decision class, and the Stage 1
ranking should not be read as a ranking of filter candidates at all.

The question: **why does ranking quality fail to predict usable rejection here,
and is the failure specific to support or general to the measurement?**

## Strongest Alternative

The alternative is that support is simply a weak signal and its high area under
the curve is noise or an artifact of the parent unit's clustering. Under that
alternative, the effect would not decompose cleanly, and coordinate likelihood
would show the same zero-rejection behaviour.

The control that separates this: run the identical decomposition on coordinate
likelihood, which is known to reject at a usable rate. If coordinate likelihood
retains nonzero rejection under the same decomposition while support collapses,
the failure is a property of support's distribution rather than of the analysis.

## Method

Two decompositions, both in sample and without folds, because this explains an
already-decided result rather than producing a new estimate. The
decision-bearing out-of-sample numbers remain owned by the parent unit.

**Population decomposition.** A filter must retain greedy-missed
union-recovered owners; it never has to retain the true positives greedy
already found, whose retention is free. Report, per feature, area under the
curve against the catastrophic tail computed three ways: over all true
positives, over recovered owners only, and over the remaining true positives
only.

**Discreteness decomposition.** Trajectory support is an integer in 1..16 with a
large atom at 1. A threshold that must retain 95% of recovered owners cannot cut
into an atom holding a substantial share of them, regardless of how well the
feature ranks within the rest of the range. Report, per checkpoint, the share of
recovered owners and of catastrophic clusters at support exactly 1.

## Artifact Handle

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-29-three-checkpoint-human-refined12-max3084/likelihood-mining-v1/ranking-vs-operating-point.json`

Produced by `scripts/research/analyze_ranking_versus_operating_point.py`, which
imports the parent unit's `_auroc` implementation rather than reimplementing it,
so the numbers are comparable to Stage 1 by construction.

**Replay note.** Producer scripts deleted from `research-probes` on 2026-08-28 (reclaim-research-probes-lifecycle); replay them from tag `research-base-v2`: `git worktree add <tmp> research-base-v2`.

## Scope and Non-Goals

Panel, clusters, labels, likelihoods, and the retention target are inherited
unchanged. This unit changes no threshold, promotes nothing, and does not
revisit any verdict. It explains a mechanism and constrains how a metric should
be used in the successor design.

## Terminology

Inherited from the parent unit. One addition:

- **Free-retention population**: true positive clusters that the native greedy
  decode already found. A confidence filter over the sampled union is not
  required to retain them, so a signal's ability to rank them above the
  catastrophic tail does not contribute to the filter's usable operating point.
