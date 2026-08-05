---
title: Likelihood Filter Robustness at Equal Owner Count and Panel Annotation Validity
description: Tests whether Sorted's likelihood-filter pass and Permutation's failure are artifacts of unequal recovered-owner counts, and establishes how complete the twelve-image panel's ground truth actually is.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-07-29-likelihood-filter-robustness-and-panel-annotation-validity
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-29
---

# Likelihood Filter Robustness at Equal Owner Count and Panel Annotation Validity

Execution is complete. The decision-bearing interpretation is owned by
[results.md](results.md).

This unit answers the next discriminator recorded by
[2026-07-29 Sampled Object-Span Likelihood and Consensus Filtering](../2026-07-29-sampled-span-likelihood-and-consensus-union-filtering/results.md),
and additionally closes a validity threat that the parent unit did not name.

No new generation, replay, or GPU work. Both probes are reanalysis of the
parent unit's frozen artifacts plus the panel definition file.

## Decision and Questions

The parent unit concluded that coordinate likelihood, not trajectory consensus,
can reject badly grounded union clusters, with Sorted passing the pre-registered
bar at 31.0% catastrophic rejection, Random passing by one cluster, and
Permutation failing at 15.2%. Two things could make that conclusion an artifact.

1. **Composition.** The retention denominators are unequal — 88 recovered owners
   for Sorted, 44 for Random, 60 for Permutation. A 5% retention budget buys 4
   droppable owners on Sorted and only 2 on Random. Sorted may pass merely
   because it has more owners, not because its likelihood is more informative.
2. **Annotation completeness.** A "catastrophic false positive" is defined by
   maximum same-class ground-truth intersection over union below 0.10. If the
   panel's ground truth is incomplete, an unlabeled but real object counts as
   catastrophic, and "rejecting the catastrophic tail" would partly mean
   rejecting correct detections. Earlier work in this investigation explicitly
   named an incomplete-annotation confound and required refined labels; the
   parent unit did not verify that its ground truth was the refined version.

## Strongest Alternative

For question 1, the alternative is that the filter has no checkpoint-specific
power at all and the three verdicts are a function of denominator size. Under
that alternative, equalizing the denominator collapses the difference between
Sorted and Permutation.

## Method

**Equal-owner-count resampling.** For each checkpoint, draw 200 random subsets
of size 44 — Random's full count — from its recovered-owner clusters. For each
draw, redo the leave-one-image-out threshold selection using only the drawn
subset as the retention constraint, and measure rejection over the **full**
catastrophic tail. Per-fold feature mappers and cluster scores do not depend on
the drawn subset, so they are computed once per fold and reused, which is what
makes 200 draws cheap.

Random's recovered-owner set is exactly 44, so its resampling is degenerate: it
returns one repeated value and supplies no robustness information. This is
reported rather than hidden, because a zero-variance column is easy to misread
as a stable result.

**Panel annotation provenance.** Count ground-truth objects in the panel
definition by origin. The refinement pass marks human-added objects with a
**negative** `coco_ann_id`; the presence of the key is not the discriminator,
and testing for presence rather than sign gives the opposite answer.

## Artifact Handle

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-29-three-checkpoint-human-refined12-max3084/likelihood-mining-v1/robustness-common-owner-count.json`

Produced by `scripts/research/analyze_common_owner_count_robustness.py`, which
imports the parent unit's `analyze_cluster_confidence_retention.py` rather than
reimplementing scoring, thresholding, or fold construction.

## Scope and Non-Goals

Panel, checkpoints, clusters, labels, likelihoods, and the pre-registered
20%-rejection bar are inherited unchanged from the parent unit. This unit does
not re-derive them, does not change any threshold, and does not extend the panel.
It cannot answer whether the parent conclusion survives a larger or differently
annotated cohort.

## Terminology

Terms are inherited from the parent unit. One addition:

- **Degenerate resampling**: a checkpoint whose recovered-owner count is at or
  below the common subsample size, so every draw returns the same full set and
  the resulting spread is zero by construction rather than by stability.
