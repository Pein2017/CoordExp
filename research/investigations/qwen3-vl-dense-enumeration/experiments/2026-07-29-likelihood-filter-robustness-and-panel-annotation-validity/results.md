---
title: Likelihood Filter Robustness and Panel Annotation Validity Results
description: Equalizing the recovered-owner count does not collapse the difference — Sorted still passes and Permutation still fails — so the checkpoint split is real, not a denominator artifact; and 49% of the panel's ground truth is human-added, which materially supports the catastrophic false-positive definition.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-07-29-likelihood-filter-robustness-and-panel-annotation-validity
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-07-29
---

# Likelihood Filter Robustness and Panel Annotation Validity Results

Protocol is owned by [unit.md](unit.md). The parent result is
[2026-07-29 Sampled Object-Span Likelihood and Consensus Filtering](../2026-07-29-sampled-span-likelihood-and-consensus-union-filtering/results.md).

## Decision

Both threats are cleared. The parent unit's checkpoint-specific verdict stands,
and its catastrophic false-positive definition rests on ground truth that is
roughly twice as complete as raw COCO on these images.

## Composition is not the explanation

With every checkpoint's retention denominator subsampled to a common 44
recovered owners, 200 draws each:

| Family | Checkpoint | Needed | Median rejection | p10 – p90 | Draws passing |
|---|---|---:|---:|---|---:|
| coordinate only | Sorted | 40/197 | 0.289 | 0.132 – 0.355 | **150/200** |
| coordinate only | Permutation | 28/138 | 0.116 | 0.116 – 0.167 | **1/200** |
| coordinate only | Random | 27/131 | 0.191 | degenerate | 0/200 |
| support + likelihood | Sorted | 40/197 | 0.348 | 0.107 – 0.477 | **143/200** |
| support + likelihood | Permutation | 28/138 | 0.087 | 0.058 – 0.101 | **0/200** |
| support + likelihood | Random | 27/131 | 0.214 | degenerate | 200/200 |

Sorted still passes in three quarters of draws at half its original owner count.
Permutation fails in 199 of 200 and 200 of 200 draws. The gap between them is
**not** a function of denominator size, so the alternative in the unit is ruled
out: Permutation's failure is a property of that checkpoint, not of its smaller
recovery set.

Random's row is **degenerate and carries no robustness information**. Its
recovered-owner set is exactly 44, so every draw is the same full set and the
zero spread is arithmetic, not stability. Its 200/200 is the same one-cluster
margin the parent unit already flagged, repeated 200 times. It must not be read
as the most robust cell in the table — it is the least informative.

## The panel is substantially better annotated than COCO

| Source | Objects |
|---|---:|
| COCO original | 176 |
| Human-added by the refinement pass | **170 (49.1%)** |
| Total | 346 |

Nearly half the panel's ground truth does not exist in COCO. On these dense
scenes — 15 to 50 objects per image — raw COCO would have under-labeled by
roughly a factor of two, and a catastrophic-false-positive count computed
against raw COCO would have been badly inflated by real but unlabeled objects.
The parent unit's definition rests on the refined labels, so that inflation does
not apply to it.

This also corrects a trap in the artifact: human-added objects are marked by a
**negative** `coco_ann_id`, not by absence of the field. Testing for presence
returns "zero human-added objects" and yields exactly the wrong conclusion about
panel quality.

## Observed

1. At equal owner count, Sorted passes the pre-registered bar in 150/200
   (coordinate only) and 143/200 (support plus likelihood) draws.
2. At equal owner count, Permutation passes in 1/200 and 0/200 draws.
3. Sorted's spread at n=44 is wide (0.107–0.477 for support plus likelihood),
   consistent with the parent unit's warning that these denominators are small.
4. 170 of 346 panel ground-truth objects are human-added.

## Supported

The parent unit's central claim — coordinate likelihood carries real grounding
information that consensus cannot supply at the operating point — is robust to
the most obvious confound, and its outcome measure is not an artifact of
incomplete annotation.

## Ruled out

- **Denominator size** as the explanation for the Sorted/Permutation split.
- **Incomplete annotation** as an explanation for the size of the catastrophic
  tail on this panel.

## Unresolved

- **Why Permutation fails** remains unexplained. Ruling out composition makes
  this sharper, not answered: something about the permutation-bundle arm makes
  its badly grounded rows carry coordinate likelihood indistinguishable from its
  correct ones.
- **Random remains undecided.** Nothing here adds information about it; the
  degenerate resampling cannot.
- Sorted's wide spread at n=44 means the panel is still the binding constraint
  on precision, and no threshold is transferable.

## Not claimed

No filter, policy, training change, or threshold is promoted. Twelve images
remain twelve images; ruling out two confounds does not make the result
generalize. Annotation completeness is established relative to COCO, not
absolutely — no claim is made that the refined panel is exhaustive.

## Next discriminator

Permutation is now the informative case. The cheapest probe that could explain
it: compare, within Permutation, the coordinate-likelihood distributions of its
catastrophic clusters against Sorted's, conditioned on matched object category
and box scale. If Permutation's catastrophic rows are confidently placed on
plausible-looking but wrong locations while Sorted's are not, that separates a
grounding failure from a calibration failure, and it needs no new generation.
