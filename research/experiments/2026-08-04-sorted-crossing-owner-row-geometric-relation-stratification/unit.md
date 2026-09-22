---
title: Sorted Crossing Owner-Row Geometric Relation Stratification
description: CPU-only test of whether C/E geometric proximity explains the coordinate-delta tail and whether greedy displacers are geometric neighbors of their targets.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete
unit_id: 2026-08-04-sorted-crossing-owner-row-geometric-relation-stratification
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-08-04
---

# Sorted Crossing Owner-Row Geometric Relation Stratification

## Decision and outcome

The completed [crossing-boundary unit](../2026-08-03-sorted-crossing-boundary-owner-release-realization/results.md)
routes a fragile displaced branch and shows that inserting clean skipped owner
`C` frequently lowers likelihood of exact downstream row `E`. The scientific
review identifies two unresolved explanations:

1. generic added-row, recency, or position sensitivity; and
2. `E` may be a geometrically corrupted rendition of `C`, so a clean `C`
   insertion suppresses a near duplicate rather than damaging a distinct
   downstream owner.

This unit asks the smallest discriminator that requires no new model work:

> Are the primary P+C->E coordinate changes concentrated in geometrically
> overlapping C/E pairs, and are the primary greedy displacers merely
> geometric neighbors of their targets?

The outcome is one owner-level geometric census over the frozen 26 crossing
owners and the twelve frozen greedy displacement pairs. It may close the local
catch-up-damage interpretation, or make exactly one small added-row control
eligible. It cannot promote training or architecture.

## Immutable inputs

Read only these sealed products:

- crossing plan:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-03-sorted-crossing-boundary-owner-release-realization/20260804T013856Z/plan/`;
- primary owner rows:
  `.../20260804T020853Z/primary-analysis-fragility-v2/owner-rows.jsonl`;
- secondary owner rows:
  `.../20260804T020853Z/secondary-analysis-v3/secondary-owner-rows.jsonl`;
- secondary merge rows for exact scored token spans:
  `.../20260804T020853Z/secondary-merged-v2/secondary-compatibility-rows.jsonl`;
- the plan's sealed lineage to the canonical owner ledger.

The analyzer must bind every input digest and source hash before emitting an
output. It must not open a model, use a GPU, rescore a token, change a branch,
or alter the 26-owner denominator.

Image `2299` is not in these frozen bytes and is not added to this denominator.
It is reserved for the following prospective thirteen-image panel, using only
`/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl`
as GT authority. The admission receipt to carry forward is:

- complete annotation file SHA-256:
  `81d674070d4b588488a2cb911c09f765b63c0e6d035b50db27ee0a41ff2a1894`;
- exact source-line SHA-256 for image `2299`:
  `ce19853c74a595f22cc183ce450e561f2da3216e54a1e499cfbca1be7e1c425b`;
- image SHA-256:
  `cd7199a37188c9ac6481520175866cbd78fa6fba35f4290bb0b60c78afcb2df3`;
- `46` physical-owner annotations: `38 person` and `8 tie`; and
- among persons, `12` positive COCO annotation IDs and `26` manual negative
  IDs.

The first row of the older
`2026-07-20-own-prefix-entity-transition-and-coordinate-boundary-calibration-training-screen/smoke-b-v1/ordinary-inference-input.coord.jsonl`
is byte-identical to this source line and has the same line SHA. That establishes
annotation comparability only. Historical outputs are not admitted into the
new thirteen-image denominator without their own checkpoint/runtime and
matcher review.

## Exact geometric fields

All boxes use the sealed norm-1000 `x1,y1,x2,y2` token values. No pixel-space
round trip is permitted. A valid box requires `x2 > x1` and `y2 > y1`.
Widths, heights, areas, intersections, and IoU use continuous xyxy geometry
without a `+1` endpoint convention. Center containment includes the boundary,
and normalized center distance divides Euclidean bin distance by
`1000 * sqrt(2)`.

Recover each bin as `coord_token_id - 151670`. The analyzer must bind that
start ID to the sealed census tokenizer/coordinate-token identity and reject
any result outside `[0,999]`; it may not infer the offset from the observed
minimum or convert through decoded pixels.

For inserted owner `C` and exact downstream row `R`, emit the fields below.
`R=E` for the primary P+C->E arm. `R=F` for the optional
P+E+C->F sensitivity arm. Never reuse E geometry for an F likelihood delta.

- IoU;
- intersection over C area and intersection over R area;
- Euclidean center distance divided by the norm-1000 canvas diagonal;
- signed `dx`, `dy`, `dw`, and `dh`, each divided by 1000 and defined as
  `R - C` (center displacement for dx/dy, extent displacement for dw/dh);
- R/C area ratio and absolute log area ratio;
- whether C's center is inside R and whether R's center is inside C;
- exact description equality after the frozen normalization;
- R strict-match status and owner ID if matched; and
- sorted-key rank gap: insert R's `(y1,x1)` key into the image's same-category
  physical-owner order and compare it with C's rank, using owner ID as the
  deterministic tie breaker. The row receives the stable synthetic tie key
  `row:{variant}:{gt_owner_id}` and sorts after every physical owner with the
  exact same `(y1,x1)` key. Here same-category means the same sealed normalized
  description within the image. Both owner and row keys are derived from
  sealed norm-1000 coordinate tokens; the pixel-space `owner_sort_key` is
  provenance only and is never mixed into this rank.

For each of the twelve greedy displacement pairs, emit the same fields between
the target GT owner and the displacing GT owner, plus their exact same-category
sorted-rank gap and the displacer's frozen native TP/FN disposition.

Reject invalid boxes, missing owners, inconsistent descriptions, nonfinite
fields, duplicate owner IDs, and any failure to reproduce `26` C/E rows or
`12` greedy pairs.

## Frozen joins and strata

Join, never recompute:

- primary branch and displacement sub-tags;
- matched-E versus unmatched-E;
- exact C/R description equality, using sealed `same_description_as_e` for E
  and the sealed F description for the sensitivity arm;
- P+C->E coordinate and complete-row exact paired deltas;
- P+E+C->F coordinate and complete-row exact paired deltas; and
- the one benign-substitution coordinate delta for the same image.

For each catch-up arm define an image-referenced coordinate change:

```text
relative_coordinate_delta = crossing_coordinate_delta
                            - same_image_benign_coordinate_delta
```

This is a descriptive reference, not a causal correction. The benign row is a
same-length replacement and does not control insertion length or position.

Predeclare these geometric labels:

- `high_overlap`: IoU >= 0.5;
- `any_overlap_or_center_containment`: IoU > 0 or either center lies inside the
  other box;
- `clearly_separated`: IoU == 0 and neither center lies inside the other; and
- `material_negative`: `relative_coordinate_delta <= -1.0` nat, meaning at
  least an `e^-1` likelihood-ratio reduction beyond the same-image benign
  reference. Each delta is the sum over its own scored row's four coordinate
  tokens: the crossing delta over the arm's downstream row E or F, and the
  benign delta over that image's benign control's downstream row.

No cutoff may be changed after reading output.

## Reports

Publish:

1. one row for every primary C/E owner and every optional C/F sensitivity
   owner with every exact geometry and joined field;
2. one row for every target/displacer pair;
3. exact counts for the Cartesian strata:
   same/different description x matched/unmatched E x geometric label x
   material/nonmaterial change;
4. per-image values and within-image summaries;
5. Spearman rank association between the two continuous geometry measures
   (IoU and center distance) and relative coordinate delta, reported as a
   descriptive association with leave-one-image-out coefficients; and
6. readable scatterplots and image crops for every material-negative case,
   with C and E or target and displacer assigned stable colors and IDs.

Raw likelihood deltas are never summarized by one cross-image median, range,
or quantile. Exact owner rows and rank associations are allowed because this
unit prospectively owns them.

## Decision rule and stop boundary

Let `M_E` be the material-negative C/E rows from the primary P+C->E arm. The
optional C/F arm is a named sensitivity and may qualify but never change the
primary decision. The same owner appearing in both arms is never counted as
two independent votes.

- **identity-slippage / duplicate-suppression eligible**: `|M_E| >= 3`, at
  least 75% of M_E are same-description plus
  `any_overlap_or_center_containment`, and there are zero material-negative
  clearly-separated different-description rows. Close the catch-up-damage
  interpretation and do not run another GPU probe.
- **separated competition survives**: at least three rows in M_E are clearly
  separated from C. Generic insertion or within-category
  competition remains live; exactly one small matched-length added-row control
  may be proposed, but is not authorized by this unit.
- **inconclusive**: every other result. Stop without GPU, training, or
  architecture.

The eligible routes are evaluated in fixed order: `separated competition
survives` is tested first. `identity-slippage / duplicate-suppression eligible`
may be routed only when fewer than three rows in M_E are clearly separated. If
both conditions would otherwise hold, route `separated competition survives`.

A non-inconclusive route additionally requires that its qualifying rows span
at least two distinct images: the same-description overlap-or-containment rows
of M_E for identity slippage, and the clearly separated rows of M_E for
separated competition. Rows from one image share that image's single benign
reference delta and are never counted as independent votes. If all qualifying
rows come from one image, route inconclusive.

The twelve greedy displacement pairs are a descriptive compatibility check;
they cannot change this decision rule. If most are adjacent/overlapping, call
the primary route identity-local. If most are distant, call it distributed
same-category reprioritization. Do not infer causality.

## Claim boundary

This unit may describe geometric association and owner identity ambiguity. It
cannot establish a visual-tower absence, a causal insertion effect, eventual
owner retention, free-rollout behavior, a final-set value, or a population
estimate. It cannot add image `2299` to any completed denominator.
