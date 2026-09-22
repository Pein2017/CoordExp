---
title: Sorted Crossing-Boundary Owner Release and Realization - Results
description: Verified owner-level decomposition at the exact sorted crossing boundary, with a bounded local downstream-compatibility readout.
type: investigation
role: research-results
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-08-03-sorted-crossing-boundary-owner-release-realization
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-08-04
---

# Results

## Verdict

The unit is complete and independently verified.

At the exact native boundary where the geometry-sorted route crosses one of
the `26` supported false-negative owners, `24` deterministic cases are
interpretable. Their frozen primary branches are:

| Branch | Count |
| --- | ---: |
| displaced | `17/24` |
| release lost | `6/24` |
| target-conditioned realization failed | `1/24` |

The remaining two owners are ambiguous. The frozen two-thirds rule therefore
routes the `displaced` branch: `17 >= ceil(2 * 24 / 3) = 16`.

This route is real but thin and heterogeneous. Of the 17 displaced owners,
ten have both likelihood and coordinate-greedy displacement evidence, five
have likelihood evidence only, and two have coordinate-greedy evidence only.
Removing only the two greedy-only cases leaves `15/22`, exactly the required
threshold. Two additional, non-frozen single-channel sensitivities fall below
two thirds: removing the five likelihood-only cases leaves `12/19`, and
keeping only the ten two-channel cases leaves `10/17`.

The strongest bounded interpretation is therefore:

> At a material subset of exact sorted crossing boundaries, the decoder's
> target-conditioned geometry path is dynamically reprioritized toward a
> different still-uncovered owner of the same category. This is evidence for
> fragile within-category identity scheduling, not for STOP, an already-
> covered-owner ledger failure, suppression by the exact crossing row, or a
> final-set causal effect.

No training objective, auxiliary head, slot, owner-commit token, coverage
ledger, detector, RL policy, or architecture is promoted.

## Evidence boundary

| Item | Executed evidence |
| --- | --- |
| Checkpoint and panel | Geometry-sorted step `4887`; frozen human-refined twelve-image panel |
| Plan | `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-03-sorted-crossing-boundary-owner-release-realization/20260804T013856Z/plan/` |
| Primary run | `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-03-sorted-crossing-boundary-owner-release-realization/20260804T020853Z/primary-merged/` |
| Primary analysis | `.../20260804T020853Z/primary-analysis-fragility-v2/` |
| Secondary merge | `.../20260804T020853Z/secondary-merged-v2/` |
| Secondary analysis | `.../20260804T020853Z/secondary-analysis-v3/` |
| Primary cohort | `26` U-bound crossing owners: `12` matched-E and `14` unmatched-E |
| Interpretable primary denominator | `24`: `11` matched-E and `13` unmatched-E |
| Secondary requests | `64`: `26` P+C->E, `26` P+E+C->F, and `12` benign substitutions |
| New free rollout | None |
| Optional coordinate sampling | Not activated; deterministic evidence assigned 24 cases and left two ambiguous |

All conclusion-bearing strings and row segments use sealed token IDs. The
secondary HF capture passed cached-versus-uncached parity with maximum selected
logit difference `1.71661376953125e-05`, below the frozen `1e-3` tolerance,
and no request was quarantined.

## What displacement means here

The two displacement channels are not interchangeable:

- `15` owners have a different physical owner as the best frozen candidate;
- `12` owners have a coordinate-only greedy box that strict-matches a
  different physical owner; and
- only `10` owners satisfy both.

Every identified displacer was still uncovered before `P`. None of the twelve
greedy displacers is the physical owner of crossing row `E`. Nine of those
twelve displacers are eventually native true positives and three remain native
false negatives. For likelihood displacers the corresponding split is nine
and six.

This rejects two simple readings:

1. the model is merely returning to an already covered owner; and
2. the exact owner emitted in `E` directly replaces the target.

It does not establish that adding `E` caused the displacement. The descriptive
timing controls show analogous within-category reprioritization away from an
exact crossing boundary, and the matcher cannot detect cross-category
displacement because the candidate family is category-local.

Sharing a normalized description is construction-determined for physical
owners in the same candidate family. It is an audit property, not evidence for
a discovered same-description mechanism.

## Secondary local compatibility

The exact clean GT row for skipped owner `C` was inserted as an oracle row,
then the exact native downstream row was teacher-forced under both roots.
This readout is local and sequence-level only.

For both P+C->E and P+E+C->F:

| Segment | Positive | Negative |
| --- | ---: | ---: |
| description | `9/26` | `17/26` |
| coordinates | `8/26` | `18/26` |
| complete row | `5/26` | `21/26` |

The twelve benign same-length substitutions are not sign-neutral either:
description is negative for `8/12`, coordinates for `7/12`, and the complete
row for `6/12`. Consequently, sign counts alone do not identify a special
catch-up penalty.

The verified artifact reports every exact owner delta and only within-image
raw summaries. It intentionally contains no cross-image raw-delta median,
range, quantile, or fitted compatibility class.

The scientific review found a conclusion-shaping pattern that remains a
successor hypothesis, not a result of this frozen unit: large coordinate
changes appear concentrated where inserted `C` and downstream `E` share a
description, especially when `E` is unmatched. The benign control replaces a
same-length row in place; it does not control adding a whole row, recency, or
position shift. A clean inserted `C` may also suppress a geometrically nearby
corrupted rendition of the same physical owner, which would be healthy
duplicate suppression rather than downstream damage.

## Observed

- `17/24` interpretable crossing owners enter the frozen displaced branch.
- All identified displacers are uncovered before `P`.
- No greedy displacer equals the strict-matched owner of crossing row `E`.
- The displacement evidence is channel-heterogeneous and only one owner above
  the frozen route threshold.
- Exact clean-row insertion frequently changes downstream selected-token
  likelihood, but the admissible benign control also changes it and does not
  isolate insertion length or position.

## Supported

- Within-category owner identity and scheduling are unstable at some exact
  crossing boundaries even after the target description is forced.
- A small CPU-only geometric relation probe is warranted before another GPU
  intervention.

## Not supported

- No claim that `E` causally suppresses `C`.
- No claim that the model forgot a covered set.
- No claim that same-description crowding is the mechanism merely because the
  same-category candidate family shares descriptions.
- No pooled cross-image downstream effect size.
- No final-set retention, eventual recovery, natural-stop, or free-rollout
  conclusion from the secondary readout.
- No population prevalence beyond the frozen twelve images.
- No training or architecture promotion.

## Next discriminator

Run exactly one CPU-only geometric-relation successor over the existing sealed
bytes. For each crossing owner, compute the relation between inserted `C` and
downstream `E`: IoU, normalized center distance, extent/area relation, and
sorted-key adjacency. For the twelve greedy displacement cases, compute the
same relation between target and displacer. Join these fixed geometric fields
to the already sealed per-owner coordinate deltas and branch tags without
fitting a threshold.

- If the large negative coordinate tail is confined to high-overlap C/E pairs,
  reinterpret it as identity slippage or duplicate suppression and stop.
- If it persists among clearly separated C/E pairs, generic within-category
  competition survives and one small matched-length added-row GPU control may
  become eligible.
- If cells are too small or incoherent, stop inconclusive and do not add GPU
  work.

Image `2299` is not retroactively inserted into this frozen twelve-image
denominator. It is admitted prospectively to the following versioned
thirteen-image panel from the training-side
`rescale_32_1024_bbox_len12000/val.coord.jsonl` authority: `46` owners,
including `38` persons and `8` ties.
