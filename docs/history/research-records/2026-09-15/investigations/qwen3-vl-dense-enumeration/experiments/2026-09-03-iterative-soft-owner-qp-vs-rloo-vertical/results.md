---
title: Iterative Soft-Owner QP versus Refreshed RLOO Results
description: A refreshed trajectory update continues train-side progress, while the registered local soft-owner QP correction has no incremental support.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-09-03-iterative-soft-owner-qp-vs-rloo-vertical
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-03
---

# Iterative soft-owner QP versus refreshed RLOO result

## Outcome

- Mechanics: **`MECHANICALLY_VALID`**.
- Arm A: **`TRAIN_GAIN_RECORDED_REFRESHED_RLOO`**.
- Arm B: **`SOFT_OWNER_QP_NO_INCREMENTAL_SUPPORT`**.

Starting from the preceding broad-support unmerged DoRA adapter, one fresh K4
rollout-and-update iteration improves the registered natural-greedy train-248
owner-count vector again.  The paired QP correction is also positive versus the
anchor, but is worse than the ordinary update it corrects.

| threshold | anchor | Arm A: refreshed RLOO | A gain/loss/net | Arm B: soft QP | B gain/loss/net | B - A |
|---|---:|---:|---:|---:|---:|---:|
| IoU50 | 1,266 | 1,267 | 4 / 3 / **+1** | 1,267 | 5 / 4 / **+1** | 0 |
| IoU60 | 1,197 | 1,201 | 8 / 4 / **+4** | 1,199 | 6 / 4 / **+2** | -2 |
| IoU80 | 902 | 908 | 8 / 2 / **+6** | 907 | 7 / 2 / **+5** | -1 |
| summed thresholds | 3,365 | 3,376 | - / - / **+11** | 3,373 | - / - / **+8** | **-3** |

The sum counts the same owner separately at three IoU thresholds.  Relative to
the original C adapter (`1,259/1,186/896`), the cumulative train movement is
`+8/+15/+12` for A and `+8/+13/+11` for B.  This is evidence that rerolling the
updated policy and optimizing again can continue improving this finite training
cohort; it is not yet a learning curve, repeatability result, or generalization
claim.

The authoritative reducer is
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-03-iterative-soft-owner-qp-vs-rloo-vertical/train248-analysis-v2.json`,
SHA-256
`5b97acdeb01fd863b1d84439c32c4990504a5bc63f44eafc6e6b2a425d7ae0c8`.

## What the QP did—and did not do

The QP operated on the same internal 588-tensor DoRA surface as RLOO; it was
not an output-head payload or external per-image memory.  From 32 selected
fragile IoU50 incumbents, eight first-order owner rows opposed the Arm-A
proposal.  The elastic solve was active, converged in five iterations, and had
KKT residual `2.612e-10`.

The correction was deliberately small: Arm A update norm was `0.01059157`, Arm
B norm was `0.01058039`, and their requested-direction cosine was `0.999635`.
It nevertheless changed 14 of 248 natural decodes.  All eight QP-threatened
owners remained IoU50 hits under both A and B, so none was a realized Arm-A
loss for the QP to rescue.  B instead exchanged owners elsewhere and finished
three summed threshold hits behind A.

This rejects the registered **local frontier and penalty at this dose** as an
incremental improvement.  It does not show that constrained optimization in
general is useless.  The direct lesson is narrower: rollout-graded local owner
gradient signs did not identify the discrete greedy-route losses that actually
distinguished these two nearby updates.

## Data-quality and cap sensitivity

The primary cohort retains all 248 rows.  The nine pre-flagged dense repeated
book/fruit/vegetable/cup images contribute `+1/+2/+3` to both A and B.  Excluding
them leaves:

| candidate | IoU50 | IoU60 | IoU80 | summed |
|---|---:|---:|---:|---:|
| Arm A | 0 | +2 | +3 | **+5** |
| Arm B | 0 | 0 | +2 | **+2** |

Excluding both those nine rows and the two length-cap rows leaves the same
`+5` for A and `+1` for B.  Thus A's full IoU50 gain comes from the flagged
dense partition, but its IoU60/80 improvement survives on the remaining clear
rows.  Visual inspection also found clear non-dense gain/loss cases; the result
cannot be reduced to annotation noise alone.

The descriptive sensitivity receipt is `sensitivity-analysis.json` under the
same output root, SHA-256
`8f0fbb6c10cbaa6205602819ce8caf523c414e55777ae68681d81744cb4d167c`.

Image 287484 is a simple bed scene with only a few bedside books, but both
candidates emit roughly 85 repeated book rows and hit the cap.  B's IoU50 book
owner swap on this image is therefore not evidence of useful precision.

## Natural behavior and mechanics

- A and B retain natural `im_end` on 246/248 rows and the same two length caps
  as the anchor;
- duplicate candidates move `42 -> 41` for both; strict physical duplicates
  remain `2`;
- parser-dropped predictions move `555 -> 549` for A and `552` for B;
- invalid predictions move `2 -> 3` for both;
- ordering violations move `78 -> 81` for A and `79` for B and remain monitors,
  never admission failures;
- A changes 70/248 anchor decodes, B changes 69/248, and A/B differ on 14 rows.

The eight-rank paired run evaluated 146 complete K4 groups and 584 trajectory
VJPs.  Arm A's raw gradient norm was `1.278560`, clipped to `0.999999` before
one fresh AdamW step.  Maximum observed allocated CUDA memory was 28.64 GiB,
reserved CUDA memory 35.84 GiB, host RSS 11.76 GiB, and runtime 111.81 seconds.
Both candidates passed cold adapter-only reload and full train-248 inference;
the live model was restored exactly to the anchor.  No merged checkpoint was
created.

The aggregate update receipt is
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-03-iterative-soft-owner-qp-vs-rloo-vertical/paired-update/receipt.json`,
SHA-256
`2c0f108d40fce199c4e35e99d0ededc09c6d6d53607c97d0b929542ebef299e9`.
Arm A's unmerged adapter fingerprint is
`738ce575b7dac37b39aa4b830d6abe093f8a867a48e7bc728936c2900c9ca380`;
Arm B's is
`c7a07bc7b77ad76ffc458faedffabf681d06c2b1b537d117dd5b5c6248d01254`.

The first reducer had a display-only dense-partition join bug caused by numeric
versus zero-padded image ids.  Its primary 248-row metrics were unaffected; it
is explicitly superseded by v2 in
`train248-analysis-v1-superseded.json`, SHA-256
`68960221635efe626f32628911160e8c26860afe1087f655c5ac43514900f745`.

## Claim boundary and stop

**Observed:** one refreshed current-policy trajectory update continues
train-248 progress from an already updated adapter.  The registered local
soft-owner QP is mechanically sound but provides no incremental behavioral
benefit over that update.

**Not established:** validation transfer, repeatability across seeds, F1/mAP,
full-scene precision under incomplete labels, production readiness, or a
general failure of QP-based optimization.

Per the registered stop rule, this unit ends after the paired train read.  Arm
A is the preferred train-side candidate for a separately decided validation
read or another rerolled iteration; neither is part of this unit.

Closeout checks: 11 relevant tests pass; Ruff, compilation, and strict OpenSpec
validation pass.  A fresh reducer replay reproduces the authoritative v2
artifact byte-for-byte.  No OpenSpec archival or production promotion occurred.
