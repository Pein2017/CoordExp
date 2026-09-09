---
title: Iterative Soft-Owner QP versus Refreshed RLOO Vertical
description: One current-policy K4 round comparing ordinary trajectory RLOO with a bounded elastic owner-preservation correction.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-09-03-iterative-soft-owner-qp-vs-rloo-vertical
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-03
---

# Iterative soft-owner QP versus refreshed RLOO

Completed evidence and disposition: [results](results.md).

## Frozen question

> From the exact broad-support unmerged DoRA anchor, does one fresh
> current-policy K4 round produce a train-248 gain under ordinary
> coverage-graded RLOO, and can a soft owner-frontier QP improve that same
> proposal by trading fewer incumbent losses against missing-owner gains?

The decision-owning behavior is natural-greedy, category-consistent global
one-to-one annotated-owner coverage on the frozen train-248 cohort.  The
primary scalar is the candidate-minus-anchor change in the sum of IoU50,
IoU60, and IoU80 owner hits; every threshold and owner gain/loss identity is
also reported.  A positive train result is retained even without validation.
This unit makes no F1, mAP, full-scene precision, image-disjoint transfer,
repeatability, or production claim.

## Immutable specimen

- **Base composition:** the base model and step-2444 selected-token embedding
  delta recorded by the anchor receipt below.
- **Anchor adapter:**
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-03-broad-support-coverage-graded-static-rloo-one-update/adapter`,
  fingerprint
  `370abdc4611b893942f4738d9eaca2f46eafd13db59bafd30d0f7f6cac10addc`;
  `adapter_model.safetensors` SHA-256
  `861c24cf85d068a44fac51314d70e4779e34656937af691f5409e19c31c190e5`.
- **Anchor receipt:**
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-03-broad-support-coverage-graded-static-rloo-one-update/receipt.json`,
  SHA-256
  `803c48ab36c95509defaa2d7cf0400cffb04d9b9ea4ead0cb4040373530c1e94`.
- **Anchor train read:**
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-03-broad-support-coverage-graded-static-rloo-train248-read/analysis.json`,
  SHA-256
  `d81dd644cc4bc401bd9c059e4edee49c29dbcba7769a2e5d3af7bba7eab1de0a`.
- **Train data:**
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-01-annotated-owner-direct-c-d0-pilot/materialization-v1/train-common-base.jsonl`,
  SHA-256
  `86d34cc2efbce9814847dd12fc12cab2f46d04168ce39905bfd62a93ced783fd`.
  Its provenance is the COCO `1024 * global max len 12000`, `xy_sorted`
  source; transcript sorting is a training preference, never a natural-decode
  admission invariant.
- **Inference leaf:**
  `configs/coordexp_swift/infer/qwen3_vl_2b_broad_support_coverage_graded_static_rloo_train248.yaml`,
  SHA-256
  `54d1d63f90c6fb2da336c1f3dfba1dd9f035e468e6b778221d91a61ca45c25b6`.
- **Output root:**
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-03-iterative-soft-owner-qp-vs-rloo-vertical`.

All new outputs are write-once.  The final aggregate receipt binds the source
commit, all paths and hashes, adapter-only composition, and cold-read evidence.

## Fresh action bank and fixed population

Acquire four complete natural sampled trajectories per image with seeds
`2026090201,2026090202,2026090203,2026090204`, temperature `1.0`, top-p `1.0`,
repetition penalty `1.0`, maximum 3,084 new tokens, and natural `im_end` stop.
Reusing seeds controls random-number schedule; generation is fresh because the
behavior policy is the broad-support anchor, not C.

The completed acquisition is bound by
`acquisition/launch-plan.json` SHA-256
`8047b86332d4cf7ad72ea10f1202c860b383bf2938255ab2f65b2df79be3be44`
and `acquisition/analysis.json` SHA-256
`cf66435db8f775490d5c1da40a42a5b9f441f4a2ee7e095a00129f92e1fa5c45`.
It contains all 992 cells, 951 clean trajectories, no cap or malformed
rollout, and 218 complete-clean K4 images.  Within the fixed prior 218-image
optimizer population, one image is newly unusable and remains a zero term;
146 groups execute 584 trajectories.  These are action-bank and signal-supply
facts, not candidate-quality evidence.

The canonical paired plan is `plan.json` SHA-256
`13de2af7a5c310fcaa7d1979871122d49fde65bf7c47dd85d82847ad978dc20a`;
it selects 32 fragile incumbents from 466 candidates across 23 images.

All 248 images and 1,798 annotated owners are audited.  Optimization uses the
same 218 image ids admitted by the preceding broad-support unit.  If a member
is not complete-clean in the fresh bank, it contributes zero while remaining
in the fixed 218-image denominator.  No replacement image enters.  Valid
unmatched predictions remain unknown; duplicate, invalid, dropped, malformed,
cap, EOS, length, and ordering events are descriptive debt monitors.

For complete-row owner `o`:

```text
u_iko = .90 I(o matched at IoU50)
      + .05 I(o matched at IoU60)
      + .05 I(o matched at IoU80)
r_ik  = sum_o u_iko / |O_i|
A_ik  = r_ik - mean_{j != k}(r_ij)
```

This decomposes exactly into owner-specific leave-one-out score gradients.
Current greedy IoU50 matches are incumbents.  A miss is flippable only when at
least one admitted current-policy rollout matches it at IoU50.  An incumbent is
fragile when its graded credit varies across K4.  Zero-support owners remain a
reported limitation, not negative examples.

## Paired interventions

Both candidates start independently from the immutable anchor, consume the
same action bank, use the same fixed denominator, and touch only the same 588
shared language-DoRA tensors.

### A: refreshed RLOO

Capture the exact one-step fresh AdamW proposal using learning rate `2.5e-6`,
betas `(0.9, 0.999)`, epsilon `1e-8`, zero weight decay, and global gradient
clip `1.0`.  There is no token-length normalization, whitening, PPO ratio,
critic, reference model, KL term, scheduler, or saved optimizer state.

### B: soft owner-frontier QP

Rank fragile incumbents by descending number of sampled IoU50 losses, then
descending summed graded-credit shortfall from the current greedy credit, then
numeric image id and owner id.  Retain at most 32.  Compute each retained
owner's complete-trajectory policy-gradient row, normalize it to unit
Euclidean norm, and keep only rows with negative dot product against the exact
Arm-A parameter delta `d0`.

Solve once, without tuning:

```text
minimize_d,s  0.5 ||d-d0||_2^2 + 0.5 sum_i s_i^2
subject to    a_i^T d + s_i >= 0
              s_i >= 0
```

The dual is bounded by at most 32 rows and is accepted only with finite
primal/dual/KKT evidence and identical reconstructed update hashes on all
ranks.  If no row is threatened, B equals A by construction; record an
inactive QP and do not publish or evaluate a duplicate adapter.  This is a
local gain-retention approximation.  It does not guarantee preservation under
natural decoding and does not protect owners absent from sampled variation.

## Bounds and gates

- exactly eight FP32/SDPA ranks and one resident model per rank;
- fixed 218-image denominator and complete image groups owned by one rank;
- at most 872 trajectory forwards/backwards; each trajectory score gradient is
  reused to accumulate the total RLOO gradient and all selected owner rows;
  semantic-zero groups with no selected row may be elided without changing the
  denominator;
- at most 32 disk-backed owner rows, one streamed 68.7 MiB FP32 CUDA row buffer,
  and one `32 x 32` Gram matrix; the realized plan places at most seven local
  rows (480.8 MiB scratch) on one rank, while rank-zero threatened-row scratch
  is bounded by 2.15 GiB; the QP adds no model forward;
- one proposal and at most two distinct adapter payloads; no further round,
  dose, penalty, cap, or seed sweep;
- expected peak below 36 GiB CUDA and 16 GiB host RSS per rank, artifact
  payload below 4 GiB, and 90-minute update ceiling after acquisition absent a
  concrete fault.

Identity mismatch, nonfinite gradient, invalid owner decomposition, solve/KKT
failure, cross-rank mismatch, overwrite attempt, merged weight, or failed cold
read is `MECHANICAL_INVALID` for the affected candidate.  It is repaired only
from the immutable anchor and never interpreted as a model result.

## Decision table and stop

After cold natural-greedy train-248 reads:

1. Record every candidate whose summed IoU50/60/80 change versus anchor is
   positive as train-side progress.
2. Iterative refreshed RLOO receives support only if A is positive.
3. The soft-QP correction receives incremental support only if B is distinct,
   positive versus anchor, and strictly exceeds A on the primary scalar.  A
   tie may still be reported as lower-churn descriptive evidence but does not
   establish QP benefit.
4. If both are nonpositive, stop this dose/profile.  If either is positive,
   stop this unit anyway and bring the result to the user; validation or a
   later rerollout is a separately authorized unit.

Dense scenes with repeated books, fruit, vegetables, or cups remain included.
Their pre-existing annotation-quality flags are reported as an overlapping
sensitivity partition, not used to silently change the primary population.
