---
title: Continuous geometric deduplication for annotated-owner recovery
description: CPU preparation and a proposed matched repair contrast from Rweak64.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_separately
unit_id: 2026-09-08-coco-owner-recovery-dedup
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-08
---

# Scope and current gate

## User authorization update (2026-09-08)

User: "不设上限.若可以 8 卡的话更好" and then
"直接拉起,不用管当前的gpu usage". The GPU round is now authorized with
no elapsed/GPU-hour ceiling, using all eight GPUs without occupancy-based
waiting or stopping unrelated processes. The fixed scientific scope is unchanged:
one lambda0/lambda1 matched continuation,16 extra updates, refresh at0/+8 and
one final holdout512 evaluation. No sweep or confirmation was authorized.
Use four training ranks per arm on0–3 /4–7, global32 and microbatch2 unchanged;
the new real-boundary slice will check this topology. Final decode uses four
ranks per arm with the same global four-image groups. This ruling supersedes
the resource ceilings and pending-approval wording below, not the stop rule.
The originating CPU package remains completed historical evidence.

Final phase: both arms completed80, their own train256 refresh and holdout512.
The dedup second-update collective-order failure was repaired and resumed from
its valid65 checkpoint, with no added scientific updates; see [execution](execution.md).
Both durable chains ended; lead reduction and all-image checks completed.
See [results](results.md): fewer FP/repeats, owner net+5 versus control with
bootstrap interval[-11,+27]; improved recovery is not established. Stop rule
reached with no extension or promotion.

The [handoff](handoff.md) authorizes the CPU/design package only. GPU work below
is a **proposal awaiting user authorization**, not inherited launch authority.
Lead owns this directory and new loss code; one direct worker owns only the
new recount script/test and CPU recount artifacts. Prior uncommitted research,
old acceptance gates, shared trainer/consumer and experiment index are untouched.
Checkout HEAD: `73b8b3cc2614db052055c7c49fc33d6207b0f80a`.

**From Rweak64, does adding continuously active geometric-repeat unlikelihood
to a matched 16-update Rweak continuation retain/improve distinct annotated
owner recovery at IoU50 while reducing repetition and annotation-relative FP?**
This tests repair, not prevention from original Source. Both arms receive the
same extra optimization; comparing treatment only to old Rweak64 is not causal.

Primary evidence is natural, unpostprocessed category-compatible one-to-one
matches at IoU >= .50: TP/FP/FN, precision/recall, durable recovered/lost/net owner
IDs treatment-minus-control and each versus Source and Rweak64. All images,
including empty/failed/capped cases, remain in the denominator. Failed artifact
production prevents a complete contrast; it is not silently scored as success.
IoU60/80, invalid/drop, cap and length are diagnostics, not new vetoes. Report
trade-offs without inventing FP tolerance. This panel supports a bounded result,
not independent confirmation, physical completeness, AP, or hallucination claims.

## Hypothesis and strongest alternative

Repeated high-overlap generated rows provide a model-origin negative signal
that can suppress repetition while the unchanged Rweak objective retains owner
recovery. The strongest alternative is geometric jitter or general output
suppression: lower repeat counts without retaining useful owners. A sampled
four-coordinate surrogate does not penalize the entire high-IoU region, so
nearby replacement boxes may evade the rule. Natural TP/FP and owner sets are
the cheapest decision-bearing discriminator; no proxy-loss gate establishes
scientific success. Low signal on train256 may also limit transfer to holdout.

## Minimal differentiable intervention

`scripts/research/coco_geometric_dedup_loss.py` is a probe-level loss, NOT an
integrated trainer. Reuse `TrajectoryScorer` from the existing trainer: its full
current-prefix replay returns full-vocabulary CE aligned to each action token.
The existing canonical anchor, R teacher, mask, fixed denominator and per-image
`sum(M_mask)/sum(R_mask)` correction multiplier remain unchanged.

For each detached natural model-generated trajectory, parse complete rows with
the existing token grammar. Flag a later geometry-valid row once iff any earlier
valid row has prediction-to-prediction IoU **strictly > .95**, regardless of
category/description or GT. Compare to all earlier rows, including earlier
flagged rows; this is not greedy NMS. Report malformed tails and invalid rows.
Use the consumer's `coord_bins_to_pixel_xyxy` with the actual image dimensions
before IoU. It rounds to integer pixels: bin-space IoU is NOT interchangeable
near the strict threshold. Pixel-degenerate boxes are reported as invalid.

For flagged row j, let `s_j` be the mean of current-model CE at its four coordinate
tokens, conditioned on the exact generated prefix including its description.
`p_j = exp(-s_j)` is geometric-mean coordinate confidence, NOT the joint
probability of that coordinate tuple. The mean avoids multiplying four small
probabilities: the initial joint-probability version had saved-train image-mean
loss only0.00016805. This is a train-only design diagnostic, not gradient-norm
matching or evidence that the normalized surrogate improves natural generation.
Use `u_j = -log((1 - p_j + 1e-8)/(1 + 1e-8))` and
`Ldup(image) = sum_j u_j / number_of_valid_generated_rows`. No eligible row or
no valid row contributes differentiable zero. Normalize once per image, then
use the unchanged global image mean (effective batch32); never normalize by
eligible images or pair count. Every flagged later row contributes once even
if it overlaps many earlier rows. Fixed proposed coefficient: **lambda=1**.

Total objective: `L_Rweak(image) + lambda * Ldup(image)` at every update.
Gradients flow through current replay CE/logits to the existing language DoRA
parameters, not through discrete parsing/selection or generation. This is
sampled coordinate-confidence unlikelihood, not differentiable IoU, GT-only
supervision, NMS training, or a guarantee against all duplicate coordinates.

Refresh natural greedy train256 generations at continuation update0 and +8,
then replay the corresponding exact token IDs for the following eight updates.
Both arms use this schedule, with a shared initial capture at identical Rweak64
and separate +8 captures. Between refreshes it is lagged replay, not on-policy
sampling every update. The regularizer stays enabled throughout optimization.
No holdout or confirmation outcomes supply replay or coefficient selection.

## Single proposed GPU round

- Initialize both branches from
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-owner-focus-ablation/focus-v1/Rweak/checkpoint-000064/`;
  checkpoint ID `6b1f5b32dc1a5b7503ad81b691855ffcda3571f6d5433bc09838de07a88bf964`.
  Restore identical optimizer state and continue the schedule at updates64–79.
  Separate successor identity/output roots; do not weaken old resume-root or
  cold-consumer identity guards to pretend these are old owner-focus runs.
- Control lambda0, treatment lambda1; 16 extra global updates each, same
  train256, batch32, microbatch2, seed20260908, LR1e-5, AdamW betas(.9,.999),
  epsilon1e-8, no weight decay, grad clip1, FP32 SDPA and language DoRA surface.
  Bank ID `37ccd23a0ea1263e3da217cf126c5397ef9ee3863917631dac1d41deb6eeeb5c`.
- Natural refresh and final decode: greedy, repetition penalty1, cap3084,
  per-device batch4, two ranks per arm, no postprocessing. Final panel is the
  same `focus-v1/inputs-v1/holdout512.jsonl` (512 images / 3759 owners).
  The unused confirmation512 remains untouched; no second evaluation round.
- Request **at most four GPUs simultaneously, four hours elapsed and sixteen
  GPU-hours total**, including one bounded two-rank mechanics slice, refresh,
  training, final decode and reduction. These are safety ceilings, not timing
  promises. No coefficient/batch/seed sweep or automatic extension.
- At most three train256 refresh panels (768 image generations; the shared
  initial one may reuse exact admitted Rweak64 tokens if available), two final
  holdout512 panels (1024), plus <=8 smoke images. Each has at most3084 new
  tokens. Each arm has512 image presentations across16 updates, at most1024
  existing anchor/correction trajectory forwards and512 additional replay
  trajectories; activation recomputation is extra and must be reported.
  Keep zero-eligibility images in objective accounting; ranks must retain a
  fixed collective schedule rather than skipping synchronization by eligibility.
- Historical Rweak64 receipt: ~2587s for64 updates, peak allocated34.0GB,
  reserved62.6GB, host RSS13.6GB per recorded maximum. These are baseline
  evidence, not a new-loss capacity claim. Smoke measures added replay memory,
  per-update/refresh wall time, actual model forwards, RSS and checkpoint size.
  No new payload family; bound replay cache to one current train256 capture,
  release GPU logits between component backwards. Abort on OOM/nonfinite loss,
  broken identities/collectives/cold readback, or the first resource ceiling;
  preserve incomplete receipts and do not change scientific recipe to fit.

## Remaining real-boundary check and stop

CPU checks cover strict/cross-description eligibility, causal token offsets,
no-repeat zero, normalization, descent direction and finite extreme losses.
After authorization, implement the smallest successor trainer/consumer seam
and run one <=2-update two-rank mechanics slice, including a generated repeat
replay, lambda0 parity, frozen-parameter invariants, synchronized empty-repeat
behavior, checkpoint save/cold consume and raw-owner reduction. This boundary
is genuinely new and is **not** closed by the CPU module or old tests.

Stop this package when CPU results and this proposal suffice for a user ruling.
After a later approved round, stop at16 updates plus the fixed final readout
or the safety ceiling. Report dominance if present and trade-off/uncertainty if
not; no automatic promotion, confirmation, longer dose, or new search.
