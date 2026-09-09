---
title: Earlier weak-correction checkpoints and prospective confirmation
description: At most two small rounds using existing checkpoints and native evaluation.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-09-07-coco-weak-correction-dose
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-07
---

# Weak-correction dose recovery

## Final disposition

**Complete after round1; no eligible early dose; round2 not launched.**
[Results](results.md) own the readout. On the same512 dose-selection panel,
Rweak16 changes Source coverage by +15/+10/-17 at IoU50/60/80, with duplicate
candidates21→27 and caps11→15. Rweak32 changes coverage by +25/+15/-15, with
duplicates21→343 and caps11→13. Both fail the fixed joint criterion, so the
conditional confirmation branch closes without GPU work. The result is a
bounded negative for the two tested earlier stops, not all possible doses.

Both new512-image decodes completed in49m30s; the lead ran the saved analysis,
verified the completed input/step/settings boundaries and paired set counts,
and accepted the no-confirmation decision. No new training, extra checkpoint
search, repeated hash audit or production-hardening work was performed.
The following sections preserve the protocol and execution history; this
disposition governs the current state. STOP reached; no further run authorized.

User authorized one or two autonomous rounds and prioritizes useful research
over hash/production ceremony. This is a new experiment, not an extension of
the [completed ablation](../2026-09-07-coco-owner-focus-ablation/results.md).
Keep ordinary input/checkpoint/metric correctness; add no framework, broad
review, repeated hash audit, new training, new model architecture or decoder.

## Question and minimum contrast

Does ending Rweak training earlier retain useful annotated-owner recovery
without the late-checkpoint repetition cost? The alternative is that repetition
already accompanies the useful update, not merely excess exposure. Existing
holdout evidence shows 249 of Rweak64's 276 strict duplicate candidates on one
image (566923); nine images are affected versus Source's eight. Report both
concentration and totals, but never remove that image to improve the score.
The scored Rweak64 output on image566923 contains 256 predictions, including
246 exact copies of the same full-frame person box; Source has 14 predictions
with no exact repeated box there. This is a concrete repetition failure, not
just an aggregate unmatched-annotation count.

## Round 1: two saved checkpoints, no retraining

- Evaluate existing Rweak update16 and update32 from `focus-v1/Rweak/` on the
  previous holdout512. Reuse the exact Source and Rweak64 outputs on these rows.
  This 512-image set now owns dose selection, NOT an independent final test.
- Same native greedy FP32 SDPA, RP1.0, cap3084, per-device batch4 and evaluator.
  Use four GPUs per new candidate if native global four-image decode groups
  remain unchanged by rank count; verify that cheap scheduling property first.
  Otherwise retain the already qualified two-rank topology. No GPU benchmark
  or extra qualification matrix. Two candidates may run concurrently.
- Report owner50/60/80 gains/losses against Source and 64, strict duplicates,
  duplicate-affected images/concentration, invalid/drop/cap and output length.
- A dose is eligible for confirmation if IoU50 exceeds Source, IoU60/80 do not
  decrease, and strict duplicate, invalid and capped-image totals do not exceed
  Source. Among eligible 16/32 doses, choose greatest IoU50; ties choose fewer
  updates. No search of the other saved checkpoints and no tuning the rule.

## Round 2: conditional fresh confirmation only

If neither early dose is eligible, STOP after round1 with a bounded negative.
Otherwise evaluate only the chosen checkpoint and original Source on another
512 existing COCO-val images. Freeze this panel outcome-blind before reading
round1 results: seed20260909, exclude the previous512 and documented prior local
train/selection IDs. Same decoder and batches, two independent four-GPU jobs
if the unchanged-group property holds. Source historical COCO-val forward
evaluation exposure remains disclosed; no claim of never-seen data.

The same coverage/debt eligibility rule owns confirmation. Report image-paired
uncertainty as diagnostic, not a claim of a statistically established population
benefit from a single seed. Failure closes the route; no rescue or third round.
Train256 is not decoded again: its known Rweak64 behavior is not the current
uncertainty. Retain all evaluation images, including repetitive/capped ones.

## Execution and stop

Reuse current cold-entry and owner-matching code. Keep one concise results JSON
and this unit/results record, not duplicate planning artifacts or audit packets.
Existing cheap loader validation may remain; do not add checksum workflows.
One package owner may prepare/launch the paired evaluations and reduce the
saved outputs; lead owns the round2 decision and final scientific acceptance.
GPU inventory at start: eight free 80GB devices. Bound each decode panel at the
existing 12-hour safety ceiling, not a forecast; preserve failures/no retries
with altered semantics. Use durable launchers and event-driven waits.

Output root:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-07-coco-weak-correction-dose/dose-v1/`.
Maximum new decode work: 2x512 images in round1, optionally 2x512 in round2.
STOP after that fixed budget: no new training, extra seed/dose, promotion,
commit, push, archive or shared-skill edits. Plain path/step/count receipts and
one proportionate result check suffice for this research decision.

## Round1 execution

The saved16/32 pair launched at `2026-09-07T17:31:53Z` in tmux
`coco-weak-dose-round1`, driver PID2162300, on GPU0-3 and GPU4-7 respectively.
Lead inspected the small launch/analysis helpers, replayed eligibility boundary
checks and verified that all128 four-image decode groups are unchanged by the
2→4 rank placement. No inference or training code was modified.

The optional confirmation panel was frozen before round1 readout: seed20260909,
512 images / 3,886 owners, disjoint from the preceding512 and all documented
prior exclusions. Its plain manifest and row IDs live in
`dose-v1/confirmation-input/`; no extra hash workflow was added. The lead ran
`dose.py analyze` after both exits were0; it selected no eligible checkpoint.
The confirmation panel remains unexecuted, not a missing or negative result.
