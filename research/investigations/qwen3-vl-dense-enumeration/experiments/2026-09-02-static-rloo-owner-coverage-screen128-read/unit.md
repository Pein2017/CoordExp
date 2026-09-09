---
title: Static RLOO Image-Disjoint Screen-128 Read
description: Evaluate the unchanged train-positive static-RLOO adapter once on the registered image-disjoint 128-image screen.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-09-02-static-rloo-owner-coverage-screen128-read
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-02
---

# Static RLOO image-disjoint screen-128 read

## Question and decision

> Does the exact static-RLOO adapter that improved train-248 transfer in the
> same direction to the frozen image-disjoint 128-image screen relative to C?

This is one read of an already selected candidate, not checkpoint selection or
additional learning.  Compare C and the unchanged candidate under identical
screen rows, images, prompt, tokenizer, FP32/SDPA backend, RP1 greedy decode,
3,084-token cap, parser, and category-consistent global one-to-one matching.

Report the complete IoU50/60/80 vector, gains and losses, common-owner geometry,
prediction count, duplicates, invalid/dropped rows, EOS/caps, length, and
natural order.  There is no perfection gate: positive, negative, or mixed
image-disjoint movement is recorded as observed.  Summed owner-threshold hits
across IoU50/60/80 are a compact descriptive aggregate, not mAP.  Unmatched
valid predictions remain unknown, so this unit cannot identify full precision,
F1, or mAP.

Stop this exact adapter after the single read.  A positive result may motivate
a separately frozen graded-reward training successor; a negative result does
not erase the verified train-side gain or reject policy gradient generally.

## Execution outcome

The registered read completed once through the native eight-GPU path.  The
artifact sets validate, and the exact result is recorded in
[results.md](results.md).  The candidate is count-negative on the screen at
IoU50/60/80 (`-6/-4/-2`) while its matched localization geometry improves
slightly at all three thresholds.  One candidate row reaches the length cap;
an exact 127-row sensitivity shows that this cap is separate structural debt,
not the cause of the negative owner-count result.

## Immutable packet

- image-disjoint screen input:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-01-annotated-owner-direct-c-d0-pilot/materialization-v1/screen-dev.jsonl`,
  SHA-256
  `10ae2a3b87e03d395d83469e4b10dceb6ee4b5509ca14b8254c85a20026320b8`;
- C screen rows:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-01-annotated-owner-direct-c-d0-pilot/infer/c/qwen3-vl-2b-annotated-owner-direct-c-screen128/gt_vs_pred.jsonl`,
  SHA-256
  `7719a4aea9f9998f7bd0074c3bb43fc89ff2c5eeed41c74ed1e8b29619bcb9fc`;
- C IoU50/60/80 owner counts: `630 / 598 / 461` over 891 annotated
  owners;
- candidate adapter fingerprint:
  `9cbfde447f9164875180cd8bbe6122d329ceb49d732882f66cb9b03e7caadc1d`;
- candidate cold-read receipt SHA-256:
  `b5d8ec481289f2fedf0e39eb3ee1d5e3a1380b58fa3bc28dc02949e8aad29182`;
- train-248 analysis-v2 SHA-256:
  `1f519f55d9dbcc93fc63f2e5faffec1c8fec043c731c98ab4478553191c04219`;
- inference config:
  `configs/coordexp_swift/infer/qwen3_vl_2b_static_rloo_owner_coverage_screen128.yaml`,
  SHA-256
  `5b5d754d63dae7f201115f69b842ff3304685506af20e651496811f89ce8d6e6`.

Use the existing native eight-GPU controller-worker path and canonical merge,
then the existing clean-rollout owner comparator at IoU50/60/80.  No new
runner, sharder, merger, optimizer, or checkpoint is needed.
