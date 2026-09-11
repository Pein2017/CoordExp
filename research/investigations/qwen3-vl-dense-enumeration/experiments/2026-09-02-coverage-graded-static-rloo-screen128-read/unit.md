---
title: Coverage-Graded Static RLOO Image-Disjoint Screen-128 Read
description: One descriptive image-disjoint natural read of the unchanged train-positive coverage-graded adapter before pausing.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-09-02-coverage-graded-static-rloo-screen128-read
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-02
---

# Coverage-graded static RLOO screen128 read

Executed result: [results.md](results.md).  The unchanged adapter remains
train-positive but moves image-disjoint IoU50/60/80 owner counts by
`-6/-4/-2`, while matched geometry improves slightly at all three thresholds.

## Question and boundary

> Does the unchanged train-positive coverage-graded adapter also improve
> natural-greedy annotated-owner behavior on the registered image-disjoint
> 128-image screen relative to exact C?

This is one read only: no training, resampling, reward adjustment, dose sweep,
or checkpoint merge.  Compare the same Source, screen rows, prompt, tokenizer,
FP32/SDPA backend, RP1 greedy policy, 3,084-token cap, parser, and
category-consistent global one-to-one matcher at IoU50/60/80.  Report owner
gains/losses, matched geometry, prediction/duplicate/drop/cap behavior, and
natural ordering.

Any improvement is recorded.  A flat or negative screen result does not erase
the verified train gain; it only says this exact one-step adapter has not yet
produced a joint train-eval win.  Unmatched valid predictions remain unknown,
and this screen does not identify full precision, F1, or mAP under incomplete
owner annotation.

## Immutable candidate

- update receipt:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-coverage-graded-static-rloo-one-update/receipt.json`,
  SHA-256
  `d4304d09a9b9c87b126dfca94ea1db4a2aebb3f0fde6ab889851d3c8d52679f8`;
- candidate adapter fingerprint:
  `17063dca07e5b0d914b99eee6809b362955e535091c575338fb085668ee8c595`;
- train248 analysis:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-coverage-graded-static-rloo-train248-read/analysis.json`,
  SHA-256
  `ed7f7eca42b423aa96868d3b9d93410074ad2664ba09e33c8e78b67787849c53`.

Use
`configs/coordexp_infras/infer/qwen3_vl_2b_coverage_graded_static_rloo_screen128.yaml`
through the repository-native eight-GPU controller-worker path.  Stop after
the canonical read, comparison, and compact result record.
