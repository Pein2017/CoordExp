---
title: Static RLOO Train-248 Breadth Read
description: Read the completed eight-image static-RLOO adapter across the registered 248-image training cohort before choosing another optimizer.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: configuration_only
unit_id: 2026-09-02-static-rloo-owner-coverage-train248-read
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-02
---

# Static RLOO train-248 breadth read

Executed result: [results.md](results.md).  The unchanged adapter is positive
at all three reported thresholds (`+1/+2/+3` IoU50/60/80 owners), with mixed
owner exchange and four of six aggregate threshold-owner gains occurring on
the 240 images outside its optimization panel.

## Question and boundary

> Does the selected-panel geometry movement from the completed one-update
> static RLOO adapter remain positive when cold natural greedy decoding is
> widened from its eight optimization images to the registered 248-image
> training cohort?

This is the shortest train-first discriminator after the fixed unit returned
unchanged IoU50 owner count but `38 -> 40` IoU80 owners.  It performs no
training and changes no adapter.  It does not claim held-out generalization.

Compare frozen C with the exact saved candidate under the same 248 input rows,
images, prompt, tokenizer, FP32/SDPA backend, RP1 greedy policy, 3,084-token
cap, parser, and category-consistent global one-to-one matcher.  Report
IoU50/60/80 owner gains and losses, matched-IoU geometry, prediction counts,
duplicates, unmatched valid predictions, drops, caps/EOS, lengths, and natural
ordering separately.  Unmatched valid predictions remain unknown.

This read has no perfection gate.  Any positive aggregate train movement is
recorded, mixed movements remain mixed, and an image-disjoint read is deferred
until the breadth result is known.  Stop only this exact adapter if the wider
train vector is uniformly nonpositive or mechanically invalid; do not infer
that policy gradient or DoRA is generally ineffective.

## Immutable inputs

- C rows:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-c-anchored-owner-mechanism-audit/full/infer/c/qwen3-vl-2b-c-anchored-owner-audit-c-train248/gt_vs_pred.jsonl`,
  SHA-256
  `9f8aa4f478883468ace45daa4a4de9b90d055938057ca8e2a1add83eff2077ed`;
- candidate update receipt:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-static-rloo-owner-coverage-one-update/receipt.json`,
  SHA-256
  `fd6c4004ab3781c32dc16beb77d279efd6cb73cee0c6309ec1662d96c4a77659`;
- candidate cold receipt: SHA-256
  `b5d8ec481289f2fedf0e39eb3ee1d5e3a1380b58fa3bc28dc02949e8aad29182`;
- candidate adapter fingerprint:
  `9cbfde447f9164875180cd8bbe6122d329ceb49d732882f66cb9b03e7caadc1d`;
- input JSONL:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-01-annotated-owner-direct-c-d0-pilot/materialization-v1/train-common-base.jsonl`,
  SHA-256
  `86d34cc2efbce9814847dd12fc12cab2f46d04168ce39905bfd62a93ced783fd`;
- inference config:
  `configs/coordexp_infras/infer/qwen3_vl_2b_static_rloo_owner_coverage_train248.yaml`,
  SHA-256
  `7e5f50ddca46a043a45460e35022ecf65d2ef8625590296dcfd312936e7b9660`.

Use the repository's native controller-worker inference path with eight visible
GPUs.  It owns deterministic sharding, rank-local complete artifact families,
and canonical ordered merge; no experiment-specific sharder or merger is
added.  Compare the merged candidate rows to C using the existing
`scripts/research/compare_clean_rollout_owner_coverage.py` at IoU thresholds
`0.50`, `0.60`, and `0.80`.
