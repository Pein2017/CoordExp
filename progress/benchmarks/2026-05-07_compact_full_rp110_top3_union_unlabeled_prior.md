---
title: Compact-Full RP1.10 Top-3 Rollout Union and Unlabeled-Object Prior
date: 2026-05-07
status: completed-benchmark
topics: [stage1, compact-full, rp1.10, val200, et-rmp-ce, sft, unlabeled-positive, bootstrap-union]
tags: [benchmarks, val200, compact-full, repetition-penalty, et-rmp-ce, sft, unlabeled-positive, bbox-union]
summary: Records the top compact-full rp=1.10 val200 checkpoints and derives a bbox-only bootstrap-union estimate for unlabeled positive objects relative to COCO GT annotation count.
---

# Compact-Full RP1.10 Top-3 Rollout Union and Unlabeled-Object Prior (2026-05-07)

This note records the useful results from the compact detection `rp=1.10`
rollout/eval sequence and the follow-up unlabeled-positive prior derivation.

The main outputs are:

- a ranked `val200` compact-full `rp=1.10` top-3 checkpoint comparison
- a corrected bootstrap-union estimate of unique unlabeled objects
- a Notion-import recommendation for research memory

This is a `progress/benchmarks/` note because it is measured evidence with a
fixed dated scope. It is not a stable current workflow contract.

## Scope

Shared benchmark contract:

- dataset:
  `public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`
- slice:
  first 200 validation examples (`val200`)
- sequence format:
  `compact_full`
- prompt:
  `prompt_variant=coco_80`
- bbox format:
  `xyxy`
- coordinate surface:
  coord tokens
- decoding:
  `temperature=0.0`, `top_p=0.9`, `repetition_penalty=1.10`,
  `max_new_tokens=3084`
- scoring:
  confidence post-op with `bbox_logprob_confidence_exp`
- primary metric:
  raw scored `eval/metrics.json` `bbox_AP`
- guarded metric:
  duplicate-control guarded `eval/metrics_guarded.json`

Guarded metrics are reported as an additive precision-control view, not as the
primary AP headline.

## Top-3 RP1.10 Val200 Results

| Rank | Run | AP | AP50 | AP75 | AR100 | F1@0.50 full micro | Guarded AP |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `ET-RMP/support2 tokenrows_v2 ckpt3664` | `0.4247` | `0.5752` | `0.4477` | `0.4720` | `0.6138` | `0.4116` |
| 2 | `random SFT bsz1/accum16 ckpt3664` | `0.3992` | `0.5577` | `0.4140` | `0.4714` | `0.4433` | `0.3827` |
| 3 | `random SFT ckpt2200` | `0.3702` | `0.5088` | `0.3915` | `0.4273` | `0.4257` | `0.3576` |

The `ET-RMP/support2` checkpoint remains the strongest current compact-full
`rp=1.10` point in this `val200` slice. The newer random-SFT checkpoint is
healthy and substantially better than capped/broken compact runs, but it remains
behind the multi-positive/support-reweighted checkpoint by about `+0.0255 AP`.

## Top-3 Artifact Paths

Rank 1:

- run:
  [/data/CoordExp/output_remote/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_max3084_chatfix_4gpu](/data/CoordExp/output_remote/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_max3084_chatfix_4gpu)
- checkpoint:
  `/data/CoordExp/output_remote/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ce_support2_bsz16_4epoch_tokenrows_v2/compact-full-et-rmp-ce-support2-bsz16-4epoch-tokenrows-v2/v0-20260504-071356/checkpoint-3664`

Rank 2:

- run:
  [/data/CoordExp/output_remote/infer/recursive_detection_ce_latest/compact_full_random_sft_bsz1_accum16_ckpt3664_val200_bsz4_temp0_rep1p10_max3084_chatfix_8gpu](/data/CoordExp/output_remote/infer/recursive_detection_ce_latest/compact_full_random_sft_bsz1_accum16_ckpt3664_val200_bsz4_temp0_rep1p10_max3084_chatfix_8gpu)
- checkpoint:
  `/data/CoordExp/output_remote/stage1_2b/recursive_detection_ce_latest/compact_full_random_sft_bsz1_accum16_4epoch_tokenrows_v4_chatfix_max12k/compact-full-random-sft-bsz1-accum16-4epoch-tokenrows-v4-chatfix-max12k/v0-20260506-021259/checkpoint-3664`

Rank 3:

- run:
  [/data/CoordExp/output_remote/infer/recursive_detection_ce_latest/compact_full_random_sft_ckpt2200_val200_bsz4_temp0_rep1p10_max3084_chatfix_4gpu](/data/CoordExp/output_remote/infer/recursive_detection_ce_latest/compact_full_random_sft_ckpt2200_val200_bsz4_temp0_rep1p10_max3084_chatfix_4gpu)
- checkpoint:
  `/data/CoordExp/output_remote/stage1_2b/recursive_detection_ce_latest/compact_full_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-random-sft-bsz16-4epoch-tokenrows-v2/v0-20260505-044157/checkpoint-2200`

Compact machine-readable summary:

- [2026-05-07_compact_full_rp110_top3_union_summary.json](artifacts/2026-05-07_compact_full_rp110_top3_union_summary.json)

## Corrected Union Definition

The final requested definition is bootstrap-union style:

```text
For each image:
1. Collect predictions from the top-3 rollout sequences.
2. Deduplicate predictions by bbox overlap only.
3. Build the union of unique predicted objects.
4. Match that prediction union against GT by bbox overlap only.
5. unlabeled_count = |unique_prediction_union| - |matched_GT_objects|
```

Important: class/description text is intentionally ignored for object identity.
Two predictions are treated as the same object when their bboxes overlap enough.

Operational thresholds used here:

- prediction union deduplication:
  bbox IoU `>= 0.50`
- union-vs-GT subtraction:
  bbox IoU `>= 0.50`
- source prediction surface:
  `gt_vs_pred_scored_guarded.jsonl`

The guarded prediction surface is used to remove ordinary within-run duplicate
bursts before cross-run union. Heavy collapse still appears in a few run-image
cells, so the primary relation below drops one run-image contribution when:

```text
guarded_pred_count > max(20, 2 * gt_count)
```

This removed `4` run-image contributions. The unfiltered result is retained as
a sensitivity bound.

## Unlabeled-vs-GT Relation

Primary bbox-only union with heavy-burst run-image filtering:

```text
GT total = 1444
unique prediction union total = 1374
matched GT clusters = 825
unlabeled union total = 549
unlabeled / GT = 0.3802
```

Linear fit:

```text
U_union ~= max(0, -0.35 + 0.43 * G)
R^2 = 0.5625
Pearson r = 0.7500
```

Through-origin fit:

```text
U_union ~= 0.40 * G
R^2 = 0.5588
```

Recommended compact prior:

```text
unlabeled_count ~= 0.38 to 0.40 * gt_annotation_count
effective_object_count ~= 1.38 to 1.40 * gt_annotation_count
```

This replaces the earlier median-per-run FP proxy. That intermediate estimate
was useful as a conservative diagnostic, but it did not implement the intended
bootstrap union of unique objects across multiple rollouts.

## Bucketed Bbox-Union Relation

Primary heavy-burst-filtered bbox-only union:

| GT Count Bucket | Images | Mean GT | Mean Unlabeled Union | Median Unlabeled Union | U / G |
|---:|---:|---:|---:|---:|---:|
| `1` | 19 | 1.00 | 0.37 | 0 | 0.368 |
| `2-3` | 65 | 2.34 | 0.48 | 0 | 0.204 |
| `4-5` | 30 | 4.53 | 0.73 | 0.0 | 0.162 |
| `6-10` | 37 | 7.68 | 3.51 | 3 | 0.458 |
| `11-20` | 39 | 14.67 | 6.51 | 5 | 0.444 |
| `21+` | 10 | 28.10 | 10.50 | 9.0 | 0.374 |

Interpretation:

- low-count images are noisy and often have zero extra union objects
- the relation stabilizes more visibly in the `6+` GT-count region
- crowded images support a rough `0.38` to `0.40` unlabeled/GT prior after
  removing collapse contributions

## Sensitivity Bound

If heavy burst/collapse run-image contributions are not removed:

```text
GT total = 1444
unique prediction union total = 1509
matched GT clusters = 825
unlabeled union total = 684
unlabeled / GT = 0.4737
U_union ~= 0.46 * G
```

This is a useful upper sensitivity bound, but not the recommended prior. The
unfiltered relation is visibly dominated by a few collapsed run-image cases.

## Research Interpretation

The most useful read is not that every false positive is truly an unlabeled
positive. The more careful interpretation is:

- high-scoring, cross-rollout prediction diversity provides a proxy for object
  support outside the COCO label set
- taking the bbox-only union across several strong checkpoints approximates a
  bootstrap proposal set
- subtracting GT from that union gives an empirical prior on extra object
  density
- on this `val200` compact-full slice, the prior is roughly:

```text
extra plausible objects ~= 0.4 * COCO GT annotations
```

This should be treated as a research prior, not as a dataset truth claim.

## Notion Import Recommendation

This result is suitable for Notion, but it should be imported as research
memory, not as an executable contract.

Recommended Notion placement:

- Research Unit:
  `Compact-full rp1.10 bootstrap-union unlabeled-object prior`
- Claims Ledger entry:
  `On val200 top-3 rp=1.10 compact-full rollouts, bbox-only prediction union suggests U ~= 0.38 to 0.40 * GT after burst filtering.`
- optional Decision Log:
  only if we decide to use the `0.4 * GT` prior in a training objective,
  dataset expansion rule, or evaluation correction.

Recommended claim status:

- `Supported` for the scoped empirical claim:
  `val200`, top-3 `rp=1.10` compact-full rollouts, bbox-only IoU `0.50`,
  guarded prediction surface, burst-filtered.
- `Untested` for any generalization claim to full-val, other checkpoints,
  other prompts, other RP settings, or true unlabeled object count.

Do not paste the full per-image JSON into Notion. Link the repo note and exact
artifact paths instead:

- this note:
  [2026-05-07_compact_full_rp110_top3_union_unlabeled_prior.md](/data/CoordExp/progress/benchmarks/2026-05-07_compact_full_rp110_top3_union_unlabeled_prior.md)
- compact summary:
  [2026-05-07_compact_full_rp110_top3_union_summary.json](/data/CoordExp/progress/benchmarks/artifacts/2026-05-07_compact_full_rp110_top3_union_summary.json)
- per-image bbox-union artifact:
  [2026-05-07_compact_full_rp110_top3_bbox_union_per_image.json](/data/CoordExp/progress/benchmarks/artifacts/2026-05-07_compact_full_rp110_top3_bbox_union_per_image.json)

## Follow-Up Questions

Good next checks before promoting this into a stable method:

- rerun the union estimate on full-val or a larger fixed slice
- sweep bbox-union IoU thresholds such as `0.40`, `0.50`, and `0.60`
- compare `rp=1.00`, `1.05`, and `1.10`
- compare compact-full against canonical JSON-style Stage-1 output
- inspect top union-only images visually before using the prior for training
