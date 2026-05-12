---
title: Compact-Full RP1.10 Top-3 Rollout Union and Unlabeled-Object Prior
date: 2026-05-07
updated: 2026-05-12
status: completed-benchmark
topics: [stage1, compact-full, rp1.10, val200, et-rmp-ce, sft, unlabeled-positive, bootstrap-union]
tags: [benchmarks, val200, compact-full, repetition-penalty, et-rmp-ce, sft, unlabeled-positive, bbox-union]
summary: Records the top compact-full rp=1.10 val200 checkpoints, derives a bbox-only bootstrap-union estimate for unlabeled positive objects relative to COCO GT annotation count, and appends the 2026-05-12 A3/A4 prefix-rollin follow-up.
---

# Compact-Full RP1.10 Top-3 Rollout Union and Unlabeled-Object Prior (2026-05-07)

This note records the useful results from the compact detection `rp=1.10`
rollout/eval sequence and the follow-up unlabeled-positive prior derivation.

The main outputs are:

- the original `val200` compact-full `rp=1.10` top-3 checkpoint comparison
- a corrected bootstrap-union estimate of unique unlabeled objects
- an A3/A4 prefix-rollin follow-up that should be treated as unstable
  diagnostic evidence rather than a replacement for the original top-3 prior

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

## Original Top-3 RP1.10 Val200 Results

This table is the original 2026-05-07 top-3 set used for the bootstrap-union
unlabeled-object prior. The 2026-05-12 A3/A4 follow-up below is intentionally
kept separate because those checkpoints show more unstable duplicate/collapse
behavior and should not be folded into the prior without another gate.

| Rank | Run | AP | AP50 | AP75 | AR100 | F1@0.50 full micro | Guarded AP |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `ET-RMP/support2 tokenrows_v2 ckpt3664` | `0.4247` | `0.5752` | `0.4477` | `0.4720` | `0.6138` | `0.4116` |
| 2 | `random SFT bsz1/accum16 ckpt3664` | `0.3992` | `0.5577` | `0.4140` | `0.4714` | `0.4433` | `0.3827` |
| 3 | `random SFT ckpt2200` | `0.3702` | `0.5088` | `0.3915` | `0.4273` | `0.4257` | `0.3576` |

The `ET-RMP/support2` checkpoint remains the strongest current compact-full
`rp=1.10` point in this `val200` slice. The newer random-SFT checkpoint is
healthy and substantially better than capped/broken compact runs, but it remains
behind the multi-positive/support-reweighted checkpoint by about `+0.0255 AP`.

## A3/A4 Prefix-Rollin Follow-Up (2026-05-12)

This follow-up appends the newer A3/A4 checkpoints to the same first-200 COCO
`compact_full`, `rp=1.10`, `temperature=0.0`, `max_new_tokens=3084`, confidence
post-op, and duplicate-control evaluation surface.

The ablation labels used in the current discussion are:

| ID | Objective | Short read |
|---|---|---|
| A0 | random SFT hard CE | random-order one-hot baseline |
| A2 | multi-positive support+balance | older support/balance row; still best compact-full AP here |
| A3 | prefix-rollin support+balance | prefix-closed rollout objective without EOS-trust prior |
| A4 | A3 + EOS-trust prior | calibrated weak-EOS objective using the missing-label prior |

No equally scoped A1/support-only `val200` artifact was found in the existing
progress docs at the time of this update. Therefore this note records A3/A4
against the available A0/A2 comparators and does not invent an A1 result.

### Comparable Result Table

| Run | AP | AP50 | F1@0.50 | Recall | Precision | Pred | Invalid or bad-geom | Suppressed | Guard AP | Guard F1@0.50 | Read |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A0 random SFT | `0.3992` | `0.5577` | `0.4433` | `0.5499` | `0.3714` | `2261` | `~647-652` | `1148` | `0.3827` | `0.5479` | High recall pressure, but heavy duplicate/bad-geometry burden. |
| A2 support+balance | `0.4247` | `0.5752` | `0.6138` | `0.5409` | `0.7094` | `1140` | `0` | `232` | `0.4116` | `0.6004` | Best compact-full AP and cleanest precision/validity tradeoff so far. |
| A3 prefix-rollin | `0.3980` | `0.5444` | `0.5597` | `0.4917` | `0.6496` | `1162` | `11` | `301` | `0.3854` | `0.5485` | Cleaner than A4, but lower AP/F1 than A2 and below random-SFT AP. |
| A4 EOS-trust | `0.4001` | `0.5502` | `0.5612` | `0.5173` | `0.6133` | `1300` | `72` | `401` | `0.3915` | `0.5638` | Opens recall relative to A3, but also opens more duplicate/collapse tail. |

A0's invalid/bad-geometry counter is approximate here because the scorer and
artifact-entry counters use slightly different event accounting. The only
decision-relevant point is that A0 is orders of magnitude noisier than A2/A3/A4
on this field.

Training-side metrics do not fully predict free-rollout reliability:

| Run | Final ckpt | Trainer best ckpt | Final eval CE | Coord top1 | Type mass | EOS trust |
|---|---:|---:|---:|---:|---:|---:|
| A3 | `3664` | `3664` | `1.5964` | `0.1315` | `0.9705` | `1.0000` |
| A4 | `3664` | `3600` | `1.5710` | `0.1318` | `0.9727` | `0.3113` |

A4's final training surface is slightly better than A3's, and the EOS-trust
prior does reduce terminal pressure in teacher-forced probes, but its free
rollout is less predictable. It has higher recall and guarded F1 than A3, while
also increasing invalid/border/collapse events.

### Teacher-Forced Boundary Probe

Artifact:

```text
temp/a4_rp110_tf_probe_and_manual_review_20260512/tf_probe_high_ge10_32_summary/summary.md
```

Scope:

```text
high-density subset of first val200, 32 images with GT_count >= 10
GT prefixes K=0,1,3,5,10,N
generated-prefix boundary from each run's own rp=1.10 rollout
```

Core boundary margins:

| Run | Generated-boundary n | Generated sep-minus-EOS mean | Generated sep <= 0 | Generated entry mean | GT sep mean | GT sep <= 0 | True-end EOS prob |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 random SFT | `9` | `6.215` | `0.556` | `17.340` | `9.622` | `0.000` | `0.435` |
| A2 support+balance | `27` | `-1.306` | `1.000` | `15.162` | `8.525` | `0.056` | `0.880` |
| A3 prefix-rollin | `22` | `-1.000` | `1.000` | `15.099` | `7.087` | `0.048` | `0.858` |
| A4 EOS-trust | `18` | `-0.667` | `1.000` | `16.066` | `7.387` | `0.032` | `0.692` |

Read:

- On GT prefixes, A2/A3/A4 still know how to continue and start a valid object
  row.
- On their own generated prefixes, A2/A3/A4 still locally prefer EOS at the
  free boundary in this high-density probe (`sep <= 0` is `1.0`).
- A4 weakens EOS relative to A3 (`true-end EOS prob` `0.858 -> 0.692` and
  generated margin `-1.000 -> -0.667`), but it does not reliably flip the
  generated free-boundary decision into continuation.
- Once continuation is forced, object-entry confidence remains healthy. The
  hard problem is the free-boundary and subsequent coordinate stability, not
  simply the schema entry token.

This is exposure/off-policy boundary evidence. It should not be used as a
single production diagnostic or a checkpoint selector by itself.

### Manual Review Read

Manual audit artifact:

```text
temp/a4_rp110_tf_probe_and_manual_review_20260512/manual_audit_a4_vs_a3_image20_v2_pixelgt/manual_audit_a4_vs_a3_image20_v2_pixelgt.csv
temp/a4_rp110_tf_probe_and_manual_review_20260512/manual_audit_a4_vs_a3_image20_v2_pixelgt/manifest_no_gt.json
temp/a4_rp110_tf_probe_and_manual_review_20260512/manual_audit_a4_vs_a3_image20_v2_pixelgt/manual_audit_labels.jsonl
```

Important visualization caveat:

```text
The earlier GT-green overlay had a pixel-vs-norm1000 coordinate rendering bug.
Use the v2 pixel-GT/no-GT overlays for review; do not use the old green-GT view
as evidence.
```

Qualitative synthesis from the 20 reviewed images:

- A4 often emits useful extra objects that look like plausible unlabeled
  positives, especially in crowded scenes and dense table/person/vehicle cases.
- A4 also frequently creates purple/red border or top-left coordinate collapse,
  plus duplicate bursts in dense object regions.
- Magenta duplicate-guarded candidates are mixed: many are true duplicates, but
  some are separate or near-valid crowded instances, so duplicate control is
  necessary but can over-suppress.
- A3/blue is often cleaner and tighter; A4/yellow/red is often higher recall.
  The two behaviors co-occur, so the 20-image audit does not support a simple
  "A4 is better" or "A3 is better" conclusion.
- The practical read is that A4 opens continuation/recall but lacks a reliable
  acceptance gate. Current A3/A4 checkpoints are more unpredictable than A2 and
  should be treated as research probes.

### Follow-Up Decision

Do not fold A3/A4 into the original top-3 bootstrap-union unlabeled prior yet.
The original prior remains based on the previously selected top-3 `rp=1.10`
compact-full rollout surfaces. A3/A4 are valuable diagnostic evidence about
prefix-rollin and EOS trust, but their duplicate/collapse behavior makes them a
poor source for a cleaner unlabeled-object prior without an additional
confidence and duplicate-risk gate.

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

## Repo Artifacts

Keep this result in repo-linked evidence rather than a separate management
surface.

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
