---
title: Raw-Text vs Coord-Token Repetition-Penalty Sweep
date: 2026-04-23
status: completed-benchmark
topics: [stage1, val200, repetition-penalty, raw-text, coord-token, detection, checkpoint-comparison]
tags: [benchmarks, val200, repetition-penalty, raw-text, coord-token, bbox, detection]
summary: Matched val200 detection benchmark comparing the raw-text adapter checkpoint against the coord-token checkpoint across repetition_penalty 1.00, 1.05, and 1.10, with a scorer repair that restored valid raw-text confidence post-op on fresh runs.
---

# Raw-Text vs Coord-Token Repetition-Penalty Sweep (2026-04-23)

This note records the `val200` detection benchmark comparing two checkpoints under
three `repetition_penalty` settings:

- raw-text adapter baseline:
  `output/stage1_2b/coco_bbox_max60-coco80-desc_first-1024-lvis_proxy-raw_text_xyxy-pure_ce/epoch_4-raw_text_xyxy-pure_ce-coco80-desc_first-1024-lvis_proxy-from-base-2B/v1-20260417-084341/checkpoint-552`
- coord-token full checkpoint:
  `output_remote/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332`

The benchmark question was:

- are raw numerical text boxes sufficient relative to specialized coord tokens?
- how sensitive are the two surfaces to `repetition_penalty` at `1.00`, `1.05`,
  and `1.10`?

## Scope

Shared benchmark contract:

- dataset:
  `public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/val.coord.jsonl`
- slice:
  first 200 validation examples (`val200`)
- prompt controls:
  `prompt_variant=coco_80`, `object_field_order=desc_first`,
  `object_ordering=sorted`, `bbox_format=xyxy`
- decoding:
  `temperature=0.0`, `top_p=0.9`, `max_new_tokens=3084`, `batch_size=1`,
  `seed=42`
- primary metric view:
  `coco_real`

Important provenance notes:

- raw-text `1.05` and `1.10` were rescored after repairing the confidence
  post-op path in [src/eval/confidence_postop.py](/data/CoordExp/src/eval/confidence_postop.py).
- raw-text `1.00` had an incomplete first attempt
  (`coco1024_val200_compare_rawtext_ckpt552_rp1p00_20260423T101256Z`), so the
  valid benchmark cell comes from the later successful run
  `coco1024_val200_compare_rawtext_ckpt552_rp1p00_20260423T101630Z`.
- coord-token `1.00` used an 8-GPU run, raw-text `1.00` also used an 8-GPU run,
  while raw-text `1.05` and `1.10` used 4-GPU runs. Accuracy is directly
  comparable across all cells; throughput is directly comparable within a given
  launch shape and should be read with care across different GPU counts.

## Main Results

| Model | RP | AP | AP50 | AP75 | F1@0.50 loc | Infer wall | Per-sample | Throughput | Kept / total preds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw-text | 1.00 | 0.2756 | 0.3534 | 0.2983 | 0.5522 | `1690.91s` | `8.455s` | `0.1183` samp/s | `960 / 960` |
| Raw-text | 1.05 | 0.3500 | 0.4497 | 0.3790 | 0.6060 | `2105.24s` | `10.526s` | `0.0950` samp/s | `1181 / 1181` |
| Raw-text | 1.10 | 0.3782 | 0.5024 | 0.4042 | 0.6211 | `2098.70s` | `10.494s` | `0.0953` samp/s | `1195 / 1195` |
| Coord-token | 1.00 | 0.4532 | 0.6177 | 0.4755 | 0.6626 | `771.02s` | `3.855s` | `0.2594` samp/s | `1518 / 1518` |
| Coord-token | 1.05 | 0.4584 | 0.6307 | 0.4770 | 0.7056 | `847.60s` | `4.238s` | `0.2360` samp/s | `1315 / 1315` |
| Coord-token | 1.10 | 0.4419 | 0.6064 | 0.4579 | 0.7002 | `828.44s` | `4.142s` | `0.2414` samp/s | `1285 / 1285` |

## Readout

Five conclusions are safe.

1. Coord tokens beat raw text at every tested repetition penalty on `val200`.
2. Coord-token accuracy peaks at `1.05`.
3. Raw-text accuracy improves monotonically across the tested penalties and is
   best at `1.10`.
4. Coord-token decoding is materially faster than raw-text decoding in every
   valid run we collected.
5. Raw-text is workable after scorer repair, but it remains both less accurate
   and more operationally fragile than the coord-token surface.

The most important accuracy comparisons:

- best raw-text cell:
  `1.10` with `AP=0.3782`
- best coord-token cell:
  `1.05` with `AP=0.4584`
- coord-token advantage at each penalty:
  - `1.00`: `+0.1776 AP`
  - `1.05`: `+0.1084 AP`
  - `1.10`: `+0.0637 AP`

Interpretation:

- raw numerical text is sufficient to produce valid and measurable detections
  once the scorer follows the numeric-text path correctly.
- specialized coord tokens still provide a clear measurable gain in both
  localization quality and runtime efficiency.
- the raw-text surface appears more sensitive to decode/scoring contract
  assumptions, while the coord-token surface is the more stable benchmark lane.

## Repetition-Penalty Trends

Coord-token family:

- `1.00 -> 1.05`:
  AP improves from `0.4532` to `0.4584`
- `1.05 -> 1.10`:
  AP drops to `0.4419`
- practical read:
  `1.05` is the best accuracy point for this checkpoint family

Raw-text family:

- `1.00 -> 1.05`:
  AP improves from `0.2756` to `0.3500`
- `1.05 -> 1.10`:
  AP improves again to `0.3782`
- practical read:
  within this sweep, `1.10` is the strongest raw-text setting

## Scorer Repair

The raw-text fresh runs originally looked broken because confidence post-op
discarded every predicted object with `pred_alignment_mismatch`.

The repaired behavior is now:

- raw-text `1.05`:
  `1181 / 1181` predictions kept
- raw-text `1.10`:
  `1195 / 1195` predictions kept

Root-cause summary:

- the raw-text records were being scored as if they were on the coord-token
  geometry surface
- once the scorer used the numeric-text span-matching path instead, the same
  traces became valid and measurable

This matters because the earlier apparent raw-text `1.10` “collapse” was a
scoring-contract bug, not a true inference collapse.

## Artifact Paths

Raw-text:

- `1.00`:
  [proxy_eval_bundle_summary.json](/data/CoordExp/output/infer/coco1024_val200_compare_rawtext_ckpt552_rp1p00_20260423T101630Z/proxy_eval_bundle_summary.json)
  [timing_summary.json](/data/CoordExp/output/infer/coco1024_val200_compare_rawtext_ckpt552_rp1p00_20260423T101630Z/timing_summary.json)
- `1.05`:
  [proxy_eval_bundle_summary.json](/data/CoordExp/output/infer/coco1024_val200_compare_rawtext_ckpt552_20260423T080326Z/proxy_eval_bundle_summary.json)
  [timing_summary.json](/data/CoordExp/output/infer/coco1024_val200_compare_rawtext_ckpt552_20260423T080326Z/timing_summary.json)
- `1.10`:
  [proxy_eval_bundle_summary.json](/data/CoordExp/output/infer/coco1024_val200_compare_rawtext_ckpt552_rp1p10_20260423T091701Z/proxy_eval_bundle_summary.json)
  [timing_summary.json](/data/CoordExp/output/infer/coco1024_val200_compare_rawtext_ckpt552_rp1p10_20260423T091701Z/timing_summary.json)

Coord-token:

- `1.00`:
  [proxy_eval_bundle_summary.json](/data/CoordExp/output/infer/coco1024_val200_compare_coordtoken_ckpt1332_rp1p00_20260423T095845Z/proxy_eval_bundle_summary.json)
  [timing_summary.json](/data/CoordExp/output/infer/coco1024_val200_compare_coordtoken_ckpt1332_rp1p00_20260423T095845Z/timing_summary.json)
- `1.05`:
  [proxy_eval_bundle_summary.json](/data/CoordExp/output/infer/coco1024_val200_compare_coordtoken_ckpt1332_20260423T080326Z/proxy_eval_bundle_summary.json)
  [timing_summary.json](/data/CoordExp/output/infer/coco1024_val200_compare_coordtoken_ckpt1332_20260423T080326Z/timing_summary.json)
- `1.10`:
  [proxy_eval_bundle_summary.json](/data/CoordExp/output/infer/coco1024_val200_compare_coordtoken_ckpt1332_rp1p10_20260423T091701Z/proxy_eval_bundle_summary.json)
  [timing_summary.json](/data/CoordExp/output/infer/coco1024_val200_compare_coordtoken_ckpt1332_rp1p10_20260423T091701Z/timing_summary.json)

## Recommendation

For future checkpoint comparisons on this surface:

- use coord-token `1.05` as the strongest accuracy reference
- use coord-token `1.00` when a faster benchmark point is useful
- if raw-text must be benchmarked, prefer `1.10`
- keep the repaired confidence scorer in the loop; otherwise fresh raw-text runs
  can be misread as failed
