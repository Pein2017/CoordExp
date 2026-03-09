---
doc_id: progress.benchmarks.stage1-coco-2b-softce-2026-02-27
layer: progress
doc_type: benchmark
status: historical-benchmark
domain: research-history
summary: Historical Stage-1 benchmark for the 2B mixed CE+softCE checkpoint.
updated: 2026-03-09
---

# COCO Bench (200 samples): 2B Mixed CE+softCE Checkpoint @ max_pixels=32*32*{768,1024} (2026-02-27)

This note records measured detection metrics for the **2B** stage-1 mixed objective checkpoint (hard CE + softCE + W1 + gate), evaluated on COCO val (`limit=200`) under two `public_data` rescale presets:

- `rescale_32_768_bbox_max60`
- `rescale_32_1024_bbox_max60`

Reference style/precedent:
- `progress/benchmarks/stage1_coco80_4b_res_768_vs_1024_2026-02-26.md`

## 0) Checkpoint + eval mapping

- Training config: `configs/stage1/ablation_2b/ce_soft_ce_mixed.yaml`
- Adapter ckpt: `output/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332`
- Merged model: `output/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate_merged-1332`

## 1) Shared evaluation settings

All runs below use:

- Eval split: COCO val JSONL from `public_data`
- Sample limit: `200`
- Mode: `coord`
- Coord decode: `pred_coord_mode=auto`
- Decoding: `temperature=0.01`, `top_p=0.95`, `max_new_tokens=1024`, `repetition_penalty=1.05`, `batch_size=16`
- Metrics:
  - COCO bbox from scored artifacts (`confidence post-op` -> `gt_vs_pred_scored.jsonl`)
  - F1-ish from `gt_vs_pred.jsonl` (`f1ish_iou_thrs=[0.5, 0.3]`, `f1ish_pred_scope=annotated`)

## 2) Results (single model, two rescale presets)

| model | ckpt | rescale | bs | mAP | AP50 | AP75 | AR1 | F1@0.50 | F1@0.30 | pred_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ce_soft_ce_mixed_2b | output/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate_merged-1332 | 768 | 16 | 0.3896 | 0.5628 | 0.3905 | 0.3509 | 0.6569 | 0.7262 | 1253 |
| ce_soft_ce_mixed_2b | output/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate_merged-1332 | 1024 | 16 | 0.3879 | 0.5599 | 0.3963 | 0.3582 | 0.6444 | 0.7149 | 1267 |

## 3) Delta (1024 - 768)

| model | mAP | AP50 | AP75 | AR1 | F1@0.50 | F1@0.30 | pred_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ce_soft_ce_mixed_2b | -0.0018 | -0.0028 | +0.0058 | +0.0073 | -0.0125 | -0.0113 | +14 |

## 4) Takeaways (bounded to these 200 samples)

- For this 2B mixed checkpoint, `768` and `1024` are very close on mAP (`0.3896` vs `0.3879`).
- `1024` is slightly better on stricter localization (`AP75`, `AR1`), while `768` is better on F1 (`@0.50`, `@0.30`) on this slice.
- Net effect on this 200-sample subset is small; avoid over-interpreting the preset delta without larger-sample repeats.

## 5) Provenance: run directories

- 768 run dir: `output/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate_merged-1332_eval200_768_bs16/`
  - COCO metrics: `eval_coco_scored/metrics.json`
  - F1-ish metrics: `eval_f1ish/metrics.json`
- 1024 run dir: `output/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate_merged-1332_eval200_1024_bs16/`
  - COCO metrics: `eval_coco_scored/metrics.json`
  - F1-ish metrics: `eval_f1ish/metrics.json`

## 6) Note on earlier 768 quick check

An earlier 768 quick eval (before batch-size harmonization) was recorded under:
- `output/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate_merged-1332_eval200/`

For final comparison in this note, both `768` and `1024` are reported from matched `batch_size=16` runs.
