# COCO80 Bench (200 samples): 4B Checkpoints @ max_pixels=32*32*{768,1024} (2026-02-26)

This note records *measured* detection metrics for three **4B-sized** CoordExp stage-1 checkpoints on the **COCO80 val** eval set (limited to **200** samples), evaluated under two `public_data` rescale presets:

- `rescale_32_768_bbox_max60`
- `rescale_32_1024_bbox_max60`

We treat the three models as:

- **soft_ce_only**: stage-1 softCE(+W1/gate) style baseline checkpoint
- **pure_ce_only**: stage-1 pure CE baseline checkpoint
- **soft_ce_hard_ce_mixed**: stage-1 mixed objective (hardCE + softCE (+W1/gate)), previously mislabeled as “2B” in discussion but confirmed to be **4B**

## 0) Important caveat about “768 vs 1024” on COCO val

For COCO val, the rescale presets do **not** actually change image resolution in practice:

- `max(width*height)` in `public_data/coco/rescale_32_{768,1024}_bbox_max60/val.coord.jsonl` is **640*640 = 409,600** pixels.
- This is below both:
  - `32*32*768 = 786,432`
  - `32*32*1024 = 1,048,576`

So the “768 vs 1024” runs should be interpreted as *the same underlying images*, and any deltas are mostly due to decoding stochasticity (we use `temperature=0.1`) and, in a couple runs, different `batch_size`.

## 1) Shared evaluation settings

All runs below use:

- **Eval set**: COCO80 val from `public_data` (`val.coord.jsonl`), `limit=200`
- **Prompt template**: `prompt_variant=coco_80`
- **Object field order**: `desc_first`
- **Decoding**: `temperature=0.1`, `top_p=0.9`, `max_new_tokens=3084`, `seed=42`
- **Metrics**: bbox-only detection metrics (`--metrics both --no-segm`)
- **Artifacts per run** (under each run dir):
  - `summary.json` (generation settings, checkpoint path)
  - `eval/metrics.json` (mAP/F1/AP50/AP75/AR1/etc)
  - `pred_token_trace.jsonl`, `pred_confidence.jsonl`
  - `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`

## 2) Checkpoint mapping (all are 4B-sized)

### soft_ce_only (4B)

- Checkpoint: `output/stage1/coco_bbox_max60-coco80-desc_first/epoch_4-softce_w1-coco80-ckpt_1832-merged`

### pure_ce_only (4B)

- Checkpoint: `output/stage1/coco_bbox_max60-coco80-desc_first-pure_ce/ckpt-1932_merged`

### soft_ce_hard_ce_mixed (4B) (previously mislabeled “2B”)

- Checkpoint evaluated in benches: `output/stage1_2b/pure_ce-ckpt_1344-merged/`
- Notes:
  - This directory name is misleading; the merged weights are **4B**.
  - This checkpoint is also byte-identical to: `output/stage1_2b/hardce_softce-ckpt_1344-merged/`

## 3) Results (COCO80 val, 200 samples)

### Rescale preset: 768

| model | ckpt | bs | mAP | AP50 | AP75 | AR1 | F1@0.50 | F1@0.30 | pred_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| soft_ce_only | output/stage1/coco_bbox_max60-coco80-desc_first/epoch_4-softce_w1-coco80-ckpt_1832-merged | 8 | 0.2237 | 0.3090 | 0.2301 | 0.1861 | 0.5605 | 0.6056 | 991.0000 |
| pure_ce_only | output/stage1/coco_bbox_max60-coco80-desc_first-pure_ce/ckpt-1932_merged | 8 | 0.2534 | 0.3488 | 0.2577 | 0.2112 | 0.5635 | 0.6082 | 1004.0000 |
| soft_ce_hard_ce_mixed | output/stage1_2b/pure_ce-ckpt_1344-merged/ | 16 | 0.3856 | 0.5430 | 0.4047 | 0.3718 | 0.6453 | 0.7117 | 1369.0000 |

### Rescale preset: 1024

| model | ckpt | bs | mAP | AP50 | AP75 | AR1 | F1@0.50 | F1@0.30 | pred_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| soft_ce_only | output/stage1/coco_bbox_max60-coco80-desc_first/epoch_4-softce_w1-coco80-ckpt_1832-merged | 16 | 0.2305 | 0.3159 | 0.2313 | 0.1926 | 0.5665 | 0.6027 | 972.0000 |
| pure_ce_only | output/stage1/coco_bbox_max60-coco80-desc_first-pure_ce/ckpt-1932_merged | 16 | 0.2556 | 0.3457 | 0.2665 | 0.2144 | 0.5694 | 0.6165 | 1034.0000 |
| soft_ce_hard_ce_mixed | output/stage1_2b/pure_ce-ckpt_1344-merged/ | 16 | 0.3891 | 0.5448 | 0.4036 | 0.3711 | 0.6408 | 0.7014 | 1403.0000 |

### Delta (1024 - 768)

| model | mAP | AP50 | AP75 | AR1 | F1@0.50 | F1@0.30 | pred_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| soft_ce_only | 0.0068 | 0.0070 | 0.0011 | 0.0064 | 0.0060 | -0.0030 | -19.0000 |
| pure_ce_only | 0.0021 | -0.0031 | 0.0088 | 0.0031 | 0.0059 | 0.0083 | 30.0000 |
| soft_ce_hard_ce_mixed | 0.0035 | 0.0019 | -0.0011 | -0.0007 | -0.0044 | -0.0104 | 34.0000 |

## 4) Takeaways (bounded to these 200 samples)

- **soft_ce_hard_ce_mixed** is substantially stronger than both single-objective baselines on this slice:
  - mAP: ~`0.386-0.389` vs pure CE ~`0.253-0.256` vs soft CE ~`0.224-0.231`
  - AR1 and F1@0.50 show the same pattern (mixed >> baselines).
- **pure_ce_only** > **soft_ce_only** by mAP on this slice, but the gap is much smaller than mixed vs pure.
- The “768 vs 1024” differences are small and should not be over-interpreted on COCO val (see caveat above).

## 5) Provenance: run directories (dumps + metrics)

- soft_ce_only:
  - 768: `output/bench/softce_w1_1832_merged_coco_val_limit200/`
  - 1024: `output/bench/softce_w1_1832_merged_coco_val_limit200_res1024/`
- pure_ce_only:
  - 768: `output/bench/pure_ce_1932_merged_coco_val_limit200/`
  - 1024: `output/bench/pure_ce_1932_merged_coco_val_limit200_res1024/`
- soft_ce_hard_ce_mixed:
  - 768: `output/bench/pure_ce_2b_1344_coco_val_limit200_res768/`
  - 1024: `output/bench/pure_ce_2b_1344_coco_val_limit200_res1024/`

