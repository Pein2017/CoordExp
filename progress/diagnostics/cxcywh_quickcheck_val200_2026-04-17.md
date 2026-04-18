---
title: cxcywh Quick Check on COCO val200
date: 2026-04-17
status: in_progress
owner: codex
---

# `cxcywh` Quick Check on COCO `val200`

## Scope

- checkpoint under test:
  - `output/stage1_2b/cxcy_wh-pure_ce-merged`
- quick validation artifact root:
  - `output/infer/coco1024_val200_lvis_proxy_cxcywh_pure_ce_merged`
- question:
  - is the `cxcywh` run still suffering from the old non-canonical decode failure mode?
  - if AP remains slightly below the `0.28` baseline band, does the gap look more like
    duplication-tail drag or broader geometry weakness?

## Main Result

On the reproducible `200`-image COCO val quick check:

- `coco_real bbox_AP = 0.2725262341`
- `coco_real bbox_AP50 = 0.4003027885`
- `coco_real bbox_AP75 = 0.2840814996`
- `f1@0.50 = 0.4652624961`

This is not the old "wrong decode / near-zero AP" failure mode.

The scored artifact confirms the correct non-canonical compatibility path:

- `pred_score_source = "cxcywh_constant"`
- `pred_score_version = 1`
- predictions are evaluated only after `cxcywh -> xyxy` standardization

## Quick Symptom Audit

### 1. No widespread parse or gibberish failure

From `gt_vs_pred.jsonl` and `pred_token_trace.jsonl`:

- `errors = {"empty_pred": 1}`
- `raw_output_json_none = 0`
- `invalid_json = 0`
- `invalid_geometry = 0`
- `invalid_coord = 0`
- no suspicious class-name gibberish in the emitted `desc` strings

Interpretation:

- the run is producing parseable object lists
- this does not look like a JSON-collapse or garbled-label regime

### 2. There is still a concentrated duplication tail

Across `200` images:

- total predictions: `1774`
- images with `max_same_desc_count >= 50`: `7`
- those `7` images contribute `716 / 1774 = 40.36%` of all predictions

Worst cases:

- row `2`: `123x "book"`
- row `135`: `123x "vase"`
- row `0`: `117x "vase"`
- row `41`: `116x "bottle"`
- row `178`: `86x "person"`
- row `82`: `74x "carrot"`
- row `137`: `61x "book"`

Interpretation:

- the duplication/collapse pattern is still present
- but it is concentrated in a small number of scenes rather than global

### 3. On non-burst images, AP already rises back above the `0.28` line

I split the `val200` scored artifact into:

- `burst50`
  - images with `max_same_desc_count >= 50`
  - `7` images
- `nonburst50`
  - remaining `193` images

Results:

- full `val200`:
  - `bbox_AP = 0.2725262341`
- burst50 only:
  - `bbox_AP = 0.0693458237`
- nonburst50 only:
  - `bbox_AP = 0.2879403670`

Interpretation:

- on this quick slice, the run clears the `0.28` band once the severe burst tail is removed
- this suggests the current gap vs the baseline line is dominated by a small but very expensive duplication subset

### 4. Some decode inefficiency is still visible after valid JSON

Token-trace audit:

- median generated token count: `317`
- `95th` percentile generated token count: `3084`
- median trailing `<|endoftext|>` tail: `124`
- `95th` percentile trailing `<|endoftext|>` tail: `1968`
- rows with trailing `<|endoftext|> >= 500`: `30`

Interpretation:

- many samples finish the useful JSON early but keep decoding special tokens until the generation budget ends
- this is inefficient and may correlate with the scenes that drift into excessive output length
- however it is not the same as garbled or unparsable output

## Current Read

The `cxcywh` family currently looks healthier than the earlier `cxcy_logw_logh` retrain:

- no catastrophic decode mismatch
- no widespread gibberish or parse collapse
- AP is already close to the desired `0.28` band on the quick slice

The remaining issue on `val200` looks primarily like:

- a concentrated same-desc duplication tail,
- not a broad semantic failure,
- and not the old broken non-canonical inversion problem

## Artifacts

- quick run root:
  - [output/infer/coco1024_val200_lvis_proxy_cxcywh_pure_ce_merged](/data/CoordExp/output/infer/coco1024_val200_lvis_proxy_cxcywh_pure_ce_merged)
- quick bundle summary:
  - [proxy_eval_bundle_summary.json](/data/CoordExp/output/infer/coco1024_val200_lvis_proxy_cxcywh_pure_ce_merged/proxy_eval_bundle_summary.json)
- burst subset summary:
  - [proxy_eval_bundle_summary.json](/data/CoordExp/temp/cxcywh_val200_burst50_eval/proxy_eval_bundle_summary.json)
- non-burst subset summary:
  - [proxy_eval_bundle_summary.json](/data/CoordExp/temp/cxcywh_val200_nonburst50_eval/proxy_eval_bundle_summary.json)

## Next Step

The authoritative answer should come from the full-val run:

- active full-val infer root:
  - `output/infer/coco1024_valfull_lvis_proxy_cxcywh_pure_ce_merged_vllm8`

Once that run completes, the next update should answer:

- whether the concentrated duplication tail remains the main limiter on full val
- whether full-val `bbox_AP` stays near or above the `0.28` quick-check band
