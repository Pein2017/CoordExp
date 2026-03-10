---
doc_id: docs.eval.coco-test-submission
layer: docs
doc_type: runbook
status: canonical
domain: eval
summary: End-to-end runbook for 1024-budget COCO test-dev inference, submission export, and official scoring.
tags: [eval, coco, test-dev, submission]
updated: 2026-03-10
---

# COCO Test-Dev Submission

Use this runbook when you want a **real official COCO benchmark** after training finishes.

This path assumes:

- the model expects **offline smart-resized** input with
  - `image_factor=32`
  - `max_pixels=32*32*1024`
  - `min_pixels=32*32*4`
- resized inference runs on the 1024-budget test-dev images
- official submission boxes must be projected back to the **original COCO test-dev resolution**
  before upload

## 1. Download Official-Test Assets

```bash
./public_data/run.sh coco download -- --include-test
./public_data/run.sh coco convert -- --include-test --test-split test-dev
```

Artifacts:

- original-resolution source JSONL:
  - `public_data/coco/raw/test-dev.jsonl`
- image-info JSON:
  - `public_data/coco/raw/annotations/image_info_test-dev2017.json`

## 2. Build The 1024-Budget Inference Input

This is the actual inference input for the model.

```bash
PYTHONPATH=. conda run -n ms python public_data/scripts/rescale_jsonl.py \
  --input-jsonl public_data/coco/raw/test-dev.jsonl \
  --output-jsonl public_data/coco/rescale_32_1024_bbox/test-dev.jsonl \
  --output-images public_data/coco/rescale_32_1024_bbox \
  --image-factor 32 \
  --max-pixels $((32*32*1024)) \
  --min-pixels $((32*32*4)) \
  --relative-images
```

Why this matters:

- the model sees the largest 32-aligned image that fits the 1024 budget
- aspect ratio stays as close as possible to the original COCO image
- the exporter later uses the original `raw/test-dev.jsonl` to project predictions
  back to official COCO pixel space

## 3. Run Real COCO Test-Dev Inference

Point the config to your final merged checkpoint and run:

```bash
PYTHONPATH=. conda run -n ms python scripts/run_infer.py \
  --config configs/infer/ablation/coco80_testdev_desc_first.yaml
```

Reference config:

- `configs/infer/ablation/coco80_testdev_desc_first.yaml`

Expected artifact:

- `<run_dir>/gt_vs_pred.jsonl`

## 4. Score The Predictions

```bash
PYTHONPATH=. conda run -n ms python scripts/postop_confidence.py \
  --config configs/postop/confidence.yaml
```

Expected artifact:

- `<run_dir>/gt_vs_pred_scored.jsonl`

## 5. Export The Official Submission JSON

Use the original-resolution source JSONL here, not the resized inference JSONL.

```bash
PYTHONPATH=. conda run -n ms python scripts/export_coco_submission.py \
  --config configs/eval/coco_submission.yaml
```

What the exporter does:

- reads scored predictions from the resized inference run
- reads `public_data/coco/raw/test-dev.jsonl` for the original COCO `image_id`, `width`, and `height`
- converts the scored boxes from resized pixel space back to original COCO test-dev resolution
- writes a valid official detection submission JSON

Expected artifacts:

- `<run_dir>/coco_submission.json`
- `<run_dir>/submission_summary.json`
- optional `<run_dir>/semantic_desc_report.json`

## 6. Upload And Record The Official Score

Upload `coco_submission.json` to the official COCO **detection test-dev** evaluation server.

Record these metrics in your benchmark note after the server returns them:

- `AP`
- `AP50`
- `AP75`
- `APs`
- `APm`
- `APl`

Recommended note location:

- `progress/benchmarks/`
- or `progress/diagnostics/` if you are still iterating

## 7. Local Val Anchor

For the closest local sanity reference, use a 1024-budget COCO val recipe after training.

Good starting handle:

- `configs/bench/pure_ce_2b_1344_coco_val_1024_limit200.yaml`

Clone or adapt that config so the checkpoint, prompt variant, backend, and generation settings
match your final test-dev run as closely as possible.
