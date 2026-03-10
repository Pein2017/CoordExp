---
doc_id: docs.eval.workflow
layer: docs
doc_type: runbook
status: canonical
domain: eval
summary: YAML-first runbook for inference, confidence post-processing, evaluation, and visualization.
tags: [eval, infer, runbook]
updated: 2026-03-09
---

# Evaluation Workflow

This page describes the current production path from inference to scored evaluation artifacts.

## Default Flow

```text
input JSONL + checkpoint
  -> inference
  -> gt_vs_pred.jsonl
  -> confidence post-op (when COCO scoring is needed)
  -> gt_vs_pred_scored.jsonl
  -> evaluation
  -> metrics.json / per_image.json / optional overlays
```

## YAML-First Commands

Run inference:

```bash
PYTHONPATH=. conda run -n ms python scripts/run_infer.py \
  --config configs/infer/pipeline.yaml
```

Run confidence post-op:

```bash
PYTHONPATH=. conda run -n ms python scripts/postop_confidence.py \
  --config configs/postop/confidence.yaml
```

Run evaluation:

```bash
PYTHONPATH=. conda run -n ms python scripts/evaluate_detection.py \
  --config configs/eval/detection.yaml
```

## Artifact Expectations

After inference:

- `gt_vs_pred.jsonl`
- `summary.json`
- `resolved_config.json` when using the YAML pipeline

After confidence post-op:

- `pred_confidence.jsonl`
- `gt_vs_pred_scored.jsonl`
- `confidence_postop_summary.json`

After evaluation:

- `metrics.json`
- `per_image.json`
- optional `per_class.csv`, `matches.jsonl`, and overlays

## Official COCO Test-Dev Submission Flow

Use this path when you need a ready-to-upload COCO detection JSON for the
official test-dev server.

```text
COCO test-dev JSONL
  -> inference
  -> gt_vs_pred.jsonl
  -> confidence post-op
  -> gt_vs_pred_scored.jsonl
  -> official submission export
  -> coco_submission.json
```

Prepare the original-resolution source test-dev JSONL:

```bash
./public_data/run.sh coco download -- --include-test
./public_data/run.sh coco convert -- --include-test --test-split test-dev
```

Build the 1024-budget resized inference input:

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

Run inference and confidence post-op:

```bash
PYTHONPATH=. conda run -n ms python scripts/run_infer.py \
  --config configs/infer/ablation/coco80_testdev_desc_first.yaml

PYTHONPATH=. conda run -n ms python scripts/postop_confidence.py \
  --config configs/postop/confidence.yaml
```

Export the official submission JSON:

```bash
PYTHONPATH=. conda run -n ms python scripts/export_coco_submission.py \
  --config configs/eval/coco_submission.yaml
```

Expected export artifacts:

- `coco_submission.json`
- `submission_summary.json`
- optional `semantic_desc_report.json` when semantic label mapping was needed

Current caveat:

- The shared public-data preset pipeline still manages `train` / `val` splits only.
- The recommended official-submission flow therefore uses:
  - original-resolution source JSONL under `public_data/coco/raw/`
  - resized inference JSONL under `public_data/coco/rescale_32_1024_bbox/`
- `scripts/export_coco_submission.py` projects detections back to the original
  COCO test-dev resolution before writing `coco_submission.json`.

## Validation Checklist

- The run directory contains the expected artifacts for the requested stages.
- `gt_vs_pred.jsonl` records parse as JSON dicts.
- Geometry is pixel-ready and structurally valid.
- Metrics are finite and diagnostic counters are reasonable on a small subset.

## Read Next

- [CONTRACT.md](CONTRACT.md)
- [../ARTIFACTS.md](../ARTIFACTS.md)
