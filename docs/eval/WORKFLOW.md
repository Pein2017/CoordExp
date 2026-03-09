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

## Validation Checklist

- The run directory contains the expected artifacts for the requested stages.
- `gt_vs_pred.jsonl` records parse as JSON dicts.
- Geometry is pixel-ready and structurally valid.
- Metrics are finite and diagnostic counters are reasonable on a small subset.

## Read Next

- [CONTRACT.md](CONTRACT.md)
- [../ARTIFACTS.md](../ARTIFACTS.md)
