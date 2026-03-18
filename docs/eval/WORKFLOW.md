---
doc_id: docs.eval.workflow
layer: docs
doc_type: runbook
status: canonical
domain: eval
summary: YAML-first runbook for inference, confidence post-processing, evaluation, and visualization.
tags: [eval, infer, runbook]
updated: 2026-03-13
---

# Evaluation Workflow

This page describes the current production path from inference to scored evaluation artifacts, plus the additive Oracle-K repeated-sampling analysis workflow.

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

Run Oracle-K analysis:

```bash
PYTHONPATH=. conda run -n ms python scripts/evaluate_oracle_k.py \
  --config configs/eval/oracle_k.yaml
```

## Artifact Expectations

After inference:

- `gt_vs_pred.jsonl`
- `summary.json`
- `resolved_config.json` when using the YAML pipeline
- verify `infer.prompt_variant` and `infer.object_ordering` in both
  `summary.json` and `resolved_config.json` when comparing prompt/order ablations

After confidence post-op:

- `pred_confidence.jsonl`
- `gt_vs_pred_scored.jsonl`
- `confidence_postop_summary.json`

After evaluation:

- `metrics.json`
- `per_image.json`
- optional `per_class.csv`, `matches.jsonl`, and overlays
- `vis_resources/gt_vs_pred.jsonl` when the shared GT-vs-Pred reviewer is
  materialized for `scripts/run_vis.sh` or evaluator overlays

After Oracle-K analysis:

- `summary.json`
- `per_image.json`
- `fn_objects.jsonl`
- optional materialized inference run folders when Oracle-K is asked to generate repeated samples

## Oracle-K Repeated-Sampling Workflow

Use Oracle-K when you want to measure whether baseline false negatives are recoverable under repeated stochastic decoding, and how often they are recovered across `K` rollouts.

```text
baseline artifact or run spec
  + one or more Oracle artifact or run specs
  -> Oracle-K analysis
  -> summary.json / per_image.json / fn_objects.jsonl
```

The Oracle-K workflow is additive:

- standard `scripts/evaluate_detection.py` behavior does not change
- Oracle-K reuses the same F1-ish matching semantics for IoU thresholds, semantic matching, and prediction scope
- cross-run alignment is validated in record order and requires consistent `file_name` provenance; `record_idx` + `gt_idx` remains the normative object key and `image_id` / `file_name` are preserved for downstream visualization analysis

## Shared GT-vs-Pred Review Flow

The default repo visualization path now goes through the canonical sidecar:

```text
gt_vs_pred.jsonl
  -> vis_resources/gt_vs_pred.jsonl
  -> shared 1x2 GT-vs-Pred reviewer
  -> vis_*.png
```

Key points:

- `scripts/run_vis.sh` and `vis_tools/vis_coordexp.py` materialize the canonical
  sidecar before rendering.
- evaluator overlays reuse the same shared reviewer semantics instead of a
  second renderer-local box contract.
- post-eval audit materialization may reuse `matches.jsonl` and `per_image.json`
  to preserve canonical matching and join keys.

The YAML config can work in two modes:

- consume pre-generated `gt_vs_pred.jsonl` artifacts for one baseline run plus one or more Oracle runs
- materialize repeated inference runs through the standard infer pipeline before aggregation

Oracle-K v1 is intentionally object-level:

- it records per-run pairing for each baseline-FN GT object
- it preserves trace provenance when available through `pred_token_trace.jsonl`
- it does not require exact token-span-to-object alignment for matched predictions

Recommended validation checks for Oracle-K:

- all runs use the same subset and agree on GT content in the same record order
- the chosen IoU thresholds match the intended F1-ish study
- the output includes both `ever recovered` and `recover_count` / `recover_fraction`
- location-only and semantic+location recovery are inspected separately

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
