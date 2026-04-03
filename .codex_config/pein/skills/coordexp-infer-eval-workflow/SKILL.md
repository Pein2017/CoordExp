---
name: coordexp-infer-eval-workflow
description: Run or audit the CoordExp inference-to-evaluation workflow when the user wants to launch inference, score predictions, evaluate detection metrics, bundle COCO plus LVIS-proxy views from one scored artifact, or summarize the resulting metrics and artifacts. Use this for YAML-first run execution and reproducible artifact checks instead of ad hoc command reconstruction.
---

# CoordExp Infer Eval Workflow

Use the repo's YAML-first production path.
Do not invent one-off CLI flags when an existing config already captures the run.

## Primary References

- canonical runbook:
  - `docs/eval/WORKFLOW.md`
- standard infer entrypoint:
  - `scripts/run_infer.py`
- confidence scoring:
  - `scripts/postop_confidence.py`
- single-view evaluation:
  - `scripts/evaluate_detection.py`
- one-run proxy bundle evaluation:
  - `scripts/evaluate_proxy_detection_bundle.py`

## Default Flow

```text
config + checkpoint + input JSONL
  -> inference
  -> gt_vs_pred.jsonl
  -> confidence post-op
  -> gt_vs_pred_scored.jsonl
  -> evaluation
  -> metrics.json / per_image.json / matches.jsonl / summaries
```

## Core Commands

Always run from repo root with `PYTHONPATH=.` and `conda run -n ms`.

Inference:

```bash
PYTHONPATH=. conda run -n ms python scripts/run_infer.py \
  --config <infer_config.yaml>
```

Confidence post-op:

```bash
PYTHONPATH=. conda run -n ms python scripts/postop_confidence.py \
  --config <postop_config.yaml>
```

Single evaluation:

```bash
PYTHONPATH=. conda run -n ms python scripts/evaluate_detection.py \
  --config <eval_config.yaml>
```

One-run proxy bundle evaluation:

```bash
PYTHONPATH=. conda run -n ms python scripts/evaluate_proxy_detection_bundle.py \
  --config <bundle_eval_config.yaml>
```

## Recommended Proxy Workflow

For COCO runs trained with LVIS proxy supervision:

1. Infer once.
2. Score once.
3. Evaluate three GT views from the same scored artifact.

Use these concrete configs when the user is working on the COCO-1024 proxy run:

- infer:
  - `configs/infer/coco_1024/val_200_lvis_proxy_merged.yaml`
- confidence:
  - `configs/postop/coco_1024/val_200_lvis_proxy_merged.yaml`
- bundled eval:
  - `configs/eval/coco_1024/val_200_lvis_proxy_bundle.yaml`

Those configs produce:

- `output/infer/coco1024_val200_lvis_proxy_merged/gt_vs_pred.jsonl`
- `output/infer/coco1024_val200_lvis_proxy_merged/gt_vs_pred_scored.jsonl`
- `output/infer/coco1024_val200_lvis_proxy_merged/eval_coco_real/`
- `output/infer/coco1024_val200_lvis_proxy_merged/eval_coco_real_strict/`
- `output/infer/coco1024_val200_lvis_proxy_merged/eval_coco_real_strict_plausible/`
- `output/infer/coco1024_val200_lvis_proxy_merged/proxy_eval_bundle_summary.json`

Interpretation:

- `coco_real` is the benchmark-aligned COCO headline.
- `coco_real_strict` is the best secondary proxy-expanded view.
- `coco_real_strict_plausible` is analysis-only and should not replace the COCO headline.

## What To Verify

Before launch:

- infer config points to the intended `gt_jsonl`
- infer config uses the intended checkpoint
- prompt controls match training when required:
  - `infer.prompt_variant`
  - `infer.object_field_order`
  - `infer.object_ordering`

After inference:

- `<run_dir>/summary.json` exists
- `<run_dir>/gt_vs_pred.jsonl` exists
- prompt/order settings in `summary.json` match the config

After confidence post-op:

- `<run_dir>/confidence_postop_summary.json` exists
- `<run_dir>/gt_vs_pred_scored.jsonl` exists

After evaluation:

- bundle summary exists when using proxy bundle eval:
  - `<run_dir>/proxy_eval_bundle_summary.json`
- each expected eval directory exists
- each eval directory contains `metrics.json`
- use the summary JSON as the default source for reporting cross-view metrics

## Reporting Guidance

When the user asks for performance:

- report `bbox_AP`, `bbox_AP50`, `bbox_AP75`, and the main F1-ish metric if available
- lead with `coco_real`
- clearly label `strict` and `strict_plausible` as additive proxy views
- mention GT counts when they materially explain score shifts

## Failure Modes

- If `metrics: both` on a COCO proxy artifact seems to trigger LVIS-federated assumptions, inspect `src/eval/detection.py` routing and verify dataset-policy detection before trusting the output.
- If images cannot be re-opened for visualization from derived artifacts, inspect `provenance.source_jsonl_dir` in the canonical visualization resource.
- If proxy-expanded GT counts look wrong, validate `metadata.coordexp_proxy_supervision.object_supervision` and the proxy-tier split before blaming the evaluator.

## Avoid

- Do not re-run inference when only evaluation views changed.
- Do not compare proxy-expanded numbers against standard COCO baselines without labeling them.
- Do not override config semantics with ad hoc shell flags unless the user explicitly asks for a one-off debug run.
