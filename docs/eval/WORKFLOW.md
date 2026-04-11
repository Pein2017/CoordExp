---
doc_id: docs.eval.workflow
layer: docs
doc_type: runbook
status: canonical
domain: eval
summary: YAML-first runbook for inference, confidence post-processing, evaluation, and visualization.
tags: [eval, infer, runbook]
updated: 2026-04-03
---

# Evaluation Workflow

This page describes the current production path from inference to scored evaluation artifacts, plus the additive Oracle-K repeated-sampling analysis workflow.

Implementation ownership note:
- pipeline orchestration lives in `src/infer/pipeline.py`
- generation/backend selection lives in `src/infer/engine.py` and `src/infer/backends.py`
- infer/eval artifact writing lives in `src/infer/artifacts.py`, `src/eval/orchestration.py`, and `src/eval/artifacts.py`

## Default Flow

```text
input JSONL + checkpoint
  -> inference
  -> gt_vs_pred.jsonl
  -> confidence post-op (when COCO scoring is needed)
  -> gt_vs_pred_scored.jsonl
  -> evaluation
  -> raw metrics/artifacts
  -> optional duplicate-control guard
  -> guarded metrics/artifacts
  -> metrics.json / metrics_guarded.json / per_image.json / per_image_guarded.json / optional overlays
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

Duplicate-control guard note:

- `scripts/evaluate_detection.py` accepts the YAML-only toggle
  `duplicate_control.enabled: true`
- when enabled, the evaluator keeps the raw input artifact authoritative and
  additionally emits guarded companions plus a duplicate-control report
- prefer reporting both raw and guarded metrics together when comparing runs

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
- `resolved_config.path` next to `gt_vs_pred.jsonl` when the YAML pipeline is
  responsible for artifact materialization
- verify `infer.prompt_variant`, `infer.object_field_order`, and
  `infer.object_ordering` in both `summary.json` and `resolved_config.json`
  when comparing prompt/order ablations
- use `resolved_config.path` when a downstream eval or visualization job is
  consuming `gt_vs_pred.jsonl` from outside the original run directory
- if the checkpoint was trained with non-default dense prompt controls, keep
  those infer-time values aligned with training so evaluation does not measure
  prompt drift instead of model behavior
- training-only bbox regression parameterizations such as
  `bbox_geo.parameterization: center_size` do not change this infer/eval
  artifact split: base predictions still write `gt_vs_pred.jsonl`, scored
  predictions still write `gt_vs_pred_scored.jsonl`, and downstream jobs should
  continue using `resolved_config.path` to recover the authoritative
  `resolved_config.json`

After confidence post-op:

- `pred_confidence.jsonl`
- `gt_vs_pred_scored.jsonl`
- `confidence_postop_summary.json`

After evaluation:

- `metrics.json`
- `metrics_guarded.json` when `duplicate_control.enabled: true`
- `per_image.json`
- `per_image_guarded.json` when `duplicate_control.enabled: true`
- `duplicate_guard_report.json` when `duplicate_control.enabled: true`
- `matches_guarded.jsonl` and `matches@<thr>_guarded.jsonl` when match exports
  are enabled under duplicate control
- optional `per_class.csv`, `matches.jsonl`, and overlays
- `vis_resources/gt_vs_pred.jsonl` when the shared GT-vs-Pred reviewer is
  materialized for `scripts/run_vis.sh` or evaluator overlays

Guarded-artifact rule:

- non-COCO raw evaluation continues to consume `gt_vs_pred.jsonl`
- score-aware COCO evaluation continues to consume `gt_vs_pred_scored.jsonl`
- guarded companions follow the same input family:
  - `gt_vs_pred_guarded.jsonl`
  - `gt_vs_pred_scored_guarded.jsonl`
- treat the raw artifact as the main research/debug surface and the guarded
  artifact as the safety/post-op surface

## COCO + LVIS Proxy Evaluation

For COCO runs trained with LVIS proxy supervision, keep the benchmark headline
explicit:

- report standard COCO metrics on the original COCO GT objects only
- treat LVIS proxy-expanded GT as additive analysis, not as the replacement
  headline benchmark

Recommended flow:

```text
gt_vs_pred_scored.jsonl
  -> materialize proxy GT views
  -> coco_real / coco_real_strict / coco_real_strict_plausible JSONLs
  -> run the standard evaluator on each view separately
```

Use `scripts/materialize_proxy_eval_views.py` on the scored artifact to create:

- `coco_real`
  - original COCO GT only (`proxy_tier = real`)
- `coco_real_strict`
  - COCO GT plus strict LVIS proxies (`same_extent_proxy`)
- `coco_real_strict_plausible`
  - COCO GT plus strict and plausible LVIS proxies

For a one-inference / one-scored-artifact workflow, use
`scripts/evaluate_proxy_detection_bundle.py` to:

- reuse one `gt_vs_pred_scored.jsonl`
- materialize the proxy GT views under the same run directory
- run the standard evaluator once per view
- write side-by-side outputs such as:
  - `eval_coco_real/`
  - `eval_coco_real_strict/`
  - `eval_coco_real_strict_plausible/`
  - `proxy_eval_bundle_summary.json`

Interpretation guidance:

- `coco_real` is the benchmark-aligned number to compare against standard COCO
  baselines
- `coco_real_strict` estimates recoverable misses where LVIS adds same-extent
  annotations
- `coco_real_strict_plausible` is the broadest supervision view and is useful
  for recall analysis, but it is the least comparable to standard COCO

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
