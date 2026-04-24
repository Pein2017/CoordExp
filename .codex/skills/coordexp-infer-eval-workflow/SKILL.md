---
name: coordexp-infer-eval-workflow
description: Use when launching, repairing, auditing, or summarizing CoordExp inference, confidence scoring, duplicate-control, COCO/LVIS-proxy evaluation, Oracle-K analysis, or benchmark artifact provenance.
---

# CoordExp Infer Eval Workflow

Use the repo's YAML-first production path.
Do not invent one-off CLI flags when an existing config already captures the run.
When running interactively, wrap noisy commands with `rtk` if useful, but keep `PYTHONPATH=.` and `conda run -n ms python`.

## Primary References

- canonical docs:
  - `docs/eval/WORKFLOW.md`
  - `docs/eval/CONTRACT.md`
  - `docs/ARTIFACTS.md`
- reusable runtime seams:
  - `src/infer/pipeline.py::run_pipeline`
  - `src/infer/engine.py::InferenceEngine.infer`
  - `src/infer/artifacts.py::build_infer_summary_payload`
  - `src/eval/detection.py::evaluate_and_save`
  - `src/eval/artifacts.py`
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
  -> score materialization
  -> gt_vs_pred_scored.jsonl
  -> evaluation and optional duplicate-control guard
  -> metrics.json / metrics_guarded.json / per_image.json / matches.jsonl / summaries
```

Score materialization depends on the coordinate surface:

- `bbox_format: xyxy` with coord tokens: run confidence post-op.
- raw-text `xyxy` norm1000: run confidence post-op with numeric-text span alignment; set `infer.mode: text`, `infer.pred_coord_mode: norm1000`, and do not rely on `auto`.
- `bbox_format: cxcy_logw_logh` or `cxcywh`: do not run confidence post-op; use the unified pipeline's deterministic constant-score compatibility artifact only for checkpoints trained on that serialization.

## Core Commands

Always run from repo root. Use YAML configs, not ad hoc shell overrides, for stable workflows.

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

Oracle-K analysis:

```bash
PYTHONPATH=. conda run -n ms python scripts/evaluate_oracle_k.py \
  --config <oracle_k_config.yaml>
```

One-run proxy bundle evaluation:

```bash
PYTHONPATH=. conda run -n ms python scripts/evaluate_proxy_detection_bundle.py \
  --config <bundle_eval_config.yaml>
```

## COCO + LVIS Proxy Workflow

For COCO runs trained with LVIS proxy supervision:

1. Infer once.
2. Score once.
3. Evaluate three GT views from the same scored artifact.

The standard view labels are:

- `coco_real`: original COCO GT only; this is the benchmark-aligned headline.
- `coco_real_strict`: COCO GT plus strict same-extent LVIS proxies.
- `coco_real_strict_plausible`: broad analysis view; useful for recall, least comparable to standard COCO.

Use existing configs under `configs/infer/`, `configs/postop/`, `configs/eval/`, and `configs/bench/`. Only reach for old concrete configs such as the COCO-1024 `val_200_lvis_proxy_*` set when the user is working on that exact historical run:

- `configs/infer/coco_1024/val_200_lvis_proxy_merged.yaml`
- `configs/postop/coco_1024/val_200_lvis_proxy_merged.yaml`
- `configs/eval/coco_1024/val_200_lvis_proxy_bundle.yaml`

Expected bundle outputs:

- `<run_dir>/eval_coco_real/`
- `<run_dir>/eval_coco_real_strict/`
- `<run_dir>/eval_coco_real_strict_plausible/`
- `<run_dir>/proxy_eval_bundle_summary.json`

## What To Verify

Before launch:

- infer config points to the intended `gt_jsonl`
- infer config uses the intended checkpoint or adapter shorthand
- root image directories resolve from config/provenance, especially when running from `temp/` or sharded work dirs
- prompt controls match training when required:
  - `infer.prompt_variant`
  - `infer.object_field_order`
  - `infer.object_ordering`
- coordinate surface is intentional:
  - `infer.mode`
  - `infer.pred_coord_mode`
  - `infer.bbox_format`
- benchmark scope is explicit:
  - dataset path
  - slice such as `val200`, `limit=200`, or full-val
  - decoding knobs and GPU launch shape when reporting timing

After inference:

- `<run_dir>/summary.json` exists
- `<run_dir>/gt_vs_pred.jsonl` exists
- `<run_dir>/resolved_config.json` exists when using the YAML pipeline
- `resolved_config.path` is present next to `gt_vs_pred.jsonl` when downstream jobs need to recover the authoritative config
- prompt/order settings in `summary.json` and `resolved_config.json` match the config

After confidence post-op:

- `<run_dir>/confidence_postop_summary.json` exists
- `<run_dir>/pred_confidence.jsonl` exists for confidence-scored paths
- `<run_dir>/gt_vs_pred_scored.jsonl` exists

After non-canonical constant-score compatibility scoring:

- `<run_dir>/gt_vs_pred_scored.jsonl` exists
- do not expect `pred_confidence.jsonl` or `confidence_postop_summary.json`

After evaluation:

- raw metrics exist:
  - `metrics.json`
  - `per_image.json`
- guarded companions exist when `duplicate_control.enabled: true`:
  - `metrics_guarded.json`
  - `per_image_guarded.json`
  - `duplicate_guard_report.json`
- bundle summary exists when using proxy bundle eval:
  - `<run_dir>/proxy_eval_bundle_summary.json`
- each expected eval directory exists
- each eval directory contains `metrics.json`
- use the summary JSON as the default source for reporting cross-view metrics

For long or sharded runs:

- verify top-level merged artifacts and summaries, not just per-shard logs
- rerun only failed or missing shards when possible
- preserve canonical image roots or rewrite them explicitly before launching from scratch space

## Reporting Guidance

When the user asks for performance:

- report `bbox_AP`, `bbox_AP50`, `bbox_AP75`, and the main F1-ish metric if available
- lead with `coco_real`
- clearly label `strict` and `strict_plausible` as additive proxy views
- state scope before comparing numbers: `val200`, `limit=200`, first-200, full-val, proxy view, raw-text vs coord-token, checkpoint id, and repetition penalty if relevant
- mention GT counts, kept/total prediction counts, or scorer repairs when they materially explain score shifts
- compare throughput only across compatible launch shapes; GPU count differences can make timing non-comparable even when accuracy is comparable

## Failure Modes

- If `metrics: both` on a COCO proxy artifact seems to trigger LVIS-federated assumptions, inspect `src/eval/detection.py` routing and verify dataset-policy detection before trusting the output.
- If eval behavior is unclear, inspect `src/eval/detection.py::EvalOptions` and `src/eval/detection.py::evaluate_and_save`; `scripts/evaluate_detection.py` is a wrapper.
- If infer behavior is unclear, inspect `src/infer/pipeline.py::run_pipeline` and `src/infer/engine.py::InferenceEngine.infer`; `scripts/run_infer.py` is a wrapper.
- If images cannot be re-opened for visualization from derived artifacts, inspect `provenance.source_jsonl_dir` in the canonical visualization resource.
- If proxy-expanded GT counts look wrong, validate `metadata.coordexp_proxy_supervision.object_supervision` and the proxy-tier split before blaming the evaluator.
- If raw-text predictions appear to collapse after scoring, verify the confidence post-op used the numeric-text path instead of coord-token geometry alignment.
- If a non-canonical bbox-format result looks strong or weak, confirm the checkpoint was trained against that exact serialization before treating it as evidence.

## Avoid

- Do not re-run inference when only evaluation views changed.
- Do not compare proxy-expanded numbers against standard COCO baselines without labeling them.
- Do not compare `val200` or `limit=200` against full-val without saying so.
- Do not treat guarded metrics as a replacement for raw model-output inspection; guarded artifacts are additive post-op views.
- Do not override config semantics with ad hoc shell flags unless the user explicitly asks for a one-off debug run.
