---
doc_id: docs.eval.contract
layer: docs
doc_type: reference
status: canonical
domain: eval
summary: Contract for CoordExp inference and detection-evaluation artifacts.
tags: [eval, contract, jsonl]
updated: 2026-03-22
---

# Evaluation Contract

This page defines the current infer/eval artifact contract.

## Primary Input Artifacts

- pipeline artifact:
  - `gt_vs_pred.jsonl`
- score-aware COCO artifact:
  - `gt_vs_pred_scored.jsonl`
- canonical visualization sidecar:
  - `vis_resources/gt_vs_pred.jsonl`

## Pipeline Record Shape

- Each record is a JSON object.
- Inline GT is required for evaluation-time consumption; evaluator-facing
  workflows do not use a separate GT file.
- Canonical required keys are:
  - `image`
  - `width`
  - `height`
  - `mode`
  - `gt`
  - `pred`
  - `coord_mode`
  - `raw_output_json`
  - `raw_special_tokens`
  - `raw_ends_with_im_end`
  - `errors`
  - `error_entries`
- Geometry objects live under `gt` and `pred`; legacy top-level prediction
  aliases are not canonical.

## Coordinate Handling

- `coord_mode: "pixel"` means evaluator consumers use `gt` and `pred` points as
  pixel coordinates directly.
- `coord_mode: "norm1000"` means evaluator consumers denormalize via per-record
  `width` and `height`, then clamp and round.
- Records missing `width` or `height` are skipped and counted because geometry
  validation is undefined without image size.

## Scoring Rules

- F1-ish-only evaluation may consume the base pipeline artifact.
- COCO evaluation consumes the scored artifact `gt_vs_pred_scored.jsonl`.
- COCO scoring uses `pred[*].score` from the scored artifact.
- Scored COCO inputs must also include:
  - `pred_score_source`
  - `pred_score_version`
- Missing or invalid scores are contract violations for COCO evaluation.
- Unscored legacy artifacts are not supported for COCO metrics.

## Failure Policy

- Parsing failures remain path-and-line explicit in evaluator diagnostics.
- Invalid or degenerate geometries are counted and surfaced in diagnostics.
- Unsupported geometry types fail or are rejected according to evaluator policy.

## Output Artifacts

- always:
  - `metrics.json`
  - `per_image.json`
- when F1-ish matching is enabled:
  - `matches.jsonl`
  - optional threshold-specific `matches@<thr>.jsonl`
- when COCO is enabled:
  - `per_class.csv`
  - `coco_gt.json`
  - `coco_preds.json`
- when shared-review overlays are materialized:
  - `vis_resources/gt_vs_pred.jsonl`

## Shared Visualization Contract

- The shared reviewer consumes canonical `vis_resources/gt_vs_pred.jsonl`
  records with:
  - top-level `schema_version`, `source_kind`, `record_idx`, `image`, `width`,
    `height`, `coord_mode`, `gt`, and `pred`
  - bbox-only per-object payloads: `index`, `desc`, `bbox_2d`
  - canonical `matching` with `pred_index_domain=canonical_pred_index` and
    `gt_index_domain=canonical_gt_index`
- Canonical visualization records preserve prediction order.
- Shared-review rendering fails fast if canonical `matching` is missing.

## Read Next

- [WORKFLOW.md](WORKFLOW.md)
- [../ARTIFACTS.md](../ARTIFACTS.md)
- [../training/METRICS.md](../training/METRICS.md)
