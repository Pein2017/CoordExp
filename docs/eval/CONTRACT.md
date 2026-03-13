---
doc_id: docs.eval.contract
layer: docs
doc_type: reference
status: canonical
domain: eval
summary: Contract for CoordExp inference and detection-evaluation artifacts.
tags: [eval, contract, jsonl]
updated: 2026-03-13
---

# Evaluation Contract

This page defines the current infer/eval artifact contract.

## Primary Input Artifacts

- Base inference artifact:
  - `gt_vs_pred.jsonl`
- Score-aware COCO artifact:
  - `gt_vs_pred_scored.jsonl`
- Canonical visualization sidecar:
  - `vis_resources/gt_vs_pred.jsonl`

## Required Record Shape

- Each record is a JSON object.
- Inline GT is required for evaluation-time consumption.
- Geometry objects live under `gt` and `pred`.
- Width and height come from inline GT metadata.
- Geometry must already be pixel-ready for evaluator consumption.

## Scoring Rules

- COCO scoring uses `pred[*].score` from the scored artifact.
- Missing or invalid scores are contract violations for COCO evaluation.
- Unscored legacy artifacts are not supported.

## Failure Policy

- Parsing strictness is configuration-controlled.
- Invalid or degenerate geometries are counted and surfaced in diagnostics.
- Unsupported geometry types fail or are rejected according to evaluator policy.

## Output Artifacts

- always:
  - `metrics.json`
  - `per_image.json`
- when COCO is enabled:
  - `per_class.csv`
  - `coco_gt.json`
  - `coco_preds.json`
- when F1-ish matching is enabled:
  - `matches.jsonl`
  - optional threshold-specific `matches@<thr>.jsonl`

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
