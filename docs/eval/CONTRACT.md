---
doc_id: docs.eval.contract
layer: docs
doc_type: reference
status: canonical
domain: eval
summary: Contract for CoordExp inference and detection-evaluation artifacts.
tags: [eval, contract, jsonl]
updated: 2026-07-03
---

# Evaluation Contract

This page defines the current infer/eval artifact contract.

## Primary Input Artifacts

- pipeline artifact:
  - `gt_vs_pred.jsonl`
- score-aware COCO artifact:
  - `gt_vs_pred_scored.jsonl`
- CoordExp-Swift score-aware provenance:
  - `gt_vs_pred_scored.jsonl.provenance.json`
- canonical visualization sidecar:
  - `vis_resources/gt_vs_pred.jsonl`

CoordExp-Swift standardized detection evaluator:

- consumes `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, and
  `gt_vs_pred_scored.jsonl.provenance.json` from the same artifact directory;
- requires raw/scored rows to preserve row order, `row_id`, `row_index`,
  `example_id` when present, `image_path`, `image_width`, `image_height`, and
  `gt` exactly;
- uses scored `pred` only for already-scored prediction objects and does not
  reparse raw decode text;
- converts inline GT `bbox` values from norm1000 coord-bin `xyxy` to pixel
  `xyxy` with the row image dimensions;
- treats scored prediction `bbox` values as already parser-normalized pixel
  `xyxy`;
- writes aggregate bbox COCO metrics only in V1.

CoordExp-Swift validation scope:

- the fixed val200 inference/eval run is sufficient V1 validation evidence when
  it has scored artifacts, valid score provenance, and mAP/mRecall output from
  `src/eval/detection_consumer.py`;
- tiny debug/smoke runs are implementation gates only;
- full validation-dataset evaluation is optional and not required for the V1
  readiness claim.

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
- `raw_output_json` is the parsed best-effort raw payload from the shared
  salvage/parser path, not a verbatim raw-text mirror.
- Compact detection inference parses model rollouts at object-span granularity.
  Valid object spans are salvaged into `raw_output_json.objects` and `pred`;
  invalid spans are dropped into `dropped_pred_objects` with a concrete
  `reason`, optional `detail`, and the offending `raw_text`. Compact rows also
  expose `parse_status`, `raw_object_spans_total`,
  `valid_pred_object_count`, and `dropped_pred_object_count`.
- `empty_pred` is reserved for outputs with no valid prediction objects. A
  compact rollout with some valid objects and some invalid spans should report
  `dropped_invalid_object` rather than erasing the whole prediction row.

CoordExp-Swift rebuilt inference rows use the narrower fields `row_id`,
`row_index`, `example_id`, `image_path`, `image_width`, `image_height`, `gt`,
`pred`, `raw_decode_text`, `parser_id`, `parser_policy`, `metric_bearing`,
`parse_status`, `valid_prediction_count`, `dropped_prediction_count`, and
`dropped_predictions`. The standardized Swift evaluator consumes this shape.

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
- Official COCO/LVIS/both metric reducers and COCO submission export fail fast
  unless the consumed artifact has comparable score-bearing provenance. The
  artifact family must be score-bearing, not the raw `gt_vs_pred.jsonl` family,
  and the carrier must include `score_policy_fingerprint`.
- legacy/mainline evaluators may consume non-canonical `cxcy_logw_logh` or
  `cxcywh` scored artifacts materialized with deterministic constant-score
  provenance. The direct CoordExp-Swift V1 evaluator does not consume that
  constant-score family; it requires selected-token score provenance from the
  rebuilt inference artifact writer.
- Scored COCO inputs must also include:
  - `pred_score_source`
  - `pred_score_version`
- For CoordExp-Swift V1, each scored prediction carries `pred_score_source`
  inside the prediction object. That source must be a selected-token provenance
  mapping whose `row_id`, `object_span_id`, and `score_policy_fingerprint`
  match the evaluated row, prediction object, and scored-artifact provenance
  sidecar.
- Missing or invalid scores are contract violations for COCO evaluation.
- Unscored legacy artifacts are not supported for COCO metrics.
- confidence post-op remains `xyxy`-only; `cxcy_logw_logh` and `cxcywh` do not
  support confidence-derived score reconstruction in V1.

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
- CoordExp-Swift direct evaluator V1:
  - `metrics.json`
  - `coco_gt.json`
  - `coco_predictions.json`
  - no `per_class.csv` or `per_image.json` in V1
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
