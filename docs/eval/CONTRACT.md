---
doc_id: docs.eval.contract
layer: docs
doc_type: reference
status: canonical
domain: eval
summary: Current Swift inference and detection-evaluation artifact boundaries.
tags: [eval, contract, jsonl]
updated: 2026-09-09
---

# Evaluation Artifact Reference

This page explains current Swift artifacts. Normative compatibility is owned
by the [scoring/artifact spec](../../openspec/specs/coordexp-swift-infer-scoring-artifacts/spec.md)
and [detection-evaluator spec](../../openspec/specs/coordexp-swift-detection-evaluator/spec.md).
The executable owners are [`src/inference/artifacts.py`](../../src/inference/artifacts.py)
and [`src/eval/detection_consumer.py`](../../src/eval/detection_consumer.py).

## Primary Input Artifacts

The direct evaluator consumes these siblings from one run directory:

- `gt_vs_pred.jsonl`: raw predictions and parsing metadata.
- `gt_vs_pred_scored.jsonl`: predictions with selected-token scores.
- `gt_vs_pred_scored.jsonl.provenance.json`: file hashes, row binding and
  model/config/score identity.
- `pred_token_trace.jsonl`: evidence for replaying selected-token scores.

The run manifest, when present, is checked against scored provenance. Preserve
the complete run directory rather than exporting only the scored rows.

## Pipeline Record Shape

| Carrier | Fields emitted by the current writer |
| --- | --- |
| Both raw and scored rows | `row_id`, `row_index`, `example_id`, `image_path`, `image_width`, `image_height`, `gt`, `pred` |
| Raw row additions | `raw_decode_text`, `decode_stop_reason`, `parser_id`, `parser_policy`, `metric_bearing`, `parse_status`, `valid_prediction_count`, `dropped_prediction_count`, `dropped_predictions` |
| Scored prediction additions | `score`, `pred_score_source`, `pred_score_version`, with selected-token provenance checked by the consumer |

This is a field map, not a replacement schema. The writer's
`_raw_artifact_row` and `_scored_artifact_row` define serialization. Raw and
scored rows retain the same row order, identities, image metadata and GT.
Scored predictions are consumed directly; the evaluator does not reparse raw
text. Preserve prediction identity and its score evidence when transforming
artifacts.

The old `image/width/height/coord_mode/raw_output_json` pipeline shape and
`vis_resources/` sidecar shape belong to the
[historical reference](../history/evaluation/2026-09-09-legacy-eval-reference.md).
They are not interchangeable with current Swift JSONL rows.

## Coordinate Handling

Inline GT `bbox` values are norm1000 coordinate-bin `xyxy`; the direct consumer
converts them to pixels using `image_width` and `image_height`. Scored prediction
`bbox` values are already parser-normalized pixel `xyxy`. Do not convert both
sides identically or add an old `coord_mode` heuristic. Dimensions must be
positive integers. Geometry normalization and invalid-box accounting belong
to the consumer; see [interpretation](INTERPRETATION.md#geometry-labels-and-coco-namespaces)
for units and category namespaces.

## Scoring Rules

The consumer verifies artifact hashes and row bindings, required identity
fields, supported per-prediction score provenance and score-policy identity.
It then recomputes selected-token scores against `pred_token_trace.jsonl`
before reduction. Arbitrary constant confidence, a renamed raw JSONL, or an
old confidence-postop carrier does not satisfy this contract.

Use the owning spec/source for exact required fields and score-channel rules;
[WORKFLOW.md](WORKFLOW.md) gives the current command. The interpretation page
separately explains [what a metric means](INTERPRETATION.md).

## Failure Policy

Missing required evidence, mismatched provenance, invalid row binding and
unsupported score provenance fail through the consumer's artifact errors.
Invalid geometry handling is explicit in normalization/conversion metrics;
it is not permission to skip malformed dimensions using historical behavior.
A successful tiny smoke does not establish benchmark eligibility or quality.

## Output Artifacts

The direct consumer writes `metrics.json` (or the requested metrics filename),
`evaluation_receipt.json`, `coco_gt.json`, and `coco_predictions.json`.
The COCO sidecars share evaluator-local category IDs. They are not official
COCO test-server exports. `per_image.json`, `matches*.jsonl`, guarded metrics
and overlays are not outputs of this aggregate consumer.

## Shared Visualization Contract

Current Swift review is owned by [`src/vis/`](../../src/vis/) through
[`scripts/visualize_detection.py`](../../scripts/visualize_detection.py).
It consumes run artifacts via its own loader and matching/rendering contract;
use the [review command](WORKFLOW.md#review-predictions). Do not require a
legacy `vis_resources/gt_vs_pred.jsonl` sidecar for this direct Swift route.

## Read Next

Select [WORKFLOW.md](WORKFLOW.md) for operation or
[INTERPRETATION.md](INTERPRETATION.md) for metric meaning. Use
[ARTIFACTS.md](../ARTIFACTS.md) only for broader artifact ownership.
