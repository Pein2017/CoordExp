# detection-evaluator Specification (delta: score is mandatory and always honored)

## Purpose
Enable confidence-sensitive COCO AP/mAP evaluation by always honoring per-object prediction scores. Legacy fixed-score behavior (overwriting/ignoring input scores) is removed.

This delta modifies only COCO scoring behavior; all other base `detection-evaluator` requirements remain unchanged unless explicitly modified below.

## Requirements

## MODIFIED Requirements

### Requirement: COCO artifacts and scoring modes
This requirement applies when COCO artifacts/metrics are requested. Runs that compute only non-COCO metrics (e.g., f1ish-only) MAY accept unscored artifacts.

Milestone scope note:
- `confidence-postop` v1 computes confidence for `bbox_2d` only. As a result, `gt_vs_pred_scored.jsonl` will contain only bbox predictions (polygon predictions are dropped as unscorable).
- Therefore, COCO evaluation over `gt_vs_pred_scored.jsonl` SHOULD be bbox-only (do not run segm metrics) until polygon confidence is supported.

- The evaluator SHALL emit `coco_gt.json` and `coco_preds.json` with images, annotations, categories, and predictions using xywh bbox format; segmentation is included when polygons are present.
- Scores SHALL be taken from each prediction object’s `score` field and exported as the COCO `score` for ranking (AP/mAP).
- Each prediction object MUST include a finite numeric `score` satisfying `0.0 <= score <= 1.0`. Missing, non-numeric, `NaN`, infinite, or out-of-range scores MUST fail fast with actionable diagnostics (record index + object index).
- Input records MUST include `pred_score_source` (string, non-empty) and `pred_score_version` (int). This change intentionally rejects unscored legacy artifacts for COCO evaluation.
- When scores tie, export order SHALL remain stable using input order.
- Image ids in COCO export SHALL be derived deterministically from the JSONL index (0-based) to avoid collisions across shards.

#### Scenario: COCO export honors prediction scores
- **GIVEN** a prediction record with two bbox predictions with `score=0.9` and `score=0.1`
- **WHEN** the evaluator exports COCO predictions
- **THEN** `coco_preds.json` contains those same score values for the corresponding exported predictions.

#### Scenario: Missing score fails fast
- **GIVEN** a prediction object missing the `score` field (or with a non-numeric score)
- **WHEN** the evaluator attempts to export COCO predictions
- **THEN** evaluation terminates with a clear error identifying the offending record and object index.

#### Scenario: Unscored legacy artifact is rejected for COCO export
- **GIVEN** an input JSONL record missing `pred_score_source` / `pred_score_version`
- **WHEN** the evaluator attempts to export COCO predictions
- **THEN** evaluation terminates with a clear error explaining that scored artifacts are mandatory for COCO evaluation and the input is not score-provenanced.
