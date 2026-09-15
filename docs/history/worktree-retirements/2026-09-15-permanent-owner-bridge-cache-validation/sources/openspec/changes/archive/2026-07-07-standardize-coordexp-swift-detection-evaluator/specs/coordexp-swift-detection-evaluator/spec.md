## ADDED Requirements

### Requirement: Parser owns generated-text salvage
The system SHALL perform generated-text prediction salvage only in the inference
parser before scoring or evaluation. The evaluator MUST NOT parse raw decode
text, recover dropped spans, reinterpret malformed spans, or mutate prediction
objects from parser diagnostics.

#### Scenario: Valid span plus malformed span
- **WHEN** generated text contains one valid compact object span and one
  malformed object span
- **THEN** the parser records the valid prediction, records the malformed span
  in diagnostics, and the evaluator consumes only the scored prediction objects
  already present in `gt_vs_pred_scored.jsonl`

#### Scenario: All generated spans dropped
- **WHEN** parser output contains no valid prediction objects after salvage/drop
  handling
- **THEN** the scored artifact row remains present with `pred: []`
- **AND** the evaluator does not reparse raw decode text to recover predictions

### Requirement: Scoring owns score-bearing prediction eligibility
The system SHALL treat `gt_vs_pred_scored.jsonl` as the metric-bearing
prediction input. Each prediction used for official mAP/mRecall MUST have a
finite `score` in `[0.0, 1.0]`, structured row-local `pred_score_source`, and a
supported integer `pred_score_version`. For the V1 selected-token scorer,
`pred_score_source` MUST include `kind`, `row_id`, `object_span_id`,
selected-token evidence, `selected_count`, and `score_policy_fingerprint`. The
row-local source MUST match the scored row id, the prediction object span id,
and the provenance sidecar score-policy fingerprint. The evaluator MUST fail
before metric computation when score-bearing prediction provenance is missing,
malformed, unsupported, or policy-mismatched.

#### Scenario: Complete scored prediction
- **WHEN** a scored prediction has finite score, row-local score source, and
  integer score version
- **THEN** the evaluator may include it in COCO prediction conversion

#### Scenario: Missing score provenance
- **WHEN** a scored prediction lacks `pred_score_source` or
  `pred_score_version`
- **THEN** evaluator consumption fails before metric computation

#### Scenario: Score provenance policy mismatch
- **WHEN** a scored prediction row-local `score_policy_fingerprint` differs
  from the scored artifact provenance sidecar
- **THEN** evaluator consumption fails before metric computation

#### Scenario: Score provenance row mismatch
- **WHEN** a scored prediction row-local `row_id` or `object_span_id` does not
  match the evaluated row and prediction object
- **THEN** evaluator consumption fails before metric computation

#### Scenario: Invalid score
- **WHEN** a scored prediction score is NaN, Inf, below `0.0`, or above `1.0`
- **THEN** evaluator consumption fails before metric computation

### Requirement: Evaluator validates artifact binding before metrics
The evaluator SHALL validate `gt_vs_pred_scored.jsonl` together with
`gt_vs_pred_scored.jsonl.provenance.json` before metric computation. The
provenance MUST bind the scored artifact to the raw artifact, row identity,
prompt policy, decode or generation policy, model identity, processor identity,
template identity, parser policy, and score policy. Raw and scored rows MUST
preserve row order, row id, row index, example id when present, image identity,
image dimensions, and GT payload exactly; scoring may add or filter prediction
score fields but MUST NOT rewrite immutable raw row fields. The evaluator MUST
read `gt_vs_pred.jsonl` only for raw parser/drop normalization counters and row
parity validation.

#### Scenario: Provenance binding matches
- **WHEN** raw and scored artifacts exist, SHA bindings match, row identities
  match, and required provenance fields are present
- **THEN** evaluator normalization and metric computation may proceed

#### Scenario: Raw and scored row mismatch
- **WHEN** raw and scored artifact row counts or row ids differ
- **THEN** evaluator consumption fails before metric computation

#### Scenario: Raw and scored payload mismatch
- **WHEN** raw and scored artifact rows have the same `row_id` but differ in
  image path, image dimensions, example id, row index, or GT payload
- **THEN** evaluator consumption fails before metric computation

#### Scenario: Raw artifact SHA mismatch
- **WHEN** provenance raw SHA does not match `gt_vs_pred.jsonl`
- **THEN** evaluator consumption fails before metric computation

### Requirement: Empty predictions remain false negatives
The evaluator SHALL preserve one evaluation image row per valid artifact row.
Prediction presence MUST NOT determine whether a row participates in official
metrics. Rows with valid image identity, dimensions, GT, and provenance but
`pred: []` SHALL enter COCO evaluation as images with zero detections.

#### Scenario: Empty prediction row with GT
- **WHEN** a scored row has valid GT and image metadata but `pred: []`
- **THEN** the row remains in `coco_gt.json`
- **AND** no prediction entry is emitted for that row in `coco_predictions.json`
- **AND** official recall/AP reflect the missed GT object

#### Scenario: Parser diagnostic-only row
- **WHEN** raw parser metadata says all spans were dropped
- **THEN** the evaluator counts the parser status diagnostically
- **AND** the row still contributes false negatives when GT is present

### Requirement: COCO bbox coordinate normalization
The evaluator SHALL convert CoordExp-Swift GT bbox values from norm1000 `xyxy`
coordinate-bin space into per-image pixel `xyxy` before COCO conversion.
Prediction `bbox` values from scored Swift inference artifacts MUST already be
parser-normalized pixel `xyxy`; the evaluator MUST NOT apply a second
norm1000-to-pixel conversion to predictions.

#### Scenario: Non-1000 image dimensions
- **WHEN** a GT object has bbox `[100, 200, 300, 400]` and image dimensions
  `1248x832`
- **AND** the corresponding scored prediction is the parser-normalized pixel
  box for the same coord tokens
- **THEN** `coco_gt.json` and `coco_predictions.json` use the same pixel-space
  bbox
- **AND** the perfect prediction reports `mAP` and `mRecall` as `1.0` within
  numeric tolerance

### Requirement: COCO-80 category normalization
The evaluator SHALL use a canonical COCO-80 closed-class registry that is the
V1 source of truth for the rebuilt Swift prompt/eval path. Category text
normalization MUST be limited to lowercasing and whitespace collapse in V1.
The evaluator MUST NOT perform semantic remapping, alias expansion, or
description embedding matching in V1. Unknown GT categories MUST fail before
metric computation. Unknown prediction categories MUST be excluded from COCO
prediction conversion and counted in metrics.

#### Scenario: Canonical category match
- **WHEN** GT and prediction descriptions normalize to a COCO-80 class name
- **THEN** they are converted to the corresponding COCO category id

#### Scenario: Unknown GT category
- **WHEN** a GT object description is not in the canonical COCO-80 registry
- **THEN** evaluator consumption fails before metric computation

#### Scenario: Unknown prediction category
- **WHEN** a prediction description is not in the canonical COCO-80 registry
- **THEN** that prediction is excluded from `coco_predictions.json`
- **AND** `metrics.json` increments `unknown_category_pred_count`

### Requirement: Official COCO bbox metrics
The evaluator SHALL compute official COCO bbox metrics with `pycocotools` for
V1. `metrics.json` MUST include `mAP` for bbox AP averaged across IoU
thresholds `.50:.95`, `mAP_50`, `mAP_75`, and `mRecall` for bbox AR@100. It
MUST also preserve raw COCO bbox aliases such as `bbox_AP`, `bbox_AP50`, and
`bbox_AR100`.

#### Scenario: Perfect single-object prediction
- **WHEN** one scored prediction exactly matches one GT object with the correct
  COCO-80 category
- **THEN** `metrics.json` reports `mAP`, `mAP_50`, and `mRecall` as `1.0`
  within numeric tolerance

#### Scenario: No scored predictions
- **WHEN** valid GT exists but `coco_predictions.json` is empty
- **THEN** `metrics.json` reports headline `mAP` and `mRecall` as `0.0`

### Requirement: Evaluator artifacts
The evaluator SHALL write `metrics.json`, `coco_gt.json`, and
`coco_predictions.json` under the requested output directory. `metrics.json`
MUST include benchmark metric identity, row/object counts, parser/drop
normalization counters, category normalization counters, official COCO metrics,
and references to the COCO conversion artifacts. V1 is aggregate-only and SHALL
NOT claim `per_class.csv` or `per_image.json` emission from this direct Swift
evaluator.

#### Scenario: Successful evaluation output
- **WHEN** evaluator consumption and metric computation succeed
- **THEN** `metrics.json`, `coco_gt.json`, and `coco_predictions.json` exist in
  the output directory

#### Scenario: Normalization counters present
- **WHEN** raw parser diagnostics include dropped predictions or parse statuses
- **THEN** `metrics.json` includes parser status counts and raw dropped
  prediction counts

### Requirement: Direct artifact evaluator CLI
The repository SHALL provide a direct offline evaluator command for existing
inference artifact directories:

```bash
python scripts/evaluate_detection.py --artifact-dir RUN_DIR --out-dir RUN_DIR/eval
```

The CLI MUST also accept `--pred-jsonl` as a compatibility alias for a direct
path to `gt_vs_pred_scored.jsonl`, resolving the artifact directory from its
parent. The CLI MUST NOT require re-running inference. The CLI SHALL print the
metrics path plus one compact JSON metrics summary on success, and SHALL report
contract failures as structured stderr without a Python traceback.

#### Scenario: Artifact directory evaluation
- **WHEN** the user runs the evaluator with `--artifact-dir` pointing at a valid
  scored inference run
- **THEN** the evaluator writes official metrics and COCO conversion artifacts
  under `--out-dir`

#### Scenario: Scored artifact path alias
- **WHEN** the user runs the evaluator with `--pred-jsonl` pointing at
  `gt_vs_pred_scored.jsonl`
- **THEN** the evaluator resolves the artifact directory from the parent path
  and produces the same outputs as `--artifact-dir`
