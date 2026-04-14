## ADDED Requirements

### Requirement: Inference can parse model-facing cxcy-logw-logh bbox slots and emit canonical artifacts
The unified inference engine SHALL support the V1 model-facing
`cxcy_logw_logh` bbox parameterization while keeping standardized artifacts
canonical.

Normative behavior:

- the supported inference-side key SHALL be `infer.bbox_format`,
- supported values SHALL be exactly:
  - `xyxy`
  - `cxcy_logw_logh`
- the default SHALL be `xyxy`,
- `infer.bbox_format` applies only to `bbox_2d`,
- GT ingestion remains canonical `xyxy`,
- when `infer.bbox_format=cxcy_logw_logh`, prediction parsing /
  standardization MUST:
  - interpret the four bbox slots as `[cx, cy, u(w), u(h)]`
  - invert the shared log-size chart on width/height
  - canonicalize predictions to pixel-space `xyxy` before standardized artifact
    emission
- standardized artifacts MUST remain canonical `xyxy`, including:
  - `gt_vs_pred.jsonl`
  - `vis_resources/gt_vs_pred.jsonl`
  - visualization/eval-ready copies derived from the canonical standardized
    artifact
- `raw_output_json` MUST remain the parsed best-effort raw payload produced by
  the shared salvage/parser path.

#### Scenario: Center-log-size inference emits canonical standardized artifacts
- **WHEN** an inference run resolves `infer.bbox_format=cxcy_logw_logh`
- **THEN** generation messages request cxcy-logw-logh bbox output
- **AND** standardized prediction artifacts are emitted as canonical pixel-space
  `xyxy`.

#### Scenario: Raw output remains parsed best-effort
- **WHEN** an inference run resolves `infer.bbox_format=cxcy_logw_logh`
- **THEN** `raw_output_json` preserves the parsed model-facing generated bbox
  payload
- **AND** only standardized artifact views are canonicalized.

### Requirement: Confidence post-processing remains unsupported for `cxcy_logw_logh` in V1
The V1 inference path SHALL keep confidence post-processing out of scope until
its raw-bin contract is updated for `[cx, cy, u(w), u(h)]`.

Normative behavior:

- when `infer.bbox_format=cxcy_logw_logh`, the following outputs/surfaces MUST
  be rejected or skipped with an explicit fail-fast diagnostic:
  - `pred_confidence.jsonl`
  - any post-op that assumes raw bbox bins are `(x1, y1, x2, y2)`
- official score-aware evaluation MAY still consume:
  - `gt_vs_pred_scored.jsonl`
  - `gt_vs_pred_scored_guarded.jsonl`
  when those artifacts are materialized from canonical standardized predictions
  via a deterministic constant-score compatibility policy
- canonical raw duplicate-control on already-standardized `xyxy` artifacts MAY
  remain available, including `gt_vs_pred_guarded.jsonl`
- `xyxy` inference remains responsible for the existing confidence post-op path.

#### Scenario: Confidence post-op request fails fast
- **WHEN** an inference/post-processing run combines
  `infer.bbox_format=cxcy_logw_logh` with a confidence-postop request
- **THEN** the run fails fast with guidance that V1 does not support
  confidence-based score reconstruction for cxcy-logw-logh bbox slots.

#### Scenario: Official eval uses a constant-score compatibility artifact
- **WHEN** an inference/eval run resolves `infer.bbox_format=cxcy_logw_logh`
- **AND** official score-aware metrics are requested
- **THEN** the run may materialize `gt_vs_pred_scored.jsonl` from canonical
  standardized predictions using a deterministic constant-score provenance
- **AND** the evaluator consumes that scored compatibility artifact instead of a
  confidence-reconstructed artifact.

## MODIFIED Requirements

### Requirement: Unified inference CLI
The system SHALL provide a single inference entrypoint that is primarily
configured via YAML under `configs/` with a minimal CLI wrapper that accepts
`--config`.

Minimum YAML schema MUST include:
- the existing required inference keys
- `infer.object_field_order`
- `infer.bbox_format` (`xyxy | cxcy_logw_logh`; default `xyxy`)

#### Scenario: YAML config requests cxcy-logw-logh inference
- **WHEN** a user runs inference with `infer.bbox_format=cxcy_logw_logh`
- **THEN** the run succeeds
- **AND** standardized artifacts remain canonical `xyxy`.
