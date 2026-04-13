## ADDED Requirements

### Requirement: Inference can parse model-facing center-log-size bbox slots and emit canonical artifacts
The unified inference engine SHALL support the V1 model-facing
`center_log_size` bbox parameterization while keeping standardized artifacts
canonical.

Normative behavior:

- the supported inference-side key SHALL be `infer.bbox_format`,
- supported values SHALL be exactly:
  - `xyxy`
  - `center_log_size`
- the default SHALL be `xyxy`,
- `infer.bbox_format` applies only to `bbox_2d`,
- GT ingestion remains canonical `xyxy`,
- when `infer.bbox_format=center_log_size`, prediction parsing /
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
- `raw_output_json` MUST remain the verbatim model-facing payload.

#### Scenario: Center-log-size inference emits canonical standardized artifacts
- **WHEN** an inference run resolves `infer.bbox_format=center_log_size`
- **THEN** generation messages request center-log-size bbox output
- **AND** standardized prediction artifacts are emitted as canonical pixel-space
  `xyxy`.

#### Scenario: Raw output remains verbatim
- **WHEN** an inference run resolves `infer.bbox_format=center_log_size`
- **THEN** `raw_output_json` preserves the original model-facing generated bbox
  payload
- **AND** only standardized artifact views are canonicalized.

### Requirement: Confidence/scored post-processing remains unsupported for `center_log_size` in V1
The V1 inference path SHALL keep confidence post-processing and scored artifact
generation out of scope until their raw-bin contracts are updated for
`[cx, cy, u(w), u(h)]`.

Normative behavior:

- when `infer.bbox_format=center_log_size`, the following outputs/surfaces MUST
  be rejected or skipped with an explicit fail-fast diagnostic:
  - `pred_confidence.jsonl`
  - `gt_vs_pred_scored.jsonl`
  - `gt_vs_pred_scored_guarded.jsonl`
  - any post-op that assumes raw bbox bins are `(x1, y1, x2, y2)`
- canonical raw duplicate-control on already-standardized `xyxy` artifacts MAY
  remain available, including `gt_vs_pred_guarded.jsonl`
- `xyxy` inference remains responsible for the existing confidence/scored paths.

#### Scenario: Confidence post-op request fails fast
- **WHEN** an inference/post-processing run combines
  `infer.bbox_format=center_log_size` with a confidence/scored artifact request
- **THEN** the run fails fast with guidance that V1 supports canonical
  unscored artifacts only.

## MODIFIED Requirements

### Requirement: Unified inference CLI
The system SHALL provide a single inference entrypoint that is primarily
configured via YAML under `configs/` with a minimal CLI wrapper that accepts
`--config`.

Minimum YAML schema MUST include:
- the existing required inference keys
- `infer.object_field_order`
- `infer.bbox_format` (`xyxy | center_log_size`; default `xyxy`)

#### Scenario: YAML config requests center-log-size inference
- **WHEN** a user runs inference with `infer.bbox_format=center_log_size`
- **THEN** the run succeeds
- **AND** standardized artifacts remain canonical `xyxy`.
