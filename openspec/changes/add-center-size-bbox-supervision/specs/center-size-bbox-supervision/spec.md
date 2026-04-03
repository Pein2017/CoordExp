## ADDED Requirements

### Requirement: Center-wise supervision preserves the canonical external bbox contract
The system SHALL treat center-wise supervision as an internal loss-space decomposition only.

Normative behavior:
- Raw JSONL records, assistant payloads, CoordJSON rendering, rollout parsing, inference artifacts, and evaluator-facing artifacts MUST continue to use canonical `bbox_2d` arrays with `xyxy` semantics.
- This change MUST NOT introduce a new public geometry key or reinterpret canonical `bbox_2d` serialization in data or evaluation contracts.
- Matching, visualization, and evaluator consumers MUST continue to receive canonical `xyxy` boxes.
- The unchanged evaluation artifact split MUST remain explicit:
  - F1-ish evaluation may still consume `gt_vs_pred.jsonl`,
  - COCO evaluation still requires `gt_vs_pred_scored.jsonl`,
  - existing evaluator recovery paths that rely on `resolved_config.path` remain unchanged.

#### Scenario: Eval artifacts remain canonical `xyxy`
- **WHEN** center-wise supervision is enabled for training
- **THEN** emitted inference and evaluation artifacts still serialize boxes as canonical `bbox_2d: [x1, y1, x2, y2]`
- **AND** downstream evaluator consumers do not need a contract migration.

### Requirement: Center-wise supervision is config-first and opt-in across Stage-1 and pipeline-driven Stage-2
The system SHALL expose center-wise supervision only through existing YAML bbox geometry surfaces.

Normative behavior:
- Stage-1 MUST author the feature under `custom.bbox_geo`.
- Stage-2 AB MUST author the feature under `stage2_ab.pipeline.objective[*].config` for the `bbox_geo` module.
- Rollout-aligned Stage-2 MUST author the feature under `rollout_matching.pipeline.objective[*].config` for the `bbox_geo` module.
- The canonical authored knobs for the new mode are:
  - `parameterization`
  - `center_weight`
  - `size_weight`
- No new CLI flags or public runtime side channels may be required to enable the feature.

#### Scenario: Stage-1 enables center-wise supervision through nested bbox_geo config
- **WHEN** a Stage-1 config enables `custom.bbox_geo` and sets `parameterization: center_size`
- **THEN** the feature is enabled entirely through resolved YAML config
- **AND** training does not require new CLI flags.

### Requirement: BBox geometry supervision supports an opt-in center-wise loss-space mode
The system SHALL support an opt-in `center_size` loss-space mode for bbox geometry supervision while retaining canonical `xyxy` as the default outward representation.

Normative behavior:
- The bbox geometry configuration surface MUST support `parameterization: xyxy | center_size`.
- `xyxy` remains the default behavior and preserves the current regression semantics.
- Existing two-key bbox geometry configs that author only `smoothl1_weight` and `ciou_weight` MUST remain valid and MUST resolve to default `xyxy` regression.
- When `parameterization: center_size` is enabled, the regression term MUST derive `(cx, cy, log_w, log_h)` from decoded predicted and GT boxes that remain canonical `xyxy`.
- The center-wise loss tuple MUST be `(cx, cy, log_w, log_h)` derived from canonical `xyxy`, using a positive epsilon guard for width and height before log conversion.
- The same forward pass MUST still compute CIoU on canonical `xyxy` boxes.
- The published decoded-box state consumed by downstream loss modules MUST remain canonical `xyxy`, including predicted boxes, target boxes, group weights, and coord-state tensors already owned by the bbox geometry path.

#### Scenario: Default mode preserves current behavior
- **WHEN** `bbox_geo` is configured without an explicit center-size parameterization override
- **THEN** bbox regression uses the existing `xyxy` behavior
- **AND** current configs remain behaviorally backward-compatible.

#### Scenario: Legacy two-key bbox_geo config remains valid
- **WHEN** a config authors only `smoothl1_weight` and `ciou_weight` for `bbox_geo`
- **THEN** config validation still succeeds
- **AND** the resolved behavior is default `xyxy` regression.

#### Scenario: Center-size mode changes only loss-space terms
- **WHEN** `bbox_geo.config.parameterization` is `center_size`
- **THEN** the regression term is computed from `(cx, cy, log_w, log_h)` derived from decoded and target boxes that remain canonical `xyxy`
- **AND** CIoU continues to consume canonical `xyxy` boxes in the same step.

### Requirement: Center-size bbox regression stays semantically aligned across Stage-1 and Stage-2 surfaces
The system SHALL keep the bbox regression decomposition for the `center_size` mode semantically aligned across the Stage-1 and Stage-2 training surfaces touched by this change.

Normative behavior:
- When both Stage-1 and Stage-2 implement `parameterization: center_size`, they MUST apply the same regression decomposition and mean-like reduction semantics for equivalent canonical decoded predicted and target `xyxy` boxes plus optional box weights.
- The aligned semantics MUST cover:
  - derivation of `(cx, cy, log_w, log_h)` from canonical `xyxy`,
  - epsilon-guarded width and height handling before log conversion,
  - mean-like reduction for the regression branch.
- Stage-specific wrappers MAY continue to own data extraction, batching or masking, metric emission, and published pipeline state.
- A shared internal helper is the recommended implementation path for this alignment, but the contract does not require one specific helper shape.
- This maintainability cleanup MUST NOT rename public modules such as `bbox_geo`, `bbox_size_aux`, or `coord_reg`, and it MUST NOT change the outward bbox contract.

#### Scenario: Stage-1 and Stage-2 do not fork center-size regression semantics
- **WHEN** both Stage-1 and Stage-2 support `bbox_geo.parameterization=center_size`
- **THEN** they apply the same internal regression decomposition and reduction semantics on canonical decoded boxes
- **AND** only wrapper-level extraction and logging behavior may differ.

### Requirement: Center and size contributions are independently weighted in loss-space
The system SHALL allow center and size components of bbox regression to carry different relative strength in loss-space.

Normative behavior:
- The bbox geometry config surface MUST support positive-or-zero `center_weight` and `size_weight` knobs when `parameterization: center_size` is used.
- The center component covers `cx` and `cy`.
- The size component covers `log_w` and `log_h`.
- The total regression term in `center_size` mode MUST be the weighted combination of the center and size components, normalized in a mean-like way.
- The intended lightweight default for this mode is center-strong and size-soft supervision, with CIoU still carried separately on `xyxy`.
- When both `center_weight` and `size_weight` are zero for `center_size` mode, configuration MUST fail fast with actionable guidance.

#### Scenario: Center-strong supervision is configurable
- **WHEN** `bbox_geo.config.parameterization` is `center_size`
- **AND** `center_weight` is greater than `size_weight`
- **THEN** the regression loss emphasizes center errors more strongly than size errors
- **AND** no parser or evaluator contract changes are required.

#### Scenario: Size-light supervision remains valid
- **WHEN** `bbox_geo.config.parameterization` is `center_size`
- **AND** `center_weight` is positive and `size_weight` is zero
- **THEN** center-size regression remains a valid configuration
- **AND** the implementation still computes CIoU on canonical `xyxy` boxes for the same decoded boxes.
