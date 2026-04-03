## MODIFIED Requirements

### Requirement: Rollout pipeline module configs are strict and canonical (no aliases)
Rollout pipeline module configs MUST be strict and MUST reject unknown keys and legacy alias keys.

Normative behavior:
- `bbox_geo.config` MUST accept only:
  - `smoothl1_weight`
  - `ciou_weight`
  - `parameterization`
  - `center_weight`
  - `size_weight`
- `bbox_size_aux.config` MUST accept only:
  - `log_wh_weight`
  - `oversize_penalty_weight`
  - `oversize_area_frac_threshold`
  - `oversize_log_w_threshold`
  - `oversize_log_h_threshold`
  - `eps`
- `coord_reg.config` MUST accept only canonical keys, including:
  - `coord_ce_weight`
  - `soft_ce_weight`
  - `w1_weight`
  - `coord_gate_weight`
  - `text_gate_weight`
  - `temperature`
  - `target_sigma`
  - `target_truncate`
- Legacy alias keys (e.g., `bbox_smoothl1_weight`, `coord_soft_ce_weight`, `coord_w1_weight`) MUST be rejected.
- The new `bbox_geo.config` keys are additive:
  - existing two-key configs that author only `smoothl1_weight` and `ciou_weight` MUST remain valid,
  - omitted `parameterization` MUST resolve to the default `xyxy` behavior,
  - `center_weight` and `size_weight` MUST remain optional unless `parameterization: center_size` is explicitly authored.

#### Scenario: Alias key in rollout module config fails fast
- **WHEN** `rollout_matching.pipeline.objective[*].name=bbox_size_aux`
- **AND** the module config contains `bbox_smoothl1_weight`
- **THEN** configuration parsing fails fast
- **AND** the error indicates the canonical `bbox_size_aux.config.*` key family
  must be used.

#### Scenario: Rollout-aligned bbox_geo can opt into center-size supervision
- **WHEN** `rollout_matching.pipeline.objective[*].name=bbox_geo`
- **AND** the module config authors `parameterization: center_size`, `center_weight`, and `size_weight`
- **THEN** strict config validation accepts the config
- **AND** rollout-aligned training can use the same center-size supervision mode without changing the rollout parser contract.

#### Scenario: Existing rollout bbox_geo config remains valid
- **WHEN** `rollout_matching.pipeline.objective[*].name=bbox_geo`
- **AND** the module config authors only `smoothl1_weight` and `ciou_weight`
- **THEN** strict config validation still accepts the config
- **AND** the resolved behavior remains default `xyxy`.
