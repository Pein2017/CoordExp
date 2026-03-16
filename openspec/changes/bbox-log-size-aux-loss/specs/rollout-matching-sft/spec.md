# rollout-matching-sft Specification (Delta)

## MODIFIED Requirements

### Requirement: Rollout pipeline module configs are strict and canonical (no aliases)
Rollout pipeline module configs MUST be strict and MUST reject unknown keys and legacy alias keys.

Normative behavior:
- `bbox_geo.config` MUST accept only:
  - `smoothl1_weight`
  - `ciou_weight`
- `bbox_size_aux.config` MUST accept only:
  - `log_wh_weight`
  - `log_area_weight`
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

#### Scenario: Alias key in rollout module config fails fast
- **WHEN** `rollout_matching.pipeline.objective[*].name=bbox_size_aux`
- **AND** the module config contains `bbox_smoothl1_weight`
- **THEN** configuration parsing fails fast
- **AND** the error indicates the canonical `bbox_size_aux.config.*` key family
  must be used.

## ADDED Requirements

### Requirement: Rollout-aligned training SHALL reuse the same decoded-box size auxiliaries through `bbox_size_aux`
The rollout-aligned trainer MUST reuse the same decoded-box size auxiliaries
through `bbox_size_aux` that Stage-2 AB uses.

When `custom.trainer_variant: stage2_rollout_aligned`, the rollout-aligned
teacher-forcing path SHALL support the same optional decoded-box size
auxiliaries through `bbox_size_aux` that Stage-2 AB uses.

Normative behavior:

- matched decoded-box size auxiliaries MUST reuse the shared `bbox_size_aux`
  module
  and shared decoded-box helper implementation,
- the rollout-aligned path MUST NOT fork a second geometry implementation just
  for `log_wh`, `log_area`, or oversize penalty,
- `bbox_size_aux` MUST consume the current bbox coord slots in the same public
  `bbox_2d: [x1, y1, x2, y2]` order used elsewhere in the stack,
- the default rollout-aligned behavior SHOULD mirror the conservative Stage-2
  default:
  - small `log_wh` weight only when explicitly enabled,
  - `log_area` and `oversize` off unless explicitly authored.

#### Scenario: Rollout-aligned matched geometry can include log-size aux
- **GIVEN** a rollout-aligned config with `bbox_size_aux.config.log_wh_weight > 0`
- **WHEN** rollout-context matched geometry loss is computed
- **THEN** the matched log-width/log-height auxiliary contributes through the
  shared `bbox_size_aux` implementation
- **AND** no trainer-specific duplicate geometry path is required.
