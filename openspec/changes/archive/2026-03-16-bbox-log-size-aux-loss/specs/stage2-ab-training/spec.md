# stage2-ab-training Specification (Delta)

## MODIFIED Requirements

### Requirement: Stage-2 AB module configs are strict and canonical (no aliases)
Stage-2 AB pipeline module configs MUST be strict and MUST reject unknown keys and legacy alias keys.

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
  - `a1_log_wh_weight`
  - `a1_log_area_weight`
  - `a1_oversize_penalty_weight`
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

#### Scenario: Alias key in module config fails fast
- **WHEN** `stage2_ab.pipeline.objective[*].name=bbox_size_aux`
- **AND** the module config contains `bbox_smoothl1_weight`
- **THEN** configuration parsing fails fast
- **AND** the error indicates the canonical `bbox_size_aux.config.*` key family
  must be used instead.

## ADDED Requirements

### Requirement: Stage-2 AB can add matched decoded-box size auxiliaries through `bbox_size_aux`
Stage-2 AB SHALL support optional decoded-box size auxiliaries on the existing
matched geometry path without changing bbox parameterization or decode format.

Normative behavior:

- when `bbox_size_aux.config.log_wh_weight > 0`, the trainer MUST add matched
  log-width/log-height supervision on canonicalized decoded boxes,
- when `bbox_size_aux.config.log_area_weight > 0`, the trainer MUST add matched
  log-area supervision on canonicalized decoded boxes,
- when `bbox_size_aux.config.oversize_penalty_weight > 0`, the trainer MAY add the
  thresholded oversize penalty on decoded boxes for the same context,
- Channel-A and Channel-B applicability MUST remain controlled by the authored
  `channels` field on the `bbox_size_aux` module entry,
- `bbox_size_aux` MUST remain separate from `bbox_geo` in the authored pipeline
  so the new size loss is an independently removable plugin module,
- `bbox_size_aux` MUST consume the current four bbox coord slots in the existing
  `xyxy` order rather than introducing a new bbox expression,
- when any `bbox_size_aux.config.a1_*` weight is non-zero, Channel-A MUST also
  support the plugin on the optional A1 anchor forward,
- the default canonical Stage-2 profile behavior SHOULD enable only the matched
  `log_wh` term at a small weight and keep `log_area` / `oversize` off.

#### Scenario: Channel-A matched geometry can include log-size aux
- **GIVEN** a Stage-2 AB config with `bbox_size_aux.channels: [A]`
- **AND** `bbox_size_aux.config.log_wh_weight > 0`
- **WHEN** Channel-A computes matched geometry loss from decoded boxes
- **THEN** the matched log-width/log-height auxiliary contributes through the
  `bbox_size_aux` plugin
- **AND** existing SmoothL1 / CIoU terms remain intact.

#### Scenario: Channel-B matched rollout geometry can include log-size aux
- **GIVEN** a Stage-2 AB config with `bbox_size_aux.channels: [B]`
- **AND** `bbox_size_aux.config.log_wh_weight > 0`
- **WHEN** Channel-B computes matched rollout geometry loss from decoded boxes
- **THEN** the matched log-width/log-height auxiliary contributes on the same
  matched-clean + FN supervision set
- **AND** unmatched clean extras remain outside positive geometry supervision.

#### Scenario: Channel-A A1 anchor path can include log-size aux immediately
- **GIVEN** a Stage-2 AB config with `bbox_size_aux.channels: [A]`
- **AND** `bbox_size_aux.config.a1_log_wh_weight > 0`
- **WHEN** the Channel-A anchor forward (`A1`) is supervised
- **THEN** `bbox_size_aux` contributes `bbox_log_wh` on the A1 decoded-box path
- **AND** the A1 term remains explicitly opt-in through the `a1_*` weights.
