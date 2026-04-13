## ADDED Requirements

### Requirement: Stage-1 center-log-size experiment uses hard coord-token CE plus positive gating
The Stage-1 bbox-parameterization experiment SHALL keep the coord-token loss
surface minimal by using pure CE on bbox coord slots while retaining gating
terms that separate coord and non-coord token families.

Normative behavior:

- for this V1 change, the Stage-1 canonical config surface extends
  `custom.coord_soft_ce_w1` with:
  - `text_gate_weight` (float; default `1.0`)
- when `custom.bbox_format=center_log_size`, Stage-1 bbox coord slots MUST be
  supervised through the `custom.coord_soft_ce_w1` surface in a pure-CE profile:
  - `enabled` MUST be `true`
  - `ce_weight` MUST be `> 0`
  - `soft_ce_weight` MUST be `0`
  - `w1_weight` MUST be `0`
- gating MUST remain enabled for the experiment:
  - `gate_weight` continues to supervise coord-vocab mass at coord positions
  - `text_gate_weight` SHALL supervise low coord-vocab mass at supervised
    `struct|desc` positions
- `gate_weight` and `text_gate_weight` MUST both be `> 0`,
- soft-target shaping knobs MUST remain at compatibility defaults in this
  profile:
  - `temperature` MUST be `1.0`
  - `target_sigma` MUST be `2.0`
  - `target_truncate` MUST be `null`
- standard CE on non-coord tokens remains unchanged,
- this V1 change MUST emit stable observability for both gates:
  - `coord_softce_w1/gate`
  - `coord_softce_w1/text_gate`
  - `coord_diag/gate`
  - `coord_diag/text_gate`
- this V1 change MUST reject decoded-box regression losses.

#### Scenario: Stage-1 center-log-size runs in pure-CE mode
- **GIVEN** `custom.bbox_format=center_log_size`
- **WHEN** config validation runs for the V1 experiment
- **THEN** `ce_weight` is active for coord slots
- **AND** `custom.coord_soft_ce_w1.enabled` is `true`
- **AND** `soft_ce_weight` and `w1_weight` are both zero.

#### Scenario: Gating remains active across coord and non-coord positions
- **GIVEN** `custom.bbox_format=center_log_size`
- **WHEN** the loss is computed
- **THEN** coord positions receive coord-gate supervision
- **AND** supervised `struct|desc` positions receive text-gate supervision.

#### Scenario: Stage-1 text gate is explicitly configurable
- **GIVEN** `custom.coord_soft_ce_w1.text_gate_weight`
- **WHEN** config validation runs for the V1 experiment
- **THEN** the key is accepted as part of the Stage-1 coord supervision surface
- **AND** negative values fail fast.

#### Scenario: Non-pure coord settings fail fast
- **GIVEN** `custom.bbox_format=center_log_size`
- **WHEN** config validation sees `soft_ce_weight > 0`, `w1_weight > 0`, or
  non-default soft-target shaping knobs
- **THEN** validation fails fast with guidance that the V1 profile is hard-CE
  only.

### Requirement: Regression-style bbox losses are rejected in the V1 experiment
The V1 center-log-size experiment SHALL defer regression-style bbox losses so
the parameterization can be evaluated under minimal loss complexity.

Normative behavior:

- the V1 path MUST reject:
  - `custom.bbox_geo.enabled=true`
  - `custom.bbox_size_aux.enabled=true`
  - W1 bbox-slot supervision
  - soft-CE bbox-slot supervision
- future changes MAY reintroduce those losses after the Stage-1 CE-only
  experiment is evaluated, but any future size-slot soft labels or
  distributional targets MUST be defined in the shared `u(*)` log-size domain
  rather than on raw width/height.

#### Scenario: V1 does not depend on decoded-box regression
- **WHEN** the Stage-1 center-log-size experiment is enabled
- **THEN** training remains well-defined without `bbox_geo` or `bbox_size_aux`
- **AND** the parameterization experiment is evaluated under CE-plus-gating
  supervision only.

#### Scenario: Decoded-box regression surfaces fail fast
- **GIVEN** `custom.bbox_format=center_log_size`
- **WHEN** config validation sees `custom.bbox_geo.enabled=true` or
  `custom.bbox_size_aux.enabled=true`
- **THEN** validation fails fast with guidance that V1 is limited to
  hard-CE-plus-gating Stage-1 supervision.
