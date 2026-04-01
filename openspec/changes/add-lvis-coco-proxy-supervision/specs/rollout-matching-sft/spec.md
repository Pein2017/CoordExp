# rollout-matching-sft Specification (Delta)

## MODIFIED Requirements

### Requirement: Rollout pipeline module configs are strict and canonical (no aliases)
Rollout-aligned Stage-2 SHALL support metadata-driven object weighting through
canonical module config keys only.

Normative behavior:

- `token_ce.config` MUST accept `object_weight_mode`,
- `bbox_geo.config` MUST accept `object_weight_mode`,
- `coord_reg.config` MUST accept `object_weight_mode`,
- `object_weight_mode` MUST support:
  - `none`
  - `metadata`
- unknown object-weight config keys or alias keys MUST fail fast.

#### Scenario: Rollout-aligned config rejects unsupported object weight mode
- **WHEN** a rollout-aligned pipeline module config authors an unsupported
  `object_weight_mode`
- **THEN** configuration parsing fails fast before trainer initialization.

### Requirement: Rollout-aligned Stage-2 supports metadata-driven proxy weighting without changing rendered targets
Rollout-aligned Stage-2 SHALL support proxy-supervision weighting as a
training-time concern rather than a prompt-format change.

Normative behavior:

- metadata-driven object weighting MUST consume object-local desc / coord
  weights from the shared context,
- rendered object syntax in the teacher-forced target MUST remain unchanged,
- structure CE remains globally supervised even when plausible proxy objects are
  present,
- missing metadata in `object_weight_mode=metadata` MUST fall back to weight
  `1.0`.

#### Scenario: Rollout-aligned metadata weighting preserves target syntax
- **WHEN** rollout-aligned Stage-2 trains on augmented COCO+LVIS proxy records
- **THEN** the rendered target format remains ordinary CoordJSON
- **AND** proxy weighting changes only desc / coord supervision strength.
