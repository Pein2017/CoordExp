## ADDED Requirements

### Requirement: Trainer metric keys expose promoted teacher-forcing type loss
The trainer metrics contract SHALL expose the promoted Stage-1
teacher-forcing type-family loss with stable unsuffixed metric keys.

Normative metrics:

- `teacher_forcing/loss/token_type_mass`
- `teacher_forcing/loss/token_type_mass/contribution`
- `teacher_forcing/type/schema_mass_at_schema`
- `teacher_forcing/type/coord_mass_at_coord`
- `teacher_forcing/type/desc_mass_at_desc`
- `teacher_forcing/type/stop_mass_at_stop`

Normative behavior:

- `teacher_forcing/loss/token_type_mass` MUST be the raw mean type-family loss
  over atoms where `token_type_mass` is active.
- `teacher_forcing/loss/token_type_mass/contribution` MUST be the raw mean
  multiplied by `objective.terms.token_type_mass.weight`.
- Family mass metrics MUST report the active-family softmax mass at atoms whose
  target family matches the metric suffix.
- These six promoted keys MUST be stable flattened metric keys emitted through
  ordinary `MetricEvent` weighted means. They MUST use active atom counts as
  denominators.
- New metric keys for this term MUST NOT use `_weighted` suffixes.
- Missing denominator groups MUST remain absent rather than being reported as
  misleading zero-valued ratios.

#### Scenario: Raw and contribution metrics are distinct
- **GIVEN** `objective.terms.token_type_mass.enabled: true`
- **AND** `objective.terms.token_type_mass.weight: 0.2`
- **WHEN** a teacher-forcing batch emits metrics
- **THEN** `teacher_forcing/loss/token_type_mass` reports the unweighted raw
  mean
- **AND** `teacher_forcing/loss/token_type_mass/contribution` reports `0.2`
  times that mean.

#### Scenario: Weighted suffix is not emitted
- **WHEN** `token_type_mass` metrics are flattened or logged
- **THEN** no metric key ending in `_weighted` is emitted for the promoted
  type-family term.
