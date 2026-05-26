# retired-lightweight-pseudopositive-v1 Specification

## Status

Retired. The active Stage-2 rollout-correction contract removed the
clean-prefix pseudo-positive extension and its public split-channel namespace.

## Requirements

### Requirement: Pseudo-positive clean-prefix knobs are rejected

Active Stage-2 configs MUST reject old pseudo-positive clean-prefix knobs
instead of interpreting them as aliases.

#### Scenario: pseudo-positive namespace is authored

- **WHEN** a config includes
  `stage2_rollout_correction.correction.pseudo_positive`
- **THEN** config loading fails fast
- **AND** the error says the pseudo-positive clean-prefix knob has been removed
  from unified Stage-2 rollout correction.

### Requirement: Active replacement is residual correction

Current Stage-2 configs MUST express correction behavior through the
`residual_set_correction` objective and rollout-correction runtime knobs, not
through pseudo-positive clean-prefix supervision.
