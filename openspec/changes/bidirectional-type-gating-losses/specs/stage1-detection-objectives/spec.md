# stage1-detection-objectives Delta

## ADDED Requirements

### Requirement: Hard-SFT token-type mass is configured through objective terms

The promoted Stage-1 hard-SFT detection stack SHALL configure the stable
token-family objective through `objective.terms.token_type_mass`.

Normative behavior:

- the promoted hard-SFT stack MUST author
  `objective.terms.token_type_mass.enabled: true`;
- `objective.terms.token_type_mass.enabled: true` MUST select the stable
  bidirectional exclusive type-family objective over `schema`, `coord`, `desc`,
  and `stop`;
- `objective.terms.token_type_mass.weight` MUST scale only this additive
  type-family objective contribution;
- `objective.terms.token_type_mass` MUST NOT require or accept a stable `mode`
  key to choose type-family behavior;
- `conditional_valid_set_likelihood` and `within_valid_coverage` MUST remain
  disabled for the requested hard-SFT stack unless a separate promoted contract
  enables them;
- pure-CE or no-type-loss comparator behavior MUST be represented as an
  explicit ablation/comparator outside the promoted hard-SFT type-gating stack,
  not by changing the meaning of `enabled: true`.

#### Scenario: Enabled token_type_mass selects the stable four-family objective

- **GIVEN** a Stage-1 detection config for the promoted hard-SFT type-gating
  stack
- **AND** `objective.terms.token_type_mass.enabled: true`
- **WHEN** config parsing and objective resolution run
- **THEN** the resolved objective includes an additive token-type mass term over
  `schema`, `coord`, `desc`, and `stop`
- **AND** no `mode` key is required to select that behavior.

#### Scenario: Token-type mode is not a stable config knob

- **GIVEN** a Stage-1 detection config with
  `objective.terms.token_type_mass.mode`
- **WHEN** strict config validation runs
- **THEN** validation fails fast
- **AND** the error points authors to the stable
  `objective.terms.token_type_mass.enabled` and `weight` fields.

#### Scenario: Valid-set and coverage terms stay disabled for hard-SFT type gating

- **GIVEN** the requested hard-SFT type-gating stack
- **WHEN** the objective terms are resolved
- **THEN** the token-type mass term is the promoted stable auxiliary objective
- **AND** valid-set likelihood and within-valid coverage are not enabled by this
  change.
