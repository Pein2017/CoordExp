## ADDED Requirements

### Requirement: Stage-1 research teacher forcing supports token-type-mass pressure
The compact Stage-1 research teacher-forcing surface SHALL support
`objective.terms.token_type_mass` as the stable type-family mass term.

Normative behavior:

- the current public implementation route is
  `pipeline.id: stage1_research_teacher_forcing` with
  `objective.id: research_teacher_forcing`,
- this change MUST NOT migrate or rename legacy OpenSpec objective identifiers
  outside the `token_type_mass` term contract,
- `objective.terms.token_type_mass` MUST expose only `enabled` and `weight`,
- `objective.terms.token_type_mass.weight` MUST be a finite nonnegative scalar
  and MUST default to `1.0`,
- `objective.profile=hard_sft` MAY enable `objective.terms.token_type_mass`,
  because token-family pressure is compatible with ordinary hard-token CE,
- `objective.profile=hard_sft` MUST continue to reject
  `conditional_valid_set_likelihood`, `within_valid_coverage`, and
  `continuation_margin`,
- pure-CE comparator configs MAY explicitly disable `token_type_mass` so the
  comparator remains interpretable,
- this term MUST remain YAML/config driven and MUST NOT require a new stable
  CLI flag.

#### Scenario: Hard SFT may use mandatory type-family pressure
- **GIVEN** a compact Stage-1 config with
  `pipeline.id: stage1_research_teacher_forcing`
- **AND** `objective.id: research_teacher_forcing`
- **AND** `objective.profile: hard_sft`
- **AND** `objective.terms.token_type_mass.enabled: true`
- **AND** `objective.terms.token_type_mass.weight: 1.0`
- **WHEN** config parsing and runtime payload resolution run
- **THEN** the config is accepted
- **AND** the resolved payload records the token-type-mass enabled state and
  weight.

#### Scenario: Hard SFT still rejects target-IR-only terms
- **GIVEN** a compact Stage-1 config with `objective.profile: hard_sft`
- **WHEN** it enables `conditional_valid_set_likelihood`,
  `within_valid_coverage`, or `continuation_margin`
- **THEN** config parsing fails fast
- **AND** the error points to the unsupported `objective.terms.*` key.

#### Scenario: Token-type-mass has no mode knob
- **WHEN** `objective.terms.token_type_mass` includes a key named `mode`
- **THEN** config parsing fails fast
- **AND** the error points to `objective.terms.token_type_mass`.
