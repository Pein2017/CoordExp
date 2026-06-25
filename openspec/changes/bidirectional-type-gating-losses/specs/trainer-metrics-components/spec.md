# trainer-metrics-components Delta

## ADDED Requirements

### Requirement: Token-type mass metrics distinguish raw mean from contribution

Trainer metrics for the promoted token-type mass objective SHALL make the raw
family objective value and any weighted contribution accounting unambiguous.

Normative behavior:

- `teacher_forcing/loss/token_type_mass` MUST be the raw mean token-type mass
  loss over eligible supervised atoms before applying
  `objective.terms.token_type_mass.weight`;
- the raw mean MUST be mean-like and scale-invariant with respect to the number
  of eligible atoms in a micro-step;
- the token-type mass contribution to `teacher_forcing/loss/total` MUST equal
  `teacher_forcing/loss/token_type_mass *
  objective.terms.token_type_mass.weight`;
- if a separate standalone contribution metric is emitted, it MUST be named
  `teacher_forcing/loss/token_type_mass/contribution`;
- metric payload or reduction metadata MAY record the contributing weight,
  eligible count, or aggregation denominator, but it MUST NOT replace, alias, or
  rename the canonical standalone contribution metric when that metric is
  emitted;
- metric producers MUST NOT introduce a parallel `_weighted` metric family for
  token-type mass;
- metric producers MUST NOT emit underscore-style standalone aliases for this
  contribution metric;
- valid-set likelihood, within-valid coverage, coverage ledger, continuation,
  geometry, smoke, or salvage metrics MUST NOT be added by this token-type mass
  contract.

#### Scenario: Raw token-type mass metric is unweighted

- **GIVEN** `objective.terms.token_type_mass.weight: 0.25`
- **WHEN** a training step emits `teacher_forcing/loss/token_type_mass`
- **THEN** the emitted scalar is the raw mean family loss
- **AND** it is not pre-multiplied by `0.25`.

#### Scenario: Contribution accounting matches total-loss aggregation

- **GIVEN** a raw token-type mass mean `m`
- **AND** `objective.terms.token_type_mass.weight: w`
- **WHEN** the objective total is assembled
- **THEN** the token-type mass contribution to
  `teacher_forcing/loss/total` is `m * w`
- **AND** any explicit standalone contribution metric reports the same value at
  `teacher_forcing/loss/token_type_mass/contribution`.

#### Scenario: Token-type mass does not create weighted-suffix aliases

- **WHEN** trainer metrics are flattened for the promoted hard-SFT type-gating
  stack
- **THEN** `teacher_forcing/loss/token_type_mass` is present when the term is
  eligible
- **AND** `teacher_forcing/loss/token_type_mass_weighted` is not emitted.
