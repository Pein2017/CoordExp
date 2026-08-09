## ADDED Requirements

### Requirement: Explicit Rollout-Calibration Loss Override

Normal supervised training SHALL continue to require protected full-row base
cross-entropy and token-type-gate losses under the existing defaults. An
explicit rollout-calibration research profile MAY omit full-row base
cross-entropy only when it consumes a validated frozen rollout state bank,
enables at least one approved rollout-calibration objective, and applies a
positive-weight token-type gate to every selected research site with a declared
intended token type. This exception MUST NOT change defaults or validation for
ordinary supervised-training profiles.

#### Scenario: Ordinary supervised run omits base cross-entropy

- **WHEN** a normal supervised-training config sets full-row base
  cross-entropy to zero or omits it
- **THEN** existing protected-loss validation MUST continue to fail.

#### Scenario: Valid rollout-calibration run omits base cross-entropy

- **WHEN** an explicit rollout-calibration profile uses a validated frozen
  state bank, enables an approved research objective, and configures a
  positive rollout-site token-type-gate weight
- **THEN** loss configuration MAY omit full-row base cross-entropy
- **AND** MUST compute no teacher-forced loss over historical prefix tokens or
  unrelated canonical rows.

#### Scenario: Calibration run disables its token-type gate

- **WHEN** a rollout-calibration profile has a zero or missing rollout-site
  token-type-gate weight
- **THEN** strict loss configuration MUST fail before model forward.
