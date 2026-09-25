## ADDED Requirements

### Requirement: Clean-source training continuation

Decision-bearing training SHALL bind the current clean Git commit, tree and
required regular source paths before accelerator, model or run-output effects.
Training resume state SHALL use schema version 2 and a separately readable JSON
source gate binding its exact payload bytes. The gate and current source SHALL
be verified before state deserialization; existing configuration, schedule,
optimizer and data compatibility checks SHALL still apply. Missing/legacy gates,
dirty source, mismatched commit/tree/path or changed state bytes MUST reject
continuation. A newly qualified run MUST NOT rewrite an old receipt or claim
historical numerical replay. A source digest is integrity, not authentication of
untrusted model files.

#### Scenario: Current qualified state resumes

- **WHEN** a clean current source identity and payload hash match the state gate and the existing resume compatibility checks pass
- **THEN** the same supported state restoration path may restore optimizer, scheduler and RNG; no GPU equivalence beyond its evidence is asserted

#### Scenario: Legacy or dirty continuation is requested

- **WHEN** a state lacks a current JSON source gate or the current source does not match
- **THEN** it fails as historical/unsupported before deserialization and before accelerator/model/output initialization

#### Scenario: Payload changes after qualification

- **WHEN** the state file bytes differ from the gate's hash
- **THEN** continuation is rejected without loading or repairing the payload
