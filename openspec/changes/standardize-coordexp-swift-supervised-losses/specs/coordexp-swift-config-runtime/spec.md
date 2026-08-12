## ADDED Requirements

### Requirement: Strict Supervised Loss Configuration

Every supported supervised config SHALL declare `losses.normalizer` as
`segment_balanced` and SHALL contain a strict protected-loss object with base
CE and token-type gate only. Base CE MUST have weight exactly `1.0`.
Token-type gate MUST contain exactly the ordered groups `desc_text`, `schema`,
`coordinate`, and `eos`, and MUST use either enabled mode with weight exactly
`0.1` or the explicitly named zero-weight-ablation mode with weight exactly
`0`. Coordinate Gaussian/RPS MUST be accepted only in the typed auxiliary-loss
object. Unknown loss names, implementation hooks, incompatible mode/weight
pairs, and the legacy protected placement MUST fail before model mutation.

Current supported configs under the canonical CoordExp-Swift config roots MUST
be migrated deliberately to this shape. Historical configs and completed
resolved-config artifacts SHALL remain unchanged historical evidence and MUST
NOT be interpreted as valid inputs to the new schema merely because they were
valid under an older commit.

#### Scenario: Canonical enabled SFT baseline

- **WHEN** a supported supervised config selects the normal protected baseline
- **THEN** strict resolution MUST retain base CE weight `1.0`, gate enabled
  mode and weight `0.1`, and the exact four gate groups in
  `resolved_config.json`.

#### Scenario: Canonical gate ablation

- **WHEN** a supported supervised config selects the named zero-weight gate
  ablation
- **THEN** strict resolution MUST require gate weight `0`
- **AND** MUST preserve the ablation identity in `resolved_config.json`.

#### Scenario: Mode and weight disagree

- **WHEN** gate enabled mode has a weight other than `0.1` or gate ablation
  mode has a weight other than `0`
- **THEN** config validation MUST fail before vocabulary or loss construction.

#### Scenario: Coordinate auxiliary uses legacy placement

- **WHEN** a supported config places `coord_gaussian_rps` under
  `losses.protected`
- **THEN** strict validation MUST fail with a migration-oriented field error
- **AND** MUST NOT silently move or alias the field.

#### Scenario: Optional coordinate auxiliary has zero weight

- **WHEN** coordinate Gaussian/RPS is present under the typed auxiliary object
  with weight `0`
- **THEN** the resolved config MUST retain the authored zero weight
- **AND** runtime MUST treat the term as omitted rather than as a diagnostic
  ablation.

#### Scenario: Dynamic loss hook authored

- **WHEN** a config authors a Python import path, callable reference, arbitrary
  loss options map, or unknown loss name
- **THEN** strict validation MUST reject it as an unknown or unsupported field
  before model mutation.

#### Scenario: Historical config is inspected

- **WHEN** an archived config or completed resolved-config artifact contains
  the legacy loss layout
- **THEN** it MUST remain unmodified and attributable to its historical code
  and schema version
- **AND** MUST NOT be promoted into a current supported config without an
  explicit migration.
