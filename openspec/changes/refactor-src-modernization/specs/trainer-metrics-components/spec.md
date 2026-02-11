# trainer-metrics-components Spec Delta

This is a delta spec for change `refactor-src-modernization`.

## ADDED Requirements

### Requirement: Metrics consume neutral payload contracts
Trainer metrics components SHALL consume neutral payload contracts/events rather than importing trainer-internal symbols.
Metrics logic MUST remain executable in isolation from trainer implementation modules.

#### Scenario: Metrics computation runs with contract payload only
- **GIVEN** a valid neutral metrics payload contract
- **WHEN** metric components compute and report metrics
- **THEN** computation succeeds without importing trainer implementation internals.

### Requirement: Neutral metrics payload schema includes minimum fields and explicit versioning
The neutral metrics payload contract SHALL include a version field and minimum required fields for compatibility checks.
At minimum, payloads MUST include:
- `schema_version` (integer major contract version identifier; initial major version `1`),
- `mode` (`train` or `eval`),
- `global_step` (optimizer-step index),
- `metrics` (key/value map of numeric metrics).

Optional sections (for diagnostics/context) MAY include batch-extras and token/coord summaries, but optional sections MUST NOT be required to parse baseline payloads.
Consumers MUST treat missing or non-integer `schema_version` as invalid payloads.
Consumers MUST fail fast (or explicitly reject) unsupported major schema versions instead of silently mis-parsing payloads.

#### Scenario: Unsupported payload version is rejected explicitly
- **GIVEN** a payload with an unsupported major `schema_version`
- **WHEN** a metrics consumer validates the payload
- **THEN** the payload is rejected with explicit version-mismatch diagnostics
- **AND** the failure is not silently ignored.

#### Scenario: Non-integer payload version is rejected explicitly
- **GIVEN** a payload with missing or non-integer `schema_version`
- **WHEN** a metrics consumer validates the payload
- **THEN** the payload is rejected with explicit schema-version diagnostics
- **AND** the consumer does not attempt fallback parsing.

### Requirement: Metric key schema remains backward compatible during refactor
The metrics capability SHALL preserve existing documented metric key names and evaluation prefix behavior during refactor migration.
Any additive keys MUST be additive-only and MUST NOT rename existing stable keys.

#### Scenario: Metric key parity is preserved after internal refactor
- **GIVEN** equivalent training settings before and after refactor
- **WHEN** metrics are emitted
- **THEN** pre-existing stable metric keys remain unchanged
- **AND** evaluation keys preserve the documented `eval_` prefix convention.

### Requirement: Diagnostics remain best-effort with explicit first-failure signaling
Diagnostics-only metric paths SHALL remain best-effort but MUST emit an explicit warning on first unexpected exception and disable only the failing diagnostic path.
They MUST NOT silently suppress repeated failures with no signal.

#### Scenario: Unexpected diagnostics exception emits one warning and isolates failure
- **GIVEN** a diagnostics helper throws an unexpected exception
- **WHEN** a training step executes
- **THEN** a warning is emitted at first failure
- **AND** only the failing diagnostic is disabled while training continues.
