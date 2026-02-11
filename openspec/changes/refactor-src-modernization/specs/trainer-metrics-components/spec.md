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

### Requirement: Canonical module ownership for metrics helpers is unambiguous
Metrics helper implementations SHALL live under `src/metrics/*` and MUST remain importable without importing trainer implementation internals.
Legacy modules under `src/trainers/metrics/*` MAY exist as compatibility shims, but MUST only re-export the canonical implementation and MUST NOT carry divergent behavior.

#### Scenario: Legacy import paths resolve to canonical behavior
- **GIVEN** a consumer imports a metrics helper from a legacy module path
- **WHEN** the helper functions are invoked
- **THEN** behavior matches the canonical `src.metrics.*` implementation
- **AND** no duplicated metric logic exists in the legacy module.

### Requirement: Neutral payload contract has a single canonical implementation
The neutral trainer-metrics payload contract SHALL have a single canonical implementation at `src/metrics/payload_contract.py`.
Any legacy or trainer-side module paths MAY re-export the contract types/helpers for compatibility, but MUST NOT duplicate validation/building logic.

#### Scenario: Payload parsing logic is not duplicated across module paths
- **GIVEN** a consumer imports the payload contract from either canonical or legacy module path
- **WHEN** payloads are validated/built
- **THEN** the same implementation is used in both cases (re-export), preserving consistent validation semantics.

### Requirement: Metrics ownership remains authoritative across overlapping deltas
Within active changes, this capability SHALL be authoritative for trainer-metrics ownership boundaries:
- canonical implementations live under `src/metrics/*`,
- `src/trainers/metrics/*` remains compatibility-shim surface only.

Other active deltas MUST NOT redefine a conflicting canonical home for metrics helper implementations.

#### Scenario: Overlapping change references keep canonical ownership consistent
- **GIVEN** another active change touches trainer-metrics helper imports
- **WHEN** OpenSpec artifacts are reviewed together
- **THEN** canonical ownership remains `src/metrics/*`
- **AND** trainer-side paths are treated as shim/re-export paths only.
