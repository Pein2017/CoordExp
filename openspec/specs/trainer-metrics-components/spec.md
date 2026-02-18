# trainer-metrics-components Specification

## Purpose
Define the canonical training metrics/logging component contract, including required keys, aggregation semantics, and removal of legacy metrics.

## Requirements
### Requirement: Stable batch extras contract
The system SHALL define a centralized contract for "batch extras" fields produced by collators and consumed by trainer hooks.
Batch extras MUST NOT be forwarded into `model(**inputs)`.

The contract MUST define the canonical set of batch extras keys. At minimum, it MUST include:
- `dataset_labels`
- `dataset_segments`
- `pack_num_samples`
- `token_types`
- `instability_meta_json`

#### Scenario: Extras are stripped before model forward
- GIVEN a collator returns a batch dict containing both model inputs and batch extras
- WHEN `Trainer.compute_loss` executes
- THEN batch extras are removed from the dict before calling `model(**inputs)`
- AND the extras are still available for logging/monitoring logic

#### Scenario: New extra is registered
- GIVEN a new diagnostics feature needs to add a new batch extra key
- WHEN the feature is implemented
- THEN the key is added to the centralized batch-extras contract
- AND the trainer strips it before model forward

### Requirement: Stable metric and batch key names
The system SHALL preserve the semantics documented in `docs/training/METRICS_LOSSES.md`.
The system MAY remove low-signal or duplicated metric keys when the docs are updated to the new canonical set.
Compatibility aliases are optional and MAY be omitted.

The system SHALL preserve the existing batch-extra key names listed in "Stable batch extras contract".

#### Scenario: Canonical-only metric contract
- GIVEN a training run after metric minimalization
- WHEN only canonical keys are emitted
- THEN removed legacy keys are absent from logs
- AND docs define the canonical key set.

#### Scenario: Key parity
- GIVEN a run with some features enabled/disabled (e.g. token-type metrics, coord loss, packing)
- WHEN running training/evaluation with the refactor enabled
- THEN every emitted metric key matches a documented train key in `docs/training/METRICS_LOSSES.md`
- OR matches the same key prefixed with `eval_` during evaluation (as described in the doc)
- AND existing metric keys are not renamed
- AND feature-conditional keys MAY be absent when their feature is disabled or skipped for a batch


### Requirement: Aggregate-only telemetry
The system SHALL keep metric emission aggregate-only by default. Per-dataset buckets MUST NOT be emitted.

#### Scenario: No per-dataset metrics
- WHEN aggregate metrics are logged during training/evaluation
- THEN keys do not include dataset-specific suffixes/prefixes
- AND no metric dict is created per dataset label

### Requirement: Best-effort diagnostics
Diagnostics-only logic (metrics computation, logging/reporting, and monitoring) MUST be best-effort:
- failures MUST NOT block training
- expected skip conditions (missing required inputs, known alignment mismatches) MUST skip only the affected batch and continue
- unexpected exceptions MUST emit a warning once per diagnostic
- unexpected exceptions MAY disable the failing diagnostic for the remainder of the run

#### Scenario: Token-type misalignment skips the batch
- GIVEN token-type metrics are enabled
- WHEN token-type alignment fails for a batch (e.g., concatenated token_types length differs from labels length)
- THEN token-type metrics are skipped for that batch
- AND training continues without crashing

#### Scenario: Unexpected exception does not block training
- GIVEN a diagnostics module throws an unexpected exception
- WHEN the training step runs
- THEN training continues without crashing
- AND a warning is emitted once indicating the diagnostic failed (and may be disabled)

### Requirement: Fail-fast loss components
Objective-changing logic (loss composition / label masking) MUST fail fast when enabled.
If an enabled loss component cannot be computed, the system MUST raise an error to avoid silently changing the training objective.

#### Scenario: Enabled coord loss failure raises
- GIVEN coord-token distributional supervision is enabled
- WHEN coord loss computation fails for a batch
- THEN the training step raises with a clear error message
- AND training does not proceed with a silently altered objective

### Requirement: Sparse gauge aggregation avoids gradient-accumulation dilution
When optimizer-step metrics are aggregated from micro-step buffers, gauge-like keys that may be absent on some micro-steps MUST be averaged over key-observation count (presence count), not total micro-step count.

Counter-like keys MUST remain additive totals.

#### Scenario: Key present on one micro-step only
- GIVEN gradient accumulation with 32 micro-steps
- AND a gauge-like rollout config key present on exactly one micro-step with value `1.0`
- WHEN step-level aggregation finalizes
- THEN the logged value is `1.0` (not `1/32`).


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
