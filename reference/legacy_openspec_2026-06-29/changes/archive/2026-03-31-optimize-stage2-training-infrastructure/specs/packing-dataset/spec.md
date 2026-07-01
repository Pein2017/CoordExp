# packing-dataset Specification (delta: static-pack reuse and deterministic precompute safety)

## Purpose
Extend the packing dataset contract so static packing can reuse deterministic prepared or encoded work safely, without relying on side-effectful hot-path sample fetches.

## Requirements

## ADDED Requirements

### Requirement: Static pack planning SHOULD reuse deterministic prepared or encoded work
When static pack-plan mode needs per-sample planning lengths, the system SHALL support a deterministic reuse path that avoids re-entering the full render-plus-encode hot path when valid prepared or encoded sidecar data already exists.

Normative behavior:
- This reuse path applies to dataset-level static pack planning and shared deterministic sample-fetch helpers; it does NOT redefine trainer-owned dynamic post-rollout packing governed by `stage2-ab-training` and `rollout-matching-sft`.
- Static planning MAY use a prepared-record or encoded-sample sidecar when it is valid for the active dataset and template contract.
- When a valid warm sidecar exists, planning MUST obtain lengths from that sidecar rather than re-running the full hot-path sample construction for every planned sample.
- Sidecar reuse MUST preserve the same planning length that the active training forward would consume.

#### Scenario: Warm static planning bypasses repeated encode work
- **GIVEN** static packing is enabled
- **AND** a valid deterministic prepared or encoded sidecar exists for the active dataset contract
- **WHEN** rank 0 builds or reloads the static pack plan
- **THEN** planning lengths are read from the sidecar
- **AND** the system does not re-run the full render-plus-encode path for each sample during plan creation.

### Requirement: Prepared-record sidecars MUST obey explicit eligibility and fingerprint rules
When the system reuses a prepared-record sidecar before token encoding, that sidecar MUST be governed by an explicit validity contract rather than treated as generically reusable.

Normative behavior:
- Prepared-record sidecars MUST be limited to deterministic dataset paths whose pre-encode content is a pure function of stable source identity plus fixed configuration.
- The validity fingerprint for a prepared-record sidecar MUST cover every pre-encode surface that can change the prepared payload or planning length, including:
  - dataset source identity,
  - prompts and system prompts,
  - object ordering and object field ordering,
  - coord-token behavior,
  - offline image-budget invariants,
  - and any fetch-time nondeterminism or epoch-sensitive behavior.
- If those conditions are not met, the system MUST bypass prepared-record sidecar reuse or fail fast with actionable guidance.

#### Scenario: Prepared sidecar reuse is rejected for an epoch-varying dataset path
- **GIVEN** a dataset path whose prepared pre-encode payload can change across epochs or fetch contexts
- **WHEN** static planning attempts to reuse a prepared-record sidecar
- **THEN** the system does not treat that sidecar as a valid reusable planning source
- **AND** it bypasses reuse or fails fast according to the configured policy.

### Requirement: Static length precompute MUST be side-effect free or fall back to a safe execution mode
When static packing computes planning lengths with concurrency, the precompute path MUST be side-effect free for the relevant dataset operations.

Normative behavior:
- If the dataset length path mutates shared template state, shared RNG state, or other fetch-time mutable state, the system MUST NOT use unsafe shared-object threaded precompute.
- In those cases the system MUST either:
  - use an immutable helper path for length precompute,
  - or fall back to a safe serial or process-isolated mode.
- For a fixed dataset, template, and configuration, repeated length precompute runs MUST be deterministic.

#### Scenario: Mutable dataset state disables unsafe shared-object threaded precompute
- **GIVEN** static packing is enabled for a dataset whose length path mutates shared state
- **WHEN** packing precompute initializes
- **THEN** the system does not use unsafe shared-object threaded precompute
- **AND** it selects an immutable or otherwise safe execution mode instead.
