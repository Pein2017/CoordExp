# trainer-metrics-components Spec Delta

This is a delta spec for change `stage2-rollout-metrics-canonicalization-2026-02-10`.

## MODIFIED Requirements

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

## ADDED Requirements

### Requirement: Sparse gauge aggregation avoids gradient-accumulation dilution
When optimizer-step metrics are aggregated from micro-step buffers, gauge-like keys that may be absent on some micro-steps MUST be averaged over key-observation count (presence count), not total micro-step count.

Counter-like keys MUST remain additive totals.

#### Scenario: Key present on one micro-step only
- GIVEN gradient accumulation with 32 micro-steps
- AND a gauge-like rollout config key present on exactly one micro-step with value `1.0`
- WHEN step-level aggregation finalizes
- THEN the logged value is `1.0` (not `1/32`).
