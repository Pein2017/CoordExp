# stage2-ab-training Specification (delta: coordinated Stage-2 reduction, provenance, and restartability)

## Purpose
Extend the Stage-2 AB training contract so distributed step coordination, executed-runtime provenance, and restartability intent are explicit, testable parts of the supported training behavior.

## Requirements

## ADDED Requirements

### Requirement: Stage-2 AB distributed step coordination MUST be explicit and shared across channels
When Stage-2 AB runs under DDP, the trainer SHALL use one explicit distributed-step coordination contract for per-step barriers, rank-symmetric failure propagation, and optimizer-step metric reduction.

Normative behavior:
- Channel-A and Channel-B MUST use the same coordination ownership model for bounded phase barriers and rank-symmetric failure handling.
- The active Stage-2 path MUST NOT maintain multiple independent optimizer-step reduction systems for the same training step.
- Any per-step distributed coordination MUST remain bounded and fail clearly rather than waiting indefinitely.

#### Scenario: Channel-A and Channel-B share the same bounded coordination contract
- **GIVEN** a Stage-2 AB training run under DDP
- **WHEN** Channel-A and Channel-B each require bounded synchronization within the training step
- **THEN** both channels use the same shared coordination contract
- **AND** the job does not depend on separate ad hoc barrier policies for otherwise equivalent step-boundary coordination.

### Requirement: Stage-2 AB runs MUST emit executed-runtime provenance in addition to authored config
Stage-2 AB runs SHALL persist artifacts that describe the executed runtime after launcher and bootstrap mutation, not only the authored typed config.

Normative behavior:
- Stage-2 AB runs MUST emit:
  - `effective_runtime.json`
  - `pipeline_manifest.json`
- Stage-2 AB runs MUST also emit stable train-data provenance and, when eval data is configured, eval-data provenance sidecars that describe the resolved source identity for the executed run.
- These artifacts MUST describe the executed runtime surfaces that affect training behavior, including the resolved pipeline structure and relevant post-mutation runtime settings.
- These artifacts complement rather than replace the authored-config record.

#### Scenario: Operator can inspect executed runtime after launch mutation
- **GIVEN** a Stage-2 AB run where launcher or bootstrap logic mutates runtime settings derived from the authored config
- **WHEN** the run starts successfully
- **THEN** the output artifacts include an executed-runtime record and pipeline manifest
- **AND** an operator can distinguish authored config from executed runtime without re-deriving it from code.

### Requirement: Stage-2 AB checkpoint intent MUST distinguish artifact-only from restartable resume
When Stage-2 AB persists checkpoints, the run MUST make checkpoint intent explicit rather than leaving model-selection artifacts and restartable checkpoints ambiguous.

Normative behavior:
- `artifact_only` is the compatibility-preserving default mode for model-selection artifacts.
- `artifact_only` checkpoints MAY omit optimizer, scheduler, RNG, and repo-owned future-affecting runtime state, and MUST NOT claim exact resume fidelity.
- `restartable` MUST be an explicit opt-in mode.
- `restartable` checkpoints MUST include, at minimum:
  - model weights,
  - optimizer state,
  - scheduler state,
  - RNG state,
  - trainer state sufficient to restore `global_step`,
  - repo-owned future-affecting Stage-2 runtime state,
  - and restart-sensitive callback state or a recompute-safe equivalent.
- Resume preflight for `restartable` mode MUST fail fast when any required artifact is missing or incompatible.
- This requirement extends resume correctness while preserving the existing rule that schedule continuity depends on the restored `global_step`.

#### Scenario: Incomplete restartable checkpoint fails before resume
- **GIVEN** a Stage-2 AB run authored for `restartable` checkpoint mode
- **AND** a candidate checkpoint is missing required optimizer, RNG, runtime-state, or callback-state artifacts
- **WHEN** resume preflight runs
- **THEN** the run fails fast with actionable guidance
- **AND** it does not proceed as if the checkpoint were restartable.
