# training-pipeline-audit Specification

## Purpose
Define audit contracts that keep Stage-1/Stage-2 training entrypoints
reproducible, testable, and diagnosable as the codebase evolves.

## Requirements
### Requirement: Audit scope is anchored to operational entrypoints
The audit process SHALL explicitly cover the pipeline exercised by the following
operational entrypoints:
- `scripts/train.sh` with a canonical Stage-1 profile
- `scripts/train_stage2.sh` with a canonical
  `configs/stage2/rollout_correction/` profile

The audit SHALL map `data -> transforms/packing -> training/inference ->
artifacts` and SHALL enumerate code/module owners for each boundary.

#### Scenario: Audit starts from runnable entrypoints
- **WHEN** a training-pipeline audit is performed
- **THEN** it begins from canonical Stage-1 and Stage-2 runnable entrypoints
- **AND** it maps data, transforms, training/inference, and artifacts to owners.

### Requirement: High-risk invariants have CPU-only test coverage
The system SHALL maintain CPU-only unit tests that fail fast when any of the
following invariants are violated:
- geometry invariants (never drop/reorder coords; training keeps
  `do_resize=false`),
- assistant output boundary and template compatibility,
- packing invariants: `labels`, `attention_mask`, and `position_ids` remain
  aligned under packing,
- Stage-2 rollout-prefix plus residual/GT correction semantics do not regress,
- vLLM server-mode contract parsing and DDP-safe control-flow remain
  deterministic.

#### Scenario: High-risk invariant regression is caught by CPU tests
- **WHEN** a high-risk geometry, packing, prompt, rollout, or server-mode
  invariant regresses
- **THEN** a CPU-only test fails before production training is launched.

### Requirement: Objective-changing failures are fail-fast
The training pipeline SHALL fail fast with actionable error messages when an
invariant violation would change the training objective or invalidate
evaluation.

#### Scenario: Unknown critical config key fails fast
- **WHEN** a Stage-2 rollout-correction profile contains an unknown key under
  `stage2_rollout_correction` or `rollout_matching`
- **THEN** config loading fails before training starts
- **AND** the error includes the full dotted-path key and migration guidance.

### Requirement: Stage-2 rollout behavior is observable for diagnosis
Stage-2 rollout-correction training SHALL log sufficient aggregate diagnostics
to support diagnosis and audit, including:
- `stage2_rollout_correction/invalid_rollout`
- `stage2_rollout_correction/strict_drop/N_valid_pred`
- `stage2_rollout_correction/strict_drop/N_drop_invalid`
- rollout timing/throughput metrics when rollout correction executes
- residual-set metrics under `stage2_rollout_correction/residual_set/*`

#### Scenario: Stage-2 rollout diagnostics are emitted
- **WHEN** Stage-2 rollout-correction training executes rollout correction
- **THEN** aggregate invalid-rollout, strict-drop, timing, and residual-set
  diagnostics are available for audit.

### Requirement: Upstream dependency provenance is recorded in run artifacts
For paper-ready reproducibility, training SHALL persist upstream dependency
provenance into run artifacts. At minimum, run artifacts SHALL include versions
for `transformers`, `torch`, `vllm`, and `swift`, plus rollout-server launch
flags when server-mode is enabled.

#### Scenario: Dependency provenance is present in artifacts
- **WHEN** training run artifacts are written
- **THEN** they include upstream dependency versions
- **AND** server-mode runs include rollout-server launch flags.
