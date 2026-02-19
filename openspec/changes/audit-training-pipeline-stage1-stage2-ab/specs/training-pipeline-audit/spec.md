# training-pipeline-audit Specification

## Purpose
Define a reproducibility-grade audit + verification harness for CoordExp stage_1 and stage_2 AB training.
This capability focuses on correctness, reproducibility, and evaluation validity by codifying high-risk
pipeline invariants as CPU-runnable tests and deterministic diagnostics.

## ADDED Requirements

### Requirement: Audit scope is anchored to operational entrypoints
The audit process SHALL explicitly cover the pipeline exercised by the following operational entrypoints:
- `scripts/train.sh` with `configs/stage1/ablation/geometry_first_coco80.yaml`
- `scripts/train_stage2.sh` with `configs/stage2_ab/prod/ab_mixed.yaml`

The audit SHALL map `data -> transforms/packing -> training/inference -> artifacts` and SHALL enumerate
code/module owners for each boundary.

#### Scenario: Audit produces a pipeline map
- **WHEN** the audit is executed for the anchored entrypoints
- **THEN** it produces a pipeline map that lists module owners for:
  - dataset ingestion/validation,
  - chat-template rendering/tokenization,
  - packing/collation/masks,
  - trainer forward/loss composition,
  - evaluation + metric emission,
  - checkpoint/artifact persistence,
  - vLLM server-mode rollout integration.
- **AND** each stage lists the relevant YAML key paths used by the entrypoints.

### Requirement: High-risk invariants have CPU-only test coverage
The system SHALL maintain CPU-only unit tests that fail fast when any of the following invariants are violated:
- geometry invariants (never drop/reorder coords; training keeps `do_resize=false`),
- assistant output boundary: CoordJSON rendering is transpiled before strict JSON parsing,
- packing invariants: `labels`, `attention_mask`, and `position_ids` remain aligned under packing,
- Stage-2 AB Channel-A vs Channel-B masking semantics do not regress (FP-neutral, FN injection, closure supervision),
- vLLM server-mode contract parsing and DDP-safe control-flow remain deterministic.

#### Scenario: CI runs pipeline invariant tests without GPUs
- **WHEN** running `PYTHONPATH=. conda run -n ms python -m pytest -q` on a CPU-only host
- **THEN** the invariant tests execute without requiring GPUs or a live vLLM server
- **AND** failures are reported with actionable diagnostics (what invariant failed and where).

### Requirement: Objective-changing failures are fail-fast
When an invariant violation would change the training objective or invalidate evaluation (e.g.,
tokenizer/template drift, misaligned packing masks, invalid rollout-prefix construction), training SHALL
fail fast with actionable error messages rather than silently continuing.

#### Scenario: Unknown critical config key fails fast
- **WHEN** a Stage-2 AB profile contains an unknown key under `stage2_ab` or `rollout_matching`
- **THEN** config loading fails before training starts
- **AND** the error includes the full dotted-path key and migration guidance.

### Requirement: Stage-2 rollout behavior is observable for diagnosis
Stage-2 AB training SHALL log sufficient aggregate diagnostics to support diagnosis and audit, including:
- realized scheduler telemetry (`stage2_ab/b_ratio_realized`),
- Channel-B strict-drop + invalid-rollout counters,
- rollout timing/throughput metrics (when Channel-B executes).

#### Scenario: Stage-2 AB emits required diagnostics keys
- **WHEN** running `custom.trainer_variant: stage2_ab_training`
- **THEN** the metrics payload includes:
  - `stage2_ab/b_ratio_realized`,
  - `stage2_ab/channel_b/invalid_rollout`,
  - `stage2_ab/channel_b/strict_drop/N_valid_pred`,
  - `stage2_ab/channel_b/strict_drop/N_drop_invalid`.

### Requirement: Upstream dependency provenance is recorded in run artifacts
For paper-ready reproducibility, training SHALL persist upstream dependency provenance into run artifacts.
At minimum, run artifacts SHALL include:
- versions for `transformers`, `torch`, `vllm`, and `swift`,
- ms-swift source provenance (path + git SHA, and dirty status if available),
- the rollout-server launch flags actually used when server-mode is enabled (dtype, DP/TP, max model len, eager mode, memory utilization).

#### Scenario: Run manifest includes upstream provenance
- **WHEN** a training run starts via the operational entrypoints
- **THEN** the run’s manifest (or equivalent run metadata artifact) contains explicit upstream provenance fields
- **AND** the provenance is sufficient to reproduce an equivalent environment or to detect drift across reruns.

### Requirement: Upstream integration surfaces have CPU-only contract tests
CoordExp depends on upstream APIs and contracts from transformers and ms-swift. The system SHALL maintain
CPU-only contract tests that fail fast when these upstream surfaces drift in incompatible ways.
These tests MUST NOT require GPUs or a live rollout server.

The minimum contract-test coverage SHALL include:
- transformers `Trainer` helper methods used by optimizer wiring (signature/behavior invariants),
- ms-swift rollout request configuration fields required by Stage-2 (e.g., `return_details=True` assumptions),
- ms-swift / vLLM interface payload shapes used for Qwen3-VL multimodal rollouts (serialization/type invariants).

#### Scenario: Contract tests run without GPUs
- **WHEN** running `PYTHONPATH=. conda run -n ms python -m pytest -q` on a CPU-only host
- **THEN** upstream contract tests execute without importing CUDA-only modules
- **AND** they fail with actionable diagnostics when a required upstream API surface is missing or incompatible.
