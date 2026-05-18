# rollout-matching-sft Specification

## Purpose

This spec records the retired public rollout-matching SFT trainer surface. It is
not an active training contract.

Active Stage-2 training MUST use:

- `custom.trainer_variant: stage2_two_channel`
- `stage2_ab.pipeline.objective[]`
- `stage2_ab.pipeline.diagnostics[]`
- runtime rollout/backend settings under `rollout_matching.*`

The shared implementation that prepares rollout prompts, dispatches HF/vLLM or
server rollouts, manages post-rollout packing, and materializes eval rollout
artifacts lives in `src/trainers/stage2_rollout_runtime.py`. That module is a
runtime base/helper surface only; it is not a public trainer variant.

### Requirement: Retired Trainer Variants Are Rejected

The system MUST reject legacy public Stage-2 rollout variants and guide users to
`stage2_two_channel`.

#### Scenario: legacy rollout variant is rejected

- **WHEN** a training config sets `custom.trainer_variant` to
  `rollout_matching_sft`, `stage2_rollout_aligned`, or `stage2_rollout_runtime`
- **THEN** config/runtime-plan validation fails fast
- **AND** the error recommends `stage2_two_channel`

### Requirement: Retired Rollout Objective Pipeline Is Rejected

The old `rollout_matching.pipeline.*` objective namespace MUST NOT be accepted
as an active training objective surface.

#### Scenario: rollout_matching pipeline is authored for active Stage-2

- **WHEN** a training config sets `custom.trainer_variant: stage2_two_channel`
- **AND** `rollout_matching.pipeline` is present
- **THEN** config validation fails fast
- **AND** the error tells the author to use `stage2_ab.pipeline`

### Requirement: Rollout Runtime Settings Remain Supported

The `rollout_matching.*` namespace remains the active home for rollout backend,
decoding, vLLM/server dispatch, eval rollout, and post-rollout packing runtime
settings.

#### Scenario: active trainer uses rollout runtime settings

- **WHEN** a training config sets `custom.trainer_variant: stage2_two_channel`
- **AND** `rollout_matching.rollout_backend`, decode batch sizes, vLLM/server
  settings, or eval-detection rollout settings are present
- **THEN** those runtime settings are interpreted by the shared Stage-2 rollout
  runtime
- **AND** objective ownership remains under `stage2_ab.pipeline`
