# rollout-matching-sft Specification

## Purpose

This spec records the retired public rollout-matching SFT trainer surface. It is
not an active training contract.

Active Stage-2 training MUST use:

- `custom.trainer_variant: stage2_rollout_correction`
- top-level `stage2_rollout_correction`
- `stage2_rollout_correction.pipeline.objective[]` with exactly one enabled
  `residual_set_correction`
- runtime rollout/backend settings under `rollout_matching.*`

The shared implementation that prepares rollout prompts, dispatches HF/vLLM or
server rollouts, manages post-rollout packing, and materializes eval rollout
artifacts lives in `src/trainers/stage2_rollout_runtime.py`. That module is an
internal runtime base/helper surface only; it is not a public trainer variant.

## Requirements

### Requirement: Retired trainer variants are rejected

The system MUST reject legacy public Stage-2 rollout variants and guide users to
`stage2_rollout_correction`.

#### Scenario: legacy rollout variant is rejected

- **WHEN** a training config sets `custom.trainer_variant` to
  `rollout_matching_sft`, `stage2_rollout_aligned`, or
  `stage2_rollout_runtime`
- **THEN** config/runtime-plan validation fails fast
- **AND** the error recommends `stage2_rollout_correction`.

### Requirement: Retired rollout objective pipeline is rejected

The old `rollout_matching.pipeline.*` objective namespace MUST NOT be accepted
as an active training objective surface.

#### Scenario: rollout_matching pipeline is authored for active Stage-2

- **WHEN** a training config sets `custom.trainer_variant:
  stage2_rollout_correction`
- **AND** `rollout_matching.pipeline` is present
- **THEN** config validation fails fast
- **AND** the error tells the author to use
  `stage2_rollout_correction.pipeline`.

### Requirement: Rollout runtime settings remain supported

The `rollout_matching.*` namespace remains the active home for rollout backend,
decoding, vLLM/server dispatch, eval rollout, and post-rollout packing runtime
settings. It does not own Stage-2 objectives.
