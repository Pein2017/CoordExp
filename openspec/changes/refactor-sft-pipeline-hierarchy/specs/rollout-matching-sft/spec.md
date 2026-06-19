## MODIFIED Requirements

### Requirement: Retired trainer variants are rejected

The retired public rollout-matching SFT trainer surface SHALL remain inactive.

Active Stage-2 training MUST use:

- `pipeline.id: stage2_rollout_correction`
- top-level `stage2_rollout_correction`
- `stage2_rollout_correction.pipeline.objective[]` with exactly one enabled
  `residual_set_correction`
- runtime rollout/backend settings under `rollout_matching.*`

The shared implementation that prepares rollout prompts, dispatches HF/vLLM or
server rollouts, manages post-rollout packing, and materializes eval rollout
artifacts lives in Stage-2 runtime/trainer modules. Those modules are internal
runtime helper surfaces and are not public trainer variants.

#### Scenario: Legacy rollout variant is rejected

- **WHEN** a training config sets `custom.trainer_variant` to
  `rollout_matching_sft`, `stage2_rollout_aligned`, or
  `stage2_rollout_runtime`
- **THEN** config/runtime-plan validation fails fast
- **AND** the error recommends `pipeline.id: stage2_rollout_correction`.

#### Scenario: Former active custom selector is rejected

- **WHEN** a training config sets `custom.trainer_variant:
  stage2_rollout_correction`
- **THEN** config/runtime-plan validation fails fast
- **AND** the error recommends `pipeline.id: stage2_rollout_correction`.

### Requirement: Retired rollout objective pipeline is rejected

The old `rollout_matching.pipeline.*` objective namespace MUST NOT be accepted
as an active training objective surface.

#### Scenario: Rollout matching pipeline is authored for active Stage-2

- **WHEN** a training config sets `pipeline.id: stage2_rollout_correction`
- **AND** `rollout_matching.pipeline` is present
- **THEN** config validation fails fast
- **AND** the error tells the author to use
  `stage2_rollout_correction.pipeline`.

### Requirement: Rollout runtime settings remain supported

The `rollout_matching.*` namespace SHALL remain a runtime handle for rollout
backend, decoding, vLLM/server dispatch, eval rollout, and post-rollout packing
runtime settings until a separate schema-owned replacement seam is complete. It
MUST NOT own Stage-2 objectives or the public trainer concept.

#### Scenario: Rollout settings stay outside objective ownership

- **WHEN** a Stage-2 rollout-correction config authors runtime rollout settings
- **THEN** backend/decode/eval settings remain under `rollout_matching.*`
- **AND** objectives remain under `stage2_rollout_correction.pipeline`.
