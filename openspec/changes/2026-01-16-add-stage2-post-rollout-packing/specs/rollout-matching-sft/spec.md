## RENAMED Requirements
- FROM: `### Requirement: Packing is rejected under rollout-matching training`
- TO: `### Requirement: Post-rollout packing is supported under rollout-matching training`

## MODIFIED Requirements

### Requirement: Post-rollout packing is supported under rollout-matching training
When rollout-matching training is enabled (`custom.trainer_variant: rollout_matching_sft`), the system SHALL support packing for the *post-rollout teacher-forced forward pass* to improve training efficiency.

Packing MUST remain YAML-driven and MUST NOT require new CLI hyperparameter flags.

When packing is enabled for rollout-matching training:
- Rollout generation MUST remain un-packed (no sequence packing during autoregressive generation).
- The trainer MUST generate rollouts first, then build each sample's teacher-forced `Y_train` sequence (rollout prefix + mandatory FN append).
- The trainer MUST perform **dynamic packing** based on the actual constructed `Y_train` lengths (after rollout + FN append), not on pre-rollout dataset length estimates.
- The trainer MUST treat each sample as an atomic segment and MUST NOT split a single sample across multiple packed forwards.
- The trainer MUST preserve per-sample loss semantics by maintaining correct per-token supervision masks after packing (offset-correct in the packed row).
- The trainer MUST maintain a rank-local carry buffer for leftover segments and reuse the existing YAML packing knobs:
  - `training.packing_buffer`, `training.packing_min_fill_ratio`, `training.packing_drop_last`
- For carry-only mode, `training.packing_drop_last` MUST be `true` (the trainer does not run "extra flush steps" after max_steps/epoch end).
- If the carry buffer would exceed the configured `packing_buffer`, the trainer MUST fail fast with actionable error text.
- The trainer MUST keep the existing sanity checks:
  - prompt prefix tokenization matches generation,
  - supervised coord indices fall within the assistant span.

#### Scenario: Packing enabled does not break rollout-matching training
- **GIVEN** a YAML config sets `custom.trainer_variant: rollout_matching_sft`
- **AND** packing is enabled for training
- **WHEN** one training step executes
- **THEN** rollouts are generated without sequence packing
- **AND** the teacher-forced forward pass uses packed sequences
- **AND** loss masking logic remains correct and validated by sanity checks.

#### Scenario: Packing is never applied during rollout generation
- **GIVEN** rollout-matching training is enabled with packing
- **WHEN** the trainer performs autoregressive generation for rollouts
- **THEN** each rollout is generated using a standard padded batch encoding (un-packed)
- **AND** the packing mechanism is applied only after the rollout is complete.

#### Scenario: Carry buffer preserves segments without splitting
- **GIVEN** rollout-matching training is enabled with packing
- **AND** the rank-local carry buffer contains leftover segments from a previous step
- **WHEN** the next training step executes
- **THEN** the trainer packs forward using a subset of segments
- **AND** any leftover segments remain buffered for future steps
- **AND** no segment is split across packed forwards.

#### Scenario: Carry buffer overflow fails fast with actionable guidance
- **GIVEN** rollout-matching training is enabled with packing
- **AND** `training.packing_buffer` is too small for the configured raw batch size / sequence lengths
- **WHEN** the trainer would exceed the configured buffer cap
- **THEN** the trainer fails fast with an error message that suggests at least one mitigation (e.g. reduce raw batch size, increase `packing_buffer`, or enable multi-pack-per-step in a future change).

## ADDED Requirements

### Requirement: Rollout generation supports a vLLM colocate backend (default)
When rollout-matching training is enabled (`custom.trainer_variant: rollout_matching_sft`), the system SHALL support generating rollouts using a vLLM backend in **colocate** mode, while keeping teacher-forced forward/backprop on the normal training model.

Backend selection MUST be YAML-driven under `custom.extra.rollout_matching`:
- `rollout_backend` MUST accept `"vllm"` or `"hf"`.
- `rollout_backend` MUST default to `"vllm"`.

When `rollout_backend: "vllm"`:
- The trainer MUST configure vLLM from `custom.extra.rollout_matching.vllm` (mapping).
- Server mode MUST NOT be used for this trainer variant (colocate only).
- The vLLM backend MUST return:
  - `response_token_ids` (assistant token ids, stop-trimmed),
  - `prompt_token_ids` (prompt token ids used by vLLM),
  so stage_2 can enforce strict prompt-prefix token-id alignment.
- Default vLLM settings MUST be conservative to preserve training headroom on 4-GPU runs:
  - `gpu_memory_utilization: 0.45`
  - `tensor_parallel_size: 4`

#### Scenario: vLLM backend produces token ids suitable for strict alignment
- **GIVEN** rollout-matching training is enabled
- **AND** the rollout backend is set to vLLM colocate
- **WHEN** one training step executes
- **THEN** the trainer obtains per-sample `response_token_ids` and `prompt_token_ids` from vLLM
- **AND** the existing prompt-prefix sanity check is applied using those token ids
- **AND** the rest of parsing/matching/loss computation proceeds unchanged.

#### Scenario: Invalid vLLM configuration fails fast
- **GIVEN** rollout-matching training is enabled
- **AND** the rollout backend is set to vLLM
- **WHEN** vLLM is unavailable, tensor-parallel settings are incompatible, or LoRA sync is not possible
- **THEN** the trainer fails fast with an actionable error message
- **AND** the user can explicitly switch back to HF rollout via `rollout_backend: "hf"`.

### Requirement: Rollout generation supports microbatched decoding within each rank
When rollout-matching training is enabled, the trainer SHALL support generating rollouts for multiple samples in a single `generate` call (padded batch), controlled by a YAML knob under `custom.extra.rollout_matching`.

The default behavior MUST remain equivalent to the current implementation (microbatch size = 1).

#### Scenario: Microbatching increases decode parallelism without changing outputs format
- **GIVEN** rollout-matching training is enabled
- **AND** rollout generate batch size is set to `M > 1`
- **WHEN** the trainer generates rollouts for a batch of `M` samples on one rank
- **THEN** the trainer performs one batched generate call for those `M` samples
- **AND** it returns per-sample `response_token_ids` suitable for strict token-aligned parsing.
