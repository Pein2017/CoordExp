# rollout-matching-sft Specification

## Purpose
TBD - created by archiving change 2026-01-19-update-stage2-post-rollout-packing-binpacking. Update Purpose after archive.
## Requirements
### Requirement: Stage-2 post-rollout packing selection uses deterministic ms-swift-like binpacking
When rollout-matching training is enabled (`custom.trainer_variant: rollout_matching_sft`) and post-rollout packing is enabled (`training.packing: true`), the trainer SHALL select which buffered segments are included in the packed teacher-forced forward pass using a deterministic, ms-swift-like constant-volume binpacking heuristic.

Definitions:
- A “segment” is one sample’s teacher-forced encoding of `Y_train` (rollout prefix + mandatory FN append), and is treated as an atomic unit.
- `packing_length` is the maximum packed length derived from `global_max_length` / `template.max_length` and is a hard cap per packed forward.

Selection requirements:
- The trainer SHALL maintain a rank-local buffer of pending segments.
- For each packed forward, the trainer SHALL select a subset of buffered segments whose total `encoded_len` is `<= packing_length`.
- The trainer MUST NOT split a single segment across multiple packed forwards.
- To avoid starvation, the trainer SHALL always include the oldest buffered segment in the selected subset.
- The trainer SHOULD attempt to improve fill ratio (larger total length) beyond the FIFO-greedy baseline, consistent with ms-swift’s constant-volume binpacking intent (e.g., via `binpacking.to_constant_volume`).
- The trainer MUST NOT produce a selection with a lower total selected length than the FIFO-greedy baseline for the same buffer state.
  - A compliant approach is: compute the FIFO-greedy baseline; compute a binpacking candidate constrained to include the oldest; pick whichever has higher total length; use stable tie-breaking.
- The selection MUST be deterministic: with identical buffered segments in identical insertion order and identical lengths, the selected subset and its order SHALL be identical across runs.
  - Tie-breaking MUST be stable.
  - If the selection logic encounters ties in its own scoring (e.g., equal total length), it MUST break ties deterministically, for example:
    - prefer fewer selected segments, then
    - prefer the lexicographically-smallest index set (in insertion-order indices).
- The selected subset SHOULD be ordered by insertion order (oldest-first) to minimize behavior change and keep packing deterministic.

Safety requirements:
- If any single buffered segment has `encoded_len > packing_length`, the trainer MUST fail fast at segment creation / buffer insertion time (not only when it becomes the oldest) with actionable guidance (e.g., increase `global_max_length`, reduce `max_new_tokens`, or disable packing).
- If post-rollout packing selection requires `binpacking` and the `binpacking` module is not available at runtime, the trainer MUST fail fast with actionable guidance (e.g., install `binpacking` or disable `training.packing`) rather than silently falling back to another heuristic.
- The trainer MUST preserve per-token supervision semantics under packing by maintaining correct offsets for all supervision masks/indices after packing.

The selection algorithm MUST reuse existing YAML knobs and MUST NOT require new CLI flags:
- `training.packing_buffer`
- `training.packing_min_fill_ratio` (telemetry/warn threshold)
- `training.packing_drop_last` (carry-only mode requirement remains unchanged)

#### Scenario: Multiple short segments pack efficiently under the same cap
- **GIVEN** rollout-matching training is enabled with post-rollout packing
- **AND** the buffer contains multiple segments whose individual `encoded_len` are all `< packing_length`
- **WHEN** the trainer selects segments for the next packed forward
- **THEN** it selects a subset whose total length is `<= packing_length`
- **AND** the selection includes the oldest segment
- **AND** the resulting fill ratio is at least as high as the FIFO-greedy baseline for the same buffer state.

#### Scenario: Deterministic selection
- **GIVEN** identical buffered segments in identical insertion order with identical `encoded_len`
- **WHEN** selection runs twice
- **THEN** it returns the same selected subset in the same order both times.

#### Scenario: Oversized segment fails fast
- **GIVEN** a segment with `encoded_len > packing_length`
- **WHEN** the trainer prepares the segment for post-rollout buffering / insertion
- **THEN** it raises an error that includes at least one mitigation suggestion.

#### Scenario: Missing binpacking dependency fails fast
- **GIVEN** rollout-matching training is enabled with post-rollout packing
- **AND** the runtime environment does not provide the `binpacking` module
- **WHEN** the trainer attempts to select segments for post-rollout packing
- **THEN** it raises an error that includes at least one mitigation suggestion (e.g., install `binpacking` or disable `training.packing`).

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

