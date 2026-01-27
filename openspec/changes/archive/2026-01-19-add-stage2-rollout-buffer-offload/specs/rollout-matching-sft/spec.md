## ADDED Requirements

### Requirement: Rollout-matching SFT supports a buffered rollout window (E-step / M-step)
When rollout-matching training is enabled (`custom.trainer_variant: rollout_matching_sft`), the system SHALL support
an opt-in rollout buffer that reuses one completed rollout window across multiple optimizer steps to improve
throughput when rollout generation is slow.

This behavior MUST be YAML-driven under `custom.extra.rollout_matching.rollout_buffer` and MUST NOT require new
hyperparameter CLI flags.

Configuration:
- `custom.extra.rollout_matching.rollout_buffer.enabled` MUST accept a boolean and MUST default to `false`.
- `custom.extra.rollout_matching.rollout_buffer.m_steps` MUST accept a positive integer and MUST default to `1`.

Terminology (normative):
- **micro-step**: one invocation of `Trainer.training_step(...)` on one dataloader batch.
- **optimizer step**: one parameter update step (after all micro-steps in the current gradient accumulation window),
  which increments `TrainerState.global_step`.
- **accumulation window**: the sequence of `gradient_accumulation_steps` micro-steps that contribute gradients to a
  single optimizer step.
- **raw sample**: one dataset record (one element of the list emitted by the identity collator).
- **raw micro-batch**: the list of raw samples received by the trainer on a micro-step (size equals
  `per_device_train_batch_size` for this trainer variant).
- **prepared model batch**: the tensor batch consumed by the model forward/backprop for one micro-step, produced by
  stage_2 rollout generation + matching + teacher-forced encoding (and post-rollout packing when enabled).
- **packed row**: when post-rollout packing is enabled, each prepared model batch is a single padding-free packed row
  (forward batch size = 1) that concatenates multiple segments; this packed row is treated as the atomic "training unit"
  for effective batch size accounting.

Effective batch size accounting (normative):
- When post-rollout packing is enabled, the *global effective batch size in packed-row units* per optimizer step is:
  `world_size * gradient_accumulation_steps * 1`.
- The `1` packed row MAY contain multiple raw samples (segments). Those segments MUST NOT affect the definition of the
  packed-row unit; they MAY be logged as a throughput diagnostic only.
- This packed-row accounting definition is diagnostic/semantic only. It MUST NOT change how the existing
  `training.effective_batch_size` YAML knob is interpreted or auto-computed elsewhere in the training stack.

Semantics when enabled:
- `m_steps` MUST be interpreted in **optimizer-step units** (i.e., in terms of `TrainerState.global_step`), not
  micro-step units.
- For each new rollout window ("E-step"), the trainer MUST prepare one full accumulation window worth of training work:
  - consume `gradient_accumulation_steps` raw micro-batches from the dataloader,
  - for each micro-batch, perform rollout generation + parse/match + `Y_train` construction + teacher-forced encoding
    (and post-rollout packing if enabled),
  - produce a list of prepared model batches to be used by the subsequent forward/backprop micro-steps.
  - Note: In HF/Swift Trainer implementations, this preparation is expected to be built incrementally as those
    `gradient_accumulation_steps` micro-steps arrive (not by prefetching multiple dataloader batches inside a single
    `training_step` call).
- During the subsequent `m_steps - 1` optimizer steps ("M-steps") in the same rollout window, the trainer MUST reuse the
  prepared micro-step batches in the same order, and MUST NOT generate new rollouts.
- The trainer MUST keep the rollout parsing/matching/append semantics unchanged relative to the non-buffered mode; only
  scheduling is changed.
- If stage_2 post-rollout packing is enabled, the trainer MUST treat the packing carry buffer as part of the E-step:
  - the carry buffer MAY be updated during E-steps,
  - the carry buffer MUST NOT be updated during M-steps (cached batches are reused as-is; no repacking).
- Cached batch immutability:
  - The trainer MUST treat cached prepared model batches as read-only.
  - When reusing a cached batch, the trainer MUST pass a safe copy into the underlying HF/Swift training stack, because
    `compute_loss` and other trainer internals may mutate the input dictionaries (e.g., via `pop`).

End-of-epoch / partial accumulation windows:
- The trainer SHOULD construct rollout windows from full accumulation windows (exactly `gradient_accumulation_steps`
  micro-steps).
- If the underlying dataloader yields a final partial accumulation window (fewer than `gradient_accumulation_steps`
  micro-batches), the trainer MUST process that final partial window without reuse (i.e., it MUST behave as if
  `m_steps=1` for that final window) and MUST NOT attempt to repeat it.

Eval / predict behavior:
- Rollout buffering is a training-only optimization. During evaluation/prediction, the trainer MUST disable reuse and
  MUST behave as if `m_steps=1` (even if the config enables buffering), to avoid confusing metrics.

Checkpoint/resume semantics:
- The rollout buffer is runtime-only state and MUST start empty on resume-from-checkpoint; the first step after resume
  MUST regenerate a new rollout window.

#### Scenario: Buffered rollout mode reuses one rollout window across multiple updates
- **GIVEN** rollout-matching training is enabled
- **AND** `rollout_buffer.enabled: true` and `rollout_buffer.m_steps: 4`
- **WHEN** 4 optimizer steps execute
- **THEN** rollouts are generated only during the first optimizer step of the rollout window
- **AND** the prepared micro-step batches from that E-step are reused for steps 2-4.

#### Scenario: Gradient accumulation does not change micro-batch composition
- **GIVEN** rollout-matching training is enabled
- **AND** `gradient_accumulation_steps: 3`
- **AND** `rollout_buffer.enabled: true` and `rollout_buffer.m_steps: 2`
- **WHEN** two optimizer steps execute
- **THEN** each optimizer step uses three micro-steps
- **AND** the raw micro-batches used for those three micro-steps are identical between the two optimizer steps.

Implementation guidance (non-normative but recommended):
- The trainer SHOULD perform a lightweight runtime sanity check that the raw micro-batch fingerprint is stable across
  M-steps (e.g., hash of `base_idx`/`sample_id` list for that micro-step) and warn or error if it diverges, as this
  typically indicates an incorrect dataloader repeater implementation.

#### Scenario: Cached prepared batches are safe to reuse
- **GIVEN** rollout-matching training is enabled with rollout buffering
- **WHEN** the trainer reuses a cached prepared model batch on a later micro-step
- **THEN** required keys (e.g., `_rollout_matching_meta`) are still present
- **AND** reuse does not crash due to prior in-place mutation of the cached batch.

### Requirement: Buffered rollout mode repeats accumulation windows to avoid skipping dataset samples
When rollout buffering is enabled with `m_steps > 1`, the system SHALL NOT silently skip dataset samples due to the
trainer reusing cached batches.

Instead, the training dataloader MUST repeat each full *accumulation window* `m_steps` times (per rank) so that the
trainer sees the same raw micro-batches for the duration of the rollout window.

This behavior MUST be stage_2-only (rollout-matching trainer variant) and MUST NOT affect baseline stage_1 SFT runs.

#### Scenario: Repeating dataloader prevents silent dataset skipping
- **GIVEN** rollout-matching training is enabled with `gradient_accumulation_steps: 2`
- **AND** `rollout_buffer.m_steps: 3`
- **WHEN** the training loop executes 3 optimizer steps
- **THEN** each optimizer step receives the same pair of raw micro-batches on that rank (repeated accumulation window)
- **AND** no dataset samples are dropped due to buffering.

### Requirement: Rollout-matching supports an offload context during colocate vLLM rollouts
When rollout-matching training uses vLLM rollouts in colocate mode, the system SHALL support an opt-in offload context
to reduce peak GPU memory usage during rollout inference.

Offload MUST be YAML-driven under `custom.extra.rollout_matching.offload`:
- `custom.extra.rollout_matching.offload.enabled` MUST accept a boolean and MUST default to `false`.
- `custom.extra.rollout_matching.offload.offload_model` MUST accept a boolean and MUST default to `false`.
- `custom.extra.rollout_matching.offload.offload_optimizer` MUST accept a boolean and MUST default to `false`.

Semantics when enabled:
- Offloading MUST occur only during rollout generation (no-grad inference).
- The trainer MUST restore model/optimizer state before teacher-forced forward/backprop.
- When rollout backend is not vLLM colocate (e.g., HF rollouts), offload settings MUST be ignored (no-op).
- When vLLM is lazily initialized, the offload context MUST cover all vLLM-side allocations required for rollout
  generation, including vLLM engine initialization and LoRA adapter loading/synchronization, not only the infer call.
- If offload is requested but cannot be applied safely under the current setup, the trainer MUST fail fast with an
  actionable error message that suggests at least one mitigation (e.g., disable offload, switch rollout backend to HF,
  or adjust DeepSpeed/ZeRO settings).

#### Scenario: Offload context does not break the training step
- **GIVEN** rollout-matching training is enabled with vLLM colocate rollouts
- **AND** offload is enabled for optimizer and/or model
- **WHEN** one training step executes
- **THEN** rollout inference completes without allocating training activations on GPU
- **AND** teacher-forced forward/backprop still executes successfully after offload restoration.
