## MODIFIED Requirements

### Requirement: Channel-B step mode is step-budgeted in raw rollouts and learns-to-completion under packing
When Stage-2 AB training is enabled, Channel-B SHALL interpret the Channel-B batch size in terms of **raw rollouts per optimizer step**, not “packed sequences per optimizer step”.

Config contract (normative):
- `training.effective_batch_size` MUST be provided.
- `training.effective_batch_size` MUST be divisible by:
  - `training.per_device_train_batch_size × learner_world_size`,
  so `training.gradient_accumulation_steps` is an exact integer (no ceil overshoot).
- `training.gradient_accumulation_steps` MAY be omitted (recommended; derived from `effective_batch_size`).
  - If `training.gradient_accumulation_steps` is provided explicitly, it MUST equal the derived exact value implied by `effective_batch_size` and topology, otherwise config validation MUST fail fast with actionable guidance.
- `stage2_ab.channel_b.mode` MUST NOT be configurable. If provided, config parsing MUST fail fast with actionable guidance.
- `stage2_ab.channel_b.async` MUST NOT be configurable. If provided, config parsing MUST fail fast with actionable guidance.
- `stage2_ab.channel_b.rollouts_per_step` MUST NOT be configurable. If provided, config parsing MUST fail fast with actionable guidance.
- `stage2_ab.channel_b.enable_pipeline` MUST NOT be configurable. If provided, config parsing MUST fail fast with actionable guidance.
- `stage2_ab.channel_b.rollout_decode_batch_size` MUST NOT be configurable. If provided, config parsing MUST fail fast with actionable guidance.

Normative behavior:
- Define `rollouts_per_step := training.effective_batch_size`.
- The trainer MUST collect `rollouts_per_step` raw rollouts **globally across all train ranks** for the optimizer step.
  - Under DDP, each rank MUST collect a deterministic share `local_rollouts_per_step` such that the sum over ranks equals `rollouts_per_step`.
- The trainer MUST then construct per-sample teacher-forced segments (rollout prefix + mandatory FN append).
- When `training.packing=true`, the trainer MUST pack these segments into a **variable** number of packed sequences under the `packing_length` cap derived from `global_max_length`.
- The trainer MUST run forward/backward once per packed sequence and accumulate gradients, then perform **exactly one** optimizer update for the optimizer step.

#### Scenario: 32 raw rollouts pack into fewer than 32 packed sequences
- **GIVEN** `training.effective_batch_size=32`
- **AND** `training.packing=true` and `global_max_length=12000`
- **WHEN** Channel-B executes for one optimizer step
- **THEN** the trainer collects 32 raw rollouts
- **AND** packs them into `N_packs` packed sequences where `N_packs` MAY be less than 32
- **AND** performs one optimizer update for the step.

#### Scenario: Channel-B executes only on the final micro-step under grad accumulation
- **GIVEN** `training.per_device_train_batch_size=1`, `learner_world_size=4`, and `training.gradient_accumulation_steps=8`
- **WHEN** Channel-B is selected for one optimizer step
- **THEN** each rank buffers its raw rollouts across the first 7 micro-steps without running the Channel-B loop
- **AND** the full Channel-B loop (rollout→pack→learn-to-completion) runs on the 8th (final) micro-step
- **AND** the outer Trainer performs exactly one optimizer update for the step.

### Requirement: Channel-B step mode supports an in-step bounded pipeline queue between rollout and learning
When Channel-B executes with packing enabled and rollout backend is vLLM server mode (dedicated rollout GPUs), the trainer SHALL overlap rollout generation with learner compute within the optimizer step using a bounded producer/consumer queue (size 1 is sufficient).

Normative safety guardrail:
- Under vLLM server mode overlap, rollouts MUST run on dedicated GPUs via vLLM server mode.
  - Concretely: the trainer MUST require `custom.extra.rollout_matching.rollout_backend=vllm` and `custom.extra.rollout_matching.vllm.mode=server`.
  - If this condition is not met, the trainer MUST error fast with a clear message (to avoid unsafe concurrent rollout+train on the same process/device).

#### Scenario: Rollout and learner overlap within a step
- **GIVEN** rollout runs on dedicated GPUs via vLLM server mode
- **AND** learner training runs on a separate GPU
- **WHEN** Channel-B executes one optimizer step
- **THEN** the trainer overlaps rollout generation and learner forward/backward where feasible
- **AND** the trainer does not build an unbounded rollout pool.

### Requirement: Channel-B rollout decode batching is configurable and independent of learner microbatch
When Channel-B executes, the trainer SHALL allow configuring rollout decode batching independently of learner microbatch size (which remains 1 under packing).

Configuration (normative):
- The decode batching knob MUST be `custom.extra.rollout_matching.decode_batch_size` (int).
- It MUST denote the maximum number of sequences decoded per rollout GPU in one backend generation call.

#### Scenario: Rollout decode batch size 4 with learner microbatch 1
- **GIVEN** `training.per_device_train_batch_size=1`
- **AND** `custom.extra.rollout_matching.decode_batch_size: 4`
- **WHEN** rollouts are generated
- **THEN** the rollout backend generates responses with per-device decode batch size bounded by 4
- **AND** learner training still runs one packed sequence per forward/backward.
