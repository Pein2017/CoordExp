# stage2-ab-training Specification (Delta)

## ADDED Requirements

### Requirement: Channel-B step mode is step-budgeted in raw rollouts and learns-to-completion under packing
When Stage-2 AB training is enabled and Channel-B executes in step mode (`custom.extra.stage2_ab.channel_b.mode=step`), the trainer SHALL interpret the Channel-B batch size in terms of **raw rollouts per optimizer step**, not “packed sequences per optimizer step”.

Normative behavior:
- The trainer MUST collect `rollouts_per_step` raw rollouts **globally across all train ranks** for the optimizer step.
  - Under DDP, each rank MUST collect a deterministic share `local_rollouts_per_step` such that the sum over ranks equals `rollouts_per_step`.
  - If `custom.extra.stage2_ab.channel_b.rollouts_per_step` is unset, the trainer MUST default it to the **derived global effective batch size**:
    - `training.per_device_train_batch_size × world_size × training.gradient_accumulation_steps`
    - Note: when using ms-swift `training.effective_batch_size`, `training.gradient_accumulation_steps` is auto-derived (ceil), so the derived global effective batch size MAY be >= the user-requested value.
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
- **GIVEN** `training.per_device_train_batch_size=1`, `world_size=4`, and `training.gradient_accumulation_steps=8`
- **AND** `custom.extra.stage2_ab.channel_b.mode=step`
- **WHEN** Channel-B is selected for one optimizer step
- **THEN** each rank buffers its raw rollouts across the first 7 micro-steps without running the Channel-B loop
- **AND** the full Channel-B loop (rollout→pack→learn-to-completion) runs on the 8th (final) micro-step
- **AND** the outer Trainer performs exactly one optimizer update for the step.

### Requirement: Channel-B step mode supports an in-step bounded pipeline queue between rollout and learning
When Channel-B executes in step mode with packing enabled, the trainer SHALL support overlapping rollout generation with learner compute within the optimizer step using a bounded producer/consumer queue (size 1 is sufficient).

Normative safety guardrail:
- If the in-step pipeline queue is enabled, rollouts MUST run on dedicated GPUs via vLLM server mode.
  - Concretely: the trainer MUST require `custom.extra.rollout_matching.rollout_backend=vllm` and `custom.extra.rollout_matching.vllm.mode=server`.
  - If this condition is not met, the trainer MUST error fast with a clear message (to avoid unsafe concurrent rollout+train on the same process/device).

#### Scenario: Rollout and learner overlap within a step
- **GIVEN** rollout runs on dedicated GPUs via vLLM server mode
- **AND** learner training runs on a separate GPU
- **WHEN** Channel-B executes one optimizer step
- **THEN** the trainer overlaps rollout generation and learner forward/backward where feasible
- **AND** the trainer does not build an unbounded rollout pool.

### Requirement: Channel-B rollout decode batching is configurable and independent of learner microbatch
When Channel-B executes, the trainer SHALL allow configuring rollout decode batching (e.g., 2) independently of learner microbatch size (which remains 1 under packing).

#### Scenario: Rollout decode batch size 2 with learner microbatch 1
- **GIVEN** `training.per_device_train_batch_size=1`
- **AND** Channel-B rollout decode batch size is configured as 2
- **WHEN** rollouts are generated
- **THEN** the rollout backend generates responses for 2 samples in one decode call
- **AND** learner training still runs one packed sequence per forward/backward.
