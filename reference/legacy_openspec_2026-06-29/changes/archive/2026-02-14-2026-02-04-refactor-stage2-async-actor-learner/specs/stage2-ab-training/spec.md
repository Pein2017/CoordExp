# stage2-ab-training Spec Delta

This is a delta spec for change `2026-02-04-refactor-stage2-async-actor-learner`.

## MODIFIED Requirements

### Requirement: Channel selection is deterministic and step-driven
The trainer SHALL choose between Channel-A and Channel-B **deterministically** as a function of (`global_step`, `stage2_ab.schedule.b_ratio`).

Definition (normative):
- `global_step` MUST refer to the **optimizer-step** counter (post gradient-accumulation), i.e. the value that increments exactly once per optimizer update.
- The selected channel for a given `global_step` MUST remain fixed for the entire accumulation window (all micro-batches that contribute to that optimizer update).
- On resume from checkpoint, the schedule MUST continue from the restored `global_step` (no re-randomization).

Schedule definition (normative minimum):
- `stage2_ab.schedule.b_ratio` MUST be a float in `[0.0, 1.0]`.
- `stage2_ab.schedule.b_ratio` MUST be explicitly provided (no implicit default).
- Let optimizer step be `s` (0-indexed).
- The trainer MUST select Channel-B at step `s` iff:
  - `floor((s+1) * b_ratio) > floor(s * b_ratio)`.
  - Otherwise it MUST select Channel-A.

Special cases:
- If `b_ratio == 0.0`, the trainer MUST always select Channel-A.
- If `b_ratio == 1.0`, the trainer MUST always select Channel-B.

Legacy schedule handling (normative):
- The legacy list-based schedule knob `schedule.pattern` is not supported.
- If a config provides `stage2_ab.schedule.pattern`, configuration parsing MUST fail fast with guidance to migrate to `stage2_ab.schedule.b_ratio`.

DDP safety requirements (new/clarified for this change):
- Under `world_size > 1`, rank0 MUST decide the effective step kind (`A` vs `B`) **once per optimizer step** and broadcast it.
- The chosen step kind MUST remain fixed for the entire accumulation window (`gradient_accumulation_steps` micro-steps).
- Mid-accumulation switching (A↔B) is forbidden.

#### Scenario: b_ratio=0.5 alternates deterministically by optimizer step
- **GIVEN** `stage2_ab.schedule.b_ratio: 0.5`
- **WHEN** the trainer selects channels for `global_step` `s = 0, 1, 2, 3`
- **THEN** it selects Channel-A at steps `0` and `2`
- **AND** it selects Channel-B at steps `1` and `3`.

#### Scenario: b_ratio edge cases are deterministic
- **GIVEN** `stage2_ab.schedule.b_ratio: 0.0`
- **WHEN** the trainer selects channels for any `global_step`
- **THEN** it always selects Channel-A.

- **GIVEN** `stage2_ab.schedule.b_ratio: 1.0`
- **WHEN** the trainer selects channels for any `global_step`
- **THEN** it always selects Channel-B.

#### Scenario: Channel selection continues across checkpoint resume
- **GIVEN** a run that has completed optimizer step `global_step = s`
- **WHEN** training resumes from a checkpoint that restores `global_step = s`
- **THEN** the channel selected for step `s` is identical to the pre-resume selection for step `s`.

#### Scenario: stage2_ab.schedule.pattern fails fast
- **GIVEN** a training config with `custom.trainer_variant: stage2_ab_training`
- **AND** `stage2_ab.schedule.pattern: ["A","B"]` is provided
- **WHEN** config is parsed/materialized
- **THEN** it fails fast with guidance to use `stage2_ab.schedule.b_ratio`.

#### Scenario: Missing b_ratio fails fast
- **GIVEN** a training config with `custom.trainer_variant: stage2_ab_training`
- **AND** `stage2_ab.schedule.b_ratio` is not provided
- **WHEN** config is parsed/materialized
- **THEN** it fails fast with guidance to set `stage2_ab.schedule.b_ratio`.

#### Scenario: Multi-process learner uses rank0 broadcast for step kind
- **GIVEN** Stage-2 AB training is enabled under `torchrun` with `world_size=2`
- **WHEN** one optimizer step executes
- **THEN** rank0 broadcasts the step kind (`A` or `B`) for that optimizer step
- **AND** all ranks execute the same step kind for all micro-steps in the accumulation window.

## ADDED Requirements

### Requirement: Channel-B supports async actor-learner mode (versioned ready-pack queues)
Stage-2 AB SHALL support an async actor-learner mode configured as:
- `stage2_ab.channel_b.mode: async`

Topology / backend requirements (v1, robustness-first):
- Async mode MUST require server-mode rollouts:
  - `custom.extra.rollout_matching.rollout_backend: vllm`
  - `custom.extra.rollout_matching.vllm.mode: server`
  - `custom.extra.rollout_matching.vllm.sync.mode: full`
- Async mode MUST NOT use HF rollouts or vLLM colocate rollouts in v1.

Queue model (per-rank):
- Each learner rank MUST maintain its own FIFO queue of “ready packs”.
- Each ready pack MUST represent exactly **one** packed micro-batch dict suitable for one forward/backward.
- Queue depth MUST be bounded by `stage2_ab.channel_b.async.queue_limit`:
  - when full, the system MUST drop the oldest items first (drop-oldest).
- The prefetcher SHOULD target a steady-state queue depth driven by `stage2_ab.channel_b.async.prefetch_target_packs`.

Freshness / versioning:
- Rank0 maintains a monotonic sync-counter `ver` and broadcasts it to all ranks at safe boundaries.
- Each ready pack MUST be tagged with the `ver` used for its rollout generation.
- Each ready pack MUST be **version-pure**:
  - all segments inside a pack MUST have been generated under the same `ver`
  - a pack MUST NOT mix segments from multiple `ver` values.
- Learner consumption MUST enforce freshness:
  - only consume packs with `ver >= current_ver - version_window`
  - stale packs MUST be dropped and counted.

Policy vs feasibility gate:
- Policy gate: `stage2_ab.schedule.b_ratio` decides whether an optimizer step *wants* Channel-B.
- Feasibility gate: Channel-B may execute only if all ranks have at least `gradient_accumulation_steps` eligible packs available
  at optimizer-step start.
- If policy wants B but feasibility fails, the learner MUST execute Channel-A for that optimizer step and log:
  - `stage2_ab/async/b_step_skipped_due_to_queue = 1`

#### Scenario: Async B step is skipped when queues are empty
- **GIVEN** `stage2_ab.channel_b.mode: async`
- **AND** `stage2_ab.schedule.b_ratio` selects B for a step
- **AND** one or more ranks have fewer than `gradient_accumulation_steps` eligible ready packs
- **WHEN** the optimizer step begins
- **THEN** the trainer executes Channel-A for that step
- **AND** logs `stage2_ab/async/b_step_skipped_due_to_queue = 1`.

#### Scenario: Stale packs are dropped under a tight version window
- **GIVEN** async mode is enabled with `version_window: 1`
- **AND** the ready queue contains a pack with `ver < current_ver - 1`
- **WHEN** the learner attempts to consume a ready pack for Channel-B
- **THEN** it drops the stale pack and increments a stale-drop counter
- **AND** it does not train on the stale pack.

#### Scenario: Ready packs are version-pure
- **GIVEN** async mode is enabled
- **AND** the prefetcher has buffered leftover segments from a previous step
- **WHEN** `ver` increments due to a policy sync
- **THEN** the prefetcher does not combine old-version segments with new-version segments into a single ready pack
- **AND** any old-version leftover segments are either flushed into old-version packs or dropped before building new-version packs.

### Requirement: DDP-safe Channel-B execution semantics for multi-GPU learners
When `world_size > 1`, Channel-B MUST be executed in a DDP-safe way:
- Each micro-step MUST perform exactly one packed forward/backward per rank.
- The trainer MUST NOT run any inner loops that cause different ranks to perform different numbers of forwards within the same micro-step.

Legacy guardrail (v1):
- Under `world_size > 1`, the legacy `stage2_ab.channel_b.mode: step` MUST fail fast with actionable guidance to use `async`.

#### Scenario: Legacy step mode fails fast under DDP
- **GIVEN** Stage-2 AB is launched with `world_size=2`
- **AND** config sets `stage2_ab.channel_b.mode: step`
- **WHEN** training starts
- **THEN** the trainer fails fast with guidance to use `stage2_ab.channel_b.mode: async`.
