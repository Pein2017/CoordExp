## Context
Stage-2 AB training has two channels:
- **Channel-A**: iterative soft self-context (no rollout), teacher-forced.
- **Channel-B**: on-policy rollout + strict parse/match + teacher-forced learning.

This change refactors **Channel-B** to align batch semantics with the intended workflow and to maximize throughput under:
- learner constraint: `global_max_length <= 12000` per forward/backward on a single GPU (`per_device_train_batch_size=1`)
- rollout constraint: decode batch size 2 is desirable to utilize rollout GPUs
- preference: packing for maximal efficiency and minimal waiting on large rollout pools

Constraints (normative):
- JSON-only assistant schema (no tag-based outputs).
- Deterministic scheduling and rollout seeding.
- Qwen3-VL padding-free packing contract preserved (4-row `position_ids`).
- Do not modify upstream HF model files.

## Goals
- Make Channel-B “effective batch size” semantics match raw rollout count:
  - `rollouts_per_step ≈ 32` (configurable) means collect ~32 trajectories, regardless of how many packed sequences that yields.
- Improve throughput by overlapping rollout generation with learner compute via a bounded 1-step queue (pipeline) within the optimizer step.
- Preserve existing strict parse/match behavior, FN-append behavior, and bbox-only v1 guardrails.
- Keep the system YAML-driven; do not add new CLI flags.

## Non-Goals
- No bespoke RL loops or architecture changes.
- No poly training support (bbox-only v1 remains).
- No multi-learner server-mode support changes.

## Design: Channel-B step-budgeted pipelined loop

### Definitions
- “raw rollout / trajectory”: one dataset sample’s prompt (messages + image) → one decoded JSON output.
- “segment”: one sample’s teacher-forced encoding of `Y_train` (rollout prefix + mandatory FN append), treated as an atomic packing unit.
- “pack”: a list of segments whose total `encoded_len <= packing_length` where `packing_length` derives from `global_max_length`.

### Step Budgeting (normative)
For Channel-B optimizer step `s`:
- Let `rollouts_per_step` be:
  - `custom.extra.stage2_ab.channel_b.rollouts_per_step` if provided, else
  - the **derived global effective batch size** for one optimizer update:
    - `training.per_device_train_batch_size × world_size × training.gradient_accumulation_steps`
    - Note: when using ms-swift `training.effective_batch_size`, `training.gradient_accumulation_steps` is auto-derived (ceil), so the derived global effective batch size MAY be >= the user-requested value.
- This behavior applies when `custom.extra.stage2_ab.channel_b.mode=step`.
- The trainer MUST collect exactly `rollouts_per_step` raw samples globally (distributed across micro-steps if the outer loop uses gradient accumulation).

### Pipeline Queue (normative)
Within the Channel-B step, the trainer SHOULD overlap:
- rollout generation + parse/match + teacher encoding (producer),
with
- packed-sequence forward/backward (consumer).

A compliant implementation is a bounded producer/consumer queue of size 1:
- Producer produces segments (or pre-packed packs) in decode micro-batches (e.g., 2).
- Consumer packs available segments into a packed sequence (<= `packing_length`) and runs forward/backward.

This keeps staleness bounded to the current step and avoids building a long rollout pool.

### Learning-to-completion (normative)
Channel-B MUST:
- consume all `rollouts_per_step` rollouts for the step,
- pack into `N_packs >= 1` packed sequences (variable),
- run forward/backward once per pack (microbatch 1),
- perform exactly one optimizer update for the step.

Implementation note (performance):
- Under DDP, a naive “one backward per pack” loop will all-reduce gradients once per pack on the final micro-step.
- A compliant implementation SHOULD wrap all-but-the-final pack in `no_sync` so gradients synchronize exactly once per optimizer step.

### Determinism (normative)
- `rollout_seed_base` MUST be derived deterministically from `(training_seed, global_step)` and logged.
- Any internal batching/concurrency MUST preserve stable ordering of samples for:
  - strict prefix-token-id sanity checks,
  - deterministic metrics logging.

## Telemetry Requirements
The trainer MUST log at least:
- `stage2/raw_rollouts` (expected ~32)
- `train/samples_total` (from meta; should match raw rollouts)
- `train/micro_steps` (number of packed sequences consumed for the step)
- `time/rollout_generate_s`, `time/forward_s`
- rollout quality counters: parse_truncated, parse_dropped_invalid, drop_poly, fn_count
