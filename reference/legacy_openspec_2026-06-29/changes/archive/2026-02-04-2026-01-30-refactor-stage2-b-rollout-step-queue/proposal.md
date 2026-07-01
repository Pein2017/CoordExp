# Change: Refactor Stage-2 Channel-B rollout into a pipelined, step-budgeted loop (1-step queue), bbox-only v1

## Why
Stage-2 Channel-B (rollout → strict parse/match → teacher-forced learning) can become rollout-bound and can exhibit confusing “effective batch” semantics under packing:

- In the current implementation, Channel-B work is naturally coupled to HF/Trainer micro-steps (gradient accumulation).
- Post-rollout packing may be forced into a fixed number of micro-batches (e.g., `gas`), which can underfill packs and waste compute.
- When rollout requests are issued serially or with small batches, vLLM server GPUs can be underutilized, inflating `time/rollout_generate_s` and dominating overall wall-clock.

This change proposes a refactor that preserves existing correctness/determinism guardrails while improving throughput and aligning batch semantics with the intended “roll out a small batch (~32) then learn immediately” workflow.

## What Changes
- **Channel-B becomes step-budgeted in terms of raw trajectories**, decoupling “how many rollouts to collect” from “how many packed sequences are produced”.
  - This behavior is enabled by `custom.extra.stage2_ab.channel_b.mode: step`.
  - Default: if `custom.extra.stage2_ab.channel_b.rollouts_per_step` is unset, it defaults to the **derived global effective batch size** for one optimizer update:
    - `training.per_device_train_batch_size × world_size × training.gradient_accumulation_steps`
    - Note: when using ms-swift `training.effective_batch_size`, `training.gradient_accumulation_steps` is auto-derived (ceil), so the derived global effective batch size MAY be >= the user-requested value.
- **Channel-B learns-to-completion**:
  - Build post-rollout segments for the step.
  - Pack those segments into a variable number of packed sequences under `global_max_length` (e.g., 12000).
  - Run teacher-forced forward/backward once per packed sequence (microbatch = 1) and accumulate gradients.
  - Perform one optimizer update for the step (preserving existing HF/Swift optimizer-step semantics).
- Add a **bounded 1-step pipeline queue** between rollout generation and learner training *within the same optimizer step* so that:
  - while the learner is computing forward/backward on packed sequence `k`,
  - rollout generation + parsing + teacher encoding can prepare the next segments/packs.
  - This overlaps rollout-server GPU time with learner GPU time.
- Preserve determinism and bbox-only v1 guardrails:
  - deterministic seeding (`rollout_seed_base`) and stable ordering,
  - strict invalid-rollout fallback,
  - drop predicted `poly` deterministically; fail fast on GT `poly`.

## Impact
- Affected specs:
  - **MODIFIED**: `stage2-ab-training`
  - (No change required to `rollout-matching-sft` spec unless we later choose to align additional server-mode timeout semantics.)
- Affected code (planned):
  - `src/trainers/stage2_ab_training.py` (Channel-B rollout + packing + training control flow)
  - `tests/test_stage2_ab_training.py` (new regression tests for step-budgeted Channel-B)
