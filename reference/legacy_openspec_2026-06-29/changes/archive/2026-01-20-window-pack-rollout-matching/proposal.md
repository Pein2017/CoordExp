## Why
Stage_2 rollout-matching frequently emits `post-rollout packing underfilled` warnings during large runs. This indicates the trainer is packing immediately after producing a single post-rollout segment (often short), which wastes the long-context budget and reduces throughput.

We want to preserve training math and reproducibility while improving scheduling so that post-rollout segments are accumulated across a full gradient-accumulation window before packing and teacher-forced SFT forward/backward.

## What Changes
- Add a config-driven scheduling option to make post-rollout packing *window-aware* (within-window only): the trainer
  accumulates segments for a full gradient-accumulation window and computes packing selections with visibility over
  the whole window.
- When enabled, the trainer:
  1) generates rollouts (vLLM/HF) per microbatch as before,
  2) parses/matches and builds per-micro post-rollout segments,
  3) accumulates these segments in a window-local buffer,
  4) computes packing selections with visibility over the accumulated segments,
  5) runs teacher-forced SFT forward/backward for exactly `gradient_accumulation_steps` micro-steps (no GA collapse).

This is a scheduling change intended to increase packing fill ratio and reduce wall time without changing correctness.

## Scope
In scope:
- Stage_2 trainer variant `rollout_matching_sft` only.
- Training loop only (not eval/predict). Eval should keep current behavior.
- Deterministic/seeded behavior preserved; no new CLI flags.

Out of scope:
- Changing losses/targets, parsing, matching, or rollout decoding.
- Cross-step carry of post-rollout segments (explicitly within-window only).

## Impact
Expected improvements:
- Higher `packing/post_rollout_fill` (fewer underfilled packs).
- Fewer teacher-forced forwards on tiny sequences; better GPU utilization.
- Reduced overall step time for rollout-heavy runs.

Risks:
- Incorrect loss scaling if the packed batch reduction differs from micro-by-micro behavior.
- Memory regression if packing produces more long sequences than expected.

Mitigations:
- Add a strict "math equivalence" validation mode for greedy rollouts: given fixed rollouts, ensure identical labels masks and loss (within numeric tolerance) between old and new scheduling.

## Rollout/Experiment Plan
- Add config knob defaulting to current behavior.
- Enable knob on a monitoring run (small sample limit) and compare:
  - wall time per optimizer step
  - packing fill ratio distribution
  - train loss curves and rollout metrics (f1/valid_pred_rate/truncation)
