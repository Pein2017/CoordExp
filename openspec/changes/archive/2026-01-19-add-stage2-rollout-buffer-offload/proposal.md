# Change: Add stage_2 rollout buffering + colocate-rollout offload context

## Why
Stage_2 rollout-matching SFT (`custom.trainer_variant: rollout_matching_sft`) is currently bottlenecked by rollout
generation. In practice this causes:
- extreme step time variance (stragglers dominate), and
- too-few optimizer steps, leading to noisy/unstable training signals.

In addition, vLLM colocate rollouts can hit OOM due to multimodal prefill peaks and KV-cache pressure when training
and rollout share the same GPUs.

ms-swift's RLHF trainers (notably GRPO) contain robust infrastructure patterns for:
- temporally separating rollout inference from training (via offload contexts), and
- buffering rollouts across multiple training updates to amortize generation cost.

This change proposes reusing those patterns for rollout-matching SFT without changing the loss definition or the
token-aligned matching logic.

## What Changes
- Add an opt-in "rollout buffer" for stage_2 rollout-matching SFT:
  - generate rollouts + run matching once ("E-step"),
  - reuse the resulting teacher-forced training batch across multiple optimizer steps ("M-steps"),
  - keep rollout parsing/matching/append semantics unchanged.
- Add a stage_2 dataloader wrapper that repeats raw batches when rollout buffering is enabled, so we do not silently
  skip dataset samples while reusing buffered rollouts.
- Add an opt-in "rollout offload context" for vLLM colocate rollouts to reduce peak GPU memory during inference by
  temporarily offloading the training model and/or optimizer state to CPU.
- Keep configuration YAML-driven (no new hyperparameter CLI flags).

## Non-Goals
- No changes to the rollout parsing, Hungarian matching, or coord supervision semantics.
- No vLLM server-mode rollout support in this change (server vs colocate remains a separate decision).
- No new RLHF losses (we are not adopting GRPO/PPO losses here).

## Impact
- Affected capability: `rollout-matching-sft` (runtime scheduling + memory policy).
- Affected code (expected):
  - `src/trainers/rollout_matching_sft.py` (buffering + offload context around rollouts)
  - `src/sft.py` (stage_2 dataloader wrapper wiring)
  - `docs/STAGE2_ROLLOUT_MATCHING_RUNBOOK.md` (document new knobs)
  - `configs/dlora/*` (example configs / defaults)
- Risk: buffered rollouts are stale/off-policy across M-steps; must be explicitly enabled and bounded by config.

