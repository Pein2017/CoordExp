# Change: Add stage_2 post-rollout packing + vLLM colocate rollouts (default)

## Why
Stage_2 rollout-matching SFT (`custom.trainer_variant: rollout_matching_sft`) currently does:

rollout (no grad, per-sample generate) -> parse -> match -> teacher-forced forward (padded batch)

This is correct but inefficient:
- Rollout generation is currently called once per sample (`_rollout_one`), which underutilizes GPU even when the trainer receives many samples per step (e.g., from dataset packing) and is especially painful under DeepSpeed ZeRO2/3 where generation can trigger repeated parameter gathering.
- The teacher-forced forward pass after rollouts is padded, wasting compute (and forcing conservative batch sizes).
- `training.packing: true` is currently rejected in `src/sft.py` even though packing is only fundamentally incompatible with *generation*, not with the post-rollout *forward* pass.

The goal is to keep rollout generation un-packed, but enable post-rollout packing to recover stage_1-like throughput.

## What Changes
- Allow "post-rollout packing" for rollout-matching training so multiple samples can be concatenated into one packed forward pass after rollouts complete.
- Packing is **dynamic** and based on the actual constructed `Y_train` lengths (after rollout + FN append). Leftover segments are carried in a rank-local buffer (carry-only; no segment splitting).
- Add rank-local batched rollout generation (microbatching) so decode runs in batch (padded), not sample-by-sample.
- Allow `training.packing: true` under `custom.trainer_variant: rollout_matching_sft` by enforcing "no packing during generation" inside the trainer via temporary template flag overrides.
- Reuse existing `training.packing_*` YAML knobs (buffer/min_fill/drop_last) to control dynamic stage_2 packing behavior; dataset-level packing wrapper is not used for stage_2.
- Keep configuration YAML-first (no new CLI flags); knobs live under existing YAML namespaces.
- Add instrumentation and recommendations for rollout/forward phase utilization and straggler mitigation.
- Integrate vLLM rollout generation as the default rollout backend for stage_2:
  - vLLM is used **only** for no-grad autoregressive rollout decode.
  - teacher-forced forward/backprop stays on the training model (HF + DeepSpeed).
  - colocate mode only (server mode out-of-scope).
  - safe initial default: `gpu_memory_utilization=0.45` and `tensor_parallel_size=4` on 4 GPUs.
  - fail-fast on vLLM incompatibilities/OOM; fallback to HF is explicit via YAML.

## Impact
- Affected capability: `rollout-matching-sft` (packing policy + runtime behavior).
- Affected code (expected):
  - `src/trainers/rollout_matching_sft.py` (rollout batching, post-rollout packing, packed loss metadata)
  - `src/sft.py` (remove/relax packing rejection for rollout-matching; wire YAML knobs)
  - `configs/rollout_matching_sft_template.yaml` and `docs/STAGE2_ROLLOUT_MATCHING_RUNBOOK.md` (document knobs and recommended settings)
  - tests: extend `tests/test_rollout_matching_sft.py` with packed-mode invariants
