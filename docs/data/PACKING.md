# Packing Mode Guide (Default: 12k, eff_bs=12)

Note:
- This guide applies to baseline SFT runs (stage_1 style) where training uses standard
  padding/packing dataset wrappers.
- Stage_2 rollout-matching SFT (`custom.trainer_variant: rollout_matching_sft`) supports
  **post-rollout packing inside the trainer** when `training.packing: true`:
  - rollout generation remains un-packed (padded batch),
  - each post-rollout `Y_train` is treated as an atomic segment (no splitting),
  - carry-only mode requires `training.packing_drop_last: true`,
  - `training.packing_buffer` / `training.packing_min_fill_ratio` control the dynamic packer.
- Stage_2 runbook: `../training/STAGE2_ROLLOUT.md`.

## Why this is the new default
- Dramatically cuts padding waste (≈0% slack vs ~40–50% with padding).
- Keeps per-update scale close to padding: ~117 base samples/update vs 128 baseline.
- Safer memory headroom on A100 80GB than 20k while still reducing micro-steps ~5×.
- Covers >99.9% of LVIS samples without truncation (p99 text length ~11k).

## Recommended training knobs
```
global_max_length: 16000
per_device_train_batch_size: 1
effective_batch_size: 12        # world=4 → grad_accum ≈ 3
num_train_epochs: 4
packing_buffer: 256
packing_min_fill_ratio: 0.7
packing_drop_last: true
eval_packing: true
```
- For logging/checkpoint cadence at ~852 opt steps/epoch: `eval_steps: 80`, `save_steps: 80`, `save_delay_steps: 200`.
- Run name example: `epoch_4-dlora-lrs_2_1_4-sorted-text_only-packed-12k`.

## Equivalence vs padding
- Padding baseline (per_device=2, eff_bs=128, world=4): grad_accum=16, ~777 opt steps/epoch.
- Packing 12k: packs/epoch ≈10,224 → micro steps/epoch ≈2,556; grad_accum≈3 → opt steps/epoch ≈852.
- Base samples/update ≈ 9.72 (samples/pack) × 4 GPUs × 3 accum ≈ 117 (close to 128 baseline).
- Formula: `grad_accum_packed ≈ ceil(128 / (avg_pack_samples * world * per_device))`.

## Image token estimate (post-merge)
- Qwen3-VL uses patch_size=16 and spatial_merge_size=2.
- Effective vision tokens per image: `ceil(H / 32) * ceil(W / 32)`.
- Packing length uses **text tokens**; vision tokens are for memory forecasting, not packing fit.

## How to regenerate stats
```
conda run -n ms python scripts/analysis/token_length_analysis.py \
  --config configs/dlora/sft_text_only.yaml --binsize 512
```
- Outputs mean/median/p95/p99, histograms, and packing sims for 12k/16k/20k with world=4, per_device=1.
- Adjust `--pack-lengths` to explore other caps; set `--per-device-train-batch` if changing per-device batch.

## When to try 20k
- If profiling shows higher tokens/sec end-to-end and memory is stable, you may raise `global_max_length` to 20000 while keeping `effective_batch_size: 12` and per_device=1.
- Expect fewer opt steps (~688/epoch) but heavier attention; watch for OOM and step-time regression.

## Migration checklist
1) Update configs to the defaults above (already applied to `configs/dlora/sft_text_only.yaml`).
2) Align `eval_steps`/`save_steps` to ~80 and `save_delay_steps` to ~200 for 4-epoch runs.
3) Monitor GPU memory; if headroom shrinks, drop to `global_max_length: 12000` and keep eff_bs=12.
4) Keep ROOT_IMAGE_DIR set for dataset paths; packing requires non-lazy tokenize.
5) If comparing to padding runs, match total samples (epochs) rather than optimizer steps.


