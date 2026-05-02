---
doc_id: docs.data.packing
layer: docs
doc_type: reference
status: canonical
domain: data
summary: Packing policy, defaults, and efficiency tradeoffs.
updated: 2026-05-02
---

# Packing Mode Guide (Default: 12k, eff_bs=12)

Note:
- This guide applies to baseline SFT runs (stage_1 style) where training uses standard
  padding/packing dataset wrappers.
- Stage-2 trainers (`custom.trainer_variant: stage2_rollout_aligned` for the legacy rollout-matching
  compatibility path and `stage2_two_channel` for the active two-channel path) support
  **post-rollout packing inside the trainer** when `training.packing: true`:
  - rollout generation remains un-packed (padded batch),
  - each post-rollout `Y_train` is treated as an atomic segment (no splitting),
  - for `stage2_two_channel` clean-prefix v2, that packed Channel-B segment is built from the canonical clean teacher-forced target derived from `accepted_objects_clean`, not from the raw rollout prefix token ids,
  - `stage2_ab.channel_b.insertion_order` determines whether that final Channel-B target keeps the historical retained-anchor prefix plus FN tail (`tail_append`, default) or applies a final top-left sort over retained anchors plus FN objects (`sorted`),
  - `training.packing_buffer` / `training.packing_min_fill_ratio` control the dynamic packer.
  - `training.packing_drop_last: true` is required (no end-of-run flush steps; `stage2_rollout_aligned` uses a carry buffer).
  - `stage2_two_channel` (step-budgeted) uses a *pool-aware* selector that prioritizes minimizing the total number of packed
    sequences per optimizer step (fewer forward/backward calls) and secondarily avoids tiny remainder packs.
    - This may select a shorter current pack than FIFO-greedy when it reduces the overall number of packs for the per-step pool.
- Stage-2 runbook: [`../training/STAGE2_RUNBOOK.md`](../training/STAGE2_RUNBOOK.md).

Stage-1 packing guardrails (current implementation):
- Stage-1 dataset-level packing requires `training.packing_mode: static` (default). `training.packing_mode: dynamic` is deprecated/unsupported and fails fast.
- If you need multi-dataset mixing *and* Stage-1 static packing, materialize an offline merged JSONL first. Runtime fusion config authoring is temporarily disabled in the canonical training surface.
- Static packing may forward `set_epoch` into the raw dataset only for length-invariant per-epoch changes such as `custom.object_ordering: random`; `raw_plan` and `aligned_plan` stay fixed across epochs for eligible datasets.
- If epoch-varying content changes per-index planning length or sample schedule, static packing fails fast; use a length-invariant ordering configuration or disable `training.packing`.
- For sorted-vs-random ordering ablations, pin `training.encoded_sample_cache.enabled` explicitly in YAML. Random-order runs remain cache-ineligible, and the sorted arm should not keep an implicit cache-only advantage.
- Packed dataset wrappers expect the template to expose `packing` and `padding_free` attributes (ms-swift templates do; custom templates must implement them).
- Stage-1 static packing now defaults to a dataset-local auto-cache root instead of a run-scoped `training.output_dir/static_packing` folder. When `training.static_packing_cache.root_dir` is omitted or `null`, the runner resolves the base cache under `<jsonl_dir>/cache/static_packing/global_max_length_<N>/{train,eval}/`.
- Static packing artifacts are stored under a fingerprinted subdirectory beneath that base root. Legacy direct-root caches are not reused; the runner treats stale or incompatible packing artifacts as disposable and regenerates the current fingerprinted cache on launch.
- Each length bucket also writes an `INDEX.json` marker at the base root. When prompt/order/template or other packing-relevant fingerprint fields change, the runner warns and rewrites that marker to the latest setup before rebuilding any affected cache artifacts.
- `training.static_packing_cache.root_dir` is optional and only needed when you want to override the default dataset-local base root.
- Stage-1 static packing uses one hard length cap: `global_max_length` / `template.max_length`.
- Static packing probes each atomic sample at full length before building the pack plan. If any sample exceeds that hard cap, packing now fails fast instead of silently truncating or skipping it.
- `custom.trainer_variant: stage1_set_continuation` rejects both
  `training.packing: true` and `training.eval_packing: true` in v1. The trainer
  samples prefixes and constructs objective-specific rows inside
  `compute_loss`, so dataset-level pack-plan construction would mix independent
  prefix/object states and make token spans ambiguous.
- Candidate-balanced set-continuation branch scoring, energy/logZ candidate
  objectives, chunk-level MP, and candidate-branch CE are retired as production
  objectives. Do not treat branch-packing work for those objectives as a
  production packing direction.
- For the promoted Stage-1 ET-RMP-CE path, keep the production runtime on
  `smart_batched_exact` full-suffix rows. The packed-varlen branch-packing
  experiments showed that dense offline sample envelopes are possible, but the
  candidate-branch packed MP scoring path did not beat smart batching in the
  rough 2026-04-28 8-GPU probe. Any future padding-free packed runtime must
  preserve ET-RMP semantics: one prefix-conditioned full suffix per row,
  entry-trie support/balance targets, and hard CE for schema/control/separator
  and stop tokens.
- `training.encoded_sample_cache` is also ineligible for
  `custom.trainer_variant: stage1_set_continuation` because subset/candidate
  branches or full-suffix rows are sampled at runtime. With
  `training.encoded_sample_cache.ineligible_policy: error`, startup fails fast.
  With `ineligible_policy: bypass`, train/eval continue uncached and run
  artifacts record `status: bypassed`, `policy: bypass`, and
  `reason: stage1_set_continuation_branch_sampling`.
- `custom.sft_structural_close.enabled: true` also rejects packing. That
  ordinary-SFT ablation attaches per-token weights to the final global CoordJSON
  close sequence `]}` and therefore requires one un-packed assistant response
  per row.

## Why this is the new default
- Dramatically cuts padding waste (≈0% slack vs ~40–50% with padding).
- Keeps per-update scale close to padding: ~117 base samples/update vs 128 baseline.
- Safer memory headroom on A100 80GB than 20k while still reducing micro-steps ~5×.
- Covers >99.9% of LVIS samples without truncation (p99 text length ~11k).

## Recommended training knobs
```
global_max_length: 12000
per_device_train_batch_size: 1
effective_batch_size: 12        # world=4 → grad_accum ≈ 3
num_train_epochs: 4
packing_buffer: 256
packing_min_fill_ratio: 0.7
packing_drop_last: true
eval_packing: true
```
- For logging/checkpoint cadence at ~852 opt steps/epoch: `eval_steps: 80`, `save_steps: 80`, `save_delay_steps: 200`.
- Run name example: `epoch_4-stage1-coco80-sorted-text_only-packed-12k`.

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
  --config configs/stage1/sft_base.yaml --binsize 512
```
- Outputs mean/median/p95/p99, histograms, and packing sims for 12k/16k/20k with world=4, per_device=1.
- Adjust `--pack-lengths` to explore other caps; set `--per-device-train-batch` if changing per-device batch.

## When to try 20k
- If profiling shows higher tokens/sec end-to-end and memory is stable, you may raise `global_max_length` to 20000 while keeping `effective_batch_size: 12` and per_device=1.
- Expect fewer opt steps (~688/epoch) but heavier attention; watch for OOM and step-time regression.

## Migration checklist
1) Update configs to the defaults above in the current Stage-1 tree (`configs/stage1/`).
2) Align `eval_steps`/`save_steps` to ~80 and `save_delay_steps` to ~200 for 4-epoch runs.
3) If any atomic sample exceeds `global_max_length`, fix that upstream in preprocessing or route it to a separate run; Stage-1 static packing will not truncate it.
4) Keep ROOT_IMAGE_DIR set for dataset paths; packing requires non-lazy tokenize.
5) If comparing to padding runs, match total samples (epochs) rather than optimizer steps.
