---
doc_id: docs.data.packing
layer: docs
doc_type: reference
status: canonical
domain: data
summary: Surface-specific packing policy, hard caps, cache behavior, and efficiency tradeoffs.
updated: 2026-06-15
---

# Packing Policy Matrix

Note:
- This guide applies to baseline SFT runs (stage_1 style) where training uses standard
  padding/packing dataset wrappers.
- Stage-2 rollout-correction training (`custom.trainer_variant:
  stage2_rollout_correction`) supports **post-rollout packing inside the
  trainer** when `training.packing: true`:
  - rollout generation remains un-packed (padded batch),
  - each rollout-correction train segment is atomic (no splitting),
  - the rollout prefix is roll-in context and the trainable target is the
    residual/GT correction target IR,
  - `stage2_rollout_correction.correction.insertion_order` controls how
    recovered GT objects are serialized for correction,
  - `training.packing_buffer` / `training.packing_min_fill_ratio` control the dynamic packer.
  - `training.packing_drop_last: true` is required (no end-of-run flush steps).
  - rollout correction uses a *pool-aware* selector that prioritizes minimizing the total number of packed
    sequences per optimizer step and secondarily avoids tiny remainder packs.
    - This may select a shorter current pack than FIFO-greedy when it reduces the overall number of packs for the per-step pool.
- Stage-2 runbook: [`../training/STAGE2_RUNBOOK.md`](../training/STAGE2_RUNBOOK.md).

## Current Surface Matrix

| Surface | Length cap | Effective batch | Packing support | Notes |
|---|---:|---:|---|---|
| Stage-1 baseline `configs/stage1/sft_base.yaml` | `12000` | `32` | static dataset packing | Uses `training.packing: true` and `training.eval_packing: true` where supported. |
| Stage-1 shared 4B coord recipes | `12000` | `128` | static dataset packing | Match comparisons by samples/epochs and record exact config. |
| Stage-1 compact recursive detection latest | `12000` | `128` | disabled | Packing remains disabled until sidecar target-position offset rewriting is implemented and validated. |
| Stage-2 rollout-correction base | `12000` | `64` | post-rollout trainer packing | Rollout generation remains padded/unpacked; correction segments are atomic. |
| Historical 12k packing probe | `12000` | `12` | historical probe | Useful as prior efficiency evidence, not the global default. |

Branch-provenance notes for segment-aware packing, coord-repel exact remapping,
and prefix-denoising hybrid packing live under `progress/` and `docs/history/`.
They are not current packing contract until their code/config surfaces are
merged and this matrix is updated.

## Effective Batch Source Of Truth

`training.effective_batch_size` is the primary optimizer-step budget when it is
present. `training.gradient_accumulation_steps` is derived from it and must not
be authored in YAML at the same time.

Runtime manifests record both the requested value and the realized optimizer-step
batch:

- `effective_batch_size`: requested/source value from YAML when authored.
- `actual_global_effective_batch_size`: `per_device_train_batch_size *
  gradient_accumulation_steps * world_size` after derivation.
- `effective_batch_rounding`: `exact` when the realized value matches the
  request, otherwise `ceil`.

Two execution regimes use the same source-of-truth rule:

- Packed / padding-free future runtime:
  `per_device_train_batch_size` is forced to `1`. The effective batch counts
  long packed sequence units per optimizer step. `global_max_length` /
  `template.max_length` is the pack cap for each packed unit, and the runtime
  derives gradient accumulation from `effective_batch_size` and `world_size`.
- Non-packed padded runtime:
  `per_device_train_batch_size` may be greater than `1`, padding follows the
  active tokenizer/template policy, and `global_max_length` is the per-sample
  hard cap. The optimizer-step sample budget is still
  `effective_batch_size`, with gradient accumulation derived from
  `effective_batch_size / (per_device_train_batch_size * world_size)`.

Latest compact recursive detection currently uses the non-packed padded regime.
Packing and padding-free packed runtime remain disabled until recursive sidecar
target-position rewriting is implemented and validated.

## Latest Compact Recursive Detection Packing Owner

Latest compact recursive detection uses top-level `packing` as the semantic
authoring owner. The current runtime still consumes adapter fields under
`training`, so latest configs must keep the semantic and runtime views aligned:

```yaml
training:
  packing: false
  eval_packing: false
packing:
  static_packing: false
  padding_free_packed: false
```

Until recursive sidecar target-position offset rewriting is implemented and
validated, canonical Stage-1 detection teacher-forcing configs under
`configs/stage1/detection_teacher_forcing/` must not enable dataset/static
packing or padding-free packed runtime. The retired recursive-detection config
root is archived under
`configs/archive/detection_scene_clean_break/stage1/recursive_detection_ce/`
for historical evidence only. Expected-failure packing examples belong under an
explicit `contract_failures/` or fixture location, not under positive `smoke/`
profiles.

## Stage-1 Packing Guardrails

Current implementation:
- Stage-1 dataset-level packing requires `training.packing_mode: static` (default). `training.packing_mode: dynamic` is deprecated/unsupported and fails fast.
- If you need multi-dataset mixing *and* Stage-1 static packing, materialize an offline merged JSONL first. Runtime fusion config authoring has been removed from the canonical training surface.
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
- Latest compact recursive detection surfaces keep packing and encoded-sample cache fail-fast until sidecar target-position rewriting is explicitly implemented and validated.
- Compact-full token-row runs train 1002 rows through the persisted
  `coord_offset_adapter` module name: 1000 coord rows plus the two compact
  structural rows `<|object_ref_start|>` and `<|box_start|>`. The module name is
  historical; the current contract is token-row adaptation, not coord-only
  adaptation.
- `training.encoded_sample_cache.max_resident_shards` bounds the number of shard
  files kept resident by the cache store. The default is `4`; raise it only when
  repeated shard reloads dominate dataset fetch time.
- `custom.sft_structural_close.enabled: true` also rejects packing. That
  ordinary-SFT ablation attaches per-token weights to the final global CoordJSON
  close sequence `]}` and therefore requires one un-packed assistant response
  per row.

## FlashAttention Varlen Boundary

Packed / padding-free runs that use FlashAttention should materialize explicit
segment boundaries at collator or batch-builder time when CoordExp sidecars are
active:

```text
cu_seq_lens_q
cu_seq_lens_k
max_length_q
max_length_k
```

These tensors describe attention isolation for the upstream forward pass.
CoordExp supervision sidecars describe loss ownership. The two maps must agree:
labels, image placeholders, coordinate tokens, loss masks, image-grid slices,
and sidecar ranges must be flattened in the same physical order. Do not rely on
a plain 2D `attention_mask` to express multiple packed examples inside one row.
For detailed upstream constraints, see
[`../standards/upstream/FLASH_ATTENTION.md`](../standards/upstream/FLASH_ATTENTION.md).

## Historical 12k Packing Probe
- Dramatically cuts padding waste (≈0% slack vs ~40–50% with padding).
- Keeps per-update scale close to padding: ~117 base samples/update vs 128 baseline.
- Safer memory headroom on A100 80GB than 20k while still reducing micro-steps ~5×.
- Covers >99.9% of LVIS samples without truncation (p99 text length ~11k).

## Historical Probe Knobs
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
python scripts/analysis/token_length_analysis.py \
  --config configs/stage1/sft_base.yaml --binsize 512
```
- Outputs mean/median/p95/p99, histograms, and packing sims for 12k/16k/20k with world=4, per_device=1.
- Adjust `--pack-lengths` to explore other caps; set `--per-device-train-batch` if changing per-device batch.

## When to Revisit 20k
- If profiling shows higher tokens/sec end-to-end and memory is stable, you may raise `global_max_length` to 20000 while keeping `effective_batch_size: 12` and per_device=1.
- Expect fewer opt steps (~688/epoch) but heavier attention; watch for OOM and step-time regression.

## Migration checklist
1) Update configs to the defaults above in the current Stage-1 tree (`configs/stage1/`).
2) Align `eval_steps`/`save_steps` to ~80 and `save_delay_steps` to ~200 for 4-epoch runs.
3) If any atomic sample exceeds `global_max_length`, fix that upstream in preprocessing or route it to a separate run; Stage-1 static packing will not truncate it.
4) Keep ROOT_IMAGE_DIR set for dataset paths; packing requires non-lazy tokenize.
5) If comparing to padding runs, match total samples (epochs) rather than optimizer steps.
