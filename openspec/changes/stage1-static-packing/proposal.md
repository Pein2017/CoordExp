## Why

Stage-1 SFT currently enables dataset-level packing via an `IterableDataset` wrapper (`PackedCaptionDataset`). Because the packed dataset has no stable `__len__`, `src/sft.py` auto-sets a finite `max_steps` using a heuristic average “samples per pack” (`training.packing_avg_samples`). This makes `num_train_epochs` an *approximate* control knob under packing: the realized number of optimizer steps and the realized raw-sample exposure vary with length distributions, packing randomness, DDP sharding, and worker scheduling.

For fair ablations (especially across packing settings and length distributions), we need *fixed, deterministic training iteration counts* when the dataset and `num_train_epochs` are fixed, with a deterministic subset policy when `N % world_size != 0`. These step counts are deterministic and exactly derivable given a fixed `world_size`, but are not invariant to changing `world_size` (because DDP alignment necessarily changes what is dropped/repeated).

## What Changes

- Add a **deterministic, countable Stage-1 packing mode** that precomputes (or loads) per-sample token lengths and builds a **static pack plan** (list of packed sequences).
  - For multi-sample packs, the pack MUST satisfy `sum(length_i) <= packing_length` (derived from `template.max_length`, with a `train_args.max_model_len` fallback).
  - When `packing_allow_single_long=true`, a single-long pack MAY exceed `packing_length` (current behavior: emit `[sample]` when `length >= packing_length`).
  - Packs MUST be formed without truncating/splitting samples.
- When this static packing mode is enabled, Stage-1 training SHALL use a **map-style packed dataset** with a stable `__len__`, so:
  - `len(train_dataloader)` and “packs per epoch” are known deterministically given (dataset, seed, packing config).
  - Total optimizer steps for a run are derived from (`num_train_epochs`, dataloader length, `gradient_accumulation_steps`) rather than the current heuristic `packing_avg_samples`-based `max_steps` auto-fill.
- Clarify and enforce the unit semantics under packing:
  - With packing enabled, one per-device dataloader item is one **packed sequence** (a list of raw encoded samples that the padding-free collator flattens).
  - `training.effective_batch_size` continues to drive `gradient_accumulation_steps`, but its “unit” under packing is **packed sequences**, not raw samples.
- Define deterministic DDP remainder handling for static packing:
  - Under the current ms-swift sized-dataset dataloader path, sharding floors `total_samples // world_size`. To implement standard DDP semantics:
    - `training.dataloader_drop_last=true`: truncate to an even multiple of `world_size` (drop a deterministic remainder subset).
    - `training.dataloader_drop_last=false`: deterministically pad/repeat packs so the pack plan length becomes an even multiple of `world_size` and all original packs are consumed.
- Remove a packing footgun for fair ablations:
  - Packing forces `per_device_train_batch_size=1`; when `training.effective_batch_size` is used, the system MUST fail fast or recompute `gradient_accumulation_steps` after this adjustment so the realized effective batch does not silently drift.
- Keep Stage-2 behavior unchanged (Stage-2 AB already budgets raw consumption per optimizer step via `training.effective_batch_size`).

## Capabilities

### New Capabilities
<!-- None: this is a requirement-level extension of the existing packing capability. -->

### Modified Capabilities
- `packing-dataset`: Extend the packing wrapper contract to support a deterministic, static pack-plan mode for Stage-1, including fixed `__len__` semantics and deterministic DDP remainder behavior under packing.

## Impact

- `src/sft.py`: Remove/disable the `packing_avg_samples`-based `max_steps` heuristic when static packing is enabled; ensure Stage-1 step/epoch semantics are derived from true packed length.
- `src/datasets/wrappers/packed_caption.py` (or a new wrapper module under `src/datasets/wrappers/`): Introduce a map-style “static packed dataset” implementation and length-cache/pack-plan plumbing.
- Config surface (`configs/stage1/*.yaml`): Add a YAML-first knob to select packing mode (dynamic/iterable vs static/countable) without introducing new CLI flags.
- Documentation/specs: Update `openspec/specs/packing-dataset/spec.md` via a delta spec to describe the new static mode and its determinism/step semantics.
