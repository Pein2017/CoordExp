## 1. Config Surface + Validation

- [x] 1.1 Add `training.packing_mode` (`dynamic|static`) as a packing-only YAML knob (config schema + loader-side stripping so it is consumed in `src/sft.py`).
- [x] 1.2 Update `src/sft.py` packing selection to default to static packing when `packing_mode` is unset, and fail fast for Stage-1 `packing_mode=dynamic`.
- [x] 1.3 In static mode, make epoch semantics explicit: pack plan is epoch-invariant; per-epoch ordering (if enabled) comes from the sampler shuffling pack indices. Always log `train_dataloader_shuffle`, `N_raw_packs`, `N_aligned_packs`, and stable checksums for `raw_plan` and `aligned_plan(world_size, dataloader_drop_last)`.

## 2. Length Cache (Planning Lengths)

- [x] 2.1 Implement a persistent length-cache format with a strict fingerprint (template + prompts + resolved `packing_length` + `global_max_length`/template max-length inputs + dataset-affecting knobs) and fail-fast mismatch diagnostics.
- [x] 2.2 Implement rank0 length precompute and deterministic distribution to other ranks (cache file or broadcast), so all ranks share the same planning lengths.
- [x] 2.3 Add lazy “compute-missing-lengths” support so interrupted runs can resume without recomputing everything.
- [x] 2.4 Add a guardrail for order-sensitive / non-deterministic datasets (e.g., fusion): static packing MUST fail fast with actionable guidance rather than silently caching unstable lengths (recommended detection: probe a small index set in two different access orders and require identical planning lengths).
- [x] 2.5 Add static-cache operational safeguards for large datasets: configurable inter-rank wait timeout and bounded/adaptive length-cache persistence cadence.
- [x] 2.6 Add rank0 multiprocessing for missing-length precompute with deterministic `(index -> length)` writeback and YAML control via `training.packing_length_precompute_workers` (default `8`; `1` serial, `>1` workers).

## 3. Static Pack-Plan Dataset Wrapper

- [x] 3.1 Implement a map-style `StaticPackedCaptionDataset` wrapper that yields one packed sequence per `__getitem__` and exposes `__len__ == N_packs`.
- [x] 3.2 Implement deterministic pack-plan construction (epoch-invariant) honoring `packing_length`, `packing_min_fill_ratio`, `packing_drop_last`, and `packing_allow_single_long` (stable tie-breaking + stable intra-pack ordering).
- [x] 3.3 Define and enforce deterministic pack ordering (e.g., sort each pack by index; sort packs by their first/min index) and log a checksum of the serialized plan.
- [x] 3.4 Share the pack plan deterministically across ranks (compute-once broadcast or cache file) to guarantee DDP consistency.
- [x] 3.5 Add exact packing telemetry for static mode (packs per epoch realized, avg fill, single-long count, skipped-long count).

## 4. Stage-1 Trainer Integration (Step/Epoch Semantics)

- [x] 4.1 Update `src/sft.py` to skip the `packing_avg_samples`-based `max_steps` heuristic when static packing is enabled; rely on true dataloader length + `num_train_epochs`.
- [x] 4.2 Implement deterministic DDP remainder semantics via pack-plan alignment:
  - `dataloader_drop_last=true`: truncate the tail of `raw_plan` to a `world_size` multiple.
  - `dataloader_drop_last=false`: pad by repeating packs from the start of `raw_plan` in order (`aligned_plan = raw_plan + raw_plan[:pad_needed]`).
  - Always log `world_size`, `dataloader_drop_last`, `pad_needed`, and the repeated pack indices (when padding occurs).
- [x] 4.3 Fix the `effective_batch_size` + packing footgun: if packing forces `per_device_train_batch_size=1`, recompute/override `gradient_accumulation_steps` derived from `effective_batch_size` so the realized effective batch does not silently drift.
- [x] 4.4 Add a clear runtime log line for the “unit” of `effective_batch_size` under packing (“packs”, not raw JSONL records).
- [x] 4.5 Route Stage-1 eval packing through the same static pack-plan pipeline as train packing, default `training.eval_packing=true`, and keep `training.eval_packing=false` as explicit opt-out.

## 5. Verification (Tests + Smoke)

- [x] 5.1 Add a unit test asserting static pack plans are identical across two runs given identical (dataset, seed, packing knobs), including pack ordering.
- [x] 5.2 Add a unit test asserting:
  - `dataloader_drop_last=true` drops by truncating the tail (exact dropped-pack set is deterministic),
  - `dataloader_drop_last=false` pads by repeating the *prefix* packs (`aligned_plan = raw_plan + raw_plan[:pad_needed]`),
  - final plan length is divisible by `world_size`, and
  - all original packs appear at least once.
- [x] 5.3 Add a minimal Stage-1 smoke YAML config enabling `packing_mode: static` and run it twice with the same seed to assert identical `optimizer.step()` counts and identical “packs per epoch” logs.
