## Context

Stage-1 SFT currently enables dataset-level packing via `PackedCaptionDataset` (rank-local, iterable, binpacking-based). This wrapper intentionally does not expose a stable `__len__`, so the training loop cannot compute “packs per epoch” exactly.

To keep runs finite, `src/sft.py` auto-fills `training_args.max_steps` under Stage-1 packing using a heuristic average samples-per-pack (`training.packing_avg_samples`). This makes `training.num_train_epochs` approximate under packing: the realized number of optimizer steps and raw-sample exposure depend on the length distribution and the packing dynamics.

This change adds a deterministic, countable alternative for Stage-1 packing: precompute per-sample lengths, build a static pack plan, and train from a map-style packed dataset with a stable `__len__`. The user requirement is fixed and comparable iteration counts (optimizer steps) across experiments when dataset length and epochs are fixed, while allowing a deterministic subset policy when `N % world_size != 0`.

Constraints:
- YAML-first (no new ad-hoc CLI flags).
- Preserve multimodal field compatibility with ms-swift padding-free collator (template flags `packing=true`, `padding_free=true`).
- No upstream HF model edits.
- Keep existing Stage-2 semantics unchanged (Stage-2 AB already budgets raw consumption per optimizer step via `training.effective_batch_size`).
- Single-node only (all ranks share a filesystem).

## Goals / Non-Goals

**Goals:**
- Provide a Stage-1 packing mode that yields a **countable** training dataset (`__len__` is defined and stable for the run).
- Make pack construction **deterministic** given (dataset, seed, packing knobs), and ensure all DDP ranks share the same pack plan.
- Ensure multi-sample packs respect the hard cap `sum(length_i) <= packing_length` and never split/truncate samples as part of packing. Single-long packs may exceed `packing_length` when `packing_allow_single_long=true` (size==1).
- Keep the “unit” under packing explicit: one per-device dataloader item is one packed sequence (list of encoded raw samples).
- Ensure DDP remainder handling is deterministic and matches standard semantics:
  - `training.dataloader_drop_last=true`: drop a deterministic remainder subset.
  - `training.dataloader_drop_last=false`: deterministically pad/repeat so all original packs are consumed and per-rank lengths are equal.

**Non-Goals:**
- Enforce “every raw record is seen exactly E times” under DDP. Deterministic drop/pad is sufficient.
- Change Stage-2 AB sample budgeting or rollout mechanics.
- Introduce new training CLI flags.
- Improve packing optimality beyond ms-swift-like constant-volume binpacking; the goal is determinism + countability.

## Decisions

1) **Config gating via a single YAML knob (`training.packing_mode`)**
- Decision: Keep `training.packing_mode ∈ {dynamic, static}`, but enforce Stage-1 dataset-level packing as static-only.
  - `static` (default): map-style static pack-plan behavior.
  - `dynamic`: deprecated/unsupported for Stage-1 dataset-level packing (fail-fast with guidance).
- Decision: Stage-1 eval packing follows the same static map-style pipeline as train packing by default.
  - `training.eval_packing` defaults to `true` and uses static pack-plan packing for eval datasets.
  - `training.eval_packing=false` is supported as an explicit opt-out.
  - Eval static packing uses non-dropping semantics (`packing_drop_last=false`, `dataloader_drop_last=false`) to preserve validation coverage.
- Decision: keep no extra strictness knob; enforce exact global pack-step semantics directly when `training.effective_batch_size` is set.
  - If `training.effective_batch_size % world_size != 0`, fail fast under packing.
  - In static mode, retain warning telemetry for epoch-boundary partial accumulation windows.
- Alternatives considered:
  - Add a second boolean like `training.packing_static: true`. Rejected as less extensible (future packing modes would multiply flags).
  - Implicitly switch based on `packing_avg_samples` presence. Rejected as ambiguous and non-obvious.

2) **Map-style packed dataset wrapper for static mode**
- Decision: Implement a `Dataset`-style wrapper (`__getitem__`, `__len__`) that stores an explicit `pack_plan: list[list[int]]`.
  - `__len__` is `len(pack_plan)` (packs per run).
  - `__getitem__(k)` returns `[base_dataset[i] for i in pack_plan[k]]`.
- Rationale: This makes dataloader length knowable and allows standard distributed sampling/remainder handling to produce fixed per-rank step counts.
- Alternatives considered:
  - Add `__len__` to an `IterableDataset` packer. Rejected because PyTorch/Trainer semantics for sized-iterables are brittle and ms-swift already treats iterable packing specially.

3) **Pack plan is computed once and shared deterministically**
- Decision: In static mode, the pack plan is computed once (epoch-invariant) and treated as part of the dataset definition for the run.
  - Per-epoch ordering differences come from the sampler shuffling pack indices (ms-swift sized-dataset path), not from regenerating the pack plan.
- Terminology:
  - `raw_plan`: the world-size-independent pack plan built from `(dataset, seed, packing knobs)`.
  - `aligned_plan(world_size, dataloader_drop_last)`: the plan actually consumed by training after deterministic DDP alignment.
- DDP alignment policy (deterministic and explicit):
  - `dataloader_drop_last=true`: drop by truncating the *tail* of `raw_plan` to a `world_size` multiple.
  - `dataloader_drop_last=false`: pad by repeating packs from the *start* of `raw_plan`, in order, until divisible by `world_size` (`aligned_plan = raw_plan + raw_plan[:pad_needed]`).
- Logging: always log `N_raw_packs`, `N_aligned_packs`, `world_size`, `dataloader_drop_last`, `pad_needed`, and checksums for both `raw_plan` and `aligned_plan`.
- Rationale: Under ms-swift, the epoch hook (`dataloader.set_epoch`) reaches the sampler (`BatchSamplerShard`) but does not call `dataset.set_epoch`. Avoiding per-epoch repacking removes a fragile integration point and keeps determinism straightforward.
- Alternatives considered:
  - Rebuild pack plan per epoch. Deferred: would require an explicit dataloader/sampler override to forward epoch state to the pack planner.

4) **Length planning uses a persistent length cache keyed by a fingerprint**
- Decision: Static mode uses per-sample planning lengths that are precomputed once and stored (or lazily extended) in a cache that is validated by a fingerprint.
  - Fingerprint MUST include at minimum: template identity (prompt variant/system prompt), `packing_length`/`global_max_length`, and any dataset-side switches that can affect tokenization (e.g., `custom.object_field_order`, summary/dense mode).
  - Fingerprint MUST include dataset source identity (`custom.train_jsonl` / `custom.fusion_config`, resolved path identity) so stale caches cannot be reused across different source files in the same output directory.
  - Cache is used only for planning; the actual training sample is still produced by the normal dataset encoding path.
- Cache scope: write caches under `training.output_dir` (run-scoped) to avoid accidental cross-run reuse with mismatched prompts/templates.
- Operational safeguards:
  - `training.packing_wait_timeout_s` controls how long non-rank0 workers wait for rank0-produced cache artifacts (`lengths.json`, `plan_ws*.json`). Default is `7200` seconds; `0` means wait indefinitely.
  - `training.packing_length_cache_persist_every` optionally sets periodic full-cache flush cadence while computing missing lengths.
  - `training.packing_length_precompute_workers` optionally controls rank0 multiprocessing fanout for missing-length computation (default `8`; `1` serial, `>1` multiprocessing workers).
  - When `training.packing_length_cache_persist_every` is unset, runtime chooses an adaptive persist interval that bounds the number of full-cache rewrites for large datasets.
- Multiprocessing determinism:
  - Rank0 parallel workers compute `(index -> length)` pairs only; rank0 writes results back by index into the canonical `lengths` array.
  - This preserves deterministic length arrays and pack plans while reducing wall-clock precompute time on CPU-bound datasets.
- Determinism guardrail: before writing or trusting a cache, run a small order-sensitivity probe (compute planning lengths for the same index set in two different access orders and require identical results). If the probe fails, static packing MUST fail fast with actionable guidance (disable static packing; avoid fusion/order-sensitive datasets; set deterministic preprocessing).
- Rationale: Planning must not require a full extra encoding pass every epoch, but must remain correct when prompts/templates change.
- Alternatives considered:
  - Compute lengths as `len(dataset[i]["input_ids"])` during pack-plan build each epoch. Rejected: doubles encode cost and makes pack planning scale poorly.
  - Use approximate text-length heuristics (character counts). Rejected: violates the hard cap and breaks determinism when tokenization changes.
  - Support fusion / order-sensitive datasets. Deferred: fusion sampling can be order-dependent under multi-worker prefetch and may rely on `set_epoch` to resample/mix; static packing is scoped to deterministic map-style datasets and MUST reject fusion-style schedules.

5) **Packing algorithm: ms-swift-like constant-volume binpacking with stable ordering**
- Decision: Reuse the existing constant-volume binpacking approach (`binpacking.to_constant_volume`), but enforce deterministic behavior:
  - Stable input order is the dataset index order (0..N-1).
  - Stable intra-pack ordering is preserved (oldest/lowest index first).
  - Underfill handling follows existing semantics (`packing_min_fill_ratio`, `packing_drop_last`).
- Rationale: Keeps behavior aligned with the existing packing contract and avoids introducing a separate greedy policy that changes fill statistics.
- Alternatives considered:
  - FIFO-greedy packing. Rejected: changes fill ratio distribution and would alter compute/token throughput in a way that complicates comparisons.

6) **Use sampler shuffle for per-epoch ordering (no repacking)**
- Decision: In static mode, per-epoch ordering randomness comes from the sampler shuffling pack indices (ms-swift `train_dataloader_shuffle`), not from regenerating pack plans.
- Rationale: This aligns with ms-swift’s sized-dataset training loop (`DataLoaderShard.set_epoch` → `BatchSamplerShard.set_epoch`) and avoids relying on dataset-level `set_epoch` propagation.
- Alternatives considered:
  - Disable sampler shuffle and rely purely on pack-plan order. Not chosen by default: it would make every epoch visit packs in the same order unless additional logic is added.

## Risks / Trade-offs

- **[Risk] Length cache mismatch after prompt/template changes →** Mitigation: fingerprint validation and fail-fast guidance (“delete cache” or “use a different cache dir”) rather than silently using stale lengths.
- **[Risk] Dataset preprocessing/augmentation changes token length across epochs →** Mitigation: constrain length computation to the same deterministic encode path as training (or document that static packing requires length-deterministic preprocessing); optionally compute planning lengths per epoch-index when hard sample plans introduce duplicates.
- **[Risk] Large datasets make length precompute expensive →** Mitigation: allow lazy caching + resume; store progress and avoid redoing work across runs; keep the cache opt-in via `packing_mode=static`.
- **[Risk] Non-rank0 workers can timeout while rank0 builds caches on large datasets →** Mitigation: configurable `training.packing_wait_timeout_s` with long default, plus `0` for explicit wait-forever behavior.
- **[Risk] Repeated full-cache rewrites can become an I/O bottleneck →** Mitigation: adaptive/default larger persist interval with optional `training.packing_length_cache_persist_every` override.
- **[Risk] DDP remainder requires explicit alignment under ms-swift sharding →** Mitigation: implement plan truncation/padding to an even multiple of `world_size` (drop vs pad/repeat) and log original vs aligned pack counts.
- **[Risk] Padding repeats some packs when drop_last=false →** Mitigation: define deterministic pad policy and log how many repeated packs are introduced for DDP alignment.
- **[Risk] Epoch-boundary partial accumulation windows can make packs/step inexact →** Mitigation: detect `per_rank_batches < GAS` or `per_rank_batches % GAS != 0` in static mode and warn with actionable guidance.
- **[Tradeoff] Strict dataset-source fingerprinting can reduce cache reuse →** Mitigation: include resolved source identity + file stat metadata for safety (fail-fast correctness), accept that path aliasing or metadata-only updates may invalidate run-scoped caches.
- **[Risk] Breaking comparability to old “epoch semantics” →** Mitigation: Stage-1 static-only is now explicit policy for reproducibility-focused runs; dynamic mode remains available only for rollout-matching trainer internals.

## Migration Plan

1) Stage-1 default behavior is now static-only:
   - Stage-1 configs with `training.packing=true` and no `training.packing_mode` now use `static`.
   - Stage-1 configs setting `training.packing_mode: dynamic` fail fast and must migrate to `static`.
2) Static packing for fair-comparison runs:
   - Keep `training.num_train_epochs` fixed to control total step count via true dataloader length.
   - Ensure `training.effective_batch_size` (when set) is divisible by `world_size`; otherwise packing fails fast by design.
   - Choose `training.dataloader_drop_last`:
     - `true`: drop deterministic remainder packs,
     - `false`: deterministically pad/repeat to include all original packs.
3) Rollback for Stage-1:
   - Disable packing (`training.packing=false`) if static mode is not desired.

## Open Questions

- Length-keying granularity for hard sample plans (`target_epoch_size > base_len`): cache by dataset `index` (exact for duplicates) vs cache by `base_idx` (smaller, assumes duplicates have identical length).
- Fusion datasets: whether to support static packing for fusion at all. Under multi-worker prefetch, fusion sampling can be order-sensitive and may rely on `set_epoch` to resample/mix; supporting static packing would likely require a deterministic, pre-materialized fusion schedule (out of scope for this change).
