# packing-dataset Specification

## Purpose
Define the packing dataset wrapper contract used to pack variable-length sequences for efficient training under a fixed `global_max_length`.
## Requirements
### Requirement: Rank-local VL packing wrapper
The system SHALL provide a packing wrapper for vision-language training datasets that groups already-encoded samples (prompt + image → text) into packed sequences using ms-swift’s bin-packing heuristic.

#### Scenario: Packing enabled for training
- GIVEN `training.packing=true`
- AND a train dataset sample includes `input_ids`, `labels`, `length`, and multimodal fields
- WHEN the wrapper builds the dataloader item
- THEN it emits a list of samples whose summed `length` ≤ `packing_length`
- AND sets `template.packing=true` and `template.padding_free=true`
- AND SHALL auto-set `per_device_train_batch_size` to 1 if a larger value is provided, logging a warning, while preserving effective batch via `gradient_accumulation_steps`.

### Requirement: Bin-packing with carry-over buffer
The system SHALL bin-pack buffered samples per rank using `calculate_matched_group` (binpacking.to_constant_volume), carrying the final underfilled bin forward until the buffer is refilled or the epoch ends.

#### Scenario: Buffer fill and emission
- GIVEN a buffer of encoded samples with known `length`
- WHEN bin-packing runs with `packing_length` and `packing_min_fill_ratio`
- THEN groups with fill ratio ≥ threshold are emitted immediately
- AND the last underfilled group is retained for the next buffer
- AND at epoch end it is emitted or dropped based on `packing_drop_last`.

### Requirement: Oversized sample handling
The system SHALL detect any sample whose `length` ≥ `packing_length`; if `packing_allow_single_long=true` it SHALL emit it as a single-sample pack and log the event, otherwise it SHALL skip the sample and log a warning.

#### Scenario: Single long sample
- GIVEN a sample longer than `packing_length`
- WHEN `packing_allow_single_long=true`
- THEN the wrapper emits it as a one-sample pack without additional truncation
- AND logs that the sample exceeded `packing_length`
- AND ensures labels beyond any upstream truncation remain masked.
- WHEN `packing_allow_single_long=false`
- THEN the wrapper drops the sample and logs a warning about the skip.

### Requirement: Multimodal field preservation and position_ids
The system SHALL preserve multimodal tensors/metadata (e.g., `pixel_values`, `image_grid_thw`/`video_grid_thw`, `channel`) for each element in a pack so that the ms-swift padding_free collator can merge them and generate correct position_ids (including Qwen VL mRoPE logic).

#### Scenario: Packed collator compatibility
- GIVEN a packed list produced by the wrapper
- WHEN the ms-swift template collator runs
- THEN it receives all required multimodal fields, produces `position_ids`, and flattens the pack into a single batch element.

### Requirement: Fusion integration
The system SHALL apply packing after fusion scheduling so dataset mix ratios remain intact, and it SHALL forward `set_epoch` so packing rebuilds after each epoch’s fusion reshuffle.

#### Scenario: Fusion with packing enabled
- GIVEN `FusionCaptionDataset` is used with ratios
- WHEN `set_epoch` is called for a new epoch
- THEN the fusion schedule is rebuilt first
- AND the packing buffer/index state is reset to reflect the new order.

### Requirement: Metric and telemetry behavior
The system SHALL record only aggregate training metrics (loss/token_acc) when packing is enabled and SHALL drop per-dataset telemetry; it SHALL log pack fill ratios and the percentage of single-long or skipped-overlength packs per epoch for observability.

#### Scenario: Metrics under packing
- GIVEN packing is enabled during training
- WHEN metrics are reported
- THEN only aggregate metrics are recorded
- AND logs include fill ratio statistics and counts of single-long packs.

### Requirement: Configuration surface
The system SHALL expose packing controls: `packing_length` (default template max_length), `packing_buffer` size, `packing_min_fill_ratio`, `packing_drop_last`, and `packing_allow_single_long`; defaults SHALL favor safety (drop-last on, min_fill_ratio ≥0.6).

#### Scenario: Default config application
- GIVEN a user enables packing without overriding knobs
- WHEN training starts
- THEN the defaults above are applied
- AND a warning is emitted if an incompatible batch size is detected and auto-corrected.

### Requirement: Eval default off
The system SHALL leave evaluation dataloaders un-packed (standard padded, non-packing batching) unless explicitly requested, to avoid length/bias side effects on validation metrics.

#### Scenario: Training packed, eval not
- GIVEN packing is enabled for training
- WHEN evaluation runs without an explicit eval packing flag
- THEN eval uses standard padded (non-packing) batching.

### Requirement: Documentation alignment
The system SHALL update or override any existing documentation that claims packing is removed or padding-only to reflect the reintroduced packing capability and its constraints.

#### Scenario: Docs reconciled with packing
- GIVEN prior docs state that packing is removed
- WHEN the packing wrapper capability is added
- THEN the relevant docs are updated to describe the new packing option, its defaults, and constraints (e.g., batch size = 1, aggregate metrics only).

### Requirement: Stage-1 dataset-level packing is static-only
For Stage-1 dataset-level packing (non-rollout trainer variants), the system SHALL enforce a static-only policy:
- `training.packing_mode` defaults to `static` when omitted.
- `training.packing_mode=dynamic` is deprecated/unsupported for Stage-1 dataset-level packing and MUST fail fast with actionable guidance.
- `training.eval_packing` defaults to `true` for Stage-1 dataset-level packing and MUST use the same static map-style pack-plan pipeline as train packing.
- `training.eval_packing=false` MAY be used to disable eval packing explicitly.
- When `training.eval_packing=true`, eval static packing MUST preserve evaluation coverage by using non-dropping alignment semantics (`packing_drop_last=false`, `dataloader_drop_last=false`).
- `training.packing_wait_timeout_s` SHALL control non-rank0 wait time for static cache artifacts; default is `7200` seconds, and `0` SHALL mean wait indefinitely.
- `training.packing_length_cache_persist_every` MAY be provided as a positive integer to override periodic length-cache flush cadence; when omitted, runtime SHALL use an adaptive flush interval suitable for large datasets.
- `training.packing_length_precompute_workers` MAY be provided as a positive integer to control rank0 multiprocessing worker count for static length precompute; default is `8` (`1` means serial, `>1` means multiprocessing).

#### Scenario: Stage-1 dynamic packing mode fails fast
- **GIVEN** Stage-1 dataset-level packing is enabled (`training.packing=true`)
- **AND** `training.packing_mode=dynamic`
- **WHEN** training initializes packing policy
- **THEN** the system fails fast and instructs the user to set `training.packing_mode=static`.

#### Scenario: Stage-1 eval packing defaults to static pack-plan behavior
- **GIVEN** Stage-1 dataset-level packing is enabled (`training.packing=true`) and `training.eval_packing` is omitted
- **WHEN** eval dataloader packing is initialized
- **THEN** eval packing is enabled by default
- **AND** eval uses the same static map-style pack-plan pipeline as train packing.

### Requirement: Static pack-plan mode yields a countable packed dataset
When training packing is enabled and static pack-plan mode is selected, the system SHALL provide a map-style packed dataset wrapper that:
- precomputes a “pack plan” as a list of packs, where each pack is an ordered list of raw-sample indices,
- yields one dataloader item per pack as `List[Dict]` (atomic samples), and
- exposes a stable `__len__` equal to the number of packs in the plan (and therefore for each epoch).

Definitions:
- A “raw sample” is one encoded training example (including all multimodal fields) treated as an atomic unit.
- A “pack” is a list of raw samples concatenated by the padding-free collator into a single training sequence.
- `global_max_length` is the run-level hard cap used by CoordExp; configuration typically sets `template.max_length` and `model.max_model_len` from it unless explicitly overridden.
- `packing_length` is the hard per-pack cap derived at runtime from `template.max_length` (preferred) or, if unset, from `training_args.max_model_len` as a fallback. `training.packing_length` is deprecated/unsupported and MUST be rejected by config validation.

Normative behavior:
- For multi-sample packs (pack size >= 2), the pack’s total length MUST satisfy `sum(length_i) <= packing_length`.
- The packer MUST NOT split a single raw sample across multiple packs.
- The packer MUST NOT truncate raw samples as part of packing.
- Oversized samples (where `length >= packing_length`) MUST follow existing long-sample policy:
  - when `training.packing_allow_single_long=true`, emit as a single-sample pack and log it;
  - when `training.packing_allow_single_long=false`, drop deterministically and log it.
- When `training.packing_allow_single_long=true`, a single-sample pack MAY exceed `packing_length` (current behavior treats `length >= packing_length` as “single-long” and emits `[sample]`).
- Within each emitted pack, raw samples MUST retain stable intra-pack order (deterministic and consistent with the pack plan).

#### Scenario: Static packing produces a fixed-length pack dataset for an epoch
- **GIVEN** `training.packing=true` and `training.packing_mode=static`
- **WHEN** the training dataloader is built for an epoch
- **THEN** `len(train_dataset)` equals the number of computed packs for that epoch
- **AND** each dataloader item is either:
  - a multi-sample pack whose summed lengths do not exceed `packing_length`, or
  - a single-long pack (size==1) emitted when `training.packing_allow_single_long=true` and `length >= packing_length`.

### Requirement: Static pack plans are deterministic and epoch-invariant
When static pack-plan mode is selected (`training.packing_mode=static`), the system SHALL compute a pack plan deterministically such that:
- for a fixed base dataset, fixed dataset seed, and fixed packing configuration, the resulting `raw_plan` is identical across runs, and
- for a fixed `world_size` and fixed `training.dataloader_drop_last`, the resulting `aligned_plan(world_size, dataloader_drop_last)` is identical across runs, and
- both plans are identical across DDP ranks (either computed identically or shared from a single source of truth).

Determinism scope:
- Determinism MUST include which raw samples are included in the plan, their grouping into packs, and:
  - each pack’s intra-pack order, and
  - the ordering of packs in the plan.
- Tie-breaking in any heuristic (e.g., binpacking) MUST be stable and deterministic.
- The system SHALL enforce a concrete deterministic ordering:
  - Within each pack, raw sample indices SHALL be ordered ascending.
  - Packs within the plan SHALL be ordered by the smallest raw index in the pack (ascending), with deterministic tie-breaking by lexicographic comparison of the full index lists.

Epoch semantics:
- Static pack-plan mode MUST NOT rely on dataset-level `set_epoch` hooks to reshuffle or rebuild the plan.
- Per-epoch ordering randomness (if desired) is provided by the dataloader sampler shuffling pack indices, not by regenerating the plan.
- Static pack-plan mode MUST be restricted to datasets where per-index encoding is order-invariant and epoch-invariant. Datasets that resample per epoch or depend on `set_epoch` to change the sample schedule (e.g., fusion/mixing schedulers) MUST be rejected (fail fast) when `training.packing_mode=static`.

#### Scenario: Identical inputs produce identical pack plans
- **GIVEN** the same dataset, seed, and packing knobs across two runs
- **WHEN** pack-plan construction runs in static mode
- **THEN** the emitted sequence of packs (index sets and order) is identical between runs.

### Requirement: Static packing enforces deterministic DDP remainder semantics (pad vs drop)
When static pack-plan mode is selected (`training.packing_mode=static`), the system SHALL ensure all DDP ranks execute the same number of training steps per epoch and that `training.dataloader_drop_last` has deterministic semantics under the current ms-swift sized-dataset path (which floors `total_samples // world_size`):
- Let `raw_plan` be the deterministic, world-size-independent pack plan produced by static packing (before any DDP alignment).
- Let `aligned_plan(world_size, dataloader_drop_last)` be the pack plan after applying DDP alignment (this is what training actually consumes).
- Let `N_raw_packs` be `len(raw_plan)`.
- Let `world_size` be the number of learner ranks.
- If `world_size <= 1`, no DDP alignment is needed.
- If `training.dataloader_drop_last=true`, the system MUST truncate the pack plan to `floor(N_raw_packs / world_size) * world_size` packs (drop remainder).
- If `training.dataloader_drop_last=false`, the system MUST deterministically pad the pack plan to `ceil(N_raw_packs / world_size) * world_size` packs by repeating packs (pad/repeat), so all original packs are included and all ranks receive the same number of packs.

Clarification:
- Under the current ms-swift sized-dataset dataloader, if the pack plan length is not a multiple of `world_size`, remainder packs are not indexed at all. Therefore, static packing MUST implement pad/drop semantics by adjusting the pack plan length itself (rather than relying on the sampler).

Normative pad/drop selection rule:
- **Drop** (`training.dataloader_drop_last=true`): `aligned_plan = raw_plan[: floor(N_raw_packs/world_size) * world_size]` (i.e., drop the final `N_raw_packs % world_size` packs in the deterministic plan order).
- **Pad/Repeat** (`training.dataloader_drop_last=false`):
  - Let `pad_needed = (world_size - (N_raw_packs % world_size)) % world_size`.
  - If `pad_needed == 0`, `aligned_plan = raw_plan`.
  - If `pad_needed > 0`, `aligned_plan = raw_plan + raw_plan[:pad_needed]` (repeat packs from the start of the already deterministically ordered `raw_plan`, in order).
  - If `N_raw_packs == 0`, the system MUST fail fast (static packing produced no packs).

Logging / determinism requirements:
- The system SHALL log `N_raw_packs`, `N_aligned_packs=len(aligned_plan)`, `world_size`, and `training.dataloader_drop_last`.
- When padding occurs, the system SHALL log `pad_needed` and the indices of packs repeated from `raw_plan` (e.g., `[0..pad_needed-1]`).
- The system SHALL log a stable checksum for:
  - `raw_plan` (world-size independent), and
  - `aligned_plan(world_size, dataloader_drop_last)` (what training consumes).

#### Scenario: DDP drop_last pads/repeats deterministically when disabled
- **GIVEN** a static pack plan with `N_raw_packs % world_size != 0`
- **AND** `training.dataloader_drop_last=false`
- **WHEN** static packing applies DDP alignment
- **THEN** the final plan length equals `ceil(N_raw_packs/world_size) * world_size`
- **AND** all original packs appear at least once in the aligned plan
- **AND** the additional packs are deterministic repeats.

#### Scenario: DDP drop_last drops a deterministic remainder subset when enabled
- **GIVEN** a static pack plan with `N_raw_packs % world_size != 0`
- **AND** `training.dataloader_drop_last=true`
- **WHEN** static packing applies DDP alignment
- **THEN** the final plan length equals `floor(N_raw_packs/world_size) * world_size`
- **AND** the dropped remainder is deterministic for that run
- **AND** because the plan is epoch-invariant, dropped packs are not consumed in any epoch of that run (note: `world_size` changes which packs are dropped, but for a fixed run configuration it is deterministic).

### Requirement: Static packing length measurement supports multimodal templates
When static pack-plan mode is selected, the system SHALL be able to obtain a deterministic per-sample `length` suitable for pack planning for vision-language samples.

Normative behavior:
- The length used for planning MUST correspond to the encoded token length that will be consumed by the teacher-forced forward pass under the active template.
- The system MUST support computing and storing lengths ahead of packing (e.g., via a cached length store) so the pack plan can be computed without repeatedly re-encoding the full dataset.
- Length computation MUST be deterministic for a fixed dataset record, template, and configuration.
- Any static length/plan cache fingerprint MUST include dataset source identity (for example, resolved `custom.train_jsonl` / `custom.fusion_config` identity) so stale cache reuse across different dataset sources fails fast.
- The run-scoped static cache directory under `training.output_dir` is the primary safety boundary; changing uncaptured length-affecting factors SHOULD use a fresh output directory.
- Static pack-plan mode MUST be restricted to datasets whose per-index encoding (and therefore per-index length) does not depend on call order or non-deterministic preprocessing. If this cannot be guaranteed, the system MUST fail fast with actionable guidance (e.g., disable static packing).
- Static length precompute MAY use rank0 multiprocessing, but computed lengths MUST be written back by dataset index so results are deterministic and byte-identical to serial precompute for the same inputs.
- Non-rank0 ranks MUST continue to use the same cache synchronization path (`training.packing_wait_timeout_s`) regardless of rank0 worker count.

#### Scenario: Multimodal samples have deterministic planning lengths
- **GIVEN** a vision-language dataset sample with an image input under the active template
- **WHEN** static packing computes the sample’s planning length
- **THEN** the computed `length` is deterministic across runs for the same configuration
- **AND** the length is usable to enforce `sum(length_i) <= packing_length` without truncating or splitting samples.

### Requirement: Packing preserves effective_batch_size semantics when batch size is forced to 1
When dataset-level packing wrappers are enabled (dynamic or static) and the system forces `per_device_train_batch_size=1`, the system SHALL preserve the intended effective batch semantics driven by `training.effective_batch_size`:
- If `training.effective_batch_size` is set, `gradient_accumulation_steps` MUST be derived using the *effective* per-device batch size after packing adjustments (i.e., batch size 1).
- If `training.effective_batch_size` is not set, the system MUST still preserve the user’s implied global effective batch after forcing `per_device_train_batch_size=1` by recomputing `gradient_accumulation_steps` from the pre-adjust configuration.
- The system MUST NOT silently change the realized effective batch because `gradient_accumulation_steps` was computed assuming a larger batch size that is later overridden by packing.

Clarification (unit semantics):
- When packing is enabled, `training.effective_batch_size` is defined in units of **packed sequences (“packs”) per `optimizer.step()`**, globally across all ranks.
- Because packing forces `per_device_train_batch_size=1`, one per-device dataloader item corresponds to exactly one packed sequence.
- Therefore, the realized global packs per optimizer step is `world_size * gradient_accumulation_steps` (in packs). The system MUST fail fast when `training.effective_batch_size` is set but not divisible by `world_size`. The system SHOULD log requested vs realized packs/step.
- In static mode, the system SHOULD warn when per-rank packed batches in an epoch produce a partial accumulation window (`per_rank_batches < gradient_accumulation_steps` or `per_rank_batches % gradient_accumulation_steps != 0`), because boundary-step packs/`optimizer.step()` become inexact.

#### Scenario: effective_batch_size is honored after packing forces batch_size=1
- **GIVEN** `training.packing=true` and `training.effective_batch_size` is set
- **AND** the user config sets `training.per_device_train_batch_size` to a value other than 1
- **WHEN** training starts and packing forces `per_device_train_batch_size=1`
- **THEN** the system recomputes/overrides `gradient_accumulation_steps` so the realized global effective batch matches the user’s `training.effective_batch_size` (in units of packed sequences).

#### Scenario: global effective batch is preserved when effective_batch_size is unset
- **GIVEN** `training.packing=true` and `training.effective_batch_size` is not set
- **AND** the user config sets `training.per_device_train_batch_size` to a value other than 1
- **WHEN** training starts and packing forces `per_device_train_batch_size=1`
- **THEN** the system recomputes/overrides `gradient_accumulation_steps` so the realized global effective batch (in units of packed sequences) matches what the user configured before the packing adjustment.

#### Scenario: fail fast when effective_batch_size is not world-size divisible
- **GIVEN** `training.packing=true` and `training.effective_batch_size` is set
- **AND** requested global packs per optimizer step are not exactly realizable by `world_size * gradient_accumulation_steps`
- **WHEN** training initializes packing runtime config
- **THEN** the system fails fast with actionable guidance to set `training.effective_batch_size` divisible by `world_size`.

#### Scenario: static mode warns on partial epoch accumulation windows
- **GIVEN** `training.packing=true` and `training.packing_mode=static`
- **AND** per-rank packed batches in an epoch are smaller than `gradient_accumulation_steps` or leave a remainder
- **WHEN** training validates dataloader/accumulation alignment
- **THEN** the system logs a warning that boundary-step packs per `optimizer.step()` will be inexact for that epoch.

