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
