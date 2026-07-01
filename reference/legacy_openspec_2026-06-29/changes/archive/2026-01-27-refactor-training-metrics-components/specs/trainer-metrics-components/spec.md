## ADDED Requirements

These requirements apply to the Stage-1 / standard SFT training path (i.e., `custom.trainer_variant != rollout_matching_sft`) unless explicitly stated otherwise.

### Requirement: Stable batch extras contract
The system SHALL define a centralized contract for "batch extras" fields produced by collators and consumed by trainer hooks.
Batch extras MUST NOT be forwarded into `model(**inputs)`.

The contract MUST define the canonical set of batch extras keys. At minimum, it MUST include:
- `dataset_labels`
- `dataset_segments`
- `pack_num_samples`
- `token_types`
- `instability_meta_json`

#### Scenario: Extras are stripped before model forward
- GIVEN a collator returns a batch dict containing both model inputs and batch extras
- WHEN `Trainer.compute_loss` executes
- THEN batch extras are removed from the dict before calling `model(**inputs)`
- AND the extras are still available for logging/monitoring logic

#### Scenario: New extra is registered
- GIVEN a new diagnostics feature needs to add a new batch extra key
- WHEN the feature is implemented
- THEN the key is added to the centralized batch-extras contract
- AND the trainer strips it before model forward

### Requirement: Stable metric and batch key names
The system SHALL preserve the existing metric key names and semantics documented in `docs/training/METRICS_LOSSES.md`.
The system SHALL preserve the existing batch-extra key names listed in "Stable batch extras contract".

#### Scenario: Key parity
- GIVEN a run with some features enabled/disabled (e.g. token-type metrics, coord loss, packing)
- WHEN running training/evaluation with the refactor enabled
- THEN every emitted metric key matches a documented train key in `docs/training/METRICS_LOSSES.md`
- OR matches the same key prefixed with `eval_` during evaluation (as described in the doc)
- AND existing metric keys are not renamed
- AND feature-conditional keys MAY be absent when their feature is disabled or skipped for a batch

### Requirement: Aggregate-only telemetry
The system SHALL keep metric emission aggregate-only by default. Per-dataset buckets MUST NOT be emitted.

#### Scenario: No per-dataset metrics
- WHEN aggregate metrics are logged during training/evaluation
- THEN keys do not include dataset-specific suffixes/prefixes
- AND no metric dict is created per dataset label

### Requirement: Best-effort diagnostics
Diagnostics-only logic (metrics computation, logging/reporting, and monitoring) MUST be best-effort:
- failures MUST NOT block training
- expected skip conditions (missing required inputs, known alignment mismatches) MUST skip only the affected batch and continue
- unexpected exceptions MUST emit a warning once per diagnostic
- unexpected exceptions MAY disable the failing diagnostic for the remainder of the run

#### Scenario: Token-type misalignment skips the batch
- GIVEN token-type metrics are enabled
- WHEN token-type alignment fails for a batch (e.g., concatenated token_types length differs from labels length)
- THEN token-type metrics are skipped for that batch
- AND training continues without crashing

#### Scenario: Unexpected exception does not block training
- GIVEN a diagnostics module throws an unexpected exception
- WHEN the training step runs
- THEN training continues without crashing
- AND a warning is emitted once indicating the diagnostic failed (and may be disabled)

### Requirement: Fail-fast loss components
Objective-changing logic (loss composition / label masking) MUST fail fast when enabled.
If an enabled loss component cannot be computed, the system MUST raise an error to avoid silently changing the training objective.

#### Scenario: Enabled coord loss failure raises
- GIVEN coord-token distributional supervision is enabled
- WHEN coord loss computation fails for a batch
- THEN the training step raises with a clear error message
- AND training does not proceed with a silently altered objective
