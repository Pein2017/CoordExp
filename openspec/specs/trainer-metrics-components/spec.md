# trainer-metrics-components Specification

## Purpose
Define the canonical training metrics/logging component contract, including required keys, aggregation semantics, and removal of legacy metrics.

## Requirements
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
The system SHALL preserve the semantics documented in `docs/training/METRICS_LOSSES.md`.
The system MAY remove low-signal or duplicated metric keys when the docs are updated to the new canonical set.
Compatibility aliases are optional and MAY be omitted.

The system SHALL preserve the existing batch-extra key names listed in "Stable batch extras contract".

#### Scenario: Canonical-only metric contract
- GIVEN a training run after metric minimalization
- WHEN only canonical keys are emitted
- THEN removed legacy keys are absent from logs
- AND docs define the canonical key set.

#### Scenario: Key parity
- GIVEN a run with some features enabled/disabled (e.g. token-type metrics, coord loss, packing)
- WHEN running training/evaluation with the refactor enabled
- THEN every emitted metric key matches a documented train key in `docs/training/METRICS_LOSSES.md`
- OR matches the same key prefixed with `eval_` during evaluation (as described in the doc)
- AND removed legacy keys/aliases are absent (no duplicate emission)
- AND feature-conditional keys MAY be absent when their feature is disabled or skipped for a batch


### Requirement: Objective metrics emit canonical provenance keys only (atomic objective atoms; no raw component keys)
For registry-defined objective modules, trainers MUST emit only **atomic objective contributions** under canonical `loss/<provenance>/<atom>` keys and MUST NOT emit raw component loss keys by default.

Definitions:
- An "objective atom" is a post-weighting contribution used in the trainer's total loss.
- For multi-term modules (notably bbox geometry + coord regularization), objective atoms are emitted per sub-term (no pre-summed aggregates).
- "Provenance" encodes which forward/context produced the objective for Stage-2 AB.

Normative behavior:
- Stage-2 AB and rollout-aligned trainers MUST emit only the following objective keys (minimum set), and only when the effective weight is non-zero:
  - Channel-A:
    - `loss/A1_text/{struct_ce,desc_ce}` (GT-anchor forward; token CE objective atoms)
    - `loss/A2_text/struct_ce` (final self-context forward; optional struct/EOS CE stabilizer atom)
    - `loss/A2_coord/{bbox_smoothl1,bbox_ciou}` (final self-context forward; geometry objective atoms)
    - `loss/A2_coord/{coord_token_ce,coord_soft_ce,coord_w1,coord_el1,coord_ehuber,coord_entropy,coord_gate,text_gate}` (final self-context forward; coord_reg objective atoms)
  - Channel-B (rollout context):
    - `loss/B_rollout_text/{struct_ce,desc_ce}` (rollout-context forward; token CE objective atoms)
    - `loss/B_coord/{bbox_smoothl1,bbox_ciou}` (rollout-context forward; geometry objective atoms)
    - `loss/B_coord/{coord_token_ce,coord_soft_ce,coord_w1,coord_el1,coord_ehuber,coord_entropy,coord_gate,text_gate}` (rollout-context forward; coord_reg objective atoms)
- Raw/duplicate loss keys MUST NOT be emitted by default (non-exhaustive):
  - Per-component registry metrics: `loss/token_ce`, `loss/struct_ce`, `loss/desc_ce`, `loss/geo`, `loss/coord_reg`, `loss/coord_gate`, `loss/text_gate`
  - Legacy objective suffixes: `loss/token_ce_obj`, `loss/bbox_geo_obj`, `loss/coord_reg_obj`
  - Trainer-specific aliases: `loss/ce`, `loss/coord`, `loss/coord_prefix`, `loss/coord_tail`

#### Scenario: Canonical-only objective keys
- **WHEN** a Stage-2 or rollout-aligned training step emits objective metrics
- **THEN** emitted keys include only canonical `loss/<provenance>/<atom>` keys for registry-defined objective modules
- **AND** raw component loss keys and legacy aliases are absent.

### Requirement: Coord distribution diagnostics are provenance-split in Stage-2 two-channel
When Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout) (`custom.trainer_variant: stage2_two_channel`) emits coord-distribution diagnostics, it MUST attribute them to the forward surface that produced the logits used for the coord-vocab slice.

Rationale:
- Channel-A runs two teacher-forced forwards (`A1` and `A2`) when `n_softctx_iter >= 2`.
- Stage-1 runs only one forward and reports coord distribution monitors without forward provenance.
- Provenance-splitting makes Stage-2 comparable to Stage-1 and makes self-context drift visible.

Normative behavior:
- Stage-2 two-channel MUST emit coord-vocab distribution monitors under:
  - `coord_diag/A1/*`: computed from Channel-A **A1** logits (GT anchor forward; `it==0`).
  - `coord_diag/A2/*`: computed from Channel-A **A2** logits (final softctx forward; `it==n_softctx_iter-1`), emitted only when `n_softctx_iter > 1`.
  - `coord_diag/B/*`: computed from Channel-B logits (rollout-context forward).
- The set of per-provenance keys SHOULD include (non-exhaustive):
  - `coord_diag/<prov>/coord_tokens_total`
  - `coord_diag/<prov>/{acc_top5,p_gt_mean,margin_mean,expected_bin_mae,expected_bin_abs_err_p90,coord_vocab_mass_mean}`
- Stage-2 two-channel MUST NOT emit bare `coord_diag/*` keys for these monitors (ambiguous provenance is disallowed).

### Requirement: Rollout-only metrics are sparse-emitted
Trainers MUST NOT emit rollout-specific monitor metrics on steps where no rollout was executed.

Normative behavior:
- “Rollout executed” MUST be determined by runtime evidence (e.g., non-zero rollout generation time, non-zero parsed rollout length, or equivalent authoritative signal), not merely by decode configuration.
- For Stage-2 AB Channel-B, `stage2/raw_rollouts > 0` SHOULD be treated as the authoritative runtime signal that a rollout was executed for the step.
- When rollout was not executed, the trainer MUST omit (not emit with `0.0`) rollout-specific keys, including (non-exhaustive):
  - `rollout/precision`, `rollout/recall`, `rollout/f1`
  - `rollout/*` parse/gating/length/coverage diagnostics
  - `time/rollout_generate_s`, `time/rollout_parse_match_s`, `time/rollout_teacher_encode_s`

#### Scenario: A-only Stage-2 does not spam rollout zeros
- **WHEN** Stage-2 AB runs with `stage2_ab.schedule.b_ratio = 0.0`
- **AND** no rollout is executed for the current optimizer step
- **THEN** the emitted training log line contains no `rollout/*` scalar keys (they are absent rather than constant zeros).

### Requirement: Zero-valued timing keys are sparse-emitted
To reduce constant-noise monitors, trainers SHOULD omit timing keys that are identically `0.0` for the current run.

Normative behavior:
- `time/mask_build_s` MUST be omitted when it is not measured by the current trainer (`0.0` placeholder values are disallowed).

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

### Requirement: Sparse gauge aggregation avoids gradient-accumulation dilution
When optimizer-step metrics are aggregated from micro-step buffers, gauge-like keys that may be absent on some micro-steps MUST be averaged over key-observation count (presence count), not total micro-step count.

Counter-like keys MUST remain additive totals.

#### Scenario: Key present on one micro-step only
- GIVEN gradient accumulation with 32 micro-steps
- AND a gauge-like rollout config key present on exactly one micro-step with value `1.0`
- WHEN step-level aggregation finalizes
- THEN the logged value is `1.0` (not `1/32`).


### Requirement: Metrics consume neutral payload contracts
Trainer metrics components SHALL consume neutral payload contracts/events rather than importing trainer-internal symbols.
Metrics logic MUST remain executable in isolation from trainer implementation modules.

#### Scenario: Metrics computation runs with contract payload only
- **GIVEN** a valid neutral metrics payload contract
- **WHEN** metric components compute and report metrics
- **THEN** computation succeeds without importing trainer implementation internals.


### Requirement: Neutral metrics payload schema includes minimum fields and explicit versioning
The neutral metrics payload contract SHALL include a version field and minimum required fields for compatibility checks.
At minimum, payloads MUST include:
- `schema_version` (integer major contract version identifier; initial major version `1`),
- `mode` (`train` or `eval`),
- `global_step` (optimizer-step index),
- `metrics` (key/value map of numeric metrics).

Optional sections (for diagnostics/context) MAY include batch-extras and token/coord summaries, but optional sections MUST NOT be required to parse baseline payloads.
Consumers MUST treat missing or non-integer `schema_version` as invalid payloads.
Consumers MUST fail fast (or explicitly reject) unsupported major schema versions instead of silently mis-parsing payloads.

#### Scenario: Unsupported payload version is rejected explicitly
- **GIVEN** a payload with an unsupported major `schema_version`
- **WHEN** a metrics consumer validates the payload
- **THEN** the payload is rejected with explicit version-mismatch diagnostics
- **AND** the failure is not silently ignored.

#### Scenario: Non-integer payload version is rejected explicitly
- **GIVEN** a payload with missing or non-integer `schema_version`
- **WHEN** a metrics consumer validates the payload
- **THEN** the payload is rejected with explicit schema-version diagnostics
- **AND** the consumer does not attempt fallback parsing.


### Requirement: Diagnostics remain best-effort with explicit first-failure signaling
Diagnostics-only metric paths SHALL remain best-effort but MUST emit an explicit warning on first unexpected exception and disable only the failing diagnostic path.
They MUST NOT silently suppress repeated failures with no signal.

#### Scenario: Unexpected diagnostics exception emits one warning and isolates failure
- **GIVEN** a diagnostics helper throws an unexpected exception
- **WHEN** a training step executes
- **THEN** a warning is emitted at first failure
- **AND** only the failing diagnostic is disabled while training continues.


### Requirement: Canonical module ownership for metrics helpers is unambiguous
Metrics helper implementations SHALL live under `src/metrics/*` and MUST remain importable without importing trainer implementation internals.
Legacy modules under `src/trainers/metrics/*` MAY exist as compatibility shims, but MUST only re-export the canonical implementation and MUST NOT carry divergent behavior.

#### Scenario: Legacy import paths resolve to canonical behavior
- **GIVEN** a consumer imports a metrics helper from a legacy module path
- **WHEN** the helper functions are invoked
- **THEN** behavior matches the canonical `src.metrics.*` implementation
- **AND** no duplicated metric logic exists in the legacy module.


### Requirement: Neutral payload contract has a single canonical implementation
The neutral trainer-metrics payload contract SHALL have a single canonical implementation at `src/metrics/payload_contract.py`.
Any legacy or trainer-side module paths MAY re-export the contract types/helpers for compatibility, but MUST NOT duplicate validation/building logic.

#### Scenario: Payload parsing logic is not duplicated across module paths
- **GIVEN** a consumer imports the payload contract from either canonical or legacy module path
- **WHEN** payloads are validated/built
- **THEN** the same implementation is used in both cases (re-export), preserving consistent validation semantics.


### Requirement: Metrics ownership remains authoritative across overlapping deltas
Within active changes, this capability SHALL be authoritative for trainer-metrics ownership boundaries:
- canonical implementations live under `src/metrics/*`,
- `src/trainers/metrics/*` remains compatibility-shim surface only.

Other active deltas MUST NOT redefine a conflicting canonical home for metrics helper implementations.

#### Scenario: Overlapping change references keep canonical ownership consistent
- **GIVEN** another active change touches trainer-metrics helper imports
- **WHEN** OpenSpec artifacts are reviewed together
- **THEN** canonical ownership remains `src/metrics/*`
- **AND** trainer-side paths are treated as shim/re-export paths only.
