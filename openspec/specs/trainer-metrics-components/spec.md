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
The system SHALL preserve the semantics documented in `docs/training/METRICS.md`.
The system MAY remove low-signal or duplicated metric keys when the docs are updated to the new canonical set.
Compatibility aliases are optional and MAY be omitted.

The system SHALL preserve the existing batch-extra key names listed in "Stable batch extras contract".

Normative behavior for the clean-prefix Channel-B contract:
- `docs/training/METRICS.md` MUST define the canonical training keys added by this contract.
- `docs/training/STAGE2_RUNBOOK.md` MUST define the corresponding Channel-B behavior and interpretation.
- Removed raw-prefix wording and removed legacy metric names MUST NOT linger in the canonical docs after implementation lands.

#### Scenario: Canonical-only metric contract
- GIVEN a training run after metric minimalization
- WHEN only canonical keys are emitted
- THEN removed legacy keys are absent from logs
- AND docs define the canonical key set.

#### Scenario: Key parity
- GIVEN a run with some features enabled/disabled (e.g. token-type metrics, coord loss, packing)
- WHEN running training/evaluation with the refactor enabled
- THEN every emitted metric key matches a documented train key in `docs/training/METRICS.md`
- OR matches the same key prefixed with `eval_` during evaluation (as described in the doc)
- AND removed legacy keys/aliases are absent (no duplicate emission)
- AND feature-conditional keys MAY be absent when their feature is disabled or skipped for a batch

#### Scenario: Canonical duplicate metrics are documented
- **GIVEN** a training run after the clean-prefix Channel-B feature lands
- **WHEN** duplicate-ul and duplicate-collapse metrics are emitted
- **THEN** their canonical key names are documented in `docs/training/METRICS.md`
- **AND** the Channel-B contract that produces them is documented in `docs/training/STAGE2_RUNBOOK.md`.

### Requirement: Metric namespace hierarchy is explicit and stable
The system SHALL keep the metric namespace hierarchy explicit so operators can infer meaning from the key shape alone.

Normative behavior:
- `loss/<provenance>/<atom>` denotes a post-weighting objective atom.
- `coord_diag/<metric>` is reserved for Stage-1-style bare coord diagnostics.
- `coord_diag/<provenance>/<metric>` is reserved for provenance-split Stage-2 coord diagnostics.
- `gradmon/<metric>` and `gradmon/<group>/<term>` are reserved for optional loss-gradient diagnostics.
- `rollout/*` and `eval/detection/*, eval/parsing/*, eval/description/*, eval/config/*, eval/runtime/*` remain distinct training-vs-eval families.
- Internal reducer-helper keys MUST remain underscore-prefixed and MUST NOT appear in the final logged payload.

#### Scenario: Key shape disambiguates objective atoms vs diagnostics
- **WHEN** a log record contains `loss/B_coord/coord_soft_ce` and `coord_diag/B/expected_bin_mae`
- **THEN** the former is understood as a training objective atom
- **AND** the latter is understood as a diagnostic-only monitor
- **AND** the two are not ambiguous despite sharing the same provenance token.

### Requirement: Loss-gradient monitor diagnostics are optional, sparse, and aggregation-safe
When `custom.extra.loss_gradient_monitor.enabled=true`, the system SHALL expose optional `gradmon/*`
diagnostics without changing the training objective or optimizer behavior.

Normative behavior:
- Per-term monitor keys MAY include:
  - `gradmon/loss_raw/<term>`
  - `gradmon/loss_ema_norm/<term>`
  - `gradmon/grad_norm/<term>`
  - `gradmon/cos_to_total/<term>`
- Aggregate monitor keys MAY include:
  - `gradmon/grad_norm_ratio_max_over_median`
  - `gradmon/neg_cosine_pair_frac`
  - `gradmon/neg_cosine_pair_pct`
  - `gradmon/neg_cos_to_total_frac`
  - `gradmon/num_terms`
  - `gradmon/shared_param_count`
  - `gradmon/shared_param_numel`
- `gradmon/*` keys are diagnostics-only, sparse-emitted on monitor steps, and MUST be absent rather than emitted as placeholder zeros on non-monitor steps.
- For Stage-2 training logs, `gradmon/*` gauges MUST be computed locally first and synchronized only through the existing optimizer-step reducers.
- Sparse `gradmon/*` gauges MUST preserve their observation-weighted value at the finalized optimizer-step log boundary (no dilution across unobserved micro-steps or packed forwards).
- `time/gradmon_s` MAY be emitted and MUST follow the active trainer's existing `time/*` reducer semantics.
- `gradmon/*` keys MUST NOT introduce per-dataset buckets.

#### Scenario: Sparse gradmon keys are not diluted
- **GIVEN** gradient accumulation or packed-forward execution where `gradmon/*` keys appear on only a subset of micro-steps
- **WHEN** the optimizer-step payload is finalized
- **THEN** the finalized `gradmon/*` value reflects only the observed monitor steps
- **AND** the value is not divided by the total number of unobserved micro-steps.

#### Scenario: Non-monitor steps omit gradmon keys
- **GIVEN** `loss_gradient_monitor.interval_steps` is greater than `1`
- **WHEN** a training step is not a monitor step
- **THEN** the step log omits `gradmon/*` keys
- **AND** `time/gradmon_s` is absent for that step.

### Requirement: Objective metrics emit canonical provenance keys only (atomic objective atoms; no raw component keys)
For registry-defined objective modules, trainers MUST emit only **atomic objective contributions** under canonical `loss/<provenance>/<atom>` keys and MUST NOT emit raw component loss keys by default.

Definitions:
- An "objective atom" is a post-weighting contribution used in the trainer's total loss.
- For multi-term modules (notably bbox geometry + coord regularization), objective atoms are emitted per sub-term (no pre-summed aggregates).
- "Provenance" encodes which forward/context produced the objective for Stage-2 AB.

Normative behavior:
- Stage-2 AB and rollout-aligned trainers MUST emit only the following objective keys (minimum set), and only when the effective weight is non-zero:
  - Channel-A:
    - `loss/text/{struct_ce,desc_ce}` (GT-anchor forward; token CE objective
      atoms)
    - `loss/coord/{bbox_smoothl1,bbox_ciou,bbox_log_wh,bbox_oversize,coord_token_ce,coord_soft_ce,coord_w1,coord_gate,text_gate}`
      when Channel-A bbox/coord supervision is active on the single-pass
      GT-anchor forward
  - Channel-B (rollout context):
    - `train/optimization/{loss_structure_ce,loss_description_ce,loss_dead_anchor_suppression}` (rollout-context forward; token/UL objective atoms)
    - `loss/B_coord/{bbox_smoothl1,bbox_ciou}` (from `bbox_geo`; rollout-context forward; geometry objective atoms)
    - `loss/B_coord/{bbox_log_wh,bbox_oversize}` (from `bbox_size_aux`; rollout-context forward; size-aux objective atoms)
    - `loss/B_coord/{coord_token_ce,coord_soft_ce,coord_w1,coord_gate,text_gate}` (rollout-context forward; coord_reg objective atoms)
- `loss/A1_*` objective atoms MUST NOT be emitted by active Stage-2
  two-channel training.
- `loss/A2_*` objective atoms MUST NOT be emitted by active Stage-2
  two-channel training.
- Raw/duplicate loss keys MUST NOT be emitted by default (non-exhaustive):
  - Per-component registry metrics: `loss/token_ce`, `loss/struct_ce`, `loss/desc_ce`, `loss/geo`, `loss/bbox_size_aux`, `loss/coord_reg`, `loss/coord_gate`, `loss/text_gate`
  - Legacy objective suffixes: `loss/token_ce_obj`, `loss/bbox_geo_obj`, `loss/coord_reg_obj`
  - Trainer-specific aliases: `loss/ce`, `loss/coord`, `loss/coord_prefix`, `loss/coord_tail`

#### Scenario: Canonical-only objective keys
- **WHEN** Stage-2 two-channel training emits objective metrics
- **THEN** Channel-A text metrics use `loss/text/*`
- **AND** Channel-A bbox/coord metrics use `loss/coord/*`
- **AND** the same run does not emit `loss/A1_*` or `loss/A2_*`.

#### Scenario: Channel-B emits loss_dead_anchor_suppression as a canonical objective atom
- **WHEN** a Channel-B training step applies duplicate unlikelihood
- **THEN** the emitted objective key is `train/optimization/loss_dead_anchor_suppression`
- **AND** no raw alias key for duplicate-unlikelihood is emitted.

### Requirement: Channel-B duplicate-collapse metrics are explicit and aggregation-safe
The trainer metrics contract SHALL expose duplicate-collapse diagnostics with explicit gauge-vs-counter naming.

Normative gauges:
- `dup/max_desc_count`
- `dup/saturation_rate`

Normative count-like metrics:
- `dup/near_iou90_pairs_same_desc_count`
- `dup/near_iou90_pairs_any_desc_count`
- `stage2_ab/channel_b/dup/N_raw_bbox_valid`
- `stage2_ab/channel_b/dup/N_clean_accepted`
- `stage2_ab/channel_b/dup/N_duplicates`
- `stage2_ab/channel_b/dup/N_duplicate_bursts`
- `stage2_ab/channel_b/dup/N_ul_boundaries`
- `stage2_ab/channel_b/dup/N_ul_skipped_no_divergence`

Normative behavior:
- Count-like metrics MUST use `/N_`, `_count`, `_total`, `_sum`, `_num`, or `_den` naming so optimizer-step aggregation treats them as additive totals.
- Gauge-like metrics MUST remain mean-like and MUST NOT masquerade as counters.

#### Scenario: Duplicate counters aggregate additively across micro-steps
- **WHEN** duplicate count-like metrics are emitted from multiple micro-steps in one optimizer step
- **THEN** the finalized step metric is the additive total
- **AND** the result is not diluted by mean-style aggregation.

### Requirement: Coord distribution diagnostics are provenance-split in Stage-2 two-channel
When Stage-2 Two-Channel Teacher Forcing (Expectation/Rollout) (`custom.trainer_variant: stage2_two_channel`) emits coord-distribution diagnostics, it MUST attribute them to the forward surface that produced the logits used for the coord-vocab slice.

Rationale:
- Channel-A now runs one GT-anchored teacher-forced forward.
- Stage-1 runs only one forward and reports coord distribution monitors without forward provenance.
- Provenance-splitting still matters for distinguishing Channel-A anchor
  behavior from Channel-B rollout behavior.

Normative behavior:
- Stage-2 two-channel MUST emit coord-vocab distribution monitors under:
  - `coord_diag/*`: computed from Channel-A GT-anchor logits.
  - `coord_diag/B/*`: computed from Channel-B logits (rollout-context forward).
- `coord_diag/A1/*` and `coord_diag/A2/*` MUST NOT be emitted.
- The set of per-provenance keys SHOULD include (non-exhaustive):
  - `coord_diag/<prov>/coord_tokens_total`
  - `coord_diag/<prov>/{acc_top5,p_gt_mean,margin_mean,expected_bin_mae,expected_bin_abs_err_p90,coord_vocab_mass_mean}`
- The bare `coord_diag/*` namespace is the canonical single-pass Channel-A
  coord group after self-context removal.

#### Scenario: A1/A2 coord diagnostics are absent after deprecation
- **WHEN** Stage-2 two-channel executes with coord diagnostics enabled
- **THEN** coord diagnostics are emitted only under supported provenance keys
- **AND** the same run does not emit `coord_diag/A1/*` or `coord_diag/A2/*`.

### Requirement: Loss-gradient monitor uses the normal single-pass coord group for Channel-A
When Stage-2 loss-gradient monitoring emits per-term coord diagnostics, it MUST
use the normal single-pass Channel-A coord group name rather than iterative
Channel-A provenance names.

Normative behavior:
- Channel-A coord monitor terms MUST use `coord/<atom>`.
- Channel-B coord monitor terms MAY continue to use `B_coord/<atom>`.
- `A1_coord/<atom>` and `A2_coord/<atom>` MUST NOT be emitted by the active
  loss-gradient monitor for Stage-2 two-channel training.

#### Scenario: Stage-2 gradmon no longer emits A1/A2 coord groups
- **WHEN** Stage-2 two-channel loss-gradient monitoring is enabled
- **THEN** emitted coord monitor term names may include `coord/<atom>` and
  `B_coord/<atom>`
- **AND** the same monitor payload does not emit `A1_coord/<atom>` or
  `A2_coord/<atom>`.

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
To reduce constant-noise monitors, trainers SHALL omit timing keys that are identically `0.0` for the current run.

Normative behavior:
- `time/mask_build_s` MUST be omitted when it is not measured by the current trainer (`0.0` placeholder values are disallowed).

#### Scenario: Unmeasured mask-build timing is omitted
- **WHEN** a trainer does not measure mask-build timing for a step
- **THEN** `time/mask_build_s` is absent from emitted metrics
- **AND** a constant `0.0` placeholder is not emitted.

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

### Requirement: DDP metric aggregation MUST be deterministic and strict
When `torch.distributed` is initialized (`world_size > 1`), any cross-rank metric aggregation MUST satisfy:

- **Deterministic key set**:
  - Either the metric key set is statically known (preferred), OR
  - the key set is computed via a distributed union step (e.g., `all_gather_object`) and MUST succeed on all ranks.
- **Deterministic ordering**:
  - the ordered key list used to build any reduction tensor MUST be identical on every rank (e.g., `sorted(keys)` after union).
- **Strict failure semantics**:
  - aggregation MUST NOT fall back to rank-local metrics when DDP is initialized,
  - any unexpected exception in aggregation MUST abort all ranks with coordinated error propagation.
  - all ranks MUST participate in union/reduction collectives even if their local metric set is empty (missing keys MUST reduce as zeros so tensor shapes match).

#### Scenario: Metric key union failure aborts rather than “proceeding locally”
- **GIVEN** DDP is initialized
- **AND** aggregation requires a distributed key union step
- **WHEN** the key union step fails on any rank
- **THEN** all ranks terminate with a clear error message
- **AND** the system does not continue with rank-local key lists.

#### Scenario: All-reduce tensor shape is identical across ranks
- **GIVEN** DDP is initialized
- **WHEN** aggregation performs an `all_reduce` over a tensor built from metric keys
- **THEN** the tensor length and key ordering are identical on every rank
- **AND** ranks with empty local metrics still build the same ordered tensor length (zero-filled)
- **AND** the system does not hang due to mismatched tensor shapes or skipped collectives.

### Requirement: Best-effort diagnostics MUST NOT perform collective sync
Diagnostics-only best-effort paths MUST be local-only and MUST NOT perform distributed collective sync.

Under DDP, any diagnostic that requires collectives (e.g., key synchronization) MUST be:
- strict (fail-fast), or
- disabled globally via a rank-symmetric decision computed with collectives (so all ranks agree).

#### Scenario: A diagnostics failure never causes collective divergence
- **GIVEN** DDP is initialized
- **WHEN** a diagnostics helper raises unexpectedly
- **THEN** the system does not skip a required collective on only a subset of ranks
- **AND** the job either fails fast or disables the diagnostic via a rank-symmetric decision.

### Requirement: Channel-B triage metrics are explicit and aggregation-safe
The trainer metrics contract SHALL expose the v3 triage bookkeeping separately from legacy duplicate diagnostics.

Normative count-like metrics:

- `train/triage/gt_backed_count`
- `train/triage/unlabeled_consistent_count`
- `train/triage/dead_anchor_count`
- `train/triage/explorer_only_dead_count`
- `train/triage/recovered_ground_truth_count`

Normative numerator / denominator metrics:

- `train/triage/recovered_ground_truth_rate_num`
- `train/triage/recovered_ground_truth_rate_den`
- `train/triage/dead_anchor_rate_num`
- `train/triage/dead_anchor_rate_den`

Normative behavior:

- count-like metrics MUST aggregate additively across micro-steps,
- numerator / denominator metrics MUST aggregate additively across micro-steps,
- monitor-dump payloads SHOULD expose both rollout views plus the final triage decision for high-signal samples.

#### Scenario: Triage counts aggregate additively
- **WHEN** multiple micro-steps emit triage count metrics within one optimizer step
- **THEN** the finalized step metrics are additive totals rather than mean-diluted gauges.

### Requirement: Stage-1 bbox size aux SHALL use canonical geometry atom names
The Stage-1 trainer MUST emit canonical geometry atom names for the
`bbox_size_aux` plugin.

When Stage-1 bbox size auxiliary supervision is enabled through the Stage-1 aux
plugin host, the trainer SHALL emit the same geometry atom names used by later
plugin-owned geometry objectives.

Normative behavior:

- Stage-1 single-forward emission MUST use:
  - `loss/geo/{bbox_log_wh,bbox_oversize}`
- Stage-1 MUST NOT invent a second Stage-1-only loss namespace for the same
  bbox-size plugin math,
- because Stage-1 has one GT forward, these keys remain unsplit by provenance.

#### Scenario: Stage-1 plugin-owned bbox size aux uses canonical loss atoms
- **GIVEN** Stage-1 bbox size auxiliary is enabled
- **WHEN** a training step emits geometry objective atoms
- **THEN** the emitted keys use `loss/geo/{bbox_log_wh,bbox_oversize}`
- **AND** the same atom names remain recognizable relative to Stage-2
  provenance-split geometry atoms.
