## ADDED Requirements

### Requirement: Stage-2 AB exposes residual-state trie correction as explicit alias objective paths

Stage-2 AB SHALL expose the residual-state trie correction objective through
explicit YAML configuration. The public module names `stage2_trie_ce` and
`residual_set_correction` SHALL be aliases for the same residual-state dynamic
valid-set semantics, not separate supervision algorithms.

Normative behavior:

- The objective module name MAY be `stage2_trie_ce` or
  `residual_set_correction`.
- `stage2_trie_ce` SHALL support `application.preset: rollout_trie_hard_ce` as a
  compatibility-facing alias.
- `residual_set_correction` SHALL support
  `application.preset: rollout_self_prefix`.
- The module is valid only for Channel-B objective application.
- Existing `hard_sft` and `token_ce` baselines MUST remain selectable and
  unchanged unless explicitly configured otherwise.
- Residual trie configs MUST NOT use legacy edited final targets, sorted/tail
  FN insertion, privileged-segment candidate aggregation, or a
  one-merged-forward target as canonical target semantics.
- A Channel-B config MUST NOT enable both `stage2_trie_ce` and
  `residual_set_correction`; select exactly one alias.
- Removed duplicate, bbox, geometry, coordinate repair, and coord diagnostic
  modules MUST remain rejected in the residual trie path.

#### Scenario: Residual-set objective is explicit opt-in

- **WHEN** a Stage-2 AB config selects `stage2_trie_ce` or
  `residual_set_correction`
- **THEN** Channel-B target construction uses offline self-prefix correction
  samples
- **AND** both names route to the same residual-state trie target IR and valid
  set marginal loss.

### Requirement: Residual trie Stage-2 config is strict and minimal

Stage-2 AB SHALL validate residual trie config keys strictly and avoid
configuration explosion.

Normative behavior:

- `stage2_trie_ce.config` and `residual_set_correction.config` MUST require
  exactly one prepared rollout input key in v1:
  - `prepared_rollout_jsonl`
- Both alias configs MUST accept these optional keys with defaults:
  - `expected_num_rollouts`: default `4`;
  - `base_seed`: default `17`;
  - `lambda_type`: default `1.0`;
  - `lambda_inner`: default `1.0`;
  - `fallback_loss_weight`;
  - `lambda_ul_promoted`: default `0.5`;
  - `label_conflict_weight`: default `0.25`;
  - `commit_iou_threshold`: default `0.75`;
  - `duplicate_burst_iou_threshold`: default `0.95`;
  - `ul_cluster_iou_threshold`: default `0.9`;
  - `ul_gray_iou_low`: default `0.30`;
  - `ul_consensus_ratio`: default `1.0`;
  - `min_ul_valid_rollouts`: default `4`;
  - `clean_gt_sft_mix`: default `0`;
  - `strict_prepared_rollout_tokens`: default `true`;
  - `strict_builder_invariants`: default `true`;
  - `require_real_prepared_rollouts`: default `false`.
- Both alias configs MUST reject every other key, including the legacy
  candidate-trie keys `support_weight`, `balance_weight`, `struct_weight`,
  `desc_weight`, `coord_hard_ce_weight`, `eos_weight`, and `normalization`.
- The config MUST NOT expose a coordinate repair policy in v1.
- The config MUST NOT expose default-on online generation inside training.
- Unknown residual trie keys MUST fail fast.

#### Scenario: Missing prepared rollout JSONL fails fast

- **WHEN** a residual trie config omits `prepared_rollout_jsonl`
- **THEN** validation fails before trainer initialization
- **AND** the error identifies
  `stage2_ab.pipeline.objective[name=residual_set_correction|stage2_trie_ce].config.prepared_rollout_jsonl`.

#### Scenario: Coordinate repair config is rejected

- **WHEN** a residual trie config declares `coord_span_policy:
  bbox_tail_from_anchor`
- **THEN** validation fails fast
- **AND** the error explains that Stage-2 v1 does not repair raw-rollout
  coordinates.

### Requirement: Stage-2 AB prepared-rollout path is offline and replayable

Stage-2 AB SHALL treat prepared rollout artifacts as the input surface for the
residual-set objective.

Normative behavior:

- Residual trie training MUST read prepared rollout attempts from offline data.
- Fully online generate-while-training MUST NOT be the v1 default or required
  path.
- Prepared record validation MUST fail/drop on missing `response_token_ids`
  except in an explicit legacy fallback mode.
- When `require_real_prepared_rollouts=true`, prepared record validation MUST
  reject fixture/preflight records and require every loaded attempt to carry
  `producer_mode: real`.
- The training run MUST record prepared artifact provenance and generation
  config hashes where available.

#### Scenario: Offline prepared rollouts are replayed

- **WHEN** a residual-set training run starts
- **THEN** it reads fixed prepared rollout attempts
- **AND** the target builder constructs training atoms from those attempts
  without calling generation inside the optimizer loop.

#### Scenario: Fixture prepared rollouts are rejected for real mini experiments

- **WHEN** a residual-set training run sets
  `require_real_prepared_rollouts=true`
- **AND** its prepared rollout JSONL contains records without
  `producer_mode: real`
- **THEN** Channel-B training fails before target construction
- **AND** the error explains that fixture/preflight records are pipeline-smoke
  data, not train128 experiment evidence.

### Requirement: Stage-2 AB residual-set diagnostics are compact and stable

Stage-2 AB SHALL emit compact diagnostics sufficient to validate the residual
correction contract without creating many files by default.

Normative behavior:

- Step monitors SHOULD include counts for committed GT rows, committed UL rows,
  pending UL candidates, promoted UL clusters, invalid geometry, malformed rows,
  duplicate rows, clean/dirty correction events, EOS/continue targets,
  truncation drops, re-encoded legacy prefixes, clean-success skips, type mass,
  wrong-type mass, active atom count, atom weight sum, raw atom loss sum, and
  sequence loss.
- Decode-mode slices SHOULD be emitted for clean-success rate, invalid rate,
  dirty-correction events, and promoted-UL support.
- `monitor_dumps/ul_clusters.jsonl` is the canonical default UL review file
  when artifact dumping is enabled.
- A standalone `label_conflict_review.jsonl` MUST NOT be created by default.

#### Scenario: Dirty-prefix recovery is visible

- **WHEN** dirty-prefix recovery contributes atoms
- **THEN** monitor diagnostics include dirty correction counts and atom/weight
  summaries
- **AND** malformed segments that were masked but retained as context are
  counted separately.

## MODIFIED Requirements

### Requirement: Stage-2 two-channel training supports a config-declared objective and diagnostics pipeline

When `custom.trainer_variant: stage2_two_channel`, the system SHALL use an
explicit YAML-declared objective/diagnostics pipeline.

Normative behavior:

- `stage2_ab.pipeline` MUST be present.
- Canonical clean-prefix baseline ordering remains:
  1. `token_ce`
  2. `hard_sft` when the config intentionally requests the baseline
     selected-path objective.
- `stage2_trie_ce` and `residual_set_correction` are explicit opt-in aliases for
  the residual-state trie objective; they are not part of the clean-prefix
  baseline ordering and cannot both occupy Channel-B.
- Live Stage-2 AB configs MUST omit `loss_duplicate_burst_unlikelihood`; the
  removed objective has no compatibility alias.

#### Scenario: Residual trie objective is not ordered as clean-prefix baseline

- **WHEN** a Stage-2 config selects `stage2_trie_ce` or
  `residual_set_correction`
- **THEN** validation treats it as the explicit residual-state trie objective
  path
- **AND** it does not require the clean-prefix baseline objective ordering.

### Requirement: Stage-2 Two-Channel module names are stable and discoverable

Stage-2 Two-Channel SHALL provide a strict module registry for its pipeline
modules, and the module names SHALL be stable so YAML-declared experiments
remain auditable.

Normative minimum objective module names for this contract:

- `token_ce`
- `hard_sft`
- `stage2_trie_ce`
- `residual_set_correction`

Removed objective module names:

- `loss_duplicate_burst_unlikelihood`
- `bbox_geo`
- `bbox_size_aux`
- `coord_reg`

Normative behavior:

- Unknown module names MUST fail fast before training starts.
- Error messages MUST list the unknown module name and available Stage-2
  Two-Channel module names.
- The residual trie objective MUST reject removed coordinate/geometry/duplicate
  modules rather than aliasing them.

#### Scenario: Removed duplicate loss remains rejected

- **WHEN** a residual trie config declares `loss_duplicate_burst_unlikelihood`
- **THEN** validation fails fast before trainer initialization
- **AND** duplicate burst remains diagnostic/provenance only.

### Requirement: Peer-rollout Channel-B triage remains scoped outside residual trie correction

The existing clean-prefix Channel-B contract SHALL use independent peer rollout
attempts and remain scoped to legacy/baseline Channel-B objective paths. It
SHALL NOT apply to
`stage2_trie_ce` or `residual_set_correction`.

Normative behavior:

- Peer rollout views, pseudo-positive semantics, duplicate-control survivor
  editing, and current-attempt clean-sequence final targets apply only to the
  canonical clean-prefix Channel-B path and never to residual trie aliases.
- When `stage2_ab.pipeline.objective[].name` is `stage2_trie_ce` or
  `residual_set_correction`, Stage-2 AB SHALL consume offline prepared
  `rollout_attempt` records and SHALL NOT apply pseudo-positive,
  privileged-rollout aggregation, or clean-prefix target semantics.
- Any future mapping from old role-specific rollout fields into
  `rollout_attempts[]` must be an explicit offline legacy adapter, not the
  residual trie runtime contract.

#### Scenario: Residual trie path bypasses clean-prefix target construction

- **WHEN** a Stage-2 config selects `stage2_trie_ce` or
  `residual_set_correction`
- **THEN** Channel-B does not build an edited clean-prefix target
- **AND** K prepared rollout attempts are treated as equal self-prefix samples.
