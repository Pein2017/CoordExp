## ADDED Requirements

### Requirement: Stage-2 AB residual-set objective module is named and typed

Stage-2 AB SHALL expose residual-set correction through a strict objective
module named `residual_set_correction`.

Normative behavior:

- The module name SHALL be `residual_set_correction`.
- The module SHALL support `application.preset: rollout_self_prefix`.
- The module SHALL be valid only for Channel-B objective application.
- On Channel-B, `residual_set_correction` SHALL be mutually exclusive with
  legacy target-realization modules such as `token_ce`, `hard_sft`, and
  `stage2_trie_ce`; Channel-A objectives MAY remain separate pipeline entries.
- `residual_set_correction.config` SHALL be strict and SHALL accept only:
  - `rollin_policy`
  - `rollin_resample_policy`
  - `base_seed`
  - `coord_span_policy`
  - `strict_builder_invariants`
  - `lambda_ul_promoted`
  - `lambda_continue_margin`
  - `continue_margin_m`
  - `coverage_strength`
  - `num_rollouts`
  - `min_ul_valid_rollouts`
  - `ul_consensus_ratio`
  - `ul_geometry`
  - `artifact_policy`
- `rollin_policy` default SHALL be `random_valid_branch`.
- `rollin_resample_policy` default SHALL be `fixed_event`.
- `base_seed` default SHALL be `17`.
- `coord_span_policy` default SHALL be `bbox_tail_from_anchor`.
- `lambda_ul_promoted` default SHALL be `0.5`.
- `lambda_continue_margin` default SHALL be `0.0`.
- `continue_margin_m` default SHALL be `0.0`.
- `coverage_strength` default SHALL be `0.0`; nonzero coverage is an explicit
  within-valid calibration ablation and MUST NOT be required for the core
  residual-set marginal objective.
- `num_rollouts` SHALL be owned by `residual_set_correction.config` and SHALL
  NOT require `stage2_ab.channel_b.pseudo_positive.enabled=true`.
- `ul_geometry` SHALL record explicit complete-link gate names for IoU minimum,
  center-distance scale, area-ratio maximum, and aspect-ratio maximum.
- `stage2_ab.channel_b.pseudo_positive.enabled=true` SHALL be rejected when the
  residual-set objective is selected, to avoid mixing legacy pseudo-positive
  semantics with rollout-local UL promotion.

#### Scenario: Residual-set module config is accepted

- **WHEN** a Stage-2 AB pipeline includes
  `name: residual_set_correction`, `channels: [B]`, and
  `application.preset: rollout_self_prefix`
- **THEN** config validation accepts the residual-set objective module when its
  config uses only the allowlisted keys
- **AND** Channel-A objective modules remain separately configurable.

#### Scenario: Residual-set rejects legacy pseudo-positive mixing

- **WHEN** a Stage-2 AB config selects `residual_set_correction`
- **AND** `stage2_ab.channel_b.pseudo_positive.enabled=true`
- **THEN** config validation fails fast with guidance to use
  `residual_set_correction.config.num_rollouts` and UL mining settings instead.

### Requirement: Stage-2 AB exposes residual-set correction as an explicit objective path

Stage-2 AB SHALL expose the residual-set self-prefix correction objective as an
explicit YAML-selected pipeline path without silently changing existing baseline
objective behavior.

Normative behavior:

- The residual-set path MUST be selected through the typed Stage-2 pipeline
  configuration.
- Hard-SFT and existing Stage-2 trie/text objective baselines MUST remain
  selectable for ablation.
- New residual-set configs MUST NOT use the legacy edited-anchor final target,
  sorted/tail FN insertion, or one merged teacher-forced forward as their
  canonical target semantics.
- Legacy `stage2_ab.channel_b.insertion_order` and duplicate-control knobs MAY
  remain in inherited base configs, but for the residual-set objective they SHALL
  be target-construction inactive. They MAY contribute provenance/diagnostics,
  but MUST NOT prune repeated-object correction events, order corrected targets,
  or perform FN insertion.
- Config validation MUST reject removed duplicate-specific loss modules and
  removed bbox/coord auxiliary modules in the residual-set path.

#### Scenario: Residual-set objective path is selected explicitly

- **WHEN** a Stage-2 AB config selects the residual-set correction objective
- **THEN** Channel-B target construction uses independent correction samples
  from valid self-prefix rollouts
- **AND** existing baseline objective modules are not mutated implicitly.

#### Scenario: Removed duplicate loss remains rejected

- **WHEN** a residual-set Stage-2 config declares
  `loss_duplicate_burst_unlikelihood`
- **THEN** config validation fails before trainer initialization
- **AND** duplicate-like behavior remains diagnostic/provenance only.

#### Scenario: Legacy ordering knobs are inactive for residual-set targets

- **WHEN** a residual-set Stage-2 config inherits
  `stage2_ab.channel_b.insertion_order`
- **THEN** residual-set correction target construction ignores that ordering
  knob
- **AND** target ordering is determined by self-prefix residual-state actions.

### Requirement: Stage-2 AB residual-set rollouts use K-valid sample semantics

Stage-2 AB SHALL treat K rollout attempts as independent self-prefix correction
samples for the residual-set objective.

Normative behavior:

- K rollout outputs MUST NOT be collapsed into one cross-prefix multiple-target
  objective or one empirical pseudo-label distribution.
- Each grammar-valid eligible rollout MAY contribute its own correction sample.
- Invalid or malformed rollouts MUST be counted by reason and excluded from
  K-valid UL consensus denominators.
- The builder order MUST be UL consensus first, then per-rollout earliest
  actionable correction selection against rollout-local `G*_k`.
- The residual-set rollout count MUST come from
  `residual_set_correction.config.num_rollouts`, not from legacy
  `stage2_ab.channel_b.triage_posterior.num_rollouts`.

#### Scenario: Two valid rollouts produce separate correction samples

- **WHEN** two valid rollouts for the same image have different grammar-valid
  self-prefix contexts
- **THEN** the residual-set path builds separate correction samples
- **AND** it does not merge their target supports into one teacher-forced
  forward.

#### Scenario: UL consensus precedes correction selection

- **WHEN** an unmatched object is promoted by K-valid consensus
- **THEN** per-rollout correction selection uses the rollout-local universe that
  includes that promoted UL member
- **AND** the object is not selected earlier as an FP boundary for its first
  occurrence.

#### Scenario: Residual K does not require legacy pseudo-positive

- **WHEN** `residual_set_correction.config.num_rollouts` is `3` or greater
- **AND** `stage2_ab.channel_b.pseudo_positive.enabled=false`
- **THEN** config validation may accept the residual-set objective
- **AND** the legacy non-residual requirement that pseudo-positive-disabled
  triage has two rollout views does not apply to the residual-set K.

### Requirement: Stage-2 AB emits residual-set and UL observability

Stage-2 AB residual-set runs SHALL emit diagnostics that expose correction
selection, residual provenance, valid-action mass, and UL mining decisions.

Normative behavior:

- Metrics MUST distinguish labeled remaining, UL remaining, and mixed remaining
  premature-stop cases.
- Metrics MUST expose repeated-object boundary corrections separately from
  duplicate-specific losses.
- Metrics MUST expose valid-action support, selected token probability,
  illegal/STOP mass where applicable, and corrected-roll-in diagnostics.
- Residual-set Stage-2 metric keys MUST use the stable prefix
  `stage2_ab/channel_b/residual_set/`.
- When monitor/debug/smoke artifact dumping is enabled, Stage-2 AB SHALL write
  discovered UL clusters under the artifact root at relative path
  `ul_clusters.jsonl`.
- Detailed UL artifact rows MUST include support rollouts, support ratio,
  geometry stats, overlap stats to labeled GT/emitted objects, consumed-target
  overlap/quarantine evidence, decision, reason, and member boxes.

#### Scenario: Premature stop provenance is visible

- **WHEN** a residual-set run encounters raw STOP with remaining objects
- **THEN** metrics identify whether the remaining set is labeled-only,
  UL-only, or mixed
- **AND** STOP/continuation probabilities are reported without requiring a
  continuation-margin loss.

#### Scenario: UL clusters are reviewable under monitor dumps

- **WHEN** monitor dumps are enabled and UL mining discovers clusters
- **THEN** Stage-2 AB writes `ul_clusters.jsonl` under the active artifact root
- **AND** the rows contain support, geometry, overlap, decision, and member-box
  evidence.

## MODIFIED Requirements

### Requirement: Stage-2 Two-Channel module names are stable and discoverable
Stage-2 Two-Channel SHALL provide a strict module registry for its pipeline modules, and the module names SHALL be stable so
YAML-declared experiments remain auditable.

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

Normative minimum diagnostics module names: none.

Removed diagnostics module names:
- `coord_diag`

Normative behavior:
- Unknown module names MUST fail fast before training starts.
- Error messages MUST list the unknown module name and available Stage-2 Two-Channel module names.
- Baseline module and preset names used in docs, specs, config validation, and
  the runtime registry MUST be aligned before adding new residual-set baseline
  configs.

#### Scenario: Unknown Stage-2 two-channel module names fail fast
- **WHEN** `stage2_ab.pipeline` references an objective module name not present in the Stage-2 registry
- **THEN** trainer initialization fails fast
- **AND** the error includes the unknown name and allowed module names.

### Requirement: Stage-2 AB Channel-B uses anchor/explorer triage with an optional pseudo-positive multi-view extension
The legacy clean-prefix Channel-B path SHALL build its clean teacher-forced
target from one greedy anchor rollout plus one or more explorer rollouts when
`custom.trainer_variant: stage2_two_channel` selects a legacy/baseline
Channel-B objective path:

- one anchor rollout using greedy / deterministic decoding,
- one or more explorer rollouts using stochastic decoding configured under `stage2_ab.channel_b.triage_posterior`.

Normative behavior:

- this requirement applies to legacy/baseline clean-prefix Channel-B objective
  paths such as `token_ce`, `hard_sft`, and `stage2_trie_ce`, not to
  `residual_set_correction`,
- when `stage2_ab.channel_b.pseudo_positive.enabled=false`, total rollout views MUST remain `2` (`1` anchor + `1` explorer),
- when `stage2_ab.channel_b.pseudo_positive.enabled=true`, total rollout views MUST equal `stage2_ab.channel_b.triage_posterior.num_rollouts`,
- each rollout MUST independently reuse the existing bounded salvage + strict record acceptance + bbox-valid filtering + sequential dedup + configured Stage-2 assignment path,
- GT-backed semantics MUST inherit the existing Channel-B accepted-clean assignment + gating contract,
- the final positive target MUST be built by editing the **anchor** clean sequence rather than rebuilding a union order,
- pseudo-positive candidate discovery MUST start from unmatched anchor clean objects and use explorer agreement only as support evidence,
- explorer-only non-GT-backed objects MUST NOT be promoted into clean-prefix positives,
- a GT hit found only on the explorer side MUST project to `recovered_fn`, not to anchor retention.

Residual-set correction is an explicit opt-in path with separate self-prefix
correction samples and rollout-local UL promotion. It MUST NOT use this
requirement's edited-anchor final target as its canonical target semantics.

#### Scenario: Channel-B builds the final target from the anchor clean sequence
- **GIVEN** anchor and explorer rollouts were both produced for a legacy clean-prefix Channel-B sample
- **WHEN** the trainer constructs the teacher-forced target
- **THEN** it starts from the anchor clean accepted sequence
- **AND** it preserves anchor order for retained objects
- **AND** it does not rebuild a union ordering over anchor and explorer objects.

#### Scenario: Residual-set path does not build an edited anchor target
- **GIVEN** a Stage-2 config selects `residual_set_correction`
- **WHEN** Channel-B constructs correction samples
- **THEN** it builds independent self-prefix correction samples
- **AND** it does not use the edited-anchor final target as the canonical target.

#### Scenario: Explorer-only GT hit does not keep a bad anchor object positive
- **GIVEN** an anchor/explorer pair-or-singleton record where the anchor side misses GT and the explorer side matches GT
- **WHEN** the trainer projects triage evidence into training actions
- **THEN** the outcome is `recovered_fn`
- **AND** the bad anchor object is not kept as an anchor GT-backed positive.

#### Scenario: Explorer-only non-GT-backed object does not become a pseudo-positive prefix object
- **GIVEN** an explorer object that does not correspond to any unmatched anchor clean object
- **WHEN** Channel-B projects pseudo-positive evidence into the final clean prefix
- **THEN** that explorer-only non-GT-backed object is not promoted into a new prefix positive
- **AND** pseudo-positive selection remains anchored on unmatched anchor clean objects.

### Requirement: Channel-B v3 uses one merged teacher-forced forward
The canonical legacy v1 v3 clean-prefix contract SHALL realize
`L(clean_anchor) + L(explore-derived corrections)` through one merged
teacher-forced forward on the edited anchor target.

Normative behavior:

- this requirement applies only to legacy/baseline clean-prefix Channel-B
  objective paths,
- the trainer MUST run one teacher-forced forward on the final edited target for
  that legacy path,
- positive, weighted-FN, and dead-anchor UL terms MUST be derived from that same forward,
- the trainer MUST NOT require a second explore teacher-forced payload in the canonical legacy v1 contract,
- `residual_set_correction` is exempt from this one-merged-forward target
  realization and instead compiles independent correction events into
  `TeacherForcingTargetIR`.

#### Scenario: Single-forward v3 target realization
- **WHEN** a legacy clean-prefix Channel-B v3 sample is prepared
- **THEN** all loss terms are derived from a single teacher-forced forward over the edited anchor target
- **AND** no second teacher-forced explore payload is required.

#### Scenario: Residual-set realization uses correction-event IR
- **WHEN** a residual-set Stage-2 sample is prepared
- **THEN** the builder compiles correction events into `TeacherForcingTargetIR`
- **AND** the legacy one-edited-target realization does not apply.
