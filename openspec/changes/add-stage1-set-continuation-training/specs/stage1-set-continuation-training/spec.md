## ADDED Requirements

### Requirement: Stage-1 set-continuation is an opt-in setup-path variant
The system SHALL expose subset-conditioned set-continuation training through an
explicit Stage-1 trainer variant, `custom.trainer_variant:
stage1_set_continuation`.

Normative behavior:
- ordinary Stage-1 SFT remains the default when this trainer variant is absent,
- the variant MUST route to a dedicated trainer before the ordinary Stage-1
  trainer factory,
- the variant MUST exclude ordinary one-sequence Stage-1 loss mixin
  composition,
- the variant MUST use a raw-sample-preserving collator path with
  `remove_unused_columns=false`,
- the variant MUST reject static/dynamic dataset packing before any pack-plan
  dataset is built.

#### Scenario: Ordinary Stage-1 remains default
- **GIVEN** no set-continuation trainer variant is configured
- **WHEN** Stage-1 training is launched
- **THEN** ordinary Stage-1 SFT behavior SHALL remain unchanged.

#### Scenario: Set-continuation variant owns setup and loss composition
- **GIVEN** `custom.trainer_variant: stage1_set_continuation`
- **WHEN** training setup resolves trainer and collator behavior
- **THEN** setup routes to the set-continuation trainer
- **AND** ordinary Stage-1 loss mixins are not dynamically composed
- **AND** raw per-sample metadata needed for branch construction survives
  collation.

### Requirement: V1 supports coord-token object coordinates only
V1 of this paradigm SHALL support object coordinates only when the effective
candidate branch rendering contains `<|coord_*|>` coordinate literals.

Normative behavior:
- setup MUST require `custom.coord_tokens.enabled=true`,
- setup MUST reject raw-text integer coordinate training for this trainer
  variant,
- validation MUST check the effective rendered candidate surface, not merely
  filename patterns or raw JSONL value types,
- numeric source coordinates MAY be accepted only if the active coord-token
  renderer converts them to `<|coord_*|>` in candidate branches.

#### Scenario: Raw-text coordinate mode fails fast
- **GIVEN** `custom.trainer_variant: stage1_set_continuation`
- **AND** the effective candidate branch contains raw integer coordinate text
- **WHEN** setup validates the training surface
- **THEN** setup fails with actionable guidance that v1 requires
  `<|coord_*|>` object coordinates.

### Requirement: Full-entry MP objective uses coord-aware candidate scoring
The trainer SHALL compute multi-positive supervision over complete serialized
object entries, not token-wise positive mixtures.

Normative candidate score:

```text
score(o) = score_struct_desc(o) + score_coord(o)
score_struct_desc(o) = sum log P_full_vocab(y_t)
                       over non-coord candidate-entry labels
score_coord(o) = sum log P_coord_vocab(coord_t)
                 over coord-token candidate-entry labels
```

The coord-vocab score MUST be computed over the configured coord-token
sub-vocabulary and MUST NOT use ordinary full-vocab CE at coord-token slots as
the candidate-score contribution.

#### Scenario: Multiple remaining objects are all positives
- **GIVEN** prefix subset `S`
- **AND** remaining observed candidates `R = O - S`
- **WHEN** candidate scores are computed in exact mode
- **THEN** each `score(o)` equals the intact candidate-entry log probability
  under the coord-aware scoring rule
- **AND** `loss/mp = -logsumexp(score(o) for o in R)`.

#### Scenario: Coordinates are not token-wise mixed
- **GIVEN** candidate entries with description and `<|coord_*|>` bbox tokens
- **WHEN** MP loss is computed
- **THEN** each candidate is scored as an intact serialized entry
- **AND** coordinate tokens from different candidates SHALL NOT be mixed as
  independent positives.

### Requirement: Canonical object-entry and closure spans are index based
The trainer SHALL derive candidate-entry and structural-closure spans from the
same canonical CoordJSON serialization used by Stage-1 assistant targets.

Normative behavior:
- `assistant_payload.objects` is the canonical object-entry source for `O`,
  not `sample["objects"]`,
- fragment construction MUST be index-based and emit object-index span metadata,
- content-based substring search MUST NOT be used to identify object spans,
- candidate-entry labels SHALL include the object-entry delimiter and
  terminator needed for append semantics,
- candidate-entry labels SHALL exclude global schema closure, assistant suffix,
  EOS, and chat-template stop tokens.

#### Scenario: Duplicate identical entries remain addressable
- **GIVEN** two serialized object entries with identical text
- **WHEN** candidate fragments are built
- **THEN** the renderer identifies spans by object index
- **AND** each duplicate entry remains separately selectable as a candidate.

#### Scenario: Global closure is outside candidate-entry score
- **GIVEN** a candidate object entry
- **WHEN** labels are selected for `score(o)`
- **THEN** object-entry terminator tokens are included
- **AND** global schema-closure tokens are excluded from the candidate-entry
  score.

### Requirement: Subset prefix sampling is deterministic and configurable
The trainer SHALL support configurable subset-prefix sampling with empty-prefix,
random-subset, leave-one-out, and small full-prefix modes.

Recommended default ratio:

```text
empty_prefix = 0.30
random_subset = 0.45
leave_one_out = 0.20
full_prefix = 0.05
```

Normative behavior:
- prefix order SHOULD default to random and expose dataset/canonical ablations,
- sampler RNG MUST be a pure function of resolved seed, epoch, sample identity,
  rank, and documented microstep salt,
- invalid mode selections for small object counts MUST renormalize over valid
  modes deterministically,
- if `R != empty`, at least one candidate MUST be scored,
- if `R = empty` and weak structural-close weight is zero, the sample
  contributes metrics only and is excluded from the objective denominator.

#### Scenario: Leave-one-out preserves held-out recovery
- **GIVEN** object set `O`
- **WHEN** leave-one-out sampling is selected
- **THEN** the prefix equals `O - {o_i}` for one sampled object
- **AND** the remaining set contains `o_i`.

#### Scenario: Full-prefix sampling produces no MP term
- **GIVEN** `S = O`
- **WHEN** the full-prefix mode is selected
- **THEN** `R` is empty
- **AND** no MP candidate-entry loss is computed
- **AND** optional weak structural-close supervision applies only if configured.

### Requirement: Candidate scoring modes expose exact and sampled logZ semantics
The trainer SHALL support exact all-remaining candidate scoring and uniform
candidate subsampling, with explicit logZ metric names for each scope.

Normative logZ names:

```text
mp/logZ_scored_raw = logsumexp(score(o) for o in scored candidates C)
mp/logZ_remaining_exact = logsumexp(score(o) for o in all remaining R)
mp/logZ_remaining_est = mp/logZ_scored_raw + log(|R| / |C|)
```

Normative behavior:
- exact mode MUST score all remaining candidates,
- subsampled mode MUST log both remaining-object count and scored-candidate
  count,
- `mp/logZ_remaining_exact` MAY be emitted only when all remaining candidates
  were scored,
- sampled PEM MUST use `mp/logZ_remaining_est` rather than raw sampled logZ,
  unless a separately named raw-sampled PEM ablation is explicitly configured.

#### Scenario: Exact mode scores all remaining candidates
- **GIVEN** exact scoring mode
- **WHEN** `R` contains `m` objects
- **THEN** the trainer scores all `m` remaining candidates
- **AND** logs `mp/num_candidates_scored = m`.

#### Scenario: Subsampled mode records estimator scope
- **GIVEN** uniform candidate subsampling with `K`
- **WHEN** `R` contains more than `K` objects
- **THEN** the trainer scores at most `K` candidates
- **AND** records `candidate_scoring_mode=uniform_subsample`
- **AND** records whether logZ is raw scored-candidate mass or
  uniform-importance estimated remaining mass.

### Requirement: Repeated-forward branch semantics are explicit
The v1 implementation SHALL use repeated independent forwards for candidate
branches.

Normative behavior:
- each candidate branch conditions on the same image, prompt, and sampled
  prefix,
- candidate branches MUST NOT attend to each other,
- prefix gradients are non-detached but recomputed per branch,
- no shared prefix cache or branch attention mask is used in v1,
- resolved config or effective runtime artifacts MUST record these semantics.

#### Scenario: Candidate branches are isolated
- **GIVEN** candidates `A`, `B`, and `C`
- **WHEN** candidate scores are computed
- **THEN** each branch conditions on the shared image, prompt, and prefix only
- **AND** candidate `B` does not attend to candidate `A`.

### Requirement: Structural-close losses do not target chat EOS
The trainer SHALL distinguish object-entry terminators from global detection-list
structural closure and SHALL NOT use chat/EOS tokens as v1 stop targets.

Definitions:

```text
P_close_start(S) = P(first structural closure token | image, prompt, prefix)
logP_close_sequence(S) = sum_t log P(closure_t | image, prompt, prefix, closure_<t)
```

Normative behavior:
- structural closure spans MUST be derived from canonical CoordJSON
  serialization,
- `loss/anti_close_start = -log(1 - P_close_start(S))` applies only when
  `R != empty` and its weight is non-zero,
- `loss/weak_schema_close = final_close_weight * (-logP_close_sequence(S))`
  applies only when `R = empty` and its weight is non-zero,
- `<|im_end|>`, `<|end_of_text|>`, EOS, chat-template stops, and object-entry
  close tokens are not part of `P_close_start` or `logP_close_sequence`,
- this trainer MUST NOT emit `loss/eod` unless a future config explicitly
  supervises chat-template EOD/EOS tokens.

#### Scenario: Remaining observed GT exists
- **GIVEN** `R != empty`
- **WHEN** anti-close is enabled
- **THEN** the trainer penalizes the first structural closure decision token
- **AND** does not penalize chat EOS or assistant stop tokens.

#### Scenario: No observed GT remains
- **GIVEN** `R = empty`
- **WHEN** weak structural-close supervision is enabled
- **THEN** the trainer teacher-forces the full structural closure sequence
- **AND** scales it by `final_close_weight`, which may be `0.0`.

### Requirement: Positive-evidence margin supports replacement-mode in v1
The trainer SHALL support fixed-threshold PEM in v1, disabled by default, and
the source-idea Group E objective SHALL use PEM as a replacement for MP rather
than an additive penalty on top of MP.

Normative behavior:
- `positive_evidence_margin.mode` MUST support `disabled` and `replace_mp`,
- when PEM is enabled, exactly one of `rho` or `log_rho` MUST be configured,
- `positive_evidence_margin.threshold_space` MUST be recorded and v1 SHALL use
  `full_entry_logZ`,
- PEM benchmark profiles MUST record threshold calibration provenance or an
  explicit statement that the threshold is an authored fixed ablation value,
- PEM uses `logZ_remaining_exact` in exact mode and `logZ_remaining_est` in
  uniform-subsample mode,
- when `mode=replace_mp`, `loss/pem` is optimized and `loss/mp_diagnostic` is
  logged without contributing to total loss,
- additive PEM, if later supported, MUST be a separately named ablation and
  MUST NOT be described as preserving probability space for latent positives.

#### Scenario: PEM disabled
- **GIVEN** `positive_evidence_margin.mode: disabled`
- **WHEN** loss is computed for `R != empty`
- **THEN** the trainer optimizes the standard MP objective.

#### Scenario: PEM replacement mode
- **GIVEN** `positive_evidence_margin.mode: replace_mp`
- **AND** a configured threshold
- **WHEN** loss is computed for `R != empty`
- **THEN** the trainer optimizes
  `max(0, log_threshold - logZ_remaining_*)`
- **AND** does not also add `-logZ_remaining_*` to the optimized total loss.

### Requirement: Branch-local auxiliary losses are toggleable through adapters
Compatible coord and geometry auxiliary losses SHALL be toggleable only through
set-continuation branch-local adapters.

Normative behavior:
- branch-local adapters MUST reuse canonical low-level helpers where available,
- ordinary one-sequence Stage-1 mixins MUST NOT be composed for this variant,
- aux reductions MUST be mean-like within candidate entry and then uniformly
  averaged over scored candidates with valid atoms,
- responsibility-weighted auxiliary losses are not a v1 mode,
- setup MUST fail fast if an enabled aux config has no branch-local adapter.

#### Scenario: Coord soft loss uses branch-local coord positions
- **GIVEN** `custom.coord_soft_ce_w1.enabled: true`
- **WHEN** candidate branches are scored
- **THEN** coord soft CE/W1 terms apply only to coord-token positions inside
  scored candidate entries
- **AND** aux metrics are logged separately from MP/PEM.

#### Scenario: Geometry aux lacks required state
- **GIVEN** `custom.bbox_geo.enabled: true` or
  `custom.bbox_size_aux.enabled: true`
- **WHEN** branch-local decoded bbox state is unavailable
- **THEN** setup or loss computation fails fast with actionable diagnostics.

### Requirement: Mechanism metrics are canonical and aggregation-safe
The trainer SHALL emit variant-specific mechanism metrics with canonical names
and aggregation semantics.

Required metric families:
- losses: `loss/mp`, `loss/mp_diagnostic`, `loss/pem`,
  `loss/anti_close_start`, `loss/weak_schema_close`,
  `loss/aux_coord_soft_ce_w1`, `loss/aux_bbox_geo`,
  `loss/aux_bbox_size`;
- candidate counts and denominators: `mp/num_prefix_objects`,
  `mp/num_remaining_objects`, `mp/num_candidates_scored`,
  `mp/scored_candidate_fraction`, `mp/samples_with_candidates`,
  `mp/samples_full_prefix`, `mp/loss_mp_denominator_samples`;
- logZ scope: `mp/logZ_scored_raw`, `mp/logZ_remaining_exact`,
  `mp/logZ_remaining_est`, `mp/logZ_estimator`;
- responsibility: `mp/responsibility_entropy_scored`,
  `mp/max_responsibility_scored`, `mp/min_responsibility_scored`;
- length diagnostics: `mp/candidate_entry_tokens_*`,
  `mp/candidate_logprob_sum_*`, `mp/candidate_logprob_per_token_*`,
  `mp/candidate_coord_token_fraction_*`,
  `mp/candidate_logprob_per_coord_token_*`,
  `mp/candidate_logprob_per_noncoord_token_*`,
  `mp/responsibility_vs_length_corr`, `mp/valid_length_corr_samples`;
- budget: `mp/branch_forwards_per_sample`, `mp/prefix_tokens_mean`,
  `mp/candidate_tokens_scored_mean`, `mp/total_candidate_tokens_scored`,
  `mp/repeated_forward_token_ratio_vs_baseline`;
- structural close: `stop/p_close_start_when_remaining_exists`,
  `stop/p_continue_start_when_remaining_exists`,
  `stop/p_close_start_when_remaining_empty`,
  `stop/logp_close_sequence_when_remaining_empty`,
  `stop/p_final_schema_token_teacher_forced`;
- aux adapter counters: `aux/<name>/candidate_count`,
  `aux/<name>/position_count`, `aux/<name>/skipped_candidates`.

Validity rules:
- responsibility entropy over one scored candidate is `0.0`, and max/min
  responsibility are both `1.0`,
- candidate standard-deviation metrics over one candidate emit `0.0`,
- `mp/responsibility_vs_length_corr` is emitted only when at least two
  candidates are scored and candidate lengths vary; otherwise the sample is
  excluded from `mp/valid_length_corr_samples`.

#### Scenario: Full-prefix sample has no MP denominator
- **GIVEN** a full-prefix sample with `R = empty`
- **WHEN** metrics are logged
- **THEN** MP denominator metrics exclude that sample from MP loss averaging
- **AND** structural-close metrics record the empty-remaining state.

### Requirement: Run artifacts preserve set-continuation provenance
Set-continuation training SHALL preserve the existing training artifact family
and add set-continuation-specific provenance.

Normative behavior:
- `resolved_config.json` records the full defaulted
  `custom.stage1_set_continuation` block,
- `effective_runtime.json` or `pipeline_manifest.json` records repeated-forward
  semantics, branch isolation, prefix-gradient semantics, collator path,
  packing rejection, encoded-cache policy, and candidate scoring mode,
- `experiment_manifest.json` records benchmark group identity, comparator,
  hypothesis, metric scope, eval view, and artifact pointers,
- `run_metadata.json` and data-provenance sidecars remain present,
- a set-continuation metric schema version is recorded.

#### Scenario: Tiny smoke run records provenance
- **GIVEN** a tiny set-continuation training smoke run
- **WHEN** bootstrap artifacts are written
- **THEN** the run artifacts include enough information to recover sampler,
  branch, PEM, structural-close, aux, and benchmark settings after training.

### Requirement: Benchmark matrix profiles are config-addressable
The change SHALL provide checked-in or generated config profiles with stable
benchmark identities for Groups A-F.

Required groups:
- Group A: ordinary SFT baseline;
- Group B: ordinary SFT with structural schema-close masked/downweighted;
- Group C: one-prefix exact MP;
- Group D: subset MP plus anti-close-start;
- Group E: fixed-threshold PEM replacement-mode;
- Group F: leave-one-out emphasis.

These A-F labels are local to this OpenSpec change and SHALL be treated as the
canonical benchmark glossary for this implementation, even where
`progress/directions/full_idea_v5.md` used intermediate letter names
differently.

Normative behavior:
- documentation may explain groups but MUST NOT substitute for config identity,
- benchmark identity MUST live in a typed strict config surface, e.g.
  `benchmark.group_id`, `benchmark.control_group_id`,
  `benchmark.intended_variable`, and `benchmark.comparability_label`,
- each group profile MUST resolve `benchmark.group_id`,
  `benchmark.control_group_id`, dataset, prompt variant, object field order,
  seed, resolution/preset, sample/optimizer-step budget, checkpoint identity,
  inference decoding controls, and eval plan,
- each group profile MUST pin coord-token settings, effective coord-slot scoring
  surface, and aux objective settings (`coord_soft_ce_w1`, `bbox_geo`,
  `bbox_size_aux`) so group labels do not hide extra objective changes,
- canonical A-F profiles MUST use `training.packing: false`; packed SFT may be
  a separately named throughput/control ablation but MUST NOT be silently mixed
  into the A-F accuracy-comparable matrix,
- benchmark manifests MUST record the intended variable versus the comparator,
  comparability label, realized branch/token budget, realized prefix-mode
  coverage, and realized aux/coord-scoring settings,
- benchmark reports MUST state eval scope, eval view, slice/sample count,
  prediction volume, AP/AP50/AP75, precision/recall where available, and
  sparse-label caveats.

#### Scenario: A-F configs resolve to a controlled matrix
- **GIVEN** the benchmark profiles for Groups A-F
- **WHEN** configs are resolved
- **THEN** each profile has a stable group id
- **AND** any difference from its comparator is recorded as an intended
  benchmark variable.
