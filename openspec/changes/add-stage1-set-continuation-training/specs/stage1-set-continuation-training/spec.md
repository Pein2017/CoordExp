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
- **THEN** each `score(o)` equals the intact candidate-continuation log
  probability under the coord-aware scoring rule
- **AND** the scored continuation includes `entry(o) + ", "` when observed
  objects remain after appending `o`, or `entry(o) + "]}"` when `o` exhausts
  the observed remaining set
- **AND** the optimized disabled-PEM objective is candidate-balanced
  token-normalized continuation CE,
- **AND** `loss/mp_diagnostic = -logsumexp(score(o) for o in R)`.

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
  EOS, and chat-template stop tokens,
- structural-close branches SHALL use a close-ready prefix without an append
  delimiter after the last selected object.

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
- prefix order SHOULD default to random and expose a `dataset` preserved-order
  ablation,
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
- when PEM is disabled, the trainer MUST optimize candidate-balanced
  token-normalized continuation CE over scored candidates and log the MP/logZ
  objective only as `loss/mp_diagnostic`,
- budget fallback estimator metadata MAY be `sampled_raw` for disabled-PEM
  diagnostics and `uniform_importance` for PEM threshold-loss diagnostics,
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
- **AND** records raw scored-candidate mass and, when needed for PEM or
  diagnostics, uniform-importance estimated remaining mass.

### Requirement: Repeated-forward branch semantics are explicit
The v1 implementation SHALL use repeated independent branch semantics for
candidate branches, even when the physical branch runtime uses activation
checkpointing or smart candidate-branch batching to improve memory or
throughput.

Normative behavior:
- each candidate branch conditions on the same image, prompt, and sampled
  prefix,
- candidate branches MUST NOT attend to each other,
- prefix gradients are non-detached,
- model-side prefix computation is repeated or recomputed per branch according
  to the selected branch runtime,
- if the runtime groups candidates into physical branch batches, each batch row
  MUST remain an independent `prefix + candidate` sequence and candidates MUST
  NOT attend to one another,
- exact CPU/tokenization prefix encoding reuse MAY be used only when branch
  inputs, labels, masks, image tensors, and span offsets are parity-equivalent
  to uncached construction,
- no GPU KV prefix cache or branch attention mask is used in the immediate v1
  production bridge,
- resolved config or effective runtime artifacts MUST record these semantics.

#### Scenario: Candidate branches are isolated
- **GIVEN** candidates `A`, `B`, and `C`
- **WHEN** candidate scores are computed
- **THEN** each branch conditions on the shared image, prompt, and prefix only
- **AND** candidate `B` does not attend to candidate `A`.

#### Scenario: Batched candidate rows preserve independent branch semantics
- **GIVEN** the `smart_batched_exact` runtime groups candidate branches into a
  physical model batch
- **WHEN** candidate scores are computed
- **THEN** each batch row is scored as the same full `prefix + candidate`
  sequence that the legacy retained-graph runtime would have scored
- **AND** the per-candidate scores are scattered back to the original
  per-sample candidate order before MP/PEM aggregation.

### Requirement: Train-forward runtime supports exact suffix logits and explicit synchronization policies
The set-continuation trainer SHALL expose a config-first train-forward branch
runtime that can preserve exact MP semantics while reducing peak memory and
wasted branch work through supervised-suffix logits and explicit DDP
synchronization policy.

Normative behavior:
- when `custom.stage1_set_continuation.train_forward` is omitted, the resolved
  config MUST preserve current verified behavior:
  `branch_runtime.mode=retained_graph`, budget policy disabled, approximate
  fallback disabled, `logits.mode=full`, `ddp_sync.candidate_padding=max_count`,
  prefix encoding cache disabled, and GPU KV cache disabled,
- the runtime MUST support a legacy `retained_graph` mode for parity tests and
  tiny debugging fixtures,
- the runtime MUST support `smart_batched_exact` as an exact selected-candidate
  throughput bridge that batches candidate branch rows while preserving
  independent branch semantics,
- the runtime MUST support `train_forward.logits.mode=full` for backward
  compatibility and rollback,
- the runtime MUST support `train_forward.logits.mode=supervised_suffix`; in this
  mode each candidate or close branch computes the earliest supervised label
  position, passes `logits_to_keep` for logits from
  `first_supervised_label_pos - 1` onward, and crops labels/masks to the same
  suffix before loss computation,
- `supervised_suffix` MUST be represented as exact objective fidelity when all
  authored candidates are scored because it omits only unsupervised prefix
  logits,
- the runtime MUST support `train_forward.ddp_sync.candidate_padding=max_count`
  for backward-compatible padding forwards and
  `train_forward.ddp_sync.candidate_padding=none` for production execution with
  no zero-loss candidate padding forwards,
- `candidate_padding=none` under DDP MUST require
  `training.ddp_broadcast_buffers=false`; otherwise ranks with different local
  candidate counts can desynchronize on DDP forward-time buffer-broadcast
  collectives,
- `candidate_padding=none` MUST still record the local candidate forward count,
  cross-rank max candidate count, selected padding policy, and zero padding
  forwards,
- the runtime MUST support a `checkpointed_exact` mode that returns the same
  MP/PEM loss semantics while recomputing branch internals during backward
  instead of retaining all branch activations from forward,
- `checkpointed_exact` MUST keep `use_cache=false` for model forwards,
- all gradient-bearing branch objective atoms MUST come from the same
  checkpointed computation, or setup MUST reject `checkpointed_exact` for the
  active objective configuration. The immediate bridge MAY reject
  `checkpointed_exact` when branch-local aux losses are enabled,
- checkpoint recomputation MUST preserve stochastic-layer determinism for the
  active model/config, for example by preserving RNG state,
- if deterministic recompute cannot be guaranteed for the active model/config,
  setup MUST either fail before training or explicitly downgrade through an
  authored approximate fallback while recording approximate objective fidelity,
- DDP padding or synchronization forwards MUST honor the selected branch runtime
  and MUST NOT retain full branch graphs when `checkpointed_exact` is active,
- unsupported branch runtime modes MUST fail config validation with an
  actionable message,
- branch runtime mode and checkpoint settings MUST be recorded in
  `effective_runtime.json`.

#### Scenario: Smart branch batching is exact for selected candidates
- **GIVEN** `train_forward.branch_runtime.mode: smart_batched_exact`
- **AND** `train_forward.branch_batching.enabled: true`
- **WHEN** the trainer scores selected candidate branches
- **THEN** the candidate scores MUST equal the scores produced by scoring the
  same encoded branches one by one under retained-graph mode, within normal
  floating-point tolerance
- **AND** `loss/mp`, `loss/pem`, and logZ metrics MUST be reduced from those
  same per-candidate scores.

#### Scenario: Smart branch batching borrows ms-swift-style scheduling
- **GIVEN** candidate branches with heterogeneous encoded lengths
- **WHEN** the smart branch batcher forms physical branch batches
- **THEN** branch length MUST be first-class scheduling metadata
- **AND** the scheduler SHOULD use ms-swift-style constant-volume grouping
  within length/suffix buckets when the `binpacking` dependency is available
- **AND** if the dependency is unavailable, the runtime MUST use a deterministic
  fallback scheduler and expose that scheduler choice in telemetry.

#### Scenario: Dataset-level packing remains rejected
- **GIVEN** `custom.trainer_variant: stage1_set_continuation`
- **AND** `train_forward.branch_runtime.mode: smart_batched_exact`
- **WHEN** setup validates `training.packing`
- **THEN** dataset-level `training.packing=true` and `training.eval_packing=true`
  remain invalid because MP branches are sampled and materialized inside the
  trainer.

#### Scenario: True padding-free branch packing is deferred
- **GIVEN** the immediate smart-batching bridge
- **WHEN** candidate branches are physically scheduled
- **THEN** the runtime MUST NOT require Qwen3-VL padding-free packed metadata
  (`cu_seq_lens`, packed `text_position_ids`, or branch attention masks)
- **AND** any future padding-free branch-packing runtime MUST be separately
  labeled and validated against per-segment suffix-logit and multimodal
  alignment contracts.

#### Scenario: Supervised suffix logits preserve candidate score semantics
- **GIVEN** `train_forward.logits.mode: supervised_suffix`
- **AND** a candidate branch contains mixed coord-token and non-coord supervised
  labels
- **WHEN** the candidate score is computed
- **THEN** it equals the score computed from full-sequence logits
- **AND** objective fidelity is unchanged.

#### Scenario: Supervised suffix logits preserve close-loss semantics
- **GIVEN** `train_forward.logits.mode: supervised_suffix`
- **AND** close-start or close-sequence labels begin after an unsupervised prefix
- **WHEN** close losses are computed
- **THEN** they equal the losses computed from full-sequence logits.

#### Scenario: DDP no-padding executes only local candidates
- **GIVEN** `train_forward.ddp_sync.candidate_padding: none`
- **AND** the local rank has fewer candidate branches than the cross-rank maximum
- **WHEN** branch scoring runs
- **THEN** the rank executes only local real candidate forwards plus the close
  branch
- **AND** `mp/ddp_candidate_padding_forwards` is `0`.

#### Scenario: Checkpointed exact mode preserves objective semantics
- **GIVEN** `train_forward.branch_runtime.mode: checkpointed_exact`
- **AND** exact candidate selection is within budget
- **WHEN** a sample with multiple remaining candidates is scored
- **THEN** the optimized MP loss is still
  `-logsumexp(score(o) for o in all remaining candidates)`
- **AND** candidate branch internals are eligible for recomputation during
  backward rather than retained from the first forward pass.

#### Scenario: Omitted train-forward config preserves verified behavior
- **GIVEN** `custom.trainer_variant: stage1_set_continuation`
- **AND** no `custom.stage1_set_continuation.train_forward` block is authored
- **WHEN** config resolution completes
- **THEN** the trainer uses retained-graph repeated branch scoring
- **AND** budget fallback and prefix encoding cache are disabled.

#### Scenario: Checkpointed exact rejects unsupported aux semantics
- **GIVEN** `train_forward.branch_runtime.mode: checkpointed_exact`
- **AND** a branch-local auxiliary objective is enabled
- **WHEN** the implementation cannot compute that auxiliary objective from the
  same differentiable checkpointed branch computation
- **THEN** setup fails with an actionable message rather than detaching aux
  gradients.

#### Scenario: Retained-graph mode remains available for parity
- **GIVEN** `train_forward.branch_runtime.mode: retained_graph`
- **WHEN** a tiny deterministic fixture is trained
- **THEN** the trainer uses the legacy retained candidate-graph behavior
- **AND** checkpointed exact mode can be compared against it for loss and
  gradient parity.

#### Scenario: DDP padding branches follow checkpointed runtime when enabled
- **GIVEN** `train_forward.branch_runtime.mode: checkpointed_exact`
- **AND** `train_forward.ddp_sync.candidate_padding: max_count`
- **AND** distributed synchronization requires zero-loss padding branches
- **WHEN** padding forwards are executed
- **THEN** they use the checkpointed branch runtime or an equivalently
  memory-bounded synchronization path.

### Requirement: Train-forward budget policy falls back explicitly when approximation is allowed
The set-continuation trainer SHALL make long-sample/candidate budget decisions
before expensive branch scoring and SHALL use explicit approximate fallback
instead of silently changing exact MP semantics.

Normative behavior:
- the budget policy MUST be config-first under
  `custom.stage1_set_continuation.train_forward`,
- exact execution remains exact only when all remaining candidates selected by
  the authored candidate mode are scored and the authored candidate mode is
  exact,
- the runtime MUST distinguish authored candidate scoring mode from runtime
  objective fidelity,
- authored `candidates.mode=uniform_subsample` MUST remain approximate even when
  the train-forward budget policy does not fire,
- when an authored budget would be exceeded and fallback is enabled, the trainer
  MUST switch to `approximate_uniform_subsample` with the configured maximum
  candidate count,
- fallback MUST be deterministic from the same sampler identity surfaces as
  ordinary candidate sampling,
- fallback MUST record the reason, at least distinguishing candidate-count,
  branch-token, and memory-budget triggers,
- fallback MUST record `objective_fidelity=approximate_uniform_subsample` or an
  equivalent structured value and MUST NOT report the sample/run as exact,
- if approximation is disabled and a sample exceeds the authored budget, setup
  or training MUST fail with an actionable budget error rather than silently
  cap candidates.

#### Scenario: Exact plan within budget remains exact
- **GIVEN** budget policy is enabled
- **AND** a sample's exact candidate plan is within configured candidate and
  token budgets
- **WHEN** branch scoring runs
- **THEN** all remaining candidates are scored
- **AND** objective fidelity is recorded as exact.

#### Scenario: Authored subsampling remains approximate without budget fallback
- **GIVEN** `candidates.mode=uniform_subsample`
- **AND** budget policy is disabled or the sample is within budget
- **WHEN** branch scoring runs
- **THEN** the run records authored candidate scoring mode as
  `uniform_subsample`
- **AND** objective fidelity is not reported as exact.

#### Scenario: Long sample falls back to approximate MP
- **GIVEN** fallback mode `approximate_uniform_subsample`
- **AND** a sample's exact candidate plan exceeds the configured budget
- **WHEN** branch scoring runs
- **THEN** the trainer scores a deterministic uniform subset of remaining
  candidates
- **AND** logs remaining-candidate count, scored-candidate count, fallback
  reason, and approximate objective fidelity.

### Requirement: Prefix reuse distinguishes exact encoding cache from KV cache
The train-forward runtime SHALL distinguish exact prefix encoding reuse from
model-side KV prefix caching because these have different objective semantics.

Normative behavior:
- an in-sample CPU/tokenization prefix encoding cache MAY be used when it is
  proven parity-equivalent to uncached branch construction,
- CPU/tokenization prefix cache hits MUST NOT detach model gradients or enable
  `use_cache` inside model forwards,
- prefix encoding cache and GPU KV prefix caching MUST remain disabled in the
  immediate bridge,
- detached KV prefix caching, if later introduced, MUST be labeled approximate
  because it changes gradient flow through the prefix,
- exact shared-prefix branch-mask execution, if later introduced, MUST be a
  separate runtime mode with its own attention-mask and geometry/image-alignment
  tests and MUST NOT require edits to upstream HF model files.

#### Scenario: Encoding cache preserves branch inputs
- **GIVEN** prefix encoding cache is enabled
- **WHEN** candidate branches are built for one sampled prefix
- **THEN** cached construction produces the same `input_ids`, label masks,
  coord masks, image tensors, and span offsets as uncached construction.

#### Scenario: Detached KV cache is not reported as exact
- **GIVEN** a future config enables detached GPU KV prefix reuse
- **WHEN** branch scoring runs
- **THEN** the run records approximate detached-prefix objective fidelity
- **AND** does not claim exact MP comparability.

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
- `loss/weak_schema_close = final_schema_close_weight * (-logP_close_sequence(S))`
  applies only when `R = empty` and its weight is non-zero,
- `<|im_end|>`, `<|end_of_text|>`, EOS, chat-template stops, and object-entry
  close tokens are not part of `P_close_start` or `logP_close_sequence`,
- non-terminal candidate branches MUST use an append boundary rather than a
  structural close, so the one-step candidate objective does not teach “emit one
  object then close”,
- compatibility aliases such as `loss/eod`, `loss/anti_stop`, and
  `stop/p_stop_*` MAY be emitted for dashboards, but they MUST be documented as
  structural-close aliases and MUST NOT include chat-template EOD/EOS tokens.

#### Scenario: Remaining observed GT exists
- **GIVEN** `R != empty`
- **WHEN** close-start suppression is enabled
- **THEN** the trainer penalizes the first structural closure decision token
- **AND** does not penalize chat EOS or assistant stop tokens.

#### Scenario: No observed GT remains
- **GIVEN** `R = empty`
- **WHEN** weak structural-close supervision is enabled
- **THEN** the trainer teacher-forces the full structural closure sequence
- **AND** scales it by `final_schema_close_weight`, which may be `0.0`.

### Requirement: Positive-evidence margin supports threshold-loss objective in v1
The trainer SHALL support fixed-threshold PEM in v1, disabled by default, and
the source-idea objective SHALL use PEM as a threshold loss rather than an
additive penalty on top of MP.

Normative behavior:
- `positive_evidence_margin.objective` MUST support `disabled` and
  `threshold_loss`,
- when PEM is enabled for `threshold_space=full_entry_logZ`, calibrated
  `log_rho` MUST be configured and fixed `rho` MUST be rejected,
- `positive_evidence_margin.threshold_space` MUST be recorded and v1 SHALL use
  `full_entry_logZ`,
- PEM benchmark profiles MUST record threshold calibration provenance,
- PEM uses `logZ_remaining_exact` in exact mode and `logZ_remaining_est` in
  uniform-subsample mode,
- when `objective=threshold_loss`, `loss/pem` is optimized and
  `loss/mp_diagnostic` is logged without contributing to total loss,
- additive PEM, if later supported, MUST be a separately named ablation and
  MUST NOT be described as preserving probability space for latent positives.

#### Scenario: PEM disabled
- **GIVEN** `positive_evidence_margin.objective: disabled`
- **WHEN** loss is computed for `R != empty`
- **THEN** the trainer optimizes candidate-balanced token-normalized
  continuation CE, including the correct post-candidate boundary.

#### Scenario: PEM threshold-loss objective
- **GIVEN** `positive_evidence_margin.objective: threshold_loss`
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
- losses: `loss/candidate_balanced`, `loss/mp`, `loss/mp_diagnostic`, `loss/pem`,
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
  `mp/effective_candidate_count`, `mp/effective_candidate_fraction`,
  `mp/max_responsibility_scored`, `mp/min_responsibility_scored`;
- length diagnostics: `mp/candidate_entry_tokens_*`,
  `mp/candidate_logprob_sum_*`, `mp/candidate_logprob_per_token_*`,
  `mp/candidate_coord_token_fraction_*`,
  `mp/candidate_logprob_per_coord_token_*`,
  `mp/candidate_logprob_per_noncoord_token_*`,
  `mp/responsibility_vs_length_corr`, `mp/valid_length_corr_samples`;
- budget: `mp/branch_forwards_per_sample`, `mp/prefix_tokens_mean`,
  `mp/candidate_tokens_scored_mean`, `mp/total_candidate_tokens_scored`,
  `mp/repeated_forward_token_ratio_vs_baseline`,
  `mp/branch_runtime_mode`, `mp/checkpointed_branch_forwards`,
  `mp/retained_graph_branch_forwards`,
  `mp/ddp_candidate_padding_policy`,
  `mp/ddp_candidate_forward_local_count`,
  `mp/ddp_candidate_forward_max_count`,
  `mp/ddp_candidate_padding_forwards`;
- objective fidelity and fallback: `mp/objective_fidelity_exact_samples`,
  `mp/objective_fidelity_approx_samples`, `mp/fallback_applied_samples`,
  `mp/fallback_reason_candidate_budget`,
  `mp/fallback_reason_token_budget`,
  `mp/fallback_reason_memory_budget`;
- prefix reuse: `mp/prefix_encoding_cache_hits`,
  `mp/prefix_encoding_cache_misses`;
- structural close: `stop/p_close_start_when_remaining_exists`,
  `stop/p_continue_start_when_remaining_exists`,
  `stop/p_close_start_when_remaining_empty`,
  `stop/logp_close_sequence_when_remaining_empty`,
  `stop/p_final_schema_token_teacher_forced`;
- aux adapter counters: `aux/<name>/candidate_count`,
  `aux/<name>/position_count`, `aux/<name>/skipped_candidates`,
  `aux/<name>/contributing_candidates`.

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
- `effective_runtime.json` records repeated-forward semantics, branch runtime
  mode, checkpoint settings, branch isolation, prefix-gradient semantics,
  train-forward budget policy, prefix reuse mode, collator path, packing
  rejection, encoded-cache policy, candidate scoring mode, authored and
  fallback logZ estimator provenance, the runtime `mp/logZ_estimator` pointer,
  and exact versus approximate objective-fidelity counters when available,
- `experiment_manifest.json` records benchmark group identity, comparator,
  configured metric scope, eval view, benchmark-report notes, and bootstrap
  manifest-file pointers,
- `run_metadata.json` and data-provenance sidecars remain present,
- a set-continuation metric schema version is recorded.

#### Scenario: Tiny smoke run records provenance
- **GIVEN** a tiny set-continuation training smoke run
- **WHEN** bootstrap artifacts are written
- **THEN** the run artifacts include enough information to recover sampler,
  branch runtime, budget/fallback, prefix reuse, PEM, structural-close, aux,
  and benchmark settings after training.

### Requirement: Train-time detection eval shards generation under DDP
This requirement describes the already-verified Stage-1 set-continuation eval
baseline. It is outside the train-forward runtime stabilization slice and MUST
NOT be expanded by that slice.

When `custom.eval_detection.enabled=true`, train-time Stage-1 detection eval
SHALL support rank-sharded generation while preserving rank-0-owned scoring.

Normative behavior:
- `custom.eval_detection.distributed` defaults to `true`,
- when DDP world size is greater than one and distributed eval is enabled, every
  rank MUST run generation for a deterministic shard of the eval JSONL using its
  live training model replica,
- shard assignment MUST preserve the standalone inference contract based on
  source-index modulo world size,
- rank 0 MUST merge shard outputs into the canonical
  `eval_detection/step_<N>/gt_vs_pred.jsonl` before scoring,
- only rank 0 MUST run final detection metric scoring and inject
  `eval_det_*` metrics,
- non-zero ranks MUST not materialize final scored outputs or mutate the metrics
  dictionary after their shard generation completes,
- one-rank and `distributed: false` runs MUST retain the existing rank-0-only
  callback behavior.

#### Scenario: DDP eval uses all learner GPUs for generation
- **GIVEN** Stage-1 training runs with world size 8
- **AND** `custom.eval_detection.enabled=true`
- **AND** `custom.eval_detection.distributed=true`
- **WHEN** an eval step starts
- **THEN** all eight ranks decode disjoint eval shards
- **AND** rank 0 merges the shards and computes the final metric payload.

### Requirement: Production profile is config-addressable
The change SHALL provide one checked-in production config profile with a stable
benchmark identity for the all-feature set-continuation training run.

Normative behavior:
- documentation may explain earlier ablations but MUST NOT substitute for config
  identity,
- benchmark identity MUST live in a typed strict config surface, e.g.
  `benchmark.group_id`, `benchmark.control_group_id`,
  `benchmark.intended_variable`, and `benchmark.comparability_label`,
- the production profile MUST resolve `benchmark.group_id`,
  `benchmark.control_group_id`, dataset, prompt variant, object field order,
  seed, resolution/preset, sample/optimizer-step budget, checkpoint identity,
  inference decoding controls, and eval plan,
- the production profile MUST pin coord-token settings, effective coord-slot scoring
  surface, and aux objective settings (`coord_soft_ce_w1`, `bbox_geo`,
  `bbox_size_aux`) so the launch identity does not hide extra objective changes,
- the production profile MUST use `training.packing: false`; packed SFT may be
  a separately named throughput/control ablation but MUST NOT be silently mixed
  into the set-continuation production run,
- the production profile MUST enable distributed train-time detection eval so
  val200 generation uses all DDP learner ranks before rank-0 scoring,
- the production profile MUST enable exact supervised-suffix logits,
  no-padding DDP candidate synchronization, `ddp_find_unused_parameters=false`,
  `ddp_broadcast_buffers=false`, and authored cap-8 automatic approximate
  fallback for over-budget samples,
- the production profile MUST enable close-start suppression, weak/disabled
  final close supervision, fixed-threshold PEM threshold loss, and a mixed
  subset-prefix sampler with leave-one-out coverage,
- benchmark manifests MUST record the intended variable versus the comparator,
  comparability label, branch runtime mode, objective-fidelity counters,
  fallback reasons, realized branch/token budget, realized prefix-mode coverage,
  and realized aux/coord-scoring settings,
- benchmark reports MUST state eval scope, eval view, slice/sample count,
  prediction volume, AP/AP50/AP75, precision/recall where available, and
  sparse-label caveats.

#### Scenario: Production config resolves to the all-feature contract
- **GIVEN** the production set-continuation profile
- **WHEN** the config is resolved
- **THEN** it has a stable group id
- **AND** its initial model is the authored Stage-1 SFT checkpoint
- **AND** its enabled objective features are recorded as the intended benchmark
  variable.
