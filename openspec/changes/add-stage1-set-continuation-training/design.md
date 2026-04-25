# Design: Stage-1 Set-Continuation Training

## Context

The existing Stage-1 path is ordinary teacher-forced SFT over a serialized assistant response. The dataset/rendering layer already carries object metadata through encoded examples, and Stage-1 loss mixins currently assume one forward pass over one full target sequence. That makes the current path clean for fixed-order SFT but awkward for an object-level multi-positive objective.

The new paradigm should be introduced as a new trainer variant rather than a mutation of the existing Stage-1 CE path. This keeps the baseline stable and makes benchmark groups easy to interpret.

## Goals

- Train continuation from arbitrary observed object-set prefixes.
- Treat all remaining observed GT entries as valid next full entries.
- Avoid token-wise positive mixing across different objects.
- Separate object-entry end supervision from global list-stop supervision.
- Support weak or censored structural-close supervision.
- Add mechanism-level logs that explain whether MP changes continuation behavior.
- Preserve the current lm-head-only, Qwen3-VL chat-template-compatible training surface.
- Keep v1 simple enough to test and debug quickly.

## Non-Goals

- No RL, pseudo-labeling, detector head, external evaluator, or architecture change.
- No prefix-cache sharing or branch attention mask in v1.
- No runtime data augmentation that changes image geometry.
- No packed-sequence implementation in v1.
- No blind reuse of ordinary one-sequence SFT loss mixins in v1.

## Decisions

### 1. Trainer Variant Boundary

Add `custom.trainer_variant: stage1_set_continuation` and route it before the generic Stage-1 trainer factory.

The set-continuation path is a setup-path fork, not just a new trainer class name. It must be handled coherently in trainer resolution, trainer-class composition, collator selection, `remove_unused_columns`, packing validation, encoded-cache eligibility, metric defaults, and artifact manifests.

The set-continuation trainer should own `compute_loss` instead of inheriting the ordinary Stage-1 loss mixins. This avoids accidental double-counting of CE and prevents ordinary one-sequence assumptions from leaking into candidate-branch marginalization.

Allowed in v1:

- coord-token object coordinates only, serialized as `<|coord_*|>` tokens;
- ordinary full-vocab non-coord logprob inside each candidate entry;
- existing trainable modules such as coord-token adapters, if enabled by config and already compatible with LM CE;
- compatible coord and geometry auxiliary losses when implemented through branch-local set-continuation adapters.

Rejected in v1 unless explicitly redesigned:

- raw-text integer coordinate training;
- automatic composition of ordinary Stage-1 loss mixins;
- auxiliary-loss configs whose set-continuation branch adapter is not implemented;
- dataset packing for training or eval, including static pack-plan construction.

Required setup behavior:

- `resolve_trainer_cls` routes `stage1_set_continuation` to the dedicated trainer before the ordinary trainer factory.
- `compose_trainer_class` excludes ordinary one-sequence Stage-1 loss mixins for this variant.
- `remove_unused_columns` is set so raw sample metadata survives to the trainer.
- the collator path preserves the set-continuation metadata payload and strips non-model extras before model forwards.
- packing is rejected immediately after packing config resolution and before any static pack-plan dataset is built.
- benchmark and metric defaults are variant-aware rather than inherited blindly from ordinary Stage-1 token-accuracy surfaces.

### 1.1 Branch-Local Auxiliary Losses

The reason not to inherit existing Stage-1 mixins is mechanical, not philosophical. Existing mixins assume a single full target sequence and a single logits tensor. The MP trainer scores multiple branch continuations and then marginalizes object entries with log-sum-exp, so auxiliary losses need an explicit branch interface.

V1 should make compatible auxiliary losses toggleable through the set-continuation trainer:

- `coord_soft_ce_w1`: applies to coord-token positions inside each scored candidate entry;
- `bbox_geo`: applies to decoded candidate-entry bbox coordinates when all needed token positions are present;
- `bbox_size_aux`: applies to candidate-entry bbox geometry when `bbox_geo` state is available.

Recommended aggregation for v1:

```text
aux_atom(o) = mean-like atom over one candidate entry's contributing positions or boxes
loss/aux_<name> = mean(aux_atom(o) for o in scored_candidates_with_valid_atom)
```

This treats all scored remaining candidates as positives and avoids using MP responsibility collapse to decide which candidate receives geometry supervision. If a future experiment wants responsibility-weighted auxiliary losses, it should be a separately named config mode.

Each auxiliary loss must be logged separately and weighted separately. Each auxiliary adapter must also log candidate count, contributing position or box count, skipped-candidate count, and `aggregation = uniform_scored_candidate_mean`. If an auxiliary config is enabled but the required branch adapter is unavailable, setup should fail with an actionable error naming the unsupported adapter.

### 1.2 Candidate Score Mask

The MP candidate score is full-entry, but coord-token positions are scored through the coord-token vocabulary rather than ordinary full-vocab CE:

```text
score(o) = score_struct_desc(o) + score_coord(o)
score_struct_desc(o) = sum log P_full_vocab(y_t) over non-coord candidate-entry labels
score_coord(o) = sum log P_coord_vocab(coord_t) over coord-token candidate-entry labels
```

This preserves object-entry scoring while respecting the existing coord-token contract that distributional coord supervision should not treat coord-token positions as ordinary full-vocab CE targets. `coord_soft_ce_w1` remains an auxiliary branch-local objective. Its hard CE term, if enabled, is separate from `score_coord` and must be logged under the coord auxiliary namespace.

### 2. Repeated-Forward Candidate Scoring

Approach A is the v1 implementation:

```text
for candidate in candidates:
    forward(image, prompt, prefix + candidate_entry)
    score(candidate) = full-entry score with coord-vocab scoring at coord slots
loss/mp = -logZ_scored
```

Branch semantics:

- each candidate branch has the same prefix;
- candidates do not attend to each other;
- prefix gradients are not detached, but they are recomputed once per branch;
- no shared prefix cache is used;
- no branch attention mask is needed in v1.

Instrumentation must make this explicit:

- `mp/prefix_attach_mode = repeated_forward`;
- `mp/prefix_gradient = non_detached_recomputed_per_branch`;
- `mp/branch_isolation = independent_forward`;
- `mp/candidate_scoring_mode = exact | uniform_subsample`;
- repeated-forward budget stats.

### 3. Canonical Object Fragment Serialization

Candidate entries must be scored at the full object-entry level. The fragment should include the full object dictionary entry and the object-entry terminator, while excluding global list closure and assistant-message suffixes.

The implementation should derive fragments from the same serialization contract used for full assistant targets. It must use index-based span instrumentation rather than content-based substring search, because duplicate or identical object entries are valid detection cases.

A safe approach is:

1. Render the assistant payload for a selected object order using the canonical `dumps_coordjson` path.
2. Emit object-index span metadata while rendering the `objects` array.
3. Include the delimiter needed to append the entry to the current prefix.
4. Exclude the global `]`, enclosing object/list closure beyond the candidate entry, EOS, and chat-template assistant suffix from the candidate score.

This avoids reconstructing a subtly different JSON spelling for candidate branches.

The prefix should be a syntactically valid partial assistant response:

```text
{"objects": [<serialized entries from S>
```

If `S` is non-empty, candidate entries include the required comma separator before the object entry. If `S` is empty, the candidate entry starts as the first object entry.

Serialization tests must include duplicate identical object entries, same-description objects with different boxes, both supported object field orders, first/middle/last candidates, and exact delimiter/terminator boundaries.

### 4. Subset Sampling

For an image with object set `O`, sample one prefix subset `S` per example by configurable mixture:

```yaml
custom:
  stage1_set_continuation:
    subset_sampling:
      empty_prefix_ratio: 0.30
      random_subset_ratio: 0.45
      leave_one_out_ratio: 0.20
      full_prefix_ratio: 0.05
      prefix_order: random
```

V1 should include a small configurable full-prefix probability so stop behavior under `R = empty` is measured and, if configured, weakly supervised. The recommended initial value is `0.05`: large enough to log empty-remaining stop behavior, small enough not to turn sparse labels into a strong closed-world stop objective.

Recommended initial policies:

- empty-prefix: `S = {}`;
- random-subset: sample size uniformly from valid intermediate sizes, then sample objects uniformly;
- leave-one-out: sample one held-out object uniformly, `S = O - {o}`;
- full-prefix: `S = O`, `R = empty`, no MP term, optional weak closure supervision only;
- prefix order: randomize the order of `S` by default, with a `dataset`
  preserved-order option for ablations. A separate geometry-canonical ordering
  mode is not exposed in v1 because the sampler receives object indices, not
  decoded geometry.

The prefix-order recommendation is random by default because this experiment is explicitly about escaping fixed-order teacher-forcing basins. A preserved-order ablation is still useful when isolating "subset state" from "order randomization".

The sampler must log:

- `mp/num_prefix_objects`;
- `mp/num_remaining_objects`;
- the selected mode, at least as aggregate counts.

Small-object-count behavior must be deterministic:

- if `|O| = 0`, no MP term is computed; the sample may contribute structural-close metrics and optional weak structural-close loss;
- if `|O| = 1`, invalid random-intermediate modes are removed and the valid mode mixture is renormalized;
- if a selected mode has no valid subset, the sampler resamples from valid modes using deterministic seeded renormalization;
- if `R != empty`, `num_candidates_scored >= 1` is required;
- if `R = empty` and weak structural-close weight is `0`, the sample contributes metrics only and is excluded from the objective denominator.

The sampler RNG must be a pure function of resolved seed, epoch, sample identity (`sample_id` or `base_idx`), rank, and a documented microstep salt. It must not depend on global RNG state or dataloader worker timing.

### 5. Candidate Scoring and Subsampling

Candidate modes:

- exact all-remaining scoring;
- uniform subsampling with maximum `K`;
- same-budget benchmark mode, if feasible after exact mode is working.

For subsampling, v1 can optimize the sampled-candidate MP objective directly, but the scalar names must make the scope explicit:

```text
mp/logZ_scored_raw = logsumexp(score(o) for o in scored candidates C)
mp/logZ_remaining_exact = logsumexp(score(o) for o in all remaining R)  # exact mode only
mp/logZ_remaining_est = mp/logZ_scored_raw + log(|R| / |C|)             # uniform subsample estimator
```

Plain sampled MP may use `mp/logZ_scored_raw` because the missing `log(|R| / |C|)` term is a gradient-constant for fixed `S,C`. PEM must use `mp/logZ_remaining_exact` in exact mode or `mp/logZ_remaining_est` in uniform-subsample mode. Raw sampled PEM is not the default Group E objective and must be a separately named ablation if ever supported.

Logs must state when `mp/num_candidates_scored < mp/num_remaining_objects`, and benchmark artifacts must record `candidate_scoring_mode` and `logZ_estimator`.

### 6. Structural-Close Handling

Global stop is not the same as object-entry end.

For this paradigm, the global stop signal is the schema-level JSON closure continuation, not `<|im_end|>`, `<|end_of_text|>`, or any chat-template special token. The configured stop schema should identify the structural closure sequence that completes the nested detection JSON after the object list.

Define two separate probabilities:

```text
P_close_start(S) = P(first structural closure token | image, prompt, prefix)
logP_close_sequence(S) = sum_t log P(closure_t | image, prompt, prefix, closure_<t)
```

When `R != empty`, add optional anti-close:

```text
loss/anti_close_start = -log(1 - P_close_start(S))
```

When `R == empty`, optionally apply weak structural-close supervision:

```text
loss/weak_schema_close = final_close_weight * (-logP_close_sequence(S))
```

Initial recommendation:

- `anti_close_weight` configurable, default off or small for safe ablation;
- `final_close_weight` configurable, default `0.0` for sparse-label experiments;
- object-entry terminators remain part of candidate-entry CE;
- stop mass includes only the configured structural closure schema, not chat/EOS tokens and not object-close tokens.

Required implementation:

- derive the structural closure sequence from canonical serialized CoordJSON spans, not hand-authored strings;
- compute `P_close_start` at the prefix position, because that is where generation chooses to continue with another object or begin closing the list;
- compute `logP_close_sequence` by teacher forcing the full structural closure sequence only for empty-remaining/full-prefix structural-close supervision;
- also log the teacher-forced probability of the final schema closure token inside the structural closure sequence;
- do not include `<|im_end|>` or `<|end_of_text|>` in `P_close_start` or `logP_close_sequence`.

The metric namespace must use structural-close naming as the source of truth.
For dashboard continuity, v1 may also emit compatibility aliases such as
`loss/eod`, `loss/anti_stop`, and `stop/p_stop_*`; these aliases must be
documented as CoordJSON structural-close aliases and must never include
`<|im_end|>`, `<|end_of_text|>`, tokenizer EOS, or chat-template stop tokens.

### 7. Positive-Evidence Margin

Plain MP mode optimizes:

```text
loss/mp = -logZ_remaining_exact      # exact mode
loss/mp = -logZ_scored_raw           # sampled mode
```

PEM is included in v1 behind config, disabled by default. The source-idea PEM experiment is replacement-mode, not additive to MP:

```text
loss/pem = max(0, log(rho) - logZ_remaining)
total = loss/pem + structural_close + aux
```

Config shape:

```yaml
custom:
  stage1_set_continuation:
    positive_evidence_margin:
      mode: disabled          # disabled | replace_mp
      threshold_space: full_entry_logZ
      rho: null               # optional probability-space threshold
      log_rho: null           # optional log-space threshold
      threshold_calibration: null  # authored_fixed_ablation | calibration note/id
```

When `mode: replace_mp`, the trainer logs `loss/pem` as the optimized objective and logs `loss/mp_diagnostic = -logZ_*` without adding it to total loss. This preserves the intended margin behavior: observed GT mass must clear the configured threshold, but the model is not pushed to assign all probability mass to observed remaining GT after the margin is satisfied.

V1 must require exactly one threshold input when PEM is enabled: `rho` or `log_rho`. Numeric examples such as `rho: 0.90` may appear in benchmark profiles, but they should be treated as explicit experiment choices rather than silent defaults because full-entry sequence probabilities are length-sensitive.

For v1, `threshold_space` is fixed to `full_entry_logZ`. Group E profiles must
record whether the threshold is an authored fixed ablation value or came from a
calibration pass, and must preserve that provenance in the benchmark manifest.

### 8. Batch and Metadata Flow

The dataset path keeps `messages`, `assistant_payload`, `objects`, and `metadata` on encoded examples, but `assistant_payload.objects` is the reliable canonical object-entry list for this trainer. The existing `sample["objects"]` field can be dataset/runtime metadata and must not be assumed to contain serialized object entries.

Preferred v1 plumbing:

- add a raw-sample-preserving collator path for `stage1_set_continuation`;
- set `remove_unused_columns=false` for this variant;
- preserve `assistant_payload`, `messages` or enough image/prompt identity to re-encode branches, `metadata`, `sample_id`, `base_idx`, and dataset label;
- build candidate branch text through a trainer-local `Stage1SetContinuationBranchEncoder`;
- tokenize branches with the same processor/chat-template path as ordinary SFT while wrapping template state and disabling packing/padding-free assumptions;
- carry image inputs consistently with the original encoded example and run existing image-token/grid batch-contract checks in smoke tests.

The implementation should avoid re-reading JSONL from disk in the trainer. If tokenization inside `compute_loss` proves too invasive, introduce a dedicated collator for this trainer variant. The key design constraint is that branch construction must remain deterministic from the resolved config, dataset sample metadata, and seeded sampler identity.

### 8.1 Encoded Cache Eligibility

V1 should treat encoded-sample cache as ineligible for branch continuations. The cache may remain eligible only as a base encoded metadata cache if it preserves `assistant_payload`, `messages`, provenance, and sample identity without caching sampled branch continuations.

Required policy:

- if `training.encoded_sample_cache.enabled=true` and the variant cannot prove metadata-only cache eligibility, startup fails when `ineligible_policy=error`;
- if `ineligible_policy=bypass`, startup continues uncached and records an explicit bypass reason in logs and run artifacts;
- no branch continuation generated from runtime subset/candidate sampling may be read from a stale ordinary-SFT encoded cache.

### 8.2 Run Artifacts and Provenance

Set-continuation runs must preserve the existing training artifact family and add set-continuation-specific provenance:

- `resolved_config.json` includes the fully defaulted `custom.stage1_set_continuation` block, objective mode, subset ratios, prefix order, candidate scoring mode/K, logZ estimator, structural-close weights, PEM mode/threshold, aux adapter modes, and coord-token-only validation status.
- `effective_runtime.json` records repeated-forward mode, branch isolation,
  prefix-gradient semantics, raw-sample collator path, packing rejection,
  encoded-cache policy, and candidate scoring mode.
- `experiment_manifest.json` records the bootstrap-time benchmark summary:
  `benchmark_group_id`, `control_group_id`, comparator, configured eval scope,
  eval view, benchmark-report notes, and pointers to the pre-train manifest
  files. Post-train metric/eval artifacts remain produced by the existing
  training and evaluation callbacks/reports rather than being known at
  bootstrap time.
- `run_metadata.json` and data-provenance sidecars remain present and include git dirty status and upstream dependency provenance.
- a metric schema marker such as `stage1_set_continuation_metrics_version` is recorded so later metric-key changes cannot silently mix.

### 9. Metrics

Minimum logs:

- `loss/mp`;
- `loss/pem`;
- `loss/mp_diagnostic`;
- `loss/anti_close_start`;
- `loss/weak_schema_close`;
- `loss/aux_coord_soft_ce_w1`;
- `loss/aux_bbox_geo`;
- `loss/aux_bbox_size`;
- `mp/num_prefix_objects`;
- `mp/num_remaining_objects`;
- `mp/num_candidates_scored`;
- `mp/scored_candidate_fraction`;
- `mp/samples_with_candidates`;
- `mp/samples_full_prefix`;
- `mp/loss_mp_denominator_samples`;
- `mp/candidate_scoring_mode`;
- `mp/logZ_scored_raw`;
- `mp/logZ_remaining_exact`;
- `mp/logZ_remaining_est`;
- `mp/logZ_estimator`;
- `mp/responsibility_entropy_scored`;
- `mp/max_responsibility_scored`;
- `mp/min_responsibility_scored`;
- `mp/candidate_logprob_sum_mean`;
- `mp/candidate_logprob_sum_min`;
- `mp/candidate_logprob_sum_max`;
- `mp/candidate_logprob_sum_std`;
- `mp/candidate_logprob_per_token_mean`;
- `mp/candidate_logprob_per_token_min`;
- `mp/candidate_logprob_per_token_max`;
- `mp/candidate_logprob_per_token_std`;
- `mp/candidate_coord_token_fraction_mean`;
- `mp/candidate_coord_token_fraction_min`;
- `mp/candidate_coord_token_fraction_max`;
- `mp/candidate_coord_token_fraction_std`;
- `mp/candidate_logprob_per_coord_token_mean`;
- `mp/candidate_logprob_per_noncoord_token_mean`;
- `mp/candidate_entry_tokens_mean`;
- `mp/candidate_entry_tokens_min`;
- `mp/candidate_entry_tokens_max`;
- `mp/candidate_entry_tokens_std`;
- `mp/responsibility_vs_length_corr`;
- `mp/valid_length_corr_samples`;
- `mp/branch_forwards_per_sample`;
- `mp/prefix_tokens_mean`;
- `mp/candidate_tokens_scored_mean`;
- `mp/total_candidate_tokens_scored`;
- `mp/repeated_forward_token_ratio_vs_baseline`;
- `stop/p_close_start_when_remaining_exists`;
- `stop/p_continue_start_when_remaining_exists`;
- `stop/p_close_start_when_remaining_empty`;
- `stop/logp_close_sequence_when_remaining_empty`;
- `stop/p_final_schema_token_teacher_forced`;
- `aux/<name>/candidate_count`;
- `aux/<name>/position_count`;
- `aux/<name>/skipped_candidates`;
- `aux/<name>/contributing_candidates`;
- per-object coverage statistics if feasible without expensive bookkeeping.

Responsibility is:

```text
resp(o) = softmax(score(o) over candidates)
```

Entropy and max/min responsibility diagnose whether MP collapses onto one easy object. Length metrics are required because full-entry sequence log probability naturally depends on entry length; length-normalized scores are diagnostics only unless a future ablation explicitly changes the objective.

Small-n metric rules must be deterministic: one-candidate responsibility entropy
is `0.0`, one-candidate standard deviations are `0.0`, and
`mp/responsibility_vs_length_corr` is emitted only for samples with at least two
scored candidates and non-constant candidate lengths.

### 10. Benchmark Groups

The implementation should make these groups config-addressable:

- Group A: ordinary SFT baseline.
- Group B: ordinary SFT with structural schema-close masked or downweighted.
- Group C: one-prefix exact MP.
- Group D: subset MP plus anti-close-start.
- Group E: fixed-rho/log-rho PEM replacement-mode.
- Group F: leave-one-out emphasis.

These group letters are local to this OpenSpec implementation and supersede
intermediate letter labels used inside the exploratory source note. Benchmark
reports should cite `benchmark.group_id`, not the source note's letter names
alone.

Group B can be implemented either as a small ordinary-SFT loss-mask extension or as a trainer-compatible config path. It should not require the full MP trainer.

Each group must have a checked-in config profile or generated profile with a stable `benchmark.group_id`. Documentation may explain a group but must not substitute for config identity.

Add a typed top-level `benchmark` config section before authoring these
profiles. The strict parser should accept and validate at least `group_id`,
`control_group_id`, `intended_variable`, and `comparability_label`; otherwise
the profile YAMLs would rely on unknown top-level keys that current config
loading rejects.

Each profile must resolve:

- `benchmark.group_id`;
- `benchmark.control_group_id`;
- `custom.trainer_variant`;
- `custom.stage1_set_continuation.*` where applicable;
- `training.packing: false`;
- dataset JSONL, prompt variant, object field order, seed, resolution/preset, effective batch/sample budget, optimizer-step budget, checkpoint/base/adapter identity, inference decoding controls, and eval plan.
- coord-token settings, effective coord-slot scoring surface, aux objective
  settings (`coord_soft_ce_w1`, `bbox_geo`, `bbox_size_aux`), PEM threshold
  calibration provenance where applicable, realized prefix-mode coverage,
  realized branch/token budget, and eval plan.

The canonical A-F matrix should keep `training.packing: false` for ordinary SFT
groups as well as MP groups so accuracy comparisons do not mix static packing
with runtime branch sampling. A packed ordinary-SFT run can still be useful as a
throughput/control ablation, but it should use a separate group id and be marked
`not-comparable` or throughput-only unless the report explicitly justifies the
comparison.

Each run or report must include a benchmark manifest or an `experiment_manifest.json` section that records the intended variable versus the comparator group and labels comparability as `accuracy-comparable`, `throughput-comparable`, or `not-comparable`.

Benchmark reports must state eval scope and view explicitly: `val200`, `limit=200`, full-val, proxy view, sample count, prediction count totals/means, AP/AP50/AP75, recall/precision at relevant thresholds, and sparse-label caveats. COCO-real headline views and proxy-expanded analysis views must not be mixed silently.

## Architecture

```text
offline JSONL
  -> BaseCaptionDataset / JSONLinesBuilder
  -> encoded example with assistant_payload.objects + messages + metadata
  -> raw-sample-preserving set-continuation collator
  -> Stage1SetContinuationTrainer
       -> subset sampler
       -> indexed canonical prefix/candidate fragment builder
       -> candidate sampler
       -> Stage1SetContinuationBranchEncoder
       -> repeated independent branch forwards
       -> full-entry score aggregation
       -> MP / PEM / anti-close / weak-schema-close losses
       -> mechanism metrics
       -> run artifact and benchmark-manifest provenance
```

Suggested new modules:

- `src/trainers/stage1_set_continuation/__init__.py`
- `src/trainers/stage1_set_continuation/trainer.py`
- `src/trainers/stage1_set_continuation/sampling.py`
- `src/trainers/stage1_set_continuation/serialization.py`
- `src/trainers/stage1_set_continuation/branch_encoder.py`
- `src/trainers/stage1_set_continuation/losses.py`
- `src/trainers/stage1_set_continuation/metrics.py`

Keep names flexible during implementation, but keep the conceptual boundaries clear.

## Risks and Trade-Offs

- Repeated forward is slower than prefix sharing, but much easier to validate.
- Tokenization inside the trainer is viable only behind an explicit branch encoder that owns template state and image alignment.
- Canonical fragment slicing is safer than hand-rendering but requires careful tests around separators and closing tokens.
- Weak structural-close supervision may produce more predictions and apparent sparse-label false positives.
- Candidate subsampling changes reported logZ scope; exact mode should be the default for mechanism runs.
- PEM thresholds over full-entry sequence probabilities are length-sensitive and must be explicit experiment choices rather than hidden defaults.
- Existing Stage-1 metric parity tests may need explicit variant-specific expectations.

## Migration Plan

1. Add OpenSpec deltas for new and modified capabilities, then run strict validation when the CLI is available.
2. Add strict config schema and setup-path routing with fail-fast guards.
3. Add raw-sample collator and metadata-preservation tests before trainer loss code.
4. Add indexed canonical object-entry/closure span tests before trainer integration.
5. Add subset/candidate sampler tests, including deterministic RNG and small-object-count fallbacks.
6. Add branch scoring, MP, PEM, structural-close, and aux-adapter loss tests with synthetic logits.
7. Add trainer integration on a tiny fixture with no packing and one image.
8. Add artifact/provenance, metrics, and metric-key tests.
9. Add benchmark config profiles and manifest validation.
10. Update Stage-1 objective, metrics, packing, encoded-cache, and training README docs.

## Resolved Choices

1. PEM support is implemented in the first version, configurable and disabled by default.
2. Group B, weak structural-close supervision for ordinary SFT, belongs in this benchmark family.
3. The source direction note is tracked in `progress/directions/full_idea_v5.md` and synced into this worktree branch.
4. V1 supports coord-token coordinates only; raw-text integer coordinate training is rejected for this paradigm.
5. Compatible coordinate and geometry auxiliary losses remain toggleable through branch-local adapters.
6. Stop supervision targets the structural JSON closure schema, not chat/EOS tokens.
7. Prefix order defaults to random, with preserved-order ablations available.
8. Full-prefix sampling is small and configurable, recommended at `0.05`.
9. PEM Group E uses replacement-mode, while plain MP remains Group C/D.
10. Candidate scoring uses coord-vocab-normalized logprob at coord-token slots.
11. `assistant_payload.objects` is the canonical object-entry source for subset/candidate sampling.
