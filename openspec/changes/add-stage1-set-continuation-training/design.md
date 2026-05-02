# Design: Stage-1 Set-Continuation Training

Status update, 2026-05-02: this design is superseded as a production objective
by ET-RMP-CE. Candidate-balanced branch CE, candidate energy/logZ objectives,
chunk-level MP, and PEM/margin candidate-energy losses are legacy compatibility
surfaces only. Preserve prefix-conditioned sampling, but route production
training through full-suffix teacher-forced token CE, entry-trie
multi-positive token CE, support/balance reweighting, and hard CE for
schema/control/separator/stop tokens.

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
- Make the production train-step forward path OOM-preventive by trading more
  compute time for lower peak activation/logit memory and fewer wasted branch
  forwards.
- Make exact versus approximate objective fidelity explicit in config, metrics,
  and artifacts.

## Non-Goals

- No RL, pseudo-labeling, detector head, external evaluator, or architecture change.
- No train-time eval decoding redesign in this train-forward refinement.
- No prefix-cache sharing, GPU KV prefix-cache sharing, or branch attention mask
  in the immediate production bridge.
- No runtime data augmentation that changes image geometry.
- No dataset-level packed-sequence implementation in v1 for
  `stage1_set_continuation`.
- No true padding-free Qwen3-VL branch-packing implementation in the immediate
  throughput bridge. That larger framework-level work remains deferred until
  per-segment logits, packed multimodal metadata, and prefix-sharing semantics
  are designed together.
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

### 1.3 Lightweight Bidirectional Token-Type Gate

Set-continuation candidate scoring uses coord-vocab normalization at coord
slots. That is correct for ranking coordinate bins, but by itself it removes
the full-vocabulary competition that exists during free generation. The
production repair therefore adds a small native token-type gate to the
set-continuation objective. The gate is not a stronger stop/close signal; it is
a slot-type consistency signal.

For each candidate branch, define masks in label-token coordinates before the
standard next-token shift:

```text
coord_gate_label_mask =
  objective_label_mask
  AND coord_label_mask

text_gate_label_mask =
  objective_label_mask
  AND NOT coord_label_mask
  AND labels are supervised
  AND label token is not EOS, <|im_end|>, <|end_of_text|>, or padding
```

After shifting into logits coordinates, compute full-vocabulary coord mass:

```text
p_coord(t) = sum softmax(logits_full(t) / T)[coord_token_ids]

loss/coord_gate = mean(-log(p_coord(t)) over coord_gate positions)
loss/text_gate = mean(-log(1 - p_coord(t)) over text_gate positions)
```

Reduction contract:

- within one training sample, each gate loss is a token mean over all
  contributing positions across scored objective branches;
- the gate is then added once per objective-contributing sample, using the same
  sample denominator policy as `loss/candidate_balanced`;
- exact all-candidate scoring and cap-8 fallback therefore change only which
  branch tokens are observed, not the scalar weight per sample;
- the gate remains active when PEM threshold loss is enabled, even if the PEM
  margin is already satisfied, because token-type control is orthogonal to
  positive-evidence mass.

The optimized sample loss includes:

```text
loss =
  loss/candidate_balanced
  + w_coord_gate * loss/coord_gate
  + w_text_gate * loss/text_gate
  + existing structural-close/json terms
```

Required properties:

- the gate uses the exact same labels, masks, suffix crop, and logits tensor as
  the candidate branch score;
- at coord slots, `coord_vocab CE + w_coord * coord_gate` is a partial
  restoration of full-vocabulary competition: `w_coord=0` is pure coord-vocab
  scoring and `w_coord=1` equals ordinary full-vocab CE for that slot;
- prefix-only labels never contribute;
- the schema opener contributes to the text gate only for empty-prefix branches
  where it is in `objective_label_mask`;
- candidate append/close boundaries contribute to the text gate when they are
  in `objective_label_mask`;
- coord tokens contribute only to the coord gate;
- `candidate_object_label_mask` remains available for candidate-local aux
  losses but does not define the gate scope;
- branch-local `coord_soft_ce_w1` remains a separate aux adapter and MUST NOT be
  required just to enable bidirectional gating;
- retained-graph, checkpointed-exact, and smart-batched-exact runtimes must
  compute the same gate losses for the same branch tensors.
- in v1, `smart_batched_exact` bidirectional-gate runs must use
  `ddp_sync.candidate_padding=none`; `smart_batched_exact` plus
  `candidate_padding=max_count` is rejected rather than mixing real smart-batch
  candidate forwards with retained-graph padding branches.

Recommended first production weights are intentionally conservative:

```yaml
custom:
  stage1_set_continuation:
    bidirectional_token_gate:
      enabled: true
      coord_gate_weight: 0.5
      text_gate_weight: 0.1
      temperature: 1.0
      scope: objective_tokens
```

The gate must be validated by token-level and gradient-level tests before any
production run. A passing mAP smoke is not sufficient evidence that the gate is
mathematically aligned.

### 2. Repeated-Forward Candidate Scoring

Approach A is the v1 implementation:

```text
for candidate in candidates:
    forward(image, prompt, prefix + candidate_entry)
    score(candidate) = full-entry score with coord-vocab scoring at coord slots
loss/candidate_balanced = mean(-score(candidate) / tokens(candidate))
loss/mp_diagnostic = -logZ_scored
```

Branch semantics:

- each candidate branch has the same prefix;
- candidates do not attend to each other;
- prefix gradients are not detached;
- model-side prefix computation is repeated or recomputed per branch depending
  on the selected branch runtime;
- no GPU KV prefix cache is used in the immediate bridge;
- no branch attention mask is needed in v1.

Instrumentation must make this explicit:

- `mp/prefix_attach_mode = repeated_forward`;
- `mp/prefix_gradient = non_detached_recomputed_per_branch`;
- `mp/branch_isolation = independent_forward`;
- `mp/candidate_scoring_mode = exact | uniform_subsample`;
- `mp/branch_runtime_mode = retained_graph | checkpointed_exact | smart_batched_exact`;
- repeated-forward budget stats.

### 2.0.1 Smart Branch Batching Bridge

The immediate throughput bridge borrows the useful scheduler idea from
ms-swift packing without enabling dataset-level packing or true padding-free
branch packing. ms-swift's packing path groups tokenized rows with
`binpacking.to_constant_volume(...)`, then relies on padding-free Qwen/VL
metadata (`position_ids`, `text_position_ids`, `cu_seq_lens`) to preserve
segment boundaries. For set-continuation MP, the analogous schedulable unit is
not a raw dataset row but a runtime-built candidate branch.

`smart_batched_exact` therefore groups candidate branches into length-aware
padded branch batches:

```text
branch_work_item = encoded(prefix + candidate_entry)
smart batch = length-bucketed, constant-volume group of branch_work_items
```

This mode is exact for selected candidates:

- each row in a branch batch is still an independent `prefix + candidate`
  sequence;
- candidate rows do not attend to each other;
- prefix gradients remain non-detached and recomputed per branch row;
- the candidate score still uses the existing coord-aware full-entry scoring
  function;
- MP/PEM aggregation still happens per original sample after scores are
  scattered back to sample/candidate order.

The scheduler SHOULD follow ms-swift's dynamic packing spirit:

- branch length is first-class scheduling metadata;
- branches are bucketed by sequence length and supervised suffix length to
  reduce padding and avoid excessive `logits_to_keep`;
- groups are selected with a constant-volume target when `binpacking` is
  available;
- if `binpacking` is unavailable, the runtime MAY fall back to a deterministic
  length-bucketed first-fit policy while recording the scheduler used;
- underfilled batches, padding fraction, rows per branch batch, and token volume
  are logged.

This is intentionally not true padding-free branch packing. The future
padding-free runtime must solve Qwen3-VL multimodal packed metadata and
per-segment suffix-logit retention together. Until then,
`training.packing=false` remains the production contract for
`stage1_set_continuation`.

### 2.1 Train-Forward Runtime Policy

The production OOM showed that "one sample per device" is not the actual
memory unit. A single set-continuation sample can expand into many candidate
branch forwards before the MP log-sum-exp loss is returned to the outer Trainer.
The train-forward runtime therefore needs its own policy layer rather than
leaving branch expansion as inline trainer logic. This refinement targets the
retained-graph branch-memory component of the failure. It does not by itself
remove per-forward peak allocations such as the coord-offset logits-hook delta
tensor.

Introduce a config-first runtime policy under `custom.stage1_set_continuation`
for the training forward pass only. This refinement does not change eval
decoding.

For backward compatibility, omitting `train_forward` must preserve the current
verified runtime:

```yaml
train_forward:
  branch_runtime:
    mode: retained_graph
    checkpoint_use_reentrant: false
    preserve_rng_state: true
  budget_policy:
    enabled: false
    fallback:
      mode: disabled
  prefix_reuse:
    encoding_cache: false
    kv_cache:
      mode: disabled
```

Recommended shape:

```yaml
custom:
  stage1_set_continuation:
    train_forward:
      branch_runtime:
        mode: retained_graph       # retained_graph | checkpointed_exact | smart_batched_exact
        checkpoint_use_reentrant: false
        preserve_rng_state: true
      branch_batching:
        enabled: false
        strategy: ms_swift_constant_volume_buckets
        max_branch_rows: null
        max_branch_tokens: null
        min_fill_ratio: 0.70
        padding_waste_warn_fraction: 0.40
      logits:
        mode: supervised_suffix    # full | supervised_suffix
      ddp_sync:
        candidate_padding: none    # max_count | none
      budget_policy:
        enabled: true
        exact_until:
          max_candidates: 8
          max_branch_tokens_per_sample: null
        fallback:
          mode: approximate_uniform_subsample
          max_candidates: 8
          estimator: uniform_importance
          require_telemetry: true
      prefix_reuse:
        encoding_cache: false
        kv_cache:
          mode: disabled
      telemetry:
        per_rank_memory: true
        branch_budget: true
        objective_fidelity: true
```

Runtime modes:

- `retained_graph`: legacy/debug behavior. Candidate branch scores are computed
  with grad enabled, retained until the sample loss is assembled, and backpropagated
  by the outer Trainer. In the immediate production bridge, this remains the
  branch runtime but is paired with exact supervised-suffix logits and no DDP
  padding forwards.
- `checkpointed_exact`: optional exact memory fallback for MP/PEM-only runs.
  Branch differentiable atoms are wrapped in activation checkpointing so the
  trainer still returns one exact differentiable scalar loss, while branch
  internals are recomputed during backward instead of retained from forward.
  This spends more time to reduce retained activation/logit graph pressure, but
  it is not the immediate production default for the throughput bridge.
- `smart_batched_exact`: exact selected-candidate throughput mode. Candidate
  branches are materialized as work items, scheduled into ms-swift-inspired
  length-aware padded branch batches, scored in batched forwards, and then
  scattered back to per-sample MP/PEM aggregation. Close branches remain on the
  proven close-loss path in the initial bridge. This mode does not detach prefix
  gradients, does not use GPU KV cache, and does not enable dataset-level or
  padding-free branch packing.

Logit modes:

- `full`: backward-compatible mode. Model forwards may return logits for the
  whole branch sequence.
- `supervised_suffix`: exact production mode. For each candidate or close branch,
  compute the earliest supervised label position and pass `logits_to_keep` so
  Qwen3-VL returns only logits from `first_supervised_label_pos - 1` onward.
  Labels and masks are cropped to the same suffix before loss computation. This
  changes neither MP/PEM nor close-loss semantics because prefix logits outside
  the supervised masks do not contribute to the objective.

DDP candidate synchronization modes:

- `max_count`: backward-compatible mode. Ranks may execute zero-loss padding
  candidate forwards up to the per-step max candidate count.
- `none`: production throughput mode. Ranks execute local real candidates plus
  the close branch only. The cross-rank max count is still logged for skew
  diagnostics, but no padding forwards are executed. Under DDP this mode also
  requires `training.ddp_broadcast_buffers=false`; otherwise ranks with different
  candidate counts can desynchronize on forward-time buffer-broadcast
  collectives.

`checkpointed_exact` must preserve exact objective semantics:

- `loss/mp` and `loss/pem`, when applicable, are computed from the same
  branch scores and estimator semantics as retained-graph mode;
- the same branch labels, coord masks, image tensors, position semantics, and
  Qwen3-VL chat-template surface are used;
- `use_cache` remains disabled for model forwards;
- all gradient-bearing branch objective atoms must come from the same
  checkpointed computation, or setup must reject `checkpointed_exact` for the
  active objective configuration. The immediate bridge may reject
  `checkpointed_exact` when branch-local aux losses are enabled rather than
  silently detaching aux logits;
- checkpoint RNG handling must be deterministic (`preserve_rng_state: true` or
  equivalent) so dropout/stochastic layers do not silently turn exact MP into
  an approximation;
- if the implementation cannot preserve deterministic recompute for the active
  model/config, setup must either reject `checkpointed_exact` or explicitly
  downgrade through the configured approximate fallback and log that downgrade.
- DDP synchronization and padding forwards must honor the selected branch
  runtime. A rank that executes zero-loss padding branches in
  `checkpointed_exact` mode must not fall back to retained full branch graphs.

Manual branch-by-branch backward is a later option, not the first bridge. It
requires overriding more of the HF Trainer/Accelerate training step and is more
likely to interact with gradient accumulation, DDP synchronization, mixed
precision, and DeepSpeed. The checkpointed exact mode keeps the outer Trainer
contract intact: `compute_loss` still returns one scalar loss.

### 2.2 Budgeted Approximate Fallback

The production policy should not fail fast on long but scientifically valid
samples when an authored approximation is allowed. Instead, the branch planner
should decide before expensive branch scoring whether exact execution is within
configured budgets. If not, it falls back to uniform candidate subsampling and
records the fidelity change.

Budget inputs may include:

- `num_remaining_objects`;
- predicted `num_candidates_scored`;
- prefix token count;
- candidate entry token counts;
- total planned branch tokens for the sample;
- per-rank CUDA memory telemetry, if available before branch execution.

Fallback behavior:

```text
if exact plan is within budget:
    objective_fidelity = exact
    scored_candidates = all remaining candidates
else:
    objective_fidelity = approximate_uniform_subsample
    scored_candidates = uniform subset of remaining candidates
    logZ_estimator = uniform_importance
```

This policy changes estimator fidelity, so it must never be silent. Metrics and
artifacts must record:

- `mp/objective_fidelity_exact_samples`;
- `mp/objective_fidelity_approx_samples`;
- `mp/fallback_applied_samples`;
- `mp/fallback_reason_candidate_budget`;
- `mp/fallback_reason_token_budget`;
- `mp/fallback_reason_memory_budget`;
- `mp/num_remaining_objects`;
- `mp/num_candidates_scored`;
- `mp/scored_candidate_fraction`;
- `mp/logZ_estimator`.

For PEM, fallback mode must use the uniform-importance estimated remaining
logZ, not raw sampled mass. Plain sampled MP may still optimize raw sampled
mass when that is the authored estimator, but the estimator and fidelity labels
must make the scope clear.

### 2.3 Prefix Reuse Roadmap

There are three distinct prefix-reuse layers. The OpenSpec must keep them
separate because they have different correctness properties.

1. Exact CPU/tokenization prefix cache.
   - Cache the rendered/tokenized shared `image + prompt + sampled prefix`
     branch stem within one sample.
   - Append candidate-entry tokens/spans per branch.
   - This is exact if tests prove identical `input_ids`, labels, coord masks,
     image tensors, and span offsets compared with the uncached branch encoder.
   - It reduces CPU/tokenization overhead but does not remove model-side prefix
     recomputation.

2. Detached GPU KV prefix cache.
   - Run the prefix once with `use_cache=True` and reuse detached K/V for
     candidate suffixes.
   - This can improve throughput but changes gradient flow through the prefix.
   - It must be a future approximate mode, labeled for example
     `objective_fidelity = approximate_detached_prefix`.

3. Exact shared-prefix branch attention.
   - Compute the shared prefix once and evaluate candidate suffixes under a
     branch-isolating attention mask where candidates attend to the prefix and
     themselves, but not each other.
   - This is the long-term exact efficiency target, but it needs careful
     Qwen3-VL attention-mask, position-id, multimodal alignment, and
     FlashAttention validation. It must not require editing upstream HF model
     files.

The immediate bridge leaves all three layers disabled and observable in the
runtime payload. Layer 1 is still the next exact efficiency step; layers 2 and 3
remain explicit future runtime modes.

### 3. Canonical Object Fragment Serialization

Candidate continuations must be scored at the full object-entry level plus the
immediate boundary that follows the candidate. The fragment should include the
full object dictionary entry, the object-entry terminator, and either the
append boundary `, ` for a non-terminal candidate or the global list closure
`]}` for a terminal candidate, while excluding chat-template assistant suffixes.

The implementation should derive fragments from the same serialization contract used for full assistant targets. It must use index-based span instrumentation rather than content-based substring search, because duplicate or identical object entries are valid detection cases.

A safe approach is:

1. Render the assistant payload for a selected object order using the canonical `dumps_coordjson` path.
2. Emit object-index span metadata while rendering the `objects` array.
3. Include the delimiter needed to append the entry to the current prefix.
4. Include the post-candidate boundary: `, ` if additional observed objects
   remain after the candidate, or `]}` if the candidate exhausts the observed
   remaining set.
5. Exclude EOS and chat-template assistant suffixes from the candidate score.

This avoids reconstructing a subtly different JSON spelling for candidate branches.

The prefix should be a syntactically valid partial assistant response:

```text
{"objects": [<serialized entries from S>
```

If `S` is non-empty, candidate entries include the required comma separator
before the object entry. If `S` is empty, the candidate entry starts as the
first object entry. If appending the candidate still leaves observed objects,
the branch must remain append-ready rather than closing the CoordJSON object.

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
- budget-triggered automatic fallback from exact planning to uniform
  subsampling, when the configured train-forward policy allows approximation;
- same-budget benchmark mode, if feasible after exact mode is working.

For subsampling, v1 can optimize the sampled-candidate MP objective directly, but the scalar names must make the scope explicit:

```text
mp/logZ_scored_raw = logsumexp(score(o) for o in scored candidates C)
mp/logZ_remaining_exact = logsumexp(score(o) for o in all remaining R)  # exact mode only
mp/logZ_remaining_est = mp/logZ_scored_raw + log(|R| / |C|)             # uniform subsample estimator
```

Plain sampled MP uses `mp/logZ_scored_raw` because the missing
`log(|R| / |C|)` term is a gradient-constant for fixed `S,C` and because this
preserves the current verified sampled-MP behavior. PEM must use
`mp/logZ_remaining_exact` in exact mode or `mp/logZ_remaining_est` in
uniform-subsample mode. Raw sampled PEM is not the default Group E objective and
must be a separately named ablation if ever supported.

Logs must state when `mp/num_candidates_scored < mp/num_remaining_objects`, and
benchmark artifacts must record `candidate_scoring_mode`, the authored
`logZ_estimator`, the fallback estimator when configured, and the runtime
`mp/logZ_estimator` metric pointer.
When subsampling is selected by automatic fallback rather than by an authored
static candidate mode, logs and artifacts must additionally record
`fallback_applied=true`, the fallback reason, and
`objective_fidelity=approximate_uniform_subsample`. The budget policy sits on
top of the authored candidate mode: authored `candidates.mode=uniform_subsample`
is already approximate even if the train-forward budget does not fire.

### 6. Structural-Close Handling

Global stop is not the same as object-entry end.

For this paradigm, the global stop signal is the schema-level JSON closure continuation, not `<|im_end|>`, `<|end_of_text|>`, or any chat-template special token. The configured stop schema should identify the structural closure sequence that completes the nested detection JSON after the object list.

Define two separate probabilities:

```text
P_close_start(S) = P(first structural closure token | image, prompt, prefix)
logP_close_sequence(S) = sum_t log P(closure_t | image, prompt, prefix, closure_<t)
```

When `R != empty`, add optional close-start suppression:

```text
loss/anti_close_start = -log(1 - P_close_start(S))
```

When `R == empty`, optionally apply weak structural-close supervision:

```text
loss/weak_schema_close =
  final_schema_close_weight
  * annotation_completeness_weight(S)
  * (-logP_close_sequence(S))
```

Initial recommendation:

- `close_start_suppression_weight` configurable, default off or small for safe ablation;
- `final_schema_close_weight` configurable, default `0.0` for sparse-label experiments;
- `annotation_completeness_weight(S)` may be estimated from an original
  checkpoint rollout by treating localization FPs as likely unlabeled objects
  and using monotone `gt / (gt + fp_loc)` buckets by GT count;
- `json_structural_weight` adds structural CoordJSON CE over schema/key,
  punctuation, object/list boundary, and post-candidate boundary tokens while
  leaving desc payload text and coordinate values to the main candidate
  objective;
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

PEM-disabled production mode optimizes candidate-balanced continuation CE and
logs MP/logZ quantities as diagnostics:

```text
loss/candidate_balanced = mean(-score(o) / tokens(o) for scored candidates)
loss/mp_diagnostic = -logZ_remaining_exact      # exact diagnostic
loss/mp_diagnostic = -logZ_scored_raw           # sampled diagnostic
```

PEM is included in v1 behind config, disabled by default. The source-idea PEM experiment optimizes a threshold loss, not an additive penalty on top of MP:

```text
loss/pem = max(0, log(rho) - logZ_remaining)
total = loss/pem + bidirectional_token_gate + structural_close + aux
```

Config shape:

```yaml
custom:
  stage1_set_continuation:
    positive_evidence_margin:
      objective: disabled     # disabled | threshold_loss
      threshold_space: full_entry_logZ
      rho: null               # optional probability-space threshold
      log_rho: null           # optional log-space threshold
      threshold_calibration: null  # authored_fixed_ablation | calibration note/id
```

When `objective: threshold_loss`, the trainer logs `loss/pem` as the optimized
positive-evidence objective and logs `loss/mp_diagnostic = -logZ_*` without
adding it to total loss. The bidirectional token gate, if enabled, is still
optimized after the PEM margin is satisfied because it controls slot type, not
observed-GT probability mass. This preserves the intended margin behavior:
observed GT mass must clear the configured threshold, but the model is not
pushed to assign all probability mass to observed remaining GT after the margin
is satisfied.

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
- `effective_runtime.json` records repeated-forward semantics, branch runtime
  mode, checkpoint settings, branch isolation, prefix-gradient semantics,
  train-forward budget policy, prefix reuse mode, raw-sample collator path,
  packing rejection, encoded-cache policy, candidate scoring mode, and exact
  versus approximate objective-fidelity counters when available.
- `experiment_manifest.json` records the bootstrap-time benchmark summary:
  `benchmark_group_id`, `control_group_id`, comparator, configured eval scope,
  eval view, benchmark-report notes, and pointers to the pre-train manifest
  files. Post-train metric/eval artifacts remain produced by the existing
  training and evaluation callbacks/reports rather than being known at
  bootstrap time.
- `run_metadata.json` and data-provenance sidecars remain present and include git dirty status and upstream dependency provenance.
- a metric schema marker such as `stage1_set_continuation_metrics_version` is recorded so later metric-key changes cannot silently mix.

### 9. Metrics

Minimum emitted logs use the compact v2 schema:

- `loss/candidate_balanced`;
- `loss/schema_open`;
- `loss/json_structural`;
- `loss/anti_close_start`;
- `loss/weak_schema_close`;
- `mp/num_prefix_objects`;
- `mp/num_remaining_objects`;
- `mp/num_candidates_scored`;
- `mp/candidate_tokens_scored_mean`;
- `mp/schema_open_tokens_scored_mean`;
- `mp/json_structural_tokens_scored_mean`;
- `mp/annotation_completeness_weight_mean`;
- `mp/final_close_weight_mean`;
- `mp/tail_positive_samples`;
- `mp/final_gt_object_scored_samples`;
- `mp/objective_fidelity_exact_samples`;
- `mp/fallback_applied_samples`;
- `mp/selected_mode_empty_prefix`;
- `mp/selected_mode_full_prefix`;
- `mp/objective_contributing_samples`;
- `stop/p_close_start_when_remaining_exists`;
- `stop/p_continue_start_when_remaining_exists`;
- `stop/p_close_start_when_remaining_empty`.

LogZ, responsibility, branch-runtime, DDP, prefix-cache, and aux counters may be
computed internally or written to debug artifacts, but they are not production
emitted metrics in schema v2.

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
- Group D: subset MP plus close-start suppression.
- Group E: fixed-rho/log-rho PEM threshold-loss objective.
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
- dataset JSONL, prompt variant, object field order, seed, resolution/preset, effective batch/sample budget, optimizer-step budget, checkpoint/base/adapter identity, inference decoding controls, and eval plan, including whether train-time eval generation is rank-sharded under DDP.
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

Train-time detection eval using all learner ranks for generation is an existing
verified baseline for the broader Stage-1 set-continuation change, not part of
this train-forward runtime refinement. The existing inference engine already
owns source-index modulo sharding and rank-0 shard merge; the Stage-1 callback
reuses that path, then keeps final detection scoring and metric injection on
rank 0 only. Do not expand eval decoding work in the train-forward runtime
stabilization slice.

## Architecture

```text
offline JSONL
  -> BaseCaptionDataset / JSONLinesBuilder
  -> encoded example with assistant_payload.objects + messages + metadata
  -> raw-sample-preserving set-continuation collator
  -> Stage1SetContinuationTrainer
       -> subset sampler
       -> indexed canonical prefix/candidate fragment builder
       -> candidate planner
       -> train-forward budget policy
       -> Stage1SetContinuationBranchEncoder
       -> prefix reuse disabled in the immediate bridge
       -> branch scorer
            -> retained-graph exact suffix-logit production mode
            -> optional checkpoint/recompute exact mode
            -> DDP max-count or no-padding candidate synchronization
       -> full-entry score aggregation
            -> exact logsumexp
            -> explicit uniform-importance fallback estimator
       -> MP / PEM / close-start-suppression / weak-schema-close losses
       -> mechanism metrics and objective-fidelity telemetry
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
- `src/trainers/stage1_set_continuation/runtime.py`
- `src/trainers/stage1_set_continuation/budget.py`
- `src/trainers/stage1_set_continuation/branch_scorer.py`

Keep names flexible during implementation, but keep the conceptual boundaries clear.

## Risks and Trade-Offs

- Repeated forward is slower than prefix sharing, but much easier to validate.
- Supervised-suffix logits reduce full-prefix logit materialization without
  changing MP/PEM or close-loss semantics, but they rely on every supervised mask
  being cropped in the same coordinate frame.
- No-padding DDP improves throughput when ranks have uneven candidate counts,
  but rank skew must remain observable through local/max candidate telemetry.
- Checkpoint/recompute exact mode remains available for memory fallback, but it
  is slower than retained-graph suffix-logit execution and is not the immediate
  production default.
- Automatic fallback makes production more robust, but it changes estimator
  fidelity and therefore must be visible in every comparable run summary.
- CPU/tokenization prefix reuse could improve throughput without changing
  objective semantics, but it is deferred until branch-input parity tests are
  explicit. GPU KV prefix caching is more powerful but must remain a separate
  approximate/future mode unless exact gradient behavior is proven.
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
7. Extract train-forward planning, budget policy, branch scorer, and MP
   aggregator seams while keeping retained-graph behavior as a parity mode.
8. Add checkpoint/recompute exact branch scoring and prove tiny-fixture loss and
   gradient parity against retained-graph mode.
9. Add budget-triggered approximate fallback and tests proving fallback is
   explicit, deterministic, and never reported as exact.
10. Add exact suffix-logit scoring and no-padding DDP policy tests, then make the
    production profile use suffix logits, no padding, and cap-8 fallback.
11. Defer exact prefix encoding cache until branch-input parity tests cover
    labels, coord masks, image tensors, and span offsets.
12. Add trainer integration on a tiny fixture with no packing and one image.
13. Add artifact/provenance, metrics, and metric-key tests, including
    objective-fidelity and fallback counters.
14. Add benchmark config profiles and manifest validation.
15. Update Stage-1 objective, metrics, packing, encoded-cache, and training README docs.

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
