# Stage-1 SFT and ET-RMP-CE Monitoring Matrix Design

Status: Ready for final Phase-1 approval review
Owner surface: repo-local Superpowers scaffold
Created: 2026-05-04
Target worktree: `/data/CoordExp/.worktrees/compact-detection-sequence`

## Purpose

This design defines an actionable training-time monitoring matrix for Stage-1 SFT and ET-RMP-CE training. The goal is not to add generic scalar noise. The goal is to make failures separable while preserving the current objective, data contract, artifact contract, and Qwen3-VL-compatible sequence surface.

The matrix is designed to answer these debugging questions during training:

1. Is the model learning the sequence schema, or merely reducing loss on easy text?
2. Is description/class-label learning improving separately from coordinate-token learning?
3. Does token-level accuracy correspond to object-entry-level correctness?
4. Are coordinate spans healthy at the slot/category level, without pretending teacher-forced token accuracy is mAP?
5. Does ET-RMP-CE place mass on valid multi-positive continuations, or collapse toward one arbitrary continuation, invalid schema, or early stop?
6. Are malformed sequence, duplication, EOS, and length risks measurable during training without online rollout?

## Current pipeline interpretation

### Compact recursive detection Stage-1 SFT

The compact detection path constructs recursive detection examples in the dataset/objective layer, enriches batches with target metadata in the collator, and computes a teacher-forced cross entropy objective over prepared token targets.

Relevant code surfaces:

- `/data/CoordExp/.worktrees/compact-detection-sequence/src/detection/ir.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/detection/objective.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/detection/dataset.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/data_collators/enrichers.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/detection/loss.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/trainers/metrics/recursive_detection.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/metrics/events.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/metrics/detection_sequence.py`

Training computes loss from existing model logits and structured target metadata. Phase 1 metrics must reuse those logits and sidecars; they must not add a second forward pass.

The current compact sequence work also introduces semantic provenance at several distinct layers:

- Current Phase-1 loss-time metrics consume `RecursiveDetectionTargets.token_targets` and `RecursiveDetectionTargets.loss_atoms`, specifically the already-aligned `LossAtom` / `TokenTarget` surface.
- The current render/tokenization path still renders `NormalizedDetectionSample` through the compact template, records `RenderSpanEvent` values for template-local span kind, primary role, mask groups, object identity, geometry kind, and slot name, then projects those render events into token roles and masks.
- Recursive target construction then builds aligned `TokenTarget` / `LossAtom` metadata from tokenized object-entry spans and assigns loss-time fields such as `semantic_role`, `object_instance_id`, and `loss_atom_id`.
- `DetectionDocument` / `DetectionObjectEntry` / `DetectionCoordinateSlot` are the canonical future upstream IR contract for object, geometry, and coordinate-slot semantics; they are not a Phase-1 loss-time dependency unless a separate implementation proves they are wired into the exact sidecar path consumed by recursive CE.

Render-event details such as `slot_name` remain upstream provenance and are not Phase-1 loss-time fields unless separately propagated and verified. Phase 1 compact loss-time metrics should consume the aligned `LossAtom` / `TokenTarget` surface. Future slot/object/render-role metrics should use the IR/render-event provenance instead of re-deriving semantic metadata from rendered strings or ad hoc loss-layer parsing.

### Set-continuation ET-RMP-CE

The set-continuation path preserves raw metadata in the collator, builds prefix-conditioned candidate/target structures, scores suffix rows, and computes the full-suffix ET-RMP-CE objective from teacher-forced logits over branch targets.

Relevant code surfaces:

- `/data/CoordExp/.worktrees/compact-detection-sequence/src/data_collators/stage1_set_continuation_collator.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/trainers/stage1_set_continuation/trainer.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/trainers/stage1_set_continuation/full_suffix.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/trainers/stage1_set_continuation/metrics.py`

Training-time ET-RMP-CE metrics are prefix-conditioned teacher-forced proxies. They can diagnose candidate mass, balance, and prefix sensitivity, but they are not rollout parse metrics and must not be reported as detection quality.

## Scope boundaries

### Phase 1 approval scope

Phase 1 may implement metrics that are already computable from current logits and existing metadata sidecars.

Phase 1 may touch:

- Compact detection loss summarization and trainer logging.
- ET-RMP-CE full-suffix branch metric accumulation.
- Compact `MetricEvent` helpers, flattening, alias tests, and detection-sequence metric docs.
- Public metric allow-lists, config/provenance schema-version defaults, metric docs, and focused tests.

Phase 1 must not:

- Change loss semantics.
- Change candidate generation, prefix selection, sampling, decoding, or rollout behavior.
- Add extra forward passes.
- Introduce new sidecar metadata that must be cloned through full-suffix target rewriting.
- Claim parse validity, duplication, EOS safety, or mAP from teacher-forced metrics.

### Phase 2 approval scope

Phase 2 requires separate approval because it may require object-entry path and coordinate-slot sidecar propagation through shifting, retained scoring, smart-batched scoring, and padding-free packed scoring.

### Phase 3 approval scope

Phase 3 requires separate approval because it covers rollout-like or decode-proxy malformedness, duplication, EOS, and length behavior. These are high-value metrics but should not be smuggled into Phase 1 as if teacher-forced branch metrics were rollout metrics.

## Namespacing and compatibility contract

### Compact metric namespace

Compact recursive-detection metrics use typed `MetricEvent` identities as the canonical metric surface. Flat trainer/eval logs are produced by `flatten_metric_events`; canonical event keys are emitted first, and compatibility aliases may only be produced through identity-checked alias registration.

Canonical compact `MetricEvent` key segments should follow the current helper identities. In particular, description and coordinate span diagnostics use `detection_sequence/description/*` and `detection_sequence/coordinate/*`. The `desc_text` and `coord` names are public grouping labels and optional `compact/span/*` alias labels only; Phase 1 must not introduce new canonical `detection_sequence/desc_text/*` or `detection_sequence/coord/*` identities without a separately approved migration and alias-parity tests.

Existing compact recursive-detection flat keys remain compatibility aliases or legacy scalar logs. New compact metric families may expose `compact/*` dashboard aliases only if those aliases are registered through the metric-event bridge and have identity tests proving that denominator, vocab scope, template id, parser/metric surface, and diagnostic status cannot drift.

Current compact recursive-detection keys that must remain backward-compatible include:

- `loss/recursive_detection_ce`
- `recursive_detection_ce/batch_size`
- `recursive_detection_ce/trie_support_weight`
- `recursive_detection_ce/trie_balance_weight`
- `detection_sequence/objective/recursive_detection_ce/loss_per_sample`
- `detection_sequence/objective/recursive_detection_ce/batch_size`

### Set-continuation metric namespace

New ET-RMP-CE branch diagnostics use the `setcont/rmp/` namespace.

Phase 1 must preserve all existing legacy public keys unless a separate breaking migration is explicitly approved. Existing bare keys such as `rmp/*`, `loss/rmp*`, `mp/*`, and `stop/*` remain emitted and allow-listed. New `setcont/rmp/*` keys are additive aliases or additive diagnostics, not replacements.

The set-continuation metric schema version is bumped to `stage1_set_continuation_metrics_v3` in the same implementation change that exposes new default public keys. The schema bump is additive, not a removal of existing v2 keys. The v3 string must flow through metrics allow-list code, config defaults, runtime metadata/provenance, benchmark profile tests, metric docs, and metric-key parity tests.

### Public logging rule

Only intentional public keys should be logged. Internal keys such as `batch_loss`, `batch_size`, raw loss internals, or per-position scratch values must not leak merely because a loss result dictionary exists.

For compact recursive detection, public logging should flow through `MetricEvent` flattening or explicit legacy key mapping. New compact diagnostics should not bypass `MetricEvent` flattening with ad hoc scalar dictionaries.

For set-continuation ET-RMP-CE, Phase 1 remains on the existing flat allow-list surface: `EMITTED_STAGE1_SET_CONTINUATION_METRICS` plus `numeric_metric_payload`. Converting set-continuation metrics to `MetricEvent` is explicitly out of scope for Phase 1 unless separately approved.

### Zero-denominator rule

Every grouped metric must emit its denominator count when useful. If a group has zero denominator, omit the corresponding mean/rate scalar and emit the count as `0`. Do not emit misleading `0.0` means for missing groups.

## Source metadata mapping contract

Phase 1 uses existing loss-time metadata only. If these fields are absent at runtime, Phase 1 metrics must fail closed by emitting the relevant count/event denominator as `0` when represented as an explicit count event and omitting dependent means/rates. They must not infer object boundaries, semantic roles, or branch types from rendered token strings.

The canonical upstream provenance for future compact semantic metrics is:

- `DetectionDocument` for object/geometry/coordinate-slot semantics before rendering.
- `RenderSpanEvent` for template-local render semantics before tokenization.
- Tokenized role/mask projection for aligned token spans.
- `LossAtom` / `TokenTarget` for Phase 1 loss-time aggregation.

Phase 1 should not add new sidecar propagation. It should consume what is already aligned at loss time and reserve IR/render-event propagation changes for separate approval.

### Compact span category mapping

Source field:

- `LossAtom.semantic_role`, with `TokenTarget.semantic_role` as the per-token aligned source when available.

Required mapping:

| Source semantic role | Public category |
|---|---|
| `SCHEMA_CONTROL` | `schema` |
| `DESC_IDENTITY` | `desc_text` |
| `BBOX_COORD` | `coord` |
| `ENTRY_TRIE_DECISION` | `object_control` |
| `OBJECT_CONTROL` | `object_control` |
| `SEPARATOR_CONTINUE` | `separator` |
| `TERMINAL_STOP` | `stop` |
| `CHAT_STOP` | `stop` |
| missing or unknown | `other` |

If a future role is introduced, it must map to `other` until a metric-contract update assigns it to a public category.

### Compact object grouping contract

Source field:

- Prefer `LossAtom.object_instance_id` for grouping object-scoped tokens.
- Use `TokenTarget.object_instance_id` as the aligned token-level source when available.
- `object_index` is diagnostic context only and must not replace `object_instance_id` when `object_instance_id` is present.

Object metric inclusion:

- Only supervised tokens with non-null `object_instance_id` contribute to canonical object-entry `MetricEvent`s and any optional `compact/object/*` aliases.
- Object count, if surfaced, must be a canonical count/sum/last `MetricEvent` under `detection_sequence/object_entry/*`; `compact/object/object_count` may exist only as a registered alias of that canonical identity.
- If no non-null object ids are present, emit the canonical object-count event with value `0` when the implementation intentionally publishes that count. Any `compact/object/object_count` flat key must come only from the alias bridge. Omit object exact-rate means.
- Do not infer grouping from serialized text adjacency, punctuation, token spans, or `loss_atom_id` alone.

### ET-RMP-CE branch type mapping

Source field:

- `FullSuffixTargetStep.token_type` after existing normalization.

Required mapping:

| Source token type | Public bucket |
|---|---|
| `text` | `desc_text` |
| `coord` | `coord` |
| `structural` | `structural` |
| `other` | `other` |
| missing or unknown | `other` |

This Phase 1 mapping intentionally stays coarse. Separator/control/stop distinctions inside `structural` are not Phase 1 set-continuation branch-type metrics.

## Metric math contract

This section fixes formulas so tests and dashboards mean the same thing across runs.

### Token-level teacher-forced metrics

For a position with logits `z`, teacher token `y`, and softmax `p = softmax(z)`:

- `teacher_ce = -log p[y]`.
- `teacher_p = p[y]`.
- `top1_acc = 1[argmax(z) == y]`.
- `top5_acc = 1[y is in top 5 logits]`.
- `margin = z[y] - max_{j != y} z[j]`.

These are teacher-forced token metrics, not decode metrics.

### Span-category compact metrics

For each span category, aggregate over token positions mapped to that category by the source metadata mapping contract.

Required Phase 1 span categories:

- `schema`
- `desc_text`
- `coord`
- `object_control`
- `separator`
- `stop`
- `other`

Required formulas:

- Canonical compact metrics are `MetricEvent` identities under `detection_sequence/*`.
- `<semantic>` is the canonical `MetricEvent` key segment, not necessarily the public span bucket label. Phase 1 uses these canonical segments: `schema -> schema`, `desc_text -> description`, `coord -> coordinate`, `object_control -> object_control`, `separator -> separator`, `stop -> stop`, and `other -> other`.
- `detection_sequence/<semantic>/token_acc/<vocab_scope>/top1`: mean token top-1 over category.
- `detection_sequence/<semantic>/token_acc/<vocab_scope>/top5`: mean token top-5 over category.
- `detection_sequence/<semantic>/token_ce/<vocab_scope>`: mean teacher CE over category.
- If teacher probability, margin, or explicit token-count metrics are added, they must also be `MetricEvent`s, not direct raw scalars.
- `compact/span/<category>/*` may exist only as compatibility/dashboard aliases registered through the metric-event bridge.

Actionability:

- `schema` good while `desc_text`/`coord` bad means the model learned formatting but not semantics/geometry.
- `desc_text` good while `coord` bad means labels/descriptions are learning faster than localization.
- `coord` loss improving without object metrics improving suggests token memorization or weak binding.

### Compact object-token grouping metrics

For each object entry grouped by non-null `object_instance_id`, aggregate token correctness across the supervised tokens assigned to that object.

Required Phase 1 formulas:

- Canonical compact object metrics are `MetricEvent` identities under `detection_sequence/object_entry/*`.
- `detection_sequence/object_entry/exact_sequence_match/<object_scope>`: fraction of object groups where the selected object-token predicate is exact.
- Description-only, coordinate-only, and all-token object exact metrics must use distinct `object_scope` identities or registered aliases with identity tests.
- If object counts or token denominators are surfaced in flat logs, emit them through canonical count/sum/last `MetricEvent`s first, not raw side-channel scalars.
- `compact/object/*` may exist only as compatibility/dashboard aliases registered through the metric-event bridge.

Actionability:

- Token accuracy can improve while object exact rate stays flat; that means entries are still brittle.
- Description exact improves but coordinate exact does not; that points to localization/tokenization rather than label learning.
- Coordinate exact improves but all-object exact does not; that points to binding, separators, or control tokens.

### Coordinate-slot metrics

Phase 1 does not add coordinate-slot sidecars. The compact IR/render-event layer now exposes upstream slot provenance through `DetectionCoordinateSlot.slot_name` and `RenderSpanEvent.slot_name`, but coordinate-slot training metrics remain Phase 2 unless `slot_name` is already present on the exact loss-time sidecar consumed by recursive CE and no propagation work is required.

When separately approved and available, formulas should use explicit slot labels, not token string heuristics:

- Canonical slot accuracy should use the existing typed identity family `detection_sequence/coordinate/slot_acc/<geometry_type>/<coordinate_surface>/<slot>`.
- Additional slot counts, CE, numeric error, or within-threshold rates must also be typed `MetricEvent` identities with `slot_name`, `geometry_type`, `coordinate_surface`, and `metric_surface` represented in the identity.
- `compact/coord_slot/x1/token_count`, `compact/coord_slot/y1/token_count`, `compact/coord_slot/x2/token_count`, `compact/coord_slot/y2/token_count`, `compact/coord_slot/<slot>/top1_acc`, `compact/coord_slot/<slot>/teacher_ce_mean`, `compact/coord_slot/<slot>/abs_error_mean`, `compact/coord_slot/<slot>/within_1_token_rate`, and `compact/coord_slot/<slot>/within_5_token_rate` may exist only as optional dashboard aliases registered through the metric-event bridge.
- Numeric slot error and within-threshold rates are valid only on coord-token surfaces where token ids map monotonically to numeric coordinate values.

Coordinate-slot numeric error is not valid on arbitrary raw-text tokenization unless the implementation proves a stable token-to-coordinate-value mapping.

### ET-RMP-CE branch metrics

For a branch node with valid child set `V`, model distribution `p`, teacher child token `t`, normalized balance target distribution `q` over valid children, and full vocabulary `A`:

- `valid_child_mass = sum_{v in V} p[v]`.
- `invalid_child_mass = 1 - valid_child_mass`.
- `valid_invalid_margin = log(valid_child_mass + eps) - log(invalid_child_mass + eps)`.
- `support_loss = -log(valid_child_mass + eps)`.
- `p_valid[v] = p[v] / valid_child_mass` for `v in V` when `valid_child_mass > eps`.
- `balance_kl = sum_{v in V} q[v] * (log q[v] - log p_valid[v])`.
- Existing code may expose branch balance cross-entropy; do not reuse it as KL. When the existing value is cross-entropy over `p_valid`, compute diagnostic KL as `balance_cross_entropy - target_entropy` and clamp only tiny negative floating-point noise.
- `teacher_path_child_prob = p[t]` over the full vocabulary, not conditional over valid children.
- `teacher_path_child_rank`: competition rank of teacher child token `t` over the full vocabulary: `rank = 1 + count(scores > z[t])`. Ties share the same rank, and no full-vocabulary sort is required.
- `positive_child_rank`: competition rank of the highest-logit valid child in `V` over the full vocabulary. Let `target_score = max_{v in V} z[v]`; then `rank = 1 + count(scores > target_score)`. Ties share the same rank, and no full-vocabulary sort is required.
- `valid_child_effective_count = exp(-sum_{v in V} p_valid[v] * log(clamp_min(p_valid[v], eps)))`. This is entropy effective count over predicted valid-child conditional mass, with range `[1, |V|]` when `valid_child_mass > eps`.
- If `valid_child_mass <= eps`, exclude that node from `valid_child_effective_count_mean` and emit/retain the corresponding count behavior rather than forcing a fake value.
- `top1_invalid_rate = 1[argmax_A z not in V]`.
- `top1_valid_not_teacher_rate = 1[argmax_A z in V and argmax_A z != t]`.

Required Phase 1 aggregate keys:

- `setcont/rmp/branch_node_count`
- `setcont/rmp/valid_child_mass_mean`
- `setcont/rmp/valid_child_mass_p10`
- `setcont/rmp/invalid_child_mass_mean`
- `setcont/rmp/valid_invalid_margin_mean`
- `setcont/rmp/top1_invalid_rate`
- `setcont/rmp/top1_valid_not_teacher_rate`
- `setcont/rmp/positive_child_rank_mean`
- `setcont/rmp/teacher_path_child_prob_mean`
- `setcont/rmp/teacher_path_child_rank_mean`
- `setcont/rmp/valid_child_effective_count_mean`
- `setcont/rmp/effective_count_node_count`
- `setcont/rmp/balance_node_count`
- `setcont/rmp/balance_kl_mean`

Percentile rule:

- `valid_child_mass_p10` is the empirical 10th percentile over branch-node `valid_child_mass` values using the implementation's tensor percentile/quantile convention pinned by test. Omit when `branch_node_count == 0`.

Denominator rule:

- `branch_node_count` is the denominator for support/mass/rank/top1 branch metrics.
- `effective_count_node_count` is the denominator for `valid_child_effective_count_mean`.
- `balance_node_count` is the denominator for `balance_kl_mean` and aggregate balance metrics.

Actionability:

- High `valid_child_mass` but poor `balance_kl` means schema/continuation set is learned but multi-positive distribution is collapsed or biased.
- Low `valid_child_mass` means the model is spending probability outside valid continuations.
- High `top1_invalid_rate` points to malformed continuation risk.
- High `top1_valid_not_teacher_rate` with good valid mass can be acceptable under multi-positive semantics; it means the model prefers another valid continuation.
- Low `valid_child_effective_count` with many positives means collapse toward one positive continuation.

### ET-RMP-CE branch type breakdown

Use `FullSuffixTargetStep.token_type` and the branch type mapping contract above.

Required Phase 1 public buckets:

- `desc_text`
- `coord`
- `structural`
- `other`

Required Phase 1 type-conditioned keys:

- `setcont/rmp/support_loss_desc_text_mean`
- `setcont/rmp/support_loss_coord_mean`
- `setcont/rmp/support_loss_structural_mean`
- `setcont/rmp/support_loss_other_mean`
- `setcont/rmp/balance_kl_desc_text_mean`
- `setcont/rmp/balance_kl_coord_mean`
- `setcont/rmp/balance_kl_structural_mean`
- `setcont/rmp/balance_kl_other_mean`
- `setcont/rmp/desc_text_branch_node_count`
- `setcont/rmp/coord_branch_node_count`
- `setcont/rmp/structural_branch_node_count`
- `setcont/rmp/other_branch_node_count`
- `setcont/rmp/desc_text_balance_node_count`
- `setcont/rmp/coord_balance_node_count`
- `setcont/rmp/structural_balance_node_count`
- `setcont/rmp/other_balance_node_count`

Type-conditioned denominator rule:

- Each `support_loss_<type>_mean` is the arithmetic mean of `support_loss` over branch nodes in that public type bucket.
- Each `<type>_branch_node_count` is the denominator for `support_loss_<type>_mean`.
- Each `balance_kl_<type>_mean` is the arithmetic mean of diagnostic KL over branch nodes in that public type bucket with a balance target.
- Each `<type>_balance_node_count` is the denominator for `balance_kl_<type>_mean`.

Actionability:

- Structural support loss dropping first means schema is learning.
- Description support improving while coordinate support remains poor means semantic text is easier than geometry.
- Coordinate balance KL high means the model may know coordinates are valid but distribute mass poorly over coordinate alternatives.

## Recommended monitoring matrix

| Diagnostic question | Metric family | Phase | Primary keys | Consumes | Failure mode exposed |
|---|---:|---:|---|---|---|
| Schema stable vs semantic quality | Compact span categories | 1 | `detection_sequence/<semantic>/token_acc/*`, `detection_sequence/<semantic>/token_ce/*`; optional registered `compact/span/*` aliases | logits, labels, semantic role metadata | Formatting learned but semantic/coord predictions weak |
| Text/class vs coordinate learning | Compact span and object groups | 1 | `detection_sequence/description/*`, `detection_sequence/coordinate/*`, `detection_sequence/object_entry/*`; optional registered `compact/*` aliases | logits, token targets, object ids | Class text improves while boxes lag, or vice versa |
| Token accuracy vs object-entry correctness | Compact object grouping | 1 | `detection_sequence/object_entry/exact_sequence_match/*`, object-denominated events | logits, object grouping metadata | Token averages hide brittle object entries |
| Coordinate-slot quality | Slot sidecar metrics | 2 unless already safe | `detection_sequence/coordinate/slot_acc/<geometry_type>/<coordinate_surface>/<slot>` and other typed slot events; optional registered `compact/coord_slot/*` aliases | explicit slot metadata, coord-token mapping | x/y or corner-specific failure |
| Loss by semantic span | Compact teacher CE by category | 1 | `detection_sequence/<semantic>/token_ce/<vocab_scope>` | logits, labels, semantic roles | Loss dominated by easy or irrelevant spans |
| Candidate validity under ET-RMP-CE | Branch valid mass diagnostics | 1 | `setcont/rmp/valid_child_mass_mean`, `setcont/rmp/top1_invalid_rate` | branch logits, valid child sets | Probability leaks into invalid continuations |
| Multi-positive continuation quality | Branch support vs balance | 1 | `setcont/rmp/support_loss_*_mean`, `setcont/rmp/balance_kl_*_mean`, `setcont/rmp/valid_child_effective_count_mean` | branch logits, normalized targets | Learns one positive but not balanced continuation set |
| Prefix sensitivity | Prefix-conditioned branch metrics | 1 proxy, 2 richer | Same `setcont/rmp/*` split by branch/type; richer prefix metadata deferred | branch rows, prefix-conditioned targets | Certain prefix states degrade valid continuation mass |
| Object-entry likelihood | Entry sidecar metrics | 2 | `setcont/rmp/teacher_entry_nll_*` | object-entry path sidecars across target transforms | Full object continuation likely/unlikely under prefix |
| Malformed sequence risk | Teacher-forced proxies first, rollout later | 1 proxy, 3 real | `setcont/rmp/top1_invalid_rate`; rollout parse counters deferred | branch logits now; decode artifacts later | Model likely to emit invalid next token or malformed output |
| Duplication tendency | Decode or entry-path metrics | 3 | deferred duplicate/entry-repeat counters | rollout/decode or explicit entry identity sidecars | Same object repeated or continuation collapse |
| EOS and length behavior | Stop category and rollout length metrics | 1 proxy, 3 real | `detection_sequence/stop/*`; optional registered `compact/span/stop/*` aliases; real EOS/length deferred | stop tokens now; generated sequences later | Early stop, late stop, overlong generations |

## Deferred metrics requiring explicit second approval

### Phase 2 deferred metrics

- Canonical coordinate-slot `MetricEvent`s such as `detection_sequence/coordinate/slot_acc/<geometry_type>/<coordinate_surface>/<slot>` if slot metadata is not already safely available.
- Optional `compact/coord_slot/<slot>/*` aliases for those canonical slot identities.
- `setcont/rmp/teacher_entry_nll_mean`.
- `setcont/rmp/teacher_entry_nll_desc_text`.
- `setcont/rmp/teacher_entry_nll_coord`.
- `setcont/rmp/teacher_entry_nll_structural`.
- Object-entry path likelihood and object binding metrics.
- Prefix-family stratification beyond existing branch token type.

### Phase 3 deferred metrics

- Parse-valid sequence rate during sampled training probes.
- Duplicate object-entry rate during sampled probes.
- EOS selected-too-early and EOS selected-too-late rates.
- Generated length distribution and max-length hit rate.
- Exact object-entry decoded correctness against parser/eval artifacts.

## Implementation placement proposal

### Compact Stage-1 SFT

Add a small summarizer near the recursive detection loss path. It should consume existing logits, token targets, and loss atoms, emit typed `MetricEvent`s, and rely on `flatten_metric_events` for public flat logs. The trainer mixin should explicitly log selected legacy keys and flattened metric events rather than dumping the full loss-result internals.

Do not add a parallel direct-scalar `compact/*` emission path.

Likely files:

- `/data/CoordExp/.worktrees/compact-detection-sequence/src/detection/loss.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/trainers/metrics/recursive_detection.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/metrics/events.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/metrics/detection_sequence.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/tests/test_recursive_detection_ce_loss_adapter.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/tests/test_recursive_detection_ce_trainer_mixin.py`

`/data/CoordExp/.worktrees/compact-detection-sequence/src/trainers/metrics/mixins.py` is a compatibility re-export facade after the metric-mixin split. Do not implement new compact recursive-CE logging there unless the facade itself must remain import-compatible.

### ET-RMP-CE

Extend full-suffix loss metric accumulation at branch nodes. Compute new metrics from the same `log_probs`, valid child ids, teacher child token, normalized targets, and token type already used by the objective.

Likely files:

- `/data/CoordExp/.worktrees/compact-detection-sequence/src/trainers/stage1_set_continuation/full_suffix.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/trainers/stage1_set_continuation/metrics.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/config/schema.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/sft.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/docs/training/METRICS.md`
- `/data/CoordExp/.worktrees/compact-detection-sequence/tests/test_stage1_set_continuation_full_suffix.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/tests/test_stage1_set_continuation_metric_keys.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/tests/test_stage1_set_continuation_config.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/tests/test_stage1_set_continuation_benchmark_profiles.py`

## Approval recommendation

Approve Phase 1 only if the intended implementation is additive and limited to the metrics listed as Phase 1 in this document.

Do not approve Phase 2 or Phase 3 metrics as part of the first implementation pass. Those phases should be handled after the Phase 1 matrix proves useful and after sidecar propagation requirements are reviewed separately.
