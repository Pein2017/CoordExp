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

- `/data/CoordExp/.worktrees/compact-detection-sequence/src/detection/objective.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/detection/dataset.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/data_collators/enrichers.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/detection/loss.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/trainers/metrics/mixins.py`

Training computes loss from existing model logits and structured target metadata. Phase 1 metrics must reuse those logits and sidecars; they must not add a second forward pass.

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

Compact full-sequence metrics use the `compact/` namespace. This keeps them distinct from legacy aggregate token-type metrics and ET-RMP-CE prefix-conditioned metrics.

### Set-continuation metric namespace

New ET-RMP-CE branch diagnostics use the `setcont/rmp/` namespace.

Phase 1 must preserve all existing legacy public keys unless a separate breaking migration is explicitly approved. Existing bare keys such as `rmp/*`, `loss/rmp*`, `mp/*`, and `stop/*` remain emitted and allow-listed. New `setcont/rmp/*` keys are additive aliases or additive diagnostics, not replacements.

The set-continuation metric schema version is bumped to `stage1_set_continuation_metrics_v3` in the same implementation change that exposes new default public keys. The schema bump is additive, not a removal of existing v2 keys. The v3 string must flow through metrics allow-list code, config defaults, runtime metadata/provenance, benchmark profile tests, metric docs, and metric-key parity tests.

### Public logging rule

Only intentional public keys should be logged. Internal keys such as `batch_loss`, `batch_size`, raw loss internals, or per-position scratch values must not leak merely because a loss result dictionary exists.

### Zero-denominator rule

Every grouped metric must emit its denominator count when useful. If a group has zero denominator, omit the corresponding mean/rate scalar and emit the count as `0`. Do not emit misleading `0.0` means for missing groups.

## Source metadata mapping contract

Phase 1 uses existing metadata only. If these fields are absent at runtime, Phase 1 metrics must fail closed by emitting the relevant count as `0` and omitting dependent means/rates. They must not infer object boundaries, semantic roles, or branch types from rendered token strings.

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

- Only supervised tokens with non-null `object_instance_id` contribute to `compact/object/*` metrics.
- `compact/object/object_count` counts distinct non-null `object_instance_id` groups with at least one supervised object token.
- If no non-null object ids are present, emit `compact/object/object_count = 0` and omit object exact-rate means.
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

- `compact/span/<category>/token_count`: number of supervised positions in category.
- `compact/span/<category>/teacher_ce_mean`: mean teacher CE over category.
- `compact/span/<category>/top1_acc`: mean token top-1 over category.
- `compact/span/<category>/top5_acc`: mean token top-5 over category.
- `compact/span/<category>/teacher_p_mean`: mean teacher probability over category.
- `compact/span/<category>/teacher_margin_mean`: mean teacher-vs-best-other logit margin over category.

Actionability:

- `schema` good while `desc_text`/`coord` bad means the model learned formatting but not semantics/geometry.
- `desc_text` good while `coord` bad means labels/descriptions are learning faster than localization.
- `coord` loss improving without object metrics improving suggests token memorization or weak binding.

### Compact object-token grouping metrics

For each object entry grouped by non-null `object_instance_id`, aggregate token correctness across the supervised tokens assigned to that object.

Required Phase 1 formulas:

- `compact/object/object_count`: number of object groups with at least one supervised object token.
- `compact/object/all_token_top1_exact_rate`: fraction of object groups where every supervised object token is top-1 correct.
- `compact/object/desc_all_top1_rate`: fraction of object groups where all description/class tokens are top-1 correct among objects with description tokens.
- `compact/object/coord_all_top1_rate`: fraction of object groups where all coordinate tokens are top-1 correct among objects with coordinate tokens.
- `compact/object/desc_token_count`: total description/class object-token denominator.
- `compact/object/coord_token_count`: total coordinate object-token denominator.

Actionability:

- Token accuracy can improve while object exact rate stays flat; that means entries are still brittle.
- Description exact improves but coordinate exact does not; that points to localization/tokenization rather than label learning.
- Coordinate exact improves but all-object exact does not; that points to binding, separators, or control tokens.

### Coordinate-slot metrics

Phase 1 does not add coordinate-slot sidecars. Coordinate-slot metrics are Phase 2 unless explicit slot labels already exist in the current metadata and no propagation work is required.

When separately approved and available, formulas should use explicit slot labels, not token string heuristics:

- `compact/coord_slot/x1/token_count`
- `compact/coord_slot/y1/token_count`
- `compact/coord_slot/x2/token_count`
- `compact/coord_slot/y2/token_count`
- `compact/coord_slot/<slot>/top1_acc`
- `compact/coord_slot/<slot>/teacher_ce_mean`
- `compact/coord_slot/<slot>/abs_error_mean`, only on coord-token surfaces where token ids map monotonically to numeric coordinate values.
- `compact/coord_slot/<slot>/within_1_token_rate`, only on coord-token surfaces.
- `compact/coord_slot/<slot>/within_5_token_rate`, only on coord-token surfaces.

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
- `teacher_path_child_rank`: 1-based full-vocabulary rank of teacher child token `t` by descending logits.
- `positive_child_rank`: 1-based full-vocabulary rank of the highest-logit valid child in `V`; lower is better. Ties follow the deterministic order used by the implementation's descending sort.
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
| Schema stable vs semantic quality | Compact span categories | 1 | `compact/span/schema/*`, `compact/span/desc_text/*`, `compact/span/coord/*` | logits, labels, semantic role metadata | Formatting learned but semantic/coord predictions weak |
| Text/class vs coordinate learning | Compact span and object groups | 1 | `compact/span/desc_text/*`, `compact/span/coord/*`, `compact/object/desc_all_top1_rate`, `compact/object/coord_all_top1_rate` | logits, token targets, object ids | Class text improves while boxes lag, or vice versa |
| Token accuracy vs object-entry correctness | Compact object grouping | 1 | `compact/object/all_token_top1_exact_rate`, object token counts | logits, object grouping metadata | Token averages hide brittle object entries |
| Coordinate-slot quality | Slot sidecar metrics | 2 unless already safe | `compact/coord_slot/<slot>/*` | explicit slot metadata, coord-token mapping | x/y or corner-specific failure |
| Loss by semantic span | Compact teacher CE by category | 1 | `compact/span/<category>/teacher_ce_mean` | logits, labels, semantic roles | Loss dominated by easy or irrelevant spans |
| Candidate validity under ET-RMP-CE | Branch valid mass diagnostics | 1 | `setcont/rmp/valid_child_mass_mean`, `setcont/rmp/top1_invalid_rate` | branch logits, valid child sets | Probability leaks into invalid continuations |
| Multi-positive continuation quality | Branch support vs balance | 1 | `setcont/rmp/support_loss_*_mean`, `setcont/rmp/balance_kl_*_mean`, `setcont/rmp/valid_child_effective_count_mean` | branch logits, normalized targets | Learns one positive but not balanced continuation set |
| Prefix sensitivity | Prefix-conditioned branch metrics | 1 proxy, 2 richer | Same `setcont/rmp/*` split by branch/type; richer prefix metadata deferred | branch rows, prefix-conditioned targets | Certain prefix states degrade valid continuation mass |
| Object-entry likelihood | Entry sidecar metrics | 2 | `setcont/rmp/teacher_entry_nll_*` | object-entry path sidecars across target transforms | Full object continuation likely/unlikely under prefix |
| Malformed sequence risk | Teacher-forced proxies first, rollout later | 1 proxy, 3 real | `setcont/rmp/top1_invalid_rate`; rollout parse counters deferred | branch logits now; decode artifacts later | Model likely to emit invalid next token or malformed output |
| Duplication tendency | Decode or entry-path metrics | 3 | deferred duplicate/entry-repeat counters | rollout/decode or explicit entry identity sidecars | Same object repeated or continuation collapse |
| EOS and length behavior | Stop category and rollout length metrics | 1 proxy, 3 real | `compact/span/stop/*`; real EOS/length deferred | stop tokens now; generated sequences later | Early stop, late stop, overlong generations |

## Deferred metrics requiring explicit second approval

### Phase 2 deferred metrics

- `compact/coord_slot/<slot>/*` if slot metadata is not already safely available.
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

Add a small summarizer near the recursive detection loss path. It should consume existing logits, token targets, and loss atoms, and return only public `compact/*` metrics plus counts. The trainer mixin should explicitly log selected keys rather than dumping the full loss-result internals.

Likely files:

- `/data/CoordExp/.worktrees/compact-detection-sequence/src/detection/loss.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/trainers/metrics/mixins.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/tests/test_recursive_detection_ce_loss_adapter.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/tests/test_recursive_detection_ce_trainer_mixin.py`

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
