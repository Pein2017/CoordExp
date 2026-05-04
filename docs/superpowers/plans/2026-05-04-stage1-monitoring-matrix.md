# Plan: Stage-1 SFT and ET-RMP-CE Monitoring Matrix

Status: Ready for final Phase-1 approval review
Created: 2026-05-04
Target worktree: `/data/CoordExp/.worktrees/compact-detection-sequence`
Design doc: `/data/CoordExp/.worktrees/compact-detection-sequence/docs/superpowers/specs/2026-05-04-stage1-monitoring-matrix-design.md`

## Approval boundary

This plan is intentionally split by approval phase.

Phase 1 may be implemented only after explicit user approval. Phase 1 adds additive training-time monitoring only.

Phase 1 must not change:

- Training objective semantics.
- Dataset serialization.
- Candidate generation.
- Prefix selection.
- Rollout or decoding behavior.
- Model forward count.
- Geometry or image alignment.

Phase 2 and Phase 3 are documented for continuity only. They are not approved by approving Phase 1.

## Compatibility decision

Phase 1 uses additive namespacing.

- Add new compact metrics under `compact/*`.
- Add new set-continuation branch diagnostics under `setcont/rmp/*`.
- Preserve existing legacy set-continuation public keys, including existing bare `rmp/*`, `loss/rmp*`, `mp/*`, and `stop/*` keys.
- Bump the set-continuation schema version to `stage1_set_continuation_metrics_v3` because new default public keys are added.
- Treat v3 as additive. Do not remove or rename existing v2 keys in Phase 1.
- Update producer tests, allow-list tests, config/provenance tests, benchmark profile tests, and docs in the same change so `numeric_metric_payload` cannot silently drop new metrics and runtime metadata cannot claim v2 while emitting v3 keys.

## Source metadata mapping contract

Phase 1 uses existing source fields only. Do not infer grouping or categories from rendered token strings.

### Compact span categories

Source field:

- `LossAtom.semantic_role`, with `TokenTarget.semantic_role` as the aligned token-level source when available.

Mapping:

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

### Compact object grouping

Source field:

- `object_instance_id` from `LossAtom` or aligned `TokenTarget`.

Rules:

- Only supervised tokens with non-null `object_instance_id` contribute to `compact/object/*`.
- `compact/object/object_count` is the number of distinct non-null `object_instance_id` groups with at least one supervised token.
- If no non-null object ids are present, emit `compact/object/object_count = 0` and omit object exact rates.
- Do not group by `loss_atom_id`, text adjacency, punctuation, serialized span boundaries, or `object_index` alone.

### ET-RMP-CE branch type buckets

Source field:

- `FullSuffixTargetStep.token_type` after existing normalization.

Mapping:

| Source token type | Public bucket |
|---|---|
| `text` | `desc_text` |
| `coord` | `coord` |
| `structural` | `structural` |
| `other` | `other` |
| missing or unknown | `other` |

## Phase 1 metric key list

### Compact Stage-1 SFT keys

Required span categories:

- `schema`
- `desc_text`
- `coord`
- `object_control`
- `separator`
- `stop`
- `other`

Required compact span key pattern:

- `compact/span/<category>/token_count`
- `compact/span/<category>/teacher_ce_mean`
- `compact/span/<category>/top1_acc`
- `compact/span/<category>/top5_acc`
- `compact/span/<category>/teacher_p_mean`
- `compact/span/<category>/teacher_margin_mean`

Required compact object keys:

- `compact/object/object_count`
- `compact/object/all_token_top1_exact_rate`
- `compact/object/desc_all_top1_rate`
- `compact/object/coord_all_top1_rate`
- `compact/object/desc_token_count`
- `compact/object/coord_token_count`

Keep existing compact recursive detection keys:

- `loss/recursive_detection_ce`
- `recursive_detection_ce/batch_size`

### ET-RMP-CE set-continuation keys

Required aggregate keys:

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

Required type-conditioned keys:

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

Keep existing legacy set-continuation keys unless a separate migration is approved.

## Phase 1 metric formulas to pin in tests

### Token metrics

For logits `z`, teacher token `y`, and `p = softmax(z)`:

- `teacher_ce = -log p[y]`.
- `teacher_p = p[y]`.
- `top1_acc = 1[argmax(z) == y]`.
- `top5_acc = 1[y is in top 5 logits]`.
- `teacher_margin = z[y] - max_{j != y} z[j]`.

### ET-RMP-CE branch metrics

For valid child set `V`, model distribution `p`, teacher child token `t`, normalized target distribution `q`, and logits `z`:

- `valid_child_mass = sum_{v in V} p[v]`.
- `invalid_child_mass = 1 - valid_child_mass`.
- `valid_invalid_margin = log(valid_child_mass + eps) - log(invalid_child_mass + eps)`.
- `support_loss = -log(valid_child_mass + eps)`.
- `p_valid[v] = p[v] / valid_child_mass` for `v in V` when `valid_child_mass > eps`.
- `balance_kl = sum_{v in V} q[v] * (log q[v] - log p_valid[v])`.
- If implementation reuses an existing branch balance cross-entropy helper, compute `balance_kl = branch_balance_cross_entropy - target_entropy_for_step`; do not log cross-entropy as KL.
- `teacher_path_child_prob = p[t]` over the full vocabulary.
- `teacher_path_child_rank` is 1-based full-vocabulary rank of `t` by descending logits.
- `positive_child_rank` is 1-based full-vocabulary rank of the highest-logit valid child in `V`; lower is better. Ties follow the implementation's deterministic descending-sort order and are pinned in a synthetic test.
- `valid_child_effective_count = exp(-sum_{v in V} p_valid[v] * log(clamp_min(p_valid[v], eps)))`.
- If `valid_child_mass <= eps`, exclude the node from `valid_child_effective_count_mean` and count only nodes included in `effective_count_node_count`.
- `top1_invalid_rate = mean(1[argmax(z) not in V])` over branch nodes.
- `top1_valid_not_teacher_rate = mean(1[argmax(z) in V and argmax(z) != t])` over branch nodes.
- `valid_child_mass_p10` is the empirical 10th percentile over branch-node valid masses using the implementation's tensor quantile convention, pinned in test.

Denominators:

- `branch_node_count` covers support, mass, rank, and top1 branch metrics.
- `effective_count_node_count` covers `valid_child_effective_count_mean`.
- `balance_node_count` covers aggregate balance KL metrics.
- `<type>_branch_node_count` covers `support_loss_<type>_mean`.
- `<type>_balance_node_count` covers `balance_kl_<type>_mean`.

Zero-denominator behavior:

- Emit count keys with `0`.
- Omit mean/rate keys for empty groups.
- Never emit `0.0` as a missing-group mean.
- Preserve legacy `rmp/*` zero-denominator behavior separately from new `setcont/rmp/*` behavior.

## Task 1: Add compact metric tests first

Files likely touched:

- `/data/CoordExp/.worktrees/compact-detection-sequence/tests/test_recursive_detection_ce_loss_adapter.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/tests/test_recursive_detection_ce_trainer_mixin.py`

Add synthetic tests that build small logits/targets/loss atoms and assert:

- Each source semantic role maps to the exact public category in the mapping table.
- Missing or unknown semantic roles route to `other`.
- Span categories emit counts and teacher-forced metrics with hand-computed values.
- Schema, description, coordinate, object-control, separator, stop, and other categories are independently counted.
- Object exact metrics use non-null `object_instance_id`, not `loss_atom_id`, text adjacency, or `object_index` alone.
- Object exact metrics distinguish token-level correctness from all-token object correctness.
- No-object-id cases emit `compact/object/object_count = 0` and omit object exact rates.
- Zero-denominator groups emit counts and omit mean/rate metrics.
- Existing `loss/recursive_detection_ce` and `recursive_detection_ce/batch_size` behavior remains intact.
- Bare `batch_loss` and `batch_size` do not leak into public logging.

Suggested test names:

- `test_recursive_detection_metrics_map_semantic_roles_to_public_span_categories`
- `test_recursive_detection_metrics_split_schema_desc_coord_and_object_spans`
- `test_recursive_detection_object_exact_metrics_use_object_instance_id`
- `test_recursive_detection_object_exact_metrics_distinguish_token_from_entry_correctness`
- `test_recursive_detection_metrics_zero_denominator_omit_means_and_emit_counts`
- `test_recursive_detection_ce_trainer_does_not_log_internal_metric_keys`

Targeted command after implementation:

```bash
conda run -n ms python -m pytest tests/test_recursive_detection_ce_loss_adapter.py tests/test_recursive_detection_ce_trainer_mixin.py -q
```

Expected outcome after implementation:

- New tests pass.
- Existing recursive detection CE tests still pass.

## Task 2: Implement compact metric summarizer and logging

Files likely touched:

- `/data/CoordExp/.worktrees/compact-detection-sequence/src/detection/loss.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/trainers/metrics/mixins.py`

Implementation notes:

- Reuse logits already passed into `compute_recursive_detection_ce_batch_loss`.
- Reuse existing target/loss atom metadata and the exact mapping contract above.
- Add a small summarizer function with explicit public key emission.
- Do not dump arbitrary loss-result internals into trainer logs.
- Keep all current recursive detection CE metrics backward-compatible.

Done when:

- Compact tests from Task 1 pass.
- Public logged keys are exactly intentional `compact/*` plus existing compact recursive detection CE keys.

## Task 3: Add ET-RMP-CE branch metric tests first

Files likely touched:

- `/data/CoordExp/.worktrees/compact-detection-sequence/tests/test_stage1_set_continuation_full_suffix.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/tests/test_stage1_set_continuation_metric_keys.py`

Add synthetic tests that assert hand-computed branch metrics for a tiny vocabulary and explicit valid child sets.

Test these conditions:

- Each current source `token_type` maps to the exact public type bucket.
- High valid mass but poor balance KL is distinguishable from low valid mass.
- `balance_kl_*` is KL, not existing balance cross-entropy; include a case where target entropy makes those values differ.
- `top1_invalid_rate` catches invalid top-1 logits.
- `top1_valid_not_teacher_rate` treats a non-teacher but valid child as valid under multi-positive semantics.
- `positive_child_rank_mean` uses 1-based full-vocabulary rank of the highest-logit valid child.
- `teacher_path_child_rank_mean` uses 1-based full-vocabulary rank of the teacher child.
- `valid_child_effective_count_mean` uses entropy effective count over predicted conditional valid mass with `clamp_min`; include `[1.0, 0.0] -> 1` and `[0.5, 0.5] -> 2` cases.
- `effective_count_node_count` excludes nodes where `valid_child_mass <= eps`.
- `valid_child_mass_p10` follows the pinned quantile convention.
- Empty branch/type groups emit counts and omit means.
- Type buckets have both branch counts and balance counts.
- New `setcont/rmp/*` metrics are emitted alongside legacy keys.
- Legacy `rmp/*` zero-denominator behavior is preserved separately from new `setcont/rmp/*` zero-denominator behavior.

Suggested test names:

- `test_full_suffix_branch_metrics_map_token_types_to_public_buckets`
- `test_full_suffix_branch_metrics_distinguish_support_mass_from_balance_kl`
- `test_full_suffix_branch_metrics_balance_kl_is_not_cross_entropy`
- `test_full_suffix_branch_metrics_rank_valid_and_teacher_children`
- `test_full_suffix_branch_metrics_effective_count_formula_and_denominator`
- `test_full_suffix_branch_type_zero_denominator_omits_means_and_emits_counts`
- `test_stage1_set_continuation_metric_allowlist_preserves_legacy_and_adds_setcont_v3_keys`

Targeted command after implementation:

```bash
conda run -n ms python -m pytest tests/test_stage1_set_continuation_full_suffix.py tests/test_stage1_set_continuation_metric_keys.py -q
```

Expected outcome after implementation:

- New branch metric tests pass.
- Existing ET-RMP-CE tests still pass.
- Legacy metric keys remain public.
- New `setcont/rmp/*` keys are not silently dropped.

## Task 4: Implement ET-RMP-CE branch metric accumulation

Files likely touched:

- `/data/CoordExp/.worktrees/compact-detection-sequence/src/trainers/stage1_set_continuation/full_suffix.py`

Implementation notes:

- Extend the existing branch-node loop in `compute_full_suffix_loss`.
- Reuse existing `log_probs`, valid child ids, teacher token id, normalized targets, support loss, balance cross-entropy, target entropy, and token type.
- Map source token types using the exact branch type mapping table.
- Track branch-node counts, effective-count node counts, and balance-node counts separately.
- Track type-conditioned branch counts and type-conditioned balance counts for `desc_text`, `coord`, `structural`, and `other`.
- Emit `setcont/rmp/*` diagnostics additively.
- Preserve existing bare legacy keys and objective behavior.

Explicit public assertions:

- All aggregate keys listed in the Phase 1 metric key list are produced when denominators exist.
- All type-conditioned keys listed in the Phase 1 metric key list are produced when denominators exist.
- Count keys are produced even for zero denominators.
- Mean/rate keys are omitted for zero denominators.

Done when:

- Task 3 tests pass.
- No loss scalar changes except for expected floating-point identity from refactoring-free metric accumulation.

## Task 5: Update set-continuation public metric schema, config/provenance, and docs

Files likely touched:

- `/data/CoordExp/.worktrees/compact-detection-sequence/src/trainers/stage1_set_continuation/metrics.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/config/schema.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/sft.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/docs/training/METRICS.md`
- `/data/CoordExp/.worktrees/compact-detection-sequence/tests/test_stage1_set_continuation_config.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/tests/test_stage1_set_continuation_benchmark_profiles.py`

Implementation notes:

- Bump `STAGE1_SET_CONTINUATION_METRIC_SCHEMA_VERSION` to `stage1_set_continuation_metrics_v3`.
- Update `Stage1SetContinuationConfig.metric_schema_version` default to `stage1_set_continuation_metrics_v3`.
- Update the `src/sft.py` runtime metadata fallback to `stage1_set_continuation_metrics_v3`.
- Update config and benchmark profile tests that assert the schema version.
- Add new `setcont/rmp/*` keys to `EMITTED_STAGE1_SET_CONTINUATION_METRICS`.
- Retain all existing legacy emitted keys.
- Document that v3 is additive and preserves v2 keys.
- Document every new Phase 1 key with denominator and interpretation.
- Document that teacher-forced branch diagnostics are not rollout parse metrics or mAP.

Done when:

- Allow-list tests prove new keys survive `numeric_metric_payload`.
- Config/provenance tests prove runtime metadata and resolved config do not still claim v2.
- Docs mention additive v3 compatibility.

## Task 6: Add metric parity and no-leak tests

Files likely touched:

- `/data/CoordExp/.worktrees/compact-detection-sequence/tests/test_stage1_metric_key_parity.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/tests/test_stage1_set_continuation_metric_keys.py`

Add tests that assert:

- Every planned public `setcont/rmp/*` Phase 1 key is allow-listed.
- Legacy set-continuation keys remain allow-listed.
- Schema version is exactly `stage1_set_continuation_metrics_v3` everywhere it is surfaced.
- Internal keys like `batch_loss` do not leak through public metric filtering.
- Compact trainer logging uses explicit compact keys and does not dump raw loss-result internals.
- Docs mention the public key families added in Phase 1.

Targeted command after implementation:

```bash
conda run -n ms python -m pytest tests/test_recursive_detection_ce_loss_adapter.py tests/test_recursive_detection_ce_trainer_mixin.py tests/test_stage1_set_continuation_full_suffix.py tests/test_stage1_set_continuation_metric_keys.py tests/test_stage1_set_continuation_config.py tests/test_stage1_set_continuation_benchmark_profiles.py tests/test_stage1_metric_key_parity.py -q
```

Expected outcome after implementation:

- All new metric contract tests pass.
- Existing Stage-1 metric/config tests in the touched files still pass.

## Deferred Task 7: Coordinate-slot sidecars and numeric coord-token error

Status: Not approved in Phase 1.

This task requires separate approval if coordinate slot identity is not already safely available in current metadata.

Potential files:

- `/data/CoordExp/.worktrees/compact-detection-sequence/src/detection/objective.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/detection/loss.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/tests/test_recursive_detection_ce_loss_adapter.py`

Required before implementation:

- Prove slot labels `x1`, `y1`, `x2`, `y2` survive all relevant dataset/collator/loss paths.
- Prove numeric token error is only computed on a valid coord-token surface.
- Add tests for raw-text tokenization surfaces where numeric coord error must be omitted.

## Deferred Task 8: Object-entry likelihood and prefix-family sidecars

Status: Not approved in Phase 1.

Potential files:

- `/data/CoordExp/.worktrees/compact-detection-sequence/src/trainers/stage1_set_continuation/full_suffix.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/src/trainers/stage1_set_continuation/trainer.py`
- `/data/CoordExp/.worktrees/compact-detection-sequence/tests/test_stage1_set_continuation_full_suffix.py`

Deferred metrics:

- `setcont/rmp/teacher_entry_nll_mean`
- `setcont/rmp/teacher_entry_nll_desc_text`
- `setcont/rmp/teacher_entry_nll_coord`
- `setcont/rmp/teacher_entry_nll_structural`
- Object-entry path likelihood metrics.
- Prefix-family metrics beyond current branch token type.

Required before implementation:

- Identify exact object-entry path metadata.
- Preserve metadata through `_target_step_with_position`, shifting, retained scoring, smart-batched scoring, and padding-free packed scoring.
- Add tests proving sidecars are not dropped or misaligned.

## Deferred Task 9: Rollout/decode malformedness, duplication, EOS, and length probes

Status: Not approved in Phase 1.

Potential approach:

- Add tiny sampled training probes or reuse existing inference/eval artifact paths.
- Keep rollout parse metrics separate from teacher-forced training metrics.
- Report exact probe scope and artifact root.

Deferred metrics:

- Parse-valid sequence rate.
- Duplicate object-entry rate.
- EOS selected-too-early and EOS selected-too-late rates.
- Generated length distribution.
- Max-length hit rate.

## Final Phase 1 implementation checklist

Before coding:

- User explicitly approves Phase 1 implementation.
- No Phase 2 or Phase 3 work is included.

During coding:

- Write tests first for each metric family.
- Use existing logits only.
- Preserve existing public keys.
- Add only explicit public metrics.
- Keep count denominators visible.
- Omit zero-denominator means/rates.
- Preserve legacy zero-denominator behavior for legacy keys while applying the stricter v3 behavior to new `setcont/rmp/*` keys.

After coding, with user permission to run tests:

```bash
conda run -n ms python -m pytest tests/test_recursive_detection_ce_loss_adapter.py tests/test_recursive_detection_ce_trainer_mixin.py tests/test_stage1_set_continuation_full_suffix.py tests/test_stage1_set_continuation_metric_keys.py tests/test_stage1_set_continuation_config.py tests/test_stage1_set_continuation_benchmark_profiles.py tests/test_stage1_metric_key_parity.py -q
```

Approval request:

- Approve only Phase 1 additive monitoring metrics from this plan.
- Do not approve deferred Phase 2 coordinate-slot/object-entry sidecars.
- Do not approve deferred Phase 3 rollout/decode probes.
