---
doc_id: docs.training.metrics
layer: docs
doc_type: reference
status: historical-reference
domain: training
summary: Canonical training metric families for Stage-1 and the active Stage-2 single-pass contract.
updated: 2026-07-11
---

# Training Metrics and Losses

> Historical metric vocabulary and old-run reference. Verify current metric
> ownership in `src/artifacts/`, `src/losses/`, `src/metrics/`, and the stable
> Swift specs before making a current claim.

This reference describes the canonical metric families for Stage-1 and the
active single-pass Stage-2 contract.

## Observability Event Contract

Current training observability is typed at the producer boundary:

- `MetricEvent` is the canonical metric record for objective losses, counts,
  gauges, weighted means, ratios, and legacy aliases.
- `MetricEvent` identities include the metric key plus axes such as `stage`,
  `channel`, `objective_id`, `provenance`, and reducer/unit metadata. Flattening
  for ms-swift happens only at the logging boundary through
  `flatten_metric_events`.
- New writers should produce clean current keys. Removed training-mechanism
  writer keys are rejected by `src/training/observability/service.py`.
- Legacy flat metric keys are tolerated only through read/adaptation helpers
  such as `src/training/observability/legacy.py::adapt_legacy_metric`; tolerant
  reads do not authorize new writes.

Structured non-scalar diagnostics use `DiagnosticEvent`:

- `DiagnosticEvent` is for bounded payloads that should not become scalar
  training metrics.
- Diagnostic profiles are `off`, `standard`, and `debug`.
- `standard` keeps a small scalar summary and drops nested payloads.
- `debug` keeps richer nested payloads, but still enforces entry, item, depth,
  and string-length bounds.
- Event payloads may be truncated; consumers must check the event `truncated`
  flag before treating a diagnostic as complete evidence.

When interpreting any metric run, join the metric stream against the resolved
run artifacts documented in [`../ARTIFACTS.md`](../ARTIFACTS.md), especially
`resolved_config.json`, `effective_runtime.json`, and
`experiment_manifest.json`.

## Stage-1 Baseline Metric Families

Stage-1 remains aggregate-only. The documented keys below are the canonical
Stage-1 training families that parity tests expect to stay user-visible.

### Runtime And Accumulation

- `accum/grad_steps`
- `accum/current_grad_steps`
- `pack/num_samples`

### Base CE

- `base_ce/loss`
- `base_ce/loss_per_sample`
- `base_ce/noncoord_tokens`
- `base_ce/noncoord_tokens_per_sample`
- `stage1/total_loss_per_sample_est`

### Coord Objective

- coord objective atoms:
  - `coord_softce_w1/loss`
  - `coord_softce_w1/ce`
  - `coord_softce_w1/soft_ce`
  - `coord_softce_w1/w1`
  - `coord_softce_w1/gate`
  - `coord_softce_w1/text_gate`

Stage-1 non-canonical bbox note:

- The `cxcy_logw_logh` and `cxcywh` Stage-1 profiles use
  `coord_softce_w1/ce`, `coord_softce_w1/gate`, and
  `coord_softce_w1/text_gate` while forcing `soft_ce` and `w1` to zero.

### Token-Type Aggregates And Coord Monitors

- shared token aggregates:
  - `token_acc_top5`
  - `text_token_acc`
- token-type-conditioned aggregates:
  - `coord_token_acc`
  - `coord_token_acc_top5`
  - `coord_token_frac`
  - `desc_token_acc`
  - `desc_token_acc_top5`
  - `desc_token_frac`
  - `format_token_acc`
  - `format_token_acc_top5`
  - `format_token_frac`
- coord-monitor probes:
  - `coord_monitor/coord_vocab_mass_at_gt_text`
  - `coord_monitor/coord_vocab_mass_at_gt_coord`
  - `coord_monitor/coord_vocab_mass_at_gt_desc`
  - `coord_monitor/coord_vocab_mass_at_gt_format`
  - `coord_monitor/flip_text_to_coord`
  - `coord_monitor/flip_coord_to_noncoord`
  - `coord_monitor/flip_desc_to_coord`
  - `coord_monitor/flip_format_to_coord`

### Stage-1 Research Teacher-Forcing Type-Family Metrics

The compact Stage-1 research teacher-forcing objective may publish promoted
type-family mass metrics when `objective.terms.token_type_mass.enabled=true`.
These metrics are typed `MetricEvent` weighted means, not diagnostic-only side
channels, and no `_weighted` aliases are emitted.

- `teacher_forcing/loss/token_type_mass`: raw mean type-family loss over active
  teacher-forcing atoms.
- `teacher_forcing/loss/token_type_mass/contribution`: configured
  `token_type_mass.weight` multiplied by the raw type-family loss mean.
- `teacher_forcing/type/schema_mass_at_schema`: schema-family probability mass
  at schema-family targets.
- `teacher_forcing/type/coord_mass_at_coord`: coord-family probability mass at
  coord-family targets.
- `teacher_forcing/type/desc_mass_at_desc`: description-family probability
  mass at description-family targets.
- `teacher_forcing/type/stop_mass_at_stop`: stop-family probability mass at
  stop-family targets.

The raw and contribution loss metrics use the active teacher-forcing atom count
as denominator. Family mass metrics use the active atom count for the matching
target family and are omitted when a batch has no targets for that family.
The semantic stop family is `<|im_end|>`; tokenizer text terminators such as
`<|endoftext|>` and `<|end_of_text|>` are not stop-family targets.

Aggregate coordinate token accuracies are published through canonical
`MetricEvent` identities, with legacy aliases generated by the metric-event
alias bridge implemented in `src/metrics/events.py`:

- `coord_token_acc/full_vocab/top1` -> `coord_token_acc`
- `coord_token_acc/full_vocab/top5` -> `coord_token_acc_top5`

`src/metrics/events.py` owns the `MetricEvent` record shape and
`flatten_metric_events`. Flattening publishes canonical identities first and
adds aliases only for registered legacy identities. The reserved-alias collision
guard rejects events whose canonical identity would collide with a reserved
legacy alias, so new event identities must not reuse legacy flat keys directly.

### Compact Recursive Detection CE Metrics

Recursive-detection CE publishes canonical typed `MetricEvent` identities from
the recursive detection loss result and explicit legacy scalar keys from the
trainer logging boundary.

Canonical recursive CE objective `MetricEvent` keys:

- `detection_sequence/objective/recursive_detection_ce/loss_per_sample`
- `detection_sequence/objective/recursive_detection_ce/batch_size`

Diagnostic-only recursive CE objective keys expose internal multi-positive and
ordinary EOS CE behavior without changing the loss tensor:

- `recursive_detection_ce/trie_valid_mass`
- `recursive_detection_ce/support_loss`
- `recursive_detection_ce/balance_loss`
- `recursive_detection_ce/trie_valid_children`
- `recursive_detection_ce/type_gate_loss`
- `recursive_detection_ce/type_gate_allowed_tokens`
- `recursive_detection_ce/type_gate_weight`
- `recursive_detection_ce/eos_unweighted_ce`

`support_loss` and `balance_loss` are unweighted branch-local components.
`type_gate_loss` is the weighted allowed-type-mass contribution.
`eos_unweighted_ce` tracks ordinary teacher-forced `<|im_end|>` CE at stop
targets.

Canonical compact recursive-detection Phase-1 semantic `MetricEvent` keys:

- `detection_sequence/schema/token_acc/full_vocab/top1`
- `detection_sequence/schema/token_acc/full_vocab/top5`
- `detection_sequence/schema/token_ce/full_vocab`
- `detection_sequence/description/token_acc/full_vocab/top1`
- `detection_sequence/description/token_acc/full_vocab/top5`
- `detection_sequence/description/token_ce/full_vocab`
- `detection_sequence/coordinate/token_acc/full_vocab/top1`
- `detection_sequence/coordinate/token_acc/full_vocab/top5`
- `detection_sequence/coordinate/token_ce/full_vocab`
- `detection_sequence/object_control/token_acc/full_vocab/top1`
- `detection_sequence/object_control/token_acc/full_vocab/top5`
- `detection_sequence/object_control/token_ce/full_vocab`
- `detection_sequence/separator/token_acc/full_vocab/top1`
- `detection_sequence/separator/token_acc/full_vocab/top5`
- `detection_sequence/separator/token_ce/full_vocab`
- `detection_sequence/stop/token_acc/full_vocab/top1`
- `detection_sequence/stop/token_acc/full_vocab/top5`
- `detection_sequence/stop/token_ce/full_vocab`
- `detection_sequence/other/token_acc/full_vocab/top1`
- `detection_sequence/other/token_acc/full_vocab/top5`
- `detection_sequence/other/token_ce/full_vocab`
- `detection_sequence/object_entry/exact_sequence_match/object_entry`

Semantic token accuracy and CE denominators are the supervised token count in
the corresponding semantic category. The object-entry exact-sequence-match
denominator is the count of object groups with a non-null `object_instance_id`.
These are teacher-forced training diagnostics, not parse validity, duplicate
control, rollout quality, or mAP.

For ratio or weighted-mean `MetricEvent`s, `flatten_metric_events` omits the
flat mean/rate key when the event denominator is zero. Count events may still
publish explicit zero counts when the producer intentionally emits them, but
missing ratio/mean groups must not be represented as misleading `0.0` values.

No `compact/*` flat aliases are emitted for these semantic/object metrics
unless an identity-checked alias is registered through the `MetricEvent` alias
bridge. The canonical `detection_sequence/*` identities above are the public
contract.

Explicit legacy scalar keys that remain backward-compatible:

- `loss/recursive_detection_ce`
- `recursive_detection_ce/batch_size`
- `recursive_detection_ce/trie_support_weight`
- `recursive_detection_ce/trie_balance_weight`

Flattening still publishes canonical identities first and adds aliases only for
registered identities. New compact recursive-detection diagnostics should follow
the same typed-event-first contract instead of adding direct `compact/*` scalar
side channels.

### Trainer Metric Module Boundaries

`src/trainers/metrics/mixins.py` is a compatibility re-export facade. Current
trainer metric implementation entrypoints are:

- `src/trainers/metrics/batch_contract.py`
- `src/trainers/metrics/structural_close.py`
- `src/trainers/metrics/recursive_detection.py`
- `src/trainers/metrics/aggregate_tokens.py`
- `src/trainers/metrics/coord_losses.py`

Use these modules for source-level changes. Keep `mixins.py` import-compatible
for existing trainer imports and downstream tests.

## Interpreting Key Stage-2 Families

- `loss/<...>`:
  - post-weighting objective atoms
- `dup/raw/<...>` and `stage2_rollout_correction/correction/dup/<...>`:
  - pre-match duplicate-control diagnostics and policy counters
- `rollout/<...>`:
  - rollout parsing, matching, and coverage diagnostics
- `eval/...` or `eval_det_*`:
  - training-time evaluation outputs
- `snapshot/<metric_key>`:
  - carry-forward last-seen Stage-2 metrics surfaced for operator continuity
  - emitted when the current step did not freshly observe that metric family
  - live current-step namespaces such as `rollout/*` remain sparse and are not reused for stale values

## Removed Stage-2 Clean-Prefix Families

The old GT-anchored clean-prefix branch is not part of active Stage-2 training.
Do not interpret these metric families as current rollout-correction evidence:

- `loss/text/struct_ce`
- `loss/text/desc_ce`

## Stage-2 Rollout-Correction Objective Families

Rollout-correction keeps rollout-specific provenance:

- assignment policy:
  - `stage2_rollout_correction/correction/assignment/strategy_greedy_iou_count`
  - `stage2_rollout_correction/correction/assignment/iou_threshold`
- rollout-text atoms:
  - `stage2_rollout_correction/residual_set/*`
- duplicate-burst UL objective loss keys are retired; `train/optimization/loss_duplicate_burst_unlikelihood`
  is no longer a live training metric
- removed rollout-context coord/bbox atoms such as `loss/B_coord/*` are not
  part of the active Stage-2 objective pipeline

Interpretation note:

- duplicate control now runs on the assembled current-attempt plus peer-attempt
  object surface before GT matching
- non-exempt non-survivors disappear from the positive clean prefix and only
  contribute duplicate-control diagnostic metadata and counters; live
  duplicate-burst UL loss keys remain retired

## Rollout Evidence And Arbitrary-K Notes

The active residual-correction path emits a residual target IR from `1` current
rollout attempt plus `K-1` peer attempts.

Operational semantics:

- `stage2/raw_rollouts`
  - total number of rollout generations used for the batch
  - under the default pseudo-positive profile this is `4` per eligible sample
- `rollout/peer/*`
  - canonical peer-attempt aggregate metrics
  - interpreted as means over valid peer views under arbitrary `K`
  - with legacy `K=2`, these reduce to the single peer-attempt values
- `rollout/explorer/*`
  - preserved as compatibility aliases for older dashboards
  - mirrors `rollout/peer/*`; prefer `rollout/peer/*` in new analysis
- `rollout/temperature_config/ordinal_<i>/*`
  - records the configured decode request for rollout ordinal `i`
  - includes `temperature`, `top_p`, `top_k`, and `do_sample`
- `rollout/by_temperature/t<value>/*`
  - aggregates all valid current and peer rollout views for the same sampling
    temperature, for example `t0`, `t0p3`, `t0p5`, `t0p7`
  - includes `raw_rollouts`, `invalid_rollout_rate`,
    `parse_dropped_invalid`, `parse_truncated_rate`, `pred_objects`,
    `valid_pred_objects`, `gen_new_tokens_mean`, `gen_new_tokens_p90`,
    `unique_sequence_count`, and `unique_sequence_rate`
  - use these keys to check whether invalid outputs correlate with sampling
    temperature and whether non-greedy samples provide enough sequence
    diversity for a learning signal
- `train/triage/unlabeled_consistent_count`
  - total shielded-anchor count
  - includes support-positive-but-subthreshold anchors and cluster-demoted pseudo-positive candidates
  - these are retained unmatched anchor objects that stay in the clean prefix as context
- `train/triage/pseudo_positive_candidate_count`
  - unmatched anchors that meet the promotion floor before overlap clustering
- `train/triage/pseudo_positive_subthreshold_count`
  - current implementation logs the retained shielded-anchor total as a compatibility counter
  - that means it includes both support-positive anchors that stay below the promotion threshold and cluster-demoted pseudo-positive candidates
  - use `train/triage/pseudo_positive_cluster_demoted_count` to isolate the cluster-demoted slice
- `train/triage/pseudo_positive_selected_count`
  - final pseudo-positive winners after clustering
- `train/triage/pseudo_positive_cluster_demoted_count`
  - pseudo-positive candidates demoted back to shielded due to overlap clustering
- `train/triage/pseudo_positive_support_rate_num`
  - summed peer-support numerators over pseudo-positive candidates
- `train/triage/pseudo_positive_support_rate_den`
  - summed peer-support denominators over pseudo-positive candidates
- `train/triage/pseudo_positive_selected_support_rate_num`
  - summed peer-support numerators over selected pseudo-positive winners
- `train/triage/pseudo_positive_selected_support_rate_den`
  - summed peer-support denominators over selected pseudo-positive winners
- supervision note for interpretation:
  - selected pseudo-positive winners contribute fixed-weight prefix bbox/coord supervision
  - support-positive retained shielded anchors that are not cluster-demoted contribute support-rate-weighted prefix bbox/coord supervision
  - cluster-demoted pseudo-positive candidates remain structure-only prefix context
- `train/triage/recovered_ground_truth_rate_num`
  - summed peer-hit numerators for recovered GT objects missed by the current attempt
- `train/triage/recovered_ground_truth_rate_den`
  - summed valid-peer denominators for those recovered GT objects
- `train/triage/recovered_ground_truth_rate`
  - `rate_num / rate_den` when the denominator is non-zero
- `train/triage/current_preparation_dropped_count`
  - enabled pseudo-positive samples dropped because current-attempt accepted-clean preparation was malformed
- `train/triage/anchor_preparation_dropped_count`
  - compatibility alias for `train/triage/current_preparation_dropped_count`

Failure telemetry:

- malformed rollouts that remain invalid after salvage parsing abort the step
  by default instead of emitting an ordinary finalized `train/triage/*` counter
- with `stage2_rollout_correction.correction.invalid_rollout_policy=dump_and_continue`, the
  trainer logs `stage2_rollout_correction/correction/invalid_rollout_sample_dropped` and
  `stage2_rollout_correction/correction/invalid_rollout_sample_dropped_rate`
- compact-full uses a different invalid/empty policy:
  `fallback_gt_fn_append_only`. Malformed compact output, empty compact output,
  or compact rows whose bboxes are dropped before any valid survivor remain
  trainable as GT/FN-only correction targets. These fallback samples do not
  count as valid rollouts for readiness gates.
- compact-full fallback diagnostics:
  - `rollout/invalid_fallback_gt_fn_count`
  - `rollout/invalid_fallback_gt_fn_rate`
  - `rollout/empty_valid_object_rate`
  - `rollout/fallback_loss_share`
  - `rollout/fallback_dominance_warning`
  - `rollout/fallback_gt_fn_append_only_count`
  - `rollout/fallback_loss_weight`
- compact-full explorer rollouts that enter fallback are included in raw
  rollout/fallback metrics but excluded from posterior-support denominators,
  including `valid_explorer_count`, support rates, recovered-GT rates, and
  pseudo-positive selection.
- treat those aborts as failure telemetry / run outcome, not as a step-level
  rolling metric

## Duplicate And Rollout Diagnostics

Canonical duplicate/rollout families include:

- `dup/raw/*`
- `stage2_rollout_correction/correction/dup/N_*`
- `rollout/*`
- `time/rollout_*`

Duplicate-control gauges are emitted on the raw pre-match object surface and
finalize as weighted means:

- `dup/raw/max_desc_count`
- `dup/raw/saturation_rate`
- `dup/raw/duplicate_like_max_cluster_size`
- `dup/raw/desc_entropy`

Raw duplicate-pathology counters are also emitted on the raw pre-match object
surface, but remain additive counts:

- `dup/raw/near_iou90_pairs_same_desc_count`
- `dup/raw/near_iou90_pairs_any_desc_count`

Canonical rollout-correction duplicate-control counters remain additive
diagnostic metadata only. Duplicate-burst UL is not part of the current
canonical objective list:

- `stage2_rollout_correction/correction/dup/N_raw_bbox_valid`
- `stage2_rollout_correction/correction/dup/N_clean_accepted`
- `stage2_rollout_correction/correction/dup/N_clusters_total`
- `stage2_rollout_correction/correction/dup/N_clusters_exempt`
- `stage2_rollout_correction/correction/dup/N_clusters_suppressed`
- `stage2_rollout_correction/correction/dup/N_objects_suppressed`
- `stage2_rollout_correction/correction/dup/N_duplicate_control_first_divergence_boundaries`
- `stage2_rollout_correction/correction/dup/N_duplicate_control_first_divergence_skipped_no_divergence`

Use `docs/training/STAGE2_RUNBOOK.md` for the contract that produces these
families and `docs/ARTIFACTS.md` for where the corresponding monitor dumps and
run artifacts live.

## Retired Stage-1 Continuation Metrics

The former continuation metric family is no longer an active logging contract. Current metric claims should use the active baseline, compact recursive detection, or Stage-2 metric families documented above, with exact scope labels and artifact references.

## Training-Time Evaluation Families

Two distinct eval surfaces exist during training:

- offline evaluator callback:
  - `eval_det_*`
- trainer-native Stage-2 rollout eval:
  - owned by `stage2_rollout_correction` and implemented through the internal shared Stage-2 rollout runtime
  - `eval/detection/*`
  - `eval/parsing/*`
  - `eval/description/*`
  - `eval/config/*`
  - `eval/runtime/*`

## Removed Historical Families

Legacy iterative clean-prefix provenance groups are no longer part of the active
contract:

- `loss/A1_*`
- `loss/A2_*`
- `coord_diag/<...>`
- `coord_diag/A1/*`
- `coord_diag/A2/*`
- `eval_rollout/*`

If they appear in old logs, treat them as historical artifacts rather than
current contract surfaces.

## Historical Reference

The legacy deprecation rationale lives in:

- `progress/diagnostics/2026-03-20_stage2_channel_a_self_context_iter_ablation.md`
