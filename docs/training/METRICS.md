---
doc_id: docs.training.metrics
layer: docs
doc_type: reference
status: canonical
domain: training
summary: Canonical training metric families for Stage-1 and the active Stage-2 single-pass contract.
updated: 2026-04-25
---

# Training Metrics and Losses

This reference describes the canonical metric families for Stage-1 and the
active single-pass Stage-2 contract.

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

### Coord Objective And Diagnostics

- coord objective atoms:
  - `coord_softce_w1/loss`
  - `coord_softce_w1/ce`
  - `coord_softce_w1/soft_ce`
  - `coord_softce_w1/w1`
  - `coord_softce_w1/gate`
  - `coord_softce_w1/text_gate`
  - `coord_softce_w1/adjacent_repulsion`
- coord diagnostics:
  - `coord_diag/enabled`
  - `coord_diag/loss`
  - `coord_diag/loss_per_sample`
  - `coord_diag/ce`
  - `coord_diag/soft_ce`
  - `coord_diag/w1`
  - `coord_diag/gate`
  - `coord_diag/text_gate`
  - `coord_diag/adjacent_repulsion`
  - `coord_diag/coord_tokens`
  - `coord_diag/coord_tokens_per_sample`
  - `coord_diag/coord_vocab_mass`
  - `coord_diag/text_coord_vocab_mass`
  - `coord_diag/adjacent_repulsion_pair_count`
  - `coord_diag/adjacent_repulsion_applied_count`
  - `coord_diag/adjacent_repulsion_copy_score_mean`
  - `coord_diag/acc_top5`
  - `coord_diag/p_gt_mean`
  - `coord_diag/margin_mean`
  - `coord_diag/expected_bin_mae`
  - `coord_diag/expected_bin_abs_err_p90`
  - `coord_diag/w1_to_delta`

Stage-1 non-canonical bbox note:

- The `cxcy_logw_logh` and `cxcywh` Stage-1 profiles use
  `coord_softce_w1/ce`, `coord_softce_w1/gate`, and
  `coord_softce_w1/text_gate` while forcing `soft_ce` and `w1` to zero.

### BBox Geo

- `loss/geo/bbox_geo`
- `loss/geo/bbox_smoothl1`
- `loss/geo/bbox_ciou`
- `bbox_geo/loss_per_sample`
- `bbox_geo/groups_total`
- `bbox_geo/groups_per_sample`
- `bbox_geo/coord_slots_total`
- `bbox_geo/skipped_incomplete_rows`
- `bbox_geo/skipped_incomplete_coord_slots`

Interpretation note:

- `loss/geo/bbox_smoothl1` is the stable key for the configured bbox regression
  term
- with `parameterization: xyxy`, it is the canonical decoded-box regression term
- with `parameterization: center_size`, it is the internal center-strong plus
  soft `log_w` / `log_h` regression term derived from canonical `xyxy`
- `loss/geo/bbox_ciou` remains CIoU on canonical `xyxy` across both modes
- compare `bbox_smoothl1` across runs only after joining against
  `resolved_config.json`

### BBox Size Aux

- `loss/geo/bbox_size_aux`
- `loss/geo/bbox_log_wh`
- `loss/geo/bbox_oversize`
- `bbox_size_aux/loss_per_sample`
- `bbox_size_aux/groups_total`
- `bbox_size_aux/groups_per_sample`
- `bbox_size_aux/coord_slots_total`
- `bbox_size_aux/skipped_incomplete_rows`
- `bbox_size_aux/skipped_incomplete_coord_slots`
- `bbox_size_aux/mean_width`
- `bbox_size_aux/mean_height`
- `bbox_size_aux/mean_log_area`

### BBox Geometry Aux

- `loss/geo/bbox_geo`
- `loss/geo/bbox_smoothl1`
- `loss/geo/bbox_ciou`
- `bbox_geo/loss_per_sample`
- `bbox_geo/groups_total`
- `bbox_geo/groups_per_sample`
- `bbox_geo/coord_slots_total`
- `bbox_geo/skipped_incomplete_rows`
- `bbox_geo/skipped_incomplete_coord_slots`

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

## Interpreting Key Stage-2 Families

- `loss/<...>`:
  - post-weighting objective atoms
- `coord_diag/<...>`:
  - coord-distribution diagnostics
- `dup/raw/<...>` and `stage2_ab/channel_b/dup/<...>`:
  - pre-match duplicate-control diagnostics and policy counters
- `rollout/<...>`:
  - rollout parsing, matching, and coverage diagnostics
- `eval/...` or `eval_det_*`:
  - training-time evaluation outputs
- `snapshot/<metric_key>`:
  - carry-forward last-seen Stage-2 metrics surfaced for operator continuity
  - emitted when the current step did not freshly observe that metric family
  - live current-step namespaces such as `rollout/*` remain sparse and are not reused for stale values

## Stage-2 Channel-A Objective Families

Channel-A uses the normal single-pass GT-anchor groups only:

- `loss/text/struct_ce`
- `loss/text/desc_ce`
- `loss/coord/bbox_smoothl1`
- `loss/coord/bbox_ciou`
- `loss/coord/bbox_log_wh`
- `loss/coord/bbox_oversize`
- `loss/coord/coord_token_ce`
- `loss/coord/coord_soft_ce`
- `loss/coord/coord_w1`
- `loss/coord/coord_gate`
- `loss/coord/text_gate`
- `coord_diag/*`
- `gradmon/*/coord/*` when gradient monitoring is enabled

Interpretation note:

- `loss/coord/bbox_smoothl1` keeps the same public key even when
  `bbox_geo.config.parameterization: center_size` is enabled
- the key means “the configured bbox regression term” and therefore must be
  interpreted together with `resolved_config.json`
- `loss/coord/bbox_ciou` remains canonical `xyxy` CIoU

## Stage-2 Channel-B Objective Families

Channel-B keeps rollout-specific provenance:

- rollout-text atoms:
  - `loss/B_rollout_text/struct_ce`
  - `loss/B_rollout_text/desc_ce`
- duplicate suppression:
  - `train/optimization/loss_duplicate_burst_unlikelihood`
- rollout-context coord atoms:
- `loss/B_coord/bbox_smoothl1`
- `loss/B_coord/bbox_ciou`
  - `loss/B_coord/bbox_log_wh`
  - `loss/B_coord/bbox_oversize`
  - `loss/B_coord/coord_token_ce`
  - `loss/B_coord/coord_soft_ce`
  - `loss/B_coord/coord_w1`
  - `loss/B_coord/adjacent_repulsion`
  - `loss/B_coord/coord_gate`
  - `loss/B_coord/text_gate`
- coord diagnostics:
  - `coord_diag/B/*`
- gradient monitors:
- `gradmon/*/B_coord/*` when enabled

Interpretation note:

- `loss/B_coord/bbox_smoothl1` follows the same configured regression semantics
  as Channel-A
- `loss/B_coord/bbox_ciou` remains canonical `xyxy` CIoU
- duplicate control now runs on the assembled anchor plus explorer object
  surface before GT matching
- non-exempt non-survivors disappear from the positive clean prefix and only
  contribute the unchanged collapsed UL payload shape

## Channel-B Pseudo-Positive And Arbitrary-K Notes

When `stage2_ab.channel_b.pseudo_positive.enabled=true`, Channel-B still emits a
single clean teacher-forced forward, but the rollout evidence path widens from
`1` anchor + `1` explorer to `1 + (K-1)` views.

Operational semantics:

- `stage2/raw_rollouts`
  - total number of rollout generations used for the batch
  - under the default pseudo-positive profile this is `4` per eligible sample
- `rollout/explorer/*`
  - preserved as compatibility metrics
  - now interpreted as means over valid explorer views under arbitrary `K`
  - with legacy `K=2`, these still reduce to the single explorer values
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
  - summed explorer-support numerators over pseudo-positive candidates
- `train/triage/pseudo_positive_support_rate_den`
  - summed explorer-support denominators over pseudo-positive candidates
- `train/triage/pseudo_positive_selected_support_rate_num`
  - summed explorer-support numerators over selected pseudo-positive winners
- `train/triage/pseudo_positive_selected_support_rate_den`
  - summed explorer-support denominators over selected pseudo-positive winners
- supervision note for interpretation:
  - selected pseudo-positive winners contribute fixed-weight prefix bbox/coord supervision
  - support-positive retained shielded anchors that are not cluster-demoted contribute support-rate-weighted prefix bbox/coord supervision
  - cluster-demoted pseudo-positive candidates remain structure-only prefix context
- `train/triage/recovered_ground_truth_rate_num`
  - summed explorer-hit numerators for recovered GT objects missed by the anchor
- `train/triage/recovered_ground_truth_rate_den`
  - summed valid-explorer denominators for those recovered GT objects
- `train/triage/recovered_ground_truth_rate`
  - `rate_num / rate_den` when the denominator is non-zero
- `train/triage/anchor_preparation_dropped_count`
  - enabled pseudo-positive samples dropped because anchor accepted-clean preparation was malformed

Failure telemetry:

- malformed rollouts that remain invalid after salvage parsing abort the step
  by default instead of emitting an ordinary finalized `train/triage/*` counter
- with `stage2_ab.channel_b.invalid_rollout_policy=dump_and_continue`, the
  trainer logs `stage2_ab/channel_b/invalid_rollout_sample_dropped` and
  `stage2_ab/channel_b/invalid_rollout_sample_dropped_rate`
- treat those aborts as failure telemetry / run outcome, not as a step-level
  rolling metric

## Duplicate And Rollout Diagnostics

Canonical duplicate/rollout families include:

- `dup/raw/*`
- `stage2_ab/channel_b/dup/N_*`
- `rollout/*`
- `time/rollout_*`

Duplicate-control gauges are emitted on the raw pre-match object surface and
finalize as weighted means:

- `dup/raw/max_desc_count`
- `dup/raw/saturation_rate`
- `dup/raw/duplicate_like_max_cluster_size`
- `dup/raw/desc_entropy`
- `dup/raw/near_iou90_pairs_same_desc_count`
- `dup/raw/near_iou90_pairs_any_desc_count`

Canonical Channel-B duplicate-control counters remain additive:

- `stage2_ab/channel_b/dup/N_raw_bbox_valid`
- `stage2_ab/channel_b/dup/N_clean_accepted`
- `stage2_ab/channel_b/dup/N_clusters_total`
- `stage2_ab/channel_b/dup/N_clusters_exempt`
- `stage2_ab/channel_b/dup/N_clusters_suppressed`
- `stage2_ab/channel_b/dup/N_objects_suppressed`
- `stage2_ab/channel_b/dup/N_ul_boundaries`
- `stage2_ab/channel_b/dup/N_duplicate_burst_unlikelihood_skipped_no_divergence`

Use `docs/training/STAGE2_RUNBOOK.md` for the contract that produces these
families and `docs/ARTIFACTS.md` for where the corresponding monitor dumps and
run artifacts live.

## Stage-1 Set-Continuation Metrics

`custom.trainer_variant: stage1_set_continuation` emits a separate mechanism
metric family. These keys are not expected from ordinary Stage-1 SFT, and
ordinary metric-key parity tests assert that MP keys do not leak into the
baseline trainer.

Objective atoms:

- `loss/mp`: optimized MP loss when PEM is disabled.
- `loss/mp_diagnostic`: MP loss logged even when the PEM threshold loss is the
  optimized objective.
- `loss/pem`: positive-evidence margin loss when
  `positive_evidence_margin.objective=threshold_loss`.
- `loss/anti_close_start`: weighted close-start suppression objective when
  observed GT remains.
- `loss/anti_stop`: compatibility alias for `loss/anti_close_start`.
- `loss/weak_schema_close`: weighted weak global CoordJSON close-sequence loss
  for empty-remaining prefixes.
- `loss/eod`: compatibility alias for weak structural-close loss in this v1
  surface. It does not mean chat-template EOS.
- `loss/aux_coord_soft_ce_w1`: branch-local coord auxiliary loss, averaged
  uniformly over scored candidates with valid coord atoms.
- `loss/aux_bbox_geo`: branch-local bbox geometry auxiliary loss, averaged
  uniformly over scored candidates with valid bbox atoms.
- `loss/aux_bbox_size`: branch-local bbox size auxiliary loss, averaged
  uniformly over scored candidates with valid bbox atoms.

Prefix and candidate state:

- `mp/num_prefix_objects`
- `mp/num_remaining_objects`
- `mp/num_candidates_scored`
- `mp/scored_candidate_fraction`
- `mp/samples_with_candidates`
- `mp/samples_full_prefix`
- `mp/loss_mp_denominator_samples`
- `mp/objective_contributing_samples`
- `mp/selected_mode_empty_prefix`
- `mp/selected_mode_random_subset`
- `mp/selected_mode_leave_one_out`
- `mp/selected_mode_full_prefix`
- `mp/configured_ratio_<mode>`
- `mp/resolved_valid_ratio_<mode>`

Branch semantics and estimators use numeric codes in trainer metrics and
string values in run artifacts:

- `mp/candidate_scoring_mode`: `0=exact`, `1=uniform_subsample`.
- `mp/logZ_estimator`: `0=exact`, `1=sampled_raw`, `2=uniform_importance`.
- `mp/prefix_attach_mode`: `0=repeated_forward`.
- `mp/branch_isolation`: `0=independent_forward`.
- `mp/prefix_gradient`: `0=non_detached_recomputed_per_branch`.

LogZ and responsibility diagnostics:

- `mp/logZ_scored_raw`
- `mp/logZ_remaining_exact`
- `mp/logZ_remaining_est`
- `mp/logZ_remaining`
- `mp/responsibility_entropy`
- `mp/responsibility_entropy_scored`
- `mp/max_responsibility`
- `mp/max_responsibility_scored`
- `mp/min_responsibility`
- `mp/min_responsibility_scored`
- `mp/valid_length_corr_samples`
- `mp/responsibility_vs_length_corr`

Candidate full-entry logprob statistics:

- `mp/candidate_entry_tokens_mean`
- `mp/candidate_entry_tokens_min`
- `mp/candidate_entry_tokens_max`
- `mp/candidate_entry_tokens_std`
- `mp/candidate_logprob_sum_mean`
- `mp/candidate_logprob_sum_min`
- `mp/candidate_logprob_sum_max`
- `mp/candidate_logprob_sum_std`
- `mp/candidate_logprob_per_token_mean`
- `mp/candidate_logprob_per_token_min`
- `mp/candidate_logprob_per_token_max`
- `mp/candidate_logprob_per_token_std`
- `mp/candidate_coord_token_fraction_mean`
- `mp/candidate_coord_token_fraction_min`
- `mp/candidate_coord_token_fraction_max`
- `mp/candidate_coord_token_fraction_std`
- `mp/candidate_logprob_per_coord_token_mean`
- `mp/candidate_logprob_per_noncoord_token_mean`

Stop/continue diagnostics:

- `stop/p_close_start_when_remaining_exists`
- `stop/p_continue_start_when_remaining_exists`
- `stop/p_stop_when_remaining_exists`
- `stop/p_continue_when_remaining_exists`
- `stop/p_close_start_when_remaining_empty`
- `stop/p_stop_when_remaining_empty`
- `stop/logp_close_sequence_when_remaining_empty`
- `stop/p_final_schema_token_teacher_forced`

Budget diagnostics:

- `mp/prefix_tokens_mean`
- `mp/total_candidate_tokens_scored`
- `mp/candidate_tokens_scored_mean`
- `mp/repeated_forward_token_ratio_vs_baseline`

Branch-local aux counters:

- `aux/<name>/candidate_count`
- `aux/<name>/position_count`
- `aux/<name>/skipped_candidates`
- `aux/<name>/contributing_candidates`

Ordinary SFT Group B close-control metrics:

- `sft_structural_close/final_close_weight`
- `sft_structural_close/final_close_tokens`
- `sft_structural_close/weighted_token_sum`
- `loss/sft_structural_close_base_ce`
- `loss/eod`, where `eod` means the final global CoordJSON close sequence `]}`,
  not chat-template EOS.

## Training-Time Evaluation Families

Two distinct eval surfaces exist during training:

- offline evaluator callback:
  - `eval_det_*`
- trainer-native Stage-2 rollout eval:
  - shared by `stage2_two_channel` and `stage2_rollout_aligned`
  - `eval/detection/*`
  - `eval/parsing/*`
  - `eval/description/*`
  - `eval/config/*`
  - `eval/runtime/*`

## Removed Historical Families

Legacy iterative Channel-A provenance groups are no longer part of the active
contract:

- `loss/A1_*`
- `loss/A2_*`
- `coord_diag/A1/*`
- `coord_diag/A2/*`
- `eval_rollout/*`

If they appear in old logs, treat them as historical artifacts rather than
current contract surfaces.

## Historical Reference

The deprecation rationale lives in:

- `progress/diagnostics/stage2_channel_a_self_context_iter_ablation_2026-03-20.md`
