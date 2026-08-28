# `scripts/research` Entropy Ledger

Change: `reclaim-research-probes-lifecycle`, tasks 5.1-5.5. Method: `reclaim-code-entropy`
(Prove Or Reject A Candidate / Implement A Proven Cut / Validate The Result), classes per
`design.md` D6. Frozen 2026-08-28 against `research-probes` @ `57d988351`.

## Method

For each of the 315 tracked `scripts/research/*.py`, every occurrence of its module stem
(`scripts.research.<stem>`, `from scripts.research import <stem>`, `scripts/research/<stem>.py`,
bare basename) was collected by identifier-token scan over 9160 text files and classified by
the containing path:

| Bucket | Paths |
| --- | --- |
| (a) production | `src/**`, `scripts/**` outside `scripts/research/`, `configs/**`, `vis_tools/`, `repo_lifecycle/`, `manifests/`, `handoff/`, `reference/`, root manifests |
| (SR) intra-directory | `scripts/research/**` (resolved through the import/reference graph, not counted as production) |
| (b) tests | `tests/**` |
| (c) research records | `research/**`, `memories/**` |
| (d) current docs | `docs/**` except `docs/history/**`, plus non-archive `openspec/changes/*/` |
| (e) historical | `docs/history/**`, `progress/**`, `openspec/changes/archive/**` |
| (f) live direction worktree | `/data/CoordExp/.worktrees/image2299-mechanism-microscope`, tracked + untracked, minus `.git/` and `outputs/` |

`image2299-mechanism-microscope` is a full fork of this tree, so **every** script has a byte-identical
mirror there. A (f) hit therefore counts only when the containing file is absent from `research-probes`
or differs from it (md5); the script's own mirrored copy never counts. Without this filter every one of
the 315 scripts would classify SHARED and the class would carry no information.

Intra-directory references are resolved as a graph: a script referenced by a kept script is kept,
transitively. A script referenced only by scripts that are themselves deletable is deletable (`SR_ONLY`).

## Class counts

| Class | Count | Disposition |
| --- | ---: | --- |
| SHARED (production consumer, or live-direction consumer, or transitively imported by a kept script) | 125 | keep |
| RECORD_CITED (`research/` or `memories/` names it) | 42 | keep (evidence obligation) |
| DOCS_CITED (current docs only) | 22 | HOLD, task 5.4 |
| TEST_ONLY | 88 | 49 deleted (batch 2), 39 HOLD (31 blocked by a shared test module, 8 reverted as load-bearing) |
| ZERO | 29 | deleted (batch 1), minus 1 prior-decision HOLD |
| SR_ONLY (referenced only by deletable scripts) | 9 | deleted (batch 1) |
| **total** | **315** | |

## Sanity check against the lead's coarse scan

The lead's coarse scan over `src/ scripts/ tests/ research/ docs/` reported 38 / 87 / 190.

| Lead | This ledger | Agreement |
| --- | --- | --- |
| 38 referenced nowhere | 29 ZERO + 9 SR_ONLY = **38** | exact |
| 87 referenced by exactly one place | 88 TEST_ONLY | +1 |
| 190 referenced by >= 2 places | 125 SHARED + 42 RECORD_CITED + 22 DOCS_CITED = **189** | -1 |

The +1/-1 is one script whose single extra reference is an intra-`scripts/research` mention that the
coarse scan counted as a second place and the graph resolves as a dead edge. No material difference.

## KEEP list

### SHARED (125) - keep

`adjudicate_sorted_all_person_route_landscape_provenance`, `analyze_human13_k_union`, `analyze_s_natural_boundary_k_n_h_evidence`, `analyze_sorted_all_person_route_landscape`, `analyze_sorted_crossing_boundary_owner_release`, `analyze_sorted_crossing_owner_row_geometry`, `analyze_sorted_supported_fn_native_prefix_reachability_prevalence`, `analyze_span_likelihood_separation`, `analyze_trajectory_owner_set_adjudication_salvage_gate`, `assemble_row_local_owner_stop_state_bank`, `attest_sorted_all_person_route_landscape`, `build_earliest_shared_prefix_branch_cases`, `build_human13_k_union_manifest`, `build_sorted_all_person_greedy_boundary_census`, `build_sorted_all_person_route_landscape`, `build_sorted_fn_mechanism_registry`, `build_sorted_owner_basin_candidates`, `build_sorted_owner_basin_census`, `build_sorted_owner_basin_cohorts`, `build_sorted_owner_basin_contexts`, `build_sorted_owner_basin_runtime_identity`, `capture_natural_boundary_support_source_bindings`, `census_human13_k_union_trie`, `check_research_graph`, `collect_human13_k16_vllm`, `collect_vllm_trajectory_panel`, `execute_human13_k_union`, `finalize_natural_boundary_routing_history_evidence`, `finalize_s_k10_h20_crossover`, `generate_research_probe_admission_cpu_receipts`, `human13_adamw_proposal_preservation`, `human13_all_hf_vertical`, `human13_continuation_projection`, `human13_cuda_cpu_adapter`, `human13_forced_continuation`, `human13_graph_owner`, `human13_greedy_compiler`, `human13_hf_census`, `human13_hf_native_one_image_owner`, `human13_hf_native_projection`, `human13_hf_shared_surface`, `human13_hf_shared_surface_live`, `human13_live_eval`, `human13_live_model`, `human13_on_policy_runtime`, `human13_one_image_services`, `human13_proposal_checkpoint`, `human13_rp_crossover_live_composition`, `human13_rp_crossover_matrix_contracts`, `human13_rp_crossover_production`, `human13_rp_crossover_production_backend`, `human13_rp_crossover_runtime`, `human13_rp_crossover_witness`, `human13_source_surface_reconciliation`, `human13_training_transaction`, `human13_trajectory_credit`, `materialize_continuation_locality_owner_compositionality`, `materialize_human13_no_update_census`, `materialize_natural_boundary_census_v3`, `materialize_natural_boundary_pre_gpu_evidence`, `materialize_s_k10_h20_crossover_plan`, `materialize_s_natural_boundary_k_n_h_pre_gpu_evidence`, `materialize_static_dynamic_owner_interface_cohort`, `merge_natural_boundary_support_completion`, `merge_s_natural_boundary_k_n_h_shards`, `merge_sorted_all_person_route_landscape`, `merge_sorted_crossing_boundary_owner_release`, `merge_sorted_crossing_boundary_owner_release_secondary`, `merge_sorted_fn_successor_score_shards`, `merge_sorted_owner_basin_landscape_shards`, `natural_boundary_attention_actuators`, `natural_boundary_residual_actuators`, `plan_natural_boundary_owner_support_completion`, `plan_s_natural_boundary_k_n_h_execution`, `prepare_sorted_crossing_boundary_owner_release_realization`, `prepare_sorted_fn_successor_inputs`, `prepare_sorted_owner_basin_inputs`, `reanalyze_sorted_fn_fixed_budget_controls`, `research_probe_admission_consumers`, `resumable_natural_boundary_support_completion`, `run_batch_coordinate_logit_invariance`, `run_complete_candidate_row_scoring`, `run_coordinate_history_crossover`, `run_dynamic_history_and_crossover_probe`, `run_exact_prefix_owner_compositionality`, `run_exact_prefix_sampled_rescue`, `run_fixed_encoding_query_scoped_object_centered_spatial_eligibility`, `run_fixed_prefix_complete_box_coherence`, `run_greedy_prefix_forced_owner_path`, `run_human13_all_hf_shared_surface_vertical`, `run_human13_k_union_overfit`, `run_local_branch_causal_value`, `run_natural_boundary_routing_history_probe`, `run_natural_boundary_support_completion`, `run_next_row_likelihood_change`, `run_research_probe_admission_vertical`, `run_resumable_natural_boundary_support_shard`, `run_row_four_coordinate_factorial`, `run_s_k10_h20_crossover_shard`, `run_s_natural_boundary_k_n_h_cohort`, `run_s_natural_boundary_k_n_h_shard`, `run_s_primary_natural_boundary_gate`, `run_same_covered_set_prefix_order_probe`, `run_same_parent_complete_row_intervention`, `run_same_parent_final_horizon`, `run_sampled_history_prefix_sufficiency_ladder`, `run_sampled_history_target_reachability`, `run_sorted_fn_successor_behavior`, `run_static_dynamic_gradient_path_audit`, `run_static_dynamic_owner_interface_experiment`, `run_static_dynamic_owner_support_probe`, `run_static_post_llm_image_field_probe`, `s_natural_boundary_k_n_h_live_executor`, `score_sorted_all_person_route_landscape`, `score_sorted_crossing_boundary_owner_release`, `score_sorted_crossing_boundary_owner_release_secondary`, `score_sorted_fn_fixed_budget`, `score_sorted_owner_basin_landscape`, `seal_natural_boundary_pre_gpu_receipt`, `seal_s_k10_h20_crossover_pre_gpu_receipt`, `seal_s_natural_boundary_k_n_h_pre_gpu_receipt`, `sorted_owner_basin_landscape`, `summarize_sorted_owner_basin_landscape`, `validate_constant_dose_trajectory_panel_union`, `validate_sampled_history_prefix_sufficiency_ladder_union`

### RECORD_CITED (42) - keep

`analyze_cluster_confidence_retention`, `analyze_common_owner_count_robustness`, `analyze_image2299_near_complete_relabel_successor_transition`, `analyze_individual_trajectory_union_support`, `analyze_ranking_versus_operating_point`, `analyze_sorted_crossing_neutral_row_control`, `analyze_trajectory_owner_set_admission_census`, `assemble_constant_dose_breadth_state_banks`, `assemble_positive_path_imitation_state_bank`, `assemble_source_preservation_multi_route_state_banks`, `assemble_trajectory_owner_set_adjudication_review`, `build_cluster_confidence`, `build_common_object_prefix_permutation_cases`, `build_inference_coordinate_boundary_state_bank`, `build_matched_random_sorted_candidate_score_manifest`, `build_sorted_owner_accessibility_census_plan`, `build_sorted_prospective_13_image_panel`, `compute_sampled_union_f1_metrics`, `derive_likelihood_mining_baseline`, `human13_rp_crossover_live_packs`, `merge_sorted_crossing_neutral_row_control`, `merge_sorted_owner_accessibility_census_shards`, `prepare_sorted_crossing_neutral_row_control`, `run_current_seeded_sampled_rollouts`, `run_earliest_shared_prefix_branch_pilot`, `run_fixed_encoding_downstream_residual_state_portability`, `run_fixed_encoding_object_centered_spatial_eligibility_crossover`, `run_fixed_prefix_nonboundary_box_grammar_image19432`, `run_historical_random_sorted_image2299_screen`, `run_human13_rp_crossover_parity_v5`, `run_native_commit_redistribution`, `run_native_sibling_branch_replay`, `run_person25_commit_closeout`, `run_sampled_rescue_transition`, `run_span_likelihood_replay`, `score_person25_y2_competition`, `score_sorted_crossing_neutral_row_control`, `score_sorted_owner_accessibility_census_shard`, `select_trajectory_owner_set_adjudication_review_sample`, `summarize_same_covered_set_prefix_order_probe`, `verify_likelihood_mining_contracts`, `visualize_sorted_owner_accessibility_visual_atlas`

`DOCS_CITED` and the blocked `TEST_ONLY` scripts are also kept by this change; they are listed under
HOLD below because their retention decision is the user's, not evidence of a live consumer.

## Batch 1 - ZERO and SR_ONLY (37 scripts)

No hit in (a), (b), (c), (d), or (f). Remaining hits are (e) historical mentions in archived OpenSpec
changes, and intra-batch imports among the candidates themselves.

```
[low / low] scripts/research/analyze_own_terminal_forced_opener_panel.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/analyze_own_terminal_forced_opener_panel.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/attest_request_scoped_sampling_source.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/attest_request_scoped_sampling_source.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/compare_image2299_random_sorted_accessibility.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/compare_image2299_random_sorted_accessibility.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/compose_three_detection_panels.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/compose_three_detection_panels.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/materialize_forced_continue_behavior_review.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/materialize_forced_continue_behavior_review.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[high / low] scripts/research/materialize_forced_continue_full_trajectory_visuals.py
evidence: intra-batch importers: `materialize_forced_continue_post_native_prefix_visuals` (all in this batch); (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/materialize_forced_continue_full_trajectory_visuals.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/materialize_forced_continue_post_native_prefix_visuals.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/materialize_forced_continue_post_native_prefix_visuals.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/materialize_row_local_long_milestone_infer_configs.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/materialize_row_local_long_milestone_infer_configs.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/materialize_source_transition_forced_opener_visual_review.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/materialize_source_transition_forced_opener_visual_review.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/materialize_state_bank_inference_panel.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/materialize_state_bank_inference_panel.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/materialize_transition_step36_transfer_infer_configs.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/materialize_transition_step36_transfer_infer_configs.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/qwen3_vl_native_text_coordinate_val200.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/qwen3_vl_native_text_coordinate_val200.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[high / low] scripts/research/render_coordinate_confidence_boxes.py
evidence: intra-batch importers: `render_forced_continue_coord_confidence` (all in this batch); (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/render_coordinate_confidence_boxes.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/render_forced_continue_coord_confidence.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/render_forced_continue_coord_confidence.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/render_forced_continue_exact_full_trajectories.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/render_forced_continue_exact_full_trajectories.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[high / low] scripts/research/run_batch_precision_selected_transition_prevalence.py
evidence: intra-batch importers: `run_repeated_first_differing_slot_coordinate_logits` (all in this batch); (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/run_batch_precision_selected_transition_prevalence.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/run_donor_state_late_coordinate_transport_image7818.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/run_donor_state_late_coordinate_transport_image7818.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/run_fixed_encoding_count_balanced_soft_cross_region_earlier_and_row_query_spatial_bias.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/run_fixed_encoding_count_balanced_soft_cross_region_earlier_and_row_query_spatial_bias.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[high / low] scripts/research/run_fixed_encoding_cross_region_earlier_and_row_query_spatial_key_eligibility_hybrid.py
evidence: intra-batch importers: `run_fixed_encoding_count_balanced_soft_cross_region_earlier_and_row_query_spatial_bias` (all in this batch); (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/run_fixed_encoding_cross_region_earlier_and_row_query_spatial_key_eligibility_hybrid.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[high / low] scripts/research/run_fixed_encoding_downstream_geometry_state_portability.py
evidence: intra-batch importers: `run_fixed_encoding_persistent_hard_geometry_donor_eligibility_screen`, `run_fixed_encoding_persistent_hard_geometry_state_portability_image7818` (all in this batch); (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/run_fixed_encoding_downstream_geometry_state_portability.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[high / low] scripts/research/run_fixed_encoding_earlier_query_only_spatial_key_eligibility_factorial.py
evidence: intra-batch importers: `run_fixed_encoding_cross_region_earlier_and_row_query_spatial_key_eligibility_hybrid` (all in this batch); (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/run_fixed_encoding_earlier_query_only_spatial_key_eligibility_factorial.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[high / low] scripts/research/run_fixed_encoding_persistent_hard_geometry_donor_eligibility_screen.py
evidence: intra-batch importers: `run_donor_state_late_coordinate_transport_image7818`, `run_fixed_encoding_persistent_hard_geometry_state_portability_image7818` (all in this batch); (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/run_fixed_encoding_persistent_hard_geometry_donor_eligibility_screen.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[high / low] scripts/research/run_fixed_encoding_persistent_hard_geometry_state_portability_image7818.py
evidence: intra-batch importers: `run_donor_state_late_coordinate_transport_image7818` (all in this batch); (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/run_fixed_encoding_persistent_hard_geometry_state_portability_image7818.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/run_fixed_encoding_soft_spatial_key_bias_dose_response.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/run_fixed_encoding_soft_spatial_key_bias_dose_response.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/run_own_terminal_forced_opener_panel.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/run_own_terminal_forced_opener_panel.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/run_paired_terminal_forced_opener_release.py
evidence: (e) historical only: `openspec/changes/archive/2026-08-06-add-hf-exact-history-evidence-seam/design.md`:66; (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/run_paired_terminal_forced_opener_release.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/run_prefix_row_factorial_likelihood.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/run_prefix_row_factorial_likelihood.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/run_prevision_raw_pixel_visual_support_counterfactual_commit.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/run_prevision_raw_pixel_visual_support_counterfactual_commit.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/run_repeated_first_differing_slot_coordinate_logits.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/run_repeated_first_differing_slot_coordinate_logits.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/run_single_target_visual_feature_replay_batch_four.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/run_single_target_visual_feature_replay_batch_four.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[high / low] scripts/research/run_visual_support_counterfactual_commit.py
evidence: intra-batch importers: `run_prevision_raw_pixel_visual_support_counterfactual_commit` (all in this batch); (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/run_visual_support_counterfactual_commit.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/score_forced_continue_coord_likelihood.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/score_forced_continue_coord_likelihood.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/select_coordinate_boundary_candidate_pool.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/select_coordinate_boundary_candidate_pool.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/summarize_native_duplication.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/summarize_native_duplication.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/summarize_prefix_state_factorial.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/summarize_prefix_state_factorial.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/summarize_row_local_long_milestone_clean_greedy.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/summarize_row_local_long_milestone_clean_greedy.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

```
[low / low] scripts/research/summarize_transition_step36_transfer_clean_greedy.py
evidence: (a)=0 (b)=0 (c)=0 (d)=0 (f)=0; no dedicated test module; no config cited only by it
cut: scripts/research/summarize_transition_step36_transfer_clean_greedy.py
tradeoff: none observable; replay via research-base-v2
verify: pytest tests/research tests/artifacts failure set unchanged vs receipts/test-baseline.md
```

## Batch 2 - TEST_ONLY with a fully dedicated test module (49 scripts, 48 test modules)

Cited only by `tests/**` (plus (e) historical). A script is eligible only when **every** test module that
loads it - by `from scripts.research import ...`, `scripts.research.<stem>`, or split-path
`importlib.util.spec_from_file_location` over a `"<stem>.py"` literal - loads *nothing but* scripts in the
delete set. A test module that also loads a kept script is never touched; its script goes to HOLD-C.

```
[low / low] scripts/research/aggregate_static_dynamic_owner_interface_shards.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=2; (e) historical only: `openspec/changes/archive/2026-08-07-add-resumable-costed-research-probe-shards/verification/source-bindings-v2.json`:1, `openspec/changes/archive/2026-08-07-add-resumable-costed-research-probe-shards/verification/source-bindings-v3.json`:1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/aggregate_static_dynamic_owner_interface_shards.py + tests/research/test_aggregate_static_dynamic_owner_interface_shards.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/analyze_gradient_cohort_treatment_owner_attribution.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/analyze_gradient_cohort_treatment_owner_attribution.py + tests/research/test_analyze_gradient_cohort_treatment_owner_attribution.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/analyze_human13_row_contrast_successor.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/analyze_human13_row_contrast_successor.py + tests/research/test_analyze_human13_row_contrast_successor.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/analyze_sampled_owner_inclusion.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/analyze_sampled_owner_inclusion.py + tests/research/test_analyze_sampled_owner_inclusion.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/analyze_selected_route_added_owner_transfer.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=3; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/analyze_selected_route_added_owner_transfer.py + tests/research/test_analyze_selected_route_added_owner_transfer.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/analyze_sorted_root_position_bias.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/analyze_sorted_root_position_bias.py + tests/research/test_analyze_sorted_root_position_bias.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/analyze_source_b16_treatment_owner_ledger.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/analyze_source_b16_treatment_owner_ledger.py + tests/research/test_analyze_source_b16_treatment_owner_ledger.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/analyze_source_route_preservation_screen.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=3; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/analyze_source_route_preservation_screen.py + tests/research/test_analyze_source_route_preservation_screen.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/assemble_duplicate_trajectory_state_banks.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=2; intra-directory referrers: `materialize_reviewed_duplication_training_inputs` (all deletable); no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/assemble_duplicate_trajectory_state_banks.py + tests/research/test_assemble_duplicate_trajectory_state_banks.py, tests/research/test_materialize_reviewed_duplication_training_inputs.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/assemble_exact_greedy_terminal_rescue_state_bank.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/assemble_exact_greedy_terminal_rescue_state_bank.py + tests/research/test_assemble_exact_greedy_terminal_rescue_state_bank.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/audit_sorted_image2299_legacy_transfer_context.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/audit_sorted_image2299_legacy_transfer_context.py + tests/research/test_audit_sorted_image2299_legacy_transfer_context.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/build_exact_trace_coordinate_review_packet.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; intra-directory referrers: `build_sampled_rescue_review_packet` (all deletable); no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/build_exact_trace_coordinate_review_packet.py + tests/research/test_build_exact_trace_coordinate_review_packet.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/build_heldout_owner_churn_review_packet.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/build_heldout_owner_churn_review_packet.py + tests/research/test_build_heldout_owner_churn_review_packet.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/build_physical_owner_duplication_review_queue.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=6; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/build_physical_owner_duplication_review_queue.py + tests/research/test_build_physical_owner_duplication_review_queue.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/build_sampled_rescue_review_packet.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/build_sampled_rescue_review_packet.py + tests/research/test_build_sampled_rescue_review_packet.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/build_source_preservation_matched_dose_control_state_banks.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/build_source_preservation_matched_dose_control_state_banks.py + tests/research/test_build_source_preservation_matched_dose_control_state_banks.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/build_static_dynamic_owner_interface_h0_ledger.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=2; (e) historical only: `openspec/changes/archive/2026-08-07-add-resumable-costed-research-probe-shards/verification/source-bindings-v2.json`:1, `openspec/changes/archive/2026-08-07-add-resumable-costed-research-probe-shards/verification/source-bindings-v3.json`:1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/build_static_dynamic_owner_interface_h0_ledger.py + tests/research/test_build_static_dynamic_owner_interface_h0_ledger.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/build_static_dynamic_owner_interface_inputs.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; (e) historical only: `openspec/changes/archive/2026-08-07-add-resumable-costed-research-probe-shards/verification/source-bindings-v2.json`:1, `openspec/changes/archive/2026-08-07-add-resumable-costed-research-probe-shards/verification/source-bindings-v3.json`:1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/build_static_dynamic_owner_interface_inputs.py + tests/research/test_build_static_dynamic_owner_interface_inputs.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/build_unmatched_prediction_review.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/build_unmatched_prediction_review.py + tests/research/test_build_unmatched_prediction_review.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/collect_exact_greedy_terminal_rescue_events.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/collect_exact_greedy_terminal_rescue_events.py + tests/research/test_collect_exact_greedy_terminal_rescue_events.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/compare_constant_dose_breadth_frozen_step.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=2; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/compare_constant_dose_breadth_frozen_step.py + tests/research/test_compare_constant_dose_breadth_frozen_step.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/compare_repetition_penalty_owner_ledgers.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/compare_repetition_penalty_owner_ledgers.py + tests/research/test_compare_repetition_penalty_owner_ledgers.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/evaluate_frozen_coordinate_state_bank.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=2; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/evaluate_frozen_coordinate_state_bank.py + tests/test_frozen_coordinate_state_bank_eval.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/freeze_local_branch_cases.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/freeze_local_branch_cases.py + tests/research/test_freeze_local_branch_cases.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/human13_live_eval_matrix.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/human13_live_eval_matrix.py + tests/research/test_human13_live_eval_matrix.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/inventory_matched_objective_native_coordinate_trajectories.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/inventory_matched_objective_native_coordinate_trajectories.py + tests/research/test_inventory_matched_objective_native_coordinate_trajectories.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/materialize_candidate_pool_rollout_complement.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=5; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/materialize_candidate_pool_rollout_complement.py + tests/research/test_materialize_candidate_pool_rollout_complement.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/materialize_constant_dose_breadth_source_b16_development_eval_configs.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=2; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/materialize_constant_dose_breadth_source_b16_development_eval_configs.py + tests/research/test_materialize_constant_dose_breadth_source_b16_development_eval_configs.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/materialize_reviewed_duplication_training_inputs.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/materialize_reviewed_duplication_training_inputs.py + tests/research/test_materialize_reviewed_duplication_training_inputs.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/materialize_transition_phase0_fixed_prefix_panel.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; intra-directory referrers: `run_transition_phase0_fixed_prefix_release` (all deletable); no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/materialize_transition_phase0_fixed_prefix_panel.py + tests/research/test_transition_phase0_fixed_prefix_panel.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/mine_sorted_all_person_owner_phenotypes.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/mine_sorted_all_person_owner_phenotypes.py + tests/research/test_mine_sorted_all_person_owner_phenotypes.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/prepare_transition_phase0_candidate_scoring.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/prepare_transition_phase0_candidate_scoring.py + tests/research/test_prepare_transition_phase0_candidate_scoring.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/run_image_12576_row_mediation_crossover.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/run_image_12576_row_mediation_crossover.py + tests/research/test_run_image_12576_row_mediation_crossover.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/run_iterative_forced_continue_extreme_capacity.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/run_iterative_forced_continue_extreme_capacity.py + tests/research/test_iterative_forced_continue_extreme_capacity.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/run_native_sibling_branch_atlas.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/run_native_sibling_branch_atlas.py + tests/analysis/test_native_sibling_row_branch_atlas.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/run_transition_phase0_fixed_prefix_release.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/run_transition_phase0_fixed_prefix_release.py + tests/research/test_transition_phase0_fixed_prefix_panel.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/seal_natural_boundary_routing_history_contract.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; (e) historical only: `openspec/changes/archive/2026-08-07-add-resumable-costed-research-probe-shards/verification/source-bindings-v2.json`:1, `openspec/changes/archive/2026-08-07-add-resumable-costed-research-probe-shards/verification/source-bindings-v3.json`:1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/seal_natural_boundary_routing_history_contract.py + tests/research/test_seal_natural_boundary_routing_history_contract.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/seal_static_dynamic_owner_interface_eligible_hold_leaf.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; (e) historical only: `openspec/changes/archive/2026-08-07-add-resumable-costed-research-probe-shards/verification/source-bindings-v2.json`:1, `openspec/changes/archive/2026-08-07-add-resumable-costed-research-probe-shards/verification/source-bindings-v3.json`:1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/seal_static_dynamic_owner_interface_eligible_hold_leaf.py + tests/research/test_seal_static_dynamic_owner_interface_eligible_hold_leaf.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/select_constant_dose_breadth_shared_milestone.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=2; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/select_constant_dose_breadth_shared_milestone.py + tests/research/test_select_constant_dose_breadth_shared_milestone.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/subset_rollout_calibration_state_bank.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/subset_rollout_calibration_state_bank.py + tests/research/test_subset_rollout_calibration_state_bank.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/summarize_candidate_row_score_deltas.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/summarize_candidate_row_score_deltas.py + tests/research/test_summarize_candidate_row_score_deltas.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/summarize_heldout_owner_churn_review.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/summarize_heldout_owner_churn_review.py + tests/research/test_summarize_heldout_owner_churn_review.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/summarize_matched_random_sorted_prefix_order.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/summarize_matched_random_sorted_prefix_order.py + tests/test_summarize_matched_random_sorted_prefix_order.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/summarize_paired_terminal_forced_opener_release.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; intra-directory referrers: `summarize_source_transition_forced_opener_comparison` (all deletable); no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/summarize_paired_terminal_forced_opener_release.py + tests/research/test_paired_terminal_forced_opener_release.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/summarize_source_transition_forced_opener_comparison.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; intra-directory referrers: `materialize_source_transition_forced_opener_visual_review` (all deletable); no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/summarize_source_transition_forced_opener_comparison.py + tests/research/test_source_transition_forced_opener_comparison.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/summarize_transition_phase0_fixed_prefix_panel.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/summarize_transition_phase0_fixed_prefix_panel.py + tests/research/test_summarize_transition_phase0_fixed_prefix_panel.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/summarize_transition_phase0_robustness.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/summarize_transition_phase0_robustness.py + tests/research/test_summarize_transition_phase0_robustness.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/summarize_untouched_terminal_boundary_statistics.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/summarize_untouched_terminal_boundary_statistics.py + tests/research/test_summarize_untouched_terminal_boundary_statistics.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

```
[low / low] scripts/research/visualize_sorted_full_canvas_token_budget_intervention.py
evidence: (a)=0 (c)=0 (d)=0 (f)=0; (b)=1; no orphaned schema-version producer binding in `research/` or `memories/`
cut: scripts/research/visualize_sorted_full_canvas_token_budget_intervention.py + tests/research/test_visualize_sorted_full_canvas_token_budget_intervention.py
tradeoff: none observable; replay via research-base-v2
verify: pytest --collect-only over tests/, then tests/research tests/artifacts failure-set diff
```

## HOLD - reported, not deleted (task 5.4)

Nothing in this section is touched by this change. Each row is a user decision: either the citation is
rewritten to point at `research-base-v2` (which contains every file listed here) and the script is then
deletable, or the script stays as a live replay entry.

### HOLD-A. DOCS_CITED - cited by current docs, no production/record/live-direction consumer

| Script | Citing docs path(s) |
| --- | --- |
| `analyze_human13_k_trajectory_rp_crossover` | `docs/superpowers/plans/2026-08-14-human13-k-trajectory-rp-crossover-screen.md` |
| `build_human13_on_policy_frontier` | `docs/superpowers/plans/2026-08-13-human13-on-policy-first-bottleneck-successor.md` |
| `build_human13_row_contrast_successor` | `docs/superpowers/plans/2026-08-13-human13-row-contrast-geometry-preservation-successor.md` |
| `collect_human13_discovery` | `docs/superpowers/plans/2026-08-14-human13-k-trajectory-rp-crossover-screen.md` |
| `collect_human13_rp_crossover` | `docs/superpowers/plans/2026-08-14-human13-k-trajectory-rp-crossover-screen.md` |
| `compare_clean_rollout_owner_coverage` | `docs/superpowers/plans/2026-08-12-human13-k-union-greedy-overfit-probe.md` |
| `human13_frontier_selection` | `docs/superpowers/plans/2026-08-13-human13-on-policy-first-bottleneck-successor.md` |
| `human13_gradient_preservation` | `docs/superpowers/plans/2026-08-13-human13-row-contrast-geometry-preservation-successor.md` |
| `human13_k_trajectory_contracts` | `docs/superpowers/plans/2026-08-14-human13-k-trajectory-rp-crossover-screen.md` |
| `human13_live_census` | `docs/superpowers/plans/2026-08-13-human13-missing-arms-recovery.md` |
| `human13_live_payload` | `docs/superpowers/plans/2026-08-13-human13-missing-arms-recovery.md`; `docs/superpowers/plans/2026-08-13-human13-on-policy-first-bottleneck-successor.md`; `docs/superpowers/plans/2026-08-13-human13-row-contrast-geometry-preservation-successor.md` |
| `human13_live_segments` | `docs/superpowers/plans/2026-08-13-human13-missing-arms-recovery.md`; `docs/superpowers/plans/2026-08-13-human13-on-policy-first-bottleneck-successor.md`; `docs/superpowers/plans/2026-08-13-human13-row-contrast-geometry-preservation-successor.md` |
| `human13_on_policy_live` | `docs/superpowers/plans/2026-08-13-human13-on-policy-first-bottleneck-successor.md` |
| `human13_on_policy_scoring` | `docs/superpowers/plans/2026-08-14-human13-k-trajectory-rp-crossover-screen.md` |
| `human13_rp_policy` | `docs/superpowers/plans/2026-08-14-human13-k-trajectory-rp-crossover-screen.md` |
| `launch_human13_k_trajectory_rp_crossover` | `docs/superpowers/plans/2026-08-14-human13-k-trajectory-rp-crossover-screen.md` |
| `launch_human13_k_union_matrix` | `docs/superpowers/plans/2026-08-12-human13-k-union-greedy-overfit-probe.md`; `docs/superpowers/plans/2026-08-13-human13-row-contrast-geometry-preservation-successor.md` |
| `materialize_human13_k_union_configs` | `docs/superpowers/plans/2026-08-12-human13-k-union-greedy-overfit-probe.md`; `docs/superpowers/plans/2026-08-13-human13-row-contrast-geometry-preservation-successor.md` |
| `run_human13_live_census` | `docs/superpowers/plans/2026-08-13-human13-missing-arms-recovery.md` |
| `train_human13_k_trajectory_rp_crossover` | `docs/superpowers/plans/2026-08-14-human13-k-trajectory-rp-crossover-screen.md` |
| `train_human13_live_arm` | `docs/superpowers/plans/2026-08-13-human13-missing-arms-recovery.md`; `docs/superpowers/plans/2026-08-13-human13-row-contrast-geometry-preservation-successor.md` |
| `train_human13_on_policy_successor` | `docs/superpowers/plans/2026-08-13-human13-on-policy-first-bottleneck-successor.md` |

### HOLD-B. RECORD_CITED but idle - only citation is a closed research unit

| Script | Citing record(s) | Owning unit status |
| --- | --- | --- |
| `analyze_cluster_confidence_retention` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-29-likelihood-filter-robustness-and-panel-annotation-validity/unit.md` | complete |
| `analyze_common_owner_count_robustness` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-29-likelihood-filter-robustness-and-panel-annotation-validity/unit.md` | complete |
| `analyze_image2299_near_complete_relabel_successor_transition` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-18-image2299-near-complete-human-relabel-successor-transition/results.md` | complete |
| `analyze_ranking_versus_operating_point` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-29-ranking-quality-versus-usable-rejection/unit.md` | complete |
| `analyze_sorted_crossing_neutral_row_control` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-04-sorted-crossing-matched-length-neutral-row-insertion-control/unit.md` | complete |
| `assemble_positive_path_imitation_state_bank` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-23-trajectory-owner-set-admission-census/unit.md` | complete |
| `build_cluster_confidence` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-29-sampled-span-likelihood-and-consensus-union-filtering/unit.md` | complete |
| `build_common_object_prefix_permutation_cases` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-19-common-object-prefix-permutation-short-horizon/results.md` | complete |
| `build_matched_random_sorted_candidate_score_manifest` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-20-matched-random-sorted-prefix-order-screen/candidate-row-scoring-manifest.json` | complete |
| `build_sorted_owner_accessibility_census_plan` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-03-sorted-owner-accessibility-phenotype-census/tasks.md`; `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-03-sorted-owner-accessibility-phenotype-census/unit.md` | complete |
| `build_sorted_prospective_13_image_panel` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-04-sorted-prospective-13-image-panel-admission/tasks.md`; `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-04-sorted-prospective-13-image-panel-admission/unit.md` | complete |
| `compute_sampled_union_f1_metrics` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-29-sampled-span-likelihood-and-consensus-union-filtering/unit.md` | complete |
| `derive_likelihood_mining_baseline` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-29-sampled-span-likelihood-and-consensus-union-filtering/unit.md` | complete |
| `human13_rp_crossover_live_packs` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-14-human13-k-trajectory-rp-crossover-screen/unit.md` | complete |
| `merge_sorted_crossing_neutral_row_control` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-04-sorted-crossing-matched-length-neutral-row-insertion-control/unit.md` | complete |
| `merge_sorted_owner_accessibility_census_shards` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-03-sorted-owner-accessibility-phenotype-census/tasks.md`; `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-03-sorted-owner-accessibility-phenotype-census/unit.md`; `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-03-sorted-supported-fn-native-prefix-reachability-prevalence/unit.md` | complete |
| `prepare_sorted_crossing_neutral_row_control` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-04-sorted-crossing-matched-length-neutral-row-insertion-control/unit.md` | complete |
| `run_current_seeded_sampled_rollouts` | `memories/notes/2026-07-22-constant-dose-image-breadth-screen-checkpoint.md`; `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-21-individual-trajectory-versus-union-support-audit/results.md` | complete |
| `run_earliest_shared_prefix_branch_pilot` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-21-earliest-shared-prefix-branch-pilot/results.md` | complete |
| `run_fixed_prefix_nonboundary_box_grammar_image19432` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-17-object-specific-geometry-transport-and-cross-row-influence-horizon/results.md` | complete |
| `run_historical_random_sorted_image2299_screen` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-18-historical-random-versus-geometry-sorted-image2299-screen/results.md`; `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-18-person25-dominant-owner-commit-and-persistence-closeout/unit.md` | complete |
| `run_human13_rp_crossover_parity_v5` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-14-human13-k-trajectory-rp-crossover-screen/unit.md` | complete |
| `run_native_commit_redistribution` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-15-homogeneous-mixed-batch-coordinate-logit-invariance/unit.md` | complete |
| `run_native_sibling_branch_replay` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-18-image2299-near-complete-human-relabel-successor-transition/unit.md` | complete |
| `run_person25_commit_closeout` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-18-person25-dominant-owner-commit-and-persistence-closeout/results.md` | complete |
| `run_sampled_rescue_transition` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-16-human-audited-rare-object-trajectory-genealogy/unit.md`; `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-17-native-sibling-row-branch-value-and-commit-crossover/unit.md` | complete, redirected |
| `run_span_likelihood_replay` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-29-sampled-span-likelihood-and-consensus-union-filtering/unit.md` | complete |
| `score_person25_y2_competition` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-18-person25-dominant-owner-commit-and-persistence-closeout/results.md` | complete |
| `score_sorted_crossing_neutral_row_control` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-04-sorted-crossing-matched-length-neutral-row-insertion-control/unit.md` | complete |
| `score_sorted_owner_accessibility_census_shard` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-03-sorted-owner-accessibility-phenotype-census/tasks.md`; `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-03-sorted-owner-accessibility-phenotype-census/unit.md` | complete |
| `summarize_same_covered_set_prefix_order_probe` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-19-common-object-prefix-permutation-short-horizon/results.md`; `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-19-same-covered-set-prefix-order-equivalence/results.md` | complete |
| `verify_likelihood_mining_contracts` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-29-sampled-span-likelihood-and-consensus-union-filtering/unit.md` | complete |
| `visualize_sorted_owner_accessibility_visual_atlas` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-03-sorted-owner-accessibility-phenotype-census/tasks.md`; `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-03-sorted-owner-accessibility-phenotype-census/unit.md` | complete |

33 of the 42 RECORD_CITED scripts are idle in this sense. The rest are cited by units whose
status is `blocked`, `needs_revision`, or by `memories/notes/*` with no unit frontmatter, and are not
candidates at all.

### HOLD-C. TEST_ONLY blocked by a shared test module

Deleting these would require surgically removing test functions from a module that also covers a KEPT
script, which trades a small deletion for a real loss of coverage on a surviving mechanic. Per the
write-surface rule that path is available, but the evidence does not justify it: not taken.

| Script | Test module also covering | Kept script(s) that module loads |
| --- | --- | --- |
| `analyze_iterative_forced_continue_extreme_capacity` | `tests/research/test_iterative_forced_continue_exact_native.py` | `run_local_branch_causal_value` |
| `analyze_sorted_crossing_boundary_owner_release_secondary` | `tests/research/test_analyze_sorted_crossing_boundary_owner_release_secondary.py` | `analyze_sorted_crossing_boundary_owner_release`, `merge_sorted_crossing_boundary_owner_release`, `merge_sorted_crossing_boundary_owner_release_secondary`, `score_sorted_crossing_boundary_owner_release` ... |
| `analyze_sorted_fn_mechanisms` | `tests/research/test_analyze_sorted_fn_mechanisms.py` | `merge_sorted_fn_successor_score_shards`, `run_sorted_fn_successor_behavior` |
| `analyze_sorted_fn_mechanisms` | `tests/research/test_run_sorted_fn_successor_behavior.py` | `merge_sorted_fn_successor_score_shards`, `run_same_covered_set_prefix_order_probe`, `run_sorted_fn_successor_behavior` |
| `analyze_sorted_full_canvas_token_budget_intervention` | `tests/research/test_analyze_sorted_full_canvas_token_budget_intervention.py` | `build_sorted_owner_accessibility_census_plan`, `merge_sorted_owner_accessibility_census_shards`, `score_sorted_owner_accessibility_census_shard`, `score_sorted_owner_basin_landscape` |
| `analyze_sorted_image2299_owner_accessibility` | `tests/research/test_analyze_sorted_image2299_owner_accessibility.py` | `build_sorted_owner_accessibility_census_plan`, `score_sorted_owner_accessibility_census_shard` |
| `analyze_sorted_image2299_owner_accessibility` | `tests/research/test_analyze_sorted_image2299_supported_fn_reachability.py` | `build_sorted_owner_accessibility_census_plan` |
| `analyze_sorted_image2299_owner_accessibility` | `tests/research/test_visualize_sorted_image2299_mechanism_atlas.py` | `build_sorted_owner_accessibility_census_plan` |
| `analyze_sorted_image2299_supported_fn_reachability` | `tests/research/test_analyze_sorted_image2299_supported_fn_reachability.py` | `build_sorted_owner_accessibility_census_plan` |
| `analyze_sorted_image2299_supported_fn_reachability` | `tests/research/test_visualize_sorted_image2299_mechanism_atlas.py` | `build_sorted_owner_accessibility_census_plan` |
| `attest_sorted_fn_successor_score_run` | `tests/research/test_analyze_sorted_fn_mechanisms.py` | `merge_sorted_fn_successor_score_shards`, `run_sorted_fn_successor_behavior` |
| `attest_sorted_fn_successor_score_run` | `tests/research/test_attest_sorted_fn_successor_score_run.py` | `build_sorted_fn_mechanism_registry`, `build_sorted_owner_basin_candidates`, `merge_sorted_fn_successor_score_shards`, `prepare_sorted_fn_successor_inputs` ... |
| `attest_sorted_fn_successor_score_run` | `tests/research/test_run_sorted_fn_successor_behavior.py` | `merge_sorted_fn_successor_score_shards`, `run_same_covered_set_prefix_order_probe`, `run_sorted_fn_successor_behavior` |
| `attest_sorted_owner_basin_smoke` | `tests/research/test_attest_sorted_owner_basin_smoke.py` | `prepare_sorted_owner_basin_inputs`, `score_sorted_owner_basin_landscape`, `sorted_owner_basin_landscape`, `summarize_sorted_owner_basin_landscape` |
| `audit_sorted_image2299_calibration_transfer` | `tests/research/test_audit_sorted_image2299_calibration_transfer.py` | `build_sorted_owner_accessibility_census_plan` |
| `build_image2299_current_native_ledger` | `tests/research/test_build_image2299_current_native_ledger.py` | `build_sorted_owner_accessibility_census_plan`, `build_sorted_owner_basin_census` |
| `build_sorted_image2299_native_ledger` | `tests/research/test_visualize_sorted_image2299_mechanism_atlas.py` | `build_sorted_owner_accessibility_census_plan` |
| `build_sorted_image2299_owner_accessibility_plan` | `tests/research/test_analyze_sorted_image2299_owner_accessibility.py` | `build_sorted_owner_accessibility_census_plan`, `score_sorted_owner_accessibility_census_shard` |
| `build_sorted_image2299_owner_accessibility_plan` | `tests/research/test_build_sorted_image2299_owner_accessibility_plan.py` | `build_sorted_owner_accessibility_census_plan`, `score_sorted_owner_accessibility_census_shard` |
| `convert_image2299_case_v1_to_v2` | `tests/research/test_convert_image2299_case_v1_to_v2.py` | `run_same_covered_set_prefix_order_probe` |
| `human13_row_contrast_live` | `tests/research/test_human13_row_contrast_live.py` | `build_human13_row_contrast_successor` |
| `materialize_human13_row_contrast_successor` | `tests/research/test_train_human13_row_contrast_successor.py` | `build_human13_row_contrast_successor` |
| `prepare_sorted_full_canvas_token_budget_intervention` | `tests/research/test_analyze_sorted_full_canvas_token_budget_intervention.py` | `build_sorted_owner_accessibility_census_plan`, `merge_sorted_owner_accessibility_census_shards`, `score_sorted_owner_accessibility_census_shard`, `score_sorted_owner_basin_landscape` |
| `prepare_sorted_full_canvas_token_budget_intervention` | `tests/research/test_prepare_sorted_full_canvas_token_budget_intervention.py` | `build_sorted_owner_accessibility_census_plan` |
| `prepare_sorted_full_canvas_token_budget_intervention` | `tests/research/test_score_sorted_full_canvas_token_budget_intervention_shard.py` | `build_sorted_owner_accessibility_census_plan`, `score_sorted_owner_accessibility_census_shard` |
| `run_continuation_locality_boundary_scoring` | `tests/research/test_continuation_locality_owner_compositionality.py` | `materialize_continuation_locality_owner_compositionality`, `run_complete_candidate_row_scoring`, `run_exact_prefix_owner_compositionality`, `run_next_row_likelihood_change` |
| `run_iterative_forced_continue_exact_native` | `tests/research/test_iterative_forced_continue_exact_native.py` | `run_local_branch_causal_value` |
| `run_physical_owner_duplication_prefix_counterfactual` | `tests/research/test_run_physical_owner_duplication_prefix_counterfactual.py` | `run_complete_candidate_row_scoring` |
| `run_static_dynamic_owner_observational_census` | `tests/research/test_run_static_dynamic_owner_observational_census.py` | `run_static_dynamic_owner_interface_experiment`, `run_static_post_llm_image_field_probe` |
| `score_sorted_full_canvas_token_budget_intervention_shard` | `tests/research/test_analyze_sorted_full_canvas_token_budget_intervention.py` | `build_sorted_owner_accessibility_census_plan`, `merge_sorted_owner_accessibility_census_shards`, `score_sorted_owner_accessibility_census_shard`, `score_sorted_owner_basin_landscape` |
| `score_sorted_full_canvas_token_budget_intervention_shard` | `tests/research/test_score_sorted_full_canvas_token_budget_intervention_shard.py` | `build_sorted_owner_accessibility_census_plan`, `score_sorted_owner_accessibility_census_shard` |
| `seal_s_k10_h20_crossover_finalization_receipt` | `tests/research/test_finalize_s_k10_h20_crossover.py` | `finalize_s_k10_h20_crossover`, `natural_boundary_attention_actuators` |
| `seal_s_k10_h20_crossover_finalization_receipt` | `tests/research/test_seal_s_k10_h20_crossover_finalization_receipt.py` | `finalize_s_k10_h20_crossover` |
| `split_label_only_candidate_pool` | `tests/research/test_validate_constant_dose_trajectory_panel_union.py` | `validate_constant_dose_trajectory_panel_union` |
| `summarize_continuation_locality_owner_compositionality` | `tests/research/test_continuation_locality_owner_compositionality.py` | `materialize_continuation_locality_owner_compositionality`, `run_complete_candidate_row_scoring`, `run_exact_prefix_owner_compositionality`, `run_next_row_likelihood_change` |
| `train_human13_row_contrast_successor` | `tests/research/test_train_human13_row_contrast_successor.py` | `build_human13_row_contrast_successor` |
| `validate_same_parent_complete_row_intervention_union` | `tests/research/test_run_same_parent_complete_row_intervention.py` | `run_same_parent_complete_row_intervention` |
| `validate_sampled_history_target_reachability_union` | `tests/research/test_run_sampled_history_target_reachability.py` | `run_sampled_history_target_reachability` |
| `visualize_sorted_crossing_boundary_owner_release` | `tests/research/test_visualize_sorted_crossing_boundary_owner_release.py` | `analyze_sorted_crossing_boundary_owner_release`, `merge_sorted_crossing_boundary_owner_release` |
| `visualize_sorted_crossing_owner_row_geometry` | `tests/research/test_visualize_sorted_crossing_owner_row_geometry.py` | `analyze_sorted_crossing_owner_row_geometry` |
| `visualize_sorted_image2299_mechanism_atlas` | `tests/research/test_visualize_sorted_image2299_mechanism_atlas.py` | `build_sorted_owner_accessibility_census_plan` |
| `visualize_sorted_supported_fn_native_prefix_reachability_prevalence` | `tests/research/test_visualize_sorted_supported_fn_native_prefix_reachability_prevalence.py` | `analyze_sorted_supported_fn_native_prefix_reachability_prevalence`, `build_sorted_owner_accessibility_census_plan`, `merge_sorted_owner_accessibility_census_shards` |

### HOLD-D. Prior-decision HOLD

| Script | Class here | Blocking record |
| --- | --- | --- |
| `analyze_native_sibling_branch_value` | ZERO (only an (e) hit) | `openspec/changes/archive/2026-08-28-establish-research-probes-baseline-v1/design.md`:193 - entropy ledger row `HOLD`, "Retain a replacement capable of replaying every documented result artifact." No new evidence beats that rationale, so it is excluded from batch 1. |

### HOLD-E. Reverted during batch 2 - candidate proved load-bearing

Found by the task-5.5 residue search and by an added check that greps each candidate's own
`"<name>.v<N>"` schema literals across `research/` and `memories/`. Both catch bindings the stem scan
cannot see: a record that cites the *test module* name (`test_<stem>.py` is one identifier token), and a
record artifact bound to a producer by schema id rather than by script name. Each was reverted with
`git checkout HEAD -- <path>` before the batch was committed; none reached a commit.

| Script (and its test) | Why it is load-bearing |
| --- | --- |
| `admit_native_sibling_first_rows` + `tests/analysis/test_native_sibling_first_row_owner_admission.py` | `scripts/research/analyze_native_sibling_branch_value.py`:39 imports it - and that script is itself HOLD-D, so the import must keep resolving. |
| `analyze_image2299_horizon_branch_value` + `tests/analysis/test_analyze_image2299_horizon_branch_value.py` | `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-18-human-resolved-dense-branch-value-and-calibration-screen/results.md`:134 names `tests/analysis/test_analyze_image2299_horizon_branch_value.py` as the verification command of a recorded result. Record-cited through the test module name, which the stem scan tokenises as one word and therefore missed. |
| `build_exact_trace_coordinate_state_bank` + `tests/research/test_build_exact_trace_coordinate_state_bank.py` | sole producer of `exact_trace_coordinate_visual_review.v1`, which appears in `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-20-own-prefix-entity-transition-and-coordinate-boundary-calibration-training-screen/smoke-b-v1/coordinate-review-decisions.json`. Producer binding by schema id, not by name. |
| `build_natural_boundary_owner_admission_census` + `tests/research/test_build_natural_boundary_owner_admission_census.py` | `scripts/research/materialize_natural_boundary_pre_gpu_evidence.py`:35 (kept) binds `tests/research/test_build_natural_boundary_owner_admission_census.py` in `FOCUSED_TEST_PATHS`; deleting the test breaks a kept script's evidence contract. |
| `run_human_refined_completion_causal_micro_panel` + `tests/analysis/test_human_refined_completion_causal_micro_panel.py` | sole producer of `human_refined_completion_causal_micro_panel.v1`, carried by 7 artifacts under `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-22-human-refined-greedy-set-completion-conditions/` (`status: complete`, `evidence_status: verified`). |
| `run_human_refined_greedy_set_completion_conditions` + `tests/analysis/test_human_refined_greedy_set_completion_conditions.py` | imported by `run_human_refined_completion_causal_micro_panel`, which this pass reverted to KEEP; transitive keep. |
| `score_fixed_prompt_coordinate_branches` + `tests/research/test_score_fixed_prompt_coordinate_branches.py` | imported by `score_native_coordinate_branch_pairs`, which this pass reverted to KEEP; transitive keep. |
| `score_native_coordinate_branch_pairs` + `tests/research/test_score_native_coordinate_branch_pairs.py` | sole producer of `native_coordinate_branch_pairs.v1`, which appears in `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-18-matched-objective-coordinate-branch-signature-comparison/pure-cross-entropy-native-branch-cases.json`. |

### HOLD-F. Half-deleted mechanism pair, flagged for the user

`run_iterative_forced_continue_extreme_capacity` (batch 2, deleted) is the producer for
`analyze_iterative_forced_continue_extreme_capacity` (HOLD-C, kept only because
`tests/research/test_iterative_forced_continue_exact_native.py` also loads the kept
`run_local_branch_causal_value`). Both halves classify as entropy on their own evidence; only the
shared test module keeps the analyzer. Its record,
`research/2026-07-29-iterative-forced-continue-extreme-capacity.md`, was moved to `docs/history/` by
wave 4 of this change, so the pair has no live record consumer. The analyzer half is a clean follow-up
deletion once its test module is split.

## Configs and fixtures

The batch-1 and batch-2 scripts reference only two `configs/` *directories*
(`configs/coordexp_swift/infer/`, `configs/coordexp_swift/infer/research/`) as generation targets, never a
tracked config file. No tracked `configs/**` entry is cited by a deleted script alone, so **no config is
removed** by this change.

## Verification

Baseline: `receipts/test-baseline.md`, 88 failing node ids, `88 failed, 5129 passed, 2 skipped`,
`research-probes` @ `57d988351`. Rule (D10): compare failure **sets**, never counts.

### Batch 1 (ZERO + SR_ONLY): 37 scripts, 0 tests, 0 configs

- Narrow: `tests/artifacts/test_research_probe_admission.py tests/research/test_research_probe_admission_consumers.py tests/research/test_capture_natural_boundary_support_source_bindings.py` -> **51 passed**.
- Broad: `CUDA_VISIBLE_DEVICES=-1 conda run -n ms pytest tests/research tests/artifacts -q -p no:cacheprovider --no-header -rfE` -> `88 failed, 5129 passed, 2 skipped in 721.48s`.
- Failure-set diff vs baseline: **0 new, 0 gone**. Pass count identical, as expected: batch 1 removes no test.
- Reverts: none. Commit `1802262a2`.

### Batch 2 (TEST_ONLY, dedicated test module): 49 scripts, 48 tests, 0 configs

- `pytest tests --collect-only`: `6504 tests collected, 53 errors`. The 53 are the pre-existing
  `ModuleNotFoundError: No module named 'src.analysis.{candidate_field_cardinality_tomography,
  policy_objective_mechanism_comparison, post_x1_instance_basin_tomography,
  prefix_state_transition_tomography, sorted_random_no_newline_phenotype}'` collection errors, unchanged in
  count and identity by this change and unrelated to `scripts/research`.
- Narrow: the same admission trio -> **51 passed**.
- Broad: `85 failed, 4815 passed, 2 skipped in 689.14s`.
- Failure-set diff vs baseline: **0 new**; 3 gone, all three inside the deleted module
  `tests/research/test_run_image_12576_row_mediation_crossover.py`
  (`test_load_stage_seven_source_reconstructs_four_exact_prefixes`,
  `test_load_stage_seven_source_refuses_changed_prefix_hash`,
  `test_validate_endpoint_parity_refuses_owner_only_match`). The 314 pass-count drop is the 48 removed
  modules.
- Out-of-baseline neighbours (the 7 deleted modules outside `tests/research`/`tests/artifacts`):
  `pytest tests/analysis` (minus the 5 pre-broken packages) -> `3 failed, 91 passed`. All three failures are
  `ImportError: cannot import name '_normalized_attested_model_identity_for_runtime_comparison' /
  'canonical_float32_logprob' from 'src.inference.backend'` in
  `tests/analysis/test_assemble_source_preservation_multi_route_state_banks.py` and
  `tests/analysis/test_sampled_rescue_transition.py`, whose subject scripts are both KEPT and neither of
  which mentions any deleted stem. Pre-existing, unrelated to this change.
- Reverts: 8 scripts + 8 test modules, all before the commit; see HOLD-E.

## Residue search (task 5.5)

For every deleted basename (and the derived `test_<stem>` form), `rg --fixed-strings` across the whole
`research-probes` tree and across `/data/CoordExp/.worktrees/image2299-mechanism-microscope`
(tracked + untracked, minus `.git/` and `outputs/`): **217 hits, all accounted for**:

| Where | Meaning |
| --- | --- |
| this ledger | the evidence records themselves |
| `openspec/changes/archive/2026-08-06-add-hf-exact-history-evidence-seam/design.md`:66 | (e) historical mention of `run_paired_terminal_forced_opener_release.py` |
| `image2299-mechanism-microscope/...` | that fork's own byte-identical copies of the files deleted here, plus its mirror of the same archived design doc; the fork is independent and unaffected |
| `scripts/research/analyze_iterative_forced_continue_extreme_capacity.py`, `tests/research/test_iterative_forced_continue_exact_native.py` | substring collisions of the derived stem `iterative_forced_continue_extreme_capacity` with a **kept** module of a similar name; see HOLD-F |

No surviving file under `src/`, `scripts/`, `tests/`, `configs/`, `research/`, `memories/`, or current `docs/`
references a deleted module. `git diff --check` and `git diff --cached --check`: clean.

## Net reduction

`git diff --stat research-base-v2..HEAD -- scripts tests configs | tail -1`:

```
 134 files changed, 72807 deletions(-)
```

No other lane in this change touched `scripts/`, `tests/`, or `configs/`, so that range is exactly the two
batches below.

| Surface | Before | After | Removed |
| --- | ---: | ---: | ---: |
| `scripts/research/*.py` | 315 | 229 | **86** (37 batch 1 + 49 batch 2) |
| test modules under `tests/**` | 413 | 365 | **48** |
| `configs/**` entries | 564 | 564 | 0 |
| lines | - | - | **72,807** |
| concepts | - | - | 86 probe entrypoints and their 48 dedicated contract suites no longer have to stay coherent with the admission owner, the artifact roots, or each other |

Insertions: 0 outside this receipt. No surviving script was edited: every deletion was end-to-end, so no
dead import had to be removed.

## Kept on purpose

- 125 SHARED and 42 RECORD_CITED scripts: real production/live-direction consumers or a research unit that
  names them as replay entry or evidence producer.
- 22 DOCS_CITED, 31 TEST_ONLY blocked by a shared test module, 8 reverted load-bearing candidates, and
  `analyze_native_sibling_branch_value` (prior-change HOLD): reported above, untouched, user's decision.
- 33 of the 42 RECORD_CITED scripts are cited only by `status: complete` units (HOLD-B). They are the
  largest remaining reclaimable block and the cheapest next step: rewriting those citations to point at
  `research-base-v2` would make them deletable without losing replay.

## Recovery

Every file removed by this change is present in the annotated tag `research-base-v2`
(tag object `960c627c6`, peeled commit `8dac2d041`):

```
git checkout research-base-v2 -- scripts/research/<name>.py
```

## Wave 8 (8.1) - approved follow-through on HOLD-A/B/C/F

User approval 2026-08-28 ("完全同意. test, scripts 都可以大改动"). Lane worktree
`/data/CoordExp/.worktrees/lane-scripts`, branch `lane/scripts`, forked from `research-probes`
HEAD `b3d3be9b9`. Method unchanged from waves 5/1-2; two rules were added and are the reason the
deleted counts are lower than the HOLD lists:

1. **Transitivity as a fixpoint.** A candidate imported by a *surviving* script is kept, and a
   candidate that is itself kept then blocks its own imports. Iterating to a fixpoint over the
   86 HOLD-A/B/C candidates leaves 59 deletable and 27 blocked. The single-pass form used while
   planning would have deleted `human13_live_payload` and `human13_live_segments`, both imported
   by `train_human13_live_arm` (HOLD-A, itself kept because `human13_on_policy_runtime` imports it).
2. **A deleted script that a surviving-subject test depends on is kept.** The split rule keeps every
   test whose subject survives; when such a test cannot run without the candidate (directly or
   through a module-level fixture), deleting the candidate would delete the module rather than
   split it. Two candidates were kept this way (`train_human13_k_trajectory_rp_crossover`,
   `split_label_only_candidate_pool`).

### Batch 1 - HOLD-B (commit `06f34e736`)

19 scripts, 10 test modules, 1 config, 12 record citation rewrites.

| Deleted script | Dedicated test module removed |
| --- | --- |
| `analyze_cluster_confidence_retention` | - |
| `analyze_common_owner_count_robustness` | - |
| `analyze_ranking_versus_operating_point` | - |
| `analyze_sorted_crossing_neutral_row_control` | `tests/research/test_analyze_sorted_crossing_neutral_row_control.py` |
| `build_cluster_confidence` | - |
| `build_common_object_prefix_permutation_cases` | `tests/research/test_build_common_object_prefix_permutation_cases.py` |
| `build_matched_random_sorted_candidate_score_manifest` | `tests/research/test_build_matched_random_sorted_candidate_score_manifest.py` |
| `build_sorted_prospective_13_image_panel` | `tests/research/test_build_sorted_prospective_13_image_panel.py` |
| `compute_sampled_union_f1_metrics` | - |
| `derive_likelihood_mining_baseline` | - |
| `merge_sorted_crossing_neutral_row_control` | `tests/research/test_merge_sorted_crossing_neutral_row_control.py` |
| `prepare_sorted_crossing_neutral_row_control` | `tests/research/test_prepare_sorted_crossing_neutral_row_control.py` |
| `run_earliest_shared_prefix_branch_pilot` | `tests/research/test_earliest_shared_prefix_branch_pilot.py` |
| `run_fixed_prefix_nonboundary_box_grammar_image19432` | - |
| `run_human13_rp_crossover_parity_v5` | `tests/research/test_run_human13_rp_crossover_parity_v5.py` |
| `run_span_likelihood_replay` | - |
| `score_sorted_crossing_neutral_row_control` | `tests/research/test_score_sorted_crossing_neutral_row_control.py` |
| `summarize_same_covered_set_prefix_order_probe` | `tests/research/test_summarize_same_covered_set_prefix_order_probe.py` |
| `verify_likelihood_mining_contracts` | - |

Config removed: `configs/coordexp_swift/infer/qwen3_vl_2b_desc_first_random_permutation_bundle_step4887_human_refined12_hf_fp32.yaml`
(referenced only by `run_span_likelihood_replay` and `verify_likelihood_mining_contracts`).

### Batch 2 - HOLD-C (commit `20be41cfa`)

31 scripts (30 HOLD-C + `visualize_sorted_owner_accessibility_visual_atlas`, a HOLD-B entry
deferred out of batch 1 because `visualize_sorted_image2299_mechanism_atlas` imported it),
26 whole test modules, 6 shared modules split, 2 record citation rewrites.

| Deleted script |
| --- |
| `analyze_iterative_forced_continue_extreme_capacity` |
| `analyze_sorted_crossing_boundary_owner_release_secondary` |
| `analyze_sorted_fn_mechanisms` |
| `analyze_sorted_full_canvas_token_budget_intervention` |
| `analyze_sorted_image2299_owner_accessibility` |
| `analyze_sorted_image2299_supported_fn_reachability` |
| `attest_sorted_fn_successor_score_run` |
| `attest_sorted_owner_basin_smoke` |
| `audit_sorted_image2299_calibration_transfer` |
| `build_image2299_current_native_ledger` |
| `build_sorted_image2299_native_ledger` |
| `build_sorted_image2299_owner_accessibility_plan` |
| `convert_image2299_case_v1_to_v2` |
| `human13_row_contrast_live` |
| `materialize_human13_row_contrast_successor` |
| `prepare_sorted_full_canvas_token_budget_intervention` |
| `run_continuation_locality_boundary_scoring` |
| `run_iterative_forced_continue_exact_native` |
| `run_physical_owner_duplication_prefix_counterfactual` |
| `run_static_dynamic_owner_observational_census` |
| `score_sorted_full_canvas_token_budget_intervention_shard` |
| `seal_s_k10_h20_crossover_finalization_receipt` |
| `summarize_continuation_locality_owner_compositionality` |
| `train_human13_row_contrast_successor` |
| `validate_same_parent_complete_row_intervention_union` |
| `validate_sampled_history_target_reachability_union` |
| `visualize_sorted_crossing_boundary_owner_release` |
| `visualize_sorted_crossing_owner_row_geometry` |
| `visualize_sorted_image2299_mechanism_atlas` |
| `visualize_sorted_owner_accessibility_visual_atlas` |
| `visualize_sorted_supported_fn_native_prefix_reachability_prevalence` |

| Shared test module | Split |
| --- | --- |
| `tests/research/test_continuation_locality_owner_compositionality.py` | -2 tests (`test_binary_recovery_transitions_preserve_pairing`, `test_locality_and_owner_runners_use_the_same_stable_sharding`); 4 kept |
| `tests/research/test_finalize_s_k10_h20_crossover.py` | -5 tests + `_successor_fixture` + the two `REAL_SEALER*` constants (the whole "post-execution finalization successor" section); `test_finalize_without_a_successor_keeps_the_strict_parent_contract` kept, 52 node ids collect |
| `tests/research/test_run_same_parent_complete_row_intervention.py` | -1 test (`test_union_requires_exact_three_candidates`); 5 kept |
| `tests/research/test_run_sampled_history_target_reachability.py` | -4 `test_union_validator_*` tests; 14 kept |
| `tests/research/test_run_sorted_fn_successor_behavior.py` | -1 test (`test_v2_attestor_analyzer_admission_builds_behavior_contract_cpu_only`) + its `_analyze_v2_landscape_for_behavior` fixture; 24 kept |
| `tests/research/test_merge_sorted_crossing_boundary_owner_release_secondary.py` | docstring pointer to the deleted `test_analyze_..._secondary` module rewritten; no test removed |

HOLD-F resolved here: `analyze_iterative_forced_continue_extreme_capacity`, the surviving half of the
pair whose producer wave 5 batch 2 deleted, went with its shared module
`tests/research/test_iterative_forced_continue_exact_native.py`. The kept `run_local_branch_causal_value`
keeps its own dedicated module `tests/research/test_run_local_branch_causal_value.py`.

### Batch 3 - HOLD-A + HOLD-F (commit `b0b2763a5`)

5 scripts and their 5 dedicated test modules. All 22 HOLD-A scripts were re-verified against
current `docs/`, `progress/`, `memories/` and `research/`: every citation is a
`docs/superpowers/plans/**` plan, so the doc rule clears all 22 and no canonical or active doc cites
any of them. The 17 not deleted are held by the transitivity/test-dependency rules, not by a doc.

| Deleted script | Dedicated test module removed |
| --- | --- |
| `analyze_human13_k_trajectory_rp_crossover` | `tests/research/test_analyze_human13_k_trajectory_rp_crossover.py` |
| `build_human13_row_contrast_successor` | `tests/research/test_build_human13_row_contrast_successor.py` |
| `human13_gradient_preservation` | `tests/research/test_human13_gradient_preservation.py` |
| `launch_human13_k_union_matrix` | `tests/research/test_launch_human13_k_union_matrix.py` |
| `run_human13_live_census` | `tests/research/test_run_human13_live_census.py` |

### Citation rewrites (14 files, one `**Replay note.**` line each)

Text: "Producer scripts deleted from `research-probes` on 2026-08-28
(reclaim-research-probes-lifecycle); replay them from tag `research-base-v2`:
`git worktree add <tmp> research-base-v2`." Placed at the end of the unit's artifact/provenance
section where one exists, else at the end of the file. No result, number, or status field changed.

- `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-17-object-specific-geometry-transport-and-cross-row-influence-horizon/results.md`
- `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-19-common-object-prefix-permutation-short-horizon/results.md`
- `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-19-same-covered-set-prefix-order-equivalence/results.md`
- `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-20-matched-random-sorted-prefix-order-screen/results.md`
- `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-21-earliest-shared-prefix-branch-pilot/results.md`
- `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-29-likelihood-filter-robustness-and-panel-annotation-validity/unit.md`
- `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-29-ranking-quality-versus-usable-rejection/unit.md`
- `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-29-sampled-span-likelihood-and-consensus-union-filtering/unit.md`
- `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-03-sorted-owner-accessibility-phenotype-census/tasks.md`
- `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-03-sorted-owner-accessibility-phenotype-census/unit.md`
- `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-04-sorted-crossing-matched-length-neutral-row-insertion-control/unit.md`
- `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-04-sorted-prospective-13-image-panel-admission/tasks.md`
- `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-04-sorted-prospective-13-image-panel-admission/unit.md`
- `research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-14-human13-k-trajectory-rp-crossover-screen/unit.md`

`research/investigations/qwen3-vl-dense-enumeration/experiments/2026-07-20-matched-random-sorted-prefix-order-screen/candidate-row-scoring-manifest.json`
is the only citation of `build_matched_random_sorted_candidate_score_manifest`, but it is a produced
result artifact carrying the producer name and hash, so it was left byte-unchanged and the replay note
went into that unit's `results.md` instead.

No `memories/**` file cites any script deleted here.

### Kept, with the blocking evidence

| Class | Total | Deleted | Kept |
| --- | ---: | ---: | ---: |
| HOLD-A | 22 | 5 | 17 |
| HOLD-B | 33 | 20 | 13 |
| HOLD-C | 31 | 30 | 1 |
| **total** | **86** | **55** | **31** |

**HOLD-A-keep (17)** - every one is cited only by `docs/superpowers/plans/**`, so all are keeps by
transitivity, never by a doc:

| Kept script | Blocked by |
| --- | --- |
| `build_human13_on_policy_frontier` | `human13_continuation_projection`, `human13_greedy_compiler`, `human13_hf_native_one_image_owner`, `human13_live_eval`, `human13_on_policy_runtime`, `human13_rp_crossover_production_backend` (import) |
| `collect_human13_discovery` | `human13_live_eval`, `human13_live_model`, `human13_rp_crossover_production_backend`, `materialize_human13_no_update_census` (import) |
| `collect_human13_rp_crossover` | `human13_rp_crossover_live_composition`, `human13_rp_crossover_matrix_contracts`, `human13_rp_crossover_production_backend`, `human13_trajectory_credit` (import) |
| `compare_clean_rollout_owner_coverage` | `analyze_human13_k_union`, `build_human13_k_union_manifest`, `human13_continuation_projection`, `human13_trajectory_credit` (import) |
| `human13_frontier_selection` | `human13_continuation_projection`, `human13_on_policy_runtime` (import) |
| `human13_k_trajectory_contracts` | `human13_trajectory_credit` (import) |
| `human13_live_census` | `human13_greedy_compiler`, `human13_rp_crossover_production_backend` (import) |
| `human13_live_payload` | `train_human13_live_arm` (import; fixpoint - `train_human13_live_arm` is itself a kept HOLD-A entry) |
| `human13_live_segments` | `train_human13_live_arm` (import; fixpoint) |
| `human13_on_policy_live` | `human13_on_policy_runtime` (import) |
| `human13_on_policy_scoring` | `human13_greedy_compiler`, `human13_on_policy_runtime` (import) |
| `human13_rp_policy` | `human13_trajectory_credit` (import) |
| `launch_human13_k_trajectory_rp_crossover` | `human13_rp_crossover_production_backend` (import) |
| `materialize_human13_k_union_configs` | `execute_human13_k_union`, `human13_live_model`, `human13_rp_crossover_production_backend` (import) |
| `train_human13_k_trajectory_rp_crossover` | surviving-subject tests: 10 of 28 in `test_launch_human13_k_trajectory_rp_crossover.py` and 3 of 16 in `test_human13_rp_crossover_production.py` drive the launcher/production factory through this runner |
| `train_human13_live_arm` | `human13_on_policy_runtime` (import) |
| `train_human13_on_policy_successor` | `human13_on_policy_runtime` (import) |

**HOLD-B-keep (13)**:

| Kept script | Blocked by |
| --- | --- |
| `analyze_image2299_near_complete_relabel_successor_transition` | `analyze_image2299_horizon_branch_value` (import) |
| `assemble_positive_path_imitation_state_bank` | `analyze_trajectory_owner_set_admission_census`, `assemble_constant_dose_breadth_state_banks`, `assemble_row_local_owner_stop_state_bank`, `assemble_source_preservation_multi_route_state_banks` (import) |
| `build_sorted_owner_accessibility_census_plan` | `analyze_sorted_supported_fn_native_prefix_reachability_prevalence`, `prepare_sorted_crossing_boundary_owner_release_realization`, `run_static_dynamic_owner_support_probe`, `score_sorted_crossing_boundary_owner_release` (import) |
| `human13_rp_crossover_live_packs` | `human13_greedy_compiler`, `human13_rp_crossover_live_composition`, `human13_rp_crossover_production_backend` (import) |
| `merge_sorted_owner_accessibility_census_shards` | `analyze_sorted_supported_fn_native_prefix_reachability_prevalence`, `prepare_sorted_crossing_boundary_owner_release_realization`, `score_sorted_crossing_boundary_owner_release` (import) |
| `run_current_seeded_sampled_rollouts` | `build_inference_coordinate_boundary_state_bank`, `collect_vllm_trajectory_panel`, `human13_hf_census`, `human13_live_eval`, `human13_rp_crossover_production_backend`, `validate_constant_dose_trajectory_panel_union` (import) |
| `run_historical_random_sorted_image2299_screen` | deletable, DEFERRED: its dedicated module is `tests/analysis/test_historical_random_sorted_image2299_screen.py`, outside this lane's write surface |
| `run_native_commit_redistribution` | `run_batch_coordinate_logit_invariance` (import) |
| `run_native_sibling_branch_replay` | `analyze_native_sibling_branch_value` (HOLD-D), `run_complete_candidate_row_scoring`, `run_next_row_likelihood_change` (import) |
| `run_person25_commit_closeout` | deletable, DEFERRED: `tests/analysis/test_person25_commit_closeout.py`, outside this lane's write surface |
| `run_sampled_rescue_transition` | `run_batch_coordinate_logit_invariance`, `run_fixed_encoding_downstream_residual_state_portability`, `run_fixed_encoding_object_centered_spatial_eligibility_crossover`, `run_fixed_encoding_query_scoped_object_centered_spatial_eligibility` (import); also task 8.2 lane work |
| `score_person25_y2_competition` | deletable, DEFERRED: `tests/analysis/test_person25_y2_competition.py`, outside this lane's write surface |
| `score_sorted_owner_accessibility_census_shard` | `score_sorted_crossing_boundary_owner_release` (import) |

**HOLD-C-keep (1)**: `split_label_only_candidate_pool` - its `_inputs` fixture builds the input set for
all 10 tests of the surviving `validate_constant_dose_trajectory_panel_union`, so removing it would
delete `tests/research/test_validate_constant_dose_trajectory_panel_union.py` rather than split it.
It and `tests/research/test_split_label_only_candidate_pool.py` were restored with
`git checkout HEAD -- <path>` before batch 2 was committed; neither reached a commit in deleted form.

The three DEFERRED HOLD-B scripts above are the only reclaimable residue this lane leaves: they are
deletable on every rule, and the `tests` lane (task 8.2) or the lead can take them together with their
`tests/analysis/` modules.

### Verification

Two path-dependent facts about running in a lane worktree, both recorded rather than fixed:

- `tests/research/test_research_probe_admission_consumers.py` raises
  `ConsumerAdmissionError: execution root is not an approved worktree` at *import* time, because
  `scripts/research/research_probe_admission_consumers.py`:160 binds the compatibility receipt pair to
  `/data/CoordExp/.worktrees/research-probes` or `/data/CoordExp/outputs/research-probe-infras` only
  (commit `74609d2b1`). It is a whole-module collection error in `lane-scripts` independent of any
  deletion here, contributes no baseline failure, and was excluded with `--ignore` from every broad run.
  It must be re-run by the lead in `research-probes` after the merge.
- `tests/research/test_prepare_sorted_fn_successor_inputs.py::test_real_gt7511_17_near_gt_micro_local_index_0_is_the_exact_singleton`
  fails with `FileNotFoundError` on
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-01-sorted-owner-basin-task4-control-input-plan-final`,
  an artifact root the concurrent task-8.4 reclaim removed mid-run. Its subject
  `prepare_sorted_fn_successor_inputs` is a kept script and is untouched by this lane; nothing was
  restored. It is out of baseline in all three batches and is an artifact-state failure, not a code one.

| Batch | Admission | `tests/research` collect | Broad `tests/research tests/artifacts` | New vs baseline | Gone vs baseline |
| --- | --- | ---: | --- | ---: | ---: |
| 1 | 36 passed | - | `86 failed, 4564 passed, 3 skipped` | 1 (the 8.4 artifact root) | 3 |
| 2 | 36 passed | 3965, 0 errors | `83 failed, 3988 passed, 3 skipped` | 1 (same) | 6 |
| 3 | 36 passed | 3911, 0 errors | `83 failed, 3934 passed, 3 skipped` | 1 (same) | 6 |

Final failure-set diff vs `receipts/test-baseline.md` (88 node ids): **0 new attributable to this lane**,
**6 gone** -

| Gone node id | Why |
| --- | --- |
| `tests/research/test_run_image_12576_row_mediation_crossover.py::test_load_stage_seven_source_reconstructs_four_exact_prefixes` | module deleted by wave 5 batch 2, before this lane forked |
| `tests/research/test_run_image_12576_row_mediation_crossover.py::test_load_stage_seven_source_refuses_changed_prefix_hash` | same |
| `tests/research/test_run_image_12576_row_mediation_crossover.py::test_validate_endpoint_parity_refuses_owner_only_match` | same |
| `tests/research/test_build_sorted_image2299_native_ledger.py::test_exact_real_artifact_validate_only` | module deleted in batch 2 |
| `tests/research/test_build_sorted_image2299_native_ledger.py::test_receipt_is_self_sealed_and_task0_shapes_are_stable` | module deleted in batch 2 |
| `tests/research/test_run_sorted_fn_successor_behavior.py::test_v2_attestor_analyzer_admission_builds_behavior_contract_cpu_only` | test removed by the batch-2 split (its subject is the deleted attestor/analyzer) |

### Residue search

For each of the 55 deleted basenames and its derived `test_<stem>` form, `grep -rl --fixed-strings`
over `lane-scripts` and over `/data/CoordExp/.worktrees/image2299-mechanism-microscope`
(minus `.git/`, `outputs/`, `model_cache/`, and this ledger). Every hit is accounted for:

| Where | Meaning |
| --- | --- |
| 15 `research/**` records | the preserved citation plus its new replay note - the intended outcome |
| 5 `docs/superpowers/plans/**` | the HOLD-A plan citations; `docs/` is the docs lane's surface, so they are reported, not edited |
| `openspec/changes/archive/**` (7 files) | class (e) historical mentions, including four `source-bindings*.json` receipts of an archived change |
| `receipts/test-baseline.md`, `receipts/src-entropy-audit.md` | this change's own evidence records |
| `scripts/research/build_sorted_owner_basin_census.py`:9 | a docstring sentence naming `compute_sampled_union_f1_metrics.py`; a prose mention, not an import (the union it describes is still produced) |
| `image2299-mechanism-microscope` | 14 byte-identical mirrors (not counted, per the md5-mirror rule), 65 fork copies of files this repo no longer has, and 19 files that differ exactly by this wave's replay notes and test splits. The fork is independent and unaffected. |

No surviving file under `src/`, `scripts/`, `tests/`, or `configs/` imports or launches a deleted module.
`git diff --check` and `git diff --cached --check`: clean on all three commits.

### Net reduction

```
git diff --stat b3d3be9b9..HEAD -- scripts tests configs research | tail -1
 117 files changed, 29 insertions(+), 73969 deletions(-)
```

| Surface | Before wave 8 | After | Removed |
| --- | ---: | ---: | ---: |
| `scripts/research/*.py` | 229 | 174 | **55** |
| `tests/research/*.py` | 185 | 144 | **41** |
| `configs/**` | 564 | 563 | **1** |
| lines | - | - | **73,969** (29 inserted: the 14 replay notes) |

Cumulative over the change: `scripts/research` 315 -> 174 (**141 removed, 44.8%**);
`tests/research` 185 -> 144 with 48 more modules removed in wave 5 outside `tests/research`.

### Recovery

Every file removed by wave 8 is present in `research-base-v3` (and in `research-base-v2` for anything
that predates it):

```
git checkout research-base-v3 -- scripts/research/<name>.py
```
