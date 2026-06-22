---
doc_id: progress.diagnostics.index
layer: progress
doc_type: router
status: canonical
domain: research-history
summary: Router for failure investigations, mechanism studies, threshold sweeps, and operator notes that support diagnosis.
tags: [progress, diagnostics, investigations]
updated: 2026-06-15
---

# Diagnostics Index

Use this folder when the primary question is:

- what is going wrong?
- why is a behavior happening?
- which overlap / decoding threshold looks safe?
- which operator tool should I open to inspect a failure?

Prefer `progress/benchmarks/` when the output is mainly a measured run-vs-run or checkpoint-vs-checkpoint comparison.

## Diagnostic Clusters

- Active autoregressive duplication mechanism diagnosis
  - Start with
    [2026-06-12_autoregressive_duplication_causal_chain_synthesis.md](2026-06-12_autoregressive_duplication_causal_chain_synthesis.md)
    for the consolidated June 10-11 causal-chain read on residual localization,
    Layer 17/head-1 visual-basin routing, candidate-basin specificity, and x2
    overcorrection. This document supersedes the one-probe Phase 4 fragments
    from June 10-11.
  - Use
    [2026-06-12_fn_guidance_and_coord_basin_synthesis.md](2026-06-12_fn_guidance_and_coord_basin_synthesis.md)
    for the consolidated June 11-12 false-negative guidance branch, including
    prefix-state coordinate-basin lock, vase same-slot repair, hard book/chair
    context-band probes, and selector follow-ups.
  - Use
    [2026-06-12_pre_onset_duplication_precursor_synthesis.md](2026-06-12_pre_onset_duplication_precursor_synthesis.md)
    for the consolidated June 12 pre-onset branch, including coordinate-basin
    precursor drift, rank-moving residual selectors, matched-pair sign splits,
    and Layer 16/head-8 route-content evidence.
  - See
    [2026-06-12_diagnostics_consolidation_summary.md](2026-06-12_diagnostics_consolidation_summary.md)
    for the source-to-merged-document map and loss-check method for the June
    10-12 consolidation.
- Prefix-denoising branch launch-health notes
  - [2026-06-14_prefix_denoising_launch_health.md](2026-06-14_prefix_denoising_launch_health.md)
    records the original tiny launch-health smoke on `codex/prefix-denoising-sft`;
    it is superseded by the branch-isolation repair note for current branch
    interpretation.
  - [2026-06-15_prefix_denoising_branch_isolation_repair.md](2026-06-15_prefix_denoising_branch_isolation_repair.md)
    records the post-audit branch-isolation repair and focused verification on
    `codex/prefix-denoising-sft`. Treat it as branch provenance until the
    prefix-denoising code/config surface lands in main.
- Hard-CE coordinate-logit and token-embedding locality diagnostics
  - Start with
    [2026-05-18_hard_ce_coord_logit_embedding_locality.md](2026-05-18_hard_ce_coord_logit_embedding_locality.md)
    for the current `val200` mechanism read comparing ET-RMP-CE and
    random-shuffled pure hard-CE SFT at checkpoint `3664`, including
    coordinate-logit locality, self-prefix fragility, effective coordinate-row
    embedding geometry, and the narrowed SoftCE decision read.
  - Use
    [2026-05-18_gaussian_softce_a5_a6_coord_logit_locality.md](2026-05-18_gaussian_softce_a5_a6_coord_logit_locality.md)
    for the Gaussian SoftCE A5/A6 follow-up on checkpoint `3664`, including
    A5 Gaussian/default versus A6 CE-anchored mix-0.2 teacher-forced,
    self-prefix, embedding-locality, and guarded `val200` rollout evidence.
  - Use
    [2026-05-14_a5_a6_iou_gibbs_softce_negative_result.md](2026-05-14_a5_a6_iou_gibbs_softce_negative_result.md)
    for the earlier IoU/CIoU-Gibbs A5/A6 negative result and artifact-bundle
    routing. Keep this distinct from the later Gaussian SoftCE follow-up.
- Compact-full coord-confidence / stop-gate diagnostics
  - Start with
    [2026-05-08_compact_full_coord_confidence_stop_gate_diagnostics.md](2026-05-08_compact_full_coord_confidence_stop_gate_diagnostics.md)
    for the diagnostic-only root-cause read on low training loss but conservative
    compact-full decode, `coord_mean_logprob` as a plausible-object versus
    invalid/duplicate-tail separator, `rp=1.10` as the main decode surface,
    the HF batched compact-grammar prompt-offset bug under decoder-only left
    padding, the fixed val200 prompt-offset comparison, and the fixed-artifact
    coord-confidence / counterfactual boundary probe plan. This is preserved as
    mechanism evidence, not current stop-gate training guidance.
  - A2 EOS-Loosen Clean Ablation is concluded negative:
    [2026-05-14_a2_eos_loosen_ablation.md](2026-05-14_a2_eos_loosen_ablation.md).
    Keep it as diagnostic/historical evidence; do not route it as active
    training guidance or a current launch recommendation.
- Canonical ET-RMP continuation / repetition-penalty / FN diagnostics
  - Start with
    [2026-04-29_et_rmp_rp_continuation_bias_hypothesis.md](2026-04-29_et_rmp_rp_continuation_bias_hypothesis.md)
    for the pre-support-mass-enhancement ET-RMP objective context, `val200`
    and core-6 RP sweeps, fixed representative sample bank, FN latent probes,
    length/count close-pressure read, and hard stop-control ablation. Treat the
    stop-control material as historical diagnostics, not a stop-gate training
    recipe.
- Closed Qwen3-VL coord-token instance-binding mechanism study
  - Start with
    [2026-04-24_qwen3_vl_instance_binding_mechanism_findings.md](2026-04-24_qwen3_vl_instance_binding_mechanism_findings.md)
    for the fixed-checkpoint mechanism conclusion: partial pre-`x1` binding,
    late schema/pre-coordinate readout, and `x1/y1` as the hard commitment
    boundary.
- Active FN-rescue attention and next-row binding diagnostics
  - Start with
    [2026-06-02_fn_rescue_attention_binding_findings.md](2026-06-02_fn_rescue_attention_binding_findings.md)
    for the checkpoint-3664 full linked FN-rescue continuation study showing
    that many rollout false negatives are recoverable under desc-first
    continuation, correct `x1` is a strong instance-binding seed, wrong-control
    `x1` misdirects binding, same-desc competitors are the sharpest failure
    surface, and attention remains background/context-heavy rather than a clean
    target-object spotlight.
- Candidate-field, prefix-state, and post-X1 tomography diagnostics
  - Start with
    [2026-06-03_candidate_field_cardinality_representative8192_analysis.md](2026-06-03_candidate_field_cardinality_representative8192_analysis.md)
    for the representative8192 candidate-field cardinality analysis.
  - Use
    [2026-06-04_prefix_state_transition_phase_a3_1_launch.md](2026-06-04_prefix_state_transition_phase_a3_1_launch.md)
    and
    [2026-06-04_prefix_state_transition_phase_a3_1_analysis.md](2026-06-04_prefix_state_transition_phase_a3_1_analysis.md)
    for the A3.1 prefix-state transition launch and analysis notes.
  - Use
    [2026-06-04_a3_2_sorted_random_no_newline_smoke_findings.md](2026-06-04_a3_2_sorted_random_no_newline_smoke_findings.md)
    for A3.2 sorted-random no-newline smoke evidence.
  - Use
    [2026-06-05_a3_3_post_x1_instance_basin_real_tiny_smoke.md](2026-06-05_a3_3_post_x1_instance_basin_real_tiny_smoke.md)
    for the A3.3 post-X1 instance-basin real-tiny smoke note.
- Stage-2 residual-set refactor and launch-health notes
  - [2026-05-20_residual_set_stage2_smoke.md](2026-05-20_residual_set_stage2_smoke.md)
  - [2026-05-22_residual_set_refactor_preflight.md](2026-05-22_residual_set_refactor_preflight.md)
  - [2026-05-23_vllm_online_residual_trie_gate_and_prod_candidate.md](2026-05-23_vllm_online_residual_trie_gate_and_prod_candidate.md)
- Stage-2 birth-first Channel-B decision evidence
  - Start with
    [2026-04-22_stage2_birth_first_channel_b_decision_study.md](2026-04-22_stage2_birth_first_channel_b_decision_study.md)
    for the merged-vLLM operator fix, the paired small-fraction control versus
    birth-first result, and the recorded decision context.
- Active Stage-1 raw-text mechanism and coordinate-family investigations
  - Start here for the current raw-text mechanism read:
    [2026-04-21_raw_text_coordinate_mechanism_findings.md](2026-04-21_raw_text_coordinate_mechanism_findings.md)
  - Use
    [2026-04-22_raw_text_decode_bias_mechanism_findings.md](2026-04-22_raw_text_decode_bias_mechanism_findings.md)
    for the decode-time EOS / repeat-penalty / branchpoint follow-up on the
    same raw-text-only checkpoint pair.
  - Use [2026-04-20_coord_family_basin_and_recall_comparison.md](2026-04-20_coord_family_basin_and_recall_comparison.md)
    for the cross-family comparison.
  - Treat
    [2026-04-20_raw_text_coord_continuity_probe.md](2026-04-20_raw_text_coord_continuity_probe.md)
    and
    [2026-04-20_raw_text_and_coord_family_decision_summary.md](2026-04-20_raw_text_and_coord_family_decision_summary.md)
    as supporting historical inputs rather than the first file to read.
  - Use
    [2026-04-11_stage1_coord_basin_duplication_mechanism.md](2026-04-11_stage1_coord_basin_duplication_mechanism.md)
    and
    [2026-04-13_duplication_collapse_final_analysis.md](2026-04-13_duplication_collapse_final_analysis.md)
    for the broader Stage-1 duplication-collapse mechanism line.
  - Family-specific performance follow-ups remain in:
    [2026-04-15_cxcy_logw_logh_retrained_performance_analysis.md](2026-04-15_cxcy_logw_logh_retrained_performance_analysis.md)
    and
    [2026-04-17_cxcywh_quickcheck_val200.md](2026-04-17_cxcywh_quickcheck_val200.md).
- Active 2B Channel-A / prefix / FN investigations
  - Start with
    [2026-03-17_stage2_2b_fn_factor_results.md](2026-03-17_stage2_2b_fn_factor_results.md).
  - Use
    [2026-03-17_stage2_2b_fn_factor_artifact_guide.md](2026-03-17_stage2_2b_fn_factor_artifact_guide.md)
    as the operator-facing artifact companion.
  - Keep
    [2026-03-16_stage2_2b_stage1_vs_aonly_prefix_fn_hypotheses_plan.md](2026-03-16_stage2_2b_stage1_vs_aonly_prefix_fn_hypotheses_plan.md)
    and
    [2026-03-17_stage2_2b_prefix_random_order_followup.md](2026-03-17_stage2_2b_prefix_random_order_followup.md)
    as planning and follow-up context.
- Active Channel-A random-order self-context investigations
  - [2026-03-20_stage2_channel_a_self_context_iter_ablation.md](2026-03-20_stage2_channel_a_self_context_iter_ablation.md)
- Historical Stage-2 duplication / UL diagnostic evidence
  - Start with
    [2026-03-26_stage2_small_object_duplication_offline_synthesis.md](2026-03-26_stage2_small_object_duplication_offline_synthesis.md)
    for the small-object offline cluster. This cluster is retained for
    duplicate-diagnostic provenance only; retired duplicate-burst UL objectives
    are not current launch guidance.
  - Supporting cluster notes live in:
    [2026-03-25_stage2_small_object_duplication_offline_protocol.md](2026-03-25_stage2_small_object_duplication_offline_protocol.md),
    [2026-03-26_stage2_small_object_duplication_offline_harness_findings.md](2026-03-26_stage2_small_object_duplication_offline_harness_findings.md),
    and
    [2026-03-26_stage2_small_object_duplication_crowded_deep_dive.md](2026-03-26_stage2_small_object_duplication_crowded_deep_dive.md).
  - Use [2026-03-05_stage2_near_duplication.md](2026-03-05_stage2_near_duplication.md)
    for the earlier mechanism diagnosis and
    [2026-03-09_stage2_ul_capture_highres1024.md](2026-03-09_stage2_ul_capture_highres1024.md)
    /
    [2026-03-12_stage2_triage_posterior_coco1024_train_dynamics.md](2026-03-12_stage2_triage_posterior_coco1024_train_dynamics.md)
    for train/run context.
  - Threshold and duplicate-mass slices remain in
    [2026-03-11_gt_overlap_threshold_search.md](2026-03-11_gt_overlap_threshold_search.md),
    [2026-03-11_rollout_duplication_thresholds_ul_vs_ulv2.md](2026-03-11_rollout_duplication_thresholds_ul_vs_ulv2.md),
    and
    [2026-03-24_stage2_pseudo_positive_k4_coord_only_findings.md](2026-03-24_stage2_pseudo_positive_k4_coord_only_findings.md).
- Stage-1 coord-basin duplication diagnostic evidence
  - [2026-04-11_stage1_coord_basin_duplication_mechanism.md](2026-04-11_stage1_coord_basin_duplication_mechanism.md)
- Historical Stage-2 failure diagnoses
  - [2026-02-17_stage2_b_ratio_085_instability.md](2026-02-17_stage2_b_ratio_085_instability.md)
  - [2026-02-21_stage2_channel_a_coord_gate.md](2026-02-21_stage2_channel_a_coord_gate.md)
  - [2026-02-25_stage2_channel_a_coord_loss.md](2026-02-25_stage2_channel_a_coord_loss.md)
  - [2026-02-25_stage2_channel_a_visual_audit.md](2026-02-25_stage2_channel_a_visual_audit.md)
  - [2026-02-22_stage2_softctx_discretization_vs_stage1_bbox.md](2026-02-22_stage2_softctx_discretization_vs_stage1_bbox.md)
- Tooling / operator aids
  - [2026-03-11_visualization_tools_index.md](2026-03-11_visualization_tools_index.md)
  - [artifacts/README.md](artifacts/README.md)

## Non-Diagnostic Workflow Notes

Supported workflows should live in `docs/`, not here.

- The canonical COCO test-dev submission path is:
  - [docs/eval/COCO_TEST_SUBMISSION.md](../../docs/eval/COCO_TEST_SUBMISSION.md)
- Benchmark-style checkpoint selection notes live in `progress/benchmarks/`, including
  [2026-04-21_mixed_objective_sota_checkpoint_probe.md](../benchmarks/2026-04-21_mixed_objective_sota_checkpoint_probe.md).

If you are recording an official score or a checkpoint-selection result, prefer `progress/benchmarks/`.
