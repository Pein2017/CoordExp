---
doc_id: progress.diagnostics.index
layer: progress
doc_type: router
status: canonical
domain: research-history
summary: Router for failure investigations, mechanism studies, threshold sweeps, and operator notes that support diagnosis.
tags: [progress, diagnostics, investigations]
updated: 2026-05-01
---

# Diagnostics Index

Use this folder when the primary question is:

- what is going wrong?
- why is a behavior happening?
- which overlap / decoding threshold looks safe?
- which operator tool should I open to inspect a failure?

Prefer `progress/benchmarks/` when the output is mainly a measured run-vs-run or checkpoint-vs-checkpoint comparison.

## Current Clusters

- Canonical ET-RMP continuation / repetition-penalty / FN diagnostics
  - Start with
    [2026-04-29_et_rmp_rp_continuation_bias_hypothesis.md](2026-04-29_et_rmp_rp_continuation_bias_hypothesis.md)
    for the pre-support-mass-enhancement ET-RMP objective context, `val200`
    and core-6 RP sweeps, fixed representative sample bank, FN latent probes,
    length/count close-pressure read, and hard stop-control ablation.
- Closed Qwen3-VL coord-token instance-binding mechanism study
  - Start with
    [2026-04-24_qwen3_vl_instance_binding_mechanism_findings.md](2026-04-24_qwen3_vl_instance_binding_mechanism_findings.md)
    for the fixed-checkpoint mechanism conclusion: partial pre-`x1` binding,
    late schema/pre-coordinate readout, and `x1/y1` as the hard commitment
    boundary.
- Active Stage-2 birth-first Channel-B decision study
  - Start with
    [2026-04-22_stage2_birth_first_channel_b_decision_study.md](2026-04-22_stage2_birth_first_channel_b_decision_study.md)
    for the merged-vLLM operator fix, the paired small-fraction control versus
    birth-first result, and the current recommendation.
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
- Active Stage-2 duplication / UL investigations
  - Start with
    [2026-03-26_stage2_small_object_duplication_offline_synthesis.md](2026-03-26_stage2_small_object_duplication_offline_synthesis.md)
    for the small-object offline cluster.
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
- Active Stage-1 coord-basin duplication investigations
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
