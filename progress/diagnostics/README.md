---
doc_id: progress.diagnostics.index
layer: progress
doc_type: router
status: canonical
domain: research-history
summary: Router for failure investigations, threshold studies, and operator notes that support diagnosis.
tags: [progress, diagnostics, investigations]
updated: 2026-03-26
---

# Diagnostics Index

Use this folder when the primary question is:

- what is going wrong?
- why is a behavior happening?
- which overlap / decoding threshold looks safe?
- which operator tool should I open to inspect a failure?

Prefer `progress/benchmarks/` when the output is mainly a measured run-vs-run or checkpoint-vs-checkpoint comparison.

## Current Clusters

- Active 2B Channel-A / prefix / FN investigations
  - [stage2_2b_stage1_vs_aonly_prefix_fn_2026-03-16.md](stage2_2b_stage1_vs_aonly_prefix_fn_2026-03-16.md)
  - [stage2_2b_fn_factor_results_2026-03-17.md](stage2_2b_fn_factor_results_2026-03-17.md)
  - [stage2_2b_fn_factor_artifact_bundle_2026-03-17.md](stage2_2b_fn_factor_artifact_bundle_2026-03-17.md)
  - [stage2_2b_prefix_random_order_followup_2026-03-17.md](stage2_2b_prefix_random_order_followup_2026-03-17.md)
- Active Channel-A random-order self-context investigations
  - [stage2_channel_a_self_context_iter_ablation_2026-03-20.md](stage2_channel_a_self_context_iter_ablation_2026-03-20.md)
- Active Stage-2 duplication / UL investigations
  - [stage2_small_object_duplication_offline_synthesis_2026-03-26.md](stage2_small_object_duplication_offline_synthesis_2026-03-26.md)
  - [stage2_near_duplication_2026-03-05.md](stage2_near_duplication_2026-03-05.md)
  - [stage2_ul_capture_highres1024_2026-03-09.md](stage2_ul_capture_highres1024_2026-03-09.md)
  - [stage2_triage_posterior_coco1024_train_dynamics_2026-03-12.md](stage2_triage_posterior_coco1024_train_dynamics_2026-03-12.md)
  - [stage2_pseudo_positive_k4_coord_only_findings_2026-03-24.md](stage2_pseudo_positive_k4_coord_only_findings_2026-03-24.md)
  - [gt_overlap_threshold_search_2026-03-11.md](gt_overlap_threshold_search_2026-03-11.md)
  - [rollout_duplication_thresholds_ul_vs_ulv2_2026-03-11.md](rollout_duplication_thresholds_ul_vs_ulv2_2026-03-11.md)
- Historical Stage-2 failure diagnoses
  - [stage2_b_ratio_085_instability_2026-02-17.md](stage2_b_ratio_085_instability_2026-02-17.md)
  - [stage2_channel_a_coord_gate_2026-02-21.md](stage2_channel_a_coord_gate_2026-02-21.md)
  - [stage2_channel_a_coord_loss_2026-02-25.md](stage2_channel_a_coord_loss_2026-02-25.md)
  - [stage2_channel_a_visual_audit_2026-02-25.md](stage2_channel_a_visual_audit_2026-02-25.md)
  - [stage2_softctx_discretization_vs_stage1_bbox_2026-02-22.md](stage2_softctx_discretization_vs_stage1_bbox_2026-02-22.md)
- Tooling / operator aids
  - [visualization_tools_index_2026-03-11.md](visualization_tools_index_2026-03-11.md)

## Non-Diagnostic Workflow Notes

Supported workflows should live in `docs/`, not here.

- The canonical COCO test-dev submission path is:
  - [docs/eval/COCO_TEST_SUBMISSION.md](../../docs/eval/COCO_TEST_SUBMISSION.md)

If you are recording an official score or a checkpoint-selection result, prefer `progress/benchmarks/`.
