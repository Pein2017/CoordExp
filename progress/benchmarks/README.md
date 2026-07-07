---
doc_id: progress.benchmarks.index
layer: progress
doc_type: router
status: legacy-router
domain: research-history
summary: Router for measured run comparisons, checkpoint selection notes, and evaluation sweeps.
tags: [progress, benchmarks, evaluation]
updated: 2026-05-07
---

# Benchmarks Index

Use this legacy folder only to reconstruct old measured comparisons, selection
decisions, or scoreboard-style reports.

Typical fits:

- checkpoint-vs-checkpoint detection metrics
- decoding or temperature sweep results
- training-dynamics comparisons used to explain outcome differences
- official benchmark results after a run is complete

Prefer `research/` for new measured comparisons, root-cause analyses, and
benchmark interpretation.

Family-specific score audits should be authored in `research/`; old
`progress/diagnostics/` entries remain legacy provenance only.

## Legacy Clusters

- Stage-1 detection result reports
  - [stage1_2b_val200_leaderboard.md](stage1_2b_val200_leaderboard.md)
  - [2026-05-07_stage1_2b_coord_component_rp110_ablation.md](2026-05-07_stage1_2b_coord_component_rp110_ablation.md)
  - [2026-05-07_compact_full_rp110_top3_union_unlabeled_prior.md](2026-05-07_compact_full_rp110_top3_union_unlabeled_prior.md)
  - [2026-04-28_stage1_mp_branch_runtime_packing_probe.md](2026-04-28_stage1_mp_branch_runtime_packing_probe.md)
  - [2026-04-23_stage1_raw_text_vs_coord_token_repetition_penalty_sweep.md](2026-04-23_stage1_raw_text_vs_coord_token_repetition_penalty_sweep.md)
  - [2026-02-26_stage1_coco80_4b_res_768_vs_1024.md](2026-02-26_stage1_coco80_4b_res_768_vs_1024.md)
  - [2026-02-26_stage1_coco80_temp0_compare.md](2026-02-26_stage1_coco80_temp0_compare.md)
  - [2026-02-27_stage1_coco_2b_ce_softce_res_768_vs_1024.md](2026-02-27_stage1_coco_2b_ce_softce_res_768_vs_1024.md)
  - [2026-02-26_stage1_training_dynamics_4b.md](2026-02-26_stage1_training_dynamics_4b.md)
  - [2026-04-21_mixed_objective_sota_checkpoint_probe.md](2026-04-21_mixed_objective_sota_checkpoint_probe.md)
- Stage-2 evaluation and selection notes
  - [2026-02-01_stage2_channel_a_infer_eval.md](2026-02-01_stage2_channel_a_infer_eval.md)
  - [2026-03-11_stage2_oracle_k_first200.md](2026-03-11_stage2_oracle_k_first200.md)
  - [2026-03-11_stage2_rollout_temperature_refinement.md](2026-03-11_stage2_rollout_temperature_refinement.md)

## COCO Official Results

When an official COCO test-dev submission returns a server score, record the new
interpretation under `research/` and use
[docs/eval/COCO_TEST_SUBMISSION.md](../../docs/eval/COCO_TEST_SUBMISSION.md) as
the workflow reference. Do not create new `progress/` benchmark notes.
