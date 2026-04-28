---
doc_id: progress.benchmarks.index
layer: progress
doc_type: router
status: canonical
domain: research-history
summary: Router for measured run comparisons, checkpoint selection notes, and evaluation sweeps.
tags: [progress, benchmarks, evaluation]
updated: 2026-04-28
---

# Benchmarks Index

Use this folder when the primary output is a measured comparison, selection decision, or scoreboard-style report.

Typical fits:

- checkpoint-vs-checkpoint detection metrics
- decoding or temperature sweep results
- training-dynamics comparisons used to explain outcome differences
- official benchmark results after a run is complete

Prefer `progress/diagnostics/` when the main question is root cause or failure analysis.

Family-specific score audits can stay in `progress/diagnostics/` when the main
point is failure-mode interpretation rather than a final run-vs-run benchmark.

## Current Clusters

- Stage-1 detection result reports
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

When an official COCO test-dev submission returns a server score, record the measured result here and use
[docs/eval/COCO_TEST_SUBMISSION.md](../../docs/eval/COCO_TEST_SUBMISSION.md) as the workflow reference.
