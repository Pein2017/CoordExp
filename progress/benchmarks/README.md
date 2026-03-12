---
doc_id: progress.benchmarks.index
layer: progress
doc_type: router
status: canonical
domain: research-history
summary: Router for measured run comparisons, checkpoint selection notes, and evaluation sweeps.
tags: [progress, benchmarks, evaluation]
updated: 2026-03-11
---

# Benchmarks Index

Use this folder when the primary output is a measured comparison, selection decision, or scoreboard-style report.

Typical fits:

- checkpoint-vs-checkpoint detection metrics
- decoding or temperature sweep results
- training-dynamics comparisons used to explain outcome differences
- official benchmark results after a run is complete

Prefer `progress/diagnostics/` when the main question is root cause or failure analysis.

## Current Clusters

- Stage-1 detection result reports
  - [stage1_coco80_4b_res_768_vs_1024_2026-02-26.md](stage1_coco80_4b_res_768_vs_1024_2026-02-26.md)
  - [stage1_coco80_temp0_compare_2026-02-26.md](stage1_coco80_temp0_compare_2026-02-26.md)
  - [stage1_coco_2b_ce_softce_res_768_vs_1024_2026-02-27.md](stage1_coco_2b_ce_softce_res_768_vs_1024_2026-02-27.md)
  - [stage1_training_dynamics_4b_2026-02-26.md](stage1_training_dynamics_4b_2026-02-26.md)
- Stage-2 evaluation and selection notes
  - [stage2_channel_a_infer_eval_2026-02-01.md](stage2_channel_a_infer_eval_2026-02-01.md)
  - [stage2_oracle_k_first200_2026-03-11.md](stage2_oracle_k_first200_2026-03-11.md)
  - [stage2_rollout_temperature_refinement_2026-03-11.md](stage2_rollout_temperature_refinement_2026-03-11.md)

## COCO Official Results

When an official COCO test-dev submission returns a server score, record the measured result here and use
[docs/eval/COCO_TEST_SUBMISSION.md](../../docs/eval/COCO_TEST_SUBMISSION.md) as the workflow reference.
