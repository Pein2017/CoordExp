# Retained benchmarks evidence

Read the [current research questions](/data/CoordExp/.worktrees/research-probes/research/index.md)
for present interpretation. These originals retain distinct measurements or design
assumptions listed below. Their proposed runs and old paths are historical, not
a current execution queue. No new records belong here.

| Source | Independent retained use |
|---|---|
| [Channel-A iterations](2026-02-01_stage2_channel_a_infer_eval.md) | Original iteration comparison, exact duplicates, overwritten Run A measurements, and unmatched decode/population conditions. |
| [4B resolution presets](2026-02-26_stage1_coco80_4b_res_768_vs_1024.md) | Six checkpoint/preset measurements; COCO image pixels did not change, and the checkpoint named 2B was actually 4B. |
| [4B deterministic comparison](2026-02-26_stage1_coco80_temp0_compare.md) | Per-class/image differences and confidence/token diagnostics absent from the resolution note. |
| [4B training dynamics](2026-02-26_stage1_training_dynamics_4b.md) | Similar teacher-forced coordinate errors alongside different generation outcomes; missing-config and comparability limits. |
| [2B CE/SoftCE](2026-02-27_stage1_coco_2b_ce_softce_res_768_vs_1024.md) | Actual merged 2B checkpoints at matched batch size, distinct from the mislabeled 4B family. |
| [K8 union recovery](2026-03-11_stage2_oracle_k_first200.md) | Per-temperature contributions and baseline-FN denominators across pre/post-UL-fix checkpoints; truncation confounds. |
| [Temperature refinement](2026-03-11_stage2_rollout_temperature_refinement.md) | Two-seed recall/empty-output/overlap tradeoffs behind the historical temperature0.7 choice. |
| [Mixed-objective checkpoint](2026-04-21_mixed_objective_sota_checkpoint_probe.md) | Adapter-specific detection, verifier and finite-sampling FN evidence with summary-artifact locators. |
| [Representation/RP sweep](2026-04-23_stage1_raw_text_vs_coord_token_repetition_penalty_sweep.md) | Matched sweep and scorer repair: discarded raw-text predictions became measurable without rerunning the model. |
| [Runtime packing comparison](2026-04-28_stage1_mp_branch_runtime_packing_probe.md) | Rough eight-GPU timing/memory, changed logical batch geometry and unpassed same-batch numerical-parity gate; generated aggregate stays in artifacts/. |
| [Top-three candidate union](2026-05-07_compact_full_rp110_top3_union_unlabeled_prior.md) | BBox-union proxy, filtered/unfiltered sensitivity and A3/A4 continuation/collapse with unmatched supervision density. |
| [Coordinate-component ablation](2026-05-07_stage1_2b_coord_component_rp110_ablation.md) | Loss/RP matrix, hard-CE decode rescue and prediction-health counters; manual-sharding comparison limits. |
| [July coordinate-token census](2026-07-03_coord_token_val200_benchmark.md) | Artifact/training identity and heterogeneous comparison contracts in the later scoreboard. |
| [Historical metric crosswalk](stage1_2b_val200_leaderboard.md) | Comparable groups and full-F1 information not entirely reproduced by individual prose reports; this is not a live dashboard. |

Metric identity matters: April's coordinate-token/RP1.05 F1 **0.7056** is
localization F1; July's **0.6967** is full semantic-plus-localization F1. Do not
harmonize these numbers. The union's **549 unmatched clusters / 1,444 annotations**
is a bbox-based candidate proxy, not verified missing physical owners.
