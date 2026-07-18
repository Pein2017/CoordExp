---
title: Coordinate Objective and Decode Negative Results
type: investigation
role: historical-registry
authority: non_normative_research
status: historical-synthesis
updated: 2026-07-17
---

# Coordinate Objective and Decode Negative Results

This page records coordinate-surface, objective, and decode findings that should
prevent repeated dead-end exploration. It is not a current objective or
inference specification. Use [`docs/training/STAGE1_OBJECTIVE.md`](../../../docs/training/STAGE1_OBJECTIVE.md),
[`docs/eval/WORKFLOW.md`](../../../docs/eval/WORKFLOW.md), and stable
`openspec/specs/` contracts for current behavior.

## Registry

| ID | Negative result or lesson | Scope / source handle | Evidence and denominator | Currentity / boundary | Comparable group |
|---|---|---|---|---|---|
| COORD-001 | Teacher-forced coordinate diagnostics did not explain the large free-rollout detection gap. | Three historical 4B Stage-1 runs; `progress/benchmarks/stage1_training_dynamics_4b_2026-02-26.md`; TensorBoard scalars and 200-row COCO evaluation companion. | Coord-vocab mass converged near `0.99+` and expected-bin MAE near `21` for all arms, while mixed objective substantially outperformed pure/soft CE in rollout AP. | Reusable train-vs-rollout lesson; checkpoint family and old artifacts are historical. Require a tiny decode validation rather than inferring quality from teacher forcing alone. | `stage1_4b_objective_training_dynamics` |
| COORD-002 | Mixed hard-CE + soft-CE/W1/gate was stronger than single-objective baselines on the historical 4B COCO-80 val200 slice. | `progress/stage1_coco80_eval_4b_ckpts_768_vs_1024_2026-02-26.md`; three 4B-sized checkpoints, two nominal rescale presets. | AP was about `0.386–0.389` mixed vs `0.253–0.256` pure CE vs `0.224–0.231` soft CE over 200 rows. | Historical objective result. The “768 vs 1024” images were effectively the same 640×640 images; deltas are not a resolution ablation. | `stage1_4b_coco80_objective_comparison` |
| COORD-003 | In the clean 2B coordinate-component matrix, hard CE at `rp=1.10` was a stronger anchor than the same arm at `rp=1.05`; this is a decode × objective interaction. | `progress/benchmarks/2026-05-07_stage1_2b_coord_component_rp110_ablation.md`; COCO 1024 LVIS-proxy first 200, adapter-loaded coord tokens. | Hard CE AP `0.3307→0.3901`, F1@0.50 `0.4617→0.5582`, prediction total `1805→1555`, errors `4→0`, degenerate `13→0` when only RP changed. | Strong historical decode lesson, not a global RP default. Strict/plausible proxy views are not standard COCO metrics. | `stage1_2b_coord_component_rp_sweep` |
| COORD-004 | The historical 2B val200 leaderboard mixes incompatible families and should not be read as one causal ranking. | `progress/benchmarks/stage1_2b_val200_leaderboard.md` and `progress/benchmarks/2026-07-03_coord_token_val200_benchmark.md`. | Rows differ in compact-full vs JSON coordinate surfaces, sorted/random/geo-sorted order, dataset roots, max tokens, adapter/export mode, and duplicate-guard availability. | Registry maintenance rule: compare only within `comparable_group`; retain old rank as historical selection context. | `stage1_2b_val200_mixed_dashboard` |
| COORD-005 | Native Qwen3-VL text-coordinate baseline failures were dominated by prompt/template and stopping errors before geometry quality could be judged. | `outputs/research/qwen3-vl-native-text-coordinate-val200/benchmark.md`; repaired native-template smoke and val200. | Original greedy arm: 75 natural terminations, 125 truncations, 208 scored predictions across 200 rows; corrected `rp=1.10` arm reached mAP `0.199962` but still had severe duplicate bursts in `67/200` rows. | Useful parser/smoke lesson and ecological baseline only; not a causal serialization comparison. | `native_text_coordinate_val200_runner` |
| COORD-006 | Several coordinate/geometry objective candidates were negative or unstable on compact-full val200 and should remain controls, not promoted mechanisms. | `progress/benchmarks/2026-07-03_coord_token_val200_benchmark.md`; A5/A6 IoU-Gibbs, Gaussian SoftCE, Coord-repel, random/full-object SFT rows. | Examples: A5 AP `0.4018`, A6 AP `0.4053` with duplicate counts `447/470`; Gaussian mix0.5 AP `0.3539` with `904` duplicates; Coord-repel v10 AP `0.3186` with 58 errors and 29 invalid rows. | Run-specific negative results. Do not infer objective superiority without matching data/template/decode and raw artifact checks. | `compact_full_objective_negative_sweep` |
| COORD-007 | Local diagnostics can be numerically repaired without making the underlying model claim trustworthy. | Stage-2 crowded `v2_bboxfix` is retained here because it directly conditions coordinate/decode interpretation; source synthesis at `progress/diagnostics/2026-03-26_stage2_small_object_duplication_offline_synthesis.md`. | The bbox extraction fix changed artifact trust, not model weights. Pre-fix rows must not be pooled with corrected rows. | Trust asymmetry is a reusable evidence rule: parser fixes require a new artifact identity and supersede earlier derived metrics. | `diagnostics_parser_correction_v2_bboxfix` |

## Interpretation constraints

Coordinate objective comparisons require the full train/infer/eval identity:
coordinate serialization, parameterization, object order, dataset and slice,
checkpoint/export mode, effective decode kwargs, parser, evaluator, and
denominator policy. A lower AP row is not a negative objective result if its
decode or data contract changed. Conversely, a clean metric row without raw
rollout and invalid counters is not sufficient evidence of stability.

## Current owner links

Current objective semantics belong to the Stage-1 training docs and stable
supervision specs. Current inference/evaluation artifact flow belongs to
`docs/eval/WORKFLOW.md`. This registry deliberately stores no launch command,
config schema, or copied implementation prose.
