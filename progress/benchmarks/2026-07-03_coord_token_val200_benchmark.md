---
title: Coordinate-Token Val200 Benchmark Snapshot
date: 2026-07-03
status: draft-snapshot
topics: [stage1, coord-token, val200, detection, inference, evaluation]
tags: [benchmarks, coordinate-tokens, val200, artifact-backed]
summary: Artifact-backed snapshot of completed CoordExp inference/evaluation rows that use coordinate tokens or coordinate-token-derived compact-full detection surfaces, excluding legacy pure-text rows.
---

# Coordinate-Token Val200 Benchmark Snapshot

This benchmark snapshot is intentionally narrower than the mixed
`stage1_2b_val200_leaderboard` dashboard. It scopes to completed inference and
evaluation artifacts with coordinate tokens or coordinate-token-derived
compact-full detection surfaces. Legacy pure-text xyxy rows are out of scope.

All artifact paths below are relative to `/data/CoordExp` unless stated
otherwise.

## Scope

Included:

- completed inference/evaluation roots with metric-bearing `metrics.json`
- `infer.mode=coord`, `coord_mode=coord_tokens`, `coordinate_surface=coord_token`,
  or compact-full / closed-wrapper detection surfaces backed by coord tokens
- `val200`, first-200, or explicitly comparable validation-slice rows
- COCO-style detection AP plus F1-ish full micro evidence when available

Excluded:

- legacy/pure raw-text xyxy rows
- post-op child views such as `postops_invalid/axis_sort_repair` when they are
  transformations of a parent inference artifact rather than standalone
  inference runs
- core-six, val128, val512, full-val, proxy-expanded, and stochastic diagnostic
  rows from the topline table unless called out as caveats

Subagent artifact census:

- inspected `outputs/infer/**/eval/metrics.json` plus matching
  `resolved_config.json`, `summary.json`, `confidence_postop_summary.json`,
  `metrics_guarded.json`, and `duplicate_guard_report.json`
- found 79 completed metric roots
- found 60 val200/first-200 coordinate-token or coordinate-surface rows after
  excluding post-op children and legacy pure-text rows
- no inference/evaluation reruns were launched for this snapshot

Metric convention:

- `AP`, `AP50`, and `AP75` are `bbox_AP`, `bbox_AP50`, and `bbox_AP75`
- `F1@.50` and `F1@.30` are full semantic+localization micro F1-ish values
- `gAP` is guarded AP from duplicate-control metrics when available
- `err/invalid` comes from inference/eval parser counters; older rows may not
  record every counter in the same schema

## Topline Coordinate-Token Rows

| Rank | Run | AP | AP50 | AP75 | F1@.50 | F1@.30 | gAP | Rows err/invalid pred/dup | Scope |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | Mixed-objective coord-token ckpt1332 rp1.05 | 0.4584 | 0.6307 | 0.4770 | 0.6967 | 0.7455 | n/a | 200, err 0, invalid 0, pred 1315, dup n/a | `outputs/infer/coco1024_val200_compare_coordtoken_ckpt1332_20260423T080326Z` |
| 2 | Mixed-objective coord-token ckpt1332 rp1.00 | 0.4532 | 0.6177 | 0.4755 | 0.6544 | 0.7001 | n/a | 200, err 0, invalid 0, pred 1518, dup n/a | `outputs/infer/coco1024_val200_compare_coordtoken_ckpt1332_rp1p00_20260423T095845Z` |
| 3 | Coverage-ledger postmerge aligner DoRA ckpt928 | 0.4446 | 0.6154 | 0.4626 | 0.6508 | 0.7141 | 0.4264 | 200, err 2, invalid 1, pred 1326, dup 256 | `outputs/infer/recursive_detection_ce_latest/compact_full_prefix_rollin_balance2_postmerge_aligner_dora_ckpt928_sorted_val200_ckpt928_val200_bsz8_temp0_rp1p10_max3084_chatfix_8gpu` |
| 4 | Coverage-ledger closed hard ckpt928 | 0.4435 | 0.6109 | 0.4583 | 0.6540 | 0.7204 | 0.4256 | 200, err 0, invalid 0, pred 1330, dup 239 | `outputs/infer/recursive_detection_ce_latest/compact_full_prefix_rollin_balance2_ledger_ckpt928_val200_8gpu_ckpt928_val200_bsz1_temp0_rp1p10_max3084_chatfix_8gpu` |
| 5 | Coord Gaussian RPS ckpt900 | 0.4429 | 0.6114 | 0.4670 | 0.6551 | 0.7145 | 0.4237 | 200, err 0, invalid 0, pred 1399, dup 271 | `outputs/infer/recursive_detection_ce_latest/compact_full_prefix_rollin_balance2_coord_gaussian_rps_best900_ckpt900_val200_bsz8_temp0_rp1p10_max3084_chatfix_8gpu` |
| 6 | Mixed-objective coord-token ckpt1332 rp1.10 | 0.4419 | 0.6064 | 0.4579 | 0.6888 | 0.7364 | n/a | 200, err 0, invalid 0, pred 1285, dup n/a | `outputs/infer/coco1024_val200_compare_coordtoken_ckpt1332_rp1p10_20260423T091701Z` |
| 7 | Natural-adjacent pure CE ckpt928 rp1.00 | 0.4411 | 0.6135 | 0.4643 | 0.5470 | 0.5915 | 0.4256 | 200, err 2, invalid 1, pred 1951, dup 743 | `outputs/infer/natadj_len12000_free_val200/natadj_sorted_ckpt928_val200_free_temp0_rp1p00_max3084_8gpu` |
| 8 | Coverage-ledger premerge fp32 ckpt928 | 0.4400 | 0.6130 | 0.4578 | 0.6273 | 0.7028 | 0.4234 | 200, err 2, invalid 1, pred 1377, dup 291 | `outputs/infer/recursive_detection_ce_latest/compact_full_prefix_rollin_balance2_ledger_premerge_fp32_sorted_ckpt928_val200_ckpt928_val200_bsz8_temp0_rp1p10_max3084_chatfix_8gpu` |
| 9 | Coverage-ledger closed hard ckpt300 | 0.4343 | 0.6012 | 0.4397 | 0.6512 | 0.7103 | 0.4174 | 200, err 2, invalid 1, pred 1319, dup 243 | `outputs/infer/recursive_detection_ce_latest/compact_full_prefix_rollin_balance2_ledger_ckpt300_val200_8gpu_ckpt300_val200_bsz1_temp0_rp1p10_max3084_chatfix_8gpu` |
| 10 | Natural-adjacent pure CE ckpt928 rp1.10 | 0.4310 | 0.5985 | 0.4493 | 0.6355 | 0.6991 | 0.4162 | 200, err 6, invalid 3, pred 1458, dup 267 | `outputs/infer/natadj_len12000_free_val200/natadj_sorted_ckpt928_val200_free_temp0_rp1p10_max3084_bsz1_8gpu_symlinkbase` |
| 11 | ET-RMP support2 ckpt3664 prompt-offset | 0.4251 | 0.5751 | 0.4477 | 0.6135 | 0.6630 | 0.4120 | 200, err 0, invalid 3, pred 1141, dup 233 | `outputs/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_prompt_offset_fix_8gpu` |
| 12 | ET-RMP support2 ckpt3664 chatfix | 0.4247 | 0.5752 | 0.4477 | 0.6138 | 0.6633 | 0.4116 | 200, err 0, invalid 3, pred 1140, dup 232 | `outputs/infer/recursive_detection_ce_latest/compact_full_support2_tokenrows_v2_ckpt3664_val200_bsz8_temp0_rep1p10_max3084_chatfix_4gpu` |
| 13 | Natural-adjacent pure CE ckpt928 sampled | 0.4215 | 0.5951 | 0.4367 | 0.6410 | 0.7071 | 0.4078 | 200, err 2, invalid 1, pred 1415, dup 273 | `outputs/infer/natadj_len12000_free_val200/natadj_sorted_ckpt928_val200_sampling_t0p2_top0p9_rp1p10_max3084_bsz1_4gpu` |
| 14 | CoordExp-Swift r16/a32 EBS64 step917 | 0.4149 | 0.5664 | 0.4456 | 0.6281 | 0.6944 | 0.4051 | 200, err 0, invalid 0, pred 1243, dup 201 | `outputs/infer/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_dora_r16a32_ebs64_step917_val200_bsz4_temp0_rp1p10_max3084_8gpu` |
| 15 | Fullobj sorted SFT free-greedy ckpt3668 | 0.4072 | 0.5642 | 0.4226 | 0.5892 | 0.6471 | 0.3872 | 200, err 12, invalid 6, pred 1411, dup 311 | `outputs/infer/recursive_detection_ce_latest/compact_full_fullobj_sorted_sft_free_greedy_ckpt3668_val200_bsz8_temp0_rp1p10_max3084_chatfix_4gpu` |
| 16 | A6 cIoU Gibbs ckpt3664 | 0.4053 | 0.5653 | 0.4169 | 0.5089 | 0.5577 | 0.3965 | 200, err 0, invalid 4, pred 1296, dup 470 | `outputs/infer/recursive_detection_ce_latest/a6_ciou_gibbs_ckpt3664_val200_bsz8_temp0_rp1p10_max3084_chatfix_4gpu` |
| 17 | A5 IoU Gibbs ckpt3664 | 0.4018 | 0.5513 | 0.4246 | 0.5047 | 0.5516 | 0.3950 | 200, err 0, invalid 4, pred 1272, dup 447 | `outputs/infer/recursive_detection_ce_latest/a5_iou_gibbs_ckpt3664_val200_bsz8_temp0_rp1p10_max3084_chatfix_4gpu` |
| 18 | CoordExp-Swift EBS128 step459 | 0.4016 | 0.5559 | 0.4112 | 0.5892 | 0.6645 | 0.3946 | 200, err 6, invalid 0, pred 1098, dup 195 | `outputs/infer/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_dora_step459_val200_bsz4_temp0_rp1p10_max3084_8gpu` |
| 19 | A6 CE Gaussian mix0.2 ckpt3664 | 0.3996 | 0.5429 | 0.4178 | 0.5264 | 0.5841 | 0.3881 | 201, err 5, invalid 7, pred 1265, dup 416 | `outputs/infer/recursive_detection_ce_latest/a6_ce_gaussian_mix0p2_ckpt3664_val200_bsz4_temp0_rep1p10_max3084_chatfix_1gpu` |
| 20 | Random SFT bsz1 accum16 ckpt3664 | 0.3992 | 0.5577 | 0.4140 | 0.4433 | 0.4919 | 0.3827 | 200, err 652, invalid 21, pred 2261, dup 1148 | `outputs/infer/recursive_detection_ce_latest/compact_full_random_sft_bsz1_accum16_ckpt3664_val200_bsz4_temp0_rep1p10_max3084_chatfix_8gpu` |
| 21 | Fullobj random SFT free-greedy ckpt3668 | 0.3914 | 0.5351 | 0.4054 | 0.4659 | 0.5266 | 0.3749 | 200, err 20, invalid 10, pred 1687, dup 739 | `outputs/infer/recursive_detection_ce_latest/compact_full_fullobj_random_sft_free_greedy_ckpt3668_val200_bsz8_temp0_rp1p10_max3084_chatfix_4gpu` |
| 22 | Coord-component hard CE rp1.10 | 0.3901 | 0.5042 | 0.3954 | 0.5582 | 0.5978 | n/a | 200, err 0, invalid 0, pred 1555, dup n/a | `outputs/infer/coord_components_2b/coord_components_2b_coord_token_hard_ce_val200_adapter_rp110` |
| 23 | Fullwrap sorted prefix-denoise ckpt908 | 0.3878 | 0.5474 | 0.4040 | 0.5719 | 0.6343 | 0.3727 | 200, err 24, invalid 12, pred 1099, dup 227 | `outputs/infer/prefix_denoising_sft_v1/prefix_denoising_fullwrap_desc_sorted_fixed_fullwrap_desc_sorted_fixed_ckpt908_val200_rp1p10_ckpt908_val200_bsz8_temp0_rp1p10_max3084_chatfix_2gpu` |
| 24 | Fullwrap random prefix-denoise ckpt908 | 0.3778 | 0.5342 | 0.3905 | 0.4930 | 0.5632 | 0.3638 | 200, err 28, invalid 14, pred 1405, dup 518 | `outputs/infer/prefix_denoising_sft_v1/prefix_denoising_fullwrap_desc_random_fixed_fullwrap_desc_random_fixed_ckpt908_val200_rp1p10_ckpt908_val200_bsz8_temp0_rp1p10_max3084_chatfix_2gpu` |
| 25 | Gauss SoftCE mix0.5 ckpt1828 | 0.3539 | 0.4855 | 0.3724 | 0.4102 | 0.4559 | 0.3413 | 200, err 58, invalid 29, pred 1691, dup 904 | `outputs/infer/recursive_detection_ce_latest/compact_full_prefix_rollin_balance2_gauss_softce_mix0p5_len12000_val200_nonewline_diag_ckpt1828_val200_bsz8_temp0_rp1p10_max3084_chatfix_8gpu` |
| 26 | Coord-repel v10 ckpt1836 | 0.3186 | 0.4784 | 0.3197 | 0.4574 | 0.5091 | 0.3052 | 200, err 58, invalid 29, pred 1095, dup 354 | `outputs/infer/recursive_detection_ce_latest/compact_full_prefix_rollin_balance2_coord_repel_v10_ckpt1836_val200_bsz1_8gpu_free_decode_diag_ckpt1836_val200_bsz1_temp0_rp1p10_max3084_chatfix_8gpu_eval` |

## Readout

- Best completed coordinate-token val200 AP in this artifact snapshot remains the
  older mixed-objective coord-token ckpt1332 rp1.05 row: AP `0.4584`,
  AP50 `0.6307`, full F1@.50 `0.6967`.
- Best current closed-wrapper / compact-full style row in the swept artifacts is
  the coverage-ledger postmerge aligner DoRA checkpoint-928: AP `0.4446`,
  AP50 `0.6154`, full F1@.50 `0.6508`, guarded AP `0.4264`.
- The CoordExp-Swift r16/a32 EBS64 retrain checkpoint-917 beats the older
  CoordExp-Swift EBS128 checkpoint-459 on this val200 run:
  AP `0.4149` vs `0.4016`, full F1@.50 `0.6281` vs `0.5892`.
- The same CoordExp-Swift retrain is still below the stronger historical
  coordinate-token rows: mixed-objective ckpt1332, coverage-ledger ckpt928,
  and natural-adjacent pure CE ckpt928.

Do not read the table as one perfectly controlled experiment. Dataset surfaces
and contracts differ across rows: `bbox_max60` vs `bbox_len12000`, LVIS-proxy
vs plain COCO roots, sorted vs random vs geo-sorted order, compact-full closed
wrappers vs xyxy JSON coordinate-token rows, different maximum decode lengths,
and duplicate-guard availability.

## User-Named Postmerge Aligner DoRA Run

Inference artifact:

`/data/CoordExp/outputs/infer/recursive_detection_ce_latest/compact_full_prefix_rollin_balance2_postmerge_aligner_dora_ckpt928_sorted_val200_ckpt928_val200_bsz8_temp0_rp1p10_max3084_chatfix_8gpu`

Training run root:

`/data/CoordExp/outputs/stage1_2b/detection_teacher_forcing/coverage_ledger_closed_hard_sft_bsz32_2epoch_postmerge_aligner_dora/coverage-ledger-closed-hard-sft-bsz32-2epoch-postmerge-aligner-dora/v2-20260629-163300`

Checkpoint used by inference:

`/data/CoordExp/outputs/stage1_2b/detection_teacher_forcing/coverage_ledger_closed_hard_sft_bsz32_2epoch_postmerge_aligner_dora/coverage-ledger-closed-hard-sft-bsz32-2epoch-postmerge-aligner-dora/v2-20260629-163300/checkpoint-928`

Recorded config path:

`/data/CoordExp/.worktrees/ledger-auxiliary-loss/configs/stage1/detection_teacher_forcing/prod/coverage_ledger_closed_hard_sft_postmerge_aligner_dora.yaml`

The run is a full-closure / compact-object-box-closed coordinate-token
teacher-forcing run, not a legacy pure-text run.

### Training Config

The copied `config_source.yaml` says this production ablation extends
`coverage_ledger_closed_hard_sft.yaml`, warm-starts from:

`/data/CoordExp/outputs/stage1_2b/adapter_views/pure_ce_sorted_checkpoint_928_token_embeddings_adapter_llm_aligner_dora`

and changes the coverage-ledger object visual feature source to
`post_merge_projected`.

Resolved training hierarchy:

- pipeline: `stage1_research_teacher_forcing`
- objective: `research_teacher_forcing`
- objective profile: `hard_sft`
- detection template: `compact_object_box_closed`
- prompt: `coco_80`
- target sequence: `coord_token`, `xyxy`, `desc_first`, `sorted`, strict parse
- data:
  - train: `public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl`
  - val: `public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`
  - image root: `public_data/coco/rescale_32_1024_bbox_max60`
- model: `model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`
- warm-start adapter: pure-CE sorted checkpoint-928 token-embeddings + LLM +
  aligner DoRA adapter view
- attention: `flash_attention_2`
- dtype: `bfloat16`

Optimizer/runtime:

- epochs: `2`
- world size: `8`
- per-device train batch size: `1`
- gradient accumulation: `4`
- effective batch size: `32`
- seed and dataset seed: `42`
- LR: `2e-5`
- warmup ratio: `0.1`
- scheduler: `cosine`
- optimizer: `adamw_torch_fused` through the
  `multimodal_token_embeddings_adapter` optimizer path
- weight decay: `0.1`
- gradient checkpointing: enabled
- packing: static, length `12000`, eval packing enabled, precompute workers `32`
- eval every `100` steps, save every `300` steps, final checkpoint at step `928`

Adapter:

- PEFT type: LoRA with DoRA enabled
- rank: `16`
- alpha: `32`
- dropout: `0.0`
- trainable surface: language-model linear projections plus Qwen visual merger
  `linear_fc1` and `linear_fc2`
- VIT frozen
- modules saved with the adapter: `token_embeddings_adapter`,
  `coverage_ledger_head`

Checkpoint metadata:

- final global step: `928`
- final epoch: `2.0`
- best eval-loss checkpoint: step `900`
- inference artifact above used the final step `928`, not the best-loss step
  `900`

### Training Algorithm

The training algorithm is Stage-1 teacher-forcing over compact closed-wrapper
detection sequences with coordinate tokens. Tokenized detection metadata
produces a `TeacherForcingTargetIR`; the trainer converts that into supervision
spans, builds a role vocabulary for schema/text/coord/stop tokens, and routes
loss through `TrainerLossBridge` and the objective runner.

Main teacher-forcing terms:

- token-type mass / hard SFT token likelihood, weight `1.0`
- continuation margin, weight `0.2`
- conditional valid-set likelihood disabled
- within-valid coverage disabled
- bbox-positive-area disabled

Coverage-ledger auxiliary term:

- enabled
- coverage weight `0.1`
- region-anchor weight `0.1`
- projection dimension `256`
- temperature `0.2`
- positive class weight `1.0`
- feature source `post_merge_projected`

Mechanically, each object sidecar records the closed-wrapper object identity,
four coordinate-token spans, object/box control-token positions, normalized xyxy
box, image grid, and processed image size. During loss computation, the bridge
captures one Qwen forward pass, keeps full logits for hard teacher forcing, and
extracts post-merge visual embeddings for object regions. The
`coverage_ledger_head` projects language hidden states at coverage/anchor
positions and pooled object visual embeddings into the same 256-dim normalized
space. The auxiliary computes binary cross entropy for cumulative object
coverage and a one-vs-all row/object binding loss, then adds the weighted
coverage-ledger loss to the teacher-forcing loss.

This is why the checkpoint is best described as:

> pure-CE sorted checkpoint-928 warm start + Stage-1 hard teacher-forcing
> closed-wrapper coord-token training + post-merge visual coverage-ledger
> auxiliary + DoRA over LLM and Qwen visual merger.

## Exclusion Notes

- The raw-text rows in `progress/benchmarks/stage1_2b_val200_leaderboard.md`
  are intentionally absent here.
- Some high-scoring `core6` and stop-control runs exist under
  `outputs/infer/**/eval_detection`, but they are subset diagnostics rather
  than val200 benchmark rows.
- Some rows report `rows=201`; the table preserves artifact counts instead of
  normalizing them after the fact.
- For paper-style claims, compare raw AP and guarded AP together when guarded
  metrics exist.
