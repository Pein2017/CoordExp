---
type: idea
role: condition-registry
authority: non_normative_research
promotion_status: not_promoted
title: Prefix Denoising SFT Conditions
description: Resolver for recurring Prefix Denoising SFT condition names, checkpoints, configs, eval settings, and artifact handles.
tags: [stage1, prefix-denoising, conditions, provenance]
updated: 2026-07-09
---

# Prefix Denoising SFT Conditions

This file resolves recurring condition names used by the Prefix Denoising SFT
research units. It is non-normative research provenance, not current training
guidance or a stable config contract.

## Branch Context

- worktree: `/data/CoordExp/.worktrees/geometry-aware-denoising-sft`
- branch: `codex/prefix-denoising-sft`
- branch head in raw intake manifest:
  `d08ef3ed63f6ffbd7d572afb162ea921b9b3a7ba`
- raw intake manifest:
  `docs/history/worktree-union/2026-06-20/manifest.tsv`

## Training Conditions

### V1 Prefix-Denoising KL Checkpoint

- checkpoint:
  `/data/CoordExp/outputs/stage1_2b/detection_teacher_forcing/compact_full_prefix_denoising_kl_w0p05_2b_base_sorted_marker_bsz1x128_2epoch/compact-full-prefix-denoising-kl-w0p05-2b-base-sorted-marker-bsz1x128-2epoch/v25-20260615-124531/checkpoint-450`
- train scope:
  `COCO rescale_32_1024_bbox_max60`, sorted compact-full, LoRA r16 DoRA,
  2 epochs.
- objective scope:
  paired clean/noisy prefix views, branch-balanced hard CE, optional sparse
  local-window coordinate KL.

### Reference Baseline Used In Root-Cause Analysis

- checkpoint:
  `/data/CoordExp/outputs/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332`
- retained deprecated merged-full handle:
  `/data/CoordExp/output_remote_DEPRECATED_20260604/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332-merged-full`
- limitation:
  this baseline is not matched to the prefix-denoising recipe; it differs on
  objective, epoch/step budget, and adapter/full-merge state.

## Launch-Health Conditions

### Historical 2026-06-14 Tiny Smokes

- CE-only:
  `/data/CoordExp/.worktrees/geometry-aware-denoising-sft/temp/detection_teacher_forcing/output/compact_full_prefix_denoising_ce_only_tiny/smoke-compact-full-prefix-denoising-ce-only-tiny/v10-20260614-195716`
- KL-on:
  `/data/CoordExp/.worktrees/geometry-aware-denoising-sft/temp/detection_teacher_forcing/output/compact_full_prefix_denoising_kl_w0p05_tiny/smoke-compact-full-prefix-denoising-kl-w0p05-tiny/v5-20260614-200052`
- limitation:
  superseded as current launch-health evidence because these smokes predate the
  branch-isolated forward repair.

### Repaired 2026-06-15 Tiny Smokes

- CE-only:
  `/data/CoordExp/.worktrees/geometry-aware-denoising-sft/temp/detection_teacher_forcing/output/compact_full_prefix_denoising_ce_only_tiny/smoke-compact-full-prefix-denoising-ce-only-tiny/v18-20260615-022333`
- KL-on:
  `/data/CoordExp/.worktrees/geometry-aware-denoising-sft/temp/detection_teacher_forcing/output/compact_full_prefix_denoising_kl_w0p05_tiny/smoke-compact-full-prefix-denoising-kl-w0p05-tiny/v13-20260615-022423`
- supported claim:
  branch-isolated V1 launch wiring was healthy at tiny scope.

## Eval/Repair Conditions

### ckpt-450 Free-Decode Diagnostic

- inference/eval root:
  `/data/CoordExp/outputs/infer/prefix_denoising_sft_v1/prefix_denoising_ckpt450_val200_free_decode_diagnostic_eval_8gpu`
- decode scope:
  val200, free greedy decode, temp 0, repetition penalty 1.1,
  `max_new_tokens=3084`.

### Axis-Sort Repair

- token-trace post-op:
  `/data/CoordExp/outputs/infer/prefix_denoising_sft_v1/prefix_denoising_ckpt450_val200_free_decode_diagnostic_eval_8gpu/postops_invalid/axis_sort_repair`
- immediate repair eval:
  `/data/CoordExp/outputs/infer/prefix_denoising_sft_v1/prefix_denoising_ckpt450_val200_decode_axis_sort_repair_eval`
- repair-aware confidence:
  `/data/CoordExp/outputs/infer/prefix_denoising_sft_v1/prefix_denoising_ckpt450_val200_decode_axis_sort_repair_eval/postops_invalid/repair_aware_confidence`

## Next Condition To Resolve

The missing gate is a matched denoising-OFF hard-CE LoRA control with the same
format, sorted marker prompt, train/eval data, adapter recipe, epoch/step
budget, and decode settings. Until that control exists, keep V1 claims inside
research.
