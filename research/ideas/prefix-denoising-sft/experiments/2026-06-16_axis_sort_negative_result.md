---
type: idea
title: Prefix Denoising Axis-Sort Repair Negative Result
description: Diagnostic negative result showing bbox endpoint sorting improves materialization but does not recover localization quality.
tags: [stage1, prefix-denoising, negative-result, eval]
updated: 2026-06-20
---

# Prefix Denoising Axis-Sort Repair Negative Result

## Question

Are the invalid bbox outputs from the prefix-denoising checkpoint mainly caused
by inverted endpoint order, and can axis sorting recover the expected val200 AP?

## Method

The diagnostic started from:

```text
/data/CoordExp/outputs/infer/prefix_denoising_sft_v1/prefix_denoising_ckpt450_val200_free_decode_diagnostic_eval_8gpu
```

and checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/detection_teacher_forcing/compact_full_prefix_denoising_kl_w0p05_2b_base_sorted_marker_bsz1x128_2epoch/compact-full-prefix-denoising-kl-w0p05-2b-base-sorted-marker-bsz1x128-2epoch/v25-20260615-124531/checkpoint-450
```

It tested post-repair token-trace sorting, immediate repair during inference
materialization, and immediate repair with repair-aware confidence.

## Result

Post-repair from token traces improved materialization:

```text
original raw AP       0.1176
post-repair raw AP    0.1457
original empty_pred   64
post-repair empty_pred 1
```

Immediate repair with standard confidence produced:

```text
raw AP       0.1227
raw AP50     0.2212
guarded AP   0.1182
empty_pred   49
```

Repair-aware confidence did not change the conclusion:

```text
raw AP      0.1214
raw AP50    0.2188
empty_pred  49
pred_total  984
```

## Interpretation

Axis sorting is a useful diagnostic because it separates endpoint inversion
from broader coordinate failure. It improves artifact coverage but does not
recover localization quality. The checkpoint is not merely swapping `x1/x2` or
`y1/y2`; it also produces edge-saturated, degenerate, wrong-arity, and poorly
localized boxes.

## Artifact Handles

- token-trace post-op:
  `/data/CoordExp/outputs/infer/prefix_denoising_sft_v1/prefix_denoising_ckpt450_val200_free_decode_diagnostic_eval_8gpu/postops_invalid/axis_sort_repair`
- immediate repair eval:
  `/data/CoordExp/outputs/infer/prefix_denoising_sft_v1/prefix_denoising_ckpt450_val200_decode_axis_sort_repair_eval`
- repair-aware confidence:
  `/data/CoordExp/outputs/infer/prefix_denoising_sft_v1/prefix_denoising_ckpt450_val200_decode_axis_sort_repair_eval/postops_invalid/repair_aware_confidence`

Manifest handle:

```text
source_path=progress/diagnostics/2026-06-16_prefix_denoising_axis_sort_repair_negative_result.md
source_kind=branch-head
classification=new_content_new_path
sha256=688fa6f04eb57ce5c9d48e0178551e7188281df6150d786818afe4911812b915
snapshot_path=docs/history/worktree-union/2026-06-20/snapshots/688fa6f04eb5/progress/diagnostics/2026-06-16_prefix_denoising_axis_sort_repair_negative_result.md
```

## Sources

- `docs/history/worktree-union/2026-06-20/snapshots/688fa6f04eb5/progress/diagnostics/2026-06-16_prefix_denoising_axis_sort_repair_negative_result.md`
- `docs/history/worktree-union/2026-06-20/manifest.tsv`
- `688fa6f04eb57ce5c9d48e0178551e7188281df6150d786818afe4911812b915`
