---
type: idea
title: Prefix Denoising SFT Implementation
description: Worktree, branch, code-surface, verification, and artifact handles for the prefix-denoising SFT V1 implementation.
tags: [stage1, prefix-denoising, implementation, worktree]
updated: 2026-06-20
---

# Prefix Denoising SFT Implementation

## Worktree

```text
/data/CoordExp/.worktrees/geometry-aware-denoising-sft
```

## Branch

```text
codex/prefix-denoising-sft
```

Raw intake branch head:

```text
d08ef3ed63f6ffbd7d572afb162ea921b9b3a7ba
```

This document does not claim the implementation is merged into `main`.

Manifest rows used for implementation provenance include the branch-head rows
for the historical design spec, historical implementation plan, direction note,
audit note, and repaired launch-health note, plus the
`new_content_new_path` snapshot row
`snapshots/688fa6f04eb5/progress/diagnostics/2026-06-16_prefix_denoising_axis_sort_repair_negative_result.md`
and the `worktree-dirty` snapshot row
`snapshots/ed610b284219/progress/diagnostics/2026-06-17_prefix_denoising_inert_objective_root_cause_analysis.md`.

## Code Surfaces

Historical branch provenance identifies the intended implementation surfaces:

- `src/config/schema.py`
- `src/datasets/geometry.py`
- `src/detection/prefix_denoising/`
- `src/detection/dataset.py`
- `src/detection/runtime.py`
- `src/detection/packing.py`
- `src/data_collators/enrichers.py`
- `src/data_collators/batch_extras_collator.py`
- `src/trainers/metrics/prefix_denoising.py`
- `src/trainers/metrics/mixins.py`
- `src/bootstrap/trainer_setup.py`
- `src/sft.py`

Treat these as branch implementation handles, not current-main behavior
authority unless verified in the target checkout.

## Config Surfaces

Historical branch provenance identifies these planned config leaves:

- `configs/stage1/detection_teacher_forcing/prod/compact_full_prefix_denoising_ce_only.yaml`
- `configs/stage1/detection_teacher_forcing/prod/compact_full_prefix_denoising_kl_w0p05.yaml`
- `configs/stage1/detection_teacher_forcing/smoke/compact_full_prefix_denoising_ce_only_tiny.yaml`
- `configs/stage1/detection_teacher_forcing/smoke/compact_full_prefix_denoising_kl_w0p05_tiny.yaml`

The implementation surface is branch-local historical provenance in this pilot;
stable contract promotion would require current docs/OpenSpec updates.

## Verification Handles

Post-repair verification recorded:

- CPU regression and integration slice: `285 passed in 2.95s`.
- Focused provenance slice: `24 passed in 1.58s`.
- Compile check: exit 0.
- Patch hygiene: `git diff --check` exit 0.
- cfg-only smoke checks for CE-only and KL-on leaves: both returned
  `status=ok`.
- repaired CE-only tiny GPU smoke: completed 2/2 steps.
- repaired KL-on tiny GPU smoke: completed 2/2 steps.

These are launch-health handles, not full validation.

## Artifact Handles

- repaired CE-only tiny smoke:
  `/data/CoordExp/.worktrees/geometry-aware-denoising-sft/temp/detection_teacher_forcing/output/compact_full_prefix_denoising_ce_only_tiny/smoke-compact-full-prefix-denoising-ce-only-tiny/v18-20260615-022333`
- repaired KL-on tiny smoke:
  `/data/CoordExp/.worktrees/geometry-aware-denoising-sft/temp/detection_teacher_forcing/output/compact_full_prefix_denoising_kl_w0p05_tiny/smoke-compact-full-prefix-denoising-kl-w0p05-tiny/v13-20260615-022423`
- ckpt-450 training checkpoint:
  `/data/CoordExp/outputs/stage1_2b/detection_teacher_forcing/compact_full_prefix_denoising_kl_w0p05_2b_base_sorted_marker_bsz1x128_2epoch/compact-full-prefix-denoising-kl-w0p05-2b-base-sorted-marker-bsz1x128-2epoch/v25-20260615-124531/checkpoint-450`
- diagnostic inference root:
  `/data/CoordExp/outputs/infer/prefix_denoising_sft_v1/prefix_denoising_ckpt450_val200_free_decode_diagnostic_eval_8gpu`

## Current Implementation Read

The current implementation read is branch-provenance only. The launch path was
healthy after repair, but later evidence suggests the tested objective did not
materially change model behavior. Future implementation work should first
settle the matched-control question before promoting the branch or converting
the idea into current training guidance.

## Sources

- `docs/history/superpowers/specs/2026-06-14-prefix-denoising-sft-v1-design.md`
- `docs/history/superpowers/plans/2026-06-14-prefix-denoising-sft-v1.md`
- `docs/history/worktree-union/2026-06-20/manifest.tsv`
- `progress/diagnostics/2026-06-15_prefix_denoising_branch_isolation_repair.md`
- `docs/history/worktree-union/2026-06-20/snapshots/688fa6f04eb5/progress/diagnostics/2026-06-16_prefix_denoising_axis_sort_repair_negative_result.md`
- `docs/history/worktree-union/2026-06-20/snapshots/ed610b284219/progress/diagnostics/2026-06-17_prefix_denoising_inert_objective_root_cause_analysis.md`
