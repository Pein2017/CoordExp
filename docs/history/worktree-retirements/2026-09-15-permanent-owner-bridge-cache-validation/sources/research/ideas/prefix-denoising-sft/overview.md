---
type: idea
title: Prefix Denoising SFT
description: Explores whether Stage-1 compact detection teacher forcing can improve coordinate robustness by training on clean and coordinate-noised prefix views.
tags: [stage1, compact-detection, prefix-denoising, teacher-forcing, coordinate-robustness]
state: active
updated: 2026-06-20
---

# Prefix Denoising SFT

## Current State

Prefix Denoising SFT is a valid active idea, but it is not concluded. The V1
branch has implementation, repair, launch-health, inference, and post-analysis
records, but it does not yet have the matched control needed for a final
promotion or rejection verdict.

Initial V1 launch wiring became healthy after the branch-isolation repair in
`progress/diagnostics/2026-06-15_prefix_denoising_branch_isolation_repair.md`.
The earlier 2026-06-14 smoke is preserved as historical launch evidence, but it
is superseded as current launch-health evidence because it predates the repaired
branch-isolated forwards.

Later analysis found the tested objective likely inert for the compact-coordinate
model: clean and noisy branch CE were nearly identical, local-window KL was near
zero from step 0, and coordinate prediction appeared insensitive to previous
coordinate tokens. This makes the V1 result a negative result about the tested
implementation premise, not evidence against geometry-aware denoising as a
broader concept.

## Central Question

Can a Stage-1 compact detection model become more robust to autoregressive
coordinate-prefix drift by training on paired clean and valid coordinate-noised
prefix views, with optional sparse clean-to-noisy coordinate KL?

## Current Interpretation

The idea is still useful, but the first tested mechanism likely did not engage.
The healthy branch-isolated launch demonstrates at tiny scope that the runtime
path can train and emit the expected metrics. The inference and post-analysis
notes show that endpoint sorting improves materialization but does not recover
localization quality, and that the apparent bbox degradation is confounded by
an unmatched baseline.

The next interpretation gate is a matched denoising-OFF hard-CE LoRA control at
the same recipe, plus a coordinate-objective comparison before any claim about
denoising benefit or harm.

## Worktree And Branch Handles

- worktree: `/data/CoordExp/.worktrees/geometry-aware-denoising-sft`
- branch: `codex/prefix-denoising-sft`
- branch head in raw intake manifest:
  `d08ef3ed63f6ffbd7d572afb162ea921b9b3a7ba`
- migration raw intake:
  `docs/history/worktree-union/2026-06-20/manifest.tsv`

## Main Reading Path

- [Draft](draft.md) - original V1 idea and rationale
- [Conditions](conditions.md) - recurring checkpoint, config, eval, and artifact condition names
- [Discussion](discussion.md) - audit findings, repair decisions, and interpretation boundaries
- [Implementation](implementation.md) - worktree, branch, code/config, verification, and artifact handles
- [Launch Health And Branch Isolation](experiments/2026-06-14-launch-health-and-branch-isolation/unit.md)
- [Axis-Sort Repair Negative Result](experiments/2026-06-16-axis-sort-negative-result/unit.md)
- [Inert Objective Root-Cause Analysis](experiments/2026-06-17-inert-objective-root-cause/unit.md)

This idea is active and does not yet have a final `conclusion.md`.

## Key Evidence

- Original direction: V1 targets Stage-1 compact detection teacher forcing with
  paired `clean_full` and `noisy_full` views, branch-balanced hard CE, and
  optional sparse local-window clean-to-noisy coordinate KL.
- Audit: the research design was considered coherent, but the implementation
  plan had blocking risks around sidecar registration, packed branch isolation,
  and packed multimodal materialization.
- Repair: branch-isolated forwards were added, CPU and focused tests passed,
  cfg-only checks returned `status=ok`, and both tiny GPU smokes completed 2/2
  train steps with repaired runtime payloads.
- Axis-sort negative result: endpoint sorting improved materialization coverage
  but did not recover localization quality; the diagnostic AP stayed far below
  the expected reference range.
- Root-cause analysis: the tested objective was likely inert and the comparison
  to the stronger reference baseline was not valid because objective, training
  length, adapter/full-merge state, and recipe differed.

## Artifact Handles

- 2026-06-14 historical CE-only tiny smoke:
  `/data/CoordExp/.worktrees/geometry-aware-denoising-sft/temp/detection_teacher_forcing/output/compact_full_prefix_denoising_ce_only_tiny/smoke-compact-full-prefix-denoising-ce-only-tiny/v10-20260614-195716`
- 2026-06-14 historical KL-on tiny smoke:
  `/data/CoordExp/.worktrees/geometry-aware-denoising-sft/temp/detection_teacher_forcing/output/compact_full_prefix_denoising_kl_w0p05_tiny/smoke-compact-full-prefix-denoising-kl-w0p05-tiny/v5-20260614-200052`
- 2026-06-15 repaired CE-only tiny smoke:
  `/data/CoordExp/.worktrees/geometry-aware-denoising-sft/temp/detection_teacher_forcing/output/compact_full_prefix_denoising_ce_only_tiny/smoke-compact-full-prefix-denoising-ce-only-tiny/v18-20260615-022333`
- 2026-06-15 repaired KL-on tiny smoke:
  `/data/CoordExp/.worktrees/geometry-aware-denoising-sft/temp/detection_teacher_forcing/output/compact_full_prefix_denoising_kl_w0p05_tiny/smoke-compact-full-prefix-denoising-kl-w0p05-tiny/v13-20260615-022423`
- ckpt-450 diagnostic inference root:
  `/data/CoordExp/outputs/infer/prefix_denoising_sft_v1/prefix_denoising_ckpt450_val200_free_decode_diagnostic_eval_8gpu`
- ckpt-450 training checkpoint:
  `/data/CoordExp/outputs/stage1_2b/detection_teacher_forcing/compact_full_prefix_denoising_kl_w0p05_2b_base_sorted_marker_bsz1x128_2epoch/compact-full-prefix-denoising-kl-w0p05-2b-base-sorted-marker-bsz1x128-2epoch/v25-20260615-124531/checkpoint-450`
- reference baseline path used in the root-cause analysis:
  `/data/CoordExp/outputs/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332`
- retained deprecated merged-full reference handle:
  `/data/CoordExp/output_remote_DEPRECATED_20260604/stage1_2b/coco_bbox_max60-hard_ce_soft_ce_w1_gate/epoch_4-from-base-2B/v0-20260227-050057/checkpoint-1332-merged-full`

## Manifest Provenance

The rows below are from `docs/history/worktree-union/2026-06-20/manifest.tsv`
filtered to `branch == codex/prefix-denoising-sft`. Branch OpenSpec, config,
and docs snapshot rows are historical branch provenance, not current contracts
or current behavior sources.

| classification | source_kind | branch | head | worktree | source_path | status | sha256 | same_path_in_main | content_paths_in_main | snapshot_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| content_present_elsewhere | branch-head | codex/prefix-denoising-sft | d08ef3ed63f6ffbd7d572afb162ea921b9b3a7ba | /data/CoordExp/.worktrees/geometry-aware-denoising-sft | docs/superpowers/specs/2026-06-14-prefix-denoising-sft-v1-design.md | A | c081faf3a5f49c48cf4262136f6d4cc907e4c14c7228df5c8ab6fa7e1fd682de | no | docs/history/superpowers/specs/2026-06-14-prefix-denoising-sft-v1-design.md | - |
| new_content_new_path | branch-head | codex/prefix-denoising-sft | d08ef3ed63f6ffbd7d572afb162ea921b9b3a7ba | /data/CoordExp/.worktrees/geometry-aware-denoising-sft | docs/superpowers/plans/2026-06-14-prefix-denoising-sft-v1.md | A | 273176f078b184f7bdc25cf792242be72c54a0e2e3d5d8c7d768c1889b486b00 | no | - | snapshots/273176f078b1/docs/superpowers/plans/2026-06-14-prefix-denoising-sft-v1.md |
| same_path_divergent_new_content | branch-head | codex/prefix-denoising-sft | d08ef3ed63f6ffbd7d572afb162ea921b9b3a7ba | /data/CoordExp/.worktrees/geometry-aware-denoising-sft | progress/directions/prefix_denoising_sft_v1.md | A | 78a4705d6b22ed29b82a1363bc2dd58ad5bcfcd3f79b5f378cfef975bd4bc62d | yes | - | snapshots/78a4705d6b22/progress/directions/prefix_denoising_sft_v1.md |
| same_path_divergent_new_content | branch-head | codex/prefix-denoising-sft | d08ef3ed63f6ffbd7d572afb162ea921b9b3a7ba | /data/CoordExp/.worktrees/geometry-aware-denoising-sft | progress/audits/2026-06-14_prefix_denoising_sft_v1_audit.md | A | be6db21eb37d0bd249689af49037630ae33c2d8f835224d4aec7941dd432127f | yes | - | snapshots/be6db21eb37d/progress/audits/2026-06-14_prefix_denoising_sft_v1_audit.md |
| same_path_identical | branch-head | codex/prefix-denoising-sft | d08ef3ed63f6ffbd7d572afb162ea921b9b3a7ba | /data/CoordExp/.worktrees/geometry-aware-denoising-sft | progress/diagnostics/2026-06-14_prefix_denoising_launch_health.md | A | 25ede3d7cc22fa92e1b6bf965e50bfaee7fc65ce4a32da04db6bae37d251c821 | yes | progress/diagnostics/2026-06-14_prefix_denoising_launch_health.md | - |
| same_path_identical | branch-head | codex/prefix-denoising-sft | d08ef3ed63f6ffbd7d572afb162ea921b9b3a7ba | /data/CoordExp/.worktrees/geometry-aware-denoising-sft | progress/diagnostics/2026-06-15_prefix_denoising_branch_isolation_repair.md | A | d4a12fd1885bf3ac3cb9798fddcb9b71aa6d952a3bde67c2a2d703f432785ea9 | yes | progress/diagnostics/2026-06-15_prefix_denoising_branch_isolation_repair.md | - |
| new_content_new_path | branch-head | codex/prefix-denoising-sft | d08ef3ed63f6ffbd7d572afb162ea921b9b3a7ba | /data/CoordExp/.worktrees/geometry-aware-denoising-sft | progress/diagnostics/2026-06-16_prefix_denoising_axis_sort_repair_negative_result.md | A | 688fa6f04eb57ce5c9d48e0178551e7188281df6150d786818afe4911812b915 | no | - | snapshots/688fa6f04eb5/progress/diagnostics/2026-06-16_prefix_denoising_axis_sort_repair_negative_result.md |
| new_content_new_path | worktree-dirty | codex/prefix-denoising-sft | d08ef3ed63f6ffbd7d572afb162ea921b9b3a7ba | /data/CoordExp/.worktrees/geometry-aware-denoising-sft | progress/diagnostics/2026-06-17_prefix_denoising_inert_objective_root_cause_analysis.md | ?? | ed610b284219c2e96fe90d2a4c446978b96ededc767c0ce3118417f847846871 | no | - | snapshots/ed610b284219/progress/diagnostics/2026-06-17_prefix_denoising_inert_objective_root_cause_analysis.md |

Expanded snapshot handles:

- `docs/history/worktree-union/2026-06-20/snapshots/273176f078b1/docs/superpowers/plans/2026-06-14-prefix-denoising-sft-v1.md`
- `docs/history/worktree-union/2026-06-20/snapshots/78a4705d6b22/progress/directions/prefix_denoising_sft_v1.md`
- `docs/history/worktree-union/2026-06-20/snapshots/be6db21eb37d/progress/audits/2026-06-14_prefix_denoising_sft_v1_audit.md`
- `docs/history/worktree-union/2026-06-20/snapshots/688fa6f04eb5/progress/diagnostics/2026-06-16_prefix_denoising_axis_sort_repair_negative_result.md`
- `docs/history/worktree-union/2026-06-20/snapshots/ed610b284219/progress/diagnostics/2026-06-17_prefix_denoising_inert_objective_root_cause_analysis.md`

## Source Map

- `progress/directions/prefix_denoising_sft_v1.md` - original idea, design rationale, objective assumptions, and launch checklist.
- `progress/diagnostics/2026-06-14_prefix_denoising_launch_health.md` - superseded tiny launch smoke and historical caveat.
- `progress/diagnostics/2026-06-15_prefix_denoising_branch_isolation_repair.md` - repaired launch-health evidence and verification handle.
- `progress/audits/2026-06-14_prefix_denoising_sft_v1_audit.md` - review findings that motivated repair and tightened implementation gates.
- `docs/history/superpowers/specs/2026-06-14-prefix-denoising-sft-v1-design.md` - historical design spec for the implemented V1 slice.
- `docs/history/superpowers/plans/2026-06-14-prefix-denoising-sft-v1.md` - historical implementation plan and code/config surface map.
- `docs/history/worktree-union/2026-06-20/manifest.tsv` - raw branch/worktree provenance index.
- `docs/history/worktree-union/2026-06-20/snapshots/688fa6f04eb5/progress/diagnostics/2026-06-16_prefix_denoising_axis_sort_repair_negative_result.md` - axis-sort repair negative result.
- `docs/history/worktree-union/2026-06-20/snapshots/ed610b284219/progress/diagnostics/2026-06-17_prefix_denoising_inert_objective_root_cause_analysis.md` - inert-objective root-cause analysis and next-probe recommendation.
- `progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md` - migration governance source for the OKF-style hierarchy and authority boundaries; not idea evidence.

## Next Action

Run or locate a matched denoising-OFF hard-CE LoRA control with the same compact
format, sorted marker prompt, train/eval data, adapter recipe, epoch/step
budget, and decode settings. Without that control, keep the idea active but do
not promote the V1 result into current training guidance.

## Sources

- `progress/directions/prefix_denoising_sft_v1.md`
- `progress/diagnostics/2026-06-14_prefix_denoising_launch_health.md`
- `progress/diagnostics/2026-06-15_prefix_denoising_branch_isolation_repair.md`
- `progress/audits/2026-06-14_prefix_denoising_sft_v1_audit.md`
- `docs/history/superpowers/specs/2026-06-14-prefix-denoising-sft-v1-design.md`
- `docs/history/superpowers/plans/2026-06-14-prefix-denoising-sft-v1.md`
- `docs/history/worktree-union/2026-06-20/manifest.tsv`
- `docs/history/worktree-union/2026-06-20/snapshots/688fa6f04eb5/progress/diagnostics/2026-06-16_prefix_denoising_axis_sort_repair_negative_result.md`
- `docs/history/worktree-union/2026-06-20/snapshots/ed610b284219/progress/diagnostics/2026-06-17_prefix_denoising_inert_objective_root_cause_analysis.md`
- `progress/explorations/2026-06-20_docs_progress_okf_upgrade_alignment.md`
