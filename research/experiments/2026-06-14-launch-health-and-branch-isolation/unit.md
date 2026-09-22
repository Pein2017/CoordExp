---
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-06-14-launch-health-and-branch-isolation
topic: prefix-denoising-sft
status: complete
title: Prefix Denoising Launch Health And Branch Isolation
description: Tiny launch-health and branch-isolation repair evidence for prefix-denoising SFT V1.
tags: [stage1, prefix-denoising, launch-health, tiny]
updated: 2026-06-20
---

# Prefix Denoising Launch Health And Branch Isolation

## Scope

This note covers tiny launch-health evidence only. It verifies wiring,
branch-isolated loss computation, static packing metadata, metrics, and runtime
payloads for V1; it is not rollout, eval-quality, or exposure-bias evidence.

## Historical Smoke

The 2026-06-14 CE-only and KL-on smokes launched paired clean/noisy views and
completed 2/2 steps. They emitted standard `llm_loss`, token top1/top5 monitors,
branch CE metrics, and local-window KL diagnostics.

The historical artifact roots are:

- `/data/CoordExp/.worktrees/geometry-aware-denoising-sft/temp/detection_teacher_forcing/output/compact_full_prefix_denoising_ce_only_tiny/smoke-compact-full-prefix-denoising-ce-only-tiny/v10-20260614-195716`
- `/data/CoordExp/.worktrees/geometry-aware-denoising-sft/temp/detection_teacher_forcing/output/compact_full_prefix_denoising_kl_w0p05_tiny/smoke-compact-full-prefix-denoising-kl-w0p05-tiny/v5-20260614-200052`

## Supersession

The 2026-06-14 smoke is superseded as current launch evidence because it
predates the branch-isolated forward repair. It remains useful as historical
evidence that the first paired data/runtime path could launch.

## Repair Evidence

The 2026-06-15 repair replayed clean and noisy branch segments as separate model
forwards before computing CE and optional KL. The repair evidence recorded:

- CPU regression and integration slice: `285 passed in 2.95s`.
- Focused provenance slice: `24 passed in 1.58s`.
- Compile check: exit 0.
- `git diff --check`: exit 0.
- CE-only and KL-on cfg-only checks: both returned `status=ok`.
- repaired CE-only tiny GPU smoke: completed 2/2 steps.
- repaired KL-on tiny GPU smoke: completed 2/2 steps.

## Artifact Handles

- repaired CE-only smoke:
  `/data/CoordExp/.worktrees/geometry-aware-denoising-sft/temp/detection_teacher_forcing/output/compact_full_prefix_denoising_ce_only_tiny/smoke-compact-full-prefix-denoising-ce-only-tiny/v18-20260615-022333`
- repaired KL-on smoke:
  `/data/CoordExp/.worktrees/geometry-aware-denoising-sft/temp/detection_teacher_forcing/output/compact_full_prefix_denoising_kl_w0p05_tiny/smoke-compact-full-prefix-denoising-kl-w0p05-tiny/v13-20260615-022423`

Both repaired smoke artifacts recorded:

```text
runtime.prefix_denoising.enabled=true
runtime.prefix_denoising.packing_enabled=true
runtime.prefix_denoising.packing_mode=static
runtime.prefix_denoising.dataset.train.source_rows=4
runtime.prefix_denoising.dataset.train.eligible_rows=4
runtime.prefix_denoising.dataset.train.skipped_rows=0
```

## Interpretation Limit

The repaired branch was healthy for V1 launch wiring. The evidence does not
show that prefix denoising improves free decode, localization, recall,
coordinate robustness, or downstream AP.

## Research Unit Closeout

Observed: repaired branch-isolated CE-only and KL-on tiny GPU smokes completed
2/2 steps and emitted the expected runtime and metric payloads.

Supported: the V1 launch path was healthy at tiny scope after branch-isolation
repair.

Not supported yet: rollout robustness, localization improvement, recall
improvement, coordinate-prefix robustness, or downstream AP benefit.

Next decider: matched denoising-OFF hard-CE LoRA control and evaluation-quality
probe before any training-guidance promotion.

Promotion decision: keep as research evidence; do not promote to OpenSpec.

Manifest handles:

- `progress/diagnostics/2026-06-14_prefix_denoising_launch_health.md`: `same_path_identical`, `branch-head`, `sha256=25ede3d7cc22fa92e1b6bf965e50bfaee7fc65ce4a32da04db6bae37d251c821`
- `progress/diagnostics/2026-06-15_prefix_denoising_branch_isolation_repair.md`: `same_path_identical`, `branch-head`, `sha256=d4a12fd1885bf3ac3cb9798fddcb9b71aa6d952a3bde67c2a2d703f432785ea9`

## Sources

- `progress/diagnostics/2026-06-14_prefix_denoising_launch_health.md`
- `progress/diagnostics/2026-06-15_prefix_denoising_branch_isolation_repair.md`
- `docs/history/worktree-union/2026-06-20/manifest.tsv`
- `same_path_identical branch-head sha256=25ede3d7cc22fa92e1b6bf965e50bfaee7fc65ce4a32da04db6bae37d251c821`
- `same_path_identical branch-head sha256=d4a12fd1885bf3ac3cb9798fddcb9b71aa6d952a3bde67c2a2d703f432785ea9`
