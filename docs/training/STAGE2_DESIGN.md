---
doc_id: docs.training.stage2-design
layer: docs
doc_type: design-note
status: historical
domain: training
summary: Historical design note for the active single-pass Channel-A plus clean-prefix Channel-B Stage-2 contract.
updated: 2026-04-03
---

# Stage-2 Design History

This page is historical context, not the primary operator runbook.

The active Stage-2 contract removed Channel-A self-context iteration and keeps
one stable design frame:

- Channel-A: one GT-anchored teacher-forced forward
- Channel-B: rollout-aligned clean-prefix supervision
- Canonical Channel-A families: `loss/text/*`, `loss/coord/*`, `coord_diag/*`
- Canonical Channel-B families: `loss/B_rollout_text/*`, `loss/B_coord/*`, `coord_diag/B/*`, `dup/raw/*`, and `stage2_ab/channel_b/dup/N_*`

## Current Runtime Ownership

The public entrypoints remain stable, but the implementation is now split
across narrower runtime seams:

- training/bootstrap entrypoint:
  - `src/sft.py`
  - `src/bootstrap/`
- Stage-2 two-channel trainer surface:
  - `src/trainers/stage2_two_channel.py`
  - `src/trainers/stage2_two_channel/`
- shared rollout runtime:
  - `src/trainers/rollout_runtime/`
- server-mode launcher:
  - `scripts/train_stage2.sh`
  - `src/launchers/stage2_vllm_server.py`

## Prefer These For Active Work

- [docs/training/STAGE2_RUNBOOK.md](STAGE2_RUNBOOK.md)
- [docs/training/METRICS.md](METRICS.md)
- [`openspec/specs/stage2-ab-training/spec.md`](../../openspec/specs/stage2-ab-training/spec.md)
- [`openspec/specs/runtime-architecture-refactor-program/spec.md`](../../openspec/specs/runtime-architecture-refactor-program/spec.md)

## Historical Rationale

The removal rationale lives in:

- [progress/diagnostics/stage2_channel_a_self_context_iter_ablation_2026-03-20.md](../../progress/diagnostics/stage2_channel_a_self_context_iter_ablation_2026-03-20.md)
