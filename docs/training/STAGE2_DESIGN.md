---
doc_id: docs.training.stage2-design
layer: docs
doc_type: design-note
status: historical
domain: training
summary: Retired Stage-2 AB/two-channel design history.
updated: 2026-05-25
---

# Stage-2 Design History

This page is historical context only. It describes the retired Stage-2
AB/two-channel era and must not be used as the active operator contract.

For current work, use:

- [docs/training/STAGE2_RUNBOOK.md](STAGE2_RUNBOOK.md)
- [docs/training/METRICS.md](METRICS.md)
- [`openspec/specs/stage2-rollout-correction/spec.md`](../../openspec/specs/stage2-rollout-correction/spec.md)

The active Stage-2 contract is Stage-2 rollout correction:

- `custom.trainer_variant: stage2_rollout_correction`
- top-level `stage2_rollout_correction`
- exactly one enabled `residual_set_correction` objective
- `application.preset: rollout_self_prefix`
- rollout-prefix roll-in plus GT/residual correction target IR

The retired AB/two-channel public names, per-channel scheduler, clean-prefix
teacher-forcing objectives, and A/B metric namespaces are removed from active
configs and fail fast when authored.

Historical rationale for retiring the old split lives in progress notes, for
example:

- [progress/diagnostics/2026-03-20_stage2_channel_a_self_context_iter_ablation.md](../../progress/diagnostics/2026-03-20_stage2_channel_a_self_context_iter_ablation.md)
