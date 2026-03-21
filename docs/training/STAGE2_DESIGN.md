# Full Idea v2 (Deprecated Historical Design Note)

This document is retained only as a historical marker. The stable Stage-2
contract no longer includes Channel-A self-context iteration or iterative
Channel-A provenance groups.

## Current Design Summary

- Channel-A: one GT-anchored teacher-forced forward
- Channel-B: rollout-aligned clean-prefix supervision
- Canonical Channel-A metrics: `loss/text/*`, `loss/coord/*`, `coord_diag/*`
- Canonical Channel-B coord metrics: `loss/B_coord/*`, `coord_diag/B/*`
- Canonical presets: `anchor_text_only`, `anchor_only`, `rollout_only`

For the active contract, prefer:

- `openspec/specs/`
- `configs/stage2_two_channel/`
- `docs/training/STAGE2_RUNBOOK.md`

For the removal rationale, see:

- `progress/diagnostics/stage2_channel_a_self_context_iter_ablation_2026-03-20.md`
