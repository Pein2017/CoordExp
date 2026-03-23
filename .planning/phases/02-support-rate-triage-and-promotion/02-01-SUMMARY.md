---
phase: 02-support-rate-triage-and-promotion
plan: 01
status: retroactive-summary
subsystem: triage
tags: [stage2, channel-b, support-rate, recovered-gt, association]
requires:
  - 01-02
  - 01-03
provides:
  - Deterministic per-view anchor/explorer association under the existing one-to-one IoU rule
  - Per-anchor support-count and support-rate accounting over `valid_explorer_count`
  - Recovered-GT collapse by GT index with per-GT support counts and rates
affects: [phase-02-plan-02, phase-03, phase-04, stage2-runtime]
tech-stack:
  added: []
  patterns:
    - Independent anchor-to-explorer association for every explorer view
    - Recovered-GT aggregation keyed by GT index rather than by explorer hit
key-files:
  created: []
  modified:
    - src/trainers/stage2_two_channel/target_builder.py
    - src/trainers/stage2_two_channel.py
    - tests/test_stage2_ab_training.py
key-decisions:
  - "Use `valid_explorer_count` as the support-rate denominator so rates stay comparable across arbitrary `K`."
  - "Count recovered GT support by unique GT index and collapse repeated explorer hits before FN injection."
patterns-established:
  - "Phase 2 triage lives in `target_builder.py`, while the trainer only consumes the structured triage result."
  - "Recovered GT remains a single object-level fact even when several explorers independently recover it."
requirements-completed: [TRIA-01, TRIA-02]
completed: 2026-03-22
verification:
  - "Passing targeted tests on 2026-03-22: `test_channel_b_enabled_pseudo_positive_uses_k4_rollouts_and_keeps_zero_object_explorer`, `test_channel_b_anchor_only_gt_hit_projects_anchor_gt_backed`, `test_channel_b_recovered_ground_truth_weight_multipliers_only_apply_to_recovered_tail_objects`."
---

# Phase 2 Plan 01 Summary

**Retroactive reconciliation: support accounting, recovered-GT aggregation, and deterministic per-view association are already landed**

## Accomplishments
- `_build_channel_b_triage` associates the anchor view to each explorer view independently with the existing deterministic one-to-one IoU matcher, then computes support counts only for unmatched anchor objects whose explorer counterpart is also unmatched and not in GT-backed conflict.
- The same triage path computes support rates over `valid_explorer_count`, which keeps arbitrary-`K` behavior comparable instead of diluting or shrinking denominators opportunistically.
- Recovered GT is aggregated from explorer matches by GT index, so multiple explorer hits collapse to one recovered object with explicit support counts and support rates.
- `Stage2ABTrainingTrainer._prepare_batch_inputs_b` consumes those triage outputs directly, preserving one recovered FN object per GT while making the metadata available to later supervision and observability phases.

## Evidence
- `src/trainers/stage2_two_channel/target_builder.py` exposes `anchor_support_counts`, `anchor_support_rates`, `recovered_gt_indices`, `recovered_gt_support_counts`, and `recovered_gt_support_rates` in `_ChannelBTriageResult`.
- `src/trainers/stage2_two_channel.py` threads the triage result into per-sample metadata and additive recovered-GT metrics.
- `tests/test_stage2_ab_training.py` already exercises the enabled `K=4` path, GT-backed anchor projection, and recovered-GT weighting behavior.

## Retroactive Notes
- No remaining implementation gap was found for Plan 01. This summary is replacing a forward plan because the worktree already contains the intended implementation.

---
*Phase: 02-support-rate-triage-and-promotion*
*Completed: 2026-03-22*
