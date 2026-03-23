# Phase 2: Support-Rate Triage And Promotion - Context

**Gathered:** 2026-03-22
**Status:** Retroactively reconciled against landed implementation

<domain>
## Phase Boundary

Phase 2 owns the deterministic triage algorithm for pseudo-positive Channel-B: per-explorer anchor association, support-count and support-rate accounting, recovered-GT collapse, bucket assignment, and overlap-cluster winner selection. It stops short of treating coord-only loss projection and operator-facing docs as new work, even though the landed implementation already threads some metadata and metrics through the same runtime seam.

</domain>

<decisions>
## Implementation Decisions

### Reconciliation Outcome
- The current worktree already satisfies the Phase 2 roadmap scope, so this directory should capture retroactive summaries rather than new execution plans.
- Keep the authored promotion contract exactly as specified: unmatched anchors use geometry-first support-rate gating, promotion still requires `support_count >= 2` plus `support_rate >= 2/3`, and enabled `K=2` remains the no-promotion control.
- Treat recovered GT as explorer evidence collapsed by GT index, so one anchor-missed GT contributes at most one recovered FN object even when multiple explorers hit it.
- Resolve overlapping pseudo-positive candidates as connected components on anchor geometry, then pick one winner per component by highest support rate with anchor order as the deterministic tie-break.

</decisions>

<code_context>
## Existing Code Insights

### Normative Sources
- `openspec/changes/study-channel-b-pseudopositive-promotion/specs/stage2-ab-training/spec.md`
- `openspec/changes/study-channel-b-pseudopositive-promotion/tasks.md`
- `.planning/ROADMAP.md`
- `.planning/REQUIREMENTS.md`

### Landed Implementation Surfaces
- `src/trainers/stage2_two_channel/target_builder.py` now centralizes `_build_channel_b_triage`, bucket assignment, recovered-GT collapse, and connected-component promotion.
- `src/trainers/stage2_two_channel.py` consumes the triage result, carries the per-sample metadata, and aggregates the support-rate and recovered-GT counters used by later observability work.
- `tests/test_stage2_ab_training.py` already covers the Phase 2 contract at the trainer and target-builder seam, including the enabled `K=2` control, `K=4` support-rate promotion, GT-backed exclusions, shielded-anchor neutrality, and recovered-GT weighting.

### Verification Snapshot
- `conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py::test_channel_b_enabled_pseudo_positive_uses_k4_rollouts_and_keeps_zero_object_explorer tests/test_stage2_ab_training.py::test_channel_b_anchor_only_gt_hit_projects_anchor_gt_backed tests/test_stage2_ab_training.py::test_channel_b_shielded_anchor_stays_neutral_context tests/test_stage2_ab_training.py::test_channel_b_triage_clusters_pseudo_positive_candidates_by_support_rate tests/test_stage2_ab_training.py::test_channel_b_recovered_ground_truth_weight_multipliers_only_apply_to_recovered_tail_objects`
- `conda run -n ms python -m pytest -q tests/test_stage2_ab_training.py::test_channel_b_triage_enabled_k2_remains_no_promotion_control`

</code_context>

<specifics>
## Specific Ideas

- Record Phase 2 as complete once roadmap/state updates are allowed elsewhere; no remaining Phase 2-only implementation gap was found in the current worktree.

</specifics>

<deferred>
## Deferred Ideas

- Phase 3 still owns the higher-level one-forward loss semantics even though the current implementation already threads pseudo-positive selections into later supervision code.
- Phase 4 still owns the operator-surface and docs completion pass even though some triage metrics and metadata are already emitted.

</deferred>
