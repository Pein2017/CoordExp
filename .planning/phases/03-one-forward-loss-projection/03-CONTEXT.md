# Phase 3: One-Forward Loss Projection - Context

**Gathered:** 2026-03-22
**Status:** Retroactively reconciled against landed implementation

<domain>
## Phase Boundary

Phase 3 owns how the selected pseudo-positive anchors are projected into the
existing one-forward teacher-forced objective surface. That includes
coord-only pseudo-positive supervision, anchor-owned geometry targets, and the
narrow duplicate-like dead-anchor suppression rule. It does not own the
support-rate triage logic itself or the operator-facing docs/profile surface.

</domain>

<decisions>
## Implementation Decisions

### Reconciliation Outcome
- The current worktree already satisfies the Phase 3 roadmap scope, so this
  directory should capture retroactive summaries rather than forward plans.
- Pseudo-positive winners stay in the edited anchor prefix and reuse the
  existing bbox-group weight carrier instead of introducing a new objective
  module.
- Pseudo-positive supervision remains coord-only: bbox geometry, bbox-size
  auxiliary, and coord regularization only.
- Dead anchors remain out of the final target, and only duplicate-like
  boundary-local dead branches create explicit suppression targets.

</decisions>

<code_context>
## Existing Code Insights

### Normative Sources
- `openspec/changes/study-channel-b-pseudopositive-promotion/specs/channel-b-lightweight-pseudopositive-v1/spec.md`
- `openspec/changes/study-channel-b-pseudopositive-promotion/specs/teacher-forcing-unified-loss-registry/spec.md`
- `openspec/changes/study-channel-b-pseudopositive-promotion/tasks.md`
- `.planning/ROADMAP.md`

### Landed Implementation Surfaces
- `src/trainers/stage2_two_channel/target_builder.py` now inserts
  pseudo-positive winners into `prefix_bbox_groups` with anchor-owned
  `gt_bins` and explicit per-group `weight`.
- `src/trainers/teacher_forcing/modules/bbox_geo.py`,
  `bbox_size_aux.py`, and `coord_reg.py` already consume those weights through
  the existing bbox-group / coord-slot pipeline.
- `tests/test_stage2_ab_training.py` covers coord-only pseudo-positive target
  projection, bbox-size-aux weighting, coord-slot weighting, and the
  duplicate-like dead-anchor suppression filter.

</code_context>

<specifics>
## Specific Ideas

- Record Phase 3 as complete once roadmap/state updates are allowed elsewhere;
  no remaining Phase 3-only implementation gap was found in the current
  worktree.

</specifics>

<deferred>
## Deferred Ideas

- Phase 4 still owns the operator-visible interpretation of the emitted losses,
  metrics, and YAML profiles.
- Phase 5 still owns the real smoke run and the first best-`K` comparison on
  live training infrastructure.

</deferred>
