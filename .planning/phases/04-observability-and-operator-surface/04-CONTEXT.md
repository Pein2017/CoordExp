# Phase 4: Observability And Operator Surface - Context

**Gathered:** 2026-03-22
**Status:** Retroactively reconciled against landed implementation

<domain>
## Phase Boundary

Phase 4 owns the operator-visible meaning of the new arbitrary-`K`
pseudo-positive implementation: metadata carriers, aggregate metrics,
compatibility semantics for legacy metric names, docs, and authored YAML
profiles. It does not own the core triage or supervision behavior itself.

</domain>

<decisions>
## Implementation Decisions

### Reconciliation Outcome
- The current worktree already satisfies the Phase 4 roadmap scope, so this
  directory should capture retroactive summaries rather than forward plans.
- Keep new pseudo-positive counters under `train/triage/*` and preserve
  `rollout/explorer/*` as compatibility metrics interpreted as means over valid
  explorer views.
- Preserve `train/triage/unlabeled_consistent_count` as the shielded-anchor
  total rather than redefining it around promoted pseudo-positives.
- Expose authored pseudo-positive YAML through explicit prod/smoke leaf
  profiles instead of mutating unrelated baselines in place.

</decisions>

<code_context>
## Existing Code Insights

### Normative Sources
- `openspec/changes/study-channel-b-pseudopositive-promotion/specs/trainer-metrics-components/spec.md`
- `openspec/changes/study-channel-b-pseudopositive-promotion/tasks.md`
- `.planning/ROADMAP.md`

### Landed Implementation Surfaces
- `src/trainers/stage2_two_channel.py` emits the new per-sample metadata and
  aggregate triage counters.
- `src/trainers/stage2_two_channel/types.py` carries the expanded Channel-B
  metadata contract.
- `docs/training/METRICS.md` and `docs/training/STAGE2_RUNBOOK.md` now explain
  the arbitrary-`K` and pseudo-positive operator surface.
- `configs/stage2_two_channel/prod/ab_mixed_coco1024_bmajority_channel_b_pseudo_positive.yaml`
  and the matching smoke profile author the explicit enabled `K=4` surface.

</code_context>

<specifics>
## Specific Ideas

- Record Phase 4 as complete once roadmap/state updates are allowed elsewhere;
  no remaining Phase 4-only implementation gap was found in the current
  worktree.

</specifics>

<deferred>
## Deferred Ideas

- Phase 5 still owns the real smoke run, the first best-`K` comparison, and the
  final live-runtime validation pass.

</deferred>
