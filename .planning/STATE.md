# State: CoordExp Stage-2 Pseudo-Positive Implementation

**Current Phase:** 1
**Current Phase Name:** Config And Rollout Foundation
**Total Phases:** 5
**Current Plan:** 1
**Total Plans in Phase:** 3
**Status:** Ready to discuss and plan
**Progress:** 0%
**Last Activity:** 2026-03-22
**Last Activity Description:** Initialized project, completed brownfield research, defined requirements, and created the initial roadmap in the pseudo-positive implementation worktree.
**Paused At:** Initialization complete

## Focus

Finish Phase 1 planning for the exact OpenSpec implementation slice:

- typed `pseudo_positive` schema and invariants
- arbitrary-`K` rollout scheduling
- deterministic explorer identity
- anchor-drop / explorer-abort failure semantics
- disabled-path compatibility

## Decisions Made

| Phase | Summary | Rationale |
|-------|---------|-----------|
| Init | Scope is limited to the authored OpenSpec change `study-channel-b-pseudopositive-promotion` | Prevents a broad Stage-2 redesign and keeps planning reviewable |
| Init | The default pseudo-positive-authored profile uses `K=4`, while legacy `K=2` remains available for compatibility and control comparisons | Keeps the implementation target concrete without losing backward-compatible baselines |
| Init | Promotion uses support-rate semantics with a minimum `support_count >= 2` floor | Keeps arbitrary-`K` behavior comparable and ensures enabled `K=2` remains a no-promotion control |
| Init | Pseudo-positive supervision stays coord-only with anchor-owned geometry | Matches the authored v1 contract and minimizes text-side risk |
| Init | Dead-anchor negatives stay narrow and duplicate-like only | Preserves conservative supervision under incomplete GT |

## Blockers

None

## Performance Metrics

| Phase / Plan | Duration | Tasks | Files |
|--------------|----------|-------|-------|
| None yet | - | - | - |

## Session

**Last Date:** 2026-03-22T00:00:00Z
**Stopped At:** Ready for `$gsd-discuss-phase 1` or `$gsd-plan-phase 1`
**Resume File:** .planning/ROADMAP.md
