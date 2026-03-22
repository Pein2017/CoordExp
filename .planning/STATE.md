# State: CoordExp Stage-2 Pseudo-Positive Implementation

**Current Phase:** 2
**Current Phase Name:** Support-Rate Triage And Promotion
**Total Phases:** 5
**Current Plan:** 1
**Total Plans in Phase:** 3
**Status:** Ready to implement Phase 2 Plan 01
**Progress:** 35%
**Last Activity:** 2026-03-22
**Last Activity Description:** Completed Phase 1 by adding pseudo-positive config guardrails, arbitrary-`K` rollout scheduling, enabled-path failure semantics, and passing Stage-2 config/training regression suites.
**Paused At:** Starting Phase 2 plan 01

## Focus

Implement the support-rate triage and promotion layer for the exact OpenSpec implementation slice:

- per-explorer anchor/explorer association
- support-count and support-rate accounting
- recovered-GT aggregation across explorers
- deterministic pseudo-positive promotion

## Decisions Made

| Phase | Summary | Rationale |
|-------|---------|-----------|
| Init | Scope is limited to the authored OpenSpec change `study-channel-b-pseudopositive-promotion` | Prevents a broad Stage-2 redesign and keeps planning reviewable |
| Init | The default pseudo-positive-authored profile uses `K=4`, while legacy `K=2` remains available for compatibility and control comparisons | Keeps the implementation target concrete without losing backward-compatible baselines |
| Init | Promotion uses support-rate semantics with a minimum `support_count >= 2` floor | Keeps arbitrary-`K` behavior comparable and ensures enabled `K=2` remains a no-promotion control |
| Init | Pseudo-positive supervision stays coord-only with anchor-owned geometry | Matches the authored v1 contract and minimizes text-side risk |
| Init | Dead-anchor negatives stay narrow and duplicate-like only | Preserves conservative supervision under incomplete GT |
| Phase 1 | Legacy `rollout/explorer/*` metrics become mean-over-valid-explorer-view summaries under arbitrary `K` | Preserves operator continuity while extending runtime evidence width |
| Phase 1 | Enabled malformed anchors are dropped, while malformed explorers abort the step | Keeps multi-explorer support denominators honest without broadening fallback semantics |

## Blockers

None

## Performance Metrics

| Phase / Plan | Duration | Tasks | Files |
|--------------|----------|-------|-------|
| Phase 1 / Plan 01 | 20 min | 2 | 2 |
| Phase 1 / Plan 02 | 30 min | 2 | 2 |
| Phase 1 / Plan 03 | 25 min | 2 | 2 |

## Session

**Last Date:** 2026-03-22T00:00:00Z
**Stopped At:** Ready to implement Phase 2 support-rate triage plans
**Resume File:** .planning/ROADMAP.md
