# State: CoordExp Stage-2 Pseudo-Positive Implementation

**Current Phase:** 5
**Current Phase Name:** Validation And Best-K Readiness
**Total Phases:** 5
**Current Plan:** 2
**Total Plans in Phase:** 3
**Status:** Ready to execute Phase 5 Plan 02
**Progress:** 85%
**Last Activity:** 2026-03-22
**Last Activity Description:** Retroactively reconciled Phases 2-4 as complete from the landed implementation, marked Phase 5 Plan 01 complete via passing regressions and OpenSpec validation, and left only real smoke / best-K runtime validation open.
**Paused At:** Starting Phase 5 plan 02

## Focus

Execute the remaining live validation work for the exact OpenSpec implementation slice:

- real pseudo-positive smoke execution through `src.sft`
- runtime confirmation of one-forward and coord-only behavior
- best-`K` comparison readiness across enabled rollout counts
- enabled `K=2` no-promotion runtime confirmation

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
| Phase 2-4 | These phases are already implemented and documented, so their GSD directories are recorded as retroactive summaries rather than forward plans | Keeps GSD aligned with the landed code instead of duplicating work |
| Phase 5 | Remaining work should focus on live smoke / best-`K` evidence instead of more code churn | The code/spec/docs surface is already green; the main unknown is runtime behavior on real infra |

## Blockers

None

## Performance Metrics

| Phase / Plan | Duration | Tasks | Files |
|--------------|----------|-------|-------|
| Phase 1 / Plan 01 | 20 min | 2 | 2 |
| Phase 1 / Plan 02 | 30 min | 2 | 2 |
| Phase 1 / Plan 03 | 25 min | 2 | 2 |
| Phase 2 / Plans 01-03 | retroactive | reconciled | 3 summaries |
| Phase 3 / Plans 01-03 | retroactive | reconciled | 3 summaries |
| Phase 4 / Plans 01-03 | retroactive | reconciled | 3 summaries |
| Phase 5 / Plan 01 | retroactive | validated | regression + OpenSpec |

## Session

**Last Date:** 2026-03-22T00:00:00Z
**Stopped At:** Ready to execute `.planning/phases/05-validation-and-best-k-readiness/05-02-PLAN.md`
**Resume File:** .planning/phases/05-validation-and-best-k-readiness/05-02-PLAN.md
