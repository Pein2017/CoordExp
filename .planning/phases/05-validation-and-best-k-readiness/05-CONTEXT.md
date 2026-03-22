# Phase 5: Validation And Best-K Readiness - Context

**Gathered:** 2026-03-22
**Status:** Active phase with partial retroactive completion

<domain>
## Phase Boundary

Phase 5 owns the validation pass that must happen on top of the already-landed
implementation: full regression confirmation, real smoke execution, and the
first best-`K`-ready comparison workflow. Unlike Phases 2–4, this phase still
has genuine unfinished work because the code/test/docs surface is ready but the
live training runtime has not yet been exercised in this session.

</domain>

<decisions>
## Implementation Decisions

### Reconciliation Outcome
- Plan 05-01 is already complete retroactively because the targeted config,
  trainer, and profile regressions all pass and the OpenSpec change validates.
- Plans 05-02 and 05-03 remain open because they require live smoke execution
  and best-`K` comparison evidence rather than more code changes.
- Keep the remaining work focused on real runtime validation instead of further
  speculative refactoring.

</decisions>

<code_context>
## Existing Code Insights

### Normative Sources
- `openspec/changes/study-channel-b-pseudopositive-promotion/tasks.md`
- `.planning/ROADMAP.md`
- `.planning/STATE.md`

### Already-Completed Validation Surfaces
- `tests/test_stage2_ab_training.py`
- `tests/test_stage2_ab_config_contract.py`
- `tests/test_stage2_ab_profile_leaf_contract.py`
- `openspec validate study-channel-b-pseudopositive-promotion --type change --strict --json --no-interactive`

### Remaining Runtime Surfaces
- `configs/stage2_two_channel/smoke/b_majority_coco1024_pseudo_positive_4steps.yaml`
- `configs/stage2_two_channel/prod/ab_mixed_coco1024_bmajority_channel_b_pseudo_positive.yaml`
- `src.sft` runtime with the repo-standard `conda run -n ms python -m src.sft --config ...` entrypoint

</code_context>

<specifics>
## Specific Ideas

- Treat Phase 5 as the only still-open phase. The immediate next actions are
  smoke execution and best-`K` evidence capture, not more contract changes.

</specifics>

<deferred>
## Deferred Ideas

- Broader semantic gating or wider dead-negative redesign should stay deferred
  until the real pseudo-positive smoke and first best-`K` comparison are stable.

</deferred>
