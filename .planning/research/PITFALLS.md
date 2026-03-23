# Pitfalls Research

**Domain:** Brownfield implementation of arbitrary-K pseudo-positive Stage-2 training in CoordExp
**Researched:** 2026-03-22
**Confidence:** HIGH

## Critical Pitfalls

### Pitfall 1: Silent explorer-denominator drift

**What goes wrong:**
Explorer failures or malformed explorer views silently reduce the denominator used for support rates, making `best-K` comparisons meaningless.

**Why it happens:**
The current codebase is singular-explorer-shaped, so it is tempting to skip failed explorers and continue with “whatever valid explorers remain.”

**How to avoid:**
Keep the authored fixed-denominator rule: malformed/missing explorer preparation under enabled pseudo-positive mode aborts the optimizer step, while zero accepted-clean explorer output still counts as zero support.

**Warning signs:**
- Rates look higher at larger `K` only because fewer explorers actually counted
- Step metrics do not clearly distinguish empty explorers from failed explorers
- Operators cannot reconcile `num_rollouts` with recorded support denominators

**Phase to address:**
Phase 2

---

### Pitfall 2: Pseudo-positive leakage into text-side supervision

**What goes wrong:**
Pseudo-positive anchors accidentally create desc CE or matched-prefix structure CE, widening the supervision contract beyond the authored v1.

**Why it happens:**
The teacher-forcing objective pipeline already carries mixed text and coord surfaces, so adding pseudo-positive groups in the wrong place can leak into existing masks.

**How to avoid:**
Keep pseudo-positive projection strictly through bbox-group weights that only scale `bbox_geo`, `coord_reg`, and `bbox_size_aux`; add tests that assert no pseudo-positive text-side supervision appears.

**Warning signs:**
- `token_ce` masks grow when pseudo-positive is enabled
- pseudo-positive examples affect desc/tail CE counters
- implementation adds new text-side atoms or module-wide pseudo-positive flags

**Phase to address:**
Phase 3

---

### Pitfall 3: Over-promoting near-duplicate unmatched anchors

**What goes wrong:**
Multiple overlapping unmatched anchor objects all get promoted, leading to duplicate-like positive signals and inflated object counts.

**Why it happens:**
Support-rate promotion alone does not prevent multiple overlapping anchors from independently satisfying the threshold.

**How to avoid:**
Build connected components on anchor-side pseudo-positive candidates using `duplicate_iou_threshold`, keep exactly one winner per component, and demote the rest to `shielded_anchor`.

**Warning signs:**
- pseudo-positive selected counts scale faster than distinct visual objects
- duplicate-like dead-anchor suppression targets spike immediately after enabling pseudo-positive
- qualitative outputs show clustered repeated boxes in crowded scenes

**Phase to address:**
Phase 3

---

### Pitfall 4: Breaking compatibility metrics while adding new rates

**What goes wrong:**
Existing dashboards/tests that depend on `train/triage/unlabeled_consistent_count` or `rollout/explorer/*` become ambiguous or break under arbitrary `K`.

**Why it happens:**
The live runtime is single-explorer-shaped, so new arbitrary-`K` semantics can accidentally repurpose old keys without documenting or testing the new meaning.

**How to avoid:**
Preserve `unlabeled_consistent_count` as total shielded-anchor count, define legacy explorer metrics as mean-over-valid-explorer-view summaries, and add explicit regression tests plus doc updates.

**Warning signs:**
- tests begin asserting different meanings for the same key
- metrics docs lag the implementation
- operators cannot compare `K=2` and `K=4` runs using the same dashboard family

**Phase to address:**
Phase 4

---

### Pitfall 5: Regressing the legacy `K=2` baseline

**What goes wrong:**
The opt-in feature subtly changes existing `K=2` runs or makes legacy disabled-path configs fail.

**Why it happens:**
Schema, rollout scheduling, and triage code are centralized, so it is easy for arbitrary-`K` assumptions to bleed into the default path.

**How to avoid:**
Guard all new behavior behind `pseudo_positive.enabled`, require dedicated config-contract tests, and keep enabled `K=2` as a no-promotion control rather than a special-cased shortcut.

**Warning signs:**
- existing prod/smoke configs change behavior with the feature disabled
- `num_rollouts=2` enabled runs create pseudo-positive promotions
- stage2 baseline tests regress in code paths unrelated to pseudo-positive configs

**Phase to address:**
Phase 1 and Phase 5

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Add ad hoc lists and dicts in `stage2_two_channel.py` without typed carrier updates | Faster first patch | Makes triage metadata and observability brittle and harder to reason about | Acceptable only for temporary local scaffolding before finalizing metadata carriers. |
| Encode pseudo-positive behavior through hard-coded module name checks | Quick loss routing | Makes future registry changes fragile and obscures the intended contract | Never for the final implementation. |
| Treat abort incidents as ordinary `train/triage/*` step metrics | Looks convenient for dashboards | Conflicts with hard-abort semantics and produces unreliable telemetry | Never; use failure telemetry/reporting instead. |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Config schema <-> YAML profiles | Add config keys in YAML before schema or docs are ready | Extend `schema.py` first, then author explicit YAML profiles once validation passes. |
| Runtime <-> metrics | Emit new rate keys without pinning numerator/denominator semantics | Define exact metric names and meanings in code, tests, and docs together. |
| Target builder <-> objective modules | Add pseudo-positive groups without proving one-forward realization | Keep all losses derived from the single clean teacher-forced forward and add focused tests. |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Linear explorer fan-out without careful reuse | Step time rises sharply as `K` increases | Reuse the same per-view accepted-clean path and keep first implementation validation to small `K` ablations | Shows up immediately at `K > 2` in smoke runs |
| Excessive per-view logging payloads | Logging grows faster than useful signal | Keep monitor payloads optional and rely on canonical per-sample metadata + aggregate metrics | Breaks artifact readability at larger `K` |
| Duplicate suppression built over all dead anchors | Many dead-target rows and unstable loss magnitude | Keep suppression boundary-local and duplicate-like only | Becomes obvious in dense scenes with many unsupported anchors |

## Security / Reproducibility Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Using hidden fallback behavior for malformed anchor/explorer views | Non-reproducible experiments and misleading metrics | Keep failure/drop behavior explicit and tested. |
| Renaming knobs with version suffixes (`v*`) | Long-lived config drift and alias confusion | Keep YAML/code-facing names versionless and reject `v*` aliases in config tests. |
| Broadening semantics without spec/doc updates | Operators cannot trust runs or audits | Re-run OpenSpec validation and update runbook/metrics docs with the implementation. |

## UX / Operator Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| A config enables pseudo-positive but there is no obvious sign what changed | Hard to compare runs and debug behavior | Add explicit pseudo-positive metrics, YAML profiles, and clear runbook documentation. |
| `best-K` ablation runs report incomparable metrics | Researchers cannot tell whether larger `K` helped | Use rate metrics plus stable legacy explorer summaries and document the aggregation semantics. |
| Failure telemetry is buried or absent on aborted steps | Operators see a failed run with no clear root cause | Route explorer-abort observability through explicit failure telemetry/reporting rather than ordinary finalized step metrics. |

## Sources

- `/data/CoordExp/.worktrees/stage2b-channelb-supervision-research-proposals/.planning/PROJECT.md`
- `/data/CoordExp/.worktrees/stage2b-channelb-supervision-research-proposals/openspec/changes/study-channel-b-pseudopositive-promotion/proposal.md`
- `/data/CoordExp/.worktrees/stage2b-channelb-supervision-research-proposals/openspec/changes/study-channel-b-pseudopositive-promotion/tasks.md`
- `/data/CoordExp/.worktrees/stage2b-channelb-supervision-research-proposals/openspec/changes/study-channel-b-pseudopositive-promotion/specs/stage2-ab-training/spec.md`
- `/data/CoordExp/.worktrees/stage2b-channelb-supervision-research-proposals/openspec/changes/study-channel-b-pseudopositive-promotion/specs/trainer-metrics-components/spec.md`

---
*Pitfalls research for: CoordExp Stage-2 pseudo-positive implementation*
*Researched: 2026-03-22*
