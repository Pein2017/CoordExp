# Cold-Start Route Frontier

> Reviewed 2026-08-14. These are time-sensitive eligibility and fallback
> priors, not numeric training data or a global model ranking. Verify the live
> picker, resolved identity, tools, quota, and client before use.

## Current Frontier

| Responsibility cell | Static fallback | Conditional peer or boundary |
| --- | --- | --- |
| `read_only_scout` | Luna/medium; use high only for deeper bounded exploration | Haiku/medium is a provider-diverse read-only peer. Neither owns writes, review, diagnosis, integration, or conclusions |
| `bounded_builder` | Terra/high for stable native contracts and integration | Sonnet/medium for small explicit work; Sonnet/high for ordinary 5--15-file features and refactors; Terra/medium only when the same deterministic verifier covers the complete contract |
| `semantic_builder` | Sol/high | Opus/medium for complex cross-module implementation or debugging; Opus/high for consequential coding. Own mathematical, algorithmic, forward/autograd, research-semantic, missing-invariant, and silent-correctness work directly |
| `lifecycle_builder` | Opus/high | Own framework lifecycle, compatibility, source archaeology, serialization, durability, and broad cross-module engineering directly |
| `semantic_reviewer` | Sol/xhigh | Opus/high for important fixed-target review; Opus/xhigh for mechanism, architecture, or root-cause diagnosis; not a builder retry rung |
| `lifecycle_reviewer` | Opus/xhigh | Opus/high may review a bounded fixed tree; use xhigh for provenance, compatibility, and engineering certification. Target drift invalidates the attempt |
| `major_decision` | Sol/max | Opus/xhigh or max and Fable/xhigh or max can advise or counterargue on major research, architecture, launch, promotion, compatibility, or irreversible decisions; lead/user retain authority |

For mixed semantic and lifecycle complexity, name one implementation owner by
the dominant uncertainty and use the other provider only as an independent
reviewer of the frozen result. Do not run two writers on the same semantic
surface.

## Claude 5 Role Map

| Route | Working role | Strength | Cost or risk | Prefer for |
| --- | --- | --- | --- | --- |
| Sonnet/medium | Efficient executor | Fast, economical, sufficient reasoning | Limited deep search | Small or medium implementation with explicit acceptance |
| Sonnet/high | Primary bounded implementer | More reliable across files | Below Opus on hidden dependencies | Ordinary 5--15-file features and routine refactors |
| Opus/medium | Complex implementer | Stronger problem solving with less wandering | Slower than Sonnet | Cross-module implementation and debugging |
| Opus/high | Consequential generalist | Strong coding, review, and reasoning | Higher latency | Important implementation, difficult debugging, fixed-target review |
| Opus/xhigh | Research-lane lead or principal adviser | Broad hypotheses, counterexamples, architecture and root-cause depth | Slow and prone to over-exploration | Research, architecture, mechanism diagnosis; the main lead still owns synthesis |
| Opus/max | Exceptional-depth adviser | Largest search and verification budget | Very slow and easy to over-research | Conflicting evidence, critical decisions, or an xhigh lane that cannot converge |

Treat effort as search and checking depth, not a substitute for the right model
family: `medium` favors efficiency, `high` adds formal checking, `xhigh`
broadens hypotheses, and `max` adds alternative explanations,
self-counterargument, and repeated verification. If Sonnet/high discovers
semantic or architectural uncertainty, reroute to Opus/medium or high instead
of escalating Sonnet to xhigh/max.

For workflow planning only, use this rough role analogy:

```text
Sonnet medium/high  ~= Luna xhigh / Terra medium
Opus medium         ~= Terra high
Opus high           ~= Terra max / Sol high
Opus xhigh          ~= Sol xhigh
Opus max            ~= Sol max
```

This is not a capability equality or global leaderboard. Live availability,
resolved identity, tools, verifier coverage, and task shape still define the
eligible set. These prose priors do not activate controller slots; add or
replace a route only through the reviewed controller workflow.

## Time And Effort

Treat model x effort as an interacting pair. Effort can deepen a suitable role
but cannot repair a role mismatch. `xhigh` is a research, review, or diagnosis
boundary; Sol/max, Opus/xhigh or max, and Fable/xhigh or max are principal-
adviser routes, not default builders. The controller does not automatically
substitute a Claude route when Sol/max is unavailable: it abstains. A lead may
deliberately dispatch one from the already-reviewed eligible set for an
independent research lane or counterargument.

Optimize:

```text
time to final acceptance = builder latency + correction + review
                         + runtime wait + lead intervention
```

When quality is conservatively comparable, prefer the shorter full path; use
spend only after time and lead burden. Surface/auth/client failures are
reporting-only health evidence, not model-quality or challenger-promotion
evidence.

## Adaptive State Boundary

The static table initializes route eligibility and cold-start fallbacks with
zero numeric pseudo-counts. `references/routing-state-seed.json` fixes seven
cells and three route slots per cell. The runtime controller may update only its
fixed summaries and choose among a lead-supplied eligible set.

- Ordinary accepted work monitors a route but cannot promote it.
- Challenger promotion requires repeated comparable evidence from multiple
  recent roots with frozen task and verifier contracts.
- Unknown aliases, model/client/tool changes, or unavailable routes require a
  fresh route epoch and zeroed slot, not inherited history.
- A verified silent false acceptance quarantines the route cell.
- Major-decision routes never auto-promote; learned state may inform advice but
  cannot acquire authority.

Replace absorbed evidence instead of appending session narratives here. Raw
receipts remain in sessions or the usage ledger; only evidence that changes a
hard eligibility boundary or cold-start fallback belongs in this file.
