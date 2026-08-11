# Cold-Start Route Frontier

> Reviewed 2026-08-09. These are time-sensitive eligibility and fallback
> priors, not numeric training data or a global model ranking. Verify the live
> picker, resolved identity, tools, quota, and client before use.

## Current Frontier

| Responsibility cell | Static fallback | Conditional peer or boundary |
| --- | --- | --- |
| `read_only_scout` | Luna/medium; use high only for deeper bounded exploration | Haiku/medium is a provider-diverse read-only peer. Neither owns writes, review, diagnosis, integration, or conclusions |
| `bounded_builder` | Terra/high for stable native contracts and integration | Sonnet/high for bounded engineering; Terra/medium only when the same deterministic verifier covers the complete contract |
| `semantic_builder` | Sol/high | Own mathematical, algorithmic, forward/autograd, research-semantic, missing-invariant, and silent-correctness work directly; do not cheap-first through a bounded builder |
| `lifecycle_builder` | Opus/high | Own framework lifecycle, compatibility, source archaeology, serialization, durability, and broad cross-module engineering directly |
| `semantic_reviewer` | Sol/xhigh | Fixed-target mechanism or root-cause diagnosis; not a builder retry rung |
| `lifecycle_reviewer` | Opus/xhigh | Fixed-tree lifecycle, provenance, compatibility, and engineering certification; target drift invalidates the attempt |
| `major_decision` | Sol/max | Fable/xhigh or max can advise or counterargue on major research, architecture, launch, promotion, compatibility, or irreversible decisions; lead/user retain authority |

For mixed semantic and lifecycle complexity, name one implementation owner by
the dominant uncertainty and use the other provider only as an independent
reviewer of the frozen result. Do not run two writers on the same semantic
surface.

## Time And Effort

Treat model x effort as an interacting pair. Effort can deepen a suitable role
but cannot repair a role mismatch. `xhigh` is a review/diagnosis boundary;
Sol/max and Fable/xhigh or max are decision-adviser routes, not default
builders. The controller does not automatically substitute Fable when Sol/max
is unavailable: it abstains. A lead may deliberately dispatch a Fable adviser
from the already-reviewed eligible set for an independent counterargument.

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
