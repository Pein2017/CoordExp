---
title: Image2299 canonical five-tie protected-null sentinel
type: investigation
role: research-unit
authority: non_normative_research
unit_id: 2026-08-31-image2299-canonical-five-tie-protected-null-sentinel
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: fresh_cold_augmented_greedy_46_owner_success
updated: 2026-08-31
---

# Image2299 canonical five-tie protected-null sentinel

## Disposition

The canonical sentinel succeeded and closes this unit. The fresh-cold
composed-augmented model ordinarily greedily emitted 38 persons, 8 ties, and
46 unique strict owners with zero debt/counters and natural row-aligned EOS.
The result is single-image composed-model evidence only; it does not justify
base-model, transfer, or general enumeration claims.

Authoritative receipt: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-31-image2299-canonical-five-tie-protected-null-sentinel/20260831T-image2299-canonical-five-tie-protected-null-sentinel-v1/receipt.json`.
Receipt SHA-256: `e475bd7e9a3f21f5a8f83c77fd0fd2fd0161b94a2e7da92546eb109cd44e06a7`.
See [results](results.md) for the bounded evidence and claim boundary.

## Frozen question and contrast

From the fresh-cold 41-owner augmented model, can one additive protected-null
child output residual make the canonical missing-tie suffix ordinarily greedily
produce all 46 strict owners?

This is the production-shaped vertical for the planned 120-permutation screen.
It tests that screen's mandatory canonical sentinel first. A success stops the
broader screen as unnecessary. A negative rejects only this order and exact-row
scaffold; it does not establish absent tie support and routes to the frozen
120-order comparison.

The strongest alternative is ordering/scaffold mismatch: no existing natural,
sampled, or controlled generated route contains a strict alias for any missing
tie, so canonical target rows may be poor autoregressive states even though
their static owners are valid.

## Immutable parent and target

- Parent success receipt:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-30-image2299-dyadic-norm-release-distillation/20260830T-image2299-dyadic-norm-release-distillation-v1/receipt.json`,
  SHA-256
  `9811bdb632c5c564607537fb80892c38b66a584b31a09da97fa887b6fc90a5e6`.
- Parent route SHA-256:
  `c89c6f9900303227cf781caf4a39dc203a09a53b0d319930754cda184529530e`,
  370 tokens, 38 persons plus ties `gt:2299:{10,12,44}`, zero debt.
- Parent residual payload SHA-256:
  `10455e0587bfd103bf3c539cdc655f21b1ee778a02ef27be5f80a6e9f8c34704`,
  nine FP64 rows, normalized norm `1.0972734315870298`.
- Target library SHA-256:
  `22c24c53af14f8a0969bb09efe3150d63bdca19ca3c85baac046450d62576988`.
- Missing ties: `gt:2299:{8,9,11,43,45}`. The targeted inventory found
  zero emitted occurrence among their 40 admitted canonical/neighbor rows.

Construct exactly:

```text
parent_route[:369]
+ canonical rows in order [gt45, gt11, gt9, gt8, gt43]
+ EOS
```

The route has 415 tokens. Before teacher scoring it must pass the production
parser/global matcher with 38 persons, 8 ties, 46 unique strict predictions,
zero hard counters, and one final row-aligned EOS.

## Child residual

Load the frozen r32 checkpoint and install the parent payload as one output-head
residual. Reproduce the exact 41-owner anchor first. Capture the canonical
415-token route under that parent-augmented model.

Positive states are candidate positions whose target is not strict top-1.
Protect every anchor ordinary state except a prefix-identical released positive,
plus every candidate state already top-1. Project positives out of the protected
span exactly as in the accepted parent method. Stop if the positive projected
rank is zero, exceeds 46, or any projected/original norm is below `1e-3`.

Only unique target tokens at positive states may receive a child delta. For each
positive state, constrain its target against every other movable child row and
the strongest fixed remainder-of-vocabulary competitor. Use the repaired
HiGHS-feasibility plus SLSQP minimum-norm solver, margin `0.01`, child normalized
cap `9/8`, exhaustive runtime full-vocabulary recheck, and protected correction
`<=1e-10`.

Do not nest wrappers. Compose one persistent union payload:

```text
composed[token] = parent[token] + child[token]  # overlap
composed[token] = present row                  # otherwise
```

Bind parent, child, and composed identities separately. A zero-child composed
payload must exactly reproduce the 41-owner anchor before the candidate solve.
Existing DoRA, tied embedding delta, aligner, vision, base weights, and every
unselected output row stay frozen.

## Acceptance, bounds, and stop

Run exactly one warm ordinary greedy candidate from the original image/prompt,
with no forcing, sampling, beam, prefix controller, or logits processor. Warm
success requires 38 persons, 8 ties, exactly 46 strict owners/predictions, zero
debt, and natural row-aligned EOS. Owner-equivalent rows may pass even if token
identity differs.

Persist only a warm success, release model/GPU references, then execute this
unit's snapshotted runner in a fresh subprocess and require exact warm/cold
route, ledger, composed payload, selected rows, protected states, surfaces, and
null parity.

- One GPU/world size 1; one solve; one warm candidate; at most two model loads.
- At most 46 positives/rank, 26 child rows, 1,196 constraints/variables.
- At most 1,800 seconds, 16 GiB reserved memory, 200 MB artifacts.

Stop without fallback on identity/static-route failure, certified
infeasibility, numerical solver HOLD, child norm above `9/8`, runtime margin
failure, protected-null failure, final warm gate failure, cold mismatch, or
resource breach. A valid negative routes to the separate 120-order unit; do not
expand aliases or change the cap in this sentinel.

The only positive claim is one Image2299 composed-augmented-model ordinary-
greedy 46/46 result. It is not base-model or transfer evidence.
