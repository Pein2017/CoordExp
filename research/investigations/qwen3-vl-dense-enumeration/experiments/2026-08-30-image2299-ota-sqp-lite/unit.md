---
title: Image2299 on-policy terminal action SQP-lite
type: investigation
role: research-unit
authority: non_normative_research
unit_id: 2026-08-30-image2299-ota-sqp-lite
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: cold_verified_bounded_negative_no_owner_promotion
updated: 2026-08-30
---

# Image2299 on-policy terminal action SQP-lite

Results: [results.md](results.md).

## Decision

Run one bounded test of whether the current language-DoRA surface contains a
multi-margin direction that can turn Parent33 into its first cold natural
zero-debt proper superset.  The main arm remains the existing `r=16`,
`lora_alpha=32` language DoRA.  A function-preserving `r=32`,
`lora_alpha=64` expansion and aligner-DoRA are no-update tangent-space
diagnostics only; neither may train unless its predeclared gate opens.

This unit replaces another fixed-`gt32` guard continuation.  It does not claim
that the DoRA surface is sufficient, that the 46-owner path is reachable, or
that the MLP aligner is uninvolved.

## Final disposition

Execution completed with no owner promotion.  r16 admitted one `1/128`
working step and then no radius; function-preserving r32 passed its no-update
tangent gate and admitted one cold `1/64` working step, but still produced the
same Parent33 plus one unsupported gt22 tail.  The unit is closed.  Aligner and
tied-embedding training remain unopened because no independent visual-side
localization evidence passed their gate.

## Frozen question and contrast

At the last clean Parent33 prefix in the cold v18 working model, does a
terminal-first, multiple-alias, multi-margin trust-region direction produce a
cold natural greedy proper superset without exchanging an incumbent owner or
adding evaluator debt?

The decision-bearing contrast uses the same v18 parameters, prompt, image,
evaluator, trainable language-DoRA surface, base step scale, and eight radii:

- **Control:** the v18 fixed `gt32` first-coordinate target plus G6 equal-unit
  direction, with G0--G5 as feasibility checks.
- **Main:** choose the most feasible complete row action from aliases of every
  still-missing owner and solve one small constrained max-min problem over the
  active target margins and incumbent cuts.

The strongest alternative is a local DoRA-surface limitation: after alias
choice and simultaneous constraints are corrected, no finite-radius direction
may improve a complete missing-owner action while retaining Parent33.

## Immutable specimen and acceptance owner

- Parent promotion champion: the 33-owner cold route with token SHA-256
  `f0b762c6a54b03015e27e5ec2794c8b8c8f2403607f18eece890047b4ffb05fd`.
- Working initialization:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-29-image2299-certified-46-owner-path-overfit/parallel-match-corridor/projected-owner-corridor/20260830T-gt17-gt32-coordinate-guard-v18/checkpoint-terminal-step-16`.
- Working receipt SHA-256:
  `81827fe7d4e82600b52824207cac9b7b2845f345a2b6b2b5ef3c6509e3c18fce`.
- Prompt-token SHA-256:
  `33f11458039a9b8c01e1329eeedad91ac68824cc5454c9df4b15e47d50458ffb`.
- Image SHA-256:
  `cd7199a37188c9ac6481520175866cbd78fa6fba35f4290bb0b60c78afcb2df3`.
- Evaluation authority and global matching semantics remain those of the
  certified-46 predecessor.  Target rows never substitute for GT matching.

The v18 checkpoint is a working state, not a champion: it retains Parent33 but
has one designated unmatched/unsupported person tail.  The optimization prefix
is the real v18 prefix immediately before that tail.  A candidate repairs or
replaces the designated tail; it never appends after the debt.

The missing owner set is fixed as:

```text
gt:2299:{0,1,8,9,11,14,18,20,32,34,35,43,45}
```

## Alias catalog

For each missing owner, collect up to eight parser-clean candidate rows from
historical strict matches, the frozen target library, and bounded coordinate
neighbors.  Every candidate is re-evaluated at the current real prefix; no KV
state or prefix is transplanted from another rollout.

The production parser and global matcher must certify the static sequence
`Parent33 rows + alias + EOS` as exactly 34 unique owners with zero debt.  At
most the best two certified aliases per owner proceed to gradient scoring.
Forward ranking may reduce work; it is not evidence that rejected aliases are
globally unreachable.

This is a set-level gate, not a row-identity lock.  A global Hungarian
permutation among the 34 prediction rows is allowed when the final committed
owner set is still exactly `Parent33 union {target}`, every match is strict,
and no evaluator debt appears.

## Complete-row action score

For alias tokens `q=(q_1,...,q_L)` at real prefix `h`, define each greedy
margin against the current best non-target token:

```text
m_t(q,h;theta) = z(q_t | h,q_<t) - max_{v != q_t} z(v | h,q_<t)
```

The hard diagnostic certificate is the weakest complete-action margin over
row-open, description, box wrappers, all coordinates, row closure, and the
following EOS:

```text
psi(q,h;theta) = min_t m_t(q,h;theta)
```

Aliases are multiple positives only at the action level: select the alias with
the best feasible hard minimum.  Do not average CE across aliases or require
canonical GT coordinate tokens.  For person rows, the description competitor
set must include at least `tie` and the observed `chair` token.

## SQP-lite direction

Represent the parameter displacement in the span of the active margin
gradients, so the numerical solve is small even though the language-DoRA
surface has 18,006,016 elements.  For target margins `m_t` with gradients
`g_t`, incumbent-cut margins `a_k` with gradients `h_k`, strict safety floor
`eta > 0`, and ordinary FP32 DoRA-space trust radius `delta`, solve:

```text
maximize    rho
subject to  m_t + g_t^T Delta >= rho       for target action tokens
            a_k + h_k^T Delta >= eta       for active incumbent cuts
            ||Delta||_2 <= delta
```

Maximizing the predicted final minimum margin, rather than equal increments,
respects the different initial token deficits.  The first implementation uses
no Fisher, KL weight, per-loss coefficient, or optimizer state.  Linearized
margins guide search only; exact teacher margins, natural decoding, and the
global matcher own acceptance.

Seed the active cuts with the predecessor's G0--G6 failure boundaries and the
lowest-slack current-route decisions.  If a candidate loses an incumbent,
add its earliest causal divergence and re-solve.  Match-equivalent token or
box changes are permitted; after any accepted route change, rebind every prefix
and discard stale cuts that no longer name the current route.  Cuts do not
accumulate across champion promotion.

## Candidate and promotion gates

An intermediate working state may retain the same single designated tail debt
only when it:

- retains every Parent33 physical owner;
- adds no second malformed, ambiguous, duplicate, unmatched, unsupported,
  unknown-neutral, nontermination, or budget event;
- ends with natural row-aligned EOS;
- strictly improves the exact complete-action minimum or advances its first
  natural divergence;
- passes all active incumbent cuts after recomputation on its own prefix.

Promotion is stricter.  A cold save/reload must naturally produce a strict
proper superset of the current champion, retain every champion person and tie,
have zero hard debt of every kind, and end with row-aligned EOS.  The gained
owner need not equal the selected training target.  On promotion, stop the
branch, recompile the new on-policy prefix and missing set, and start a fresh
successor iteration.

Warm natural decoding may screen the eight-radius panel because every valid
predecessor reproduced warm behavior after cold reload.  Any selected working
checkpoint, possible promotion, and terminal result must be cold-reloaded and
compared before it can own evidence.

## Parameter-surface diagnostic

Before training, solve the same frozen alias/action and incumbent-cut panel on:

1. current language DoRA `r=16`, `alpha=32`;
2. a function-preserving language DoRA expansion `r=32`, `alpha=64`;
3. aligner-DoRA tangent directions on the eight discovered merger linears.

The r32 construction must preserve `alpha/r=2`, copy the existing A/B blocks
and DoRA magnitude vectors, initialize new A rows deterministically nonzero,
and initialize new B columns to zero.  It must reproduce current logits,
margins, cold tokens, owners, debt, and frozen-surface identity before its
tangent evidence is admissible.  Merely having a nonzero new gradient is not a
capacity result; the added surface must improve the constrained feasible
direction or an exact candidate behavior.

Training remains r16 unless the bounded r16 main arm supplies no proper
superset and the matched r32 panel supplies a feasible direction unavailable
to r16.  Aligner training remains closed unless language r16 and r32 both fail
and separate visual evidence plus the matched tangent panel localize useful
owner-specific signal to the aligner surface.  Full aligner unfreezing and tied
embedding updates are outside this unit.

## Budget and stop

- At most 16 accepted/relinearized r16 updates.
- At most eight radius candidates per update.
- At most three consecutive relinearizations with the same earliest natural
  divergence and owner set; then stop that branch.
- Stop immediately on a clean cold 34-owner promotion.
- Stop on identity, source, trainable-surface, solver-residual, gradient-sign,
  warm/cold parity, parser, matcher, or checkpoint provenance failure.
- If r16 ends negative, seal the exact tested alias catalog, prefixes, cuts,
  and trust region.  Do not call it global DoRA infeasibility.
- r32 training and aligner-DoRA training require their separate conditional
  gates; they are not automatic retries.

The sole success claim is a single-image cold natural parameterization that
adds at least one strict owner without exchanging Parent33.  It does not
establish 46/46, transfer, general set completion, or visual recognition of any
never-retrieved owner.

## Preparation receipt

The entropy pass removed one unreferenced byte-identical fixture copy while
retaining its canonical predecessor.  The CPU implementation now lives in
`scripts/research/run_image2299_ota_sqp_lite.py`, with its focused invariants in
`tests/research/test_image2299_ota_sqp_lite.py`.  Lead replay passed all nine
focused tests, Python compilation, frozen binding verification, whitespace
checks, and repository `git diff --check`.

That preparation receipt was mechanics-only.  The subsequent GPU evidence and
final claim boundary are recorded in [results.md](results.md).
