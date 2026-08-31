---
title: Image2299 global target-only protected-null distillation
type: investigation
role: research-unit
authority: non_normative_research
unit_id: 2026-08-30-image2299-global-target-only-protected-null-distillation
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: exact_feasible_minimum_norm_exceeds_cap1
updated: 2026-08-30
---

# Image2299 global target-only protected-null distillation

## Frozen question and contrast

Does the one global minimum-norm residual satisfying all 108 target-only
constraints make the frozen r32 model ordinarily greedily emit the complete
38-person trajectory?

The direct contrast is the completed staged target-only predecessor.  Its first
solve exactly reproduced stage 1.  Its second solve passed all 27 exact
full-vocabulary margins and retained the intended parent plus `gt32,gt35`
owner set, but generated 352 rather than 343 tokens and one unsupported person.
The first unprotected future decision is the stage-3 row-open at position 342;
stage 2 allowed that target row to move before constraining its following x1.

This successor removes intermediate stage gates.  It solves all twelve positive
states together and evaluates only the final natural trajectory.  It does not
change the model, output rows, basis, margin, norm cap, controlled route,
matcher, or debt policy.

The strongest alternative is global infeasibility or norm above 1.  A second
alternative is that even all real-valued margins do not reproduce the route
under runtime arithmetic, which is technical HOLD rather than model evidence.

## Immutable specimen and predecessor

- Staged predecessor receipt:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-30-image2299-target-only-protected-null-distillation/20260830T-image2299-target-only-protected-null-distillation-v1/receipt.json`,
  SHA-256
  `e1ed1e101bf0f3b8485491ee87bfdb9f226a4451580da1885222ba4cd8b4729e`.
- Terminal witness receipt SHA-256:
  `d9cb113ae3f7cbe4b080fbe4f8faf76f68d36d9ca2844940b27560d330ab4495`.
- Frozen r32 checkpoint:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-30-image2299-ota-sqp-lite/20260830T-image2299-ota-sqp-lite-r32-step-v1/checkpoint-selected-r32-step`.
- Controlled route SHA-256:
  `c89c6f9900303227cf781caf4a39dc203a09a53b0d319930754cda184529530e`.
- Recapture: 12 positive positions
  `301,324,328,342,346,351,355,360,364,365,367,369`; 664 protected
  states; rank-12 basis; protected-hidden SHA-256
  `d3ba5b61b4783da8c8e1d7973d32c337e3abae96efb2b255c48173bd875a9ead`;
  basis SHA-256
  `f913a3ae539119deb7c61641ae39458e343d7f39404f571b7e37af65907ea438`.
- Exactly nine movable output rows:
  `151645,151646,151820,151867,151935,152032,152190,152242,152305`.

Prompt, image, checkpoint readback, r32/frozen surfaces, wrapper, FP64 apply,
`1e-10` observed-state null bound, owner authority, and final acceptance are
inherited unchanged and reverified live.

## One global program

Build the same target-only partition as the predecessor over all 12 positive
states.  Each state contributes eight constraints against the other movable
target rows and one against the strongest fixed row in the rest of the
vocabulary.  Solve once from zero:

```text
108 constraints
108 normalized variables = 9 rows x rank 12
target margin >= 0.01
normalized residual norm <= 1
```

Require solver feasibility, finite values, exact nine-row selection, numerical
nullness, and a runtime-dtype exhaustive full-vocabulary recheck.  Then install
the persistent residual and run exactly one ordinary unforced greedy decode
from the original prompt/image.  There is no intermediate decode, closure,
retry, beam, sampling, prefix table, forced token, or logits processor.

The v1 execution was a technical numerical HOLD, initially misclassified as
infeasible: its legacy dual path reported minimum slack `-1.137212933599585e-6`
and produced zero warm candidates. Receipt (independently SHA-256 verified):
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-30-image2299-global-target-only-protected-null-distillation/20260830T-image2299-global-target-only-protected-null-distillation-v1/receipt.json`,
`a70a6ffe5a7379b3ad864cd6eb262966a0430106b86b382335e231bc616c819c`.

The repaired v2 execution certified all `108/108` constraints feasible. HiGHS
minimum slack was `-5.551115123125783e-15`; SLSQP minimum normalized norm was
`1.0972734315870298`, objective `0.6020044918333882`, and slack
`-5.329070518200751e-15`; the dual lower bound agrees at
`1.0972734315870256`. Thus cap 1—not feasibility—is the sole reached stop;
no warm or cold candidate was run. Receipt:
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-30-image2299-global-target-only-protected-null-distillation/20260830T-image2299-global-target-only-protected-null-distillation-v2/receipt.json`,
`3bc288a62e6b6ef23ef9f08dc53b33e9d5bafd3f690d8cb7c226f20577a28cdf`.

Warm success requires the production matcher to find exactly the controlled
owner set: 38 persons, the same 3 ties, 41 strict owners/predictions, zero hard
debt, and natural row-aligned EOS.  Token identity may differ only if this
match-level gate passes.  Persist only a warm success, release every model/GPU
reference, then fresh-subprocess load the augmented payload and require exact
warm/cold route, ledger, residual, protected-state/basis, null, and frozen-
surface parity.

## Bounds and stop

- One GPU/world size 1; one solve; one warm candidate; at most two model loads.
- 12 positives, 9 rows, rank 12, 108 constraints/variables.
- At most 1,200 seconds, 16 GiB reserved memory, and 100 MB artifacts.
- No existing parameter or unselected vocabulary row changes.

Stop without fallback on identity drift, infeasibility, non-finite solve,
normalized norm above 1, real-dtype full-vocabulary failure, final warm gate
failure, protected correction above `1e-10`, cold mismatch, or resource breach.
Do not widen rank, norm, output rows, route, margin, or parameter surface in
this unit.

The claim is exact feasible minimum-norm evidence exceeding cap 1. It is not an
ordinary-greedy result, base-r32 result, transfer evidence, general enumeration
learning, or 8/8 tie recovery. At this unit's stop, only the `9/8` norm-release
phase was pending. It later executed as [Dyadic Norm-Release Distillation](../2026-08-30-image2299-dyadic-norm-release-distillation/results.md)
and reached cold 41-owner greedy; later composed and direct-canonical successors
reached 46/46. Those outcomes do not change this cap-1 unit's claim.
