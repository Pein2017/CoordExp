---
title: Image2299 global target-only protected-null distillation results
type: investigation
role: research-result
authority: non_normative_research
unit_id: 2026-08-30-image2299-global-target-only-protected-null-distillation
status: complete
evidence_status: exact_feasible_minimum_norm_exceeds_cap1
updated: 2026-08-30
---

# Result

The global target-only program is feasible, but its exact minimum normalized
norm exceeds the frozen cap `1`. Cap 1—not feasibility—is the sole reached
stop, so no ordinary-greedy warm or cold candidate was run.

## Numerical repair lineage

V1 receipt, independently SHA-256 verified:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-30-image2299-global-target-only-protected-null-distillation/20260830T-image2299-global-target-only-protected-null-distillation-v1/receipt.json
a70a6ffe5a7379b3ad864cd6eb262966a0430106b86b382335e231bc616c819c
```

V1 is retained as a technical numerical HOLD, initially misclassified as
infeasible: its legacy dual path reported slack `-1.137212933599585e-6`, with
zero warm candidates and zero cold replay. Repaired V2 receipt:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-30-image2299-global-target-only-protected-null-distillation/20260830T-image2299-global-target-only-protected-null-distillation-v2/receipt.json
3bc288a62e6b6ef23ef9f08dc53b33e9d5bafd3f690d8cb7c226f20577a28cdf
status norm_cap_exceeded
```

HiGHS certified `108/108` constraints feasible with minimum slack
`-5.551115123125783e-15`. SLSQP independently returned minimum normalized
norm `1.0972734315870298`, objective `0.6020044918333882`, and slack
`-5.329070518200751e-15`; the dual lower bound agrees at
`1.0972734315870256`. This is exact feasible minimum-norm evidence, not solver
infeasibility.

## Scope and disposition

Both runs preserved the frozen r32 checkpoint, prompt/image, controlled route,
12 positive states, 664 protected states, rank-12 basis, nine output rows,
runtime full-vocabulary surface, FP64 null bound, and one-GPU/world-size-1
resource envelope. Each used one solve and one model load; peak CUDA reserved
was `10815012864` bytes and artifacts stayed below 100 MB. No warm/cold route,
persistent payload, checkpoint, training, or model mutation exists.

No ordinary-greedy 38-person, transfer, general enumeration-learning, or tie
recovery claim is made **for this cap-1 unit**. Its then-unexecuted `9/8` phase
later succeeded in [Dyadic Norm-Release Distillation](../2026-08-30-image2299-dyadic-norm-release-distillation/results.md),
and later successors reached 46/46; keep those claims with their own receipts.
