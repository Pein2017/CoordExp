---
title: Image2299 target-only protected-null distillation results
type: investigation
role: research-result
authority: non_normative_research
unit_id: 2026-08-30-image2299-target-only-protected-null-distillation
status: complete
evidence_status: stage1_exact_stage2_future_positive_leakage
updated: 2026-08-30
---

# Target-only protected-null distillation results

Receipt:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-30-image2299-target-only-protected-null-distillation/20260830T-image2299-target-only-protected-null-distillation-v1/receipt.json
SHA-256 e1ed1e101bf0f3b8485491ee87bfdb9f226a4451580da1885222ba4cd8b4729e
status stage_exhausted
```

The receipt was independently SHA-256 verified. The run stopped at stage 2,
after two of the five planned target-only solves.

| stage | constraints | normalized norm | route | expected length | observed length | gate |
|---|---:|---:|---|---:|---:|---|
| 1 | 9 | 0.43629900998978716 | `7a39d314548b8f8a7cbae23727a110e836b0c449aa459a0fe3e31fe93d4147d0` | exact | 325 | pass |
| 2 | 27 | 0.7897506392158977 | `c4d21bb0f8a4895e88a487248cdcc2594701bcd2adec4c49bffa7414a7c730f3` | 343 | 352 | fail |

Stage 1 passed its exact route and full-vocabulary recheck. Stage 2's solver
was feasible and its full-vocabulary runtime recheck passed, retaining the
parent owner set plus `gt:2299:32` and `gt:2299:35`. It nevertheless produced
the nine-token route-length excess and one unmatched/unsupported person; no
cold run was reached.

The future stage-3 positive at position 342 was intentionally outside the
protected set. A global nine-row residual could therefore open that row before
its `x1` was constrained. This is evidence for staged-gate/global-residual
mismatch, not target-only capacity exhaustion. The then-unexecuted one-shot
108-constraint successor later ran as [Global Target-Only Protected-Null Distillation](../2026-08-30-image2299-global-target-only-protected-null-distillation/results.md),
then reached cold greedy through the [dyadic norm release](../2026-08-30-image2299-dyadic-norm-release-distillation/results.md).
That later evidence does not retroactively promote this staged unit.

No model mutation, checkpoint, training, deployment, or architecture claim is
made. This is bounded one-image mechanics evidence only.
