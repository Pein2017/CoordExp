---
title: Fixed N2 Prox-Linear versus Margin-Adam Results
description: Both squared-margin arms stop short of complete fit; the bounded inexact QP profile costs more and yields worse terminal natural coverage than margin-Adam.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
unit_id: 2026-09-05-dora-prox-linear-n2
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-05
---

# Outcome

**Mechanics:** valid bounded comparison, including exact Source restoration and
independent-process equality of all 588 saved/live unmerged adapter tensors.

**Scientific disposition:** `NEITHER_ARM_REACHED_COMMON_COMPLETE_FIT`.
The registered bounded inexact prox-linear quadratic-program (QP) profile shows
no optimization advantage over Adam on the same squared-margin objective.
Neither produces the required 65/65 natural owner coverage or complete route
margin certificate. Stop this unit; do not expand to N4/N13 or tune the profile
after seeing its scientific outcome.

This is not a negative result about every internal-QP algorithm, exact QP, or
DoRA capacity. All QP proposals in this comparison were deliberately bounded
inexact proposals, not certified solutions of their full subproblems.

## Frozen comparison

The [unit](unit.md) and [paired launch packet](launch-paired-v1.json) bind exact
step-2444 Source, images 6040/16228, 65 owners, all 592 canonical positions,
FP32/RP1.0 runtime, and 196 magnitude vectors / 573,440 trainable scalars.
No readout residual or other DoRA/base parameters were trained. Both arms use
the same total-token-normalized squared worst-competitor margin objective and
start from identical measured Source metrics.

The immutable primary aggregate is:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-dora-prox-linear-n2/paired-v1/comparison.json`

SHA-256: `1a35a6e327603480b0796be6fa29246fb1a7ba787d9f47ec6a3754cef66c1cc5`.
It binds all four train/cold terminal hashes and the deterministic analyzer.

| Quantity | Same-margin Adam | Bounded inexact prox-linear QP |
| --- | ---: | ---: |
| Recorded parameter updates | 300 | 13 |
| Training termination | 300-step ceiling | 1,800-second ceiling during outer attempt 14 |
| Initial loss | 0.5347199440 | 0.5347199440 |
| Final / best observed loss | 0.0002465234 | 0.0104541704 |
| Minimum final target margin | -0.1335630 | -0.5136747 |
| Positions below 0.00998 | 213 / 592 | 244 / 592 |
| Natural owner matches at IoU50 | 37 / 65 | 24 / 65 |
| Natural owner matches at IoU60 | 28 / 65 | 19 / 65 |
| Natural owner matches at IoU80 | 21 / 65 | 8 / 65 |
| Duplicate / malformed / cap counters | 2 / 2 / 0 | 33 / 301 / 1 |
| Valid unmatched, unknown | 33 | 39 |
| Training process seconds | 897.102 | 1,800.968 |
| Cold verification process seconds | 67.420 | 288.714 |
| Combined single-GPU process seconds | 964.522 | 2,089.681 |
| Peak GPU reserved bytes | 31,937,527,808 | 33,386,659,840 |
| Magnitude distance from Source | 23.19594 | 1.93591 |

Time-to-common-success is undefined for both arms: neither succeeded. These
are terminal bounded-run costs, not a speedup ratio between successful fits.
The recorded distance has the analytic frozen-direction DoRA isometry meaning,
not an explicit FP32 merged-matrix subtraction measurement. The QP's smaller
distance is not a low-norm advantage at equal success, because equal success
was never achieved.

On QP image 6040, native generation has 31 duplicate candidates, 298 malformed
spans, and a length cap, while matching only two owners at every threshold.
Image 16228 contributes the other 2 duplicates and 3 malformed spans. Valid
unmatched predictions remain unknown; they are not counted as hallucinations
or included in the confirmed structural-debt total.

## What the actual QP trajectory tells us

All 13 accepted updates lower the real squared-margin loss. Their actual-to-
predicted loss-decrease ratios range from 0.8565 to 1.1032. This is evidence of
useful finite local prediction on the accepted steps, not a globally accurate
linearization or a full QP optimality certificate.

The 11 recorded rejected proposals all have nonpositive penalized prox-model
decrease and `rho=null`: they were rejected **before** a finite trial. Five also
have nonpositive unpenalized predicted loss decrease. Therefore these recorded
rejections cannot be attributed to observed nonlinear trial failure. The
bounded inner solve / changing competitor set failed to produce an admissible
descent proposal at those doses. The unfinished proposal in outer attempt 14
is accounted for in runtime/oracle counters but has no completed history row.

QP costs include 376 dual-objective calls, 751 image VJPs, 750 image JVPs,
50 cut solves, and 1,862 vocabulary chunks. Vector-Jacobian products (VJPs)
and Jacobian-vector products (JVPs) traverse the actual language network;
matrix-free does not mean inexpensive. Adam used 600 training forward/backward
passes and 602 margin forwards. The different counters must not be equated as
equal-cost operations.

**Supported:** this numerical-work profile lacks practical advantage within
the frozen run; a low average squared deficit is not a complete-generation
certificate. **Unresolved:** how much failure comes from bounded inner quality,
the chosen squared-margin objective, or outer optimization settings. **Not
established:** that nonlinear mismatch is the main obstacle, that exact
internal QP cannot work, or that more iterations would necessarily fix it.

The [historical CE+AdamW result](../2026-09-01-human13-dora-magnitude-finite-overfit/results.md)
already achieved N2 65/65 and N13 392/392 on this magnitude surface. It remains
a capacity witness, not a fresh same-loss timing comparator. The present
results do not erase that evidence or establish an Adam impossibility.

## Mechanics history and accounting

- `smoke-v1`: first inner-work profile exhausted 900 seconds without an accepted
  update. Source was restored, an unchanged adapter was persisted, and no cold
  read was admitted. Terminal SHA-256:
  `5e4c7393812b1c31a16cbf6b33633f8b94f31bb20e41e89f9558663c69b77183`.
- `smoke-v2`: executor precreated the runner-owned directory. Entry correctly
  failed before model load; the log and empty directory remain, and no GPU
  result or terminal model receipt is claimed.
- `smoke-v3`: the unchanged-contract launcher replacement used the qualified
  bounded inner profile; one accepted update took 83.099 seconds, and cold
  verification took 45.323 seconds. The paired packet binds both receipts.
- `paired-v1`: each arm ran exactly once and received one cold verification.
  There was no post-outcome tuning, restart, merged checkpoint, or larger panel.

Instrumented single-GPU process times sum to about 68.08 GPU-minutes across
the valid/partial model runs and cold reads, including qualification smokes.
The paired arms alone total about 50.90 GPU-minutes. Conda/shell launch overhead
and the pre-model directory failure are outside these instrumented sums. GPU
stress activity is a user-declared background workload, not a reserved-cluster
benchmark; no repeatability or hardware-normalized efficiency claim is made.

## Reproduction and closure

From this worktree, choose a fresh output filename:

```bash
conda run -n ms python scripts/research/analyze_dora_prox_linear_n2.py \
  --root /data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-dora-prox-linear-n2/paired-v1 \
  --output /tmp/dora-prox-linear-n2-comparison-replay.json
```

The reducer refuses overwrite and independently reconstructs the complete-fit
predicate from canonical and natural metrics. A sensitivity check using actual
Adam receipts rejects a forged `complete_fit` status; it does not trust the
executor's status summary. Fresh replay must reproduce the primary aggregate
byte-for-byte. Focused runner tests cover shared slack, joint cross-image
products, competitor switching, rho, rejection/restoration, and persistence.

Current disposition: close this fixed profile with bounded negative evidence.
Any successor must separately decide whether to test inner-solve quality,
change the loss, or pursue another optimizer; these are different interventions
and are not implicitly launched by this result. No production promotion,
OpenSpec archival, external publication, or N4/N13 continuation occurred.
