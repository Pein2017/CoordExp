---
title: Human-13 K-Trajectory RP-Crossover Screen Results
description: Verified negative admission result for cross-engine exact-on-policy trajectory credit.
type: investigation
role: results
authority: non_normative_research
unit_id: 2026-08-14-human13-k-trajectory-rp-crossover-screen
status: complete
evidence_status: verified
updated: 2026-08-15
---

# Result

The unit stopped at its predeclared sampler-versus-replay parity gate.  The
exact-on-policy trajectory-credit route is retired for this execution design;
the eighteen-cell model-quality matrix did not run.

## Observed

- Source: S step-2444, language-only rank-16 DoRA warm start, frozen special
  embedding delta.  Manifest SHA-256:
  `a8f88716c1227054ab29dc698f89462c9369c47c8d6415de3783c0937f60a6fb`.
- Case: image 1584, seeds `30001..30016`, `rp=1.0`, temperature `0.4`,
  `top_p=1.0`, 16 trajectories in four native batches of four.
- Native acquisition: 16 natural-stop trajectories, 1,573 generated tokens.
- Replay: 16 HF fp32/SDPA batch-one exact-history forwards over the same sealed
  histories and processed-policy transform.
- Sealed parity result: maximum error `0.1675825119` nats; mean error
  `0.0021682973` nats; `22/1573` tokens exceeded `0.02`.  The fixed limits were
  `0.02` per token and `0.002` group mean.
- First-failure stop: `rp=1.10` was not run.  Optimizer steps, backwards,
  checkpoints, proposals, witness/dose/compiler/owner analyses, and matrix
  cells were all zero or absent.
- Resource envelope: 103.48 s wall time, four native batches, peak CUDA
  allocated 68,762,947,072 bytes, peak CUDA reserved 69,283,610,624 bytes, and
  peak host RSS 11,223,592,960 bytes.  The generic `replay_token_count=0` and
  resource `forward_count=0` mean no replay evidence was admitted and generic
  failure-path counters were not advanced; the exact surface receipt records
  the 16 real forwards and 1,573 comparisons.

## Supported

- Aligning replay from BF16/FA2 packed execution to HF fp32/SDPA exact history
  reduced the earlier v4 mean error from `0.055759` to `0.0021683` and reduced
  over-limit tokens from `620/1573` to `22/1573`.  Numeric execution surface
  was therefore the dominant source of the prior mismatch.
- Even after that correction, vLLM fp32/TRITON sampling and HF fp32/SDPA
  exact-history replay do not meet the frozen exact-policy gate.  The remaining
  mismatch is too large to call the estimator admitted under the declared
  contract.
- The stop rule worked: no model-quality update or owner result was produced
  from non-admitted policy evidence.

## Ruled out

- Widening the tolerance after observing v4/v5 is not a valid repair for this
  unit.
- The current cross-engine design cannot support the unit's exact-on-policy
  score-function claim.
- The full-panel LR ray and eighteen-cell matrix have no admitted prerequisite
  and are retired, not merely delayed.

## Unresolved

- Whether trajectory-level credit improves sampled or greedy owner coverage.
- Whether the sparse compiler or proposal preservation can prevent owner
  exchange.
- `rp=1.10` parity and any cross-RP transfer.
- Whether a single shared sampler/gradient engine can satisfy exact parity, or
  whether an explicitly approximate/off-policy estimator is useful.

## Not claimed

No statement is made about owner gains or losses, greedy recall, duplication,
malformed output, optimization quality, K-miss learning, generalization,
deployment, or checkpoint promotion.

## Artifacts

Root:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-08-14-human13-k-trajectory-rp-crossover-screen/
  vertical-dose-qualification-v5/
```

- `terminal.json` content SHA-256: `711ad7119d172f34e9b745503afc904d851fbade55b217ed821579b222597a21`
- `rp100/parity-evidence.json` content SHA-256: `054f7bb6951ae0eb2209fa5d93c27aee58170cf3ce1a0bdcd7452ef181040cf6`
- `rp100/native-acquisition.json` content SHA-256: `e235bfb921c51ba8e032867eed2279a03958bf4e5599d35f48ff564d56d0e5a5`
- `rp100/parity-error.json` content SHA-256: `f2e681bed92df6f4e97166ca100f42f5df07c0399631600571f8f3cee68805ff`
- Source lineage: `2a5eeb739f755993f47f326f55abb349626ca12c7419c38406b6817a0511a427`
- Source checkpoint payload:
  `99678ea954c4b37abbf704432dbf43a8df5ce37cd07ebcef4e11f263782dca47`
- RP1.0 leaf config:
  `b826fa9f70ec7a5772bdf5c023e31d48b5721d04bc92159c51f606c8e67d38e3`

## Disposition and next decision

Close this unit as verified negative evidence at its admission gate.  A future
unit must choose one of two different claims before implementation:

1. use one genuinely shared numerical forward for sampling and score-function
   gradients, accepting its compute cost; or
2. declare an approximate/off-policy objective and evaluate its bias and greedy
   transfer explicitly rather than calling it exact on-policy.

Neither route is authorized or implied by this result.
