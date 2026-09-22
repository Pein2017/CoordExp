---
title: Image2299 on-policy terminal action SQP-lite results
type: investigation
role: research-result
authority: non_normative_research
unit_id: 2026-08-30-image2299-ota-sqp-lite
status: complete
evidence_status: cold_verified_bounded_negative_no_owner_promotion
updated: 2026-08-30
---

# Image2299 OTA-SQP-lite results

## Outcome

The production-shaped r16 smoke, one r16 continuation attempt, the matched
r32 tangent screen, and the sole authorized r32 step completed.  No cold
natural rollout exceeded Parent33 or removed the designated unsupported tail.

The useful positive result is narrower: function-preserving r32 adds genuine
first-order directions and admits a cold working step at radius `1/64`, while
the matched r16 continuation admits no radius.  This is capacity evidence for
the local constrained surrogate, not owner-promotion evidence.

## Decision-bearing receipts

| Receipt | SHA-256 | Status | Role |
|---|---|---|---|
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-30-image2299-ota-sqp-lite/20260830T-image2299-ota-sqp-lite-smoke-v2/receipt.json` | `ad51f46f76274fc2e68218f17ced0443982261e03204a4c9a34993c1dcde4ba0` | `smoke_selected_working_cold_reproduced` | Valid r16 vertical slice and accepted update 1 |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-30-image2299-ota-sqp-lite/20260830T-image2299-ota-sqp-lite-continuation-v3b/receipt.json` | `32f821b4f004625cd84e7599cf85fa202d57bbebb7b2b70920ec58c18f1934d8` | `continuation_bounded_negative_no_admissible_radius` | Valid r16 continuation stop |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-30-image2299-ota-sqp-lite/20260830T-image2299-ota-sqp-lite-r32-screen-v1f/receipt.json` | `31c82b14a5a16ef5bcc0fa71022b6d410d204b8c3e07501dae4678ef4fb2b47b` | `r32_tangent_gate_open` | Valid no-update r32 capacity gate |
| `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-30-image2299-ota-sqp-lite/20260830T-image2299-ota-sqp-lite-r32-step-v1/receipt.json` | `a2912cab11797ddab8a556be1c17603d537271981047d40b99ffa6f27515278c` | `r32_step_selected_working_cold_reproduced` | Valid sole matched r32 step |

Earlier `smoke-v1`, `continuation-v3`, and r32 screen `v1` through `v1e`
are mechanics or diagnostic receipts only.  They respectively exposed the raw
SLSQP coordinate scaling, an unnecessary cleanup-variable bug, rank-width
margin calibration, an inappropriate elementwise Gram comparison, and a JSON
finalizer type.  None is a model-quality negative or promotion result.

## r16 result

The smoke selected the smallest radius, `1/128`.  It retained the exact v18
natural route and Parent33, preserved the designated tail debt, and changed the
selected gt32 complete-action minimum from `-7.8491888` to `-7.8486805`.
Every larger radius produced a different `other_unknown_neutral` tail and was
inadmissible.

After cold continuation from that checkpoint, all eight radii produced the
same inadmissible unknown tail.  No update was accepted, and a fresh cold load
restored surface `b85952abf74e982980696753e006a1bc0b17386546480c53a1373eef5b8b1406`
and natural route `dd483c20f1c21bc01678f551205143f163810ad7e2c4adb6f4e7b5cbc351afad`.
The r16 branch therefore stopped without using the nominal 16-update ceiling.

## r32 tangent result

The r32 checkpoint used `r=32`, `lora_alpha=64`, and preserved `alpha/r=2`.
All 196 new A blocks were deterministically nonzero while all new B blocks
started at zero.  Cold step-zero natural tokens, owners, raw counters, causal
event, EOS, and frozen surface were exact.  Rank-width GEMM changed teacher
margins by at most `8.87e-5`, below one percent of the strict `eta=0.01`
safety floor.

All 17 new-B gradient rows were live across all 196 modules; new-A gradients
were zero.  The added-coordinate Gram was PSD with eigenvalues from `56.09` to
`210089.07`.  The matched old-coordinate control had spectral Gram drift
`3.50e-5` and rho drift `9.70e-8` from sealed r16.  At the same trust radius:

```text
r16 rho      = -7.77878317
full r32 rho = -7.76906195
advantage    = +0.00972122
```

This opened exactly one r32 step, not an r32 training loop.

## Sole r32 step

The r32 panel selected radius `1/64`, twice the largest safe first r16 radius.
The selected state cold-reproduced Parent33, the same three correlated raw
counters, one causal duplicate tail, all active cuts, and frozen-surface
identity.  Larger radii through `1/32` produced an inadmissible unknown tail.

The step changed only the natural tail after token position 301:

```text
v2 tail box:  [1102, 233, 1200, 667], strict gt22 IoU 0.6663
r32 tail box: [1119, 273, 1200, 667], strict gt22 IoU 0.5000
```

It did not emit gt32 or another missing owner.  It moved the unsupported box
while preserving its duplicate/near-miss optimization-state class.

## Mechanism interpretation

Observation: both valid SQP solves put essentially all coefficient mass on the
weakest gt32 action coordinate at relative position 4; the remaining action
and cut coefficients were negligible.  The weakest margin was about `-7.85`,
whereas the next deficits were much smaller.  Thus a nominal complete-action
max-min objective still behaves like a single bottleneck ray until that large
coordinate deficit is substantially reduced.

Observation: improving that teacher-forced gt32 coordinate margin did not move
the free rollout toward a strict gt32 row.  The r32 free tail instead shifted
farther right and down while remaining assigned to gt22.  Larger steps crossed
directly into a different unsupported description class.

Inference: the principal remaining failure is the mismatch between the local
teacher-forced action surrogate and the discontinuous autoregressive natural
action, not demonstrated lack of r16/r32 parameter capacity.  r32 improves the
surrogate geometry and safe radius, but that improvement does not compile into
set-level owner gain.

This does not prove that another alias catalog, rollout-aware objective,
dynamic natural-branch cut, parameter surface, or architecture cannot promote
an owner.  It closes only the frozen OTA-SQP-lite family and its one conditional
r32 expansion.

## Final disposition

- Stop r16 and r32 updates under this unit; no checkpoint is promoted.
- Do not open aligner or tied-embedding training.  The required independent
  visual-localization evidence is absent, while the observed failure remains
  explainable by action-surrogate mismatch.
- Any successor should change the decision-bearing objective toward the actual
  earliest natural branch or rollout-level action admission, rather than add
  more radius trials to this teacher-forced max-min ray.
