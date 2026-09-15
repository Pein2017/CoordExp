---
title: Image2299 canonical-G46 global-QP protected-null sentinel results
type: investigation-result
role: research-result
authority: non_normative_research
unit_id: 2026-08-31-image2299-canonical-g46-global-qp-protected-null-sentinel
status: complete
evidence_status: fresh_cold_direct_canonical_g46_success
updated: 2026-08-31
---

# Image2299 canonical-G46 global-QP protected-null sentinel results

## Disposition

**Success.** One global QP, solved directly from frozen r32 against a fully
canonical 46-row route, produces one output-only residual whose ordinary-greedy
warm and fresh-cold outputs both exactly equal canonical G46. No M-route payload,
completed 46/46 payload, iterative cut, norm cap, or second QP solve was used.

## Authoritative evidence

- Receipt:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-31-image2299-canonical-g46-global-qp-protected-null-sentinel/20260831T-image2299-canonical-g46-global-qp-v3/receipt.json`.
- Receipt SHA-256:
  `ff04300c19eb15ac0f6193ecace0a475e4e023b398eece913e6b7dda568a8a5d`.
- Payload SHA-256:
  `22d9df392f586979e8b191a89068ea929943356d8bfce86856e8203c0e0f51fd`.
- Runner snapshot SHA-256:
  `b53ebad9e69326b952d5fb82092ed981f2de317450529601bf7dbcc241f1effe`.

## Program and behavioral result

| Quantity | Frozen result |
|---|---|
| Route | 46 complete canonical rows + EOS; 415 tokens; SHA `96c2cbe15fdb4c09822d6a472f2510605c5701cfafcc9e33324f14ce90e0cb49` |
| Positive states / selected rows / rank | `57 / 51 / 57` |
| Variables / constraints / solves | `2907 / 2907 / 1` |
| Minimum normalized norm | `28.282770681462463` |
| Protected-null maximum | `4.440892098500626e-13 <= 1e-10` |
| Full-vocabulary recheck | pass |
| Warm | exact G46; 38 persons, 8 ties, 46 owners, empty debt, natural EOS, no divergence |
| Fresh cold | exact same route, ledger, payload identity, positive positions, basis, protected states, and gate |

The payload is one FP64 sparse output residual over 51 selected token rows,
shape `51 x 2048`. Tied input embeddings, DoRA, existing embedding delta,
aligner, vision tower, and nonselected output rows remain frozen.

## Numerical certificate and lineage

HiGHS independently found a feasible primal. SciPy SLSQP returned status `8`
(`Positive directional derivative for linesearch`) despite a finite candidate
with minimum slack `-6.441069899665308e-12`. The runner does **not** call this a
SciPy success. It admits the convex-QP candidate through an explicit certificate:

- primal objective `399.9575587100963`;
- dual objective `399.9575586896209`;
- absolute primal-dual gap `2.0475397377595073e-08 <= 1e-6`;
- feasibility slack within the registered `1e-9` tolerance.

Run v1 stopped before decoding because the old helper required the SLSQP success
flag. Run v2 then proved exact warm and cold G46 behavior, but used different
evaluator labels, so label-derived `trajectory_id/pred_row_id` strings prevented
literal ledger equality. Run v3 uses one shared evaluator label and is the sole
authority. v1/v2 are technical lineage, not negative scientific results.

## Interpretation

**Observation.** Direct canonical G41 and G46 need normalized norms
`27.98494582667317` and `28.282770681462463`. G46 is only `1.064%` larger in
norm (`2.140%` in the quadratic objective), although its basis and route differ.
Both are far more expensive than model-manifold M41 (`1.0972734315870412`).

**Inference.** A model-emitted row witness is not necessary for this fixed
specimen: canonical GT rows can be compiled directly into a single greedy 46/46
trajectory by one constrained output-head intervention. The main sequence
effect is intervention cost, not demonstrated feasibility. The five missing
ties add little total normalized cost relative to canonicalizing the first 41
owners in this particular construction, but the basis change prevents a strict
causal cost decomposition.

**Speculation.** The large M-to-G cost gap may encode mismatch between canonical
row realizations and the model's preferred autoregressive geometry. This result
does not identify that internal mechanism.

## Claim boundary

This is still single-image, oracle-route, augmented-output-head evidence. The
row contents are fully canonical, but the owner order is inherited from G41's
M-derived ledger and then extended by the accepted five-tie suffix. Therefore it
removes dependence on model-emitted token/box witnesses, but not yet dependence
on that discovered owner order. It does not establish a deterministic-order,
multi-image, held-out, scalable, base-model, or transfer result.
