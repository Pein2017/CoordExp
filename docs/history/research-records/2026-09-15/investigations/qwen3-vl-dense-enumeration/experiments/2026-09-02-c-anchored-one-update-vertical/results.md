---
title: C-Anchored One-Update Behavior Vertical Results
description: The fresh first AdamW proposal improves the eleven-row gain objective but has negative first-order changes on two protected margins, so the frozen gate stops before mutation.
status: complete; mechanically valid; SCIENTIFIC_STOP_C_ANCHORED_ONE_UPDATE_FINITE; zero updates
---

# C-Anchored One-Update Behavior Vertical Results

## Verdict

`SCIENTIFIC_STOP_C_ANCHORED_ONE_UPDATE_FINITE` is the valid result of this
unit.  The exact first fresh AdamW proposal has positive gain alignment, but it
decreases two of the three registered incumbent margins.  The frozen pre-step
gate therefore stopped before `optimizer.step()`.

No parameter was updated, no adapter was saved, and no natural decode was
launched.  This is not evidence that the eleven-image shared objective cannot
learn.  It isolates a mismatch between the safe Euclidean direction from the
predecessor and AdamW's coordinate-wise first-step geometry.

## Immutable evidence

- authoritative receipt:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-c-anchored-one-update-vertical/authoritative-v1/receipt.json`,
  SHA-256
  `ea4efd923d340de4f0432ea41118ce3b262f8256153ce0793b66753fee1a759c`;
- log:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-c-anchored-one-update-vertical/authoritative-v1.log`,
  SHA-256
  `c9e6e44f00833fc648349eb5e60dcf7a93ac8d60dccf5a578f6872fe11715e97`;
- frozen predecessor plan SHA-256:
  `f13ca3d2744ddfa5ae0db5748999043c356621995856d26f512f0bb42e956743`;
- runner commit: `9d338ca98f2f4163458d1662298ca8ee8823c21d`;
- C adapter fingerprint:
  `5eed9a1eeccbd8117ab7fec7c9c63fafc1e26b1f25233f6a4c56908c88d4bb95`;
- inherited selected-token embedding-delta fingerprint:
  `635ec008a79fd2657c2acc75772a52cfda70eb0f664ef4f91c4f05c0aa931bc6`.

The reference remained the universal base plus the shared unmerged C DoRA and
the frozen step-2444 embedding delta.

## Proposal result

The eleven-event equal-image gain gradient reproduced norm `4.4871373`.  After
the frozen global norm cap, the fresh AdamW proposal had:

- realized proposal norm: `0.04236874`;
- gain alignment `q^T d = +0.08304185`;
- parameter updates: `0`;
- optimizer state entries after the gate: `0`.

The incumbent results were:

| protected owner | anchor margin | predicted change `a_i^T d` | affine margin |
|---|---:|---:|---:|
| `101636:coco_ann:1760455` | 0.02978325 | +0.03189343 | 0.06167668 |
| `347671:coco_ann:1799145` | 0.08687401 | -0.00049559 | 0.08637842 |
| `359310:coco_ann:1172698` | 0.03891754 | -0.00260375 | 0.03631379 |

Thus the registered zero-decrease condition fails on two incumbents even
though all three first-order affine margins remain positive.  Learning-rate
shrinkage cannot change the sign of these directional derivatives; projection
would change the proposal but is not yet necessary to answer whether the
finite margin budgets already suffice.

## Mechanical validity

The run used the exact 11 gain events and 3 incumbent decisions, with 14
forwards, 11 backwards, and 3 VJPs.  It peaked at `26.56 GB` CUDA allocation and
finished in `23.12 s`.  All identities, FP32 C prefixes, complete rows, and the
588-tensor / 18,006,016-scalar trainable surface passed.  The stop occurred at
the intended scientific gate, not at loading, graph, memory, optimizer, or
artifact plumbing.

Before launch, the inference loader was also strengthened to compare every
saved and materialized adapter tensor value, rather than keys alone.  Its
targeted RED reproduced a wrong-value load that previously passed; the fix and
the surrounding DoRA tests pass (`42 passed`).  That permanent trust-boundary
upgrade was not exercised by this stopped arm because no adapter was written.

## Interpretation and next boundary

This result exposes an overstrict surrogate, not yet an observed preservation
failure.  The behavior-owned requirement is that a protected decision remains
on the positive side (and ultimately remains matched under natural decode),
whereas this unit required its margin never to decrease at all.  The observed
negative changes spend only `0.57%` and `6.69%` of the two affected anchor
margins in the first-order model.

The shortest successor is therefore not DDP, full-248 evaluation, a different
optimizer, or an immediate projection.  It is a separately frozen rerun of the
same one-step AdamW proposal under finite affine-margin admission:

\[
m_i + a_i^T d > \varepsilon_m,
\]

followed by exact post-step margins, adapter-only cold readback, and the already
specified twelve-image natural panel.  Projection becomes justified only if
that finite-budget proposal crosses the buffer or the realized/natural result
loses an incumbent.  Production and DDP/COCO scaling remain on hold.
