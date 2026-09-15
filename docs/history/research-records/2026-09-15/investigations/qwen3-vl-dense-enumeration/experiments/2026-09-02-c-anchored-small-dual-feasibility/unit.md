---
title: C-Anchored Small-Dual Local Feasibility
description: A zero-update test of whether an evidence-linked, single-successor gain direction at current C prefixes is compatible with the surviving behavior-linked incumbent margins.
status: complete; mechanically valid; GO_C_ANCHORED_SIMPLE_REFRESHED_VERTICAL; production held
---

# C-Anchored Small-Dual Local Feasibility

## Frozen question

> From the exact unmerged C adapter, does the current-prefix complete-row gain
> direction threaten any surviving Phase-A lesion incumbent, and, only if it
> does, does the corresponding hard-margin cone retain a nonzero direction of
> positive predicted gain?

This is the only successor authorized by the
[C-anchored owner-mechanism audit](../2026-09-02-c-anchored-owner-mechanism-audit/results.md).
It is a zero-update local-feasibility unit, not projected training.  Backward
passes are permitted; optimizer construction, parameter updates, checkpoints,
screen-2 evaluation, and production promotion are forbidden.

## Model and artifact identity

The reference model is the predecessor's frozen C final state:

- universal base:
  `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`;
- C adapter:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-01-annotated-owner-direct-c-d0-pilot/train/c/runs/qwen3_vl_2b_annotated_owner_direct_c_26_steps_ebs64_seed19/checkpoints/step-26/adapter`;
- C adapter fingerprint:
  `5eed9a1eeccbd8117ab7fec7c9c63fafc1e26b1f25233f6a4c56908c88d4bb95`;
- cold-readback receipt SHA-256:
  `319e8b50ec46262d096d1a9b8fe9a2768e01e2a47dbf60f876bcd9c854abc1ff`;
- frozen selected-token embedding delta: the step-2444 delta inherited by
  `qwen3_vl_2b_c_anchored_owner_audit_c_train248.yaml`, fingerprint
  `635ec008a79fd2657c2acc75772a52cfda70eb0f664ef4f91c4f05c0aa931bc6`;
- trainable tangent surface: exactly the 588 shared language-tower rank-16
  DoRA `A`, `B`, and magnitude tensors, 18,006,016 scalars;
- frozen: base weights, tied embedding/lm head, selected-token delta, aligner,
  and vision tower.

The model remains `immutable base + one shared unmerged DoRA adapter`.  No
output residual, per-image payload, adapter routing, external memory, merge,
or temporary export is used.

The authoritative predecessor inputs are:

- Phase-A audit:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-c-anchored-owner-mechanism-audit/audit-v1.json`,
  SHA-256
  `40a51a1e34950b9efaf72f25e648e1c3e2e02764929d98655195ceadb4f163af`;
- C natural decode:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-02-c-anchored-owner-mechanism-audit/full/infer/c/qwen3-vl-2b-c-anchored-owner-audit-c-train248`;
- processed 248-image input:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-01-annotated-owner-direct-c-d0-pilot/materialization-v1/train-common-base.jsonl`,
  SHA-256
  `86d34cc2efbce9814847dd12fc12cab2f46d04168ce39905bfd62a93ced783fd`;
- Source state-bank records:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-01-annotated-owner-direct-c-d0-pilot/materialization-v1/base-state-bank/records.jsonl`.

The sealed Phase-A C decode is already a fresh cold natural decode of the exact
reference and is reused as the trajectory authority.  Re-decoding the same 248
images would add cost but no new estimand.  Source prefix tokens and Source
logits are never reused.

## Frozen gain support and materialization

The evidence-linked owner universe is defined before any new model load:

1. take the 63 IoU50 owners gained by D0 over Source in Phase A;
2. retain only owners still missed by C;
3. bind each owner to its authoritative positive complete-row candidate;
4. retain only C reachability class 1 or 2: exact C token-prefix reachability,
   or the same locked physical-owner set and row boundary with different C
   tokens;
5. rematerialize the generated prefix from the C token trace at that row
   boundary and reject any dropped/malformed row before it;
6. activate at most one owner per image: choose the latest admitted C row
   boundary, then deterministic event/candidate identity as a tie-break.

The other owners remain neutral.  Class-3 owner-set/boundary mismatches and
class-4 absent boundaries are diagnostics, not silently repaired events.
Natural output disorder is legal; the historical insertion boundary already
ensures that the proposed row does not add an ordering inversion.  Every action
is a complete category-and-four-coordinate row; coordinate-wise mixed targets
are forbidden.

For selected action `e`, use the existing field-balanced positive-path loss:

\[
\ell_e=\frac12\left[
  \operatorname{mean}_{u\in\text{schema+description}}-log p(a_u)
  +\operatorname{mean}_{u\in\text{coordinates}}-log p(a_u)
\right].
\]

The gain objective is the equal-image mean
\(L_G=|E|^{-1}\sum_e\ell_e\).  There is no mean preservation term, EOS target,
legality auxiliary, or newly tuned margin temperature in this unit.  Admission
already owns row legality; adding another objective would change the direction
being tested.

## Frozen incumbent support

Start from the five exact behavior-linked Phase-A lesion owner references and
retain only those freshly realized by C.  Do not fill unused capacity with
newly low-margin owners.  Protect C's actual matched emitted row at its exact C
natural prefix, not the stale Source preservation row.

For each retained incumbent row, score every complete-row token under raw
FP32 model logits.  Freeze the weakest selected-versus-best-other decision:

\[
m_i(\theta)=z_{t_i}(\theta)-z_{r_i}(\theta),\qquad
a_i=\nabla_\theta m_i(\theta_C).
\]

All weakest-position or runner-up ties within `1e-7` logit units become
separate constraints.  The total remains capped at 16.  Nonfinite or vanishing
gradients, a negative anchor margin beyond numerical tolerance, unresolved
alignment, or capacity overflow stops the unit.

## Primal and small dual

Let

\[
q=-\nabla_\theta L_G(\theta_C),\qquad
d_0=q/\lVert q\rVert_2,
\]

and normalize each nonzero incumbent gradient
\(\bar a_i=a_i/\lVert a_i\rVert_2\).  The identity metric is frozen because
this unit asks pure local existence on the registered DoRA coordinates; it
does not claim to reproduce a fresh AdamW step.  No unavailable optimizer
state is reconstructed.

Solve only

\[
\boxed{
\min_d\ \frac12\lVert d-d_0\rVert_2^2
\quad\text{s.t.}\quad \bar A d\ge0
}
\]

with zero slack.  A finite margin budget or trust radius would require an
unauthorized step size; this direction-only test instead requires no negative
first-order change in the registered critical margins.

The dual has at most 16 variables:

\[
\min_{\lambda\ge0}
\frac12\lambda^T K\lambda+h^T\lambda,
\quad K=\bar A\bar A^T,
\quad h=\bar A d_0,
\quad d^\star=d_0+\bar A^T\lambda^\star.
\]

Only gradient inner products and the bounded active rows are materialized; no
COCO-scale Jacobian or output-head residual is built.

### Local lemmas

1. The feasible set is a closed convex cone containing zero, so the primal has
   one unique projection.
2. If \(\bar A d_0\ge0\), then \(d^\star=d_0\); projection has no demonstrated
   role.
3. Cone projection gives
   \(q^T d^\star=\lVert q\rVert_2\lVert d^\star\rVert_2^2\).  Therefore
   \(d^\star\ne0\) exactly when the registered cone contains a direction of
   strictly positive first-order gain.
4. \(a_i^Td\ge0\) prevents negative first-order change only.  It does not
   certify a finite step, complete-row execution, natural owner retention, or
   held-out transfer.

## Execution bounds

CPU plan construction precedes model load and freezes all identities,
selections, current C prefixes, candidate rows, and incumbent rows.  The live
run is one FP32 HF process on one GPU, with no workers or collectives.

- gain: one graph forward and one backward per selected image;
- incumbents: one graph forward per retained row and at most 16 VJPs total;
- cache: disabled; only the last complete-row logits are requested;
- gradient surface: explicitly freeze every loaded parameter, then re-enable
  only the 588 named DoRA tensors;
- gradient accumulation: FP32 on the 588 DoRA tensors;
- Gram, dot products, and solve: FP64 on CPU;
- persisted payload: plan plus one compact receipt; gradient vectors are
  hashed but not written;
- parameter updates, optimizer steps, checkpoints, and merged exports: zero.

Before the authoritative run, one non-authoritative sentinel uses the first
gain event and first incumbent to measure graph peak memory and verify exact
model/prefix/row/VJP plumbing.  A sentinel is mechanics evidence only.

## Decision matrix and stop rule

All identity, prefix, row, owner, processor, adapter, trainable-surface, or
parameter before/after mismatches are `MECHANICAL_INVALID_C_SMALL_DUAL`.

- If no evidence-linked C-missed gain action survives, or none of the five
  lesions remains a C incumbent: `SCIENTIFIC_STOP_C_SMALL_DUAL_SUPPORT`.
- If `q` is zero/nonfinite, constraints exceed 16, the dual cannot be
  certified, or predicted gain is numerically ambiguous:
  `SCIENTIFIC_STOP_C_SMALL_DUAL_LOCAL_FEASIBILITY`.
- If every registered incumbent has nonnegative unconstrained directional
  change: `GO_C_ANCHORED_SIMPLE_REFRESHED_VERTICAL`; projection is
  unnecessary on the evidence-supported active set.
- If at least one incumbent is threatened and the certified projection has
  nonzero norm and strictly positive predicted gain:
  `GO_C_ANCHORED_ONE_PROJECTED_UPDATE_VERTICAL`.

Either GO authorizes only a separately contracted one-update vertical slice.
This unit stops after one authoritative receipt and results record.  It may not
add constraints, targets, aliases, rank, learning rate, optimizer state, dose,
or another metric to rescue an ambiguous result.

## Claim boundary

The strongest possible claim is local: on the exact C adapter, exact selected
C prefixes, evidence-linked target actions, full shared language-DoRA tangent,
and surviving exact-lesion margins, a nonzero first-order compatible direction
was or was not found.  The unit cannot establish finite-update preservation,
natural-policy improvement, image-disjoint generalization, missing-label
precision/recall, a covered-set representation, or production readiness.
