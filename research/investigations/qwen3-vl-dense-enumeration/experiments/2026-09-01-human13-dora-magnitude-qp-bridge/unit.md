---
title: Human13 DoRA Magnitude-QP Bridge
description: A nested same-panel test of whether a minimum-norm update to the existing language-DoRA magnitude vectors can move the Human13 output policy into the unmerged network.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-09-01-human13-dora-magnitude-qp-bridge
topic: qwen3-vl-dense-enumeration
status: deferred_after_operator_smoke
evidence_status: mechanics_partial
updated: 2026-09-01
---

# Human13 DoRA Magnitude-QP Bridge

## Routing disposition

The exact operator tracer bullet is mechanically viable, but this QP is no
longer on the Human13 critical path.  The direct finite magnitude-only unit
[`2026-09-01-human13-dora-magnitude-finite-overfit`](../2026-09-01-human13-dora-magnitude-finite-overfit/unit.md)
now owns the `N2 -> N4 -> N13` fitting ladder.  QP, Farkas, distributed solver,
and recovery work remain deferred until a finite N2 candidate exists; this unit
is retained for the separate minimum-norm and norm-scaling question.

The latest source-logit-anchored live sentinel is immutable at
`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-01-human13-dora-magnitude-qp-bridge/20260901T-source-logit-anchor-v3/image-6040-operator.json`
(file SHA-256
`47c337738f8b18752ad24d6d0e9a1bbdb5f94f467aad892f23958589e0cd2a32`).
It is `OPERATOR_PASS`: actual output-wrapper replay and language Source parity
are exact, the exhaustive `152670`-row separator runs in 38 chunks, adjoint
relative error is `5.88e-7`, Source restoration passes, peak reserved GPU is
`29,867,638,784` bytes, and elapsed time is `12.54` seconds.  The separately
collapsed selected-delta head differs from actual Source logits by
`6.87e-5`, so separator base gaps are now anchored to actual wrapper logits;
effective rows are used only for the exact hidden-state derivative.  This is a
single-image mechanics sentinel, not the complete N2 QP smoke or a scientific
model result.

## Active contract

> From the exact unmerged step-2444 Source, can a source-linearized,
> full-vocabulary minimum-norm update shared by all 196 language-DoRA
> magnitude vectors survive finite unmerged replay and fresh natural greedy on
> the frozen Human13 `N2 -> N4 -> N13` ladder?

This is a same-panel overfit bridge.  Every stage may use its complete canonical
target routes.  The decision-bearing outcome is the finite unmerged model and
its natural decode, not the linearized quadratic program (QP) alone.  A change
to the Source, adapter target set, canonical routes, margin, finite replay,
natural evaluator, or nested stage membership is a phase reset.

Direction-local mechanics are preserved at
`probe/human13-output-qp-identity-generalization@a2c049453:openspec/changes/add-human13-dora-magnitude-tangent-oracle/proposal.md`;
they are not promoted into canonical `research-probes` by this records-only return.
OpenSpec acceptance proves mechanics only; this unit owns the panel, stages,
margin, scientific statuses, and claim boundary.

## Origin and semantic boundary

The predecessor used one shared output-head residual, not thirteen per-image
residuals.  It passed `65/65`, `123/123`, and `392/392` owner gates at IoU50,
IoU60, and IoU80 under fresh-cold RP1.0 natural greedy, with zero hard debt and
natural EOS.  That proves fixed-panel shared-output compilability only.  The
residual is an external output-readout intervention and is not a learned model
identity.

This unit moves the intervention into the existing language blocks.  It keeps
the universal base, special-token embedding delta, all DoRA `A/B` factors,
vision tower, aligner, tied input embedding, and output head frozen.  Only the
196 existing language-DoRA magnitude vectors may change.  The durable identity
is always:

```text
immutable universal base package + one unmerged DoRA adapter
```

No output residual, per-image payload, retrieval table, inference-time adapter
selection, or durable merged checkpoint is permitted.  A temporary in-memory
application is allowed for candidate evaluation; a temporary export is allowed
only for fresh unmerged readback and is not a model identity.

## Immutable Source and panel

- checkpoint:
  `/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444`
- universal base:
  `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`
- Source adapter SHA-256:
  `49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da`
- special-token embedding tensor SHA-256:
  `a41cbb2fd05e3f6b7ad43f28f9fc5ce973477812435acf8d79d9b2b61f19e2f2`
- base `config.json` SHA-256:
  `c7d172360d0ff881db59a6f34865c379bbef40d976ad79cfe5fbbf50483655de`
- tokenizer SHA-256:
  `ca7e80dee65c629af3b314e76a7587490db3f4e6412df4af9f3b690a9e9916f8`
- prompt/config authority:
  `configs/coordexp_swift/infer/qwen3_vl_2b_static_dynamic_owner_interface_s_step2444_h0.yaml`
- prompt/config SHA-256:
  `d2217208bc3e419bc9d8c621b4842da262d216780eb1dfc88eade0edf1358f6b`
- Human13 panel:
  `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-05-static-dynamic-owner-interface-crossover/inputs/human-refined-13.geo_sorted_xy.coord.jsonl`
- panel SHA-256:
  `5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23`
- geometry/order: four-token `XYXY`, training transcript preference
  `geo_sorted_xy=(x1,y1)`; natural ordering violations are legal model behavior,
  not image-exclusion conditions.

The runner must freshly bind executed prompt tokens, canonical target tokens,
full-vocabulary width, PEFT version, unmerged state, and the exact trainable
surface.  Expected magnitude surface: 196 vectors and 573,440 scalars.  Any
trainable `A`, `B`, base, embedding, vision, aligner, or output-head parameter is
mechanically invalid.

## Exact magnitude geometry

For language module `l`, freeze

\[
V_l = W_{0,l} + s_l B_l A_l,
\qquad
U_l[i,:] = V_l[i,:] / \lVert V_l[i,:] \rVert_2.
\]

PEFT DoRA realizes

\[
W_l(m_l) = \operatorname{diag}(m_l) U_l.
\]

Therefore, for fixed `A/B`, a finite magnitude update obeys

\[
\Delta W_l = \operatorname{diag}(\delta m_l) U_l,
\qquad
\boxed{\lVert\delta m\rVert_2^2
= \sum_l \lVert\Delta W_l\rVert_F^2}.
\]

The magnitude coordinates are thus an orthonormal basis in the direct-sum
realized-weight Frobenius metric.  This removes the factor gauge and makes the
QP norm physically interpretable.  It does **not** make whole-model logits
linear: attention, normalization, activation functions, and changes at
multiple layers produce finite higher-order terms.

## Source-linearized QP and scalable operator

At canonical decision position `p`, let target token be `t_p`, competitor be
`c`, Source logits be `z^0`, and margin be `gamma=0.01`:

\[
g^0_{pc}=z^0_{p,t_p}-z^0_{p,c},\quad
a_{pc}=\nabla_m(z_{p,t_p}-z_{p,c})\rvert_{m_0},\quad
b_{pc}=\gamma-g^0_{pc}.
\]

The registered linearized problem is

\[
\min_d \frac12\lVert d\rVert_2^2
\quad\text{subject to}\quad
a_{pc}^{\top}d\ge b_{pc}
\quad\text{for every }p\text{ and }c\ne t_p.
\]

Its nonnegative dual is

\[
\max_{\lambda\ge0}
b^{\top}\lambda-\frac12\lVert A^{\top}\lambda\rVert_2^2,
\qquad d=A^{\top}\lambda.
\]

The implementation must not materialize the full constraint Jacobian.  Let
`H(m)` be all canonical pre-head decision states and let frozen head row `E_c`
produce token `c`.  For active dual weights, construct

\[
q_p=\sum_{i:p_i=p}\lambda_i(E_{t_i}-E_{c_i}).
\]

One reverse vector-Jacobian product gives

\[
d=J_H^{\top}q,
\]

and one exact Jacobian-vector product gives `D=J_H d`.  Active dual gradients
and the exhaustive separator then use only

\[
(E_t-E_c)^{\top}D_p,
\qquad
c_p^*=\arg\max_{c\ne t_p}(z^0_{pc}+D_p^{\top}E_c).
\]

Vocabulary rows may be scanned in chunks.  This identity makes the operator
additive across image shards: distributed ranks sum their local reverse
products, then apply the shared direction to local forward products.  The old
output-QP competitors may seed cuts but never certify this problem, because a
language-block update can move every vocabulary logit.

No central finite difference may own a QP certificate.  If the production
operator lacks an exact Jacobian-vector product, the stage is
`JVP_ORACLE_HOLD`; an exact explicit-gradient N2 fallback may be designed after
that observed failure, but is not pre-built here.

## Nested stages and decision gates

Each stage independently starts from the exact Source.  A previous dual/active
set may warm-start mechanics, but a previous candidate adapter is not the next
stage's anchor.

| Stage | Frozen image IDs | Owners | Canonical decisions |
| --- | --- | ---: | ---: |
| N2 | `6040, 16228` | 65 | 592 |
| N4 | `4134, 6040, 13923, 16228` | 123 | 1,147 |
| N13 | `1584, 2299, 2685, 4134, 5001, 6040, 7511, 10707, 13348, 13923, 14038, 14439, 16228` | 392 | 3,637 |

### Stage -1: production-shaped operator smoke

On both N2 images and all 592 positions:

1. prove zero-direction Source parity and frozen-head factorization;
2. use a deterministic nonzero active dual vector;
3. compute one reverse product and one exact forward product;
4. require the adjoint identity
   `q dot (J_H d) = ||d||_2^2` within the registered numerical tolerance;
5. scan the complete vocabulary once; and
6. restore and hash-check every Source magnitude vector.

If exact forward AD requires the SDPA math backend, automatic-SDPA and
math-SDPA must preserve target IDs, worst-competitor IDs, and registered
Source margins before the operator is admitted.  The receipt records max/mean
logit difference, wall time, forwards, backwards, peak GPU reserved memory,
host RSS, vocabulary chunks, and restoration identity.  This smoke proves only
the operator path.

### Stage 0: linearized QP

- initialize at least one Source worst-competitor cut per position;
- solve the restricted dual, exhaustively separate all vocabulary competitors,
  add the worst violated competitor per position, and repeat;
- accept only with full-vocabulary primal violation at most `2e-5`, projected
  dual KKT residual at most `2e-5`, and a finite recorded primal-dual gap;
- report minimum `||d||_2`, which exactly equals the combined realized-weight
  Frobenius norm; there is no imported G0 radius or arbitrary norm cap; and
- permit one unchanged-problem numerical polish, not a different margin,
  metric, stage, route, or regularizer.

`LINEARIZED_QP_PASS` is fixed-prefix first-order evidence only.  Solver failure
without a valid certificate is neutral `QP_SOLVER_HOLD`.  A valid Farkas
certificate can establish linear infeasibility for magnitude-only, but cannot
reject full `A/B/m` DoRA.

### Stage 1: finite unmerged bridge

Apply `m_0+d` to an ephemeral unmerged adapter and freshly recompute every
canonical target-versus-full-vocabulary margin.  Restore and hash-check Source
after every accepted or rejected candidate.  Acceptance requires:

1. every finite canonical margin at least `0.01-2e-5` including terminal EOS;
2. one adapter-only save and fresh base-plus-unmerged-adapter readback with the
   same finite margins; and
3. no changed tensor outside the 196 magnitude vectors.

A linearized pass followed by finite failure is `FINITE_LINEARIZATION_FAIL`.
It rejects this one source-linearized candidate, not magnitude-only fitting or
full DoRA.

### Stage 2: natural greedy

At RP1.0, one fresh process per verification cell must decode from the original
prompt without teacher forcing and reach all owners at strict same-category,
global one-to-one IoU50.  Report IoU50/60/80, exact-route replay, duplicate,
unmatched, malformed, invalid-box, token-cap, ordering-violation, and EOS
counters.  Required stage totals are `65/65`, `123/123`, or `392/392`, zero
confirmed hard debt, and natural EOS for every image.  RP1.10 is a nonblocking
monitor.

N4 launches only after N2 passes Stages 0--2.  N13 launches only after N4
passes.  Each stage gets one scientific solve, one unchanged-problem numerical
polish, and at most one fresh rerun after a demonstrated mechanical repair.  No
line search, sequential relinearization, new norm cap, or optimizer sweep may
rescue a failed finite candidate inside this unit.

## Outcomes and claim boundary

- `MECHANICAL_INVALID`: identity, trainable-surface, operator, vocabulary,
  restoration, persistence, cold-readback, evaluator, or receipt mismatch.
- `JVP_ORACLE_HOLD`: no exact admitted Jacobian-vector product.
- `QP_SOLVER_HOLD`: no valid primal/dual disposition before the frozen limit.
- `M_ONLY_LINEAR_INFEASIBLE`: valid full-vocabulary Farkas certificate.
- `FINITE_LINEARIZATION_FAIL`: linearized QP passes but its finite candidate
  misses a canonical margin.
- `N*_NATURAL_FAIL`: finite canonical replay passes but the registered natural
  owner/debt/EOS gate fails.
- `N*_PASS`: linearized, finite, cold-unmerged, and natural gates pass for that
  stage.

The strongest N13 positive claim is:

> On the frozen Human13 panel and exact step-2444 Source, changing only the 196
> shared language-DoRA magnitude vectors produced one unmerged adapter whose
> fresh natural greedy decode compiled all 392 annotated owners without hard
> debt.

This would establish fixed-panel internal-network magnitude-surface
programmability.  It would not establish held-out or distributional
generalization, semantic sharing, SGD learnability, COCO missing-label
handling, full-DoRA necessity, architecture promotion, scalable-system
efficiency, or production readiness.  Every negative before a valid finite
natural result remains bounded to the named operator or candidate; it does not
revive or modify the closed scalable G0 `SCIENTIFIC_STOP`.

## Artifact and resource contract

Logical root:

`/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-01-human13-dora-magnitude-qp-bridge/<run-id>/`

One machine-readable receipt owns identities, stage, active cuts, QP
certificate, finite margins, natural evaluation, invocation counts, resource
peaks, adapter tensor hashes, Source restoration, and final disposition.  Raw
full-vocabulary Jacobians and durable merged weights are forbidden.

Before scale-up, Stage -1 must measure rather than assume forward/JVP/VJP wall
time and memory.  Reference bounds are 573,440 magnitude scalars; hidden
tangent storage of about 4.63 MiB for N2 and 28.41 MiB for N13 in FP32; and a
152,670-row vocabulary scanned in bounded chunks.  N13 may use eight ranks only
after the exact single-rank operator and collective order have passed a
production-shaped vertical slice.
