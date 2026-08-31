---
title: Human13 Shared Output-QP Identity and Generalization
description: A staged test of whether one leak-free shared output residual transfers owner gains across Human13 rather than compiling per-image lookup tables.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: planned
unit_id: 2026-08-31-human13-shared-output-qp-identity-generalization
topic: qwen3-vl-dense-enumeration
status: planned
evidence_status: no_execution
updated: 2026-08-31
---

# Human13 Shared Output-QP Identity and Generalization

## Decision and active estimand

This unit asks whether one output-only parameter delta shared across images can
produce **held-out** natural-greedy owner gains without consuming any held-out
target route, hidden state, constraint, route choice, or image-specific
residual.

The decision-owning outcome is fresh-cold ordinary greedy from the original
prompt. Training-route likelihood, teacher-forced margins, feasibility, norm,
and same-panel fit are mechanism diagnostics only.

The strongest alternative is lookup-table superposition: a high-dimensional
output head can independently compile a finite collection of image-specific
routes without learning a reusable owner or enumeration rule.

## Frozen claim boundary

Image2299's direct G46 result is a valid single-image finite output-policy
compilation witness. It is not a learned base-model result. This unit does not
reuse its residual at inference and does not solve a GT-conditioned residual
for an evaluation image.

The strongest permitted positive claim is:

> On the frozen Human13 panel and Source checkpoint, one shared output-only
> delta trained without the held-out image's route or states improves held-out
> strict-owner coverage under fresh-cold ordinary greedy while preserving all
> Source owners and hard debt.

This would still be panel-bounded shared-readout evidence, not production,
distribution-level generalization, a covered-set controller, or proof that the
vision tower learned new information.

This unit is a representation and transfer discriminator, not the final
internalized-learning treatment. A successful direct QP only licenses a
separate optimizer-trained shared-output-adapter unit. A language-tower update
is a later unit with a different parameter surface and acceptance contract.

## Frozen substrate before execution

The implementation must bind and verify, before the first model forward:

- Source checkpoint and tied-head identity;
- tokenizer, prompt, wrapper, geometry convention, and natural decode recipe;
- exact Human13 image JSONL, image bytes, GT owner ledger, and evaluator;
- Source natural-greedy routes at repetition penalties `1.0` and `1.10`;
- target-bearing versus preservation-only image roles;
- duplicate, unsupported, malformed, cap-stop, and ambiguity policy.

Historical records indicate 12 target-bearing images and one preservation-only
control. Re-derive this from the recovered frozen ledger; a mismatch is a data
gate failure, not a reason to silently rewrite the denominator.

The live historical panel artifacts may have been reclaimed. Rebuild only the
needed inputs from their frozen sources and verify the hashes recorded by the
[prospective panel-admission unit](../2026-08-04-sorted-prospective-13-image-panel-admission/unit.md).
Recover the historical builder from `research-base-v2` only if necessary; do
not restore a retired Human13 runner wholesale.

## Intervention identity

Keep the tied base embedding/head `E`, language tower, multimodal aligner, and
vision tower frozen:

\[
z(h)=hE^\top+h\Delta W_{out}^\top.
\]

`Delta W_out` is one shared parameter across all images. It may be solved as a
convex FP64 program during this mechanics unit or represented by a standard
Torch parameter later. No image-id lookup or per-image payload is allowed.

## Stage -1: input recovery and parity

1. Re-materialize the exact Human13 panel from frozen sources.
2. Verify all published hashes and 13 image references.
3. Reproduce the Source natural-greedy owner/debt ledger on both repetition
   penalties before constructing targets.
4. Stop on checkpoint, tokenizer, prompt, route, geometry, or evaluator drift.

This stage produces no intervention and no scientific result.

## Stage 0: pooled protected-state capacity census

Teacher-force every Source route and every intended post-insertion target route
at zero residual. Cache final hidden states and report, as images are pooled:

1. protected-state matrix shape, singular spectrum, and numerical rank;
2. exact protected-null dimension within hidden width 2048;
3. projected norm of every positive state;
4. every per-position baseline target-versus-competitor deficit;
5. selected-row overlap across images.

### Stage-0 decision

- If the pooled exact nullspace remains nontrivial with well-conditioned
  positive projections, exact protected-null may continue as a mechanics arm.
- If rank saturates hidden width or positive projected norms collapse, exact
  multi-image protected-null is retired. The primary path becomes explicit
  route-margin preservation.
- Norm explosion coupled to nullspace collapse is **capacity exhaustion**, not
  evidence against semantic sharing.

Stage 0 is the first launch target. Do not implement the joint solver until its
receipt fixes the active parameterization.

## Stage 1: N=2 joint QP and matched null

Select two target-bearing images that bracket declared target burden using only
the frozen ledger. Build the complete intended post-insertion route for each.
All preservation constraints are evaluated on those new routes, not on the old
prefixes that the intervention will no longer visit.

Compare:

1. semantic owner rows; and
2. same-image deficit-matched decoy rows.

Decoys must match category, wrapper, token count, area/aspect distribution, and
per-position baseline deficit profile, while maintaining maximum IoU below
`0.1` to every GT box. Freeze candidate generation, matching tolerance, and
tie-break before any solve.

Solve:

\[
\min_{\Delta W,\xi\ge0}
\tfrac12\lVert\Delta W\rVert_F^2+C\sum_g\xi_g
\]

with hard constraints for incumbent route-margin preservation, confirmed hard
debt, and CONTINUE/EOS behavior; gained-owner margins use owner-level slack.
Choose one alias per owner outside the convex solve by a frozen minimum-deficit
rule. Use cutting planes and a final full-vocabulary margin check.

`Delta W = 0` must be feasible for the safety-only program. Otherwise the
constraint compiler is invalid.

## Stage 2: N=4 leave-one-out transfer

Use four frozen images. For each fold, construct targets, cache states, assemble
constraints, and solve on three images only. Apply the one resulting shared
delta cold to the fourth image's original prompt.

The held-out image contributes only its pre-frozen evaluation ledger after the
solve. It contributes no route, hidden state, constraint, residual, alias, or
hyperparameter choice.

Run semantic folds first. Run matched-null folds only while needed to decide
shared structure versus lookup.

### Primary metric

Per-fold held-out net strict-owner delta at repetition penalty `1.0`, subject
to:

- zero Source-owner loss on every train and held-out image;
- zero increase in confirmed duplicate, unsupported, malformed, or cap-stop
  debt;
- valid terminal EOS;
- one warm and one fresh-cold replay with identical payload and surface hashes.

Repetition penalty `1.10` is a robustness readout. Failure there blocks a
robustness or production claim but does not erase a qualified `1.0` geometry
result.

## Stage 3: full panel

Proceed only after positive N=4 leave-one-out evidence stronger than matched
null. Use leave-one-out over the 12 target-bearing images and retain the
thirteenth as a pure preservation control. Never report a pooled total before
the legacy-12 and Image2299 slices required by the panel-admission contract.

## Required mechanism readouts

For every per-image diagnostic residual and every joint residual, report:

- selected rows, constraint counts, projected rank, and solver certificate;
- Frobenius norm and normalized norm;
- residual-row norm relative to its frozen base row and the maximum required
  target-versus-competitor logit-margin shift;
- largest-row energy share;
- participation-ratio effective rank and rank for 95% energy;
- objective with the largest single constraint removed;
- semantic and matched-null pairwise residual cosine;
- off-diagonal protection damage and target-deficit transfer.

Also measure functional blast radius on frozen Source and held-out states:
top-1 flips, coordinate-token rank changes, logit KL divergence, and owner/debt
changes. Sparse row support must not be described as a small intervention when
its logit effect is large.

Per-image residuals are diagnostics only and must never be applied in an
evaluation path.

## Decision rules and stop boundary

- **LOOKUP:** semantic and matched null have comparable feasibility, rank,
  cosine, and off-diagonal transfer. Stop without parameter escalation.
- **NO TRANSFER:** at least three of four held-out folds have nonpositive net
  owner gain, or any held-out Source owner is lost. Stop before N=13.
- **SHARED READOUT EVIDENCE:** semantic residuals beat their matched nulls on
  off-diagonal transfer/rank burden and produce positive held-out greedy gains
  with zero owner/debt loss. This licenses a separate optimizer-trained shared
  Torch-adapter unit; it does not establish hidden-state internalization.
- **MECHANICAL INVALID:** any hash, parser, ledger, route, warm/cold, full-vocab,
  or primal-dual failure invalidates that cell and supplies no scientific
  evidence.

Changing the panel, target-route construction, matcher, margin definition, or
null after inspecting a decode starts a new unit. This unit does not tune until
something works.

## Parameter escalation

Do not unfreeze language-tower DoRA, the aligner, vision tower, or tied
embeddings in this unit.

1. Test the shared output surface.
2. Consider language-tower DoRA only after a rank-matched held-out linear probe
   shows that the needed next-owner action is not decodable from final hidden
   states, or a certified shared-output constraint conflict remains.
3. Consider the aligner only if pre-aligner visual tokens separate missing
   owners but post-aligner states do not.
4. Consider the vision tower only if pre-aligner region/instance features also
   fail.
5. Keep tied embeddings last because changing `E` changes both input dynamics
   and every output row and destroys the convex readout problem.

If a separate language-tower unit is later opened, QP supplies active
constraints and feasibility certificates rather than inference-time weights.
That unit must train on complete post-insertion routes, accumulate signals
across images into shared optimizer steps, re-run natural greedy after each
accepted update, and roll back any incumbent-owner or hard-debt regression.

## Code ownership and reuse rule

The archived Image2299 implementation is preserved at
`probe-final/image2299-mechanism-microscope`. Do not copy its 30-plus-file
private import chain. Stage 0 needs only panel recovery, teacher-forced state
capture, and linear-algebra reporting. If Stage 1 proceeds, port the smallest
pure selected-row residual, constraint assembly, solve/certificate, and
full-vocabulary verification seams. Extract shared infrastructure only after
this Human13 unit proves a real second-direction consumer.

## Execution state

No code, model forward, solver run, optimizer update, GPU launch, checkpoint,
or scientific artifact has been created by this unit. Execution requires a
fresh launch decision after Stage -1 identities and Stage-0 resource bounds are
frozen.
