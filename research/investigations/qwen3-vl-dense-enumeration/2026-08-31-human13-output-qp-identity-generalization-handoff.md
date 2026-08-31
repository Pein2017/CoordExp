---
title: Human13 shared output-QP identity and generalization handoff
description: Phase-boundary transport from the Image2299 finite-policy compilation witness to a leak-free Human13 shared-readout test.
type: investigation
role: research-handoff
authority: transport_only
status: direction_forked_unit_planned
updated: 2026-08-31
---

# Human13 shared output-QP identity and generalization handoff

Direction instantiated on `2026-08-31` at
`/data/CoordExp/.worktrees/human13-output-qp-identity-generalization`, branch
`probe/human13-output-qp-identity-generalization`, forked from canonical commit
`42752b496`. The active research authority is now the [Human13 shared
output-QP unit](experiments/2026-08-31-human13-shared-output-qp-identity-generalization/unit.md).

This handoff is transport, not experimental authority. A new research unit must
freeze its own exact panel, routes, constraints, controls, attempt budget, and
stop rule before running a model.

## What Image2299 actually established

The completed [Image2299 continuation](2026-08-31-image2299-augmented-greedy-38-to-46-owner-continuation.md)
contains a fresh-cold ordinary-greedy 46/46 witness. The strongest path is the
[direct canonical G46 global QP](experiments/2026-08-31-image2299-canonical-g46-global-qp-protected-null-sentinel/results.md):

\[
z(h)=hE^\top+h\Delta W_{out}^\top .
\]

The tied base embedding/head `E`, language tower, multimodal aligner, and vision
tower stay frozen. The intervention is one sparse output-only residual over 51
vocabulary rows by hidden width 2048. It changes output logits only; it does
not change the corresponding input embeddings. The original checkpoint is not
modified, and the solve uses no `loss.backward()`.

The correct claim is therefore:

> On one fixed image and one oracle-authored complete route, a frozen Qwen3-VL
> plus sparse output-head delta can be compiled into an ordinary-greedy 46/46
> trajectory.

This is a finite output-policy programmability witness. It is not evidence that
the base model learned 46/46, that Qwen3-VL learned dense enumeration, that the
residual transfers to another image, or that production training is ready.
Re-solving a GT-conditioned residual for every test image would be oracle
leakage.

## Post-closeout residual-geometry audit

The lead independently re-read the frozen v3 payload and receipt after an
Opus-max red-team. These are observations, not a new decode:

- The `51 x 2048` direct-G46 residual has Frobenius norm `37.21948`.
- Its first singular direction contains `99.1492%` of squared singular energy;
  participation-ratio effective rank is `1.0172`, and rank for 95% energy is 1.
- Token row `152031` has residual norm `37.06019`; the median selected-row norm
  is `0.19947`, a `185.8x` ratio.
- Across the 57 positive positions, the worst zero-residual margin is
  `-12.37240` at position 301. At the registered margin `0.01`, the six largest
  squared deficits contain `87.10%` of total deficit mass and the ten largest
  contain `95.65%`.
- The model-manifold M41 residual has effective rank `3.0808`; its leading
  direction has absolute cosine `0.04644` with direct G46. The composed
  five-tie payload's leading direction has cosine `0.99948` with M41 because
  that payload contains the M41 component. The composed witness is a distinct
  construction and cold replay, not independent evidence for a shared
  direction.

These measurements narrow the interpretation: direct G46 is dominated by one
large coordinate-token readout edit and one severe local deficit. Total norm
alone is therefore a poor semantic-sharing statistic. Every future solve must
also report the largest-row energy share, effective rank, rank for 95% energy,
and a norm or objective with the largest constraint removed.

### What the concentration does and does not mean

The `99.1492%` statistic says that the solved matrix is effectively rank one:
the selected vocabulary rows mostly use one common hidden-state direction. It
does **not** show that a general owner concept lives in that direction. The
single dominant coordinate row and the near-orthogonality to M41 make a
specimen-specific readout shortcut the strongest current explanation.

The `87.10%` statistic is a concentration of squared margin deficits, not a
fraction of owners, behavior, or intelligence. It says that the minimum-norm
program is numerically driven by six unusually hard token decisions. At the
worst site, the target-versus-competitor margin must move by at least about
`12.38` logits to clear the registered margin. The intervention is therefore
sparse in parameter support but functionally large at its bottleneck states.

Neither statistic alone proves overfitting. A low-rank direction could be a
real shared bottleneck if it transfers to unseen images and beats a matched
null. In the present combination—one GT-authored image, all route states
visible to the solver, excess readout capacity, no held-out test, one dominant
coordinate row—the evidence should be treated as oracle finite-path overfit
until Human13 rejects that explanation.

Finally, this result does not localize the causal disease to `lm_head`. The
experiment allowed correction only at `lm_head`, so every fitted correction is
forced to appear there. It proves linear readout sufficiency on the visited
Image2299 states; it does not prove that frozen hidden states contain a
cross-image owner representation or covered-set memory.

## The Human13 question

The next estimand is deliberately narrower than “can QP fit 13 images?”:

> Can one shared output-only delta, learned without any held-out image's target
> route, hidden states, constraints, or per-image residual, improve natural
> greedy strict-owner coverage on that held-out image while preserving all
> protected owners and not increasing hard debt?

Same-panel joint feasibility is only a mechanics result: with roughly
`|selected rows| x 2048` free parameters and relatively few constraints, a
large readout can memorize many routes. Held-out natural greedy behavior is the
decision owner.

The frozen panel has 13 images but only 12 target-bearing images in the existing
Human13 ledger; the thirteenth is a preservation-only control. Interaction
matrices are therefore `12 x 12`, not `13 x 13`. Re-derive this count from the
frozen ledger before launch rather than trusting this transport note.

## Program objective: compilation to optimizer learning

The output QP is not the intended production model. Its durable roles are:

1. **feasibility oracle**—show whether an intended complete route can exist on
   the current frozen states;
2. **credit-assignment teacher**—identify active target, preservation, debt,
   CONTINUE, and EOS constraints at exact post-insertion states;
3. **minimum-intervention certificate**—separate infeasibility from an
   arbitrary optimizer failure;
4. **red-team control**—compare semantic routes with deficit-matched nulls.

There are two distinct learning milestones:

- **shared readout learning:** one Torch-native output adapter is updated by an
  optimizer across training images and transfers to held-out images without
  per-image QP at inference. This is genuine shared parameter learning, but it
  still changes only the readout;
- **transition-dynamics internalization:** QP-derived constraints supervise an
  optimizer update to the language tower so that owner preservation,
  remaining-owner admission, duplicate suppression, and STOP behavior become
  properties of the model's evolving hidden states.

The current Human13 unit tests whether the first milestone is even supported by
frozen representations. It must not claim the second milestone. If a later
unit trains language-tower DoRA, QP becomes a teacher rather than a payload:

1. decode current natural routes;
2. Hungarian/owner matching builds gained, preserved, debt, and STOP events;
3. teacher-force the complete **post-insertion** routes, not counterfactual
   repaired prefixes that inference will never visit;
4. accumulate multi-image event losses or a local constrained step into one
   optimizer update;
5. re-decode naturally after every accepted update;
6. reject or roll back any update that loses an incumbent owner or increases
   hard debt.

Success for internalization requires one shared checkpoint, no test-time QP,
held-out natural-greedy gain, zero protected-owner/hard-debt regression, and
evidence that the relevant hidden-state separation—not merely one giant output
row—improved. Loss curves and teacher-forced route fit do not qualify.

## Why the former consensus-span idea is retired

Do not first solve per-image residuals and search a nonnegative cone
`sum_i alpha_i d_i`. That cone is a strict subset of the joint QP decision
space, the residuals were solved in different per-image protected-null bases,
and nonnegative coefficients have no semantic justification. A joint convex
solve can return a lower-norm feasible point whenever that cone is feasible.

Per-image residuals remain diagnostic only. From cached states, compute:

- a protection-damage matrix: how much residual `d_i` consumes image `j`'s
  incumbent margins; and
- a transfer matrix: how much `d_i` reduces image `j`'s target deficits.

Compare both against deficit-matched null residuals. No per-image residual may
enter an evaluation path.

## Proposed smallest research ladder

### Stage -1: recover and bind the panel

The historical Human13 JSONL outputs may have been reclaimed from the live
artifact tree. Rebuild them from their frozen sources, recheck the published
hashes, and bind the exact Source checkpoint, prompt, tokenizer, evaluator,
owner ledger, image bytes, and geometry convention. The historical builder can
be replayed from `research-base-v2`; recover only the exact files the new unit
uses. Do not restore the retired Human13 runner wholesale.

### Stage 0: capacity census before a solver

Teacher-force each Source route and intended post-insertion target route at
zero residual, then cache final hidden states. As images are pooled, report:

1. singular spectrum and numerical rank of the protected-state matrix;
2. remaining exact protected-null dimension;
3. projected norm of every positive state;
4. the complete per-position baseline-deficit table.

Hard protected-null was valid for Image2299 because its protected bank was
smaller than hidden width 2048. Across Human13 the pooled bank is expected to
span the full hidden space. If the exact nullspace collapses, retire exact
multi-image protected-null; do not misreport the resulting norm explosion as a
semantic failure.

### Stage 1: two-image joint mechanics plus matched null

Use two target-bearing images that bracket target burden. Compare semantic
target routes with same-image deficit-matched decoy rows. Each decoy must match
wrapper, category, token count, box area/aspect distribution, and per-position
baseline deficit profile while having maximum IoU below `0.1` to every GT box.
Freeze the decoy selection rule before solving.

Solve one shared output residual directly. Replace exact hidden-state nulling
with hard margin preservation on the complete intended post-insertion routes:

\[
\min_{\Delta W,\xi\ge0}
\tfrac12\lVert\Delta W\rVert_F^2+C\sum_g\xi_g
\]

subject to:

- every incumbent-owner route token retaining its frozen safety margin;
- every confirmed duplicate or unsupported event losing only where a known
  positive alternative exists;
- CONTINUE beating EOS while coverage is incomplete and EOS winning at the
  completed terminal;
- each gained-owner alias meeting its margin with owner-level slack `xi_g`.

`Delta W = 0` must satisfy the safety-only constraints by construction. Owner
OR-alias selection stays outside the convex solve and is reselected only at a
declared outer boundary. Use cutting planes plus a full-vocabulary final check,
not a fixed small competitor list.

Decision readouts are fresh-cold ordinary greedy at repetition penalty 1.0,
with repetition penalty 1.10 as a robustness stress. Report gained, retained,
lost, duplicate, unsupported, malformed, cap-stop, STOP/EOS, norm, effective
rank, row-energy concentration, and primal-dual certificate per image.

### Stage 2: four-image leave-one-out transfer

Move held-out evidence forward. For each of four folds, build all target
routes, states, and constraints from three images only; apply the one shared
residual cold to the fourth image's original prompt. The held-out image may
contribute only its pre-frozen evaluation ledger after the solve. Run the same
folds for matched nulls when the semantic arm is viable.

Primary outcome: held-out per-image net strict-owner delta at repetition
penalty 1.0, subject to zero Source-owner loss and zero hard-debt increase on
every train and held-out image.

### Stage 3: 12 target-bearing images plus one preservation control

Proceed only if Stage 2 shows positive held-out transfer that is stronger than
the matched null. Use leave-one-out shared checkpoints; never solve or select a
route on the held-out image. The preservation-only thirteenth image must not
regress.

## Pre-registered interpretations and stop rules

- **LOOKUP:** semantic and deficit-matched null have comparable feasibility,
  rank, residual cosine, and off-diagonal transfer. Stop; same-panel success is
  finite lookup-table superposition.
- **NO TRANSFER:** at least three of four Stage-2 held-out folds have nonpositive
  net owner gain, or any held-out Source owner is lost. Stop before N=13.
- **CAPACITY EXHAUSTION:** required norm grows as pooled protected-null dimension
  collapses. Retire exact nulling and use margin preservation; this is not a
  semantic verdict.
- **SHARED READOUT EVIDENCE:** semantic residuals show stronger off-diagonal
  transfer and lower effective-rank burden than matched nulls, plus positive
  held-out natural-greedy gains with zero loss/debt. This licenses a shared
  Torch-native output-adapter training unit, not production.
- Any route, matcher, margin, null, or ledger change after seeing a decode opens
  a new unit. Mechanical invalidity is not scientific evidence.

## Parameter escalation boundary

1. Use direct shared output QP only as the cheapest representation/constraint
   discriminator. If it transfers, open a separate shared Torch `nn.Parameter`
   adapter unit trained by an optimizer; do not ship the QP payload.
2. Move to language-tower DoRA only if held-out targets are not linearly
   decodable from cached final hidden states at the same rank budget, or a
   certified shared-output constraint conflict remains after rank is relaxed.
   That successor must use QP active constraints as teacher signals and judge
   every update by natural greedy, not by teacher-forced loss.
3. Consider the multimodal aligner only if pre-aligner visual tokens separate
   the missing owners while post-aligner states do not.
4. Consider the vision tower only if pre-aligner region/instance features also
   fail to separate them.
5. Keep tied input embeddings frozen last. Updating the tied `E` changes both
   input dynamics and all output rows, destroys the convex readout problem, and
   has the largest blast radius. If readout capacity is needed, first raise the
   explicit untied output-adapter rank.

## Code and lifecycle boundary

No Image2299 QP runner is promoted into `research-probes`. The successful code
is a deeply specimen-bound chain with Image2299 routes, hashes, private helper
imports, and cold verifiers. Preserve it under the annotated
`probe-final/image2299-mechanism-microscope` tag. In the new Human13 direction,
implement only the smallest pure mechanics actually consumed. Extract a shared
module only after Human13 becomes a real second-direction consumer.

The generic `sparse_target_site_ce` helper is also not required by this first
QP unit and has only Image2299 callers today, so it is not promoted at this
boundary.

## Fresh-chat continuation contract

The recommended continuation is a new chat in this existing worktree. This is
a design-to-implementation boundary; the long Image2299 transcript is no
longer required to act safely.

First actions, in order:

1. verify `pwd`, branch, `git status`, and current `HEAD`;
2. read the active [unit](experiments/2026-08-31-human13-shared-output-qp-identity-generalization/unit.md),
   then this handoff and the minimum reading path below;
3. re-resolve the panel/checkpoint/artifact paths and hashes—historical output
   paths are volatile and may have been reclaimed;
4. implement only Stage -1 recovery/parity and the Stage-0 pooled-rank census;
5. freeze its forward-count, wall-time, memory, cache, and artifact bounds
   before any GPU launch;
6. stop after the Stage-0 receipt and choose exact protected-null versus margin
   preservation from measured rank. Do not implement Stage 1 speculatively.

Current execution state: no Human13 code, model forward, solver run, optimizer
step, GPU launch, checkpoint, or scientific artifact exists on this branch.
This handoff conveys no launch authority. A fresh chat must obtain the user's
material-cost decision after Stage -1 identities and Stage-0 bounds are frozen.

Volatile facts to reverify include live artifact existence, Source checkpoint
availability, GPU/process state, evaluator imports, and the exact 12-target plus
1-preservation ledger count. Stable facts are the branch/worktree identity,
the linked committed records, and the archived Image2299 tag.

## Reading path for the next session

1. this handoff;
2. the [Image2299 continuation](2026-08-31-image2299-augmented-greedy-38-to-46-owner-continuation.md);
3. [direct G46 results](experiments/2026-08-31-image2299-canonical-g46-global-qp-protected-null-sentinel/results.md);
4. [matched QP versus CE ablation](experiments/2026-08-31-image2299-matched-sequence-optimizer-ablation/results.md);
5. [Human13 owner-credit N/K factorial](experiments/2026-08-22-human13-owner-credit-nk-factorial/results.md);
6. [prospective 13-image panel admission](experiments/2026-08-04-sorted-prospective-13-image-panel-admission/unit.md).

`HOLD_PRODUCTION` remains the standing disposition until a shared checkpoint
shows held-out natural-greedy gain without protected-owner or hard-debt loss.
