---
title: Joint DoRA Magnitude Prox-Linear Learning on Fixed N2
description: Compare a joint soft-margin quadratic update with Adam on the same complete two-image generation routes and internal magnitude parameters.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: authorized_within_goal
unit_id: 2026-09-05-dora-prox-linear-n2
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
updated: 2026-09-05
---

# Fixed N2 internal magnitude learning

## Terminal disposition

The [result](results.md) closes this unit after its one frozen paired run:
neither arm reaches common complete fit, and the bounded inexact QP profile
has no observed optimization advantage over same-margin Adam. Mechanics and
cold persistence are valid; this is not an exact-QP impossibility result.
No additional training, numerical sweep, or panel expansion is active.

## Active contract and authority

From exact step-2444 Source, does a jointly solved, repeatedly relinearized
soft-margin quadratic program improve the compute-to-complete-fit trade-off
over Adam on the identical squared-margin objective, using only the existing
DoRA magnitude parameters and the frozen two-image panel?

The user authorized autonomous implementation and bounded execution on
2026-09-05, made all GPUs available (background stress activity is expected to
yield to research work), and explicitly created this worktree from the prior
Human13 checkout. That fork instruction supersedes the earlier proposed
research-probes base. Current lead owns the scientific contract and acceptance;
one builder owns the new runner and its tests; a fresh executor owns each
launch. No new top-level task, further delegation layer, or runtime cleanup is
authorized by this packet.

This research-local unit owns the new hypothesis. It does not reopen or change
the old tangent-oracle OpenSpec or immutable Human13 receipts, and creates no
reusable production interface requiring a new OpenSpec change.

## Frozen specimen and provenance

- worktree: `/data/CoordExp/.worktrees/dora-prox-linear-n2`
- branch: `probe/dora-prox-linear-n2`
- inherited source commit: `a2c049453aa6356320fe29299daade15c3e0efb5`
- checkpoint: `/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/2026-08-05-closeout/artifacts/training/four-coordinate-xy/checkpoints/step-2444`
- Source adapter SHA-256: `49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da`
- frozen source/panel/config identities and existing runtime bindings:
  [original N2 packet](../2026-09-01-human13-dora-magnitude-finite-overfit/n2-launch-packet-v1.json)
- images: `6040`, `16228`; 65 annotated owners; 592 complete decision positions
  (136 and 456), including EOS; four-coordinate XYXY canonical routes.
- trainable: 196 language-DoRA magnitude vectors, 573,440 scalars. Freeze all
  A/B, base, vision, aligner, embedding delta, and output-head parameters.
- runtime: existing HF FP32 unmerged runtime, exact admitted language JVP/VJP,
  full vocabulary width 152,670; natural greedy batch one, RP1.0.
- output root: `/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-09-05-dora-prox-linear-n2/`

The predecessor scripts are inherited tracked code, not cross-worktree runtime
imports. Reuse their specimen capture, magnitude surface, output wrapper,
adapter-only persistence, Source restoration, and cold natural evaluator.
Existing CE+AdamW N2 completed 70 updates in 142.683 seconds with about 30 GiB
reserved GPU memory. That is historical context, not a fresh timing comparator.

The Pro critique supplied at
`/data/CoordExp/.codex/attachments/3b980c85-8e99-4777-aad0-43e01c17f34c/pasted-text.txt`
motivates abandoning hard readout-removal continuation. It is advisory
provenance, not runtime authority. The selected route is direct internal
learning; no external output residual, alpha curriculum, or distillation arm.

## Objective and contrast

Let p index all 592 positions across both images, t_p its target, and c a
non-target token. Set gamma=0.01 and g_pc=z_pt-z_pc. Both arms minimize

    F(m) = (1/(2*592)) sum_p [max(0, max_(c != t_p)(gamma-g_pc(m)))]^2.

This preserves the predecessor's total-token weighting rather than introducing
image-normalized weighting. Both images are one mathematical batch; independent
forwards do not imply independent parameter updates. No token-position subset
may replace the declared objective.

Adam arm: AdamW with lr=0.003, betas=(0.9,0.999), eps=1e-8, weight_decay=0,
no scheduler, seed=0; fresh optimizer from Source.

QP arm: at each accepted current parameter state, recompute logits/Jacobian J
and solve the prox-linear subproblem

    min_(d,s>=0) ||d||^2/(2*eta) + sum_p s_p^2/(2*592)
    subject to g_pc + J_pc d >= gamma-s_p for every competitor c.

All competitors at a position SHARE one slack. This is not a sum of separate
competitor hinge losses. On a restricted cut set, with lambda>=0,

    dual(lambda) = b^T lambda - eta/2 ||J^T lambda||^2
                   - 592/2 sum_p (sum_(cuts at p) lambda_pc)^2,
    d = eta J^T lambda, b_pc=gamma-g_pc.

Use exact JVP/VJP and existing chunked vocabulary scanning; do not materialize
the full vocabulary Jacobian or average independently solved image updates.
Initialize cuts with current worst competitors and separate the candidate
linearized full vocabulary; record inner optimality/gap/separation residuals.
Bounded inexact solves are allowed only when labeled inexact and delivering
model descent; a restricted-cut solution is never called a full certificate.

Globalization: eta starts at 1.0. Predicted loss decrease is
`F(current)-F_hat(d)`, excluding the norm penalty; require both this decrease
and the separate penalized prox-model decrease to be positive. Accept only
positive actual same-batch decrease with rho=actual/predicted_loss >=0.1. Reject by
exactly restoring the current iterate and halving eta, at most five rejected
proposals per outer step. On rho>=0.75, double eta for the next step up to 1e4;
otherwise retain it. Stop if eta<1e-6 or no usable model descent remains.
Each accepted update must relinearize the language model; frozen vision inputs
may be reused because all pre-language parameters are fixed. Never validate
updated language parameters with an old language KV cache.

## Decision-owning evidence and boundaries

Check real full-vocabulary margins after every outer update for both arms.
The common complete-fit gate is every margin >=0.00998, followed by adapter-only
save, independent-process cold readback, and native greedy 65/65 at IoU50/60/80,
natural EOS, and zero confirmed duplicate, malformed, invalid-box, or cap debt.
Reuse the original per-image N2 evaluator, not its legacy IoU50-only aggregate
or its automatic unmatched-as-debt predicate. Valid unmatched alone is not a
HOLD condition. If complete cold route margins pass but same-semantics greedy
diverges from that route, classify a mechanical parity mismatch.
The complete route is a sufficient-condition optimization target here, not a
claim that detection requires one canonical ordering. Valid unmatched outputs
must remain separately reported, not automatically labeled hallucinations.

Primary comparison: first common success versus total measured compute; include
loading, capture, inner iterations, JVP/VJP, separation, rejection, and checks.
Report training and cold-validation costs separately and together. When a gate
is not reached, report final/best observed loss, worst gap, violated positions,
and natural outcomes if a valid candidate can be cold-read; do not call partial
fit a success. QP outer iterations alone are not a cost metric.

Secondary: final realized-weight distance from Source at equal success. Small
increments are not a minimum-norm endpoint claim. No source regularization is
added. A later low-norm claim would require ruling out ordinary Adam overshoot.

Strongest alternatives: margin loss, not QP, supplies the advantage; small
steps make QP an expensive gradient method; a source-local Jacobian cannot
produce useful finite steps. No seed-repeatability, N4/N13, held-out transfer,
full-scene recall, or production claim is included.

## Execution bounds and stop

Start with one production-shaped two-image/one-update mechanics run and cold
readback, not a distributed system. Each full arm uses one GPU and one model
load; distinct arms may run concurrently on separate available GPUs. GPU stress
occupancy is not a reason to wait, kill unrelated processes, or change precision.

Initial safety ceilings (not scientific evidence): mechanics run 900 seconds;
each full training arm 1,800 seconds from process training entry, at most 300
outer updates; each cold verification 600 seconds. Cooperatively check the time
budget within QP objective/oracle and separation loops, publish a terminal
receipt, and restore state rather than waiting for an unbounded solve. Record
actual full process wall time even if an in-flight kernel crosses a boundary.
QP inner: at most 50 optimizer iterations, 100 objective evaluations per cut
solve, and eight cut-expansion solves per outer proposal, all subordinate to
the global time ceiling. Do not spend hours solving an initial inner problem.

### Pre-science inner-work qualification

The first mechanics run, `smoke-v1/qp/terminal.json` under the output root,
exhausted its 900-second budget during the first proposal (191 dual objectives
across four cut solves). It restored Source and persisted an unchanged adapter;
no update or cold verification was admitted. This is execution-cost evidence,
not an optimizer comparison or a passed end-to-end smoke.

Before any scientific-arm outcome is available, the lead selects one bounded
inexact inner-work profile for the second and final mechanics attempt:
L-BFGS-B requested maxiter=4, maxfun=12, maxls=6; ftol=1e-12 and gtol=1e-7
unchanged; at most two cut solves per proposal. SciPy may exceed its requested
evaluation count, so the hard 100-evaluation guard and global wall limit remain
authoritative. Full-vocabulary separation, positive penalized model decrease,
the unpenalized-loss rho denominator, and actual finite-update checks remain
unchanged. Uncertified directions are labeled inexact. This is a numerical work
qualification within the already allowed inexact-QP family, not an added arm,
loss change, or tolerance relaxation. No more mechanics retries are allocated.

The `smoke-v2` executor mistakenly precreated the runner-owned `qp` output
directory. Entry failed closed with FileExistsError before model load, so that
invocation produced no GPU attempt or scientific evidence. Its log and empty
directory are preserved. The one permitted unchanged-contract technical
replacement uses `smoke-v3`, exactly the v2 runner/profile and budgets; only
the run identifier and correct parent-only directory preparation change. The
lead owns this replacement invocation to avoid another launcher translation.

`smoke-v3` passed the mechanics boundary: one nonzero accepted inexact joint
update, exact Source restoration, and independent cold equality of all 588
adapter tensors. It remained far from complete fit; its natural counts are not
a scientific comparison. The [paired packet](launch-paired-v1.json) binds the
exact evidence, measured memory/time, and frozen numerical options. Both full
arms restart from Source, not the smoke adapter.

Freeze numerical inner stopping tolerances, implementation hashes, GPUs, exact
CLI commands, and measured bounds in one launch packet after lead inspection
of the mechanics run and before the paired scientific run. Do not tune them
against scientific-arm outcomes. A projected memory requirement above one
80-GiB card closes that implementation path; do not silently switch to DDP.

This unit permits one valid paired scientific run, at most two mechanics GPU
attempts, and one unchanged-contract replacement after a demonstrated technical
defect. Failed runs remain immutable. If either arm reaches its budget, report
that bounded result; no implicit hyperparameter sweep or dataset expansion.
Stop after the paired evidence is reduced and recorded, whether positive,
negative, or technically incomplete. Further scientific exploration requires
a new decision rather than treating this authorization as an unlimited loop.
