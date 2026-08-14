# Human-13 K-Trajectory RP-Crossover Screen Design

The user approved the scientific direction and delegated technical closure to
the research lead and a Fable-5-xhigh design reviewer.  Scientific meaning,
population, estimand, outcome, and stop rules are owned by the
[research unit](../../../research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-14-human13-k-trajectory-rp-crossover-screen/unit.md).
Stable experiment-local implementation behavior is owned by the
[OpenSpec change](../../../openspec/changes/add-human13-k-trajectory-rp-crossover-screen/).
This document explains the implementation shape without
duplicating either authority.

## Architecture

Extend the existing Human-13 one-update spine with five narrow components:

1. an RP-aware batch-four sampler that stores exact token histories and
   processed chosen-token log probabilities;
2. a pure trajectory-credit projector that emits immutable row advantages;
3. a sparse Source-boundary greedy compiler over the frozen native alias bank;
4. an exact-AdamW proposal builder plus owner-wise parameter-delta projection;
5. a matrix runner and analyzer that audit every private proposal under both RP
   surfaces and always restore Source.

The design deliberately keeps policy acquisition, scientific label projection,
loss computation, optimizer proposal geometry, and behavioral evaluation in
separate modules.  Existing model assembly, multimodal packing, matcher,
transaction, checkpoint, and HF evaluation modules remain the shared spine.

## Data flow

```text
sealed Source + manifest + frozen alias bank
                |
                v
      seal Source baselines at rp=1.0/1.10
                |
                v
 fresh K16 batch decode for one (training RP, seed group)
                |
                v
 parse + match + first-hit/cost ledger + detached RLOO rows
                |
          +-----+--------------------+
          |                          |
          v                          v
 trajectory-credit packs       Source-boundary compiler pack
          |                          |
          +-------------+------------+
                        v
           exact fresh-AdamW proposal
                        |
        +---------------+----------------+
        |               |                |
        v               v                v
      arm A           arm B       arm C proposal projection
        |               |                |
        +---------------+----------------+
                        v
        private apply -> dual RP clean-greedy audits
                        |
                        v
        owner/burden receipts -> exact transaction restore
```

## Numerical contract

RP is part of the policy.  A replay token is admissible only when its prompt
history, generated history, processor order, RP value, temperature, token ID,
and processed chosen-token log probability are bound.  The vertical seals a
numeric parity tolerance before scientific execution.  A breach stops the run;
the implementation cannot silently relabel an approximate replay as exact.

Fresh owner weights are not estimated from the same K trajectories.  They are
uniform over the frozen trusted owners in each image.  The matcher, weights,
advantages, compiler selectors, and preservation witnesses are detached.
Terminal STOP retains negative credit but cannot receive positive imitation.
Logical reduction uses one global image-by-trajectory denominator; physical
packs and gradient-accumulation boundaries cannot reweight the estimator.
Legacy-M row tokens are masked from direct score-function credit but remain in
the causal history for later trusted actions.

The compiler uses repetition-penalty-processed greedy logits without the
sampling temperature.  Temperature remains part of trajectory likelihood only.

## Proposal semantics

All arms use one AdamW step at the same globally frozen learning rate.  The
default is `3e-6`; a disjoint qualification-only ray may replace it once using
the research unit's mechanics-only floor/ceiling.  Owner outcomes and gradient/
delta norms cannot select dose, and no matrix cell adapts it.  Realized
parameter norms are recorded rather than normalized after the fact.  The
preservation arm operates on the exact reconstructed parameter delta and
projects it in the fixed AdamW coordinate metric.  The applied delta, not the
unprojected gradient, is the object audited by both clean-greedy surfaces.
Finite realized witness degradation is reported and still receives both
behavioral audits; only an uncertified/non-finite projection or wrong applied
delta invalidates the cell.

Because every proposal is discarded, this screen does not implement optimizer-
moment continuation for a projected step.  A positive preservation result can
justify that later infrastructure; it cannot silently authorize it now.

## Isolation and lifecycle

- Acquisition artifacts are immutable and shared only by the three arms in the
  same `(training RP, seed group)` cell.
- Every arm receives an independent Source model and fresh optimizer state.
- Private proposals are never published as accepted checkpoints.
- Both clean-greedy audits complete before transaction restore.
- A failed audit, parity check, or rollback writes a durable failure receipt and
  stops its affected run identity; there is no blind retry.
- The matrix runner cannot adapt learning rate, utility, costs, candidates, or
  seeds after observing an outcome; it consumes one content-bound global dose
  receipt produced before matrix materialization.
- Qualification uses disjoint seeds and its owner outcomes are excluded from
  matrix analysis and dose selection.  Its sole permitted setting decision is
  the predeclared mechanical selection among
  `{3e-7,1e-6,3e-6,1e-5,3e-5}`.

## Testing and execution gates

CPU/tensor tests own policy-transform parity, first-hit credit, row return-to-go,
STOP sign, normalized compiler bounds, exact AdamW reconstruction, projection
feasibility, dual-baseline accounting, and deterministic receipts.  A real
vertical then owns sampler/replay numeric parity, the fixed qualification dose
ray, one backward/proposal per training RP at the selected global dose, witness
finite differences, two RP audits per proposal, and byte-identical rollback.  Only
that vertical can admit the eighteen-cell model-quality screen.

## Scope discipline

The change is experiment-local.  It must not create a generic reinforcement-
learning trainer, online controller service, new owner architecture, new public
inference policy, or scale-time checkpoint promotion path.  No validation or
K-miss route is included.  The shortest implementation that preserves the
research-unit semantics wins.

The independent Fable-5 reviews found no mathematical blocker and required
the corrections now built into this design: exogenous owner weights, explicit
sampler/replay parity, one-sided STOP credit, RP-specific Source baselines, and
qualification-selected globally fixed learning-rate proposal accounting.

## Research review routing

Use the repository `agent-routing` discipline only for decisions that could
change the objective, statistical interpretation, experiment boundary, or
go/no-go conclusion.  Fable-5-xhigh/max and GPT-5.6-sol-max are peer principal
research reviewers for those questions and are read-only by default.  The
research lead reconciles their evidence; neither reviewer edits authority
documents or independently authorizes implementation or execution.  Routine
implementation choices stay with the owning worker and verifier.
