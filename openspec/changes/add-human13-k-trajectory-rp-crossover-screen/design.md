## Context

See [proposal.md](proposal.md) and the owning
[research unit](../../../research/investigations/qwen3-vl-dense-enumeration/experiments/2026-08-14-human13-k-trajectory-rp-crossover-screen/unit.md).
The repository already has production-shaped Human-13 model assembly,
no-padding multimodal packing, compact-logit forwards, HF clean-greedy
evaluation, exact training-state transactions, and immutable artifact helpers.
The predecessor also supplies the sealed Source/K ledger and frozen native
alias bank.  The missing behavior is an experiment-local path from fresh
RP-aware trajectories to a shared credit ledger, an exact optimizer proposal,
and a dual-surface behavioral audit.

The design must preserve three distinctions:

- sampled-policy evidence is not the same as raw-model likelihood;
- stochastic coverage improvement is not the same as greedy transfer; and
- a locally preserved logit witness is not the same as retained decoded owner
  coverage.

## Goals / Non-Goals

**Goals:**

- Reuse all stable production-shaped seams and add only the missing research
  contracts.
- Let all K trajectories contribute signed information without reducing the
  experiment to one chosen suffix or a tied group-union reward.
- Make the compiler and preservation additions nested, observable contrasts.
- Keep one-update proposals private, independently auditable, and exactly
  reversible.
- Resolve update dose once on disjoint qualification evidence without using
  owner outcomes, then keep that learning rate fixed across the matrix.
- Fail before scientific execution when RP replay, optimizer reconstruction,
  projection, or rollback cannot be certified.

**Non-Goals:**

- A generic RL or preference-optimization framework.
- Online multi-update learning, optimizer-moment continuation after a projected
  step, K refresh, or accepted checkpoint promotion.
- K-miss supervision, a new owner architecture, production decode changes, or
  claims outside the sealed Human-13 panel.

## Decisions

### 1. Add one experiment-local evidence pipeline

The matrix runner will compose five small owners: policy acquisition, pure
trajectory-ledger projection, greedy-compiler materialization, optimizer-
proposal construction/preservation, and dual-RP analysis.  They exchange
content-addressed typed artifacts rather than model/backend objects.  Existing
Human-13 model, forward, transaction, matcher, and evaluator modules remain
unchanged unless a real vertical proves one narrow adapter is missing.

Two further experiment-local owners close the live boundary: a witness/dose
measurement owner that turns the sealed Source surfaces into the frozen
owner-wise witness bank and the qualification dose mechanics, and one live
composition that binds native acquisition, exact-surface replay, ledger
materialization, and per-cell runtime services into the public node runtime
factory default.  Both keep every live seam injectable so their contracts stay
CPU-testable.

This keeps the new behavior removable with deletion and avoids making a
same-panel research estimator a stable training API.

**Alternative rejected:** extend the generic trainer with RL, compiler, and
projection modes.  It would enlarge compatibility surface before the mechanism
has any positive evidence.

### 2. Treat RP as part of the sampled policy

The batch-four sampler records the exact token history and chosen-token policy
log probability after repetition penalty and temperature.  Replay reconstructs
that same processed distribution from the frozen history on the HF fp32/SDPA
exact-history batch-one surface, which also owns the score-function gradient
forward.  A no-update qualification compares the native sampled evidence with
replay at every generated token and seals the numeric tolerance and processor
order, and parity always runs before expensive witness/dose work.

The exact surface is a live-falsified requirement, not a preference.  The v3
activation stopped at the sealed gate without magnitudes; the quantified v4
activation, run with the gate unchanged, measured the BF16/FA2 packed replay
against the fp32 sampler at `max=0.868143` nats, `mean=0.055759` nats over
`1573` tokens, `620` tokens over the `0.02` per-token gate, at `rp=1.0` with
zero updates.  That spread is intrinsic to the numeric-surface pairing and
rules out both tolerance widening and BF16/FA2 replay for an exact-on-policy
unit.  Moving replay and gradients to the exact surface is an
execution-surface correction that preserves the frozen scientific contrast;
the image-width scale claim remains deferred, and the accepted cost is
compute, not semantics.

The implementation will not attempt to algebraically “remove” RP after
sampling.  The two training policies are independent contracts, while every
proposal is behaviorally audited under both policies.

**Alternative rejected:** sample with RP 1.10 but optimize raw logits or RP 1.0
likelihoods.  That changes the policy whose score function is being estimated
and makes cross-RP conclusions uninterpretable.

**Alternative rejected:** align by sampling from a bf16 vLLM engine instead.
That redefines the sampled policy itself, de-aligns acquisition from every
sealed fp32 audit surface in the unit, and still compares two kernel-different
bf16 surfaces that have no reason to agree within the sealed gate.

**Alternative rejected:** widen the sealed tolerance to admit the measured
spread.  A gate that admits a 0.055-nat mean silently retires the
exactly-on-policy claim while keeping its name.

### 3. Build trajectory credit as a pure detached projection

Parsing, matching, first-hit identity, owner weights, burden precedence,
returns-to-go, and leave-one-out baselines are computed in a CPU-pure stage.
Its output is an immutable row/token advantage ledger; the model-facing loss
only gathers the sealed processed log probabilities and multiplies them by
those detached advantages.

The terminal row is one-sided: missing trusted mass can make STOP negative,
but no finite union licenses positive STOP imitation.  Equal returns naturally
produce zero advantage.

A legacy-M row has zero instantaneous reward and its own score-function tokens
are masked even if downstream trusted hits would give that row positive
return-to-go.  It remains in causal history for later scored actions.  This is
an intentional biased quarantine, not an unbiased-policy-gradient claim.

Loss reduction is over logical images and trajectories.  Every microstep
contributes an unnormalized numerator and the runtime applies the one sealed
`N*K` denominator exactly once, which keeps gradient accumulation an
execution detail rather than an objective change.  Score-function loss and
gradients run on the exact fp32/SDPA surface; no-padding packing may carry
only non-score-function plumbing, and only with proof of mathematical
identity to that surface.

**Alternative rejected:** give all K samples the union reward or use
within-group normalized cardinality.  The former ties all advantages and the
latter rewards individual path size without compiling complementary paths into
one greedy completion.

### 4. Keep the greedy compiler sparse and Source-bound

For each image and RP contract, the compiler uses one sealed Source boundary
and the frozen metric-valid alias children for uncovered trusted owners.  It
uses `RP_r(raw_logits)` without temperature division, computes the normalized
valid log-mean-exp, and compares it with the realized bad child.  One image-
mean compiler term is added to the trajectory term with the fixed coefficient
owned by the research unit.

The compiler never searches full suffixes and never treats tokens outside the
frozen alias bank as invalid.  Its complete detached selector evidence is
stored, while final unconstrained clean greedy owns the transfer result.

**Alternative rejected:** teacher-force every sampled or GT suffix.  That
reintroduces order-sensitive sequence imitation and obscures whether the
sampling gradient itself carries whole-picture information.

### 5. Reconstruct the actual AdamW proposal before preservation

The runtime begins a complete training-state transaction, performs the exact
configured fresh-AdamW step in a private proposal state, and measures the
resulting parameter delta and bias-corrected diagonal denominator.  Unprojected
arms audit that actual step.  The preservation arm restores Source, projects
the measured delta in the frozen AdamW metric, and applies only the projected
delta for evaluation.

The owner-wise witness set is frozen from Source before acquisition.  The
projection uses a bounded active-set solver: screen all witness directional
changes, materialize gradients only for active constraints, solve the small
dual system, and repeat until every declared constraint and trust-radius check
passes.  Predicted changes, finite-difference checks, realized Source-witness
changes, correction size, and the exact applied parameter hash are receipted.
Failure to certify the first-order projection, finite measurement, trust
radius, or exact applied delta stops the cell.  A finite realized witness
degradation is instead retained as a scientific result and proceeds to both
behavioral audits.

Because no proposal continues, projected AdamW moments are deliberately not
defined in this change.

The witness and dose-mechanics measurement semantics are frozen by the owning
research unit and implemented by one experiment-local owner: one constraint per
`(trusted owner, Source RP membership)`, the sealed parser row's half-open
`[token_start, token_end)` span as the eligible tokens, the sign-aware
RP-processed full-vocabulary margin without temperature, minimum-margin
selection with index-then-token-id tie breaks, a frozen `(y, v*)` pair whose
float64 Jacobian is taken over the frozen `ParameterLayout`, and an HF
fp32/SDPA batch-one surface as the only margin/Jacobian/probe surface.
Certification compares `J . Delta` for the actual applied projected delta with
the re-maximized finite difference at unit step against the sealed `1e-4`
tolerance, and the dose statistics use the deduplicated compiler/witness site
union plus a teacher-forced greedy-decision comparison over both RP surfaces.

**Alternative rejected:** project the raw gradient and then let AdamW transform
it.  AdamW's coordinate-wise transformation can invalidate the intended
parameter-space constraints.

**Alternative rejected:** certify the projection with an extra small-step or
random-direction finite difference.  It would add a second, unsealed numeric
policy without measuring the delta that is actually applied and audited.

### 6. Use independent proposals and shared evidence only where causal

One `(training RP, seed group)` acquisition artifact is shared by its three
nested arms.  Each arm otherwise assembles an independent Source model and
fresh optimizer.  The runtime applies exactly one proposal, writes a private
checkpoint only when required by the existing HF evaluator, runs the two
clean-greedy audits, and restores the full transaction.  Private proposal bytes
are never promoted and are removed or retained only as failure evidence under
the existing lifecycle contract.

Before the matrix, qualification may replace the default learning rate exactly
once through the predeclared mechanics-only dose rule in Decision 7.  After
that decision is sealed, the matrix is static.  No matrix result can change
learning rate, costs, seeds, candidates, or later cells.

**Alternative rejected:** run A, then continue B and C from its weights.  That
would confound nested objective additions with optimization history.

### 7. Make the vertical decision-bearing

The vertical is now gated by a parity-only qualification: the reserved v5
root runs one image (1584, the K16 group that quantified the v4 failure)
through K16 acquisition and exact-surface replay at `rp=1.0` and then
`rp=1.10`, each against the unchanged sealed gate, with no witness, dose,
update, or owner analysis.  If either contract fails, the exact-on-policy
route is retired on that recorded result rather than tuning tolerance.  If
both pass, a fresh full-panel successor root continues the vertical below
unchanged.

The real vertical includes one complete K16 batch-four acquisition per training
RP, sampler/replay parity, all three objective constructions, one exact AdamW
preservation proposal per training RP, both HF RP audits for each proposal, and
rollback reproduction.  It also records wall time, peak device memory, packed tokens, decode tokens,
forward/backward counts, and artifact sizes.  Synthetic tensors and CPU mocks
can test mathematics but cannot admit the eighteen-proposal screen.

Qualification uses a seed group disjoint from all three matrix groups.  It
first evaluates the fixed AdamW learning-rate ray
`{3e-7, 1e-6, 3e-6, 1e-5, 3e-5}`, with `3e-6` as the default, under both RP
contracts.  This is not an online controller: every attempted `(RP, dose)`
point runs one independent C proposal from Source and fresh optimizer state,
completes both RP audits, and rolls back.  Gradient norm, delta norm, predicted
KL, and owner outcomes are recorded only as covariates.

The default `3e-6` is retained when it satisfies both RP contracts.  A dose is
mechanically admissible only when all of the following hold on both contracts:

- relative to the sealed Source surface, at least one greedy token decision
  changes and the preservation solve exposes at least one active witness
  constraint;
- no new malformed, cap-terminated, or unparseable output appears;
- witness Jacobian-vector products agree with finite differences within the
  already declared tolerance; and
- the median absolute decision-margin displacement does not exceed the median
  absolute Source decision margin over the same sealed sites.

If the default is below the floor, select the smallest larger ray point that
passes all gates; if it exceeds the ceiling, select the largest smaller ray
point that passes.  If the failure direction is mixed, the evidence is
non-monotone, or no single point passes both RP contracts, stop for a new
research decision.  The resulting content-addressed qualification receipt
freezes one global learning rate for both RPs, all A/B/C arms, and every matrix
seed.  Qualification owner identities and gains/losses are quarantined from
selection and excluded from the matrix analyzer.  No other scientific setting
may change.

## Risks / Trade-offs

- **[Native sampler and replay processor semantics differ]** -> Stop at the
  token-level parity gate; do not reinterpret the estimator.  Realized in
  v3/v4 as a numeric-surface breach; resolved by the exact-surface
  correction, never by tolerance revision.
- **[Exact fp32/SDPA parity still fails cross-engine]** -> The v5 parity-only
  root becomes the decisive negative measurement and the exact-on-policy
  route is retired; the sealed tolerance is not tuned.
- **[RLOO remains high variance at K16]** -> Use fixed paired seed groups and
  report individual cells; do not claim population inference.
- **[The compiler overfits one Source boundary]** -> Keep it a nested arm and
  let dual-RP unconstrained greedy own the result.
- **[Owner-wise projection is expensive]** -> Screen constraints before
  materializing active gradients, measure the vertical, and prefer correctness
  over scaling this exploratory panel.
- **[The nominal `3e-6` dose is mechanically silent or destructive]** -> Use
  only the sealed qualification ray and mechanics-only floor/ceiling; never
  tune from owner outcomes or adapt inside the matrix.
- **[First-order witnesses retain logits but not owners]** -> Keep zero named
  baseline-owner loss as the decision-owning behavioral outcome.
- **[RP 1.10 Source exposes a legacy-M owner]** -> Protect it in behavioral
  audit, mark it undefendable by the frozen training contract, and do not leak
  it into positive credit.

## Migration Plan

1. Implement pure policy-transform, ledger, compiler, and proposal-projection
   helpers under focused CPU/tensor tests.
2. Add the experiment-local runtime, analyzer, six leaf configs, and dry-run
   matrix receipts without launching a model.
3. Move score-function replay and gradients to the exact HF fp32/SDPA
   surface under focused CPU tests on real frozen shapes, then run the
   reserved v5 one-image parity-only qualification on both RP contracts; on
   failure, retire the exact-on-policy route on that recorded result.
4. Only after both v5 contracts pass, on a fresh full-panel successor root:
   run the qualification dose ray, freeze one global learning rate, then
   complete the real vertical and publish its bounded infrastructure
   evidence.
5. Only if the vertical passes, run the fixed eighteen-proposal matrix and
   publish same-panel results.
6. Stop and close the unit.  Any multi-update continuation or wider-image study
   requires a new research decision.

Implementation rollback is deletion of the successor-only scripts, configs,
tests, and unexecuted change artifacts.  Runtime rollback is the existing full
training-state transaction plus private-proposal cleanup; Source and historical
artifacts remain immutable.
