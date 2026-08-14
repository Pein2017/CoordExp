---
title: Human-13 K-Trajectory Credit to Greedy Repetition-Penalty Crossover Screen
description: A one-update same-panel mechanism screen of trajectory credit, sparse greedy compilation, and Source-owner preservation under repetition penalties 1.0 and 1.10.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: in_progress
unit_id: 2026-08-14-human13-k-trajectory-rp-crossover-screen
topic: qwen3-vl-dense-enumeration
status: planned
evidence_status: none
updated: 2026-08-14
---

# Human-13 K-Trajectory Credit to Greedy Repetition-Penalty Crossover Screen

## Decision

Run a bounded Human-13 mechanism screen before increasing image width.  The
screen asks whether fresh stochastic trajectories can supply a useful
whole-picture owner-coverage update, whether one sparse greedy compiler moves
that update into clean-greedy behavior, and whether a constraint on the actual
optimizer proposal can prevent the protected-owner exchange observed in every
proposal of the predecessor.

The screen crosses two repetition-penalty (`RP`) policy contracts, `1.0` and
`1.10`.  Each contract owns fresh full-support sampling and matched processed-
policy log probabilities.  Every proposed update is evaluated under clean
greedy decoding at both RP values and then rolled back.  No proposal is
accepted or used to generate a later training state.

## Question and strongest alternative

**Question:** Does owner-balanced first-hit trajectory credit produce a useful
whole-picture one-update proposal relative to the sealed Source criteria, does
a sparse valid-versus-bad margin compile that stochastic improvement into
greedy coverage, and does proposal-level preservation remove Source-owner loss
without erasing the gain?

**Strongest alternative:** K trajectories expose separate stochastic modes
that are not jointly composable.  Their average policy gradient may increase
sampling diversity without improving greedy decoding, and any fixed-dose
update large enough to add K-hit owners may continue to displace existing
owners even after local witness preservation.

## Prior evidence and claim boundary

The exact panel contains `13` images and `392` trusted annotations.  Under the
historical `rp=1.0` Source surface, the sealed ledger has `G=173` greedy owners,
`H=73` K-hit/greedy-miss owners, and `M=146` K-miss owners.  The authoritative
manifest SHA-256 remains
`a8f88716c1227054ab29dc698f89462c9369c47c8d6415de3783c0937f60a6fb`.

The predecessor made sixteen fresh-Source one-update proposals.  Every
proposal gained at least one H owner and lost at least one Source owner; only
`9/16` selected owners compiled into clean greedy.  This closes same-dose
suffix search as the next discriminator and leaves distributed trajectory
credit, greedy transfer, dose, and preservation unresolved.  The predecessor
is historical context, not a matched local-imitation arm in the new matrix.
The owning result is [the 2026-08-13 bounded pilot](../2026-08-13-human13-on-policy-first-bottleneck-successor/results.md).

This unit can support only exact-panel, one-update mechanism evidence.  It
cannot establish multi-update stability, validation or population gain,
K-miss support expansion, architecture sufficiency, full-set mastery, or a
deployable checkpoint.  Three acquisition seeds are paired replications, not
population-level statistical power.

## Frozen owner semantics

The historical trusted optimization universe is frozen as

```text
U = legacy G union legacy H.
```

Legacy M remains absent from positive credit, owner weights, compiler valid
sets, preservation witnesses, checkpoint choice, and stop rules.  A fresh
trajectory that happens to recover M is reported as incidental only; it does
not change the partition within this unit.  A valid unmatched row is not
silently converted into M or a trusted positive.

Each greedy evaluation surface receives its own deterministic Source baseline:

```text
S_1.0  = Source owners under clean greedy rp=1.0
S_1.10 = Source owners under clean greedy rp=1.10.
```

Gain and loss at evaluation RP `e` are defined relative to `S_e`:

```text
trusted_gain_e = (proposal_owners_e intersect U) minus S_e
baseline_loss_e = S_e minus proposal_owners_e.
```

If `S_e` contains a legacy-M owner, that identity remains protected by the
deployment regression audit but receives no training or preservation signal.
It is reported as `protected_but_undefendable_M`.  This keeps M neutral while
making processor-specific baseline regression visible.

Contract-local success means positive trusted gain and zero baseline loss on
the matching RP surface.  RP-robust success means positive trusted gain and
zero baseline loss on both surfaces.  Historical G/H/M identities are reported
in parallel for continuity.

## Two optimization-policy contracts

For `r` in `{1.0, 1.10}`, define the sampled policy as

```text
pi(theta,T,r)(a | history)
  = softmax(RP_r(raw_logits(theta, history)) / T)[a].
```

Both contracts use:

- temperature `T=0.4`;
- `top_p=1.0` and no top-k truncation;
- `n=1`, `max_new_tokens=512`, no minimum-token floor, and zero frequency or
  presence penalty;
- Qwen `<|im_end|>` as the only generation stop, with `ignore_eos=false`, the
  terminal token retained in evidence, and `<|endoftext|>` not added as a stop;
- physical decode batches of four with sixteen trajectories per image;
- repetition penalty applied once per distinct token ID occurring in the full
  exact prompt plus generated history before the current action, followed by
  temperature, under the same processor order during sampling and replay; and
- complete chosen-token processed log probabilities and token histories.

The RP transform is part of the policy, not a sampling-only decoration.  The
sampler and replay path must pass a no-update token-level parity gate before
scientific execution.  The gate binds backend/model/processor identities,
prompt inclusion, processor order, sampled token, and processed log probability.
Parity requires exact history and chosen-token identity, finite likelihoods,
per-token absolute processed-log-probability difference no larger than `0.02`
nats, and acquisition-group mean absolute difference no larger than `0.002`
nats.  If the fast batch sampler and packed replay cannot meet this sealed
contract, the unit stops before model-quality execution; it does not call the
estimator exactly on-policy.  A 512-token cap stop is retained and reported as
harm rather than silently treated as natural termination.

Historical `rp=1.10`, `top_p=0.95` K16 trajectories may supply frozen aliases,
trusted support, and constant metadata.  They are never score-function data
for this unit.

## Trajectory credit

For image `i`, every trusted owner in `U_i` has fixed weight
`1 / |U_i|`.  Weights are frozen before fresh acquisition and do not depend on
the estimating K trajectories.  Matching, first-hit identity, advantages, and
all selectors are detached.

For trajectory `k`, let `c_ik` be normalized first-hit trusted-owner coverage.
The positive co-occurrence utility is fixed at

```text
u(c) = (exp(c) - 1) / (exp(1) - 1).
```

Each owner contributes only at its first matched row.  Chronological
class-agnostic prediction-to-prediction `IoU>0.95` duplicates receive no owner
credit.  The signed row ledger is fixed before acquisition:

- a trusted first hit receives the exact increment
  `u(c_after)-u(c_before)`;
- a duplicate, invalid row, trusted-owner repeat, or non-M unmatched row costs
  `1/|U_i|` once, using that precedence order rather than stacking labels;
- each malformed row-equivalent span costs `1/|U_i|`;
- a legacy-M match is neutral rather than a false positive; and
- terminal STOP costs the remaining trusted mass `1-c` when `c<1`, and is zero
  otherwise.

No positive exhaustiveness reward is assigned to STOP.  A cap termination adds
the same remaining-mass cost after the last parsed row, so earlier actions see
it in return-to-go, but it has no synthetic terminal token or direct STOP term.

Row returns-to-go telescope from those first-hit credits and costs.  For a
given image and row index, the baseline is the mean return-to-go of the other
fifteen trajectories; positions after a trajectory terminates contribute zero.
All tokens in one parsed row share its detached row advantage.  The terminal
STOP token is never positively imitated: its advantage is clamped to at most
zero, while a negative premature-STOP advantage remains attached to the STOP
log probability.  Therefore STOP can receive direct downward pressure without
finite U being treated as proof of exhaustiveness.

For `N=13` images and fixed `K=16`, define

```text
L_trajectory = -(1/NK) * sum_i sum_k sum_{t in scored non-M tokens}
                 A_i,k,row(t) * log pi_i,k,t.
```

The denominator is global and is applied once after summing logical examples;
arbitrary no-padding pack boundaries or gradient-accumulation splits MUST
produce the same loss and gradient.  It is a predeclared row-level score-
function surrogate.  The unit does not overclaim exact unbiasedness for
detached matching, row grouping, STOP clamping, or any sampler/replay drift
admitted by the parity contract.

Legacy-M row tokens are masked from this score-function sum even when their
zero instantaneous reward would otherwise inherit a positive downstream
return.  They remain in the observed causal history for later scored rows, so
the surrogate can learn trusted actions after that history without directly
crediting the M-producing action.  This deliberate quarantine is another
reason the objective is not claimed as an exact unbiased policy gradient.

## Sparse greedy compiler

For each RP contract and image, use the sealed Source clean-greedy pre-STOP
boundary when trusted owners remain.  The valid set contains next tokens from
the frozen `309` complete metric-valid aliases for uncovered trusted owners;
fresh sampled rows do not enlarge it.  Aliases are normalized within owner and
owners are normalized within the valid set.

At greedy-policy logits `z = RP_r(raw_logits)` with no temperature division,
the valid score is the normalized log-mean-exp

```text
M(V) = kappa * log(sum_v alpha_v * exp(z_v / kappa)),
sum_v alpha_v = 1,
```

with fixed `kappa=1`.  Because `M(V) <= max_v z_v`, crossing the realized
STOP/bad-child margin implies that at least one trusted valid token beats that
competitor.  The compiler uses margin `1e-4`, is absent when no premature STOP
boundary exists, and never claims that aliases observed under different
histories are recursively composable.  Final clean greedy, not this
teacher-forced boundary, owns the transfer claim.

The compiler loss is
`L_compiler=(1/N) sum_i relu(1e-4 + z_bad_i - M(V_i))`, with an absent site
contributing zero and the global image denominator applied once across packs.
Arm B uses `L_trajectory + L_compiler` with coefficient `1.0`; no observed
proposal norm is used to retune that coefficient.

## Proposal-level preservation

All active cells start from the exact Source model and a fresh AdamW optimizer
with betas `(0.9, 0.999)`, epsilon `1e-8`, and zero weight decay.  The default
learning rate is `3e-6`.  Before matrix materialization, the disjoint
qualification group may replace it exactly once using the fixed dose ray

```text
3e-7, 1e-6, 3e-6, 1e-5, 3e-5.
```

This is predeclared dose calibration, not online adaptation.  Every attempted
`(RP, dose)` point runs one independent C proposal from Source and fresh
optimizer state, completes both RP audits, and rolls back; no proposal receives
more than one update.  Gradient norm, realized delta norm, predicted
Kullback-Leibler divergence, and owner outcomes are covariates; none controls
learning rate or rescales an arm after its proposal is observed.

The default `3e-6` is selected if it passes on both RP contracts.  A candidate
passes the mechanical floor and ceiling only if, on both contracts:

1. at least one greedy token decision changes relative to the matching sealed
   Source surface and at least one preservation witness constraint is active;
2. no new malformed, cap-terminated, or unparseable output appears;
3. witness Jacobian-vector products agree with finite differences within the
   already sealed tolerance; and
4. median absolute decision-margin displacement is no larger than the median
   absolute Source decision margin over the same sealed compiler/witness sites.

These four mechanics are measured exactly.  For the actual applied projected
delta `Delta` on the same private proposal checkpoint, `predicted = J . Delta`
and `fd = m_tilde(theta + Delta) - m_tilde(theta)` at unit step `alpha = 1`,
where `m_tilde` re-maximizes the competitor over the full vocabulary at the
frozen site; `jvp_fd_max_abs_error` is `max |predicted - fd|` over every
constraint witness and `jvp_fd_tolerance` is exactly the sealed
`1e-4` first-order tolerance.  Missing, non-finite, or surface-mismatched
measurement fails closed; an error above tolerance is a ceiling failure and is
never rewritten into a pass.  No optional small-step or random-direction finite
difference and no new loss term may be introduced.  Decision-margin dose sites
are the deduplicated union of compiler sites and trusted constraint-witness
sites across both surfaces, excluding legacy-M, with each site scored on its
own RP surface, and the median of an even-cardinality sample is the mean of its
two middle values.  `greedy_decision_change_count` is teacher-forced across
every sealed Source decode token under both RP surfaces, comparing the exact
processed argmax (ties broken by the smallest token id) with the sealed chosen
token; free-running dual audits own the nonnegative malformed, cap-terminated,
and unparseable output deltas.  Owner outcomes remain sealed and unread for
learning-rate selection.

If `3e-6` fails only the floor, select the smallest larger ray point passing
all gates.  If it fails the ceiling, select the largest smaller ray point
passing all gates.  Mixed/non-monotone evidence or absence of one common point
for both RPs stops the unit for a new decision.  The selected learning rate and
qualification receipt are frozen before matrix execution and are byte-
identical across both RPs, all A/B/C arms, and all matrix seeds.

The preservation arm first reconstructs the exact AdamW parameter delta for
the trajectory-plus-compiler gradient.  For every Source-emitted owner in
`U intersect S_1.0` or `U intersect S_1.10`, retain its weakest processed-logit
token margin as an owner-wise witness.  The witness set and logits are detached
before acquisition outcomes.

The witness surface is exact.  One constraint exists per
`(trusted owner, Source RP membership)` pair, so an owner emitted on both
Source surfaces yields two witnesses; legacy-M owners stay audit-only and carry
no Jacobian.  The owner row is the sealed Source clean-greedy parser/matcher
row and its eligible tokens are exactly the half-open parser span
`[token_start, token_end)`; terminal and inter-row tokens are excluded and no
hand exclusion is permitted.  At each eligible index `t` the row score is
`z = P_r(raw_logits)` under the exact sign-aware repetition-penalty transform,
without temperature division, over the full vocabulary, and
`m_t = z[y_t] - max_{v != y_t} z[v]`.  The witness is the minimum `m_t`, with
ties broken by the smallest generated token index and competitor ties by the
smallest token id; the margin must be finite and Source-greedy
(`m >= 0` within numeric tolerance).  The chosen token `y` and competitor `v*`
are then frozen and the Jacobian is `d(z[y] - z[v*])/d theta` over the frozen
`ParameterLayout` DoRA trainables, flattened to float64.  Margins, Jacobians,
and the realized probe are computed only on the HF fp32/SDPA batch-one surface
with autograd enabled for the trainable DoRA parameters, and the bank is frozen
before acquisition.

Preservation solves the minimum-change projection of the exact AdamW delta in
its fixed diagonal AdamW coordinate metric:

```text
min_delta  (delta-delta_0)^T D (delta-delta_0)
subject to J_w delta >= -1e-4                  for every witness w,
           delta^T D delta <= delta_0^T D delta_0,
where      D_j = sqrt(v_hat_j) + epsilon.
```

`D` is frozen from the bias-corrected denominator of the unprojected proposal.
The applied and audited parameter delta is the projected delta.
Witness Jacobian-vector products, finite-difference checks, feasibility,
predicted changes, realized changes, and correction norm are all receipted.
Legacy-M baseline owners remain audit-only and do not enter these constraints.
Finite realized witness degradation is a scientific outcome, not a reason to
drop the C cell: it remains eligible for both clean-greedy audits and is
reported as `realized_witness_violation`.  Only missing/non-finite measurement,
an uncertified first-order projection, or a wrongly applied delta fails closed.

This one-update screen never carries projected optimizer moments forward.  If
the preservation arm succeeds, moment-consistent continuation is a separate
scale-time infrastructure and research decision.

## Matrix

The three descriptive arms are:

- **Trajectory Credit (`A`)**: signed row-level trajectory credit only;
- **Trajectory Credit plus Greedy Compiler (`B`)**: A plus the sparse compiler;
- **Trajectory Credit plus Greedy Compiler plus Preservation (`C`)**: B with
  the actual AdamW proposal projected through owner-wise preservation.

For each RP contract, use three seed groups:

```text
31001..31016
32001..32016
33001..33016.
```

The production-shaped vertical uses the disjoint qualification group
`30001..30016` for each RP.  Its owner identities, gains, and losses are sealed
but unavailable to dose selection and excluded from the eighteen-cell
analyzer.  Qualification may change only the single global learning rate via
the rule above; it cannot change loss values, tolerances, seeds, arms, or
continuation.  Only its predeclared mechanics/semantic gates can admit matrix
execution; its owner gains or losses are not pilot data.

A, B, and C within one `(RP, seed-group)` share byte-identical trajectories,
matching, fixed weights, credits, costs, and advantages.  They differ only by
the named compiler and preservation additions.  Every cell starts from Source
and a fresh optimizer, applies one proposal, runs clean greedy under both RP
surfaces, records the results, and restores the complete transaction.  There
are exactly `2 x 3 x 3 = 18` proposals and no adaptive continuation, accepted
checkpoint, K refresh, or post-qualification dose choice.

The primary evidence is the paired A-to-B-to-C change within each RP and seed,
the named baseline-owner losses, trusted gains, and output burdens on both
surfaces.  Union@K alone is support evidence, not success.

## Infrastructure alignment

Implementation behavior and execution discipline are linked from:

- [OpenSpec change](../../../../../openspec/changes/add-human13-k-trajectory-rp-crossover-screen/);
- [implementation design](../../../../../docs/superpowers/specs/2026-08-14-human13-k-trajectory-rp-crossover-screen-design.md); and
- [Superpowers implementation plan](../../../../../docs/superpowers/plans/2026-08-14-human13-k-trajectory-rp-crossover-screen.md).

Reuse the existing:

- Human-13 manifest, parser, one-to-one owner matcher, duplicate projection,
  language-only DoRA model assembly, and frozen vision/aligner/embedding surface;
- no-padding multimodal packing and compact-logit forward path;
- HF fp32/SDPA batch-one clean-greedy evaluator at both RP values;
- full model/AdamW/scheduler/counter/CPU-and-CUDA-RNG transaction and rollback;
- private proposal lifecycle and immutable analyzer receipts.

The linked OpenSpec owns only the experiment-local RP sampler/replay contract,
trajectory-credit loss, compiler materialization, exact-proposal preservation,
dual-surface analyzer, and bounded matrix lifecycle.  It does not own this
unit's cohort, utility meaning, success rule, or scientific claim.

## Vertical gates

Before the matrix:

1. seal two deterministic Source decodes per RP and their surface-specific
   owner/burden baselines;
2. run an `eta=0` proposal through the complete transaction and show no change
   on either RP surface;
3. on at least one image per RP, prove token-by-token sampler/replay processed-
   log-probability parity under the sealed numeric tolerance and exact history;
4. prove fixed owner-weight detachment, return-to-go telescoping, leave-one-out
   constant-shift invariance, arbitrary pack/gradient-accumulation invariance,
   and one real negative STOP advantage;
5. prove the normalized compiler bound and explicit absent-boundary receipt on
   real logits;
6. reconstruct one exact fresh-AdamW delta per RP, match witness Jacobian-vector
   products to finite differences, solve a feasible preservation projection,
   and report predicted versus realized witness changes;
7. evaluate the predeclared learning-rate ray on disjoint qualification seeds,
   use only the mechanical floor/ceiling above to freeze one global dose, and
   quarantine all qualification owner outcomes from selection; and
8. admit the selected ray point for each RP only after its acquisition, packed
   backward, private parameter delta, dual clean-greedy audit, and byte-
   identical rollback are complete, while excluding every qualification owner
   outcome from matrix disposition.

These are mechanics and estimator-semantic gates, not model-quality evidence.
Stop before the matrix on wrong surface, non-finite state, sampler/replay parity
breach, Source baseline nondeterminism, projection infeasibility, or rollback
non-reproduction.  A repair requires a fresh immutable run identity and cannot
change the frozen scientific contrast.

## Decision and stop rules

The matrix ends after exactly eighteen proposals regardless of outcome.

- A-to-B isolates whether the sparse compiler moves trajectory improvement
  toward greedy behavior.
- B-to-C isolates whether actual-proposal preservation reduces named baseline
  loss at the same qualification-selected global learning rate.
- For a fixed `(arm C, training RP)` pair, contract-local success requires at
  least two of its same three seed cells to achieve positive trusted gain and
  zero baseline loss on the matching evaluation RP.
- For that same `(arm C, training RP)` pair, RP-robust success requires at least
  two same seed cells each to achieve positive trusted gain and zero baseline
  loss simultaneously on both evaluation RPs.  Passing seeds cannot be mixed
  across surfaces.
- If both training-RP pairs meet the RP-robust rule, report that stronger fact
  as bi-policy replication; it is not required for contract-local success.

Contract-local success opens only a user decision about a wider image screen;
it does not automatically launch one.  If all eighteen proposals lose baseline
owners on their matching surfaces, this globally fixed-dose one-update family is held on
this panel.  If A, B, and C are indistinguishable, the decomposition is
uninformative at the qualification-selected learning rate.  Duplicate, malformed, row, token, cap,
invalid, unmatched, incidental-M, delta-norm, predicted-KL, wall-time, and peak-
memory channels remain visible and cannot be replaced by pooled net recall.

## Artifact handle and non-goals

Planned immutable run roots live under:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-08-14-human13-k-trajectory-rp-crossover-screen/<run-id>/
```

The authorized production-shaped qualification root is frozen as

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
  2026-08-14-human13-k-trajectory-rp-crossover-screen/
  vertical-dose-qualification-v1/
```

It was confirmed absent before activation.  Any repair after partial activation
requires a new append-only run identity; this path is never overwritten.

Implementation and bounded model/GPU execution were explicitly authorized by
the user; this unit still grants no authority beyond its named tasks and roots.
No K-miss supervision, full-sequence CE control, DPO, GFlowNet, bridge,
architecture change, validation run, checkpoint promotion, or production
default belongs to the unit.

Independent Fable-5 reviews returned `PROCEED` and later recommended the fixed
qualification-only dose ray now incorporated: fixed owner weights, explicit
sampler/replay parity, one-sided STOP credit, surface-specific baseline sealing,
and one globally frozen learning rate selected without owner outcomes.  The
reviews are advisory; this unit owns the resulting protocol.

A subsequent GPT-5.6-sol-max peer audit and independent coherence re-review
returned `PASS` with no remaining P0/P1.  These verdicts are also advisory and
grant neither implementation nor execution authority.
