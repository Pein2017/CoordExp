---
title: Object-Specific Geometry Transport, Decision Phase, and Cross-Row Influence Horizon
description: Staged four-case causal study testing late donor-state transport, gating physical-owner interpretation, and separating persistent cross-row entity selectivity from local traversal inertia.
type: investigation
role: research-unit
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-17-object-specific-geometry-transport-and-cross-row-influence-horizon
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
conclusion_status: stopped_after_wave_one_common_support_failure
updated: 2026-07-17
---

# Object-Specific Geometry Transport, Decision Phase, and Cross-Row Influence Horizon

## Question

At one exact autoregressive state, does a donor-induced geometry state survive
two increasingly difficult boundaries, and when can that state be identified
as a selected physical object rather than route or box grammar:

1. later coordinate tokens within the same detection row; and
2. row closure plus one, two, or three subsequent detection rows?

The falsifiable form is:

> After earlier coordinate tokens and intermediate rows are made identical,
> does changing a trusted donor or committed owner still change later geometry
> or later unique-entity scores, and do owner-admission and alias controls rule
> out route, grammar, exact repetition, and traversal explanations?

A supported same-coordinate donor contrast first establishes late donor-state
transport. Only a contrast that also passes the owner-admissible anchor gate
supports an object-specific carrier. A negative result under valid common
support raises autoregressive box grammar, geometry-sorted traversal, or local
prefix transition as sufficient explanations for that bounded state. Neither
outcome selects a final architecture.

## Why This Is the Next Discriminator

The completed coordinate-release unit established three facts at bounded
states:

- a dense-chair `x1` intervention moves released `x2` toward the cued endpoint
  on `63/64` paired suffixes;
- a visible fork remains predominantly part-sized after its reportable whole-
  object `x1,y1` path is supplied; and
- changing the current chair geometry changes the next generated category.

Here `x1` is the left horizontal boundary, `y1` is the top vertical boundary,
`x2` is the right horizontal boundary, and `y2` is the bottom vertical
boundary.

These results reject literal coordinate independence and prove executable
cross-row prefix sensitivity. They do not distinguish:

```text
physical-object state
  versus
autoregressive width, validity, and box-shape grammar
```

or:

```text
unique-object commit and covered-set update
  versus
local successor transition and geometry-sorted traversal inertia
```

The earlier manual review also showed that category evidence can remain
correct while different box boundaries refer to parts, neighboring entities,
or multiple rows of chairs. A row therefore cannot be treated as a physical-
object commit until its owner is frozen independently of Common Objects in
Context (`COCO`) annotation completeness and loose box overlap.

## Terminology

- **Qwen3 Vision-Language (`Qwen3-VL`)**: the pretrained multimodal model
  family under investigation.
- **Common Objects in Context 80-category ontology (`COCO-80`)**: the closed
  set of reportable categories used in this investigation. Visible objects
  outside this ontology are treated as background for the task, not as proof
  that the pixels contain no entity.
- **Exact fixed-prefix case**: one frozen image, prompt, adapter, tokenizer,
  complete preceding token prefix, current row boundary, position identifiers,
  and Key-Value cache state.
- **Physical-object owner**: one visually distinguishable entity in the image,
  identified by a frozen `COCO` annotation identifier or a human identifier of
  the form `human:<image-id>:<four-digit-local-entity-index>`.
- **Unique physical-entity ledger**: the case-local table that maps accepted
  annotations, previously unlabeled objects, row variants, and model outputs
  to physical-object owners. It is intentionally local and need not claim
  exhaustive scene coverage.
- **Owner-equivalent row aliases**: two byte-distinct legal rows with the same
  description and token count whose boxes both have at least `0.90`
  Intersection over Union with one frozen primary owner box and are not closer
  to any competing ledger owner. One alias is committed and the other is
  scored, then their roles are exchanged.
- **Object-specific residual transport**: a physical-object-dependent hidden
  state continues to affect later boundaries after the intervening coordinate
  tokens have been forced to be identical.
- **Autoregressive box-grammar transport**: later boundaries depend on earlier
  coordinate tokens through width, ordering, box validity, or learned shape
  continuation without requiring a persistent physical-object owner.
- **Decision phase**: one of the row-entry or first-description decision,
  pre-`x1` location decision, or late `x2,y2` extent decision.
- **Left-top-right-bottom coordinate order (`xyxy`)**: a box encoded as left
  horizontal, top vertical, right horizontal, and bottom vertical boundaries.
- **Clamped-intermediate causal effect**: the effect of changing row `t` while
  forcing rows between row `t` and the measured future row to the same fixed
  legal token sequences in both conditions.
- **Free-trajectory total effect**: the full downstream difference when the
  model generates every row after the intervention without intermediate
  clamping.
- **Traversal inertia**: a prefix effect that changes the next canonical rank,
  list phase, or geometry-sorted successor without demonstrating suppression
  of one committed physical entity.
- **Terminal no-more-objects decision (`STOP`)**: the generated decision that
  ends enumeration. This unit does not assume that it verifies visual
  completeness.
- **Display-only interpolated crop**: an enlarged crop produced only for human
  inspection. Lanczos interpolation may make existing pixels easier to see,
  but it creates no new visual evidence and never changes model input.

## Authorization Boundary

The user authorized implementation and graphics-processing-unit execution of
this staged research unit through the active long-running goal. Authorization
is limited to the smallest probes needed to reach the sequential admission
gates. It does not authorize model training, a new loss, a final forward pass,
a persistent ledger, an object slot, a detector, or an OpenSpec change.

The user has approved the scientific design choices below:

1. the clamped-intermediate causal effect is primary and free rollout is a
   secondary total-effect sidecar;
2. the waves use sequential admission gates rather than one large crossed
   factorial;
3. any coverage claim requires a unique physical-entity ledger; and
4. the first panel contains four deeply reviewed cases and expands to eight
   only if the initial mechanism signatures conflict.

## Causal Decomposition

Let `P` denote the exact prefix immediately before row `t`. Let
`R_A_commit` and `R_B_commit` denote complete legal treatment rows referring
to physical entities `A` and `B`. For each owner, freeze a byte-distinct but
owner-equivalent probe row, `R_A_probe` and `R_B_probe`. Let `C_1` and `C_2`
denote common legal intermediate rows that are identical in both intervention
branches.

For physical entity `e`, define `S(e | context)` as the length-normalized
teacher-forced log probability of its frozen probe row under the declared
context. Every `log` is the natural logarithm. The branch contexts are:

```text
Context_A(d)
  = P, do(R_A_commit), do(intermediate rows=C_1...C_(d-1))

Context_B(d)
  = P, do(R_B_commit), do(intermediate rows=C_1...C_(d-1))
```

The per-entity controlled effect remains a diagnostic:

```text
ClampedEntityEffect(d, e)
  = S(e | Context_A(d)) - S(e | Context_B(d))
```

The primary coverage-selectivity estimand is the symmetric treatment-by-owner
interaction:

```text
SymmetricEntitySelectivity(d)
  = 0.5 * [S(A | Context_B(d)) - S(A | Context_A(d))]
  + 0.5 * [S(B | Context_A(d)) - S(B | Context_B(d))]

OwnerSuppression_A(d)
  = S(A | Context_B(d)) - S(A | Context_A(d))

OwnerSuppression_B(d)
  = S(B | Context_A(d)) - S(B | Context_B(d))

ThirdOwnerDrift(d, D)
  = abs[S(D | Context_A(d)) - S(D | Context_B(d))]
```

where distance `d=1` measures row `t+1`, distance `d=2` measures row `t+2`,
and distance `d=3` measures row `t+3`.

`SymmetricEntitySelectivity` is a summary, not a pass gate: a positive average
can hide one wrong-signed owner component. Let `epsilon_noop` be the maximum
absolute length-normalized score drift across exact no-operation replay and
self-state replacement controls. Freeze:

```text
delta_entity = max(0.05 natural-log units per token, 5 * epsilon_noop)
delta_third  = max(0.05 natural-log units per token, 5 * epsilon_noop)
```

A coverage-selective result requires both `OwnerSuppression_A` and
`OwnerSuppression_B` to exceed `delta_entity` separately in both reciprocal
commit/probe alias directions. It also requires `ThirdOwnerDrift` to be no
larger than `delta_third` and no larger than half the smaller owner-
suppression component. Exact byte-repeat suppression remains a separate
control. These gates distinguish reciprocal owner selectivity from a positive
average caused by only one owner, generic category inhibition, continuation
drift, or arbitrary mass movement.

This is an intervention-defined controlled effect. It is not called a natural
direct effect because the common intermediate rows may not be the rows each
branch would naturally generate.

The secondary free-trajectory total effect records:

- the physical entity generated at each of the first three free rows;
- cumulative unique-entity recovery through each distance;
- duplicate, unsupported, invalid, and terminal outcomes; and
- the distance at which the two trajectories first reconverge by physical
  entity and by the newly generated suffix-row sequence. The complete prefixes
  cannot become byte-identical because row `t` remains different by design.

The row-local estimand fixes one coordinate history `c=(x1,y1)` in both donor
conditions. For paired physical-object donor states `h_A` and `h_B`, define:

```text
LateCoordinateScore_A(h, c)
  = log p(x2_A | h, c)
  + log p(y2_A | h, c, x2_A)

LateCoordinateScore_B(h, c)
  = log p(x2_B | h, c)
  + log p(y2_B | h, c, x2_B)

LateGeometryMargin(h, c)
  = LateCoordinateScore_A(h, c) - LateCoordinateScore_B(h, c)

DonorStateLateCoordinateTransport(c)
  = LateGeometryMargin(h_A, c) - LateGeometryMargin(h_B, c)
```

For the phase matrix, freeze one exact candidate token segment
`Y_(e,r)` for owner `e` and readout phase `r` before execution. Define:

```text
PhaseSegmentScore(e, r | intervention q)
  = sum over k in Y_(e,r) of
    log p(Y_(e,r,k) | intervention q, frozen readout history,
          Y_(e,r,<k))

PhaseOwnerMargin(r | intervention q)
  = PhaseSegmentScore(A, r | intervention q)
  - PhaseSegmentScore(B, r | intervention q)
```

Description, `x1,y1` location, and `x2,y2` extent segments and their frozen
readout histories are identical across intervention boundaries. Greedy owner
attribution remains a separate readout.

Here `(x2_A,y2_A)` and `(x2_B,y2_B)` are the one frozen primary coherent
late-coordinate reference pair for each owner. Alternative accepted extents
are reported only as secondary sensitivity readouts and are not selected after
viewing the intervention result. Greedy coordinate release and physical-owner
attribution are separate readouts; neither is folded into the score above.

This estimand shows only whether a donor-induced state remains causally
readable after byte-identical early coordinate tokens. Define:

```text
EarlyCoordinateScore(h, c)
  = log p(x1_c | h) + log p(y1_c | h, x1_c)
```

A common history is branch-supported only when its score under each donor is
within `log(10)` of that donor's own frozen native `x1,y1` history. This fixes
the minimum sequence-probability ratio at one tenth before late-coordinate
outcomes are opened.

Let `epsilon_late_noop` be the maximum absolute selected-coordinate log-
probability drift between unrestricted replay and exact self-state replacement
over every admitted row. Freeze the deterministic late-transport effect floor
before execution as:

```text
delta_late = max(0.05 natural-log units, 5 * epsilon_late_noop)
```

The block-`23` donor contrast supports the predicted direction only when
`DonorStateLateCoordinateTransport(c) > delta_late` for every admitted common
history. A reversed history is a direction failure, a weak or mixed panel is
reported as inconclusive, and any block-`13` absolute contrast above
`delta_late` vetoes layer-specific support. Absolute magnitude without the
predeclared donor direction is not support.

The estimand is promoted to a physical-owner estimand only when a frozen
`OwnerAdmissible(c)` gate passes:

1. `c` meets the predeclared branch-wise support threshold under both donors;
2. combining `c` with each late-coordinate reference yields a valid `xyxy`
   box; and
3. the two assembled boxes have distinct, unambiguous owners in the frozen
   physical-entity ledger under the predeclared attribution rule.

For this bounded screen, a box has a unique ledger owner only when its best
Intersection over Union is at least `0.30` and exceeds the second-best owner by
at least `0.15`. Owner admission is evaluated separately for each common
history and is conjunctive with branch support. The image-`7818` smoke records
this gate only as a diagnostic and is hard-prohibited from producing a
physical-owner claim, even if a future ledger revision would change the
diagnostic attribution.

The existing image-`7818` pair does not pass this owner gate: one crossed box
is invalid and the other is owner-ambiguous. It remains a valid donor-state
survival assay, but cannot by itself support a physical-object-state claim.

## Competing Hypotheses

### Donor state survives coordinate history; physical-owner meaning is gated

**Motivation.** A decoder block-`23` donor state on image `7818` can switch the
unrestricted pre-`x1` geometry owner, while a block-`13` control cannot. The
existing result is concentrated at `x1`; it does not show that owner
information survives later coordinate tokens.

**Expected mechanism.** A donor-induced geometry or route state remains
causally readable after the same description and coordinate tokens have been
consumed. A stronger physical-owner interpretation requires an
owner-admissible common anchor rather than only donor labels.

**Predictions.**

- target and paired block-`23` donor states produce different `x2,y2`
  coordinate margins under at least one branch-supported common `x1,y1`
  history;
- the sign follows donor direction under both supported common coordinate
  histories;
- self-state replacement and the trusted block-`13` control do not reproduce
  the owner switch; and
- only an owner-admissible pair may additionally show that the late effect is
  owner-specific rather than route-specific or syntactic closure.

**Verification experiment.** Reuse the eligible image-`7818` target and paired
`wine glass` donor states only for donor-state late-coordinate transport.
Under each donor state, force each branch-supported common `x1,y1` history,
then score and release `x2,y2`. Separately pre-screen the four candidate cases
for an owner-admissible common anchor. If none exists, close the owner-specific
subquestion as unresolved rather than reusing image `7818` as owner evidence.

**Falsification experiment.** If the donor-state contrast disappears after
every branch-supported common `x1,y1` path while ordinary coordinate-only
forcing still moves `x2`, late donor-state transport is not supported at this
state. A failed owner-admission gate does not falsify physical-object state; it
means this pair cannot identify it.

**Confounds.** Donor-state capture may carry hard-routing artifacts, geometry
or traversal state rather than owner identity, and a donor may be incompatible
with the forced coordinate history. A single layer or direction cannot
establish a universal object file. Branch-wise support, valid crossed boxes,
unique owner attribution, both coordinate histories when admissible, and the
trusted negative layer are therefore required.

### Earlier coordinates drive later boundaries through box grammar

**Motivation.** The dense-chair `x1` intervention also changes the legally and
statistically plausible range of `x2`. Strong `x1`-to-`x2` dependence is
therefore expected even without object identity.

**Expected mechanism.** The model learns a conditional distribution over
valid widths, aspect ratios, shape priors, and geometry-sorted continuations.

**Predictions.**

- later horizontal boundaries move systematically under real-edge and
  synthetic non-boundary `x1` cues;
- movement is explained substantially by the numeric cue and valid-width
  constraints, even when no frozen annotated chair left boundary begins at
  the supplied coordinate;
- the same-coordinate donor contrast is weak or absent; and
- phrase identity may remain fixed while geometry changes.

**Verification experiment.** On the frozen dense-chair state, compare the two
real chair edges with the predeclared inter-edge and far dense-field
non-boundary horizontal coordinates. Keep the remaining prefix and top
boundary fixed, and record the complete `x2` coordinate distribution rather
than only the chosen bin.

**Falsification experiment.** A block-`23` donor that changes uniquely
attributed late owners through an owner-admissible common anchor cannot be
explained by coordinate grammar alone. A donor effect that fails that gate
remains compatible with geometry-route state.

**Confounds.** Background coordinates can create implausible boxes, and a
linear coordinate response is not by itself evidence of a learned width rule.
Synthetic non-boundary conditions are grammar-positive controls, not owner or
background evidence.

#### Frozen image-`19432` non-boundary grammar fixture

Fixture review before opening any new outcome logits found that the dense
chair field does not contain a defensible empty-background horizontal cue at
the fixed top boundary. The originally proposed `object-free` label is
therefore retired for this case rather than being assigned to a visually
occupied location. The executable discriminator compares real chair left
boundaries with synthetic coordinates at which no frozen annotated chair left
boundary begins. These synthetic arms test whether translation-grammar-
compatible behavior extends beyond the two frozen annotated left edges; they
do not independently establish boundary independence and are not background,
owner, or object-absence evidence.

The canonical model input is the `1152`-by-`864` image with Secure Hash
Algorithm 256-bit digest
`815ff42f8543e6da5fb5d92372ce0b001f5f14b973f9fddeb6a956319f64277d`.
The exact pre-`x1` prompt digest is
`250f52fc3dbc6e405a7d3bc1f90bb2a6c391be6812409383fec30aa0e1d0f253`.
All arms force the source-native top boundary `y1=123` and differ only in
`x1`:

| Arm | `x1` coordinate bin | Operational meaning |
|---|---:|---|
| `real_target_left_edge` | `537` | Frozen target chair `coco-ann:378536` left boundary. |
| `synthetic_inter_edge_nonboundary` | `608` | Pre-score visual-review choice between the two real left-edge cues; it lies in the dense chair field and is not an object-free claim. |
| `real_adjacent_left_edge` | `643` | Frozen adjacent chair `coco-ann:387701` left boundary. |
| `synthetic_right_dense_field_nonboundary` | `700` | Pre-score dense-field control at least `43` coordinate bins from every frozen annotated chair left boundary, with `299` bins of uncensored right-boundary support; it may lie inside another chair extent and is not an object-free claim. |

Let

```text
HistoryScore(c)
  = log p(x1=c | exact pre-x1 prefix)
  + log p(y1=123 | exact pre-x1 prefix, x1=c)
```

using full-vocabulary probabilities. Freeze a natural-log support tolerance of
`log(10)`. Let the larger of the two real-edge history scores be the reference.
Every real or synthetic arm is support-admitted only when its history score is
no more than `log(10)` below that reference. An unsupported synthetic cue is
recorded as a positivity failure and cannot be interpreted as a null grammar
effect.

For each admitted arm, store the complete 1,000-bin `x2` coordinate-logit
vector in 32-bit floating point. Let `q_c(k)` be its coordinate-normalized
probability and freeze:

```text
ValidRightMass(c) = sum over k > c of q_c(k)
ExpectedRightValid(c)
  = sum over k > c of k * q_c(k) / ValidRightMass(c)
ExpectedWidth(c)  = sum over k > c of (k-c) * q_c(k) / ValidRightMass(c)
```

An arm is valid-width admitted only when `ValidRightMass(c) >= 0.80`. Compare
both the absolute `x2` distributions and a zero-padded translated-width
probability mass function:

```text
WidthProbability(c, w)
  = q_c(c+w) / ValidRightMass(c) for 1 <= w <= 999-c
  = 0 otherwise, on the common width support w=1...999.
```

A bounded simple translation-grammar-compatible signature requires all four
arms to pass support and valid-width admission, the real adjacent-minus-target
`ExpectedRightValid` difference to be at least `20` bins, Kendall rank
correlation between `x1` and `ExpectedRightValid` to be at least `2/3`, the
fitted `ExpectedRightValid`-on-`x1` slope to lie in `[0.5, 1.5]`, the range of
`ExpectedWidth` to be at most `50` bins, and the median pairwise Jensen-Shannon
divergence between translated width distributions to be smaller than that
between absolute `x2` distributions. Invalid right-boundary mass remains a
separate reported measurement and never enters the conditional mean.

If all arms are admitted but this joint signature fails, the simple
translation-grammar explanation is insufficient at this state; that failure
does not by itself prove object identity. A passing result means only
translation-grammar compatibility beyond the two frozen annotated left edges:
incomplete dense annotations prohibit claiming that the synthetic coordinates
lack every physical or visual boundary. If either synthetic arm lacks support
or valid-right mass, this four-arm discriminator is unidentified and Wave One
stops without replacing the cue. One bounded greedy release of
`x2,y2,<|box_end|>` per arm is a secondary validity readout and never overrides
the complete-logit primary result.

### Description, location, and complete extent are phase-separated

**Motivation.** Prior units found phrase-geometry chimeras, a late pre-`x1`
owner-basin switch, and a fork whose early left and top boundaries do not
recover complete visible extent.

**Expected mechanism.** Description narrows category support; pre-`x1` state
selects a spatial basin; later coordinate positions refine or replace extent.
No single boundary is assumed to own the complete row.

**Predictions.**

- interventions have a phase-by-readout interaction rather than one uniform
  downstream effect;
- same-description owner donors can alter location without a lexical change;
- identical early location can still admit distinct late extents; and
- a late extent intervention can alter `x2,y2` without retroactively changing
  the already fixed description or `x1,y1`.

**Verification experiment.** Within one admitted case and one fixed donor
construction, apply the same owner contrast separately at row entry,
immediately before the first discriminative description token, pre-`x1`, and
pre-`x2`. Keep candidate rows, recipient-prefix family, and readout definitions
fixed. Prior cells are contextual evidence only and cannot complete this
factorial.

**Falsification experiment.** A single owner intervention that controls phrase,
early location, late extent, and closure under every matched phase would favor
one persistent row owner. Conversely, if all apparent late effects vanish
after token history is matched, the phase account reduces to ordinary
autoregressive mediation.

**Confounds.** Different phases have different token histories and donor
availability. No phase interaction is interpreted unless the case, owner
pair, donor construction, candidate definitions, and readouts are shared, and
the recipient tokens, position identifiers, model input, replacement
location, and phase-specific positive control are exact within each phase.

### A committed row creates persistent entity-specific coverage state

**Motivation.** Current-row chair geometry changes the next category, and a
coherent phrase-plus-geometry row can alter a successor. It is unknown whether
the effect is a durable object record or only a local canonical transition.

**Expected mechanism.** Emitting entity `A` writes a state that selectively
reduces later support for `A` and redistributes support toward distinct,
uncovered entities even after unrelated legal rows intervene.

**Predictions.**

- after committing one byte-distinct row variant of `A`, a second accepted
  row variant of `A` scores lower than after committing matched entity `B`,
  and the symmetric statement holds for `B`;
- the entity-specific difference survives at distance two or three under
  identical, common-support intermediate rows that equalize the declared
  geometry-sorted traversal frontier;
- an `A then B` prefix and a `B then A` prefix, which contain the same physical
  covered set and row count, converge on the same remaining-entity
  distribution; and
- the effect is not reducible to terminal suppression, row-start probability,
  or category-wide inhibition.

**Verification experiment.** Use same-description, equal-token-count coherent
rows for two frozen entities, with two byte-distinct accepted variants per
owner. Measure symmetric entity selectivity at distances one, two, and three,
repeat with commit and probe variants exchanged, score a third owner, and then
compare the order-swapped same-covered-set prefixes.

**Falsification experiment.** Persistent entity-specific coverage is not
supported if the effect vanishes after one common row, follows only the latest
row, or changes only generic continuation and termination. Dependence on `A
then B` versus `B then A` rejects a bare order-free covered set as a sufficient
state, but does not reject every richer order-sensitive entity state.

**Confounds.** Exact token repetition, `COCO` omissions, overlapping same-
class entities, canonical geometry order, incompatible clamped rows, and loose
box overlap can all mimic or hide coverage. Byte-distinct owner aliases,
common-support and canonical-frontier gates, a third-owner control, the unique
physical-entity ledger, and the order-swap control are mandatory.

### Current-row influence is local traversal inertia

**Motivation.** An autoregressive prefix can advance a learned list phase
without storing an order-free covered set.

**Expected mechanism.** A row changes the immediate successor distribution;
later differences are mediated by the generated successor sequence. Once
intermediate rows are made identical, the earlier row loses most of its
effect.

**Predictions.**

- the free-trajectory total effect is larger and longer-lived than the
  clamped-intermediate effect;
- the clamped effect is strongest at distance one and approaches zero by
  distance two or three;
- same-covered-set order swaps remain distinguishable; and
- differences follow absolute row phase or geometry-sorted rank rather than
  selective suppression of one physical owner.

**Verification and falsification.** The same clamped-distance and order-swap
panel adjudicates this alternative. Persistent symmetric entity suppression
under common intermediates falsifies a purely local traversal account.

## Sequential Admission Plan

The unit is intentionally adaptive. A later wave may run only after its parent
question has an interpretable result.

### Wave Zero: freeze cases and unique physical entities

Before any new conclusion-owning model score is observed:

1. resolve one exact fixed-prefix state per case;
2. resolve the canonical rebased model image through the frozen source-data
   row, then freeze its path, pixel dimensions, digest, processed visual grid,
   prompt, adapter, token prefix, prefix digest, row index, and accepted object
   references;
3. build a case-local unique physical-entity ledger from accepted `COCO`
   annotations plus human-confirmed unlabeled `COCO-80` entities;
4. inspect each ambiguous region in original context and one or more display-
   only interpolated crops; and
5. mark every entity and reference box as accepted, ambiguous, or excluded;
   and
6. for any cross-row coverage case, freeze two byte-distinct accepted row
   variants per treatment owner, a third-owner control distinct from both
   treatments and every possible common clamp, all candidate common clamp
   rows, and every geometry-sorted rank before clamp-support scores are opened;
   and
7. run only the exact no-operation and self-state replacement trust controls,
   then freeze `epsilon_noop`, `delta_entity`, and `delta_third` before any
   treatment-branch outcome score is opened.

The canonical rebased image resolved by the source-data row remains the only
model input. The raw `COCO` image and direct interpolation-resized crops are
human-review surfaces only. Display metadata records which source was viewed,
raw crop coordinates, interpolation method, scale factor, and source path so
that enlarged pixels cannot be confused with model evidence.

A case may support geometry or phase analysis while being excluded from
coverage claims if unique entity ownership remains ambiguous. There is no
silent replacement after model scoring begins. Any pre-score replacement is
recorded in the fixture.

### Wave One: donor-state late transport and box-grammar discrimination

Primary case: image `7818`, paired `wine glass` entities with already eligible
block-`23` donor states and a trusted block-`13` control.

Secondary grammar case: image `19432`, dense repeated chairs with real-edge,
inter-edge non-boundary, and far dense-field non-boundary horizontal cues.

Image `7818` first asks only whether donor-state information remains readable
after byte-identical early coordinate history. Its current crossed boxes do
not identify physical ownership. A physical-owner verdict is attempted only
if Wave Zero finds an owner-admissible anchor within the four predeclared
cases. If donor-state transport is unidentified and the admitted non-boundary
coordinate sweep reproduces the late movement, simple box grammar becomes the
leading bounded explanation for these states.

### Wave Two: decision-phase decomposition

Run only after Wave One identifies at least one valid donor-state or grammar-
specific handle.

The primary phase claim must be within one admitted case, using one donor-
construction procedure, one fixed owner pair, one recipient-prefix family,
and identical candidate and readout definitions at every boundary. Image
`12576` is the preferred candidate because different utensil descriptions and
part-versus-whole extents are both available. Image `7818` may provide a
same-description location/extent capability control, but cannot fill the
description cell. Historical cells remain contextual evidence and never fill
the primary factorial.

The primary output is a phase-by-readout matrix:

| Intervention boundary | Description readout | `x1,y1` location readout | `x2,y2` extent readout | Next-row readout |
|---|---|---|---|---|
| Row entry | required | required | required | optional |
| Immediately before the first discriminative description token | required | required | required | optional |
| Pre-`x1` | fixed or already emitted | required | required | optional |
| Pre-`x2` | fixed | fixed | required | optional |

The matrix must report the sequential teacher-forced candidate margins defined
above and separate physical-owner outcomes, not only hidden-state cosine
similarity or attention intensity. Every cell needs a phase-matched positive
control showing that the donor construction is effective at that boundary. If
case, donor construction, owner pair, recipient family, or readout changes
across cells, report only case-specific intervention capability; do not call
the matrix evidence of phase-separated object construction.

### Wave Three: cross-row influence horizon

Run only after at least one current-row treatment has a frozen physical owner
and valid coherent row. Primary clean case: image `7818`. Primary dense case:
image `19432`. Image `7816` is an incomplete-label stress case and may
contribute coverage evidence only if its person identities are frozen without
ambiguity.

For each admitted same-description entity pair:

1. append `R_A_commit` or `R_B_commit` with equal grammar and token count;
2. measure row `t+1` directly;
3. append the same legal row `C_1` to both branches and measure row `t+2`;
4. append the same legal rows `C_1,C_2` to both branches and measure row
   `t+3`;
5. score `R_A_probe`, `R_B_probe`, one exact-repeat control, and a third frozen
   owner under both branches, then repeat with commit and probe variants
   exchanged;
6. compare `A then B` with `B then A` to separate a same-covered-set result
   from last-row and order effects; and
7. run a small free-trajectory sidecar from the unclamped `A` and `B` branches.

The candidate clamp set is frozen before its support-only admission pass, and
downstream entity scores remain unopened until admission is final. Each common
row must be a distinct physical entity from `A`, `B`, and every earlier common
row, parse as one coherent legal row, and put both branches at the same
declared geometry-sorted
traversal frontier. Specifically, `rank(C_1)` must be later than both treatment
owners and `rank(C_2)` must be later than `C_1`.

For branch `b`, let `S_best_b` be the best length-normalized row score among
the predeclared clamp candidates. Let `S_native(C_i)` be the same row's score
under its frozen geometry-sorted teacher-forced predecessor prefix in the same
case. A common row is admitted only when:

```text
S(C_i | branch A) >= S_best_A - log(2)
S(C_i | branch B) >= S_best_B - log(2)
S(C_i | branch A) >= S_native(C_i) - log(2)
S(C_i | branch B) >= S_native(C_i) - log(2)
abs[S(C_i | branch A) - S(C_i | branch B)] <= log(2)
```

Thus its geometric-mean per-token probability is at least half that of the
best predeclared clamp in each branch and at least half its native canonical
support, while the branch-wise ratio is at most two. The thresholds are fixed
here and cannot be relaxed after downstream results are viewed. An
unsupported, malformed, owner-ambiguous, frontier-misaligned, or branch-
incompatible row is not a valid clamp. The first failed clamp ends
interpretation at the previous distance rather than being replaced after its
downstream outcome is seen.

## Predeclared Four-Case Candidate Panel

The first panel contains four predeclared candidate images. Wave Zero converts
an admitted image into an exact fixed-prefix case by recording its prefix,
owner ledger, coordinate references, and fixture digest. Different waves may
use different admitted cases, but no wave may silently substitute a second
prefix from the same image and still call it the same case.

| Image identifier | Primary role | Admission requirement |
|---|---|---|
| `19432` | Dense same-class chair geometry, box-grammar control, and cross-row dense case | Freeze target, adjacent, and relevant intervening chairs as unique physical entities; multi-chair unions are controls, not owners. |
| `12576` | Utensil description and visible part-versus-whole extent | Freeze the reportable fork, spoon, and relevant coherent extent references; the fork-head box remains a part reference. |
| `7818` | Clean repeated-class `wine glass` owner control and residual-state transport | Reproduce parent donor eligibility and the trusted block-`13` and block-`23` semantics in full-model 32-bit floating point. |
| `7816` | Dense-person, incomplete-annotation stress case | Use enlarged original-image crops to freeze distinguishable people; exclude the case from coverage if the woman, orange-shirt man, white-haired head, or neighboring person cannot be assigned consistently. |

The canonical source-data file is:

```text
/data/CoordExp/.worktrees/coordexp-infras/outputs/coordexp_swift/infer/
val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl
```

It resolves these model-input images:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images/val2017/000000019432.jpg
/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images/val2017/000000012576.jpg
/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images/val2017/000000007818.jpg
/data/CoordExp/public_data/coco/rescale_32_1024_bbox/images/val2017/000000007816.jpg
```

The image-`7818` parent donor is bound specifically to the `1248`-by-`832`
rebased image whose Secure Hash Algorithm 256-bit digest is
`d8c9bdea180ba1070daf5792c44187424f66fc319ada9e1ad1bbf86eda348272`.
The parent donor may be reused only after this digest, processed visual grid,
prefix digest, and replacement boundary all match its receipt. Any other image
input requires donor recapture and requalification.

The raw images below are retained only for full-context human review and
display-only interpolation:

```text
/data/CoordExp/public_data/coco/raw/images/val2017/000000019432.jpg
/data/CoordExp/public_data/coco/raw/images/val2017/000000012576.jpg
/data/CoordExp/public_data/coco/raw/images/val2017/000000007818.jpg
/data/CoordExp/public_data/coco/raw/images/val2017/000000007816.jpg
```

The panel expands from four to at most eight exact cases only when the clean
and dense mechanism signatures conflict. It does not expand merely because a
case is inconclusive and does not expand to estimate population metrics.

## Primary Measurements

Keep the following outputs separate:

1. complete coordinate-token logit vectors stored in 32-bit floating point;
2. per-slot and joint coherent-row log probability;
3. physical-owner attribution for description, early location, and late
   extent;
4. terminal-versus-row-start logit margin;
5. per-owner suppression components and frozen effect floors in both
   reciprocal commit/probe alias directions, symmetric entity selectivity as
   a summary, an exact-repeat control, and one third-owner stability score;
6. exact row validity and natural row closure;
7. cross-row effect at distances one, two, and three;
8. same-covered-set order sensitivity; and
9. free-trajectory reconvergence both at absolute row distance and after
   content alignment that ignores a single inserted or repeated entity; and
10. suffix-row token reconvergence without claiming that the full historical
    prefixes become identical.

Attention maps and hidden-state similarity may help localize a route, but they
cannot own any causal conclusion.

## Interpretation Table

| Observation | Bounded interpretation |
|---|---|
| Block-`23` donor changes the late-coordinate margin under supported identical `x1,y1`; block `13` does not | Supports donor-state late-coordinate transport, not physical-owner transport by itself. |
| The same contrast passes the owner-admissible anchor gate and switches uniquely attributed late owners | Supports bounded physical-owner transport beyond the first coordinate at that admitted state. |
| Donor effect is unidentified after identical `x1,y1`, while admitted non-boundary cues move `x2` with stable translated width | Supports simple autoregressive box grammar as a bounded local explanation; it does not resolve the donor-state question. |
| Within one case and donor construction, description, location, and extent respond differentially across separately controlled boundaries | Supports phase-separated object construction at that admitted state; does not require an explicit slot. |
| Only row `t+1` changes under clamping and later rows reconverge | Supports local transition or traversal inertia, not persistent coverage. |
| Effect survives to row `t+2` or row `t+3`, but changes only terminal or row-start margin | Supports persistent generic continuation state, not entity coverage. |
| Both owner-suppression components exceed the frozen floor in both reciprocal alias directions beyond an equalized frontier, while the third owner passes both stability tolerances | Supports a bounded persistent entity-specific state rather than one-sided averaging or exact token repetition alone. |
| `A then B` and `B then A` differ despite identical physical covered set and row count | Rejects a bare order-free covered set as a sufficient state; an order-sensitive persistent state remains possible. |
| Free trajectories differ but clamped trajectories do not | Downstream divergence is mediated by generated rows rather than direct persistence of row `t`. |
| Image `7816` remains visually ambiguous after enlargement | Exclude it from unique-entity coverage claims; retain it only as an ambiguity stress case. |

## Execution Semantics Required for Interpretation

- Use the geometry-sorted Gaussian-coordinate adapter at checkpoint step
  `4887`, matching the parent coordinate and portability units.
- Resolve the canonical rebased model image from the frozen source-data row;
  keep its digest, processed visual grid, processor, prompt, tokenizer,
  coordinate tokens, and `do_resize=false` semantics unchanged.
- Use repetition penalty `1.0`; the historical `1.10` heuristic is not part of
  the mechanism.
- Use physical batch size one for primary teacher-forced scoring, state
  capture, state replacement, and greedy controls.
- Use full-model 32-bit floating-point (`float32`) arithmetic with Scaled Dot
  Product Attention for conclusion-critical donor and clamped-prefix
  comparisons. Selected logits and score arithmetic are also stored as
  `float32`.
- A physical batch size four sampled sidecar is optional only after the exact
  batch-one control is reproduced and the arm cannot change the primary
  conclusion.
- The first sampled sidecar uses eight paired seeds, temperature `0.4`, top-p
  nucleus threshold `0.95`, and repetition penalty `1.0`. Expand to sixteen
  only when the eight-seed result conflicts with greedy or teacher-forced
  evidence.
- Human-view interpolation never enters model preprocessing, feature capture,
  or evaluation geometry.

## Conclusion-Critical Admission Gates

These are scientific identification checks, not a general runtime-assurance
layer:

- fail donor reuse when the completed parent portability receipt digest, any
  recaptured donor-state digest, token-history digest, position-identifier
  digest, canonical model-image digest, processed visual grid, prefix digest,
  or replacement boundary differs from the parent receipt;
- keep a same-coordinate result at donor-state level when either crossed box
  is invalid, branch-unsupported, or lacks unique ledger ownership;
- end the cross-row horizon at the previous distance when a common clamp fails
  the fixed support, branch-compatibility, or canonical-frontier rule;
- prohibit a coverage claim when commit and probe rows are byte-identical,
  fail reciprocal aliasing, or do not share one accepted physical owner;
- prohibit a phase-interaction claim when case, owner pair, donor construction,
  recipient-prefix family, candidate definition, readout, or phase-specific
  positive control differs across cells; and
- preserve every failed admission as a result rather than replacing the case,
  anchor, alias, or clamp after outcome inspection.

## Reused Surfaces and Minimal Build

Reuse before adding code:

1. the exact-prefix and target-fixture surfaces from the complete-box
   coordinate-release unit;
2. the fixed-encoding donor capture and recipient replacement seam from the
   image-`7818` geometry-state portability unit;
3. the progressive coordinate-release loop and complete coordinate-logit
   capture;
4. the canonical assistant-continuation renderer and detection-row parser;
5. the existing human-review export and accepted `COCO` annotations as
   candidate evidence from which Wave Zero builds a new unique physical-entity
   ledger;
6. the standard model loader and request-scoped sampling infrastructure.

The executed unit added only the smallest missing pieces:

- one four-case fixture and case-local unique physical-entity ledger;
- one paired prefix constructor that can append coherent treatment rows and
  common clamped intermediate rows;
- one analyzer for the phase matrix, distance-one-through-three entity scores,
  same-covered-set order control, and free-trajectory reconvergence; and
- one display-only crop helper that records source coordinates and
  interpolation metadata.

Both consumers remain under `scripts/research/`. Repeated semantic logic should
move into `src/analysis/` only after a second independent experiment needs the
same seam. No OpenSpec change was justified.

## Artifact Handle

The two immutable Wave One run roots and receipt digests are:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-17-object-specific-geometry-transport-and-cross-row-influence-horizon/image7818-donor-state-late-coordinate-float32-20260717a/
receipt SHA-256: 923a232fffc2723628255caa61719ac9b33038072c21f00cb4112d745d12f38c

/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-07-17-object-specific-geometry-transport-and-cross-row-influence-horizon/image19432-nonboundary-box-grammar-float32-20260717a/
receipt SHA-256: 14aeb6493247c930bfb101dff54a8cabb22ea7f476985406704584688c580696
```

Each compact receipt records the resolved immutable run root, source identity,
checkpoint and resolved configuration, exact fixtures and prefix digests,
conditions, raw score or continuation outputs, and the frozen primary
classification. See `results.md` for the verified interpretation boundary.

## First Smoke and Rough Cost

The first smoke is deliberately smaller than the full four-case unit:

```text
case: image 7818
conditions: target donor and paired donor
candidate shared coordinate histories: target x1,y1 and paired x1,y1
admission: exact parent image/state identity plus branch-wise early-coordinate support
controls: unrestricted, self-state replacement, and trusted block 13
readout: teacher-forced late-coordinate margin plus one greedy release
precision: full-model float32
physical batch size: one
```

Only admitted coordinate histories enter the late readout; an unsupported
cross-history is not interpreted as a null. This smoke tests whether the
existing portability seam can carry a donor-state effect beyond identical
early coordinates. It does not identify physical ownership and does not test
cross-row coverage.

If exact, add the frozen image-`19432` non-boundary coordinate sweep and enter
later waves only if it admits a positive mechanism handle. The complete first panel is still a four-
case mechanism study, not a validation-set estimate. Parallel graphics-
processing-unit execution may assign independent cases to separate devices,
but each primary request remains physical batch size one.

## Stop Rules

- Stop donor-state late-coordinate transport at image `7818` if every
  branch-supported shared coordinate history erases the donor contrast while
  every trust control passes.
- Do not make a physical-owner transport claim from image `7818`. Stop that
  subquestion as unresolved unless another predeclared case passes the owner-
  admissible common-anchor gate.
- Stop box grammar as a sufficient explanation only if an owner-admissible
  donor contrast changes unique late-owner attribution under byte-identical
  coordinates and the trusted control layer does not.
- Stop phase decomposition if no admitted intervention has a valid phase-
  matched positive control; do not interpret a null from an ineffective seam.
- Stop persistent entity-specific coverage if either owner-suppression
  component fails its frozen floor after row `t+1`, either reciprocal alias
  direction fails, the third owner exceeds either tolerance, or only generic
  continuation changes. If same-
  covered-set order swaps remain strongly different, stop only the bare order-
  free covered-set interpretation and retain richer order-sensitive state as
  unresolved.
- Stop any coverage claim for a case whose unique physical-entity ledger is
  ambiguous.
- Expand from four to at most eight cases only for conflicting clean-versus-
  dense signatures, not for inconclusive cases and not to chase significance.
- Do not start a 256-image training screen from a single positive state. A
  later training proposal requires a reproducible endogenous target, a direct
  preservation control for native detection and language capability, and a
  separate user authorization.
- Do not infer a final carrier. A positive persistent state does not decide
  whether the eventual treatment should be native loss shaping, prefix state,
  residual routing, visual feedback, or explicit memory.

## Non-Goals

- no population mean Average Precision, recall, hallucination, or prevalence
  estimate;
- no STOP suppression, repetition-penalty tuning, or decoding-policy
  optimization;
- no generic coverage percentage or remaining-count auxiliary loss;
- no object-slot, Detection Transformer-style query, external detector, or
  stronger backbone;
- no claim that interpolation recovers missing image information;
- no claim that one case generalizes across categories or scene densities;
- no training, architecture promotion, or stable compatibility contract.

## Next State

This unit is closed after Wave One. Neither Wave Two nor Wave Three is
admitted, and no training or architecture promotion follows from these
results. A future natural-overlap or graded on-manifold discriminator must be
proposed as a separate research unit with its own frozen fixtures and admission
contract; it must not repair these failed controls post hoc.
