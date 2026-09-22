---
title: PVCI Native Canonical Commit Depth
description: Tests whether self-suppression and actual-successor promotion persist across the first three canonical geo-sorted row transitions.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-11-pvci-native-canonical-commit-depth
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
tags:
  - coordexp-swift
  - research-unit
  - qwen3-vl
  - autoregressive-commit
  - causal-counterfactual
  - traversal-depth
updated: 2026-07-11
---

# PVCI Native Canonical Commit Depth

## Question

Under exact teacher-forced canonical prefixes, does the decoder repeat a
`commit current -> promote actual successor` transition beyond the first row,
or was the preceding `A0` effect only a first-row/prefix special case?

The identity-by-slot counterbalance did not support an order-robust commit
ledger. Its actual-successor control did show a bounded identity-conditioned
signature:
after appending canonical first object `A`, actual successor `G` gained; after
appending far/off-order `B`, `G` lost. This unit tests the smallest unresolved
requirement: persistence of that transition along teacher-forced canonical
prefixes. Native free-rollout persistence remains a separate later claim.

## Scope and Decision Relevance

- Functional requirement: after each canonical row, the emitted row should
  become less likely and the next canonical annotated object should become more
  likely.
- Experimental handle: teacher-forced canonical prefixes at depths 0, 1, and
  2, with direct adjacent pre/post candidate scoring.
- Decision: distinguish a reusable teacher-forced order-conditioned prefix
  transition from a first-row special case or a local coordinate/geometry
  effect.
- Outside scope: new training, slots, a detector/proposal module, complete
  coverage, COCO-background claims, or architecture promotion.

## Frozen Cohort

Reuse exactly the five frozen events from:

`/data/CoordExp/outputs/painted_gt/pvci_native_identity_slot_counterbalance/plan_step4887/identity_slot_events.jsonl`

No image may be added or removed based on scores. Every event already contains
at least four canonical geo-sorted annotated objects. Define:

```text
O0 = GT index 0, the existing canonical A
O1 = GT index 1, the existing actual successor G
O2 = GT index 2
O3 = GT index 3
```

The existing far/off-order same-class `B` remains a required scored control;
it is never substituted for a canonical chain object.

## Prefix and Transition Panel

```text
P          = ordinary prompt
P+O0       = first canonical row committed
P+O0+O1    = first two canonical rows committed
P+O0+O1+O2 = first three canonical rows committed
```

Adjacent cells:

| Cell | Pre | Post | Just committed | Actual next |
|---|---|---|---|---|
| C0 | P | P+O0 | O0 | O1 |
| C1 | P+O0 | P+O0+O1 | O1 | O2 |
| C2 | P+O0+O1 | P+O0+O1+O2 | O2 | O3 |

## Candidate Panel

At every boundary score the same score-blind panel:

- `O0/O1/O2/O3_exact`;
- deterministic `near` zero-coordinate-ID-overlap controls for each object
  when feasible;
- `B_exact` as a far/off-order control;
- STOP as a separate one-token diagnostic.

Near controls must be generated before model loading, preserve the declared
IoU band, and never silently fall back to shared coordinate IDs. An infeasible
near candidate stays unavailable and cannot enter a majority numerator.

H1 is not eligible unless, at every depth, at least four of five attempted
events have both the committed-self near control and actual-successor near
control available. Availability is frozen by the score-blind planner.

## Primary Contrasts

For depth `t`:

```text
Self_t = log P(row_Ot | post_t) - log P(row_Ot | pre_t)
Next_t = log P(row_O(t+1) | post_t) - log P(row_O(t+1) | pre_t)
Far_t  = log P(row_B | post_t) - log P(row_B | pre_t)
```

Raw direct-forward scores are primary. Report phrase, coordinate, structure,
and sequence spans separately. Offline RP1.1 is secondary. Candidate logits
and log-softmax are float32; stable aggregates are Python double; native model
and generation remain bfloat16.

## Competing Hypotheses and Falsifiers

### H1: Persistent canonical traversal state

At C0, C1, and C2, committed self is suppressed (`Self_t < 0`) while the actual
successor is promoted (`Next_t > 0`). There is no separate retention/equivalence
outcome in this unit. H1 requires, at every depth:

- exact `Self_t < 0` and exact `Next_t > 0` jointly in at least `4/5` attempted
  events;
- matched zero-coordinate-overlap near `Self_t < 0` and near `Next_t > 0`
  jointly in at least `3/5` attempted events, after passing the `4/5`
  near-availability gate;
- successor specificity `Next_t > Far_t` in at least `3/5` attempted events.

This supports a teacher-forced order-conditioned prefix transition, not an
order-free ledger and not native free-rollout persistence.

Falsifier: the pattern disappears after C0, reverses across depth, or successor
gain is not distinguishable from unrelated rows.

### H2: First-row special case

C0 reaches the exact paired `4/5` criterion, but either C1 or C2 does not. This
means the existing A0 effect cannot be used as evidence of a depth-persistent
teacher-forced transition.

Falsifier: the same paired transition persists at later depths.

### H3: Coordinate/geometry-local transition

Exact paired criteria reach `4/5`, but matched near paired criteria fail to
reach `3/5` at any depth despite the near-availability gate passing. This favors
a local spatial/serialization mechanism over object-state tracking. If the
near-availability gate itself fails, the result is `inconclusive`, not
`coordinate_local`.

Falsifier: nearby geometry inherits the depth-stable direction.

### H4: Generic prefix-length/list state

At a depth, exact `Self_t`, `Next_t`, and `Far_t` share the same sign in at
least `4/5` attempted events, while successor specificity `Next_t > Far_t`
fails to reach `3/5`. If this occurs at two or more depths while H1 is false,
label `generic_prefix_list`. This is ordinary list dynamics, not
commit-to-uncovered behavior.

Falsifier: committed self and actual successor diverge selectively while the
far control does not follow the same transition.

If more than one terminal predicate is true, apply the frozen precedence
`H1 > H2 > H3 > H4 > inconclusive`. This prioritizes persistence, then the
depth failure it was designed to test, before mechanism-locality alternatives.

## Runtime and Evidence Gates

- Freeze unit, planner, plan, scorer, analyzer, config, checkpoint, Git, and
  argv/content hashes before interpretation.
- Reconstruct all O0-O3 rows from the embedded source ledger; never trust role
  labels alone.
- Require exactly five ordered unique frozen event IDs for a final label.
- Any `--event-limit` run is a runtime gate and can only return a partial label.
- Repeat one candidate at every boundary and require exact per-token equality.
- Run a two-event runtime gate, then the canonical five only if it passes.
- Preserve every failure and use all five attempted events as the denominator.
- STOP is not conserved row mass. Unknown free continuation is not avoidance.

## Outcome Map

- H1: next test parser-clean native rollout plus order diversification/hindsight
  selection while preserving the prefix/KV route; do not add slots or explicit
  memory yet.
- H2: stop native-commit claims; test a minimal explicit current-object/commit
  control state before any full architecture.
- H3: prioritize spatial cursor/renderer fidelity and geometry-aware binding.
- H4: target list/row-state training and STOP calibration, not object memory.
- Inconclusive: name one remaining confound; do not expand the cohort or train.

## Stop Condition

Stop after a passing two-event runtime gate and one canonical full-five panel,
or immediately on contract failure. Return exactly one bounded label:
`persistent_canonical_traversal`, `first_row_only`, `coordinate_local`,
`generic_prefix_list`, or `inconclusive`. Architecture remains
`not_promoted` for every outcome.

## Planned Artifacts

- plan: `/data/CoordExp/outputs/painted_gt/pvci_native_canonical_commit_depth/plan_step4887/`;
- scorer: `/data/CoordExp/outputs/painted_gt/pvci_native_canonical_commit_depth/full5/`;
- analysis: scorer root plus `/analysis/`.

## Result

The exact frozen full-five panel completed with `5/5` scored/analyzed events,
zero failures, live step-4887 checkpoint attestation, CUDA execution, native
bfloat16 forward/generation, float32 candidate readout/log-softmax, and exact
raw/RP1.1 repeated token log-probabilities at all `20/20` event-boundaries.

The predeclared raw H1 gates all passed:

| Depth | Exact pair | Near available | Near pair | Next > Far | Common sign |
|---|---:|---:|---:|---:|---:|
| C0 | 5/5 | 5/5 | 4/5 | 3/5 | 0/5 |
| C1 | 5/5 | 5/5 | 5/5 | 4/5 | 0/5 |
| C2 | 4/5 | 5/5 | 3/5 | 3/5 | 0/5 |

Raw sequence-delta medians `(Self exact, Next exact, Far exact, Self near,
Next near)` were:

```text
C0 = (-11.2735, +5.0964, +4.7492, -2.6197, +3.8743)
C1 = ( -5.6035, +3.4514, -1.1731, -1.2049, +2.0181)
C2 = ( -4.0398, +1.5102, +1.6224, -3.6482, +0.4607)
```

The transition is predominantly coordinate-span, not phrase/structure. It is
also threshold-tight: C0 specificity and C2 near/specificity sit exactly at the
frozen `3/5` floor, and C2 exact is `4/5`. RP1.1 preserves exact self-down/next-
up at all depths but C1 successor specificity falls to `2/5`. Raw remains the
predeclared primary view.

Final bounded label: `persistent_canonical_traversal`. This means a repeated
teacher-forced, canonical-prefix transition across depths 0-2. It does not mean
an order-free object ledger or native long-horizon rollout.

Free continuation was retained as secondary side evidence, not part of H1. Of
30 one-row continuations, `17` strictly matched an annotated-uncovered object,
`13` were unmatched/unknown, and `0` strictly matched a covered object. Only
`8/30` selected the immediate canonical successor; `9/30` selected a different
annotated-uncovered object. Among the 13 unknown rows, 10 were geometrically
closer to a remaining annotation than a covered annotation and 12 had a phrase
matching some remaining annotation, but these are not counted as avoidance.
This side evidence suggests that write/selection/binding quality, rather than
the existence of the teacher-forced suppression transition alone, is the next
uncertainty.

Canonical artifacts:

- gate: `/data/CoordExp/outputs/painted_gt/pvci_native_canonical_commit_depth/gate2_v1/`;
- scorer: `/data/CoordExp/outputs/painted_gt/pvci_native_canonical_commit_depth/full5_v1/`;
- analysis: `/data/CoordExp/outputs/painted_gt/pvci_native_canonical_commit_depth/full5_v1/analysis_v1/`;
- scorer events SHA256: `fcb7ecc59a085cbe2db9d4756b3879cc373d5ddd01a1516e0671c98bdec486b0`;
- execution receipt SHA256: `ee1263458ef50837a91316615d1abb010dd779226b80cc6b9a24ffb25f4195d6`;
- analysis summary SHA256: `426a97a9675cc382074cd4bb8e829e2d1b167ffea2866276c6fe560d12ee5748`.

## Evidence Scope

- Checkout: `/data/CoordExp/.codex/worktrees/69ed/CoordExp`, dirty state
  content-hashed in the runtime receipt.
- Checkpoint: pure-CE/type-gated long-trained step-4887; live manifest SHA256
  `c8ad1ab01550fc640c67457fec9ad1f8b3bd1b8cef351cb90d41666233b80da1`.
- Config SHA256:
  `bf316926ca35afc92aa90e81b12a9c75f8bc12e9f8799f75a8423dd9c6e3d7c2`.
- Frozen plan SHA256:
  `917e89f1eb6b174a2709a9ba7788b9ca9f7e07f7e946d06a30ee8a6c0e3cc9ac`.
- Pre-run unit SHA256:
  `3e7d2ee5dc66ef8d68a392bd743dfa80e74aecd958e18a4d3b1cd3ca89c6de07`.
- Sample window: five frozen images, three adjacent canonical transitions per
  image, exact/near/far rows at four boundaries, plus 30 one-row continuations.
- Known limitation: canonical GT prefixes privilege the trained order and do
  not reproduce endogenous prefix errors; COCO annotations remain incomplete.

## Research Unit Closeout

Observed: Self suppression and actual-successor promotion repeat across C0-C2
under canonical teacher forcing and pass every frozen raw H1 gate. Free
continuation is much noisier: 17/30 strict uncovered matches, only 8/30 the
immediate successor, 13/30 unknown, and no strict covered repeat.

Supported: A depth-persistent, order-conditioned prefix transition exists in
the checkpoint. It is mostly coordinate-level and partially generalizes to
zero-coordinate-ID-overlap near geometry.

Not supported yet: Parser-clean endogenous rollout persistence, an order-free
object ledger, reliable object-to-row binding, complete enumeration, STOP
calibration, or a detection-metric improvement.

Next decider: Compare raw self-generated prefixes with parser-clean/hindsight-
canonicalized prefixes for the same generated object. This separates a bad
commit write/binding from failure to read and retain a valid commit.

Promotion decision: `not_promoted`; do not add slots or explicit memory from
this result alone.
