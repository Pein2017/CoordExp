---
title: PVCI Native Identity-by-Slot Counterbalance
description: Crosses same-class object identity with first versus second committed row to separate generic row-slot/order effects from object-specific commitment.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-11-pvci-native-identity-slot-counterbalance
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
tags:
  - coordexp-swift
  - research-unit
  - qwen3-vl
  - autoregressive-commit
  - causal-counterfactual
  - row-position
updated: 2026-07-11
---

# PVCI Native Identity-by-Slot Counterbalance

## Question

Does the sign of the native row transition follow the committed same-class
object, the first/second row slot, or the forward/reverse pair order?

The preceding five-event unit found that canonical first/on-policy `A`
suppressed itself while far, later/off-order `B` promoted itself. Because A was
the first geo object and B was later in every event, object identity, geometry,
order, and row role were completely confounded. This unit breaks the strongest
part of that confound without training or adding architecture.

## Scope and Decision Relevance

- Functional requirement: committing an object should suppress its revisit
  independently of whether it appears in row slot 0 or 1 and independently of
  whether the local pair order is `A→B` or `B→A`.
- Mechanism alternatives: generic row-slot/list dynamics, raster-direction
  state, identity/geometry priors, exact coordinate-token memory, or an
  object-specific commit transition.
- Experimental handle: symmetric canonical same-image prefixes `A`, `B`,
  `A→B`, and `B→A`, with adjacent pre/post scoring around each committed row.
- Outside scope: training, slots, a new detector, persistent coverage memory,
  full-scene completion, or a claim that COCO unlabeled space is background.

## Frozen Cohort

Reuse exactly the five score-blind A/B pairs from:

`/data/CoordExp/outputs/painted_gt/pvci_native_spatial_commit_field/plan_pure_ce_step4887/selected_spatial_commit_events.jsonl`

No new image or pair may be added based on observed results. A and B retain the
same phrase bytes, are spatially distinct, and B is not the frozen geo
successor. The actual frozen geo successor `G` must now be explicitly rendered
and scored as a control.

## Prefix/Transition Panel

All rows use canonical annotation syntax and token coordinates. For each event:

```text
P       = ordinary prompt
P+A     = A committed in slot 0
P+B     = B committed in slot 0
P+A+B   = B committed in slot 1 after A
P+B+A   = A committed in slot 1 after B
```

The four adjacent commit transitions are:

| Cell | Pre | Post | Committed identity | Slot | Pair direction |
|---|---|---|---|---:|---|
| A0 | P | P+A | A | 0 | canonical-first |
| B0 | P | P+B | B | 0 | off-order-first |
| B1 | P+A | P+A+B | B | 1 | forward A→B |
| A1 | P+B | P+B+A | A | 1 | reverse B→A |

`A→B` is only forward relative to the pair. Because B is deliberately not the
immediate geo successor, it is not called globally on-policy.

## Candidate Panel

At every boundary score the same:

- `A_exact`, `A_near`, `A_mid`;
- `B_exact`, `B_near`, `B_mid`;
- `G_exact`, the actual frozen geo successor row;
- STOP as a separate one-token diagnostic.

Near/mid A/B controls retain the frozen IoU bands and zero exact coordinate-ID
overlap. `G_exact` is not included in the A/B factorial and may have a different
phrase. It directly tests the traversal alternative missing from the previous
unit.

## Primary Contrasts

For target identity `X` and transition cell `c`:

```text
D_c(X) = log P(row_X | post_c) - log P(row_X | pre_c)
```

Primary exact/near contrasts:

- A slot effect: `D_A1(A) - D_A0(A)`;
- B slot effect: `D_B1(B) - D_B0(B)`;
- slot-0 identity contrast: `D_B0(B) - D_A0(A)`;
- slot-1 identity contrast: `D_A1(A) - D_B1(B)`;
- actual-successor gain after every cell relative to A/B candidates.

Raw direct-forward scores are primary. Offline RP1.1 is reported separately.
All candidate-token log-softmax uses float32, stable sums use Python double,
and native model/generation dtype remains bfloat16.

## Competing Hypotheses and Falsifiers

### H1: Generic slot/list transition

Both identities show similar signs when committed in the same slot; changing
slot changes the sign more than changing identity. Falsified if the committed
identity determines suppression at both slots.

### H2: Pair-direction / raster state

Forward `A→B` suppresses B or promotes G/remaining objects, while reverse
`B→A` promotes/repeats A or otherwise behaves off-policy. Falsified if forward
and reverse cells are symmetric after controlling identity and slot.

### H3: Identity/geometry prior

A remains suppressive and B remains promotive in both slots. This can arise
from box size/location/coordinate-pattern priors and is not a commit ledger.
Falsified if each identity's sign changes with slot/order.

### H4: Exact coordinate-token memory

Exact candidates change, but zero-token-overlap near/mid candidates do not
inherit the sign. Falsified by consistent local generalization.

### H5: Object-specific order-robust commit

Whichever identity is appended becomes selectively suppressed at both slots,
the effect extends to near zero-token-overlap geometry, G/other rows remain
available, and free continuation avoids the committed support without STOP or
malformed compensation. This is the only outcome that supports strengthening a
native commit substrate.

## Runtime and Evidence Gates

- Freeze plan, unit, code, config, checkpoint, Git, and argv hashes before
  interpretation.
- Scorer must recompute pair, prefix, candidate, G, token-overlap, and geometry
  invariants from the embedded source ledger.
- Repeat one candidate score at every boundary; require exact per-token equality.
- Run a two-event gate first, then all five only if the contract passes.
- Preserve every attempted event/failure and gate any majority claim by all
  five frozen attempts, not available-only denominators.
- Free continuations are attributed only by unique exact-description plus
  IoU≥0.5 matches; unknown/ambiguous outputs are never positive avoidance.
- STOP remains a different-granularity diagnostic, never a conserved row-mass
  bucket.

## Outcome Map

- H1/H2: next work targets order/row-state training; do not add explicit memory
  yet.
- H3: next match or perturb size/location/coordinate patterns before any ledger
  claim.
- H4: native semantics are token-local; explicit spatial commitment becomes
  better justified.
- H5: strengthen the native prefix/KV write-read path and test persistence in
  free rollout before architecture promotion.
- Inconclusive: do not train a new objective until one single confound is named.

## Stop Condition

This unit ends after the two-event gate and all five frozen events have explicit
outcomes, or immediately on contract failure. It must return one bounded label:
`slot/list`, `pair-direction`, `identity/geometry`, `exact-token`, `order-robust
commit`, or `inconclusive`. Architecture remains `not_promoted` in every case.

## Planned Artifacts

- plan: `/data/CoordExp/outputs/painted_gt/pvci_native_identity_slot_counterbalance/plan_step4887/`;
- scorer: `/data/CoordExp/outputs/painted_gt/pvci_native_identity_slot_counterbalance/full5/`;
- analysis: scorer root plus `/analysis/`.

## Result

Completed on the exact frozen five-event cohort with live step-4887 checkpoint
attestation. The two-event runtime gate scored `2/2` with zero failures and was
correctly restricted to `inconclusive_partial_cohort`. The canonical run scored
`5/5` with zero failures, exact repeated token log-probabilities at all five
boundaries, native bfloat16 execution, float32 candidate readout/log-softmax,
and Python-double aggregation.

Canonical artifacts:

- scorer: `/data/CoordExp/outputs/painted_gt/pvci_native_identity_slot_counterbalance/full5_v1/`;
- analysis: `/data/CoordExp/outputs/painted_gt/pvci_native_identity_slot_counterbalance/full5_v1/analysis_v4/`;
- scorer events SHA256: `1632592dd7640b9931c6ffcb6fc8e217f50b56b5806c417a2c5fee17485a0970`;
- execution receipt SHA256: `04e4e035812a471958350de4910d7357e9c88d375ad7cb7aeca44681271546e6`;
- canonical analysis summary SHA256: `aba1706feae5b7ec621a792c33f2c01c287183ca1abeed5c2d805321ccc752ae`.

Primary raw exact self-delta medians were:

```text
A0 = -11.2734880614
B0 =  +8.9273083250
B1 =  +3.0429731602
A1 =  +6.4386782291
```

`A0` suppressed itself in `5/5` exact events, while `B0` promoted itself in
`5/5`; `B1` and `A1` were mixed/mostly promotive. The effect was overwhelmingly
coordinate-span rather than phrase-span. Consequently, order-robust
object-specific commit was not supported.

The predeclared `G_exact` control showed a real but still confounded descriptive
signal in `4/5` events: G rose after appending A in either slot and fell after
appending B in either slot. Raw median G deltas were:

```text
A0 =  +5.0963803533
B0 =  -9.0588089714
B1 = -15.1346401472
A1 = +10.4478311446
```

Because G is A's frozen canonical successor, that pattern groups by committed
A/B identity and cannot independently identify raster direction, identity,
geometry, or canonical role. It is therefore recorded only as
`identity_conditioned_geo_successor_shift`, not as a raster mechanism.

Final bounded verdict: `inconclusive_mixed_transition`. Architecture remains
`not_promoted`. The next route is the separately frozen canonical-depth unit,
which tests whether `self down / actual successor up` repeats at depths 1 and 2
across different committed identities and geometries.

## Evidence Scope

- Checkout: `/data/CoordExp/.codex/worktrees/69ed/CoordExp`.
- Checkpoint: pure-CE/type-gated long-trained step-4887, live manifest SHA256
  `c8ad1ab01550fc640c67457fec9ad1f8b3bd1b8cef351cb90d41666233b80da1`.
- Config:
  `configs/coordexp_swift/infer/research/pvci_native_commit_pure_ce_step4887_probe.yaml`.
- Frozen cohort: five score-blind events from the completed spatial-commit
  unit; no result-selected replacement.
- Runtime gate: `/data/CoordExp/outputs/painted_gt/pvci_native_identity_slot_counterbalance/gate2_v1/`.
- Canonical scorer/analysis: `full5_v1/` and `full5_v1/analysis_v4/` under the
  same artifact root.
- Sample window: five same-image same-class A/B panels, each with five prefix
  boundaries, seven row candidates, STOP, and two one-row decode policies.
- Known limitation: A is always canonical index 0 and G is A's canonical
  successor; A/B identity, geometry, and canonical role remain inseparable.

## Research Unit Closeout

Observed: The exact frozen five completed without scorer failures. A0 self
suppression was `5/5`, B0 self promotion was `5/5`, later cells were mixed,
and coordinate spans dominated. G rose after A and fell after B in `4/5`, but
that grouping is identity/canonical-role confounded.

Supported: The checkpoint carries a strong teacher-forced prefix-conditioned
transition. It is not symmetric across committed identities or row slots. The
G response is a descriptive `identity_conditioned_geo_successor_shift`.

Not supported yet: An order-robust object commit ledger, visually grounded
coverage memory, persistence beyond the first canonical transition, complete
enumeration, improved STOP calibration, or a detection metric benefit.

Next decider: The frozen canonical-depth unit tests O0, O1, and O2 adjacent
commits with their actual successors and zero-token-overlap near controls.

Promotion decision: `not_promoted`; no architecture or training objective is
authorized by this result.
