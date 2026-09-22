---
title: Native Sibling-Row Branch Value and Commit Crossover Results
description: Natural complete rows change the immediate successor distribution, but no safe greedy branch-value mismatch or general physical-object covered set is established.
type: investigation
role: research-result
authority: non_normative_research
architecture_promotion_status: not_promoted
implementation_status: complete_for_unit
unit_id: 2026-07-17-native-sibling-row-branch-value-and-commit-crossover
topic: qwen3-vl-dense-enumeration
status: complete
evidence_status: verified
conclusion_status: state_local_successor_effect_without_general_commit_or_branch_value_handle
updated: 2026-07-17
---

# Native Sibling-Row Branch Value and Commit Crossover Results

## Verdict

This unit establishes a native cross-row state effect, but it does not establish
a general physical-object covered set or a greedy branch-value failure.

The bounded conclusions are:

1. **No safe greedy branch-value mismatch was identified.** At none of the
   three released owner-resolved states did a non-greedy natural sibling have a
   positive, safety-preserving Greedy Branch-Value Gap at Horizon Four
   (`G_H`). The 256-image training-screen gate therefore remains closed.
2. **One state has a strict reciprocal immediate-successor crossover.** On
   image `15254` at row zero, a natural `bowl` row makes the overlapping
   `carrot` owner the next action, while a natural `carrot` row makes the
   `bowl` owner the next action. The result passes every frozen exact-row,
   safety, and no-reversal gate in both Brain Floating Point 16-bit
   (`bfloat16`) and full-model 32-bit floating-point (`float32`) execution.
3. **The result is not a general commit mechanism.** Prefix State 56 (`P56`)
   on image `12576` and row zero on image `7574` reject reciprocal crossover;
   image `2299` is refused because its parent contains an unresolved physical
   owner. The preregistered requirement of three states across two images is
   not met.
4. **The positive state is structurally ambiguous.** Its `bowl` and `carrot`
   boxes have approximately `0.797` Intersection over Union (`IoU`), with the
   carrot box entirely inside the bowl box. It can therefore reflect a local
   compound-region serialization, semantic complementation, or two-label
   anti-repetition rather than a spatially indexed physical-object ledger.

The durable mechanism statement is:

```text
a complete naturally generated row can write executable successor state
  !=
the model has a general order-free covered-object ledger
```

## Frozen Scope

All scientific branches were complete rows naturally sampled from one exact
native parent prefix. No row token, description, coordinate, visual feature,
attention value, residual state, terminal logit, or geometry was forced or
edited.

The source-consistent runs used the step-4,887 geometry-sorted Qwen3
Vision-Language (`Qwen3-VL`) adapter with:

- physical batch size one;
- Scaled Dot-Product Attention (`SDPA`);
- `bfloat16` model parameters;
- temperature `0.4`;
- top-p nucleus threshold `0.95`;
- repetition penalty `1.0`;
- 32 one-row discovery samples per selected state;
- every admitted sampled exact row variant;
- eight fresh paired confirmation seeds per exact row; and
- one branch row plus at most four naturally generated suffix rows.

Selected reductions and bootstrap arithmetic used `float32`. The only
conclusion-changing positive state was replayed with all 2,149,097,472 model
parameters converted to `float32`, while retaining the same branch token
identities, parent-prefix hash, confirmation seeds, decode policy, `SDPA`, and
physical batch size one.

The `float32` arm is conditional robustness evidence. Its parent prefix and 32
branch rows were discovered under the source-consistent `bfloat16` runtime and
then replayed unchanged. It does not show that the same parent, branch-support
frequencies, or complete trajectory would be reached end to end under native
`float32` sampling.

## Canonical Evidence

| State | Runtime | Canonical analysis | Secure Hash Algorithm 256-bit digest |
|---|---|---|---|
| image `12576`, `P56` | `bfloat16` | `analysis-p56-v2/analysis.json` | `fe891e9350c318884ddf0375811bb409b8590e906d8148d62672a4602cba47c8` |
| image `7574`, row zero | `bfloat16` | `analysis-image7574-row0-v2/analysis.json` | `82590b21cbc4a38759fcfc95041b524bb40075f18cad04537843a7aa3ee9443f` |
| image `2299`, Prefix State 54 | `bfloat16` | `analysis-image2299-p54-v2/analysis.json` | `cca6494ff14569ab0e9551acd1d190a7c234ec354736d70acbfbeadfd0e9a1fb` |
| image `15254`, row zero | `bfloat16` | `analysis-image15254-row0-v2/analysis.json` | `fd4749f374b1328834b6323d1199908326296c93fd2c76986373ff64f5ab560e` |
| image `15254`, row zero | full-model `float32` | `analysis-image15254-row0-fp32-v2/analysis.json` | `fecdb507765e745caad83bf437f336fe815bb6a7e0ffe819c2e7b58e63d8e003` |

Every path above is relative to:

```text
/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/
2026-07-17-native-sibling-row-branch-value-and-commit-crossover/
```

The full-model `float32` replay root is:

```text
wave3-image15254-admitted-exact-variants-k8-fp32/
```

It contains 32 exact sampled variants times eight paired seeds, for 256 valid
calls. All 32 receipt checks pass; no call is invalid or dropped, and all 256
reach the four-row analysis horizon. The analyzer found no unmatched or
automatically ambiguous candidate requiring new human review.

## State-Level Results

| State | Admitted owners | Owner-level Branch Value at Horizon Four (`Q_H`), lower / upper | Formal `G_H`, lower / upper | Reciprocal commit result |
|---|---|---:|---:|---|
| image `12576`, `P56` | pizza `coco-ann:1571077`; left cup `coco-ann:678023` | pizza `4.4861 / 4.5972`; cup `4.4837 / 4.4891` | `0.0024 / 0.1081`; not significant and unsafe | rejected: pizza direction `0`; cup direction `1` |
| image `7574`, row zero | bowl `coco-ann:1535235`; microwave `coco-ann:1641005` | bowl `4.6125 / 4.6125`; microwave `4.7083 / 4.8542` | `0.0958 / 0.2417`; confidence bounds cross zero and safety fails | rejected: bowl direction `-0.375`; microwave direction `0` |
| image `2299`, Prefix State 54 | three person owners | descriptive only | refused | refused because one parent row remains review-unknown |
| image `15254`, row zero, `bfloat16` | bowl `audit:dense-union-51-audit-v1:15254:0001`; carrot `coco-ann:1563326` | bowl `5.0000`; carrot `4.9886` | `0`; greedy bowl is maximizing | supported locally |
| image `15254`, row zero, `float32` | same two owners | bowl `5.0000`; carrot `5.0000` | `0`; equal values | supported locally |

`Q_H` counts the branch row itself plus at most four suffix rows, so five is
the ceiling. This makes the branch-value analysis sensitive to a ceiling
effect. The correct conclusion is “no positive handle detected,” not “all
natural branches have identical long-horizon value.”

## No Greedy Branch-Value Handle

At image `12576`, the lower-world pizza-versus-greedy-cup point gap is only
`0.002415`; the upper-world gap is `0.108092`. Their paired confidence bounds
cross zero, and the unsupported-output safety gate fails. The result is
compatible with a weak branch-value difference but does not identify one.

At image `7574`, the non-greedy microwave branch has positive lower- and
upper-world point gaps, `0.095833` and `0.241667`. Both confidence intervals
cross zero and unsupported-output safety again fails. Sampling a different
first object therefore changes the trajectory, but the unit cannot call that
change a safe unique-object gain.

At image `15254`, the greedy bowl is already maximizing under `bfloat16`, and
both owners reach the ceiling under `float32`. There is no alternative branch
advantage.

The frozen stop rule is therefore active:

> Do not propose branch-value training when no safety-preserving `G_H` appears
> after three admitted states.

No loss, reinforcement-learning objective, policy update, or 256-image
training screen is promoted from this unit.

## Image 12576: One-Way Traversal, Not Reciprocal Commit

The pizza branch is followed by the cup, but the cup branch does not return to
the earlier pizza. The owner-level crossover contrasts are:

```text
C_pizza = 0.0, simultaneous 95-percent interval [0.0, 0.0]
C_cup   = 1.0, simultaneous 95-percent interval [1.0, 1.0]
```

All 207 exact pizza-versus-cup row-variant pairs preserve this one-way pattern.
That is strong evidence for a deterministic local transition, but it is the
signature expected from a forward geometry-sorted traversal frontier:

```text
earlier pizza -> later cup
later cup     -> do not rescan the omitted earlier pizza
```

It rejects strict reciprocal commit at this state.

## Image 7574: The First Row Changes the Trajectory without Commit

The owner-level contrasts are:

```text
C_bowl      = -0.375, simultaneous 95-percent interval [-0.4625, -0.2875]
C_microwave =  0.000, simultaneous 95-percent interval [-0.0875, 0.0875]
```

Neither direction passes. Exact variants also violate the no-reversal rule.
The microwave-first trajectory can reduce some duplicate behavior, but this
is not reciprocal physical-owner suppression and does not identify a covered
set.

## Image 2299: Correct Refusal

Three person owners pass the discovery support gate, but a parent-prefix person
row remains unresolved after blind review. The analyzer therefore refuses
owner-level `Q_H`, `G_H`, and crossover claims rather than assigning a dense
person fragment to the nearest annotation.

This refusal is scientifically useful. A dense repeated-category scene is the
most discriminating setting for physical-instance commit, but it is also where
an incorrect owner assignment can manufacture the desired crossover. The
state must be rebuilt around a fully resolved parent before reuse.

## Image 15254: Numerically Stable Reciprocal Successor State

Under `bfloat16`, the reciprocal contrasts are:

```text
C_bowl   = 0.795455, simultaneous 95-percent interval [0.579545, 1.011364]
C_carrot = 1.000000, simultaneous 95-percent interval [0.784091, 1.215909]
```

Under full-model `float32`, the point estimates are unchanged:

```text
C_bowl   = 0.795455, simultaneous 95-percent interval [0.545455, 1.045455]
C_carrot = 1.000000, simultaneous 95-percent interval [0.750000, 1.250000]
```

For both owners, in both unknown-support worlds, the lower 95-percent bound of
the good-minus-bad contrast is exactly `1.0`. All 231 exact bowl-versus-carrot
row-variant pairs pass both directional gates with no reversal in both
numerical runtimes. The effect is therefore not a `bfloat16` batch-one basin
accident or one peculiar coordinate realization.

This licenses the narrow statement:

> At one clean row-zero state, replaying either of two naturally generated
> complete rows reproducibly changes the immediate next-owner distribution to
> the other owner.

It does not identify the internal carrier. The physical scene is a meal tray;
the accepted bowl box `[714, 2, 1047, 323]` contains the carrot box
`[733.473, 17.809, 1022.015, 313.198]`. Their `IoU` is approximately `0.797`,
and the carrot box is fully contained in the bowl box. Consequently, all of
these explanations remain live:

- object-specific commit and remaining-object redistribution;
- category-level anti-repetition;
- lexical or semantic complementation between container and content;
- serialization of two labels attached to one compound visual region; and
- a finite two-owner list in which the only alternative is “the other one.”

## Mechanism Update

The unit changes the belief state in two ways.

First, cross-row prefix influence is not merely vague inertia. A complete
native row can causally select a sharply different immediate successor, and
that effect can be stable across row geometry variants and numerical precision.
The prefix is therefore an executable state carrier at least locally.

Second, the carrier is not yet shown to encode a physical covered set. The
three interpretable states express three different transition regimes:

```text
image 12576: forward-only traversal frontier
image 7574: trajectory change without reciprocal owner suppression
image 15254: reciprocal compound-region or semantic complement
```

The best current model is a mixture of state-local serialization rules rather
than one proven universal commit algorithm.

## Most Informative Next Discriminator

If this line continues, the next native-rollout unit should use one exact
parent with **three support-admitted, spatially non-overlapping instances of
the same category** and fully resolved physical ownership. It should estimate
the immediate transition matrix:

```text
T[i,j] = probability(next physical owner is j | first owner is i)
```

Every discovered exact row variant should be replayed under paired seeds, but
the horizon need not exceed one next row. Three same-category owners remove
the two strongest confounds at once:

- a third owner eliminates trivial binary complementation; and
- one shared description eliminates cross-category lexical complementation.

The predicted signatures are:

| Mechanism | Expected transition signature |
|---|---|
| physical-instance commit | the emitted owner's diagonal is selectively suppressed relative to counterfactual sibling-first histories, with released mass reaching at least one uncovered owner |
| order freedom | earlier-ranked uncovered support remains recoverable; this is tested separately from self-suppression |
| non-degenerate uncovered support | more than one uncovered owner remains reachable across paired samples; this is a support gate, not part of the commit definition |
| geometry-sorted frontier | mass moves primarily to later-ranked instances and rarely returns backward |
| category anti-repetition | cannot selectively distinguish which same-category instance was emitted |
| generic row progression | rows change concentration or termination similarly without owner-specific off-diagonal structure |

Failure to increase both off-diagonal owners would not by itself reject
physical-instance commit. It would reject only the stronger claim that one
commit update preserves broad, order-free access to every remaining instance.

A fully resolved dense-person state is the strongest candidate. The image
`2299` state cannot be reused unchanged because its parent ownership is
unresolved.

## Supported

- Multiple complete natural rows can recur from one exact parent prefix.
- Complete native rows can causally alter the immediate successor distribution.
- One overlapping bowl-carrot state has strict reciprocal successor switching
  that is stable across all exact row variants and full-model `float32` replay.
- Image `12576` has a highly stable forward-only transition compatible with a
  geometry-sorted traversal frontier.
- Owner ambiguity is conclusion-changing in dense same-category scenes and
  must remain a refusal condition.

## Not Supported

- No safe greedy branch-value mismatch is identified.
- No general physical-object, order-free covered-set mechanism is established.
- No stable commit mechanism is replicated across the required three states
  and two images.
- No influence distance beyond the immediate successor is identified.
- No claim is made about the internal hidden state, attention path, residual
  direction, or visual-versus-language location of the successor state.
- No terminal suppression, repetition-penalty change, decoder constraint,
  loss, reinforcement learning, explicit ledger, object slot, selector,
  architecture, or training screen is promoted.

## Execution Notes and Verification

The implementation added only experiment-local discovery, replay, admission,
and analysis surfaces. The combined focused test suite passed `51/51`; Python
compilation and repository diff checks pass.

Two incorrect launch attempts are excluded from every estimate:

- the first image-`15254` discovery launch accidentally included greedy on all
  eight shards and was rejected during merge; and
- the first image-`15254` suffix launch attempted too many concurrent model
  processes and was terminated, then quarantined under
  `wave2-image15254-admitted-exact-variants-k8-invalid-concurrency-20260717/`.

The canonical replacement discovery root ends in
`wave1-image15254-row0-discovery-k32-v2/`; the canonical `bfloat16` suffix root
ends in `wave2-image15254-admitted-exact-variants-k8/`; and the canonical
robustness root ends in
`wave3-image15254-admitted-exact-variants-k8-fp32/`.

The unit closes here. A new three-owner same-category transition matrix would
be a separate research unit, not an unbounded extension of this one.
