## Why

The current stack has strong coord-distribution supervision, decoded-box geometry
auxiliaries, and Stage-2 duplicate suppression, but it still lacks one specific
anti-collapse signal:

> when the model is about to localize object `j`, it is not explicitly told to
> avoid rerolling the immediately previous object's box.

That missing signal matters in two different regimes:

- in Stage-1 GT teacher forcing, the model can still learn a weak adjacent-copy
  prior where the next instance reuses the previous object's edges too closely
  even when GT keeps the boxes distinct,
- in Stage-2 rollout-context teacher forcing, local duplicate bursts often look
  like "the next object snaps back to the previous object's edges" rather than
  only exact copied boxes.

The desired first change is intentionally narrower than a global pairwise
repulsion field:

- only compare each object to the immediately previous object in sequence order,
- assume objects are pre-sorted within each sample and define adjacency over
  that canonical per-sample order,
- make the penalty distributional in coord-bin space rather than point-estimate
  only,
- define the forbidden region as edge-only decaying bands around the previous
  box edges,
- scale those bands with the previous object's bbox size,
- support a `same_desc` filter and a `global` mode that drops that filter,
- reuse current GT vs rollout context wording carefully,
- and keep decoded-box CIoU-style repulsion as a follow-up rather than the first
  normative loss definition.

This change should therefore be organized under the repo's existing loss-family
split:

- `coord_distribution`:
  - hard coord CE
  - soft coord CE
  - coord W1
  - coord/text gates
  - new `adjacent_repulsion` anti-copy atom
- `decoded_box_auxiliaries`:
  - `bbox_geo`
  - `bbox_size_aux`

The new atom belongs to the first family, not the second.

Public naming in this change:

- the shared loss-family atom is `adjacent_repulsion`
- the canonical Stage-2 rollout provenance key is `loss/B_coord/adjacent_repulsion`

## What Changes

- Introduce a new shared capability:
  - `adjacent-distributional-repulsion-loss`
- Define a new coord-side adjacent anti-copy loss term whose primary v1 form is:
  - computed from the current object's coord-bin distributions,
  - measured against four edge-only decaying bands induced by the immediately
    previous object's target box,
  - width-scaled by the previous object's bbox size on each axis,
  - applied only when an adjacent-object predicate passes.
- Define the adjacent-object predicate as:
  - previous object in current teacher-forced object order,
  - `filter_mode: same_desc | global`,
  - `same_desc` means the current object and immediately previous object share
    the same normalized description,
  - `global` means the description filter is dropped.
- Keep the primary v1 loss distributional:
  - it measures overlap between the current coord-bin distribution and a
    size-aware decaying edge band around each previous edge,
  - it combines the four slot overlaps into one box-copy score,
  - it penalizes only when that box-copy score exceeds a configured copy
    margin,
  - it does not define decoded-box CIoU-style repulsion as the primary
    requirement in v1.
- Reuse one shared implementation concept across contexts:
  - Stage-1 `gt` context through the Stage-1 coord auxiliary surface,
  - Stage-2 `rollout` context through the rollout coord-regularization surface,
  - this change does not add a Stage-2 GT-path adjacent-repulsion emission.
- Stage-1 config surface:
  - extend `custom.coord_soft_ce_w1` with adjacent-repulsion knobs and tighten
    nested-key validation for those knobs.
- Stage-2 config surface:
  - extend `coord_reg.config` with adjacent-repulsion knobs for
    `stage2_ab.pipeline` and rollout-matching pipeline users.
- Keep the Stage-2 objective pipeline shape stable:
  - do not introduce a standalone new bbox module in v1,
  - instead add the adjacent-repulsion term as a new optional coord-side
    sub-term under the existing `coord_reg` family.
- Add canonical logging for:
  - adjacent-repulsion loss contribution,
  - adjacent-pair count / applied count diagnostics,
  - adjacent box-copy score diagnostics.

## Recommended First Version

The recommended v1 is:

- one new shared capability centered on coord-bin distributions,
- Stage-1 support through `custom.coord_soft_ce_w1`,
- Stage-2 support through `coord_reg.config`,
- Stage-2 rollout-context only for Stage-2 provenance,
- immediate-previous-object adjacency only,
- edge-only decaying bands rather than a full 2D penalty field,
- `same_desc` as the default filter mode,
- `global` as an explicit opt-in ablation,
- decoded-box CIoU-style adjacent repulsion deferred to a later follow-up.

The recommended v1 loss form is:

- for each current bbox object with a valid immediately previous object,
- for each coord slot `{x1, y1, x2, y2}`,
- build a decaying edge band centered on the previous object's matching edge,
- set the band half-width as a ratio of the previous object's width or height,
- compute the current slot distribution's expected overlap with that band,
- combine the four slot overlaps into one box-copy score,
- penalize only when that box-copy score exceeds a configured copy margin.

Why this version:

- it matches CoordExp's current distributional training style,
- it backpropagates through the full coord distribution rather than through a
  hard decoded point only,
- it targets rerollout of the same box more directly than a generic
  "push-away" prior,
- it avoids over-killing valid next instances that happen to share one or more
  edges with the previous object,
- it can be realized in both `gt` and `rollout` contexts,
- it keeps Stage-2 provenance unambiguous by scoping Stage-2 emission to rollout
  context only,
- and it avoids forcing the first version to solve full decoded-box geometry
  repulsion and shared-pipeline order changes at the same time.

## Assumptions

- Adjacent-local box copying is a meaningful failure mode even without a full
  pairwise global repulsion field.
- The immediately previous object is the right first causal reference point for
  this feature.
- An edge-band box-copy score is a safer first contract than a decoded
  CIoU-only repulsion term.
- Stage-1 can support this as a GT-context anti-collapse regularizer even though
  it does not reproduce rollout-state contamination.
- Stage-2 rollout-context wording should remain aligned with the active
  clean-prefix `rollout` contract and should not revive deprecated
  `self_context` terminology.

## Non-Blocking Follow-Ups

- Add an optional decoded-box CIoU-style adjacent repulsion companion once the
  distributional term is validated.
- Generalize beyond immediate-previous-object adjacency only if the v1 signal is
  useful and stable.
- Add richer locality gates beyond `same_desc | global` if dense-scene recall
  requires them.
- Consider rollout-aligned YAML examples once the Stage-2 AB path is validated.

## Risks To Validity

- `global` mode may suppress legitimate dense-scene adjacency too aggressively.
- `same_desc` mode depends on reliable object-local description association in
  Stage-1 and Stage-2 target builders.
- If the edge bands are too wide, the term may over-penalize valid next
  instances that share some geometry with the previous object.
- If the edge bands are too narrow, the term may only catch near-exact copies.
- If the box-copy margin is too low, the term may still over-kill partial-edge
  matches.
- If the box-copy margin is too high, the term may not materially change
  collapse behavior.

## Required Evidence

- Evidence that the new term reduces adjacent duplicate-like localization
  collapse without harming crowded-scene recall.
- Evidence that `same_desc` and `global` differ meaningfully enough to justify a
  first-class toggle.
- Evidence that Stage-1 GT-context usage behaves as an anti-collapse prior
  rather than a destructive dense-scene bias.
- Evidence that rollout-context usage reduces local burst behavior on the known
  hard cases.
- Evidence that a whole-box copy margin prevents over-penalizing valid next
  instances that share only part of the previous box geometry.
- Evidence that the distributional-only v1 is worth keeping before adding a
  decoded-box CIoU companion.
- Evidence from at least one focused smoke or synthetic acceptance check, not
  only unit tests.

## Capabilities

### New Capabilities

- `adjacent-distributional-repulsion-loss`: a shared immediate-previous-object
  coord-distribution anti-copy contract that can run in Stage-1 `gt` context and
  Stage-2 `rollout` context.

### Modified Capabilities

- `coord-aux-loss`: extend the Stage-1 coord auxiliary config and logging
  contract so GT-context adjacent repulsion is a supported opt-in sub-term.
- `teacher-forcing-unified-loss-registry`: add canonical semantics for the new
  adjacent-repulsion coord-side term across `gt` and `rollout` contexts.
- `stage2-ab-training`: extend the Stage-2 AB config and rollout-context
  semantics so `coord_reg` may carry adjacent repulsion in rollout context.
- `rollout-matching-sft`: extend the rollout-matching Stage-2 pipeline contract
  so the same `coord_reg` adjacent-repulsion knobs are available in rollout
  context.
- `trainer-metrics-components`: document the canonical adjacent-repulsion loss
  atoms and diagnostics keys.

## Impact

- Immediate impact is proposal/design/spec only.
- The expected implementation surface is likely centered on:
  - `src/trainers/losses/coord_soft_ce_w1.py`
  - `src/trainers/teacher_forcing/modules/coord_reg.py`
  - `src/trainers/teacher_forcing/stage1.py`
  - `src/trainers/stage2_two_channel/target_builder.py`
  - `src/trainers/teacher_forcing/module_registry.py`
  - `src/config/schema.py`
  - `src/config/rollout_matching_schema.py`
- The first implementation is intentionally not a standalone bbox module and
  intentionally does not redefine decoded-box geometry loss ownership.
