## Context

The user wants one adjacent-object anti-collapse signal that can be discussed
coherently across:

- Stage-1 GT teacher forcing,
- and Stage-2 rollout-context teacher forcing.

The key design constraints are:

- adjacency means the immediately previous object in sequence order,
- the user will pre-sort sequence objects first, so adjacency is defined over
  canonical per-sample object order rather than incidental append order,
- `global` means dropping the `same_desc` filter,
- the signal should be distributional rather than point-estimate-first,
- the signal should prevent rerollout of the previous box more than it enforces
  generic separation,
- the penalty region should be edge-only and decay from the previous edge out to
  a size-aware boundary,
- and the wording must use the current `gt` / `rollout` context vocabulary
  rather than deprecated `self_context`.

The current repo already has the right architectural precedent:

- Stage-1 coord auxiliary supervision is distributional in
  `coord_soft_ce_w1`,
- Stage-2 rollout coord-side supervision already flows through `coord_reg`,
- decoded-box geometry auxiliaries already exist when a box-level companion is
  desired later.

That suggests a cleaner first design than "new bbox module":

> adjacent repulsion should be a new coord-side sub-term, shared across
> contexts, with Stage-1 using a thin GT-context adapter and Stage-2 using the
> existing rollout-context coord-reg path.

For implementation and documentation, this change should be grouped under the
existing coord-distribution family rather than the decoded-box auxiliary family:

- coord-distribution family:
  - coord CE
  - soft coord CE
  - coord W1
  - coord/text gates
  - adjacent repulsion
- decoded-box auxiliary family:
  - bbox geometry
  - bbox size auxiliary

## Goals / Non-Goals

**Goals**

- Define one shared adjacent-repulsion capability across `gt` and `rollout`
  contexts.
- Keep the primary v1 formulation distributional in coord-bin space.
- Make the primary v1 signal anti-copy rather than generic anti-proximity.
- Scope adjacency to the immediately previous object only.
- Support `same_desc` and `global` filter modes.
- Make Stage-1 wording explicitly GT-context.
- Make Stage-2 wording explicitly rollout-context.
- Keep the Stage-2 pipeline surface stable by making this a `coord_reg`
  sub-term rather than a standalone bbox module.

**Non-Goals**

- No deprecated `self_context` terminology.
- No full global all-pairs repulsion field in v1.
- No full 2D image-pixel penalty field in v1.
- No decoded-box CIoU-style adjacent repulsion as the primary normative v1
  contract.
- No requirement that repo-authored canonical Stage-2 profiles enable this by
  default in the first version.

## Decisions

### 1) The primary v1 signal is a size-aware edge-band box-copy score

For the current object `j` and its immediately previous object `i = j - 1`, the
loss is defined in coord-bin space.

For each coord slot `s in {x1, y1, x2, y2}`:

- let `p_{j,s}(k)` be the model distribution over coord bin `k`,
- let `b_{i,s}` be the previous object's target coord bin for that slot,
- let `w_i = max(1, x2_i - x1_i)` and `h_i = max(1, y2_i - y1_i)` in coord-bin
  units,
- let `m_x(i) = max(1, round(r * w_i))`,
- let `m_y(i) = max(1, round(r * h_i))`,
- where `r` is the configured `adjacent_repulsion_margin_ratio`, interpreted as
  a per-edge half-width ratio rather than a whole-box scale factor.

Define one edge-only decaying band per slot:

- for `x1, x2`, use half-width `m_x(i)`,
- for `y1, y2`, use half-width `m_y(i)`.

The v1 per-slot edge-band mask is linear and bounded:

- `q_{i,s}(k) = max(0, 1 - |k - b_{i,s}| / m_s(i))`

Properties:

- `q_{i,s}(b_{i,s}) = 1`,
- `q_{i,s}(k) = 0` at and beyond the band boundary,
- the band is symmetric around the previous edge coordinate,
- the band gets wider as the previous box gets larger on that axis.

This makes the public meaning of `adjacent_repulsion_margin_ratio` explicit:

- `0.05` means each edge owns a band that extends `5%` of the previous width or
  height away from that edge on its axis,
- it is not shorthand for expanding the whole box by `1.05` around its center.

The v1 per-slot soft overlap is:

- `overlap_{j,s} = sum_k p_{j,s}(k) * q_{i,s}(k)`

The v1 box-copy score is the geometric mean across the four coord slots:

- `copy_j = (overlap_{j,x1} * overlap_{j,y1} * overlap_{j,x2} * overlap_{j,y2})^(1/4)`

The v1 adjacent penalty is a thresholded hinge:

- `L_j = max(0, copy_j - tau)^2`

where `tau` is the configured `adjacent_repulsion_copy_margin`.

The batch-level adjacent repulsion term is the mean of `L_j` over contributing
objects.

Why this form:

- it is distributional and differentiable,
- it directly models "copy the previous box edges again" rather than generic
  local proximity,
- it scales with the previous object's size,
- the geometric mean prevents one or two matching edges from over-triggering the
  penalty when the full box is clearly different,
- and it matches the repo's current coord-distribution training style.

### 2) Edge-only bands are better than a full 2D region for the current repo

The user suggested a conceptually appealing full bbox margin region. For the
current architecture, the better first realization is edge-only bands.

Why:

- the model directly predicts `x1, y1, x2, y2` token distributions rather than
  a dense occupancy map,
- edge bands map cleanly onto the current slot-wise coord loss machinery,
- a full 2D region would require a heavier soft box occupancy construction that
  is not needed to validate the idea,
- edge-only bands are less likely to over-kill valid next instances that share
  only partial geometry with the previous object.

So v1 chooses edge-only bands, not a full annulus between expanded and shrunk
boxes.

### 3) The edge-band mask is a bounded taper, not a normalized probability target

The edge-band mask `q_{i,s}` must act like a penalty region, not like a target
distribution.

Why this is important:

- the user wants the penalty to decay from the previous edge to the band
  boundary and then become exactly zero elsewhere,
- if the mask were normalized into a probability distribution, larger bands
  would spread the same mass more widely and weaken exact-edge copying in an
  undesirable way,
- a bounded mask with peak value `1` preserves the interpretation of
  `overlap_{j,s}` as "how much probability mass is still copy-like for this
  edge".

### 4) Adjacency is defined on object order, not on pairwise neighborhood search

V1 adjacency means exactly one reference:

- the immediately previous object in canonical teacher-forced object order
  within the same sample.

No earlier object window, no all-pairs field, and no cross-sample adjacency are
part of the v1 contract.

Why:

- it matches the user's requested simplification,
- it is the most causal local reference in an autoregressive object stream,
- and it avoids turning the first version into an expensive or overly broad
  pairwise system.

### 5) The filter toggle is `same_desc | global`

The gating modes are:

- `same_desc`:
  - apply the adjacent repulsion term only when the current object and the
    immediately previous object share the same normalized description,
- `global`:
  - drop the description filter and apply the same adjacent repulsion test to
    any immediately previous object.

Recommended default:

- `same_desc`

Why:

- it is the lower-risk mode for crowded scenes,
- it directly matches the user's primary anti-duplication purpose,
- while `global` remains a useful ablation for a stronger anti-copy prior.

### 6) Stage-1 semantics are GT-context and must use explicit object grouping

In Stage-1:

- the current object order comes from the GT teacher-forced target,
- the previous object reference also comes from that GT order,
- the previous-object target bins come from GT target bins,
- the current object distributions come from the model's teacher-forced coord
  logits.

This is a GT-context anti-collapse regularizer, not a sampled-prefix burst
mechanism.

Implementation consequence:

- Stage-1 must be able to recover per-sample boundaries, per-object order,
  per-object edge bins, and object-local description identity deterministically
  from the teacher-forced target,
- for packed rows, Stage-1 must reset adjacency at successive top-level
  CoordJSON container boundaries rather than treating the whole packed row as
  one continuous object stream,
- if `same_desc` mode is enabled and the runtime cannot map bbox quartets to
  object-local descriptions unambiguously, the feature must fail fast rather than
  guess.

### 7) Stage-2 semantics are rollout-context and must use the edited clean target order

In Stage-2 rollout context:

- the current object order is the canonical pre-sorted edited clean
  teacher-forced target order for the active rollout-context sample,
- the immediately previous object is the previous object in that same edited
  target order,
- the previous-object target box comes from the current rollout-context target
  carrier,
- the current object distributions come from rollout-context coord logits.

Recommended v1 application rule:

- apply adjacent repulsion only to rollout-context objects that already
  participate in positive coord-side supervision,
- allow the immediately previous object reference to come from the active target
  order even if that previous object itself is context-retained,
- do not infer adjacency from `bbox_groups_prefix` append order or other
  supervision-source assembly order.

This keeps the loss aligned with the current clean-prefix rollout contract.

### 8) Runtime ownership is split across target building, grouped carrier preservation, and coord-side computation

The implementation contract should be explicit:

- Stage-1 adapter:
  - recovers canonical per-sample object order and desc association from the
    pre-sorted GT sequence,
- Stage-2 `target_builder`:
  - computes canonical clean-order adjacency inputs for rollout context,
  - computes or preserves the gate inputs needed for `same_desc | global`,
- `bbox_geo`:
  - preserves grouped object carriers instead of forcing downstream
    reconstruction from flat slot order,
- `coord_reg`:
  - computes edge-band overlaps, `copy_score`, hinge loss, and raw adjacent
    diagnostics from grouped carriers,
- Stage-2 objective logging:
  - projects rollout-context adjacent loss to canonical `loss/B_coord/*`
    metrics.

This prevents `coord_reg` from guessing desc identity or clean-order adjacency
after flattening.

### 9) The first Stage-2 surface should be a `coord_reg` extension, not a new standalone module

The cleanest first architecture is:

- Stage-1:
  - extend `custom.coord_soft_ce_w1` with adjacent-repulsion knobs,
- Stage-2:
  - extend `coord_reg.config` with the same adjacent-repulsion knobs.

Recommended first keys:

- `adjacent_repulsion_weight`
- `adjacent_repulsion_filter_mode`
- `adjacent_repulsion_margin_ratio`
- `adjacent_repulsion_copy_margin`

Recommended defaults:

- `adjacent_repulsion_weight: 0.0`
- `adjacent_repulsion_filter_mode: same_desc`
- `adjacent_repulsion_margin_ratio: 0.05`
- `adjacent_repulsion_copy_margin: 0.8`

Why this is preferred:

- it keeps the loss inside the coord-distribution family,
- it avoids introducing a new standalone bbox module,
- it avoids unnecessary canonical Stage-2 pipeline-order churn,
- and it supports one shared conceptual implementation.

### 10) Config strictness must be explicit for Stage-1 and Stage-2

Stage-2 already has strict allowlist-style config ownership for `coord_reg`.
Stage-1 does not yet have the same strict nested-key behavior for
`custom.coord_soft_ce_w1`.

So this change requires:

- Stage-2:
  - extend existing strict config ownership for the new adjacent-repulsion keys,
- Stage-1:
  - tighten `coord_soft_ce_w1` parser validation so unsupported adjacent
    repulsion keys fail fast rather than being silently ignored.

### 11) Decoded-box CIoU-style adjacent repulsion is deferred

A decoded-box companion remains interesting:

- decode the current object's soft box,
- compare it to the immediately previous object's target box,
- penalize excessive adjacent overlap in box space.

But this is not the v1 normative contract.

Why it is deferred:

- the user explicitly wants the primary idea to be distributional,
- edge-band anti-copy is the cleaner first mapping of that idea,
- and the first slice should validate whether the coord-distributional signal is
  already enough.

## Open Questions To Resolve During Implementation

- Should Stage-2 rollout-context adjacency include FN tail objects in the same
  immediate-order rule, or should v1 stop at the retained rollout prefix?
- Is the minimal Stage-2 carrier better expressed as:
  - grouped `clean_order_index + desc_key`, or
  - compact `prev_gt_bins + same_desc_flag + apply_mask`?

## Verification Strategy

- Unit-test the per-slot edge-band overlap and confirm it increases when current
  coord mass moves toward the previous object's edge bins.
- Unit-test size scaling and confirm the same relative reuse pattern produces a
  wider forbidden region for a larger previous box.
- Unit-test the whole-box copy margin and confirm a box with only partial edge
  agreement stays below the margin.
- Unit-test Stage-1 sample-boundary preservation and confirm adjacency never
  crosses examples within a batch.
- Unit-test `same_desc` and `global` gating in both GT and rollout-context
  target builders.
- Verify Stage-1 fail-fast behavior when object-desc grouping is ambiguous in
  `same_desc` mode.
- Verify Stage-2 rollout-context uses canonical clean-order adjacency rather
  than supervision append order.
- Verify Stage-2 can reference a previous clean-order object that is
  context-retained even if not positively bbox-supervised.
- Add at least one focused smoke or synthetic acceptance check showing a
  partial-edge overlap stays below `copy_margin`.
- Keep decoded-box CIoU companion out of v1 success criteria.
