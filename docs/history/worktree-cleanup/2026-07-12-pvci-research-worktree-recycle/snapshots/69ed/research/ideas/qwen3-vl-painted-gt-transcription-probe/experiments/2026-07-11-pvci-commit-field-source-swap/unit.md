---
title: PVCI Commit-Field Annotation-Empty Source Swap
description: Tests whether an annotation-empty written address is sufficient to reproduce the native coordinate-local commit field.
type: idea
role: research-unit
authority: non_normative_research
promotion_status: not_promoted
unit_id: 2026-07-11-pvci-commit-field-source-swap
topic: qwen3-vl-painted-gt-transcription-probe
status: complete
tags:
  - coordexp-swift
  - research-unit
  - qwen3-vl
  - commit-field
  - source-swap
  - objecthood
updated: 2026-07-11
---

# PVCI Commit-Field Objecthood Source Swap

## Question

Can the language decoder write the completed coordinate-local
occupancy/anti-repetition field at a spatially matched annotation-empty address,
or is this one-sided sufficiency test inconclusive?

This unit changes no architecture and trains no model. It holds the image,
canonical PRE prefix, class phrase, compact-row syntax, checkpoint, and scoring
panel fixed while swapping only the geometry written into the next row.

## Decision Relevance

- If a spatially matched annotation-empty write creates the same local
  suppression as an annotated-object write, recorded object support is not
  required for the native commit primitive. Treat it as a useful
  coordinate-addressed occupancy field, not an object ledger.
- If real J/K writes create stable local suppression but the selected
  annotation-empty write does not, this particular control failed. That
  falsifies the strongest any-coordinate sufficiency claim but cannot identify
  visual-object necessity; the next decider must hold coordinates fixed while
  changing visual content.
- If both effects exist but annotated-object addresses are consistently
  stronger, report an address advantage, not objecthood amplification.
- None of these outcomes selects a final architecture. They decide whether
  later work should preserve a native spatial field, seek a visual binding
  signal, or justify an explicit state.

## Frozen Source Panel

Source:

`/data/CoordExp/outputs/painted_gt/pvci_endogenous_commit_write_read/plan_step4887/write_read_attempts.jsonl`

Retain all ten strictly attributable source attempts from the completed bridge
unit in the top-level ledger. The source contains eight unique `(image, J)`
clusters across four images.

The balanced source-swap factorial is feasible for eight attempts, seven unique
clusters, and all four images. The two infeasible attempts are the repeated
large-J cluster:

```text
line-000009-row-000-C1
line-000009-row-000-C2
```

Their J box is approximately `173 x 491`; exhaustive deterministic placement
finds no same-size box with zero intersection against all annotations. Preserve
both as `annotation_empty_infeasible`. Do not relax them or silently change the
balanced denominator.

## Frozen Geometry Construction

For every source attempt:

### J and J-near

- `J_true` is the frozen canonical row for the strict-matched J.
- `J_near` is the existing deterministic perturbation, with zero shared
  coordinate IDs and IoU approximately `0.79-0.84` to J.

### Same-class K

Choose K only from frozen same-description objects still uncovered at PRE.
Select the object with maximum normalized center distance from J; break ties by
minimum GT index. K must remain distinct from J and annotated-uncovered.
Construct deterministic `K_near` with zero shared coordinate IDs and IoU target
`0.80` in `[0.65, 0.95]`.

### Annotation-empty E

E is not called verified background. It is a size/aspect-exact, spatially near
region with no intersection against any frozen source annotation; COCO may omit
real objects.

Let E have exactly J's pixel width and height. Enumerate top-left positions on
an 8-pixel grid, including terminal right/bottom positions, with border inset:

```text
max(8, ceil(0.01 * min(image_width, image_height)))
```

Require strict zero pixel-area intersection with every source annotation. On
the uint8 grayscale crop compute pixel variance and edge density, where edge
density is the mean fraction of horizontal/vertical absolute differences above
20. To reduce absolute-location/raster-prior confounding, select the
lexicographic minimum:

```text
(normalized_center_distance_from_J, variance, edge_density, y1, x1)
```

Construct `E_near` with zero shared coordinate IDs, IoU target `0.80` in
`[0.65, 0.95]`, valid bounds, and continued zero intersection with every source
annotation. Freeze every eligible-count and pixel-stat receipt before model
loading.

## Prefix Factorial

For each balanced attempt, use the same canonical PRE and same class phrase:

```text
PRE
WRITE_J_TRUE = PRE + phrase(J) + geometry(J)
WRITE_J_NEAR = PRE + phrase(J) + geometry(J_near)
WRITE_K_TRUE = PRE + phrase(J) + geometry(K)
WRITE_EMPTY  = PRE + phrase(J) + geometry(E)
```

Because K is same-class, `phrase(J) == phrase(K)`. All branches must have
identical structure/phrase token IDs and differ only in coordinate token IDs.

At PRE and every write branch, score the identical candidate universe:

- `J_exact`, `J_near`;
- `K_exact`, `K_near`;
- `E_exact`, `E_near`;
- all other still-uncovered same-description canonical rows;
- STOP as a separate one-token diagnostic.

Generate one additional RP1.0 row from each write branch as descriptive side
evidence. For J/J-near, move J to covered; for K, move K to covered; for E, add
no annotated object to covered. Independently record `repeat_written_address`
when the next generated box has IoU>=0.5 with the geometry written by that
branch, regardless of phrase or annotated-object attribution. Unknown is never
successful avoidance.

## Primary Contrasts

For a written location `W in {J, K, E}` and its zero-token-overlap near
candidate `W_near`:

```text
Local_W = log P(W_near | WRITE_W) - log P(W_near | PRE)

Address_W =
  Local_W
  - median(
      Delta_WRITE_W(other two source-swap near candidates)
    )
```

The address control set is exact and deduplicated:

```text
C_J = {K_near, E_near}
C_K = {J_near, E_near}
C_E = {J_near, K_near}
```

`W_exact`, `W_near`, J/K exact rows, and the additional same-description rows
cannot enter `C_W`. Additional same-description candidates remain descriptive
only. Candidate IDs must be unique, and the planner freezes every `C_W`.

`Local_W < 0` shows a non-literal local suppression field. `Address_W < 0`
shows that the field is more suppressive at the written location than at the
other same-class/source-swap locations.

The primary annotated-address advantage is size-matched between J and E:

```text
AddressAdv_J = Address_J - Address_E
```

Negative values mean the annotated J address creates a more localized field
than the annotation-empty address. This is not proof of objecthood: visual
content, coordinate prior, baseline likelihood, and local context still differ.
Report `Local_J - Local_E` descriptively. K is a second annotated address
control, but its size is not forced to match J/E, so its magnitude is not used
in the primary advantage contrast.

`WRITE_J_NEAR` is secondary: score whether displaced written geometry
suppresses `J_exact` and whether its effect remains localized around the J
neighborhood. It does not change the primary label.

Raw direct-forward scores are primary. Native model/generation remains
bfloat16; logit readout/log-softmax is float32; stable aggregation is Python
double. RP1.1 is secondary only.

## Frozen Gates

All primary counts use the eight balanced attempts. Repeated attempts are
collapsed by median within seven unique `(image, J)` clusters, then by median
within four images. A terminal primary direction requires at least `6/8`
attempts, `5/7` clusters, and `3/4` images.

Define:

```text
AddressRead_W = [Local_W < 0] AND [Address_W < 0]

Occupancy_i =
  AddressRead_J(i) AND AddressRead_K(i) AND AddressRead_E(i)
```

Within each `(image, J)` cluster, separately median-aggregate `Local_W` and
`Address_W`, then define `ClusterAddressRead_W` from their signs and
`ClusterOccupancy` as the conjunction across J/K/E. Within each image, median
the cluster-level Local and Address values separately, define
`ImageAddressRead_W`, then `ImageOccupancy` as their conjunction. For the
modifier, `ClusterAddressAdv_J` is the median of attempt-level `AddressAdv_J`
within a cluster, and `ImageAddressAdv_J` is the median of cluster-level
advantages within an image.

### H1: Generic coordinate occupancy is sufficient

- `Occupancy_i` passes `6/8` attempts;
- `ClusterOccupancy` passes `5/7` clusters and `ImageOccupancy` passes `3/4`
  images;
- exact source-row repetition is not used as evidence.

Primary label: `annotation_empty_coordinate_occupancy_supported`.

### Otherwise: annotation-empty control is inconclusive

Any failure of H1 returns `annotation_empty_control_inconclusive`. This does not
establish real-object support. Report the J/K/E marginal and paired failure
patterns, then route to a same-coordinate visual-content intervention.

### Independent annotated-address modifier

- `annotated_J_address_advantage` if `AddressAdv_J < 0` in `6/8` attempts,
  `ClusterAddressAdv_J < 0` in `5/7` clusters, and `ImageAddressAdv_J < 0` in
  `3/4` images;
- `no_consistent_annotated_address_advantage` if the reverse/neutral direction holds
  in at least `6/8`, `5/7`, and `3/4`;
- otherwise `annotated_address_advantage_inconclusive`.

The independent modifier is always reported but cannot promote an objecthood
claim.

## Evidence Gates

- Materialize all ten source attempts and preserve two E-infeasible attempts.
- Hash source plan/events, completed bridge artifacts, unit, planner, scorer,
  analyzer, geometry helpers, image bytes, config, live checkpoint, Git state,
  and argv.
- Freeze J/K/E choices and all pixel/geometry receipts before model loading.
- Materialize an E/E-near crop sheet and freeze a descriptive visual-audit
  receipt for obvious foreground content; the audit never upgrades E to
  verified background and does not change the frozen denominator.
- Fail if any K is covered, non-same-class, or absent from the frozen source
  controls.
- Fail if E or E-near intersects any frozen annotation by positive area.
- Require exact phrase/structure token equality across all branches.
- Require the identical candidate universe and exact designated-candidate
  determinism at PRE and all four write branches.
- Require source-equivalent live checkpoint/config/runtime identity.
- Two-attempt GPU gate first; the full ten-attempt ledger runs only after GO.
- Partial runs and any failed balanced attempt are ineligible for terminal
  labels.

## Stop Condition

Stop after one passing two-attempt runtime gate and the exact frozen ten-source
ledger, or immediately on contract failure. Return one primary label and one
annotated-address modifier. No architecture is promoted.

## Planned Artifacts

- plan: `/data/CoordExp/outputs/painted_gt/pvci_commit_field_source_swap/plan_step4887/`;
- scorer: `/data/CoordExp/outputs/painted_gt/pvci_commit_field_source_swap/full10/`;
- analysis: scorer root plus `/analysis/`.

## Research Unit Closeout

Observed: The two-attempt runtime gate and exact frozen ten-source ledger both
completed with zero scorer or analyzer failures. Eight attempts were balanced;
the two predeclared annotation-empty-infeasible rows remained ineligible rather
than being relaxed. Raw and RP1.1 aggregation agree:

```text
joint Occupancy: attempts 0/8, clusters 0/7, images 0/4
AddressRead_J: 7/8
AddressRead_K: 0/8
AddressRead_E: 1/8
AddressAdv_J < 0: attempts 7/8, clusters 6/7, images 4/4
```

The primary label is `annotation_empty_control_inconclusive`; the independent
modifier is `annotated_J_address_advantage`. Across 32 descriptive one-row
continuations, the token-decoded written address repeated `0/32`; 24/32 rows
matched a unique annotated-uncovered object and 8/32 were unknown. The four
write branches had identical `6/8` strict and `2/8` unknown continuation
counts, so continuation behavior does not explain the direct-forward field.

Supported: generic "write any coordinates and suppress their neighborhood"
is falsified for this frozen panel. The source free-continuation-selected J
write produces a localized non-literal suppression field substantially more
consistently than either an arbitrary same-class annotated K write or the
selected annotation-empty same-size E write. Real annotated support alone is
also insufficient because K fails `AddressRead` in `0/8`. The strongest
remaining working hypothesis is a trajectory/current-self-selected-object
gate: canonical PRE establishes an active object trajectory, and writing a row
coherent with that trajectory completes a native local redistribution.

Not supported: annotation-empty failure does not establish visual-object
necessity; E is annotation-empty rather than verified background. The result
does not establish a semantic identity ledger, durable coverage, autonomous
selection, stopping, long-horizon composition, or an architecture. Baseline
candidate likelihood and current-object status remain entangled.

Authoritative artifacts:

- scorer: `/data/CoordExp/outputs/painted_gt/pvci_commit_field_source_swap/full10_v1/`;
- analysis: `/data/CoordExp/outputs/painted_gt/pvci_commit_field_source_swap/full10_v1/analysis_v1/`;
- scorer events SHA-256: `1e369f75c23c5c24806af0b6e26cc08055dd5054f8e919dc1a8c338c5cfba599`;
- execution receipt SHA-256: `0ef5c487e29e5aab5441635c42efc3bb7ac664859ca48c35660ac235d6b97cb7`;
- scorer summary SHA-256: `833c63e7e1f97507d8f39d9918bb0c1597d333ebba9381c1f450ec8676502e5e`;
- analysis summary SHA-256: `e94bbdb53d4c1e227b67da7940b2f371dbd866863c00d31b8123fecba4e08535`;
- analysis events SHA-256: `2abfadf9f3f98824b232a3007984805eb1f3425e33c31efc7d8d7be709775956`.

Next decider: hold image, object J, phrase, coordinates, and candidate universe
fixed while placing the identical canonical J row after two canonical prefixes:
one where J remains uncovered but was not selected by the source free
continuation, and one where J was selected. This directly tests whether
current-object/trajectory status, rather than generic object or coordinate
writing, gates the read.

Promotion decision: `not_promoted`. The unit is complete and narrows the
native mechanism hypothesis; it does not justify an architecture change.
