# Clean Pre-X1 Ownership Transition Findings

Date: 2026-06-23

Scope: post-hoc reduction over existing model-backed clean pre-x1 compatibility
patch artifacts, followed by a new bounded GPU component-output patch pass over
the selected ownership-tomography panel. The reducers summarize ownership
transitions across layers 17-21 and localize the layer-17/layer-18 component
surface so train and val failures can be compared without returning to the
original person/backpack anchor.

## Tooling

Added:

```text
src/analysis/autoregressive_binding_template_ablation/formation_ownership_transition_reducer.py
scripts/analysis/run_autoregressive_binding_formation_ownership_transition.py
tests/analysis/test_formation_ownership_transition_reducer.py
```

The reducer consumes `formation_replay_patch_compatibility_rows.jsonl` and
writes:

```text
formation_ownership_transition_group_rows.jsonl
formation_ownership_transition_sequence_rows.jsonl
formation_ownership_transition_summary.json
formation_ownership_transition_summary.md
```

Ownership classes are:

```text
target_geometry
target_rank
donor
origin
worse_or_escape
other
```

`target_geometry` is reserved for exact/near target coordinate-basin landing.
`target_rank` records rank-only target improvement. This distinction matters
because many apparent improvements are not whole-basin repair.

## Artifacts

Compatible donor layer scan:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_ownership_transition/v1_v24_clean_prex1_compatible_layers17_21
```

Source compatibility rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v24_v23_compatible_donors_layers17_21_compatibility/formation_replay_patch_compatibility_rows.jsonl
```

Simple-control origin layer scan:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_ownership_transition/v2_v28_clean_prex1_simple_control_layers17_21
```

Source compatibility rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_replay_patch_layer_scan/v28_v27_simple_control_origin_layers17_21_compatibility/formation_replay_patch_compatibility_rows.jsonl
```

## Main Result

The clean pre-x1 ownership decision is not centered on the original
person/backpack example. The compatible-donor artifact covers 52 receiver
states and 595 patch rows across train and val:

```text
train rows: 245
val rows: 350
regimes: crowded, duplicate_basin_nearby, repeated_class, small_object,
         termination_tail
```

Layer 17 is mixed and comparatively soft. By layer 18, compatible donor patches
snap mostly into donor ownership and stay there through layer 21.

Compatible donors, all rows:

```text
input rows: 595
receiver states: 52
sequence rows: 119
target_like_rate: 0.1395
target_geometry_rate: 0.0689
donor_like_rate: 0.6437
origin_like_rate: 0.0824
worse_like_rate: 0.2000
mean_coord_rank_delta: +122.88
```

Earliest ownership layer counts:

```text
target-like:     layer17=29, layer18=7, layer20=1, none=82
target-geometry: layer17=11, layer18=3, layer19=1, none=104
donor-like:      layer17=9,  layer18=78, layer19=11, layer20=4, layer21=1, none=16
origin-like:     layer17=38, layer18=1,  layer19=2, none=78
```

Final ownership at layer 21:

```text
donor: 87
target_geometry: 8
target_rank: 6
worse_or_escape: 17
origin: 1
```

The most common transition signatures show the same qualitative story:

```text
17:worse_or_escape -> 18:donor -> 19:donor -> 20:donor -> 21:donor : 31
17:origin          -> 18:donor -> 19:donor -> 20:donor -> 21:donor : 27
17:target_rank     -> 18:donor -> 19:donor -> 20:donor -> 21:donor : 10
17:target_geometry -> 18:target_geometry -> 19:target_geometry -> 20:target_geometry -> 21:target_geometry : 5
```

Interpretation: the late residual stream behaves like an ownership resolver,
not a generic visual-perception repair site. Compatible hidden states carry
strong coordinate-basin identity. If the donor and receiver are not already
geometrically compatible, patching tends to transfer donor ownership instead of
recovering the receiver target.

## Train Versus Val

The same layer-18 donor snap appears in both trained rows and held-out analogs.
This keeps trained-sequence failures central: a row being in the training
dataset does not guarantee a strong pre-x1 ownership state under a GT prefix.

Compatible donors by split:

```text
train layer17: target_like=0.2449, target_geometry=0.0816, donor_like=0.0816, origin_like=0.3265, mean_rank_delta=+38.31
train layer18: target_like=0.1633, target_geometry=0.1020, donor_like=0.7347, origin_like=0.0408, mean_rank_delta=+85.04
train layer21: target_like=0.1837, target_geometry=0.1224, donor_like=0.7755, origin_like=0.0000, mean_rank_delta=+116.22

val layer17: target_like=0.2429, target_geometry=0.1000, donor_like=0.0714, origin_like=0.3143, mean_rank_delta=+93.33
val layer18: target_like=0.1143, target_geometry=0.0429, donor_like=0.7286, origin_like=0.0286, mean_rank_delta=+115.64
val layer21: target_like=0.0714, target_geometry=0.0286, donor_like=0.8286, origin_like=0.0143, mean_rank_delta=+187.27
```

Train rows are somewhat easier, but not mechanistically different enough to
support a pure unseen-val explanation. The decisive failure family remains
pre-x1 ownership and coordinate-basin selection under local autoregressive
context.

## Simple-Control Caveat

The simple-control artifact covers 57 receiver states and 605 patch rows. It is
not a repair control. It is an origin attractor:

```text
target_like_rate: 0.0
target_geometry_rate: 0.0
ownership_class_counts: origin=601, donor=3, worse_or_escape=1
last ownership at layer21: origin=121 sequence rows
```

The `donor_like_rate` is near 1.0 in this artifact because the simple-control
donor is the origin coordinate. The reducer intentionally prioritizes
`origin` over `donor` in the mutually exclusive ownership class so this overlap
does not masquerade as donor repair.

## Consequences

1. The next causal row bank should broaden further over dataset rows before
   spending GPU budget, especially trained-sequence failures under GT prefix
   and held-out analogs matched by motif.
2. The original person/backpack case remains useful as a microscope, not as a
   sampling prior.
3. Generic donor-state import is now demoted. It mostly measures donor capture.
   The next coordinate-onset probes should isolate object identity, slot phase,
   and coordinate value instead of injecting an entire donor state.
4. The layer-17 to layer-18 transition is the current priority for attention
   and value localization. Stable target-geometry sequences should be compared
   against donor-snap and origin-collapse sequences at this transition.
5. Router/closure failures are a separate mechanism. Boundary-direction
   results remain promising, but should be expanded over crowded train/val rows
   without mixing them with pre-x1 coordinate-onset claims.

## Next Execution Slice

Use the broad per128 train/val candidate bank and the v7 failure/analog panel
as the next population base. Expand if needed, but keep row selection explicitly
population-first:

```text
train-sequence pre_x1 failures under GT prefix
val analog failures matched by motif and descriptor where possible
small-object extent/final-coordinate cases
crowded closure/router cases
nonrecovered tail cases
strict clean coord-close rows versus donor-far intrusive rows
```

Concrete next probes:

```text
1. Layer 17->18 ownership tomography:
   compare stable target_geometry, donor-snap, origin-collapse, and
   worse/escape sequences using attention/value and component-output reads.

2. Slot-phase disentanglement:
   split donor state into object identity, coordinate value, and slot phase.
   Prefer donor-minus-previous-slot or projected value probes over raw donor
   imports.

3. Train-first population expansion:
   mine additional training rows that fail under GT prefix, then match val
   analogs by regime, descriptor, area, object order, repetition, tail length,
   and post_x1 recovery/nonrecovery.

4. Router expansion:
   replicate the layer-24 boundary-direction threshold on additional crowded
   train/val closure failures outside the capped staged mechanism panel.

5. Coordinate locality/smoothness:
   for small-object and final-extent rows, measure local coordinate mass,
   rank smoothness, and neighboring-bin basin shape before interpreting them
   as weak visual perception.
```

## Tomography Launch Panel

Added a read-only selector for the next layer-17 to layer-18 tomography pass:

```text
src/analysis/autoregressive_binding_template_ablation/formation_ownership_tomography_panel.py
scripts/analysis/run_autoregressive_binding_formation_ownership_tomography_panel.py
tests/analysis/test_formation_ownership_tomography_panel.py
```

Artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_ownership_tomography_panel/v1_v24_v28_layers17_18_train_val_panel
```

Input sequence rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_ownership_transition/v1_v24_clean_prex1_compatible_layers17_21/formation_ownership_transition_sequence_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_ownership_transition/v2_v28_clean_prex1_simple_control_layers17_21/formation_ownership_transition_sequence_rows.jsonl
```

Selector scope:

```text
input transition sequences: 240
candidate sequences: 224
selected panel rows: 85
max_per_role_split_regime: 3
model_ran: false
training_ran: false
```

Selected role counts:

```text
layer18_donor_snap: 27
origin_collapse: 30
persistent_worse_escape: 14
rank_only_target_repair: 6
stable_target_geometry: 8
```

Selected split/regime coverage:

```text
train: 44
val: 41

crowded: 22
duplicate_basin_nearby: 14
repeated_class: 18
small_object: 13
termination_tail: 18
```

This is now the preferred launch handle for the next attention/value or
component-output tomography pass. It keeps the train dataset in the center of
the mechanism question while preserving held-out analog rows. It also prevents
the next GPU run from silently collapsing back to a person/backpack-only story.

## Component-Output Tomography Result

Added pair-building and reduction helpers for the selected ownership-tomography
panel:

```text
src/analysis/autoregressive_binding_template_ablation/formation_ownership_component_pairs.py
scripts/analysis/run_autoregressive_binding_formation_ownership_component_pairs.py
tests/analysis/test_formation_ownership_component_pairs.py

src/analysis/autoregressive_binding_template_ablation/formation_ownership_component_reducer.py
scripts/analysis/run_autoregressive_binding_formation_ownership_component_reduce.py
tests/analysis/test_formation_ownership_component_reducer.py
```

Pair artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_ownership_component_pairs/v1_v24_v28_panel_layers17_18_selfattn_mlp
```

Model-backed component patch artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_ownership_component_patch/v1_component_layers17_18_panel_merged
```

Post-hoc basin, compatibility, and component summaries:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_ownership_component_basin/v1_component_layers17_18_panel_merged_basin
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_ownership_component_compatibility/v1_component_layers17_18_panel_merged_compatibility
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_ownership_component_reducer/v1_component_layers17_18_panel_merged
```

Scope:

```text
85 selected train/val ownership-panel sequences
340 component-output patch rows
layers: 17, 18
patch sites: self_attn, mlp
train rows: 176
val rows: 164
error_count: 0
```

Overall component reducer:

```text
target_like_rate: 0.1971
target_geometry_rate: 0.1000
donor_like_rate: 0.3441
origin_like_rate: 0.2853
worse_like_rate: 0.3735
mean_coord_rank_delta: +81.05
```

Site/layer split:

```text
self_attn / layer17:
  target_like=0.1294, target_geometry=0.0824,
  donor_like=0.7882, origin_like=0.3294,
  worse_like=0.1765, mean_rank_delta=+152.81

self_attn / layer18:
  target_like=0.2353, target_geometry=0.1176,
  donor_like=0.0941, origin_like=0.1765,
  worse_like=0.4471, mean_rank_delta=+5.15

mlp / layer17:
  target_like=0.2118, target_geometry=0.1176,
  donor_like=0.1882, origin_like=0.2588,
  worse_like=0.4824, mean_rank_delta=+78.33

mlp / layer18:
  target_like=0.2118, target_geometry=0.0824,
  donor_like=0.3059, origin_like=0.3765,
  worse_like=0.3882, mean_rank_delta=+87.91
```

Role-specific read:

```text
layer18_donor_snap:
  self_attn/layer17 donor_like=0.8889, target_geometry=0.0000,
  mean_rank_delta=+279.96.
  self_attn/layer18 donor_like=0.0370, target_geometry=0.1481,
  mean_rank_delta=+25.48.

origin_collapse:
  self_attn/layer17 donor_like=1.0000 and origin_like=0.9333,
  mean_rank_delta=+203.63.
  mlp/layer18 donor_like=0.7000 and origin_like=0.7000,
  mean_rank_delta=+155.03.

stable_target_geometry:
  self_attn/layer17 target_like=0.8750 and target_geometry=0.8750,
  but donor_like=1.0000 because these are already compatible near-target
  cases where target and donor basins overlap.

rank_only_target_repair:
  target rank can move through self_attn/layer17 or mlp/layer18, but
  target_geometry remains 0.0000.

persistent_worse_escape:
  no component site/layer reliably repairs the row family.
```

Mechanistic interpretation:

1. Layer-17 self-attention is a high-gain exposure path for the basin identity
   already compatible with the donor/control state. In bad rows this is
   donor/origin; in stable target-geometry rows it can look target-like because
   target and donor are already geometrically compatible.
2. Layer-18 self-attention is not simply continued donor capture. Donor-like
   landing drops sharply, while target-like and worse/escape outcomes mix. This
   makes layer 18 a resolver or redistribution point rather than another copy
   of the layer-17 donor exposure.
3. MLP patches look more like rank/value shaping and basin stabilization. They
   can strengthen origin collapse and sometimes target rank, but they do not
   explain the first donor-anchor exposure.
4. The next attention/value tomography should start at layer-17 self-attention
   heads and explicitly stratify source regions by ownership role. Layer 18
   should be studied as the resolver/recombination point.
5. The next population expansion should not chase the original semantic pair.
   Mine additional trained rows that fail under teacher-forced GT prefixes,
   then match val analogs by locus and motif. The current component result
   says the interesting question is which basin identity is exposed or resolved,
   not whether the object class pair is special.
