---
doc_id: progress.diagnostics.object_step_structural_residual_decomposition
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: val200-desc-first-20state-object-step-structural-residual-decomposition
domain: autoregressive-binding-template-ablation
updated: 2026-06-20
branch: codex/autoregressive-binding-template-study
---

# Object-Step Structural Residual Decomposition

## Scope

This note extends
`progress.diagnostics.object_step_value_region_continuation_bridge` with a
structural-span decomposition and independent-head pass.

Source panel:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_panel_selection/v2_desc_first_duplicate_unmatched_slots_selector_strict_v1
```

This is still the same strict 20-state `desc_first` duplicate/unmatched
object-step panel. It is a local causal continuation diagnostic, not full
rollout validation.

## Run Matrix

All six runs used:

```text
stage: trajectory-boundary-head-value-region-continuation
pair config: configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
trajectory_steps: 6
trajectory_prefix_source: state
value_source_regions: current_partial_coords,current_partial_box_span,pre_prefix_non_image_context
families: desc_first
intervention_arms: natural
target_next_kinds: coord
stop_reasons: object_step_role
```

Structural decomposition runs used `--head-direction-bases
structural_logit_gradient_span` with:

```text
direction_span_projection_subtract
direction_span_residual_subtract
direction_span_random_residual_subtract
```

The structural span components are the local logit-gradient bases:

```text
box_end_minus_object_ref_start
box_end_minus_object_ref_end
box_end_minus_im_end
object_ref_boundaries_minus_im_end
```

Independent-head decomposition used:

```text
candidate_head_group_mode: independent
value_region_patch_mode: direction_projection_subtract
head_direction_basis: logit_box_end_minus_object_ref_boundaries_minus_im_end
value_region_scales: 0,1
candidate_heads: 20:14,20:3,27:14,27:15,27:2,27:3,24:2,24:9
```

Run roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v3_desc_first_strict_harmful_combined_spanproj_regions3_scales012_steps6_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v3_desc_first_strict_harmful_combined_spanresid_regions3_scales012_steps6_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v3_desc_first_strict_harmful_combined_spanrandresid2_regions3_scales012_steps6_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v3_desc_first_strict_repair_combined_spanproj_regions3_scales012_steps6_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v3_desc_first_strict_repair_combined_spanresid_regions3_scales012_steps6_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v3_desc_first_strict_independent_allheads_dirproj_regions3_scales01_steps6_v1
```

Compact summary:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_decomposition_summary/v2_span_decomp_and_independent_dirproj
```

Summary counters:

```text
run_count: 6
aligned intervention-vs-baseline records: 1040
independent-head records: 416

harmful span projection: rows 915, continuations 180
harmful span residual:   rows 915, continuations 180
harmful random residual: rows 1808, continuations 336, random_controls 2
repair span projection:  rows 915, continuations 180
repair span residual:    rows 915, continuations 180
independent heads:       rows 4880, continuations 960
```

## Main Finding: Residual Dominates Projection

For the harmful combined group, removing the structural-span projection is much
weaker than removing the orthogonal residual from the same current
coord/box-span value route.

All-slot summary:

```text
harmful direction_span_projection_subtract, current_partial_coords, scale 2:
  changed_first_token_rate 0.333333
  target_match 0.416667 -> 0.166667
  mean absolute delta error +1.083333

harmful direction_span_residual_subtract, current_partial_coords, scale 2:
  changed_first_token_rate 0.666667
  target_match 0.416667 -> 0.166667
  mean absolute delta error +15.833333

harmful direction_span_projection_subtract, current_partial_box_span, scale 2:
  changed_first_token_rate 0.25
  target_match 0.25 -> 0.25
  mean absolute delta error +1.35

harmful direction_span_residual_subtract, current_partial_box_span, scale 2:
  changed_first_token_rate 0.55
  target_match 0.25 -> 0.15
  mean absolute delta error +10.85
```

The later-slot phase makes this sharper:

```text
harmful span projection, current_partial_coords, scale 2, later x2/y2:
  changed_first_token_rate 0.375
  target_match 0.25 -> 0.0
  mean absolute delta error +1.5

harmful span residual, current_partial_coords, scale 2, later x2/y2:
  changed_first_token_rate 0.75
  target_match 0.25 -> 0.125
  mean absolute delta error +22.625

harmful span residual, current_partial_box_span, scale 2, later x2/y2:
  changed_first_token_rate 0.75
  target_match 0.25 -> 0.125
  mean absolute delta error +21.75
```

The broader pre-prefix non-image context remains much weaker:

```text
harmful span residual, pre_prefix_non_image_context, scale 2:
  changed_first_token_rate 0.2
  target_match 0.25 -> 0.3
  mean absolute delta error +1.4
```

Interpretation: the large harmful current-partial route is not mainly inside
the small structural logit-gradient span. Most of the coordinate basin leverage
is carried in a high-dimensional residual component orthogonal to those four
structural token surfaces.

## Random Residual Control

The matched-norm random residual control is not inert:

```text
harmful direction_span_random_residual_subtract, current_partial_coords, scale 2:
  n 24
  changed_first_token_rate 0.708333
  target_match 0.416667 -> 0.125
  mean absolute delta error +4.166667
  changed_final_stop_rate 0.083333

harmful direction_span_random_residual_subtract, current_partial_box_span, scale 2:
  n 40
  changed_first_token_rate 0.7
  target_match 0.25 -> 0.125
  mean absolute delta error +6.25
```

This matters. It says the off-span residual subspace is generally sensitive at
matched norm; the real harmful residual is stronger and more structured, but a
large random vector in the same orthogonal complement can also move coordinate
basins.

The useful contrast is therefore:

```text
real harmful residual, current_partial_coords, scale 2:
  mean absolute delta error +15.833333

matched random residual, current_partial_coords, scale 2:
  mean absolute delta error +4.166667
```

The null is not zero, but the actual route still carries more aligned damage.

## Full-Vector Catastrophe Suggests A Component Interaction

The previous full-vector bridge had the most catastrophic example:

```text
source 36 / image 3255 / object 14 / y2
target 996, baseline 999
harmful full-vector current_partial_coords scale 2 -> 234
delta error +759
```

The decomposition does not reproduce that flip with either component alone:

```text
same source/object/slot, current_partial_coords, scale 2:

full_vector_subtract:
  999 -> 234
  patch norm 849.635

direction_span_projection_subtract:
  999 -> 999
  patch norm 192.176

direction_span_residual_subtract:
  999 -> 999
  patch norm 827.616

direction_span_random_residual_subtract:
  random control 0: 999 -> 999
  random control 1: 999 -> 995
```

This argues against a simple "the residual alone causes every catastrophe"
reading. The weak claim is safe: neither tested component alone reproduced the
largest prior full-vector catastrophe. A better working hypothesis is a
threshold interaction: the structural projection and high-norm residual
components jointly move the local coordinate basin across a decision boundary,
while either component alone can remain below the greedy-token flip threshold
for some states.

The residual still explains the general scale of later-slot drift; the
full-vector catastrophe may require component superposition or another
nonlinear scale-threshold effect. This is not yet directly proven because this
artifact set does not run an explicit projection-plus-residual factorial.

## Independent Heads Are Weaker And Mixed

The independent-head direction-projection run is useful because it says the
combined-group effect is not trivially reducible to one head.

Largest harmful independent-head region effects:

```text
27:14 current_partial_box_span:
  changed_first_token_rate 0.2
  target_match 0.25 -> 0.3
  mean absolute delta error +1.45

20:3 current_partial_coords:
  changed_first_token_rate 0.166667
  target_match 0.416667 -> 0.25
  mean absolute delta error +0.75

27:15 current_partial_box_span:
  changed_first_token_rate 0.1
  target_match 0.25 -> 0.25
  mean absolute delta error +1.05
```

Repair heads are also mixed, not uniformly protective:

```text
24:2 current_partial_box_span:
  changed_first_token_rate 0.2
  target_match 0.25 -> 0.35
  mean delta error -1.6

27:2 current_partial_box_span:
  changed_first_token_rate 0.1
  target_match 0.25 -> 0.2
  mean absolute delta error +0.65
```

Specific rows show individual-head nudges, but no individual head reproduced
the strongest combined direction-projection and full-vector basin flips. For
example, on `source 36 / object 14 / y2`, all eight independent heads left the
first token at `999`; the full-vector combined group at scale 2 changed it to
`234`.

Interpretation: the current evidence favors a distributed head-group route.
Individual heads carry different directional pieces, but the high-impact basin
state appears after aggregation.

## Updated Mechanistic Picture

Current best model:

```text
1. The late object-step query reads current partial coord/box text state through
   a small set of heads.

2. Harmful heads route a high-norm value vector from that state.

3. Only a small fraction of that vector lies in simple structural logit-gradient
   directions. The high-dimensional residual carries much of the coordinate
   basin leverage.

4. Some extreme coordinate flips are not produced by projection or residual
   alone. They appear when both components are removed together, suggesting
   basin-threshold interaction in the next-token logit landscape.

5. Head-level effects are distributed. Single heads nudge local basins, while
   combined groups cross stronger thresholds.
```

This points toward a richer mechanism than "duplication head attends to prior
coordinate and repeats it." For this panel, the problematic state seems to be
an autoregressive coordinate-attractor basin encoded in a distributed
high-dimensional value route, with simple structural token surfaces only
describing part of the state's causal geometry.

## Boundaries

Do not overclaim:

```text
This is still a 20-state selected panel, not a population estimate.
The random residual control uses only two controls here.
Matched random residual controls are highly active, so residual-space harm is
not direction-specific yet.
The structural span is limited to four token-gradient surfaces.
The independent-head pass used direction_projection_subtract only, not full
vector or residual decomposition per head.
The component-interaction claim is currently a working hypothesis based on
what projection-only and residual-only fail to reproduce; it should be verified
with combined projection+residual partial sweeps.
```

## Next Experiments

1. Run an exact projection/residual factorial on the harmful combined group:
   none, projection-only, residual-only, reconstructed projection+residual,
   projection plus matched/shuffled residual, and residual plus matched/shuffled
   projection. Use a dense scale grid around the suspected threshold, for
   example `{1.0,1.25,1.5,1.75,2.0,2.25}`.

2. Run a two-axis projection/residual grid on the harmful combined group:
   projection scale in `{0,0.5,1,1.5,2}` crossed with residual scale in
   `{0,0.5,1,1.5,2}`. The key is to directly map the threshold surface for
   rows like `36/14/y2`.

3. Add more random residual controls, at least 8 as supported by the default,
   and summarize quantiles rather than only means.

4. Run independent-head residual decomposition for the most implicated heads
   (`27:14`, `27:15`, `20:3`, `20:14`) to distinguish high-norm carrier heads
   from low-norm trigger heads.

5. Build row-specific coordinate surfaces, not just structural boundary
   surfaces, for the residual decomposition. A residual relative to boundary
   logits may still contain coordinate-slot directions.
