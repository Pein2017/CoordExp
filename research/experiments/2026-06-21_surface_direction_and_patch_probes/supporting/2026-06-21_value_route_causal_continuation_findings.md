---
doc_id: progress.diagnostics.value_route_causal_continuation_findings
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: tiny-four-state-y2-causal-continuation-plus-v15-random-controls-and-v16-fixed-effects
domain: autoregressive-binding-template-ablation
updated: 2026-06-21
branch: codex/autoregressive-binding-template-study
---

# Value Route Causal Continuation Findings

## Scope

This note follows:

```text
progress/diagnostics/2026-06-21_route_ridge_bridge_findings.md
```

The route-ridge bridge suggested that the immediate harmful route for selected
y2 states runs through `current_partial_coords` / `current_partial_box_span`.
This note records the first causal tests on the two strongest route-ridge
candidates:

```text
source36/object14/y2, head L27H14
source145/object20/y2, head L27H15
```

All evidence here is tiny, deterministic, and state-local. It should guide the
next mechanism probe, not stand in for population-level validation.

## Helper Added

New CPU-only reducer:

```text
scripts/analysis/run_autoregressive_binding_value_region_continuation_panel.py \
  --continuation-rows <trajectory_boundary_head_value_region_continuation_rows.jsonl> \
  --continuation-rows <another_rows.jsonl> \
  --output-root <panel_root>
```

It writes:

```text
value_region_continuation_panel_summary.json
value_region_continuation_panel_summary.md
value_region_continuation_panel_manifest.json
```

The reducer groups continuation paths by the original seed state encoded in
`state_key` or `continuation_id`, so later natural follow-up rows with
`target_next_coord_slot=null` or a new object coordinate slot remain attached
to the originating y2 intervention.

## Focused Inputs

Filtered selected-row inputs:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_route_causal_inputs/four_state_y2_v1
```

Useful files:

```text
four_state_y2_selected_rows.jsonl
source36_145_y2_selected_rows.jsonl
source36_y2_selected_rows.jsonl
source145_y2_selected_rows.jsonl
manifest.json
```

These were filtered from:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_panel_selection/v2_desc_first_duplicate_unmatched_slots_selector_strict_v1/selected_rows.jsonl
```

## Causal Artifacts

Individual structural-span continuation probes:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v5_source36_obj14_y2_l27h14_individual_span_factorial_regions2_steps6_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v5_source145_obj20_y2_l27h15_individual_span_factorial_regions2_steps6_v1
```

First-step full-vector probe-bin interventions:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_intervention/v5_source36_obj14_y2_l27h14_fullvec_probe_bins_regions2_scales00512_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_intervention/v5_source145_obj20_y2_l27h15_fullvec_probe_bins_regions2_scales00512_v1
```

Source145 threshold replay:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v7_source145_obj20_y2_l27h15_projection_threshold_coords_p0_20_r0_03_steps8_v1
```

Component-basis probes:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v8_source145_obj20_y2_l27h15_component_box_end_minus_start_coords_scales0_3_steps8_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v8_source145_obj20_y2_l27h15_component_box_end_minus_end_coords_scales0_3_steps8_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v8_source145_obj20_y2_l27h15_component_box_end_minus_imend_coords_scales0_3_steps8_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v8_source145_obj20_y2_l27h15_component_obj_boundaries_minus_imend_coords_scales0_3_steps8_v1
```

Panel roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_continuation_panel/v5_source36_145_individual_span_factorial_y2_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_continuation_panel/v7_source145_projection_threshold_with_baseline_y2_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_continuation_panel/v8_source145_component_basis_y2_v1
```

First-step tie/plateau panel:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_tie_panel/source145_y2_v7_v8_v9_first_step_v1
```

Bin-resolved source145 coordinate probe:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_intervention/v10_source145_obj20_y2_l27h15_component_start_probe_bins140_160_scales08_20_v1
```

Coord-token adapter offset smoothness:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coord_token_embedding_offset_smoothness/source145_bins140_160_desc_geometry_v1
```

Four-state span-factorial replication:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v13_four_state_y2_l27h3_individual_span_factorial_regions2_p0_2_r0_2_steps6_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v13_four_state_y2_l27h14_individual_span_factorial_regions2_p0_2_r0_2_steps6_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v13_four_state_y2_l27h15_individual_span_factorial_regions2_p0_2_r0_2_steps6_scalarfirst_v1
```

Four-state panels:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_continuation_panel/v13_four_state_y2_l27h3_14_15_span_factorial_regions2_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_tie_panel/v13_four_state_y2_l27h3_14_15_first_step_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_coordinate_copy/v13_four_state_y2_l27h3_span_factorial_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_coordinate_copy/v13_four_state_y2_l27h14_span_factorial_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_coordinate_copy/v13_four_state_y2_l27h15_span_factorial_scalarfirst_v1
```

Source36 termination-component challenge:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v14_source36_obj14_y2_l27h14_box_end_minus_imend_scales0_3_steps10_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v14_source36_obj14_y2_l27h14_obj_boundaries_minus_imend_scales0_3_steps10_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_continuation_panel/v14_source36_l27h14_termination_components_steps10_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_tie_panel/v14_source36_l27h14_termination_components_first_step_scalarfirst_v1
```

## Main Finding

The two route-ridge candidates are not the same mechanism.

| state / head | realized first-token effect | continuation effect | strongest read |
| --- | --- | --- | --- |
| source36/object14/y2 / L27H14 | `50/50` first tokens stay `<|coord_999|>` | `50/50` end at `<|im_end|>` | projection axis can raise target-prob slightly, but the high-coordinate plus immediate-termination basin is not broken by this head alone |
| source145/object20/y2 / L27H15 | local coordinate winner can jump among `154`, `152`, `151`, `145`, `144` | path continues into another object-ref span; no termination repair | this head participates in a fragile local coordinate basin, but not enough to land the exact target or fix object-level continuation |

This argues against the simplest explanation that current-prefix value-route
subtraction directly repairs duplication. The route is causal, but it appears
to control local basin geometry and schema/termination margins separately.

## Source36

Structural-span continuation:

```text
first_step_token_counts: {'<|coord_999|>': 50}
final_stop_reason_counts: {'im_end': 50}
mean_first_step_target_next_token_prob_delta: -0.00329810597
mean_first_step_box_end_minus_object_ref_boundaries_logit_delta: +0.134263916016
mean_first_step_box_end_minus_im_end_logit_delta: +1.170344238281
mean_first_step_object_ref_boundaries_minus_im_end_logit_delta: +1.036080322266
```

Projection-only cells increase target probability more than residual-only
cells, but the greedy output remains:

```text
<|coord_999|><|box_end|><|im_end|>
```

Full-vector first-step intervention agrees that L27H14 is not a greedy
coordinate repair route for this state:

```text
target: <|coord_996|>
baseline_top: <|coord_999|>
intervention_top: <|coord_999|>
target_rank: 5 -> 5
scale 2.0 target_prob_delta: about -0.006
```

Interpretation:

```text
L27H14/current-prefix route moves schema/termination margins strongly, but the
source36 y2 slot is already captured by a high-coordinate attractor that this
single head cannot dislodge.
```

## Source145

Structural-span continuation:

```text
first_step_token_counts: {'<|coord_154|>': 48, '<|coord_144|>': 2}
final_stop_reason_counts: {'max_steps': 50}
mean_first_step_target_next_token_prob_delta: +0.002970079184
mean_first_step_box_end_minus_object_ref_boundaries_logit_delta: -0.1032421875
mean_first_step_box_end_minus_im_end_logit_delta: -0.3659765625
mean_first_step_object_ref_boundaries_minus_im_end_logit_delta: -0.262734375
```

The exact target is:

```text
<|coord_146|>
```

The model usually emits:

```text
<|coord_154|>
```

The source145 threshold replay with baseline included produced:

```text
seed_count: 52
token_counts: {'<|coord_154|>': 38, '<|coord_144|>': 9, '<|coord_152|>': 4, '<|coord_151|>': 1}
```

Notable non-154 cells:

| projection | residual | first token | target prob delta |
| ---: | ---: | --- | ---: |
| 1.3 | 0.1 | `<|coord_144|>` | +0.006946902722 |
| 1.4 | 0.0 | `<|coord_144|>` | +0.000804308802 |
| 1.4 | 0.1 | `<|coord_144|>` | +0.006858743727 |
| 1.45 | 0.1 | `<|coord_144|>` | +0.006861936301 |
| 1.5 | 0.0 | `<|coord_152|>` | -0.001511353999 |
| 1.5 | 0.2 | `<|coord_151|>` | +0.002099819481 |
| 2.0 | 0.1 | `<|coord_144|>` | +0.001857455820 |
| 2.0 | 0.3 | `<|coord_152|>` | +0.000788409263 |

Interpretation:

```text
This is not a monotonic target dial. It is a local coordinate basin where small
projection/residual changes move top-1 among nearby bins. The target bin 146
gets more probability in several cells, but top-1 often jumps to neighboring
or competitor bins instead.
```

## Component Basis Results

Component probes used:

```text
head: L27H15
region: current_partial_coords
patch mode: direction_projection_subtract
scales: 0,0.5,1,1.5,2,3
steps: 8
```

First-token counts:

| component basis | first-token counts | strongest interpretation |
| --- | --- | --- |
| `box_end_minus_object_ref_start` | `154`: 3, `151`: 2, `145`: 1 | strongest coordinate-bin mover; closest to target 146 |
| `box_end_minus_object_ref_end` | `154`: 6 | raises target probability but does not change top-1 |
| `box_end_minus_im_end` | `154`: 4, `152`: 2 | can move local basin, but away from exact target |
| `object_ref_boundaries_minus_im_end` | `154`: 6 | changes schema margins and target probability without top-1 coordinate flip |

Most important component details:

```text
box_end_minus_object_ref_start:
  scale 1.0 -> <|coord_151|>, target_prob_delta +0.002327386290
  scale 1.5 -> <|coord_145|>, target_prob_delta +0.000518828630
  scale 3.0 -> <|coord_151|>, target_prob_delta +0.001429017633

box_end_minus_im_end:
  scale 1.5 -> <|coord_152|>, target_prob_delta -0.002314608544
  scale 2.0 -> <|coord_152|>, target_prob_delta -0.001816630363
```

This makes the target-neighbor movement more specific:

```text
The source145 local coordinate basin is most sensitive to the box-end vs
object-ref-start component, not to the object-ref-end component alone.
```

## Top-Probability Tie Correction

A follow-up audit found that some apparent first-token differences are not
separate intervention effects. They are exact top-probability plateaus whose
displayed winner can depend on top-k list ordering.

The key case:

```text
v8 scale 1.5 / box_end_minus_object_ref_start:
  intervention_top_token_text: <|coord_145|>
  trajectory_greedy_token_text: <|coord_145|>

v9 scale 1.5 / box_end_minus_object_ref_start:
  intervention_top_token_text: <|coord_145|>
  trajectory_greedy_token_text: <|coord_151|>
```

The scalar intervention readout is the same in both artifacts:

```text
target: <|coord_146|>
intervention_target_next_token_prob: 0.050682842731
target_next_token_prob_delta: +0.000518828630
intervention_top_token_prob: 0.065078057349
tie group: <|coord_145|>, <|coord_151|>, <|coord_154|>
```

The difference is that continuation seeding follows the first item in the
recorded top-k list, while the scalar `intervention_top_token_text` comes from
the direct argmax readout. For exact ties, the top-k ordering is not a stable
mechanistic signal.

Operational correction: future value-region continuation rows now normalize
the intervention readout so the scalar `intervention_top_token_id` is first in
`top_token_ids`, while retaining the remaining top-k entries for tie
diagnostics. Historical v8/v9 continuation rows above should therefore be read
as pre-fix evidence of plateau membership, not as distinct scalar intervention
effects.

Canonical post-fix reruns used the same v9 fine-scale settings with
`value_top_k=16`:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v12_source145_obj20_y2_l27h15_component_start_fine_coords_scales08_20_steps8_topk16_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v12_source145_obj20_y2_l27h15_component_box_imend_fine_coords_scales08_20_steps8_topk16_scalarfirst_v1
```

The paired diff artifacts are:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_continuation_diff/source145_y2_start_v9_vs_v12_topk16_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_continuation_diff/source145_y2_box_imend_v9_vs_v12_topk16_scalarfirst_v1
```

For `box_end_minus_object_ref_start`, all `14/14` first-step rows matched
between v9 and v12. The fix changed the continuation greedy token in 4 rows:

```text
scales 1.4, 1.45, 1.5, 1.55:
  before greedy: <|coord_151|>
  after greedy:  <|coord_145|>
  scalar top:    <|coord_145|>
  tie group:     <|coord_145|>, <|coord_151|>, <|coord_154|>
```

For `box_end_minus_im_end`, all `14/14` first-step rows also matched. The fix
changed the continuation greedy token in 5 rows:

```text
scales 1.3, 1.4, 1.5, 1.7, 2.0:
  before greedy: <|coord_151|>
  after greedy:  <|coord_144|>
  scalar top:    <|coord_144|>
```

In both paired diffs:

```text
before_only_count: 0
after_only_count: 0
duplicate_key_count: 0
target_enters_tie_count: 0
target_leaves_tie_count: 0
target_match_improved_count: 0
target_match_regressed_count: 0
```

This validates the operational interpretation: the scalar-first correction
removes a top-k ordering artifact in continuation seeding, but it does not turn
the intervention into target recovery. The target coordinate 146 still does not
enter the top-probability tie group in this slice.

The v12 first-step tie panel:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_tie_panel/source145_y2_v12_start_box_imend_first_step_topk16_scalarfirst_v1
```

reports:

```text
analyzed first-step rows: 28
tied first-step rows: 14
tie_size_counts: {'1': 14, '2': 5, '3': 4, '4': 1, '5': 1, '6': 3}
greedy_vs_intervention_top_mismatch_count: 0
target_in_top_tie_count: 0
```

The earlier first-step tie panel over the combined source145 v7/v8/v9 probes
reported:

```text
analyzed first-step rows: 86
tied first-step rows: 37
tie_size_counts: {'1': 49, '2': 12, '3': 15, '4': 4, '5': 3, '6': 3}
greedy_vs_intervention_top_mismatch_count: 20
target_in_top_tie_count: 0
```

The bin-resolved v10 probe then checked all coordinate bins 140-160 for the
same source145/object20/y2 state and the `box_end_minus_object_ref_start`
component. Around the most sensitive scales:

| scale | best probed bins | target 146 logit | best logit | 146 minus best |
| ---: | --- | ---: | ---: | ---: |
| 1.3 | `151`, `154` | 29.25 | 29.5 | -0.25 |
| 1.4 | `145`, `151`, `154` | 29.25 | 29.5 | -0.25 |
| 1.45 | `145`, `151`, `154` | 29.25 | 29.5 | -0.25 |
| 1.5 | `145`, `151`, `154` | 29.25 | 29.5 | -0.25 |
| 1.55 | `145`, `151`, `154` | 29.25 | 29.5 | -0.25 |
| 1.6 | `154` | 29.25 | 29.625 | -0.375 |

The operational precision caveat matters: the bridge HF loader uses
`torch_dtype=torch.bfloat16`, so logits around 29 are observed on a bf16 decode
surface with roughly 0.125 logit granularity. The exact ties are therefore
decode-surface ties, not evidence that an unobserved fp32 surface is exactly
equal.

The special-token adapter itself is also not locally smooth around this basin.
For the learned coord-token offset vectors:

| checkpoint | global neighbor mean | local 140-160 mean | largest local jump | norm(154) | norm(146) |
| --- | ---: | ---: | --- | ---: | ---: |
| desc-first converted ckpt928 | 0.075115 | 0.076987 | `154-155`: 0.125618 | 0.102295 | 0.080118 |
| geometry-first ckpt928 | 0.126258 | 0.130462 | `154-155`: 0.186783 | 0.158336 | 0.140495 |

Both checked adapters have their largest local jump in this window at
`154-155`, and bin 154 has an elevated offset norm relative to the exact target
bin 146. This does not by itself prove the final logit plateau is caused by the
embedding offset, because the recurrent/attention state and LoRA path also
shape the final hidden vector. It does make the basin origin more concrete:
the newly trained coord-token surface has a learned local discontinuity exactly
near the competing bin.

This changes the interpretation of the component sweep:

```text
The intervention often moves source145 into a flat local coordinate plateau.
The plateau includes nearby competing bins such as 144/145/151/152/154/158,
but the exact target bin 146 is not in the top-probability tie group in this
panel.
```

So the stronger mechanistic claim is not "the head chooses 145 instead of 151."
It is:

```text
L27H15/current_partial_coords can reshape the local coordinate basin into a
quantized plateau that excludes the correct coordinate. This is direct evidence
for a coordinate-slot attraction basin over the newly trained coord embeddings,
not merely ordinary greedy decoding noise. The word "quantized" here is
literal: the currently measured decode surface is bf16-quantized, and this
precision interacts with the learned coord-token basin.
```

## Current Mechanistic Picture

The working picture after this slice:

```text
1. A current-prefix value route can be causal without being sufficient for
   repair.

2. Source36 is a high-coordinate plus termination basin:
   L27H14 can move boundary-vs-im_end margins, but the greedy coordinate and
   termination trajectory are locked.

3. Source145 is a local coordinate basin:
   L27H15 interventions perturb the basin among nearby coordinate bins, and
   the box_end_minus_object_ref_start component is the clearest target-neighbor
   mover. Several "winner" changes are exact top-probability ties, so the
   plateau membership is more meaningful than the displayed single winner.

4. Exact target recovery is harder than moving the basin:
   probability mass can increase for the target bin while the top-probability
   plateau excludes it.

5. Coordinate-slot basin and object-span/termination basin are partly
   separable:
   source145 coordinate top-1 moves, but the continuation still proceeds into
   another object-ref span; source36 schema margins move, but visible output
   does not.
```

## Four-State Replication

To test whether the source36/source145 split is just a two-example story, a
four-state span-factorial continuation panel was run on:

```text
source33/object11/y2 target <|coord_417|>
source36/object14/y2 target <|coord_996|>
source123/object5/y2 target <|coord_270|>
source145/object20/y2 target <|coord_146|>
```

Settings:

```text
heads: L27H3, L27H14, L27H15
head_direction_basis: structural_logit_gradient_span
regions: current_partial_coords,current_partial_box_span
projection scales: 0,0.5,1,1.5,2
residual scales: 0,0.5,1,1.5,2
trajectory steps: 6
top_k: 8
```

All three head runs produced:

```text
source_row_count: 4
continuation_count: 200
row_count: 1050
model_perturbation_ran: true
training_ran: false
```

The combined panel had:

```text
input_record_count: 3150
first-step cells: 600
panel rows: 600
tie-panel tied first-step rows: 117
```

Key first-step and stop behavior:

| source / target | head | changed first-token cells | changed stop cells | first-token support | final stops |
| --- | --- | ---: | ---: | --- | --- |
| source33 / `417` | L27H3 | 0/50 | 0/50 | `415` only | `max_steps`: 50 |
| source33 / `417` | L27H14 | 0/50 | 0/50 | `415` only | `max_steps`: 50 |
| source33 / `417` | L27H15 | 0/50 | 0/50 | `415` only | `max_steps`: 50 |
| source36 / `996` | L27H3 | 0/50 | 0/50 | `999` only | `im_end`: 50 |
| source36 / `996` | L27H14 | 0/50 | 0/50 | `999` only | `im_end`: 50 |
| source36 / `996` | L27H15 | 0/50 | 0/50 | `999` only | `im_end`: 50 |
| source123 / `270` | L27H3 | 16/50 | 0/50 | `252`,`253`,`254` | `max_steps`: 50 |
| source123 / `270` | L27H14 | 0/50 | 0/50 | `252` only | `max_steps`: 50 |
| source123 / `270` | L27H15 | 46/50 | 0/50 | `252`,`253`,`254`,`259`,`267` | `max_steps`: 50 |
| source145 / `146` | L27H3 | 1/50 | 0/50 | `154`,`151` | `max_steps`: 50 |
| source145 / `146` | L27H14 | 27/50 | 0/50 | `154`,`144`,`145`,`151` | `max_steps`: 50 |
| source145 / `146` | L27H15 | 2/50 | 0/50 | `154`,`144` | `max_steps`: 50 |

Tie-panel target membership:

```text
target_in_top_tie_count: 0 for every source/head slice
reported_topk_first_matches_argmax_top mismatch: 0 after scalar-first fix
```

Coordinate-copy classification over all `600` continuations:

```text
coordinate_copy_class: incomplete for 600/600
emitted_coord_count: 1 for 600/600
near8_target_slot_match_count: 0 for 600/600
exact_target_slot_match_count: 0 for 600/600
```

Interpretation:

```text
The four-state panel strengthens the split but also narrows it. The value-route
grid can move coordinate top-1 for source123 and source145, but it does not
make the exact target win, enter the top-probability tie group, or alter
object-span/termination behavior. Source36 is a qualitatively separate
high-coordinate + immediate-termination lock: none of the tested heads/regions
move either the coordinate or the stop class. Source33 is also locked locally,
but in a max-steps continuation basin rather than an im_end basin.
```

## Source36 Termination Challenge

After the four-state panel, source36 was tested with direct local component
bases aimed at the termination/schema side:

```text
source36/object14/y2
head: L27H14
regions: current_partial_coords,current_partial_box_span
scales: 0,0.5,1,1.5,2,3
trajectory steps: 10
top_k: 16
patch mode: direction_projection_subtract
bases:
  box_end_minus_im_end
  object_ref_boundaries_minus_im_end
```

Results:

| basis | continuations | first token | final stop | max box_end-im_end delta | max object-boundary-im_end delta |
| --- | ---: | --- | --- | ---: | ---: |
| `box_end_minus_im_end` | 12 | `999`: 12 | `im_end`: 12 | +3.2988 | +2.7964 |
| `object_ref_boundaries_minus_im_end` | 12 | `999`: 12 | `im_end`: 12 | +2.1934 | +2.6440 |

The v14 tie panel found:

```text
analyzed first-step rows: 24
tied first-step rows: 0
```

Interpretation:

```text
For source36, even direct local directions that strongly increase
box_end/object-boundary margins over im_end do not change realized stop
behavior. The immediate `<|coord_999|><|box_end|><|im_end|>` path is not
controlled by this local first-step margin alone. This makes source36 more like
a downstream/sequential termination lock than a simple local schema-logit
failure.
```

## Matched Random Residual Control

The next control tested whether the off-span residual part of the combined
harmful route is specific, or whether matched random residuals can perturb the
coordinate basin just as easily.

Real harmful baseline:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v4_desc_first_strict_harmful_combined_span_factorial_p012_r012_regions3_steps6_v1
```

Random residual controls:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v15_desc_first_strict_harmful_combined_span_random_residual_regions3_scales012_steps6_seed101_controls2_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v15_desc_first_strict_harmful_combined_span_random_residual_regions3_scales012_steps6_seed202_controls2_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v15_desc_first_strict_harmful_combined_span_random_residual_regions3_scales012_steps6_seed303_controls2_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v15_desc_first_strict_harmful_combined_span_random_residual_regions3_scales012_steps6_seed404_controls2_scalarfirst_v1
```

Post-hoc reducers:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_random_residual_comparison/v15_real_harmful_vs_random_residual_controls8_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_factorial_summary/v15_real_harmful_vs_random_residual_controls8_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_tie_panel/v15_random_residual_controls8_first_step_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_continuation_panel/v15_random_residual_controls8_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_specificity_panel/v16_real_harmful_vs_random_residual_controls8_fixed_effect_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_coordinate_copy/v15_random_residual_seed101_controls2_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_coordinate_copy/v15_random_residual_seed202_controls2_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_coordinate_copy/v15_random_residual_seed303_controls2_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_coordinate_copy/v15_random_residual_seed404_controls2_scalarfirst_v1
```

Reusable helper added:

```text
scripts/analysis/run_autoregressive_binding_value_region_specificity_panel.py
```

It writes:

```text
value_region_specificity_records.jsonl
value_region_specificity_paired_records.jsonl
value_region_specificity_summary.json
value_region_specificity_summary.md
```

The key first-step comparison is residual-only real harmful cells
`projection_scale=0, residual_scale in {1,2}` against random residual controls
at `scale in {1,2}`:

| slice | ok cells | changed first-token | target match | near8 | catastrophic `abs_error>=100` | mean abs error | mean target prob delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real residual-only | 104 | 37 / 104 = 0.356 | 24 / 104 = 0.231 | 68 / 104 = 0.654 | 2 | 9.913 | -0.010620 |
| random residual controls | 832 | 314 / 832 = 0.377 | 168 / 832 = 0.202 | 541 / 832 = 0.650 | 3 | 9.559 | -0.005709 |

Region split:

| region | real changed first-token | random changed first-token | real target match | random target match | real near8 | random near8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `current_partial_coords` | 14 / 24 = 0.583 | 107 / 192 = 0.557 | 4 / 24 = 0.167 | 38 / 192 = 0.198 | 16 / 24 = 0.667 | 130 / 192 = 0.677 |
| `current_partial_box_span` | 17 / 40 = 0.425 | 169 / 320 = 0.528 | 7 / 40 = 0.175 | 49 / 320 = 0.153 | 24 / 40 = 0.600 | 187 / 320 = 0.584 |
| `pre_prefix_non_image_context` | 6 / 40 = 0.150 | 38 / 320 = 0.119 | 13 / 40 = 0.325 | 81 / 320 = 0.253 | 28 / 40 = 0.700 | 224 / 320 = 0.700 |

The random controls therefore weaken the strong version of the
route-specific residual claim:

```text
Off-span residual-only subtraction is not clean evidence of a named
value-route repair mechanism. Matched random residuals move the first
coordinate and land near the target at comparable rates.
```

This does not falsify the whole route story. It specifically downgrades the
raw residual-only effect. The structured projection/component hypothesis is
still alive, because real projection-only cells were cleaner in this same
comparison:

```text
real projection-only p in {1,2}, residual=0:
  ok: 104
  changed_first_token_rate: 0.192
  target_match_rate: 0.279
  near8_rate: 0.712
  catastrophic_abs_error_ge_100_count: 0
  mean_target_prob_delta: -0.001202
```

The v16 fixed-effect reducer then paired real residual-only cells against
random residual controls by:

```text
state_key, value_source_region, target_next_coord_slot, target_next_coord_bin
```

For the matched subset:

```text
paired fixed-effect count: 52
real residual-only rows per pair: residual_scale in {1,2}
random residual rows per pair: scale in {1,2}, four seeds x two controls
mean_delta_patch_norm (real - random): -0.000000923
```

The paired deltas were:

| metric | real residual-only minus random residual |
| --- | ---: |
| changed first-token rate | -0.015625 |
| target match rate | +0.028846 |
| near8 target match rate | +0.003606 |
| mean abs error | +0.354567 |
| target prob delta | -0.004910 |

Region-level paired deltas:

| region | pairs | changed first-token | target match | near8 | mean abs error | target prob delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `current_partial_box_span` | 20 | -0.131250 | +0.021875 | +0.015625 | -2.771875 | -0.005561 |
| `current_partial_coords` | 12 | -0.010417 | -0.031250 | -0.010417 | +4.994792 | -0.012081 |
| `pre_prefix_non_image_context` | 20 | +0.096875 | +0.071875 | 0.000000 | +0.696875 | +0.000043 |

This first fixed-effect pass is close to the one-off comparison but more
decisive on norm confounding:

```text
At matched state/region/slot and effectively identical patch norm, real
residual-only cells do not separate cleanly from random off-span residuals.
They have a tiny target-match advantage, no meaningful near8 advantage, and a
worse target-probability delta. The residual-only effect should be treated as
generic coordinate-basin fragility until a stricter wrong-head/wrong-region or
projection-matched control beats this result.
```

The first-step tie panel over the random controls reported:

```text
analyzed first-step rows: 1344
tied first-step rows: 220
tie_size_counts: {'0': 96, '1': 1028, '2': 118, '3': 54, '4': 40, '5': 6, '6': 1, '7': 1}
greedy_vs_intervention_top_mismatch_count: 0
target_in_top_tie_count: 326
```

This panel is broader than the four-state y2-only panel and includes easier
slots plus scale-zero rows, so it should not be compared directly to the
earlier `target_in_top_tie_count: 0` y2 result.

The continuation/coordinate-copy reducers give the stronger guardrail:

| scope | rows | class counts | emitted coord counts | final stop counts |
| --- | ---: | --- | --- | --- |
| all four random seeds | 1344 | `incomplete`: 864, `mixed`: 289, `target_like`: 95, `no_coords`: 96 | `0`: 96, `1`: 288, `2`: 288, `3`: 288, `4`: 384 | `max_steps`: 1052, `im_end`: 196, `intervention_skipped`: 96 |

Target-slot distributions:

```text
exact_target_slot_match_count: {0: 948, 1: 252, 2: 40, 3: 8, null: 96}
near1_target_slot_match_count: {0: 947, 1: 159, 2: 94, 3: 45, 4: 3, null: 96}
near8_target_slot_match_count: {0: 875, 1: 120, 2: 105, 3: 53, 4: 95, null: 96}
```

The important negative result is:

```text
exact full bbox repair: 0 / 1344
```

Random residuals can produce exact first-coordinate hits:

```text
first emitted coord exactly equals target_next_coord_bin: 288 / 1344
first exact matches with target_next_token_rank == 1: 288 / 288
```

But these mostly do not propagate:

```text
first exact matches by class:
  incomplete: 275
  target_like: 7
  mixed: 6

first exact matches by emitted coord count:
  1 coord: 3
  2 coords: 96
  3 coords: 176
  4 coords: 13
```

Concrete examples:

| case | target | emitted | read |
| --- | --- | --- | --- |
| source145/object20/pre_y2 | `[459,126,482,146]`, next `146` | `[146]` | exact first/final coord, but only one coordinate; `incomplete`, stops by `max_steps` |
| source33/object11/pre_x2 | `[119,349,146,417]`, next `146` | `[146,415]` | exact first coordinate and near y2, but bbox incomplete |
| source36/object14/pre_x2 | `[440,989,456,996]`, next `456` | `[456,999]` | exact first coordinate, then high-coordinate attraction persists and stop is `im_end` |
| source33/object10/pre_x1 | `[111,348,142,416]` | `[111,348,142,415]` | near-target four-coordinate box with 3 exact slots, but not exact repair and continuation enters another object-ref span |

Interpretation:

```text
Matched random residuals can shake the coordinate attractor, sometimes enough
to place the correct next coordinate at top-1, but this is not object-level
repair. The effect usually remains a local coordinate/top-1 perturbation or a
near-target bbox fragment whose object-span continuation still fails.
```

Mechanistic consequence:

```text
The next question is not "did the harmful route repair duplication?" The next
question is which part of the route is structured and which part is generic
fragility of a newly trained coordinate-token basin under matched-norm
perturbations.
```

## Per-Head Residual Localization

To test whether the combined-head residual result hid one localized residual
route, the same `20` selected states were rerun per head:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v17_desc_first_strict_harmful_head20h14_residual_regions3_scales012_steps6_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v17_desc_first_strict_harmful_head20h3_residual_regions3_scales012_steps6_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v17_desc_first_strict_harmful_head27h14_residual_regions3_scales012_steps6_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v17_desc_first_strict_harmful_head27h15_residual_regions3_scales012_steps6_scalarfirst_v1
```

Settings:

```text
patch mode: direction_span_residual_subtract
regions: current_partial_coords,current_partial_box_span,pre_prefix_non_image_context
scales: 0,1,2
trajectory steps: 6
top_k: 8
candidate heads: L20H14, L20H3, L27H14, L27H15 run independently
```

Reducers:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_factorial_summary/v17_per_head_residual_regions3_scales012_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_continuation_panel/v17_per_head_residual_regions3_scales012_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_coordinate_copy/v17_head20h14_residual_regions3_scales012_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_coordinate_copy/v17_head20h3_residual_regions3_scales012_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_coordinate_copy/v17_head27h14_residual_regions3_scales012_scalarfirst_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/value_region_coordinate_copy/v17_head27h15_residual_regions3_scales012_scalarfirst_v1
```

Each head produced:

```text
input_row_count: 20
selected_state_row_count: 20
continuation row_count: 915
training_ran: false
model_perturbation_ran: true
```

First-step ok rows by residual scale:

| head | scale | changed first-token | target match | near8 | mean abs error | mean target prob delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L20H14 | 0 | 0.000 | 0.288 | 0.731 | 6.654 | 0.000000 |
| L20H14 | 1 | 0.231 | 0.192 | 0.692 | 5.942 | -0.003451 |
| L20H14 | 2 | 0.269 | 0.212 | 0.615 | 7.423 | -0.005573 |
| L20H3 | 0 | 0.000 | 0.288 | 0.731 | 6.654 | 0.000000 |
| L20H3 | 1 | 0.115 | 0.231 | 0.692 | 7.308 | -0.001801 |
| L20H3 | 2 | 0.135 | 0.269 | 0.692 | 6.904 | -0.004030 |
| L27H14 | 0 | 0.000 | 0.288 | 0.731 | 6.654 | 0.000000 |
| L27H14 | 1 | 0.173 | 0.250 | 0.692 | 7.058 | -0.001421 |
| L27H14 | 2 | 0.250 | 0.212 | 0.654 | 7.808 | -0.001818 |
| L27H15 | 0 | 0.000 | 0.288 | 0.731 | 6.654 | 0.000000 |
| L27H15 | 1 | 0.212 | 0.192 | 0.673 | 6.327 | -0.004023 |
| L27H15 | 2 | 0.288 | 0.173 | 0.692 | 6.808 | -0.008701 |

Coordinate-copy classification:

| head | rows | class counts | exact-slot distribution | near8-slot distribution | final stops |
| --- | ---: | --- | --- | --- | --- |
| L20H14 | 180 | `incomplete`: 108, `mixed`: 36, `target_like`: 12, `no_coords`: 24 | `0`: 117, `1`: 33, `2`: 6, `null`: 24 | `0`: 107, `1`: 16, `2`: 15, `3`: 6, `4`: 12, `null`: 24 | `max_steps`: 132, `im_end`: 24, `intervention_skipped`: 24 |
| L20H3 | 180 | `incomplete`: 108, `mixed`: 35, `target_like`: 13, `no_coords`: 24 | `0`: 120, `1`: 29, `2`: 6, `3`: 1, `null`: 24 | `0`: 110, `1`: 16, `2`: 12, `3`: 5, `4`: 13, `null`: 24 | `max_steps`: 132, `im_end`: 24, `intervention_skipped`: 24 |
| L27H14 | 180 | `incomplete`: 108, `mixed`: 35, `target_like`: 13, `no_coords`: 24 | `0`: 121, `1`: 29, `2`: 5, `3`: 1, `null`: 24 | `0`: 110, `1`: 16, `2`: 12, `3`: 5, `4`: 13, `null`: 24 | `max_steps`: 132, `im_end`: 24, `intervention_skipped`: 24 |
| L27H15 | 180 | `incomplete`: 108, `mixed`: 37, `target_like`: 11, `no_coords`: 24 | `0`: 119, `1`: 30, `2`: 5, `3`: 2, `null`: 24 | `0`: 108, `1`: 17, `2`: 15, `3`: 5, `4`: 11, `null`: 24 | `max_steps`: 132, `im_end`: 24, `intervention_skipped`: 24 |

Interpretation:

```text
The per-head residual-only pilot does not rescue route specificity. Individual
heads can move coordinate top-1, especially L27H15 at scale 2, but the movement
does not improve target match over the scale-0 baseline and does not repair
object-span continuation. The final-stop profile is identical across heads,
and exact full-bbox repair remains absent.
```

This strengthens the v16 read:

```text
The residual component looks more like a generic perturbation of a fragile
coordinate basin than a localized, sufficient value route. The better remaining
candidate for route-specific structure is the projection/component axis, not
the off-span residual.
```

## Next Best Experiments

High-priority:

```text
1. Second-pass specificity controls:
   The first fixed-effect panel says real residual-only does not beat matched
   random residuals. The next stricter control should add same-state
   wrong-region, wrong-head same-region, and random vectors matched to the
   projection norm rather than residual norm. If those also collapse onto the
   same margin/norm curve, residual route specificity should be retired for
   this panel.

2. Projection-vs-residual equal-norm decomposition:
   Compare real projection direction, real residual direction, random vector
   with projection norm, random vector with residual norm, and random vector
   matched for projection fraction/cosine. The v15 evidence keeps structured
   projection alive while weakening raw residual specificity.

3. Continuation sufficiency test:
   For source145/object20/y2 and source36/object14/y2, cross forced first-token
   conditions with route interventions: baseline, intervention only, force
   target coordinate only, force target coordinate plus real route
   intervention, and force target coordinate plus random residual. This tests
   whether the route actually repairs object-span continuation after the
   coordinate slot is made correct.

4. Projection/component exactness test:
   The residual path still does not repair object spans. Focus the next
   model-backed run on projection/component axes and ask whether they can move
   exact target bins at lower norm without damaging target probability.

5. Margin/fragility regression panel:
   For each intervention cell, record baseline target margin, top1-vs-second
   margin, patch norm, residual norm, projection norm, region, head, slot, and
   source state. Test whether changed-first and target-entry rates are mostly
   explained by margin and norm.

6. Precision/tie robustness rerun:
   For source145 y2 and random cases that hit `<|coord_146|>`, rerun with
   saved fp32 logits, larger top-k, and tie tolerances. Do not make route
   claims from winner identity inside bf16-quantized plateaus.
```

The most promising conceptual direction is no longer simply "does this head
repair duplication?" It is:

```text
Which subspace controls coordinate-bin basin movement, which part is generic
high-norm basin fragility, and which separate subspace controls object-span
closure/termination?
```

## Forced-Coordinate Guidance Sufficiency Probe

The next implementation slice tested whether the selected false-negative /
duplicate-prone object states are visually absent, or whether the model can
continue when the language/coordinate side is nudged into the right basin.

New CPU helpers:

```text
src/analysis/autoregressive_binding_template_ablation/forced_coordinate_rows.py
scripts/analysis/run_autoregressive_binding_forced_coordinate_rows.py
src/analysis/autoregressive_binding_template_ablation/forced_coordinate_continuation_panel.py
scripts/analysis/run_autoregressive_binding_forced_coordinate_continuation_panel.py
```

Validation:

```text
python -m pytest tests/analysis/test_forced_coordinate_rows.py tests/analysis/test_forced_coordinate_continuation_panel.py -q
Pytest: 16 passed

python -m py_compile \
  src/analysis/autoregressive_binding_template_ablation/forced_coordinate_rows.py \
  scripts/analysis/run_autoregressive_binding_forced_coordinate_rows.py \
  src/analysis/autoregressive_binding_template_ablation/forced_coordinate_continuation_panel.py \
  scripts/analysis/run_autoregressive_binding_forced_coordinate_continuation_panel.py
exit 0
```

The forced-row builder consumes the strict 20-state selected panel and appends
the currently correct coordinate token into `trajectory_prefix_text`, leaving
the original state prefix audit fields intact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/forced_coordinate_selected_rows/v18_desc_first_strict_force_target_next_coord_v1
input_row_count: 20
output_row_count: 20
forced_slot_counts: {'x1': 8, 'y1': 4, 'x2': 4, 'y2': 4}
resulting target_next_kind_counts: {'coord': 16, 'box_end': 4}
```

Guidance-only continuation baseline:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v18_forced_target_coord_guidance_baseline_head27h15_prectx_scale0_steps6_v1
selected_state_row_count: 20
row_count: 111
trajectory_prefix_source: trajectory
patch mode: direction_span_residual_subtract
head: L27H15
region: pre_prefix_non_image_context
scale: 0
```

Strict suffix reducer:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/forced_coordinate_continuation_panel/v18_forced_target_coord_guidance_baseline_v2_strict_exact
```

Important reducer semantics:

```text
suffix_prefix_exact_match:
  the first expected-suffix-length greedy tokens match the remaining suffix

suffix_exact_match:
  suffix_prefix_exact_match plus no extra continuation rows after the expected
  suffix; this is a strict artificial stopping diagnostic, not the main
  object-list correctness metric
```

Guidance-only result:

| metric | value |
| --- | ---: |
| continuations | 20 |
| suffix prefix exact match | 0.200 |
| strict artificial suffix-stop match | 0.000 |
| artificial continuation ended at expected suffix | 0.000 |
| box-end on expected step | 1.000 |
| coordinate exact rate | 12 / 36 = 0.333 |
| coordinate near4 rate | 25 / 36 = 0.694 |
| coordinate near8 rate | 30 / 36 = 0.833 |
| mean extra tokens after expected suffix | 2.750 |
| final stops | `max_steps`: 17, `im_end`: 3 |

Slot split:

| forced slot | records | suffix prefix exact | strict exact | box-end expected | coord exact | coord near8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| x1 | 8 | 0.000 | 0.000 | 1.000 | 10 / 24 = 0.417 | 20 / 24 = 0.833 |
| y1 | 4 | 0.000 | 0.000 | 1.000 | 2 / 8 = 0.250 | 7 / 8 = 0.875 |
| x2 | 4 | 0.000 | 0.000 | 1.000 | 0 / 4 = 0.000 | 3 / 4 = 0.750 |
| y2 | 4 | 1.000 | 0.000 | 1.000 | n/a | n/a |

Reading:

```text
The model is not simply visually blind at these selected states. Once one
correct coordinate is forced, it almost always stays in a nearby target-like
coordinate basin and always emits box_end at the correct relative step.

But one correct coordinate is not sufficient for exact coordinate-chain repair.
The model drifts to nearby/neighbor coordinates before it closes the box. The
post-box continuation rows should not be interpreted as "failure to stop":
most selected objects are not necessarily terminal in the object list, and the
continuation harness still uses the seed row's target metadata after box_end.
This separates three mechanisms that were previously conflated:

1. structural box closure timing is easy to restore;
2. exact coordinate-chain binding remains fragile;
3. next-object / end-of-list transition after box closure requires a separate,
   next-object-aware probe.
```

Four per-head forced-coordinate plus projection/residual grids then crossed the
forced prefix with `direction_span_projection_residual_grid_subtract`:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v19_forced_target_coord_head20h14_proj_resid_grid_regions3_steps6_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v19_forced_target_coord_head20h3_proj_resid_grid_regions3_steps6_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v19_forced_target_coord_head27h14_proj_resid_grid_regions3_steps6_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v19_forced_target_coord_head27h15_proj_resid_grid_regions3_steps6_v1
```

Settings:

```text
forced rows: v18_desc_first_strict_force_target_next_coord_v1
heads: L20H14, L20H3, L27H14, L27H15
regions: current_partial_coords,current_partial_box_span,pre_prefix_non_image_context
projection scales: 0,1,2
residual scales: 0,1,2
trajectory steps: 6
top_k: 8
trajectory_prefix_source: trajectory
```

Combined strict suffix panel:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/forced_coordinate_continuation_panel/v19_four_heads_grid_vs_forced_baseline_v1
record_count: 2180
baseline continuations: 20
grid continuations: 4 * 540
```

Head-level grid result:

| run | continuations | suffix prefix | strict exact | ended at suffix | box-end expected | coord exact | coord near8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| forced baseline | 20 | 0.200 | 0.000 | 0.000 | 1.000 | 0.333 | 0.833 |
| L20H14 grid | 540 | 0.200 | 0.000 | 0.000 | 1.000 | 0.308 | 0.813 |
| L20H3 grid | 540 | 0.200 | 0.000 | 0.000 | 1.000 | 0.315 | 0.817 |
| L27H14 grid | 540 | 0.200 | 0.000 | 0.000 | 1.000 | 0.307 | 0.811 |
| L27H15 grid | 540 | 0.200 | 0.000 | 0.000 | 1.000 | 0.276 | 0.843 |

Projection/residual aggregate across the four heads:

| projection | residual | continuations | coord exact | coord near8 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 240 | 0.333 | 0.833 |
| 0 | 1 | 240 | 0.292 | 0.815 |
| 0 | 2 | 240 | 0.278 | 0.815 |
| 1 | 0 | 240 | 0.329 | 0.829 |
| 1 | 1 | 240 | 0.301 | 0.815 |
| 1 | 2 | 240 | 0.271 | 0.819 |
| 2 | 0 | 240 | 0.319 | 0.822 |
| 2 | 1 | 240 | 0.310 | 0.822 |
| 2 | 2 | 240 | 0.278 | 0.817 |

Interpretation:

```text
The route interventions do not add object-span sufficiency after the correct
coordinate has been forced. Across all four heads and all projection/residual
grid cells:

- artificial strict suffix-stop match remains 0;
- box-end timing stays perfect but was already perfect under guidance only;
- coordinate exactness is not improved over the scale-0/guidance baseline and
  usually degrades when residual scale is nonzero.

This is stronger than the previous residual-only negative. It says that for
these selected states, the tested value-route components are not the missing
mechanism for exact coordinate-chain repair once the coordinate basin has been
seeded. They can perturb local coordinate identity, but the exact coordinate
chain lives elsewhere. Post-box transition remains unresolved because this
continuation harness is not next-object aware.
```

Updated mechanism picture:

```text
False negatives in this panel are unlikely to be pure visual non-perception.
The model can produce near-target coordinates and close the box under a small
language-side coordinate nudge. The deeper failure is binding/control: exact
slot-to-slot coordinate commitment is not governed by the tested value-region
routes.

Next best probe should move from value-route subtraction to hidden-state /
logit-basin dynamics around both exact coordinate-chain commitment and the
post-box_end transition:

1. compare hidden states at expected box_end for natural, forced-coordinate,
   and exact teacher-forced suffix prefixes;
2. construct next-object-aware post-box rows, so object_ref_start / im_end /
   next-desc logits are scored against the actual remaining object sequence;
3. identify layers/subspaces that separate "close a nearby box" from "close
   the exact target box";
4. keep coordinate-token local smoothness/basin diagnostics in the loop, but do
   not expect single value-region subtraction to repair exact coordinate chains.
```
