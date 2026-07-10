---
doc_id: progress.diagnostics.object_step_value_region_continuation_bridge
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: val200-desc-first-20state-object-step-value-region-continuation
domain: autoregressive-binding-template-ablation
updated: 2026-06-20
branch: codex/autoregressive-binding-template-study
---

# Object-Step Value-Region Continuation Bridge

## Scope

This note records the causal companion pass to
`progress.diagnostics.object_step_value_contribution_tomography`.

The source states are the same strict 20-state `desc_first` val200 object-step
coordinate panel:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_panel_selection/v2_desc_first_duplicate_unmatched_slots_selector_strict_v1
```

The previous attention/value tomography said the harmful-vs-repair split is
mostly expressed through current partial object/box/coord text state, not
through direct image-token value contribution. This continuation pass asks a
more causal question: if selected head-group value contribution from a named
region is removed, does the next coordinate basin move?

This is a local perturbation diagnostic, not full validation. It uses a small
duplicated/unmatched panel, combined head groups, selected value regions, two
intervention strengths, and six continuation steps.

The panel is intentionally sharp rather than population-representative: 20
selected rows from four source lines/images (`36`, `145`, `33`, `123`), all
`desc_first`, with repeated-object cases such as broccoli, wine glass,
snowboard, and backpack. The four runs use scale `0` as the same-condition
baseline, but they do not include matched random heads or random region
controls (`value_region_random_control_count=0` in the continuation summaries).

## Run Matrix

All four runs used:

```text
stage: trajectory-boundary-head-value-region-continuation
pair config: configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
selected rows: strict object-step 20-state panel above
trajectory_steps: 6
candidate_head_group_mode: combined_all
head_direction_basis: logit_box_end_minus_object_ref_boundaries_minus_im_end
value_source_regions: current_partial_coords,current_partial_box_span,pre_prefix_non_image_context
value_region_scales: 0,1,2
trajectory_prefix_source: state
families: desc_first
intervention_arms: natural
target_next_kinds: coord
stop_reasons: object_step_role
```

Head groups:

```text
harmful = 20:14, 20:3, 27:14, 27:15
repair  = 27:2, 27:3, 24:2, 24:9
```

Run roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v2_desc_first_strict_harmful_combined_dirproj_regions3_scales012_steps6_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v2_desc_first_strict_harmful_combined_fullvec_regions3_scales012_steps6_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v2_desc_first_strict_repair_combined_dirproj_regions3_scales012_steps6_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v2_desc_first_strict_repair_combined_fullvec_regions3_scales012_steps6_v1
```

The compact bridge reducer aligns each scale 1 or 2 row to its same-condition
scale 0 baseline and writes:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation_bridge_summary/v1_harmful_repair_fullvec_dirproj_regions3_scales012_steps6
```

Bridge summary:

```text
schema_version: 1
record_count: 416
run_summaries: 4
phase summary rows: 64
slot summary rows: 88
top changed examples: 80
```

Run counters:

```text
harmful direction_projection_subtract: rows 915, continuations 180, final stops im_end 24 / intervention_skipped 24 / max_steps 132
harmful full_vector_subtract:         rows 921, continuations 180, final stops im_end 22 / intervention_skipped 24 / max_steps 134
repair direction_projection_subtract: rows 915, continuations 180, final stops im_end 24 / intervention_skipped 24 / max_steps 132
repair full_vector_subtract:          rows 915, continuations 180, final stops im_end 24 / intervention_skipped 24 / max_steps 132
```

`current_partial_coords` has expected skipped interventions on x1 states
because no prior coordinate exists inside the current object span yet.

## Main Finding

The cleanest causal signal is not final termination within six steps. It is the
first coordinate-token basin.

On later slots (`x2/y2`, n=8 per condition), removing harmful value routed from
current partial coordinate or box-span state moves the next coordinate token
substantially more than removing the broader pre-prefix non-image context:

```text
harmful direction_projection_subtract, current_partial_coords, scale 1:
  changed_first_token_rate 0.375
  baseline_target_match_rate 0.25 -> target_match_rate 0.0
  mean absolute delta error +2.875
  changed_final_stop_rate 0.0

harmful direction_projection_subtract, current_partial_box_span, scale 2:
  changed_first_token_rate 0.5
  baseline_target_match_rate 0.25 -> target_match_rate 0.0
  mean absolute delta error +1.875
  changed_final_stop_rate 0.0

harmful direction_projection_subtract, pre_prefix_non_image_context, scale 1:
  changed_first_token_rate 0.125
  baseline_target_match_rate 0.25 -> target_match_rate 0.25
  mean absolute delta error +0.125
  changed_final_stop_rate 0.0
```

The full-vector subtraction mode is a stronger stress test and can become
catastrophic on the same later-slot region:

```text
harmful full_vector_subtract, current_partial_box_span, scale 2, all slots:
  changed_first_token_rate 0.75
  baseline_target_match_rate 0.25 -> target_match_rate 0.20
  mean absolute delta error +51.35
  changed_final_stop_rate 0.05

harmful full_vector_subtract, current_partial_coords, scale 2, all available slots:
  changed_first_token_rate 0.833333
  baseline_target_match_rate 0.416667 -> target_match_rate 0.083333
  mean absolute delta error +79.833333
  changed_final_stop_rate 0.083333

harmful full_vector_subtract, pre_prefix_non_image_context, scale 2, all slots:
  changed_first_token_rate 0.20
  baseline_target_match_rate 0.25 -> target_match_rate 0.35
  mean absolute delta error +0.80
  changed_final_stop_rate 0.0

harmful full_vector_subtract, current_partial_coords, scale 2:
  changed_first_token_rate 1.0
  baseline_target_match_rate 0.25 -> target_match_rate 0.0
  mean absolute delta error +118.5
  changed_final_stop_rate 0.125

harmful full_vector_subtract, current_partial_box_span, scale 2:
  changed_first_token_rate 1.0
  baseline_target_match_rate 0.25 -> target_match_rate 0.0
  mean absolute delta error +121.5
  changed_final_stop_rate 0.125

harmful full_vector_subtract, pre_prefix_non_image_context, scale 2:
  changed_first_token_rate 0.25
  baseline_target_match_rate 0.25 -> target_match_rate 0.25
  mean absolute delta error +1.25
  changed_final_stop_rate 0.0
```

Concrete examples:

```text
source 36 / image 3255 / object 14 / x2:
  target 456, baseline 456
  harmful direction-projection current_partial_coords scale 1 -> 447
  delta error +9, final stop im_end -> im_end

source 123 / image 12670 / object 5 / x2:
  target 407, baseline 407
  harmful direction-projection current_partial_coords scale 1 -> 415
  delta error +8, final stop max_steps -> max_steps

source 36 / image 3255 / object 14 / y2:
  target 996, baseline 999
  harmful full-vector current_partial_coords scale 2 -> 234
  delta error +759, final stop im_end -> max_steps
```

This supports a causal bridge from the readout-only tomography: the harmful
group is not merely attending to current partial coord/box text state; at these
states, value carried through that route is sufficient for local coordinate
basin control.

The enormous full-vector scale-2 averages should not be read as a typical
per-row error. They are dominated by a small number of extreme later-slot basin
breaks, especially the `999 -> 234` example above. The safer conclusion is
that full-vector removal can catastrophically knock later coordinate basins off
course, while direction-projection removal gives the cleaner local readout.

## Repair Group Is Not A Simple Opposite

The repair group perturbations are smaller and mixed. On later slots under
full-vector subtraction:

```text
repair full_vector_subtract, current_partial_coords, scale 1:
  changed_first_token_rate 0.0
  baseline_target_match_rate 0.25 -> target_match_rate 0.25
  mean absolute delta error +0.0

repair full_vector_subtract, current_partial_coords, scale 2:
  changed_first_token_rate 0.25
  baseline_target_match_rate 0.25 -> target_match_rate 0.25
  mean absolute delta error +0.875

repair full_vector_subtract, current_partial_box_span, scale 2:
  changed_first_token_rate 0.375
  baseline_target_match_rate 0.25 -> target_match_rate 0.125
  mean absolute delta error +1.625
```

On onset `x1`, repair perturbations can sometimes move an initially wrong
coordinate toward the target:

```text
repair direction_projection_subtract, current_partial_box_span, scale 1, onset x1:
  changed_first_token_rate 0.25
  baseline_target_match_rate 0.0 -> target_match_rate 0.25
  mean delta error -0.75

source 145 / image 15254 / object 20 / x1:
  target 459, baseline 473
  repair full-vector current_partial_box_span scale 2 -> 460
  delta error -13
```

This keeps the slot-phase picture alive: x1 onset and later `x2/y2` coordinate
continuation are not the same mechanism. The repair group may participate in
object-onset anchoring or boundary calibration rather than simply being the
negative of the harmful local-coordinate route.

## Interpretation Boundary

Safe claim:

```text
For this strict desc_first duplicate/unmatched object-step panel, selected
harmful heads causally affect the first next-coordinate basin primarily through
current partial coord/box-span value routes. The effect is slot-phase dependent
and strongest on later slots where partial coordinate history already exists.
```

Claims to avoid:

```text
This does not prove all duplication is caused by these four heads.
This does not show visual perception is absent.
This does not establish that full-vector scale 2 is a natural intervention.
This does not make final stopping/termination the primary mechanism.
This does not identify individual head roles inside each combined group.
```

Important confounds:

```text
current_partial_coords is a subset of current_partial_box_span, so the two
regions are overlapping, not independent.

combined_all mode is useful for a first causal bridge but can hide head-level
opposition or redundancy.

scale 2 full-vector subtraction is intentionally strong and should be read as
a stress test. Direction-projection subtraction is the cleaner mechanistic
readout.

Scale 0 is a paired no-op baseline, not a random-control null. The current
region story is therefore contrastive against the broader pre-prefix region,
not yet proven against matched random heads, shuffled token regions, or
same-size off-region controls.

The panel is selected for duplicated/unmatched object-step states; it should
not be treated as a population estimate for all val200 objects.
```

## Next Experiments

1. Add matched random and off-region controls: random heads, shuffled current
   region token positions, and same-size non-current text regions.

2. Split the combined harmful and repair groups into individual heads and
   leave-one-out groups, especially `20:14`, `20:3`, `27:14`, `27:15`,
   `27:2`, `27:3`, `24:2`, and `24:9`.

3. Decompose full-vector versus projected-direction effects with residual-only,
   projection-only, and norm-matched projection interventions if the stage
   supports them or with a small extension if it does not.

4. Run bidirectional value-region exchange: remove harmful current-partial
   value, add repair/onset value, and test whether this shifts later-slot
   coordinates without creating onset drift.

5. Build a visual-side matched control for false negatives: compare missing
   objects under image-region value/readout probes versus language-side prefix
   guidance, using the same first-token basin metrics.

6. Extend the basin probe over selected pre-onset windows, not only the
   coordinate token itself, to test whether instability appears before the
   visible duplicate coordinate is emitted.
