# Component Basin Label Findings

Date: 2026-06-22

Worktree: `/data/CoordExp/.worktrees/autoregressive-binding-template-study`

## Question

The transported readout projection smoke showed that late `layer_input`
projection can repair the descriptor, but not the selected target geometry. The
next question was whether the repair lives in an isolated block component
(`self_attn` or `mlp`) or in the incoming residual/cursor state, and whether the
generated box belongs to a recognizable emitted-object basin.

## Implementation

Added emitted-basin labeling for
`trajectory-hidden-causal-activation-patch-continuation` rows.

New row fields include:

```text
generated_emitted_basin_label
generated_target_bbox_overlap
generated_completed_box_overlap
generated_nearest_gt_idx
generated_nearest_gt_desc
generated_nearest_gt_bbox
generated_nearest_gt_iou
generated_nearest_same_desc_gt_idx
generated_nearest_same_desc_gt_iou
```

Label precedence:

```text
invalid_or_incomplete
target_bbox_overlap
completed_box_basin
same_desc_gt_basin
other_gt_basin
unmatched_valid
```

The continuation writer now loads the pair-config GT JSONL and labels emitted
basins before writing rows/summaries. This makes the emitted-basin taxonomy a
default artifact surface rather than a manual post-hoc notebook step.

Verification:

```text
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -k 'emitted_basins or continuation_summary_counts_parse_statuses'
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -k 'emitted_basins or continuation or causal_activation_patch'
python -m py_compile src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
```

Results:

```text
3 passed
112 passed
py_compile passed
```

## GPU Artifacts

Common settings:

```text
stage=trajectory-hidden-causal-activation-patch-continuation
max_rows=2
case_count=1
state_row_count=2
row_count=28
patch_component_sites=layer_input,self_attn,mlp
patch_direction_bases=paired_post_box_baseline_minus_current
patch_component_projection_bases=[
  transported_desc_target_minus_boundary_variant_top,
  orthogonal_to_transported_desc_target_minus_boundary_variant_top,
  desc_target_minus_boundary_variant_top,
  orthogonal_to_desc_target_minus_boundary_variant_top
]
patch_continuation_application_modes=first_step
target_next_kinds=desc
max_new_tokens=32
```

Artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_continuation_probe/post_box_transport_static_components_m4_m1_smoke_v2
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_continuation_probe/post_box_transport_static_components_m8_m1_smoke_v2
```

Both runs realized all requested projection bases with no skipped projection
reasons.

## Result

For the flip row, baseline continuation remains:

```text
desc=person
bbox=[269,105,323,252]
basin=unmatched_valid
target_iou=0
```

At `-4 -> -1`, only `layer_input` repairs the descriptor:

| site / component | projection | desc | basin | nearest same-desc IoU |
| --- | --- | --- | --- | ---: |
| layer_input full | none | backpack | same_desc_gt_basin | 0.440 |
| layer_input transported | transported | backpack | same_desc_gt_basin | 0.423 |
| layer_input transported orthogonal | transported orthogonal | person | unmatched_valid | null |
| layer_input static projection | static | person | unmatched_valid | null |
| layer_input static orthogonal | static orthogonal | backpack | same_desc_gt_basin | 0.440 |
| self_attn full/projections | all | person | unmatched_valid | null |
| mlp full/projections | all | person | unmatched_valid | null |

At `-8 -> -1`, the same localization holds:

| site / component | projection | desc | basin | nearest same-desc IoU |
| --- | --- | --- | --- | ---: |
| layer_input full | none | backpack | same_desc_gt_basin | 0.423 |
| layer_input transported | transported | backpack | same_desc_gt_basin | 0.580 |
| layer_input transported orthogonal | transported orthogonal | person | unmatched_valid | null |
| layer_input static projection | static | person | unmatched_valid | null |
| layer_input static orthogonal | static orthogonal | backpack | same_desc_gt_basin | 0.423 |
| self_attn full/projections | all | person | unmatched_valid | null |
| mlp full/projections | all | person | unmatched_valid | null |

Aggregate basin counts:

```text
m4->m1: completed_box_basin=4, other_gt_basin=1, same_desc_gt_basin=4, unmatched_valid=19
m8->m1: completed_box_basin=3, other_gt_basin=1, same_desc_gt_basin=4, target_bbox_overlap=1, unmatched_valid=19
```

The lone `target_bbox_overlap` row in the `m8->m1` run is a baseline-role
interpolation sanity row:

```text
role=baseline
patch_label=interpolate_source_to_target_alpha_0p75
bbox=[420,165,491,320]
target_iou=0.945
```

It is not evidence that a flip-row component/projection patch recovered the
selected target geometry.

## Mechanistic Update

This round strengthens the residual-cursor interpretation:

1. The descriptor-repair carrier is visible at the late block input but not as
   an isolated `self_attn` or `mlp` output intervention.
2. The transported readout projection at `layer_input` is sufficient for
   descriptor-span repair, but it enters a real wrong same-class GT basin.
3. Static descriptor projection remains misleading: the static projection does
   not repair the descriptor, while its orthogonal residual does.
4. The useful object-state seems cumulative in the residual stream before the
   block, not a single final submodule contribution.

The next best experimental move is therefore not another isolated late-component
sweep. It is a formation-time and path-mediation map:

```text
patch earlier object-span positions and replay the suffix
then clamp/recompute self_attn and mlp paths to see where the cursor/basin state forms
```

The criterion should remain emitted-basin taxonomy, not next-token descriptor
success alone.
