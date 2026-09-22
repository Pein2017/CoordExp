# Path-Averaged Transported Projection Findings

Date: 2026-06-22

Worktree: `/data/CoordExp/.worktrees/autoregressive-binding-template-study`

## Question

The transported-readout projection probe showed that a live decoder-site
gradient can isolate enough of a post-box hidden delta to repair the next
descriptor token and generated descriptor span. But the paired hidden delta is
large, so a single local gradient at the failed active state could be misleading.

This pass adds a path-averaged transported readout direction:

```text
q_l = mean_tau grad_{component_l + tau * delta_l}(z_target - z_competitor)
tau = [0.0, 0.25, 0.5, 0.75, 1.0]
```

The goal is not to prove that descriptor repair exists. That is already visible.
The goal is to test whether the large active-to-baseline transition changes the
readout direction enough to explain the failure to recover the correct instance
geometry.

## Implementation

Added component projection bases:

```text
path_avg_transported_desc_target_minus_boundary_variant_top
orthogonal_to_path_avg_transported_desc_target_minus_boundary_variant_top
```

The path-average helper replays the prefix at several points along the
component delta, captures the selected decoder component, computes the
descriptor target-vs-boundary-variant margin gradient, and averages those
gradients before projecting the causal patch delta.

Runtime metadata records:

```text
component_patch_projection_direction_source=path_averaged_transported_readout_gradient
component_patch_projection_path_average_tau_values
component_patch_projection_margin_objective_values
component_patch_projection_margin_objective_value
```

Code:

```text
src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py
tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
```

Narrow verification:

```text
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -k 'path_avg_transported_projection_rows or normalize_patch_component_projection_bases'
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -k 'path_avg or transported_readout or component_projection or projection_basis or projection_bases or causal_activation_patch'
python -m py_compile src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
```

Results:

```text
2 passed
75 passed
py_compile passed
```

## Tiny GPU Scope

Input selected rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_box_counterfactual_boundary_selection/v1_desc_first_strict_selected_descriptor_flips_v1/post_box_counterfactual_boundary_selected_rows.jsonl
```

Pair config:

```text
configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
```

Common settings:

```text
stage=trajectory-hidden-causal-activation-patch-continuation
max_rows=2
case_count=1
state_row_count=2
row_count=18 per run
patch_component_sites=layer_input
patch_direction_bases=paired_post_box_baseline_minus_current
patch_component_projection_bases=[
  transported_desc_target_minus_boundary_variant_top,
  orthogonal_to_transported_desc_target_minus_boundary_variant_top,
  path_avg_transported_desc_target_minus_boundary_variant_top,
  orthogonal_to_path_avg_transported_desc_target_minus_boundary_variant_top
]
patch_strengths=[1.0]
patch_continuation_application_modes=[first_step]
target_next_kinds=[desc]
max_new_tokens=32
```

Artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_continuation_probe/post_box_layer_input_pathavg_transport_m4_m1_smoke_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_continuation_probe/post_box_layer_input_pathavg_transport_m8_m1_smoke_v1
```

This is a one-case backpack descriptor-basin flip smoke, not a validation run.

## Result

Both runs realized all transported and path-averaged projection bases.

Aggregate summaries:

| source -> target | rows | exact desc | different desc | mean target IoU | mean nearest same-desc GT IoU | emitted basin labels |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| -4 -> -1 | 18 | 8 | 10 | 0.000 | 0.398 | completed_box_basin=4, other_gt_basin=1, same_desc_gt_basin=4, unmatched_valid=9 |
| -8 -> -1 | 18 | 8 | 10 | 0.053 | 0.350 | completed_box_basin=3, other_gt_basin=1, same_desc_gt_basin=4, target_bbox_overlap=1, unmatched_valid=9 |

The `-8 -> -1` target-overlap row is a baseline-role interpolation sanity case:

```text
role=baseline
patch_label=interpolate_source_to_target_alpha_0p75
desc=backpack
bbox=[420,165,491,320]
target IoU=0.945
```

It is not evidence that the flip-row component/projection rescue recovered the
selected target geometry.

Flip-row behavior:

| source -> target | patch | generated desc | emitted basin | target IoU | nearest same-desc GT IoU | projection norm | cosine |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| -4 -> -1 | baseline no patch | person | unmatched_valid | 0.000 | null | null | null |
| -4 -> -1 | full layer_input delta | backpack | same_desc_gt_basin | 0.000 | 0.440 | null | null |
| -4 -> -1 | transported projection | backpack | same_desc_gt_basin | 0.000 | 0.423 | 34.618 | 0.208 |
| -4 -> -1 | transported orthogonal | person | unmatched_valid | 0.000 | null | 163.189 | 0.208 |
| -4 -> -1 | path-avg transported projection | backpack | same_desc_gt_basin | 0.000 | 0.423 | 34.827 | 0.209 |
| -4 -> -1 | path-avg transported orthogonal | person | unmatched_valid | 0.000 | null | 163.145 | 0.209 |
| -8 -> -1 | baseline no patch | person | unmatched_valid | 0.000 | null | null | null |
| -8 -> -1 | full layer_input delta | backpack | same_desc_gt_basin | 0.000 | 0.423 | null | null |
| -8 -> -1 | transported projection | backpack | same_desc_gt_basin | 0.000 | 0.580 | 18.855 | 0.206 |
| -8 -> -1 | transported orthogonal | person | unmatched_valid | 0.000 | null | 89.339 | 0.206 |
| -8 -> -1 | path-avg transported projection | backpack | same_desc_gt_basin | 0.000 | 0.423 | 17.304 | 0.190 |
| -8 -> -1 | path-avg transported orthogonal | person | unmatched_valid | 0.000 | null | 89.652 | 0.190 |

Path-average margin trace for both source depths:

```text
tau: [0.0, 0.25, 0.5, 0.75, 1.0]
target-minus-competitor objective: [-1.0, -0.375, 0.0, 0.5, 0.875]
```

Generated flip-row exact-desc boxes stayed in the same wrong backpack basin,
for example:

```text
[874,103,903,255]
[862,103,903,260]
```

The selected row target box remains:

```text
[420,165,491,329]
```

## Mechanistic Update

Path-averaging does not materially change the story for this smoke case.

The descriptor margin changes smoothly along the hidden-delta path, and the
path-averaged readout axis remains close enough to the single-point transported
axis to recover the same descriptor behavior. But neither axis carries the
selected instance geometry. Descriptor rescue and object-instance rescue are
therefore separable in the current probe family.

The strongest current statement is:

1. A late decoder-site readout direction can repair the next descriptor under a
   post-box hidden-state patch.
2. That repaired descriptor is grounded enough to land in a real same-class
   visual basin.
3. It is not bound to the selected row target instance.
4. The missing variable is not simply local descriptor-readout curvature along
   the active-to-baseline hidden path.

This narrows the failure from "descriptor not available" to "descriptor
available but instance pointer/geometry basin not co-transported by the
descriptor readout direction."

## Revised Next Directions

Promote now:

1. **Instance-pointer and geometry-basin bases.** Build projection directions
   from target-vs-wrong-same-desc box evidence, not descriptor logits alone.
   Candidate bases: row-specific bbox coord-token surfaces, target-vs-nearest
   same-desc GT basin margins, and nearest-previous/completed-box basin margins.
2. **Immediate geometry-basis falsifier.** Before any broader sweep, rerun this
   same post-box continuation smoke with the existing bases:

   ```text
   target_bbox_tokens_minus_completed_box_tokens
   orthogonal_to_target_bbox_tokens_minus_completed_box_tokens
   ```

   Keep transported descriptor projection only as a control. If this late
   `layer_input` target-bbox direction still emits `same_desc_gt_basin` or
   `unmatched_valid` with target IoU `0`, stop spending effort on descriptor
   readout variants for this case and move to coordinate destination-basin
   readouts.
3. **Coordinate destination-basin readout.** If the geometry-basis falsifier
   misses, explicitly include emitted wrong-basin coordinates as probe bins and
   compare target-bin versus wrong-bin rank movement. This decides whether the
   generated geometry is an attractor transition rather than merely failed
   descriptor conditioning.
4. **Descriptor/geometry mediation split.** Repeat transported and path-averaged
   projection over `layer_input`, `self_attn`, and `mlp`, but score outcomes by
   emitted basin label first and descriptor exactness second. The key question
   is whether a submodule carries the missing instance pointer while another
   submodule only translates it into descriptor logits.
5. **Narrow formation-time map across slots.** Probe only the positions where
   the hypothesis can change: descriptor onset, first box coordinate, box close,
   post-box boundary, and next-object onset, with basin outcomes required. Do
   not run a generic layer sweep that merely rediscovers the late descriptor
   axis.
6. **Small causal training perturbations only after basis separation.** If a
   geometry-basin basis is found, use a tiny targeted objective to strengthen or
   weaken that basis. Avoid training on descriptor rescue alone, because this
   smoke shows it can improve text while preserving wrong-instance binding.

Demote now:

```text
single-point gradient curvature as the main explanation of wrong-basin rescue
descriptor-token rescue as evidence of object rescue
static output-embedding cosine as a semantic information percentage
continuing larger descriptor-only path-average sweeps before geometry bases exist
all-steps patching as natural boundary unfolding evidence
false-negative guidance bridge as the immediate next step before emitted-basin taxonomy is stable
large cross-case low-rank transfer before target-vs-wrong-basin readout separates on a tiny panel
```

Fast falsifiers:

1. If a target-vs-wrong-same-desc geometry-basin basis repairs target IoU without
   descriptor repair, descriptor and geometry routes are partially independent.
2. If every geometry-basin basis that improves target IoU also destroys
   descriptor exactness, the model may use competing row-state attractors rather
   than a clean compositional descriptor-plus-box binding.
3. If no slot-local basis predicts emitted-basin movement before box close, the
   instance pointer may be distributed across prefix history or visual-token
   attention rather than encoded as a low-rank residual direction.
