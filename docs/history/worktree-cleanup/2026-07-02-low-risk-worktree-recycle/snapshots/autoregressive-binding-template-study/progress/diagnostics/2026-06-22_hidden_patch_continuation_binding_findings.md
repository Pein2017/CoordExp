# Hidden Patch Continuation Binding Findings

Date: 2026-06-22

Worktree: `/data/CoordExp/.worktrees/autoregressive-binding-template-study`

Relevant commits:

- `7fe86968` - `Add hidden patch continuation probe`
- `4e69f4c6` - `Clarify hidden patch continuation semantics`
- `db30f785` - `Fix hidden patch continuation report formatting`

## Question

The prior layer-input projection probe showed that the post-box descriptor rescue
rotates across late decoder sites:

- m4/m3: mostly descriptor-orthogonal residual repair
- m2: mixed descriptor and descriptor-orthogonal repair
- m1: descriptor-axis projection becomes sufficient for next-token rescue

The missing question was whether this immediate next-token rescue unfolds into a
coherent generated object span, and whether it carries the correct instance
geometry.

## Implementation

Added stage:

```text
trajectory-hidden-causal-activation-patch-continuation
```

The stage applies the same causal hidden/component patches used by
`trajectory-hidden-causal-activation-patch`, but runs `model.generate` after the
selected prefix and parses the first generated object span after
`<|object_ref_start|>`.

Key continuation fields:

```text
continuation_application_mode
patch_position_policy
patch_application_count
patch_prefill_application_count
patch_decode_application_count
generated_object_parse_status
generated_desc_exact_match_target
generated_bbox_xyxy
generated_bbox_iou_target
generated_bbox_iou_completed_box
```

`first_step` patches only the prompt-prefill boundary forward. `all_steps`
patches the last token of every generate forward, so it is a sustained
decode-step steering diagnostic, not a one-shot boundary unfolding condition.

Verification:

```text
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -k 'trajectory_hidden_causal_activation_patch_continuation'
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -k 'continuation or causal_activation_patch'
python -m py_compile src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
```

Results:

```text
11 passed
109 passed
py_compile passed
```

## Inputs

Selected post-box descriptor flip rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_box_counterfactual_boundary_selection/v1_desc_first_strict_selected_descriptor_flips_v1/post_box_counterfactual_boundary_selected_rows.jsonl
```

The selected set has 7 rows over two source images:

- `image_id=12670`, `source_line_idx=123`, target descriptor `backpack`
- `image_id=3255`, `source_line_idx=36`, target descriptor `snowboard`

Pair config:

```text
configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
```

## Artifacts

Smoke:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_continuation_probe/post_box_layer_input_m1_smoke2_v1
```

Four-site continuation panel:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_continuation_probe/post_box_layer_input_projection_site_m4_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_continuation_probe/post_box_layer_input_projection_site_m3_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_continuation_probe/post_box_layer_input_projection_site_m2_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_continuation_probe/post_box_layer_input_projection_site_m1_v1
```

Each full panel run produced:

```text
row_count=212
state_row_count=7
realized_direction_patch_count=128
continuation_application_modes=["first_step", "all_steps"]
patch_component_sites=["layer_input"]
patch_component_projection_bases=[
  "desc_target_minus_boundary_variant_top",
  "orthogonal_to_desc_target_minus_boundary_variant_top"
]
patch_strengths=[0.5,0.75,1.0,1.25]
max_new_tokens=32
```

## First-Step Result

For the flip rows, baseline and self-noop continuations emit the wrong
descriptor:

```text
backpack flips: person
snowboard flip: sk
```

Layer-input patches repair the descriptor and preserve row syntax:

| site | patch group | strength threshold for 3/3 flip rows exact-desc | complete-valid first objects | target IoU > 0 |
|---|---|---:|---:|---:|
| m4 | full layer_input | 0.75 | 3/3 | 0/3 |
| m4 | desc projection | not reached by 1.25 | 3/3 | 0/3 |
| m4 | desc-orthogonal | 0.75 | 3/3 | 0/3 |
| m3 | full layer_input | 0.75 | 3/3 | 0/3 |
| m3 | desc projection | not fully reached by 1.25 | 3/3 | 0/3 |
| m3 | desc-orthogonal | 1.00 | 3/3 | 0/3 |
| m2 | full layer_input | 0.75 | 3/3 | 0/3 |
| m2 | desc projection | 1.25 | 3/3 | 0/3 |
| m2 | desc-orthogonal | not fully reached by 1.25 | 3/3 | 0/3 |
| m1 | full layer_input | 0.75 | 3/3 | 0/3 |
| m1 | desc projection | 1.00 | 3/3 | 0/3 |
| m1 | desc-orthogonal | not reached by 1.25 | 3/3 | 0/3 |

This reproduces the earlier next-token projection rotation, but at the level of
full generated object spans: early-late sites can repair the object descriptor
through descriptor-orthogonal residual state; the final late site repairs it
through the descriptor projection.

## Geometry Binding Split

The descriptor repair does not bind to the selected row target box.

Across first-step flip rows in the four-site panel:

```text
target_bbox IoU > 0: 0 rows
generated_object_parse_status: complete_valid for all first-step rows
```

For the backpack flip, the repaired descriptor usually emits boxes such as:

```text
[874,103,903,255]
[874,167,903,259]
[862,103,903,260]
```

These do not overlap the selected row target bbox:

```text
target_bbox=[420,165,491,329]
```

But they do overlap a different GT backpack in the same image:

```text
GT image_id=12670 object 2:
desc=backpack
bbox=[858,151,905,273]
coco_ann_id=1422414
IoU with generated boxes: 0.42-0.58
```

So the patch is not merely producing arbitrary coordinates. It repairs the
descriptor into a real same-class spatial basin, but not the requested instance
binding.

The snowboard case is less clean because the public GT objects in image `3255`
do not contain a `snowboard` descriptor. The selected target bbox is close to a
small lower-image object region, while repaired continuations emit boxes near
`[414,946,429,956]`, weakly overlapping a person/skis region. Treat this case as
supporting syntax/descriptor repair, not as strong same-class GT grounding.

## All-Steps Contrast

All four layer runs show the same mode split:

```text
first_step: 106/106 complete_valid first objects per site
all_steps: 55/106 incomplete_descriptor rows per site
```

Patch counts make the interpretation explicit:

```text
first_step: prefill applications=99, decode applications=0
all_steps: prefill applications=99, decode applications=3069
```

`all_steps` often creates descriptor loops or incomplete descriptor tails. It
should not be interpreted as natural boundary-state unfolding; it is useful as a
separate sustained-steering stress test.

## Main Mechanistic Update

The current evidence separates three mechanisms that had been easy to conflate:

1. Descriptor onset basin repair: late residual patches can flip the wrong next
   descriptor and make the generated first object syntactically complete.
2. Layerwise subspace conversion: the causal descriptor repair rotates from
   descriptor-orthogonal at m4/m3 toward descriptor-projection at m1.
3. Instance geometry binding: descriptor repair alone does not restore the
   selected object geometry. In the backpack case it rebinds to another real
   same-class spatial anchor.

This suggests that duplication/false-negative failures are not only wrong-class
or wrong-token onset failures. A model can recover the correct class descriptor
while still falling into a different instance/coordinate basin.

The better framing is not "missing final descriptor logit." The post-box row
state appears displaced into a wrong object-binding / routing / cursor decision
region, and late blocks translate that row state into descriptor, box,
repetition, or termination behavior.

## Revised Next Directions

The next cycles should therefore avoid treating the static descriptor direction
`E_target - E_competitor` as the privileged coordinate system.

P0: transported-readout projection.

For each selected row/layer, compute the local causal margin direction:

```text
q_l = grad_{h_l}(z_target - z_competitor)
```

Use VJP/backward through the actual readout surface, including final norm /
output projection geometry. Compare this with the static descriptor embedding
axis. Decompose the natural hidden delta:

```text
delta_l = h_correct - h_failed
delta_parallel_q
delta_orthogonal_q
```

Patch full delta, `parallel(q_l)`, and `orthogonal(q_l)`, and score both
next-token margin and continuation taxonomy.

P1: path-averaged margin direction.

Because the hidden deltas are large, a single-point gradient may be misleading.
Compute:

```text
qbar_l = mean_k grad margin(h_failed + tau_k * delta)
eta = |Delta margin - q_l dot delta| / (|Delta margin| + eps)
```

Patch `parallel(q_l)`, `orthogonal(q_l)`, `parallel(qbar_l)`,
`orthogonal(qbar_l)`, and full delta. This directly tests whether the earlier
"descriptor-orthogonal" rescue was truly object-state repair, or merely
orthogonal to the wrong static axis.

P2: continuation taxonomy as the default outcome surface.

Continue scoring:

1. first-token rescue only
2. descriptor-span rescue
3. descriptor + valid object box rescue
4. descriptor + target-overlapping box rescue
5. transition / next-row rescue
6. sustained duplicate basin / wrong emitted-object basin
7. malformed or incomplete span

The current evidence shows a concrete category split: descriptor + valid box
rescue can still be wrong emitted-object basin.

P3: component mediation with path patching.

Do not infer "MLP is wrong" from isolated MLP output deltas. Test block-level
paths:

```text
patch layer input, recompute attention + MLP
patch layer input, clamp attention output to failed/active
patch layer input, clamp MLP output to failed/active
clamp both
```

This asks whether MLP is a wrong-basin amplifier or a conditional calibration
response to the patched attention/state.

P4: formation-time causal map.

The current evidence localizes a carrier at late post-box layer inputs; it does
not localize origin. Build a layer x token-position map over descriptor onset,
descriptor end, coordinate slots, box close, post-box boundary, next descriptor,
and previous object spans. Early-token patching must replay the subsequent
prefix and refresh cache, otherwise it is only local readout.

P5: low-rank and cross-case transport only after P0-P4.

Expand from these two semantic flip cases into duplication, low-recall,
extreme-box, termination, and strict competitor cohorts. SVD the paired deltas,
patch rank-r projections, and test cross-case transfer:

```text
same class cross-image
different class
same failure type
same image different object
desc-first vs geometry-first / checkpoint crossing
```

Interpretation guardrails:

- do not claim universal binding pulse yet;
- do not claim strict attractor basin without closed-loop continuation support;
- do not claim missing visual information without candidate-object probing;
- do not treat static descriptor cosine as semantic-information percentage;
- distinguish descriptor readout repair, object-binding state repair, and
  coverage-transition repair.
