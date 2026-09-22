# Train 418535 / Val 14439 X1 Visual-Route Basin Probe

Date: 2026-06-27

Scope: paired sample-base mechanistic probe for the sorted prefix-denoising SFT
checkpoint, using existing visual-value evidence plus new x1 route attribution and
group surgery. This is not a population metric pass.

Checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/detection_teacher_forcing/compact_object_box_closed_desc_first_prefix_denoising_kl_w0p05_k2_2b_base_sorted_bsz1x128_4epoch/compact-object-box-closed-desc-first-prefix-denoising-kl-w0p05-k2-2b-base-sorted-bsz1x128-4epoch/v5-20260623-133125/checkpoint-908
```

Note: the actual artifact rows carry the full checkpoint path. The wrapped path
above is for human identification.

## Source Cases

Train case:

```text
sample_base_id=sorted_denoise:418535:person:obj8
image_id=418535
target_desc=person
target_bbox_bins=[795, 48, 806, 83]
slot=x1
target=795
baseline_attractor=302
```

Val case:

```text
sample_base_id=coco_000000014439
image_id=14439
target_desc=person
target_bbox_bins=[729, 38, 737, 85]
slot=x1
target=729
baseline_attractor=0
candidate_family=forced_object_start_bridge
object_state_family=forced_fn_after_boundary
```

Source visual-value artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v14_train_418535_token_union_l16_h12h13_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_token_sign_reduce/v1_train_418535_l16_h12h13_contrast0_302_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_component_recompose_probe/v1_train_418535_l16_h12h13_modes6_gpu0

/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v10_sorted_14439_candidate_bands_l16_h8h12h13_scales_contrast0_1_236_249_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_token_sign_reduce/v2_val_14439_l16_h8h12h13_contrast0_1_236_249
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_component_recompose_probe/v2_val_14439_l16_h8h12h13_modes6_gpu0
```

New x1 route artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_route_plan/v1_train418535_val14439_from_visual_value_plans/position_plan_rows.jsonl

/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_route/v1_train418535_from_visual_value_layers16_20_24_27_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_route/v1_val14439_from_visual_value_layers16_20_24_27_gpu1

/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_route_group_intervention/v1_train418535_from_visual_value_top2_bridge_alpha_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_route_group_intervention/v1_val14439_from_visual_value_top2_bridge_alpha_gpu1
```

Route capture protocol:

```text
layers=16,20,24,27
regions=prompt_non_image,assistant_prefix_prior_objects,current_object_ref,current_box_start,current_forced_coords,current_partial_object,recent_16,all_prefix
group_metric=value_source_contribution_projection
top_n_per_group=2
positive_scale=2.0
negative_scale=0.0
readout_bridge_alphas=0.05,0.1,0.2
continuation_steps=5
```

## Main Result

This pair gives a cleaner distinction than a metric-only train/val comparison:

1. The model has target-supporting evidence for the missing object coordinate.
2. Direct route surgery improves target rank but usually does not flip the
   winning coordinate.
3. A small readout bridge can make the target x1 rank-1/top-1.
4. Even after target x1 is emitted, the following coordinate tail is not bound
   to the target object box.

So the failure is not simply "the model cannot perceive the object." It is more
precisely a coordinate-slot and object-span binding failure: the first coordinate
slot can be repaired independently, while the y1/x2/y2 tail remains attracted to
another local coordinate manifold.

## Existing Visual-Value Evidence

Train 418535:

```text
baseline x1 target rank: 127
baseline x1 top1: 302
target logit: 14.0
target radius16 mass: 0.0415
sign rows: 59 supports_target, 22 supports_contrast, 10 mixed_supports_target, 20 mixed_supports_contrast
max visual token projection: 0.3039
strongest token: L16H12 visual_token_offset=123, prompt_index=231, supports_target
best component recomposition: rank 127 -> 86, top1 still 302
```

Val 14439:

```text
baseline x1 target rank: 441
baseline x1 top1: 0
target logit: 14.4375
target radius16 mass: 0.0131
sign rows: 65 supports_target, 56 supports_contrast, 4 mixed_supports_target, 2 mixed_supports_contrast
max visual token projection: 0.0460
strongest token: L16H8 visual_token_offset=313, prompt_index=421, supports_target
best component recomposition: rank 441 -> 321, top1 still 0
```

Interpretation: both cases contain target-supporting visual value evidence, but
the train case has a much stronger localized visual component and an interior
coordinate attractor, while the val case has weaker/distributed target support
and a border/edge attractor.

## Route Evidence

Train 418535, target 795 vs attractor 302:

| route region | mean projection |
|---|---:|
| current_box_start | -0.2095 |
| current_partial_object | -0.2007 |
| recent_16 | -0.2582 |
| prompt_non_image | +0.0253 |
| assistant_prefix_prior_objects | -0.1609 |
| all_prefix | -0.3221 |

Layer split:

| layer | current_box_start | current_partial_object | recent_16 | prompt_non_image | prior_objects | all_prefix |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | -0.001 | +0.007 | -0.001 | +0.003 | -0.025 | +0.045 |
| 20 | +0.007 | +0.001 | -0.004 | +0.012 | -0.016 | -0.007 |
| 24 | -0.016 | -0.025 | -0.006 | -0.042 | +0.046 | -0.025 |
| 27 | -0.828 | -0.785 | -1.022 | +0.128 | -0.648 | -1.301 |

Dominant train route sites:

```text
positive: L27H14 current_box_start +3.3253, current_partial_object +3.3063
positive: L24H10 current_box_start +1.7822, current_partial_object +1.7866
negative: L27H3 current_box_start -9.0068, current_partial_object -9.6726
negative: L27H13 current_box_start -3.0654, current_partial_object -3.0748
```

Val 14439, target 729 vs attractor 0:

| route region | mean projection |
|---|---:|
| current_box_start | +0.2406 |
| current_partial_object | +0.1886 |
| recent_16 | +0.1852 |
| prompt_non_image | -0.0438 |
| assistant_prefix_prior_objects | -0.1456 |
| all_prefix | -0.0111 |

Layer split:

| layer | current_box_start | current_partial_object | recent_16 | prompt_non_image | prior_objects | all_prefix |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | -0.009 | -0.023 | -0.025 | -0.019 | -0.073 | -0.080 |
| 20 | -0.045 | -0.127 | -0.131 | -0.044 | -0.031 | -0.240 |
| 24 | +0.057 | -0.023 | -0.036 | -0.003 | +0.006 | -0.039 |
| 27 | +0.960 | +0.927 | +0.932 | -0.108 | -0.485 | +0.314 |

Dominant val route sites:

```text
positive: L27H14 current_box_start +12.0460, current_partial_object +12.0439
positive: L27H2 current_box_start +2.9704, current_partial_object +2.9439
negative: L27H15 current_box_start -1.6450, current_partial_object -1.5001
negative: L27H13 prior box_start routes, especially assistant_prefix_prior_objects -4.7281
```

Interpretation:

- Train 418535 is not merely missing visual evidence. At late layers, the local
  current-object route itself becomes net anti-target because large negative
  current box-start heads overpower positive current heads.
- Val 14439 is different. The current box-start route is strongly target-positive
  at layer 27, but prior-object/global routes and the border coordinate basin
  still keep the observed top1 at 0.

This suggests at least two false-negative origins under the same checkpoint:

1. local current-marker route conflict, where the current object marker becomes
   anti-target in the target-vs-attractor direction;
2. global/prior basin dominance, where the current object marker carries target
   evidence but the readout stays pinned to a border or repeated context basin.

## Group Surgery Evidence

Direct group surgery rows:

| case | direct rows | x1 flips | best direct effect |
|---|---:|---:|---|
| train 418535 | 7 | 0 | current positive rank 127 -> 49, top1 still 302 |
| val 14439 | 7 | 0 | all signed combo rank 441 -> 225, top1 still 0 |

Direct surgery did not repair x1 in either case. It can move the target rank, but
the discrete winner remains in the old basin.

Readout-bridge rows:

| case | bridge rows | x1 flips | rank-1 target rows | alpha behavior |
|---|---:|---:|---:|---|
| train 418535 | 21 | 11 | 12 | alpha 0.1/0.2 repairs x1; alpha 0.05 does not |
| val 14439 | 21 | 5 | 7 | only alpha 0.2 repairs x1 |

Representative repaired first tokens:

Train:

```text
baseline continuation:
<|coord_302|><|coord_54|><|coord_343|><|coord_182|><|box_end|>

bridge-repaired x1 continuation:
<|coord_795|><|coord_46|><|coord_794|><|coord_99|><|box_end|>

target bbox:
[795, 48, 806, 83]
```

Val:

```text
baseline continuation:
<|coord_0|><|coord_390|><|coord_27|><|coord_425|><|box_end|>

bridge-repaired x1 continuation:
<|coord_729|><|coord_385|><|coord_935|><|coord_671|><|box_end|>

target bbox:
[729, 38, 737, 85]
```

The repaired x1 does not pull the remaining coordinates into the target object
box. The continuation labels are `first_token_only`, not full-object repair.
`patched_followup_prefix_match_count` stays 0 for the repaired rows.

## Mechanistic Interpretation

The current evidence supports a stronger mechanism than "false negatives happen
because target evidence is weak."

The model can carry or recover target-coordinate evidence at the x1 readout, but
coordinate emission is not a single scalar decision. The object span behaves like
a coupled basin:

```text
object-ref text + box_start marker + previous object manifold + coord-slot prior
    -> x1 winner
    -> tail manifold for y1/x2/y2/box_end
```

The x1 winner can be locally forced without moving the tail manifold. That means
the binding problem lives after, or at least beyond, the first coordinate slot.
The model can "name" the right x1 while still preparing the wrong object box.

This is especially important for interpreting prefix denoising. Prefix denoising
may reduce some exposure-bias symptoms while still leaving a hidden object-span
binding failure: the model knows enough to move one coordinate, but the remaining
autoregressive coordinate slots are attracted to a different box trajectory.

## Next Most Valuable Experiment

Run a staged x1-to-y1 probe:

1. Start from the same two pre-x1 prefixes.
2. Force or bridge only the target x1 token.
3. Probe y1 readout and route attribution before y1.
4. Compare:
   - target y1 rank/mass after natural baseline x1;
   - target y1 rank/mass after forced target x1;
   - route signs on `current_forced_coords`, `current_partial_object`, and
     `current_box_start`;
   - whether the bridge-created x1 state enters the target y1 manifold or keeps
     the wrong tail basin.

Prediction:

- Train 418535 may enter a near-target but malformed small-person tail
  `[795, 46, 794, 99]`, implying local person-neighborhood binding with invalid
  geometry.
- Val 14439 may enter a lower-object or prior-backpack/person tail
  `[729, 385, 935, 671]`, implying x1 correction without target-object binding.

If this prediction holds, the central mechanism becomes:

```text
false negative != no visual perception
false negative = target evidence fails to bind to the autoregressive object-span basin
```

The follow-up should focus on how `box_start`, prior `<|coord_*|>` tokens, and
current forced coordinates interact across layers 24-27 to select the next slot
basin.
