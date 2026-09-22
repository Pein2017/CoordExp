# Train 418535 / Val 14439 X1-To-Y1 Binding Probe

Date: 2026-06-27

Scope: staged sample-base mechanistic probe for the sorted prefix-denoising SFT
checkpoint. This probe starts from the x1 findings in
`progress/diagnostics/2026-06-27_train418535_val14439_x1_visual_route_basin.md`
and asks whether repairing or forcing the first coordinate is enough to bind the
following y1 slot to the same object.

This is not a population metric pass. It is a narrow causal read on object-span
binding after the x1 slot.

Checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/detection_teacher_forcing/compact_object_box_closed_desc_first_prefix_denoising_kl_w0p05_k2_2b_base_sorted_bsz1x128_4epoch/compact-object-box-closed-desc-first-prefix-denoising-kl-w0p05-k2-2b-base-sorted-bsz1x128-4epoch/v5-20260623-133125/checkpoint-908
```

## Source Cases

Train case:

```text
sample_base_id=sorted_denoise:418535:person:obj8
image_id=418535
target_desc=person
target_bbox_bins=[795, 48, 806, 83]
baseline natural object span=[302, 54, 343, 182]
bridge-repaired-x1 object span=[795, 46, 794, 99]
```

Val case:

```text
sample_base_id=coco_000000014439
image_id=14439
target_desc=person
target_bbox_bins=[729, 38, 737, 85]
baseline natural object span=[0, 390, 27, 425]
bridge-repaired-x1 object span=[729, 385, 935, 671]
```

The x1 probe already showed that the target x1 can be recovered by a small
readout bridge. The new question is whether the subsequent y1 slot follows the
repaired object or remains in an old coordinate tail basin.

## Artifacts

Plan:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_to_y1_route_plan/v1_train418535_val14439_natural_vs_forced_x1/position_plan_rows.jsonl
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_to_y1_route_plan/v1_train418535_val14439_natural_vs_forced_x1/position_plan_summary.json
```

Route captures:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_to_y1_route/v1_train418535_natural_vs_forced_x1_layers16_20_24_27_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_to_y1_route/v1_val14439_natural_vs_forced_x1_layers16_20_24_27_gpu1
```

Group interventions:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_to_y1_route_group_intervention/v1_train418535_natural_vs_forced_x1_top2_bridge_alpha_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_to_y1_route_group_intervention/v1_val14439_natural_vs_forced_x1_top2_bridge_alpha_gpu1
```

Route protocol:

```text
probe_position=post_x1_pre_y1
target_next_coord_slot=y1
layers=16,20,24,27
regions=prompt_non_image,assistant_prefix_prior_objects,current_object_ref,current_box_start,current_forced_coords,current_partial_object,recent_16,all_prefix
ridge_k=8
ridge_border_bins=0,1,999
route rows: 1024 train + 1024 val, 0 errors
```

Group surgery protocol:

```text
group_metric=value_source_contribution_projection
top_n_per_group=2
positive_scale=2.0
negative_scale=0.0
readout_bridge_alphas=0.05,0.1,0.2
continuation_steps=4
group rows: 58 train + 58 val, 0 errors
```

## Plan Rows

| case | variant | forced x1 prefix | target y1 | contrast y1 | observed continuation |
|---|---|---:|---:|---:|---|
| train 418535 | natural_wrong_x1 | 302 | 48 | 54 | `[302,54,343,182]` |
| train 418535 | forced_target_x1 | 795 | 48 | 46 | `[795,46,794,99]` |
| val 14439 | natural_wrong_x1 | 0 | 38 | 390 | `[0,390,27,425]` |
| val 14439 | forced_target_x1 | 729 | 38 | 385 | `[729,385,935,671]` |

## Route Evidence

Mean route projection is measured in the target-y1 minus contrast-y1 direction.
Negative current-coordinate projection means the model is using the just-emitted
x1 token and its nearby object prefix as evidence against the target y1.

Train 418535, natural wrong x1:

```text
forced x1: [302]
target y1: 48
observed/ridge top1: 54
ridge top bins: [54, 56, 55, 52, 46, 49, 57, 53]
```

| region | mean projection |
|---|---:|
| current_forced_coords | -0.1833 |
| current_partial_object | -0.1991 |
| recent_16 | -0.2399 |
| prompt_non_image | -0.0295 |
| assistant_prefix_prior_objects | -0.0785 |
| all_prefix | -0.2683 |

Layer 27 means:

```text
current_forced_coords=-0.657
current_partial_object=-0.714
recent_16=-0.862
all_prefix=-0.977
```

Dominant negative sites:

```text
L27H2 current_forced_coords <|coord_302|> -4.5734
L27H2 current_partial_object -4.6463
L27H2 recent_16 -4.9300
L27H2 all_prefix -4.7879
```

Train 418535, forced target x1:

```text
forced x1: [795]
target y1: 48
observed/ridge top1: 46
ridge top bins: [46, 45, 55, 54, 60, 56, 52, 61]
```

| region | mean projection |
|---|---:|
| current_forced_coords | -0.1353 |
| current_partial_object | -0.1552 |
| recent_16 | -0.1385 |
| prompt_non_image | -0.1327 |
| assistant_prefix_prior_objects | -0.0108 |
| all_prefix | -0.2974 |

Layer 27 means:

```text
current_forced_coords=-0.427
current_partial_object=-0.477
recent_16=-0.363
all_prefix=-0.480
```

Dominant negative sites:

```text
L27H2 current_forced_coords <|coord_795|> -4.4030
L27H2 current_partial_object -4.7887
L27H2 recent_16 -5.3488
L27H2 all_prefix -5.3736
```

Interpretation for train: forcing the correct x1 moves the y1 basin from 54 to
near-target 46, but does not select the exact target 48. More importantly, the
correct x1 token itself becomes late anti-target-y1 evidence in L27H2. That is a
direct route-level version of the coordinate-slot basin hypothesis.

Val 14439, natural wrong x1:

```text
forced x1: [0]
target y1: 38
observed/ridge top1: 390
ridge top bins: [390, 382, 383, 391, 384, 385, 387, 353]
```

| region | mean projection |
|---|---:|
| current_forced_coords | -0.1871 |
| current_partial_object | -0.1813 |
| recent_16 | -0.1651 |
| prompt_non_image | -0.1223 |
| assistant_prefix_prior_objects | +0.0158 |
| all_prefix | -0.3119 |

Layer 27 means:

```text
current_forced_coords=-0.809
current_partial_object=-0.821
recent_16=-0.746
all_prefix=-0.661
```

Dominant negative sites:

```text
L27H15 current_forced_coords <|coord_0|> -13.3428
L27H15 current_partial_object -13.2092
L27H15 recent_16 -13.1128
L27H15 all_prefix -13.5225
```

Val 14439, forced target x1:

```text
forced x1: [729]
target y1: 38
observed/ridge top1: 385
ridge top bins: [385, 388, 390, 387, 383, 382, 391, 386]
```

| region | mean projection |
|---|---:|
| current_forced_coords | -0.1111 |
| current_partial_object | -0.1092 |
| recent_16 | -0.0980 |
| prompt_non_image | -0.0997 |
| assistant_prefix_prior_objects | +0.0222 |
| all_prefix | -0.2040 |

Layer 27 means:

```text
current_forced_coords=-0.461
current_partial_object=-0.484
recent_16=-0.392
all_prefix=-0.256
```

Dominant negative sites:

```text
L27H15 current_forced_coords <|coord_729|> -7.9559
L27H15 current_partial_object -7.8134
L27H15 recent_16 -7.6962
L27H15 all_prefix -7.5581
```

Interpretation for val: forcing the correct x1 only shifts the y1 winner from
390 to 385. It does not move the model into the top-object y basin at all. The
late L27H15 route uses the current x1 token as very strong anti-target evidence,
even when x1 is the correct target value.

## Group Surgery Evidence

| case | variant | direct rows | direct y1 flips | bridge rows | bridge y1 flips | best bridge behavior |
|---|---|---:|---:|---:|---:|---|
| train 418535 | natural_wrong_x1 | 7 | 0 | 21 | 10 | y1 flips to 48, tail remains `[343,182]` |
| train 418535 | forced_target_x1 | 7 | 0 | 21 | 3 | y1 flips to 48 only under template negative suppression, tail becomes `[809,97]` |
| val 14439 | natural_wrong_x1 | 7 | 0 | 21 | 0 | top1 jumps among 287/412/414/534/539, never 38 |
| val 14439 | forced_target_x1 | 7 | 0 | 21 | 0 | target rank can reach 2, but top1 stays 287 or other lower basin values |

Representative continuation outcomes:

```text
train natural y1 bridge repaired:
  <|coord_48|><|coord_343|><|coord_182|><|box_end|>
  label=first_token_only

train forced-x1 y1 bridge repaired:
  <|coord_48|><|coord_809|><|coord_97|><|box_end|>
  label=first_token_only

val natural y1 bridge best:
  <|coord_287|><|coord_27|><|coord_425|><|box_end|>
  label=first_token_not_repaired

val forced-x1 y1 bridge best:
  <|coord_287|><|coord_935|><|coord_671|><|box_end|>
  label=first_token_not_repaired
```

The group surgery makes the difference between the two cases more explicit:

- Train is locally repairable at y1, but still not span-bound. The first token
  can be forced into the target y1, while x2/y2 remain inherited from the old
  continuation basin or become malformed-near-target.
- Val is far-tail locked. Even aggressive readout bridge rows improve the
  target rank sharply, but the top1 remains in a lower-image y basin and the
  subsequent x2/y2 tail remains the old lower-object tail.

## Mechanistic Claim

The current best description is not "false negative equals no visual
perception." It is:

1. Target visual/route evidence exists and can repair x1 locally.
2. Correct x1 emission is not sufficient to bind y1/x2/y2 to the target object.
3. Coordinate tokens can act as basin keys. At the post-x1/pre-y1 state, the
   current forced coordinate token is routed through late heads as anti-target
   evidence for y1.
4. The model has at least two tail-binding regimes:
   - train-like near-tail regime: local y1 can be repaired, but the span tail is
     not reliably object-bound;
   - val-like far-tail regime: y1 remains trapped in a separate coordinate
     basin even when x1 is correct and target rank improves.

This supports the user's concern that the `<|coord_*|>` token family is not only
a coordinate readout vocabulary. It can become a causal hidden-state attractor
that shapes the next coordinate slot.

## Next Probe Direction

The most promising next step is a coordinate-token basin-key probe:

1. For train 418535, isolate L27H2 at `current_forced_coords` and
   `current_partial_object`.
2. For val 14439, isolate L27H15 at the same regions.
3. Compare three interventions at post-x1/pre-y1:
   - suppress only the late current-coordinate head contribution;
   - replace the current x1 coordinate-token residual with a neutral or
     neighbor-token residual while keeping the visual/image context fixed;
   - sweep x1 coordinate tokens near and far from the target to see whether y1
     follows object evidence, coordinate co-occurrence, or a learned global
     basin.
4. Read out y1 top1/rank and continue through x2/y2. A true object-binding
   repair must improve the whole `[y1,x2,y2]` tail, not only the first y1 token.

This would test whether the late coordinate token route is merely a symptom of
the wrong prefix, or whether it is a reusable coordinate-slot key that actively
selects the next basin.

## Verification

Checked artifact completeness and row counts:

```text
route capture: 1024 rows train, 1024 rows val, 0 errors
group intervention: 58 rows train, 58 rows val, 0 errors
plan rows: 4
```

Evidence scope: two high-value sample bases, sorted prefix-denoising SFT
checkpoint only. The mechanism claim should be replicated on additional
train/val bases and compared against at least one pure-CE baseline before being
promoted beyond a strong local hypothesis.
