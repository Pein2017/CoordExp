# Image 19432 Same-Prefix Tail Bridge Findings

Date: 2026-06-26

Scope: same-prefix cross-checkpoint tail-binding probe for COCO val image
`19432`, using the known random-denoise duplicate-heavy `chair` corridor. This
is a narrow mechanistic slice. Non-random rows reuse a random-denoise generated
prefix and are same-prefix controls, not native rollout states.

## Why This Image

Image `19432` was selected after the image-2157 contextual-tail audit because
the representative panel labels it as the cleaner recoverable-tail substrate:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/representative_sample_base_panel/v3_current_deep_panel_terminal_dense/representative_sample_base_panel.jsonl
```

Existing reduction evidence already showed staged repair under random-denoise:

```text
gt8 chair: free 0.000 -> x1 0.632 -> x1+y1 0.996
gt5 chair: free 0.000 -> x1 0.584 -> x1+y1 0.900 -> x1+y1+x2 0.973
```

The new question here was whether that repair is a random-denoise-specific tail
skill, or whether the duplicate-heavy prefix itself creates a corridor that
other checkpoints can also complete once the right coordinates are supplied.

## Source Rows

Base random-denoise anchor rows:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_plan/v2_saturated_direction_coord0_gt_candidates/anchor_escape_rows.jsonl
```

I cloned the two image-19432 `generated_object_idx=12` rows for three models:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_sameprefix_control/v1_19432_random_corridor_sorted_purece
```

Targets:

```text
gt5  chair  [351,122,458,348]
gt8  chair  [537,122,651,348]
```

Original emitted duplicate-anchor bbox for both targets:

```text
[0,0,999,300]
```

Model rows:

```text
random_denoise        native random corridor
sorted_denoise        same-prefix control
pure_ce_sorted_natadj same-prefix weak control
```

## Probe Artifacts

Slot-scaffold hidden readout:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/slot_scaffold_readout/v1_19432_random_corridor_sameprefix_random_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/slot_scaffold_readout/v1_19432_random_corridor_sameprefix_sorted_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/slot_scaffold_readout/v1_19432_random_corridor_sameprefix_purece_gpu2
```

All three completed with `error_count=0`, `plan_row_count=8`,
`row_count=80`.

Forced-tail continuations:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_tail_binding/v1_19432_random_corridor_sameprefix_random_gpu3
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_tail_binding/v1_19432_random_corridor_sameprefix_sorted_gpu4
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_tail_binding/v1_19432_random_corridor_sameprefix_purece_gpu5
```

All three completed with `error_count=0`, `plan_row_count=10`,
`row_count=10`.

## Continuation Result

Mean target IoU by forced-prefix mode:

```text
model    pre_x1_free  force_x1  force_x1_y1  force_x1_y1_x2  full_box
random   0.000        0.608     0.948        0.976           1.000
sorted   0.000        0.578     0.891        0.973           1.000
pureCE   0.000        0.463     0.758        0.985           1.000
```

Per target:

```text
random gt5: free 0.000 -> x1 0.584 -> x1+y1 0.900 -> x1+y1+x2 0.973
random gt8: free 0.000 -> x1 0.632 -> x1+y1 0.996 -> x1+y1+x2 0.978

sorted gt5: free 0.000 -> x1 0.597 -> x1+y1 0.919 -> x1+y1+x2 0.973
sorted gt8: free 0.000 -> x1 0.560 -> x1+y1 0.863 -> x1+y1+x2 0.973

pureCE gt5: free 0.000 -> x1 0.617 -> x1+y1 0.900 -> x1+y1+x2 0.973
pureCE gt8: free 0.000 -> x1 0.309 -> x1+y1 0.617 -> x1+y1+x2 0.996
```

The free prefix behavior is identical in kind across all three checkpoints:
each emits a valid non-target low-left chair-like box:

```text
random: [0,0,73,270]
sorted: [0,0,66,290]
pureCE: [0,0,47,277]
```

Interpretation: the duplicate-heavy random prefix is strong enough to drive all
three checkpoints into a low-left/non-target basin. This is not a random-denoise
exclusive tail failure.

Once x1/y1/x2 are supplied, all three checkpoints complete almost exact target
boxes. Random and sorted recover earlier after x1+y1 than pure CE, but pure CE
still recovers once x2 is supplied.

## Hidden Readout

Best hidden-readout rows are weaker and noisier than final continuation. Many
rows are `tied` or low coordinate mass.

Best target ranks:

```text
model   gt  mode                         slot  target  layer  rank  winner      top peak
random  5   force_target_x1_pre_y1       y1       122  26       10  tied        98
random  5   force_target_x1_y1_pre_x2    x2       458  25       19  target      462
random  5   force_target_x1_y1_x2_pre_y2 y2       348  23        1  tied        348
random  8   force_target_x1_pre_y1       y1       122  26       87  tied        0
random  8   force_target_x1_y1_pre_x2    x2       651  27        1  target      651
random  8   force_target_x1_y1_x2_pre_y2 y2       348  23        1  tied        348

sorted  5   force_target_x1_pre_y1       y1       122  24        2  tied        123
sorted  5   force_target_x1_y1_pre_x2    x2       458  26       18  target      454
sorted  5   force_target_x1_y1_x2_pre_y2 y2       348  23        4  tied        347
sorted  8   force_target_x1_pre_y1       y1       122  24       21  tied        0
sorted  8   force_target_x1_y1_pre_x2    x2       651  27       20  background  666
sorted  8   force_target_x1_y1_x2_pre_y2 y2       348  23        3  tied        347

pureCE  5   force_target_x1_pre_y1       y1       122  25       12  tied        0
pureCE  5   force_target_x1_y1_pre_x2    x2       458  27       30  background  467
pureCE  5   force_target_x1_y1_x2_pre_y2 y2       348  23        4  tied        347
pureCE  8   force_target_x1_pre_y1       y1       122  25       39  tied        0
pureCE  8   force_target_x1_y1_pre_x2    x2       651  27       17  tied        717
pureCE  8   force_target_x1_y1_x2_pre_y2 y2       348  25        2  tied        347
```

Taxonomy counts:

```text
model   target  tied  invalid_low_coord_mass  background
random  3       43    32                      2
sorted  1       50    26                      3
pureCE  0       51    27                      2
```

Interpretation: a direct hidden readout underestimates downstream generative
recoverability. The model state often looks ambiguous/tied locally, but the
autoregressive decoder can still turn a sufficiently supplied coordinate prefix
into the right box.

## Mechanistic Update

This sample gives a clean separation:

1. The duplicate-heavy prefix is the dominant failure driver. It pushes all
   three checkpoints to a non-target low-left box under free continuation.
2. Tail completion is not a prefix-denoise-only capability. Random-denoise,
   sorted-denoise, and pure CE all complete near-exact target boxes after enough
   coordinates are supplied.
3. Checkpoint differences are in the threshold for corridor commitment:
   random/sorted often bind strongly after x1+y1, while pure CE may need x2 for
   the harder gt8 target.
4. Hidden readout and final generation are not equivalent. Readout can remain
   tied/background while continuation is highly target-bound after forced
   coordinates.

Current best phrase: duplication creates a prefix-conditioned coordinate
corridor. The tail skill exists across checkpoints, but free rollout fails
because the corridor's initial state is in the wrong basin. The deepest next
question is therefore not "can the model complete a box?" It is "what route
selects the first few coordinates that decide the corridor?"

## Next

Use existing 19432 route artifacts to localize the first committed coordinate
transition:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route/v2_ridge_attribution_19432_random_layers22_27_allheads_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/value_region_intervention/v10_ridge_flow_19432_random_heads23_27_regions_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant/v7_coord_response_atlas_19432_random_gpu3
```

Then run counterfactual route/head interventions under the same-prefix controls
only if the existing random-denoise route points to a compact component set.
