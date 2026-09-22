# Image 19432 Same-Prefix Y2 Route Surgery Findings

Date: 2026-06-26

Scope: narrow y2-closure route and intervention study for COCO val image
`19432`, using the same duplicate-heavy random-denoise `chair` corridor from
`2026-06-26_image19432_sameprefix_tail_bridge_findings.md`.

The source prefix is not a normal native state for every checkpoint. Random is
the native corridor; sorted-denoise and pure CE are same-prefix controls. This
is intentional: the question is how different checkpoints process the same
late object-span prefix after x1/y1/x2 are already supplied.

## Source State

Targets:

```text
gt5 chair [351,122,458,348]
gt8 chair [537,122,651,348]
```

All probes operate at:

```text
continuation_application_mode=force_target_x1_y1_x2
target_next_coord_slot=y2
target_y2_bin=348
```

The forced-tail source rows are:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_tail_binding/v1_19432_random_corridor_sameprefix_random_gpu3
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_tail_binding/v1_19432_random_corridor_sameprefix_sorted_gpu4
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_tail_binding/v1_19432_random_corridor_sameprefix_purece_gpu5
```

Dry route plans, two receiver rows per model:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_plan/dryrun_19432_sameprefix_random
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_plan/dryrun_19432_sameprefix_sorted
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_plan/dryrun_19432_sameprefix_purece
```

## Natural Donor And Direct Bridge

Natural donor transplant, layer 27 `layer_output`, same image y2 donors:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant/v8_sameprefix_19432_random_layer27_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant/v8_sameprefix_19432_sorted_layer27_gpu1
```

Both completed with `error_count=0`, `plan_row_count=226`, `row_count=226`.

Summary:

```text
model   target repairs  rank improvements  donor-y2 transports
random  51              176                27
sorted  55              176                34
```

Per target:

```text
random gt5: baseline rank 6  -> 25 target repairs, 13 donor-y2 transports
random gt8: baseline rank 7  -> 26 target repairs, 14 donor-y2 transports

sorted gt5: baseline rank 5  -> 33 target repairs, 19 donor-y2 transports
sorted gt8: baseline rank 19 -> 22 target repairs, 17 donor-y2 transports
```

All exact repairs came from `target_bridge`, not raw `donor_delta`. The smallest
repair bridge scales were:

```text
random gt5: 0.1
random gt8: 0.1
sorted gt5: 0.1
sorted gt8: 0.2
```

This says natural object states can provide useful local y2 geometry, but the
raw donor delta alone is not the clean causal primitive. The effective primitive
is closer to a target-vs-current coordinate readout direction riding on a
compatible object-state vector.

Direct self bridge, layer 27 `layer_output`, no natural donors:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_factorized_donor_transplant/v4_sameprefix_19432_random_selfbridge_gpu2
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_factorized_donor_transplant/v4_sameprefix_19432_sorted_selfbridge_gpu3
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_factorized_donor_transplant/v4_sameprefix_19432_purece_selfbridge_gpu4
```

All completed with `error_count=0`, `plan_row_count=14`, `row_count=14`.

```text
model   exact target repairs / bridge rows  target-rank-improved rows
random  5 / 10                              10 / 10
sorted  4 / 10                              10 / 10
pureCE  2 / 10                               8 / 10
```

Per target:

```text
random gt5: 2 repairs, best rank 1
random gt8: 3 repairs, best rank 1

sorted gt5: 2 repairs, best rank 1
sorted gt8: 2 repairs, best rank 1

pureCE gt5: 2 repairs, best rank 1
pureCE gt8: 0 repairs, best rank 2
```

Interpretation: the y2 target direction is shared enough to move all three
checkpoints, including pure CE. Pure CE is less receptive, especially on gt8.
This looks like basin stiffness, not missing coordinate knowledge.

## Route Attribution

All-head y2 route probes:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route/v3_sameprefix_19432_random_layers22_27_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route/v3_sameprefix_19432_sorted_layers22_27_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route/v3_sameprefix_19432_purece_layers22_27_gpu2
```

All completed with `error_count=0`, `plan_row_count=2`, `row_count=1728`.

Mean value-source projection onto `target_y2 - generated_y2`:

```text
region                 random     sorted     pureCE
current_forced_coords  -0.208     +0.024     -0.157
current_partial_object -0.207     +0.025     -0.154
recent_16              -0.207     +0.027     -0.151
prompt_non_image       +0.061     +0.059     -0.044
all_prefix             -0.148     +0.088     -0.207
```

This is the cleanest cross-checkpoint route contrast so far. Under the same
object-prefix corridor, sorted-denoise is the only checkpoint whose local
forced-coordinate/recent-region value flow is positive on average toward target
y2. Random and pure CE both have negative local-region flow, but can still be
repaired by direct target bridges.

Top positive routes:

```text
random: layer23 head6 all_prefix/recent/current regions; layer26 head5 prompt/all_prefix
sorted: layer25 head12 prompt/all_prefix; layer26 head5 prompt/all_prefix; layer23 head6 current regions
pureCE: layer23 head6 current/recent/all_prefix; layer25 head9 current/recent/all_prefix
```

Top negative routes:

```text
random: layer27 head14 current_forced_coords/current_partial_object/recent/all_prefix
sorted: layer26 head15 prompt/all_prefix; weaker local-region suppression
pureCE: layer27 head15 current_forced_coords/current_partial_object/recent/all_prefix
```

The random and pure-CE suppressors are structurally parallel but not identical
heads: random uses layer27 head14, pureCE uses layer27 head15. Sorted seems to
have shifted the strongest suppressor earlier/to prompt-heavy layer26 head15,
while local y2 regions are already net positive.

## Focused Value-Region Interventions

Focused head/region interventions:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/value_region_intervention/v11_sameprefix_19432_random_route_heads_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/value_region_intervention/v11_sameprefix_19432_sorted_route_heads_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/value_region_intervention/v11_sameprefix_19432_purece_route_heads_gpu2
```

Heads:

```text
random: 23:6, 27:14
sorted: 23:6, 25:12, 26:15
pureCE: 23:6, 25:9, 27:15
```

Regions:

```text
current_forced_coords,current_partial_object,recent_16,prompt_non_image,all_prefix
```

Scales:

```text
0,0.5,1,2,4
```

All completed with `error_count=0`.

```text
model   rows  rank-improved  margin-improved  exact target top1 flips
random  100   34             46               0
sorted  150   25             43               0
pureCE  150   39             33               1
```

Continuation labels:

```text
random: near radius1=24, radius8=61, radius16=15, exact=0
sorted: near radius1=2, radius4=2, radius8=119, radius16=15, exact=0, not repaired=12
pureCE: near radius1=46, radius4=7, radius8=87, exact=1, not repaired=9
```

The exact pure-CE flip:

```text
model=pure_ce_sorted_natadj
receiver=gt8
head=layer27 head15
region=recent_16
scale=2
baseline top1=349, target rank=5
patched top1=348, target rank=1
continuation=y2_box_tail_complete
```

For random, ablation of layer27 head14 local regions consistently moves y2 from
the generated basin to the adjacent near-target basin:

```text
generated 342/343 -> patched top1 347, target rank 2, continuation radius1
```

For sorted, the same kind of local surgery mostly improves rank or near-radius
continuation without exact top1 flips. The sorted receiver already has a more
positive local route mean, so the remaining failure looks less like one
dominant local suppressor and more like a distributed basin competition.

For pure CE, the single exact flip after amplifying layer27 head15 recent-region
is notable because route attribution labels the same head as a strong negative
route in other regions/rows. The effect is sign- and region-sensitive: the head
is not simply "bad"; it can also be the movable actuator for escaping the 349
basin on gt8.

## Mechanistic Update

The same-prefix y2 corridor now supports a three-part mechanism:

1. The model retains coordinate knowledge in all checkpoints. A direct target
   bridge can move random, sorted, and pure CE toward or onto y2=348.
2. Prefix denoising changes local route balance. Sorted-denoise turns the
   current-coordinate/recent-region mean projection positive, while random and
   pure CE keep it negative under the same prefix.
3. Exact y2 emission is governed by a small late basin competition among nearby
   coordinate tokens, especially `342/343/347/348/349/350/354`. Interventions
   often land on `347`, one bin below target, before exact `348`.

This argues against the shallow story "the model cannot perceive the object" at
this late y2 boundary. At least here, the object corridor has enough visual and
language grounding to be repaired. The failure is an autoregressive coordinate
basin problem: once the prefix chooses the wrong local y2 attractor, ordinary
head/value surgery can move the basin but usually cannot fully flip exact top1
unless a target-bridge direction is injected or a very specific region/head
actuator is found.

Open question for the next slice: whether the same basin-stiffness pattern
already appears at x1/y1 onset before x2 is supplied. If yes, duplication is
mostly an onset/corridor-entry failure. If no, the failure is a late closure
competition that accumulates after partially correct grounding.
