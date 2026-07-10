# Image 2157 Prefix-Denoising Simplex Findings

Date: 2026-06-26

Scope: selected CoordExp readout-only coordinate-token geometry and competitor
simplex probes for COCO val image `2157`. This is a sample-base mechanism
slice, not a population metric.

Correction after tail-binding audit: the sorted-denoise `knife` row discussed
below is a false-positive/state-entry span on an FN-pressure image, not a clean
missing-object row. Treat this note's `false_negative_guidance` wording as the
source selection family, not the final object-level diagnosis. See:

```text
progress/diagnostics/2026-06-26_image2157_contextual_tail_binding_findings.md
```

## Why This Image

I selected image `2157` after the image-16228 microscope because it is a
non-person wine-glass/cup/knife sample base with both false-negative guidance
and duplication-anchor evidence. It is a useful test of whether the previous
coord0/neighbor-basin story generalizes, or whether false negatives can instead
come from fragile object-span continuation after a recoverable first coordinate.

The source row panel is:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/selected_position_subset/v1_broad_image_coverage_unpatched/selected_position_rows.jsonl
```

The four selected `pre_x1` rows are:

```text
sorted_denoise  wine glass  duplicate_anchor_basin   target coord3
sorted_denoise  knife       false_negative_guidance  target coord374
random_denoise  wine glass  duplicate_anchor_basin   target coord0
random_denoise  cup         false_negative_guidance  target coord3
```

## Artifacts

Coordinate-token geometry probes:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coord_token_geometry_probe/v12_2157_sorted_prex1_fn_anchor_simplex_gpu3
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coord_token_geometry_probe/v13_2157_random_prex1_fn_anchor_simplex_gpu4
```

Both runs completed with `error_count=0` and `plan_row_count=2`.

Competitor-simplex reduction:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coord_competitor_simplex_reduce/v2_2157_sorted_random_prex1_fn_anchor_simplex
```

## Simplex Summary

The reducer labels all four states as `single_top1_escape`, unlike the
image-16228 `coord133` states, which required `top8` competitor removal after
neighbor snapping.

```text
model           object      target  observed top1  target rank  coord0 rank  top1 flip  top8 flip
random_denoise  cup              3              0            3           1      0.05       0.1
random_denoise  wine glass       0              0            1           1      0.0        0.0
sorted_denoise  knife          374            414            5         415      0.005      0.005
sorted_denoise  wine glass       3              0            3           1      0.05       0.1
```

This is a different basin family from image 16228:

- Low-coordinate wine-glass/cup states are classic coord0-anchor competition:
  target `coord3` is already rank 3, while coord0 is rank 1.
- The random-denoise wine-glass state with target `coord0` is already exactly
  in the coord0 basin. This is not a hidden false-negative coordinate; it is a
  visible duplication-anchor emission.
- The sorted-denoise knife state is not coord0-trapped. It has target
  `coord374` already at rank 5, top1 at `coord414`, and coord0 only at rank
  415. Its x1 entry is close and cheap to recover.

## Knife State

The sorted-denoise knife state is the most diagnostic row:

```text
state:
  model: sorted_denoise
  generated object: knife
  target coord: 374
  observed top1 coord: 414
  observed target rank: 5
  coord0 rank: 415
  target adapter-head logit delta: 0.0846
  coord0 adapter-head logit delta: 2.8504

observed-top1 surgery:
  antagonist: coord414
  alpha=0.001 -> top1 coord414, target rank 5
  alpha=0.005 -> top1 coord374, target rank 1
```

At `alpha=0`, the top coordinate bins are:

```text
coord414  distance 40  logit 16.4719
coord407  distance 33  logit 16.4373
coord427  distance 53  logit 16.4157
coord624  distance 250 logit 16.3585
coord374  distance 0   logit 16.3494
```

At `alpha=0.005`, the target wins:

```text
coord374  distance 0   logit 16.4852
coord624  distance 250 logit 16.4781
coord407  distance 33  logit 16.4069
coord427  distance 53  logit 16.3691
coord414  distance 40  logit 16.3310
```

Interpretation: the model has a usable x1 coordinate representation for the
knife. The first coordinate is behind a thin competitor barrier, not absent.
This aligns with the earlier activation-patch evidence for the same source
state: patches can repair the first token (`<|coord_374|>`) at small alpha, but
the continuation remains `first_token_only` and does not become tail-coherent.

## Mechanism Update

Image 2157 supports a different state-entry and contextual-guidance origin than
image 16228:

1. For the sorted-denoise knife, the visual/coordinate entry signal is present
   enough to make x1 easily recoverable.
2. The sorted-denoise knife row is better framed as downstream tail coupling
   after a false-positive state entry: the model can be nudged to start the
   span at a nearby coordinate, but object-level binding is not guaranteed.
3. Low-coordinate cup/wine-glass states expose a separate coord0-anchor family:
   target `coord3` has strong adapter support and rank 3, but coord0's adapter
   logit is stronger and top1.
4. The image-16228 `coord133` case remains a multi-competitor/neighbor-snap
   basin. Image 2157 is mostly single-competitor escape. This gives us a useful
   split for future sample-base triage.

## Next

For 2157, the next high-value probe should not spend more effort on final x1
rank. The better question is x1-to-tail binding: after x1 is forced or patched
to `coord374`, which layer/site fails to propagate the object identity and
remaining box coordinates? The promising handles are multi-slot patching and
tail-slot tensor flow for the sorted knife state, compared against the
random-denoise cup/wine-glass coord3 low-coordinate anchor states.
