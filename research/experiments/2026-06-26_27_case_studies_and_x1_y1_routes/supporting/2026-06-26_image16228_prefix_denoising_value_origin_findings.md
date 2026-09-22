# Image 16228 Prefix-Denoising Value-Origin Findings

Date: 2026-06-26

Scope: sample-base-centered microscope on COCO val image `000000016228`,
using the same image/prefix family to compare prefix-denoising sorted SFT,
pure-CE sorted SFT, and random prefix-denoising emitted false-positive basins.
This is narrow mechanistic evidence, not a population metric.

## Sample Base

The selected image has a crowded same-class person cluster. The GT pair used
for the clean checkpoint comparison is:

- `object_idx=12`, `target_coord_bin=120`, bbox `[120,447,171,598]`,
  panel role `same_image_matched_control`.
- `object_idx=15`, `target_coord_bin=133`, bbox `[133,460,166,585]`,
  panel role `same_image_unmatched_fn`.

The emitted-basin random-denoise comparison uses two generated person false
positive x1 basins from the same image:

- generated object index 16, emitted bbox bins `[829,384,861,453]`,
  `target_coord_bin=829`.
- generated object index 51, emitted bbox bins `[862,385,899,437]`,
  `target_coord_bin=862`.

## Artifacts

Plan rows:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/sample_base_value_origin_plan/v1_16228_random_obj16_obj51_generated_region/sample_base_value_origin_plan_rows.jsonl
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/sample_base_value_origin_plan/v2_16228_sorted_gt_near_duplicate_pair/sample_base_value_origin_plan_rows.jsonl
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/sample_base_value_origin_plan/v3_16228_purece_sorted_gt_near_duplicate_pair/sample_base_value_origin_plan_rows.jsonl
```

Token/sign value-origin probes:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_token_probe/v6_16228_random_obj16_obj51_generated_region_l16_20_24_last_allheads_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_token_sign_reduce/v4_16228_random_obj16_obj51_generated_region_l16_20_24_last_allheads
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_token_probe/v7_16228_sorted_gt_neardup_l16_20_24_last_allheads_gpu2
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_token_sign_reduce/v5_16228_sorted_gt_neardup_l16_20_24_last_allheads
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_token_probe/v8_16228_purece_sorted_gt_neardup_l16_20_24_last_allheads_gpu3
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_token_sign_reduce/v6_16228_purece_sorted_gt_neardup_l16_20_24_last_allheads
```

Region-scale probes and fixed target-identity reductions:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v11_16228_sorted_gt_neardup_l16_h8h12h13_context_target_scales_gpu5
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v12_16228_purece_sorted_gt_neardup_l16_h8h12h13_context_target_scales_gpu6
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v13_16228_random_obj16_obj51_generated_region_l16_l24_selected_heads_scales_gpu7
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_scale_sweep_reduce/v14_16228_sorted_gt_neardup_l16_h8h12h13_fixed_target_identity
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_scale_sweep_reduce/v15_16228_purece_sorted_gt_neardup_l16_h8h12h13_fixed_target_identity
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_scale_sweep_reduce/v16_16228_sorted_vs_purece_gt_neardup_l16_h8h12h13_fixed_target_identity_combined
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_scale_sweep_reduce/v17_16228_random_obj16_obj51_generated_region_l16_l24_selected_heads_fixed_target_identity
```

Component recomposition probes and basin reductions:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_component_recompose_probe/v4_16228_random_obj16_obj51_generated_region_l16_l24_selected_heads_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_component_basin_reduce/v4_16228_random_obj16_obj51_generated_region_l16_l24_selected_heads
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_component_recompose_probe/v5_16228_sorted_gt_neardup_l16_h8h12h13_gpu2
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_component_basin_reduce/v5_16228_sorted_gt_neardup_l16_h8h12h13
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_component_recompose_probe/v6_16228_purece_sorted_gt_neardup_l16_h8h12h13_gpu4
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_component_basin_reduce/v6_16228_purece_sorted_gt_neardup_l16_h8h12h13
```

## Tooling Correction

The first combined scale-sweep reduction merged the two person targets because
the reducer grouped by model, image, desc, layer, head, and region, but did not
include target identity. That was invalid for this crowded same-class image.
The reducer now groups by target coordinate, target bbox key, and state key as
well. A regression test covers same image/desc/head/region with distinct
`target_coord_bin` values.

Verification:

```text
pytest -q tests/analysis/test_prefix_denoising_fn_visual_value_region_scale_sweep_reduce.py
```

Result: `4 passed`.

## Main Finding

For this image, the strongest evidence is not that prefix denoising improves
local object perception. It looks more like prefix denoising damps or recenters
coordinate-basin entry so the model fails to enter the nearby-person x-basin,
even though related L16 visual routes still exist.

The pure-CE sorted control is already much closer to the two GT x1 basins before
surgery:

- pure CE `coord120`: baseline target rank 15, top1 bin 105.
- pure CE `coord133`: baseline target rank 73, top1 bin 92.
- prefix-denoise `coord120`: baseline target rank 79, top1 bin 0.
- prefix-denoise `coord133`: baseline target rank 119, top1 bin 0.

Thus the prefix-denoise failure here is not just "missing visual evidence".
The relevant coordinate readout is pulled toward the global/low-coordinate
basin, especially top1 `coord0`.

## Token-Level Contrast

Both sorted models show L16 context-ring and target-region routes around h8,
h12, and h13, but the pure-CE route is stronger.

For the GT near-duplicate pair:

- prefix-denoise `coord120`: L16 h12 context-ring toward contrast 133 has
  absolute effect about `0.674`; pure CE has about `1.033`.
- prefix-denoise `coord133`: L16 h12 context-ring toward contrast 120 has
  absolute effect about `0.539`; pure CE has about `1.009`.
- prefix-denoise h8 context-ring to coord0 is weaker than pure CE
  (`0.361` vs `0.528` for coord120; `0.277` vs `0.457` for coord133).

Interpretation: prefix denoising did not invent the disambiguation circuit. It
appears to weaken or redistribute a pre-existing pure-CE L16 context-ring
coordinate route.

## Region-Scale Contrast

After fixing target identity grouping, the combined GT scale reduction has 36
rows, as expected: two target coordinates times three heads times three regions
times two models.

Prefix-denoise sorted:

- `coord133`, L16 h8 context-ring is suppressive for the target:
  scale0 delta `-6`, scale2 delta `+29`, baseline rank 119.
- `coord133`, L16 h8 target-full is weakly supportive:
  scale0 delta `+8`, scale2 delta `-10`.
- `coord120`, L16 h12 context-ring can improve rank from 79 to 66/66-ish in
  recomposition/scale views, but top1 remains `coord0`.

Pure-CE sorted:

- `coord133`, L16 h8 context-ring is strongly supportive:
  scale0 delta `+189`, scale2 delta `-10`.
- `coord133`, L16 h13 context-ring is suppressive:
  scale0 delta `-35`, scale2 delta `+14`.
- `coord120` is already rank 15 and most h8/h12/h13 region-scale rows are
  mostly neutral, consistent with being near the correct basin before surgery.

Interpretation: pure CE has a larger, more active, and more fragile local
coordinate circuit. Prefix denoising leaves a weaker circuit with much less
ability to move the final top1 away from coord0.

## Component Recomposition Contrast

Prefix-denoise sorted component reduction:

- reduced rows: 216.
- top1 movements: 2 changed-wrong-top1, 214 unchanged-wrong-top1.
- strongest improvement: h8 context-ring target-component/orthogonal variants
  improve `coord120` from rank 79 to 35, still with top1 `coord0`.
- strongest `coord133` improvement is smaller, rank 119 to about 102/108.

Pure-CE sorted component reduction:

- reduced rows: 216.
- top1 movements: 134 changed-wrong-top1, 82 unchanged-wrong-top1.
- h13 context-ring variants improve `coord133` from rank 73 to 38.
- h8 context-ring variants can also collapse `coord133` from rank 73 to
  260-266 and top1 `coord0`.

Interpretation: pure CE has more causal leverage and more opponent fragility.
Prefix denoising appears to suppress this leverage. That suppression may make
some trajectories less volatile, but on this false-negative slice it also keeps
the model from escaping the wrong global basin.

## Emitted False-Positive Basin

The random-denoise emitted false-positive basins are different from the GT
near-duplicate miss:

- generated `coord829`: baseline rank 9, top1 `coord0`.
- generated `coord862`: baseline rank 7, top1 `coord0`.
- component recomposition has 1080 reduced rows and zero top1 flips.
- L16 h8/h12 context-ring can improve `coord829` from rank 9 to 2, but top1
  remains `coord0`.
- L16 h8 far-background removal can degrade `coord862` badly, from rank 7 to
  about 121-124.
- L24 selected-head region rows are mostly neutral.

Interpretation: the emitted false-positive basin is already rank-close but
top1-locked. It does not look like a simple visual-region hallucination. It
looks more like a basin-competition state where the target-like coordinate is
available below the top but a global/low-coordinate attractor still wins.

## Mechanistic Hypothesis Update

Current best hypothesis from this sample:

1. Prefix denoising dampens pre-existing L16 coordinate disambiguation routes
   rather than creating a new object-local route.
2. False negatives can arise from coordinate-basin entry failure even when
   object-local/contextual visual evidence is present.
3. Duplication/false-positive emitted basins can be rank-close but top1-locked,
   suggesting that logit-rank availability and actual autoregressive emission
   are separated by a hard basin selection barrier.
4. Pure CE is more locally responsive but also more fragile: strong context
   routes can either rescue the target or amplify the wrong basin depending on
   component/region mode.

This refines the earlier L16-bottleneck picture: the critical question is not
only "which visual tokens support the coordinate?", but "which L16 routes can
move the final readout out of the global coord0 basin and into a local x1
basin?"

## Next Directions

Immediate high-value followups:

- Patch or ablate the L16 h8/h12/h13 context-ring routes between pure CE and
  prefix-denoise to test whether prefix-denoise mainly loses route magnitude,
  route sign, downstream readout compatibility, or basin escape.
- Add a top1-barrier probe: hold the rank-close emitted basin fixed and test
  whether small logit/readout edits can flip top1 without destroying the object
  span.
- Replicate the same microscope on at least one train image and one additional
  val image to separate checkpoint mechanics from image-specific crowd layout.

Interpretation boundary: all claims above are sample-state causal probes. They
are evidence for mechanism hypotheses and experiment selection, not final
population conclusions.

## Addendum: Coord0 Tensor-Flow Lens

After the value-origin pass, I ran `x1_hidden_vector_flow` on the same six
microscope states with antagonist coordinate bin `0` and layers
`0,4,8,12,16,20,24,last`. The join labels are:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_hidden_vector_flow_source_labels/v1_16228_value_origin_microscope
```

Tensor-flow artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_hidden_vector_flow/v6_16228_sorted_gt_neardup_coord0_l0_4_8_12_16_20_24_last_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_hidden_vector_flow/v7_16228_purece_gt_neardup_coord0_l0_4_8_12_16_20_24_last_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_hidden_vector_flow/v8_16228_random_generated_coord0_l0_4_8_12_16_20_24_last_gpu2
```

Each run selected two source states and produced 96 ok rows. The margin below
is the layer-output-state projection onto `W[target_coord] - W[coord0]`. It is
a directional lens, not a full coordinate-softmax/top1 proof.

```text
model                    coord  L0     L4     L8     L12    L16    L20    L24    last
sorted_denoise           120    2.10   0.38  -0.67  -1.10  -2.00  -0.13   1.17   0.17
sorted_denoise           133    0.01  -0.56  -2.02  -0.69  -0.94   0.98   1.27  -0.09
pure_ce_sorted_natadj    120    1.38  -0.16  -0.18  -0.37  -1.41   0.60   4.53   1.92
pure_ce_sorted_natadj    133   -0.52  -0.63  -1.05  -0.06  -1.05   1.03   2.42   0.72
random_denoise           829    3.59   3.57   2.03   2.68   1.37  -0.77   1.78   0.57
random_denoise           862    5.22   3.63   1.78   2.11   0.53  -2.06   2.32   0.84
```

This adds a sharper downstream-readout picture:

- Both sorted models pass through a mid-layer anti-target/coord0-favoring basin,
  especially around L8-L16.
- Pure CE recovers much more strongly at L24 and remains positive at the last
  captured layer for both GT targets.
- Prefix-denoise recovers at L24, but the last captured layer nearly cancels
  the target-vs-coord0 advantage (`coord120` only `0.17`, `coord133` `-0.09`).
- Random-denoise emitted false-positive basins have strong positive early
  target-vs-coord0 direction, a negative L20 interruption, and weak positive
  last-layer margins. This matches the earlier "rank-close but top1-locked"
  picture: target-vs-coord0 direction exists, but it is not enough to win the
  full coordinate basin competition.

The addendum shifts the hypothesis slightly: prefix denoising may not merely
dampen the L16 visual route. It may also weaken the late-layer recovery that
translates a local coordinate route into a durable final x1 basin. Pure CE is
more unstable, but it can re-amplify the correct x-basin late; prefix-denoise is
more muted and ends close to cancellation.

Updated high-value next step: patch or transport the late L24/last recovery
state, not only the L16 visual route. The decisive question is whether
prefix-denoise lacks a late amplification vector, or whether the vector exists
but is blocked by the final coordinate-basin competition.

## Addendum: Late Residual Readout Surgery

I then ran a direct readout-direction surgery on the same image-16228
microscope states, using antagonist coordinate bin `0` and configured layers
`16,20,24,last` with `alpha in {0.5,1,2,4}`. The artifacts are:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/residual_readout_surgery/v2_16228_sorted_gt_neardup_late_recovery_coord0_gpu3
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/residual_readout_surgery/v3_16228_purece_gt_neardup_late_recovery_coord0_gpu4
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/residual_readout_surgery/v4_16228_random_generated_late_recovery_coord0_gpu5
```

Important interpretation boundary: these rows score layer-specific readout-lens
states after an explicit residual/readout-space edit. The baseline ranks in
this table are therefore the layer-specific lens baselines for the probed
states, not necessarily the raw final next-token ranks from the earlier
region-mask probes.

Summary:

```text
model                    rows  target-rank improved  top1-distance improved  exact target flips
sorted_denoise             32                    32                      32                   8
pure_ce_sorted_natadj      32                    31                      27                  12
random_denoise             32                    32                      31                   9
```

Target-level exact flips:

```text
model                    target  flips/rows  best target rank  best top1 distance
sorted_denoise              120       8/16                 1                   0
sorted_denoise              133       0/16                 2                   1
pure_ce_sorted_natadj       120       4/16                 1                   0
pure_ce_sorted_natadj       133       8/16                 1                   0
random_denoise              829       5/16                 1                   0
random_denoise              862       4/16                 1                   0
```

The sharpest finding is the `coord133` failure mode. In prefix-denoise sorted,
the surgery strongly moves the state into the correct local coordinate
neighborhood but repeatedly snaps to neighboring bins instead of the exact
target. At layer 16 and `alpha=0.5`, top1 is `coord134` with coordinate-only
probability `0.667`, while `coord133` is rank 3 with probability `0.062`. At
the last captured layer and `alpha=0.5`, top1 is still `coord134` with
probability `0.317`; `coord133` reaches rank 2 but ties or nearly ties the
next neighbor at probability `0.192`. Increasing the surgery strength does not
fix this; it hardens the neighbor basin around `coord134`.

Pure CE sorted behaves differently on the same target. At layer 20 and
`alpha=0.5`, the edited state flips exactly to `coord133` with coordinate-only
probability `0.685`, while `coord134` and `coord132` sit behind it at
`0.153`. At the last captured layer and `alpha=0.5`, pure CE again selects
`coord133` as top1, although `coord134` is effectively tied. This means the
pure CE model is not merely "stronger" in a scalar sense; its coordinate basin
is shaped differently enough that the same target-direction edit can land on
the exact bin instead of the neighbor attractor.

The emitted false-positive basins from random-denoise are also informative.
`coord829` can be flipped exactly by layer-16 and layer-24 surgery, while
`coord862` flips exactly at the last captured layer. This supports the
rank-close/top1-locked interpretation: the model has a latent route into these
emitted coordinate basins, but ordinary decoding keeps them trapped behind a
dominant global or neighboring coordinate winner until a direct readout edit
pushes through the barrier.

This addendum makes the coordinate-token geometry question urgent. For the
missed GT `coord133`, prefix denoising does not look like a simple failure to
perceive a visual x-location. The edited state can reach the right local basin,
but the final coordinate surface prefers a nearby bin. The next probe should
therefore inspect the special coordinate-token embeddings/readout rows around
`132,133,134,135` and compare them to `120/121` plus the random emitted
basins `829/862`. The concrete question is whether prefix denoising reshapes
the coordinate-token basin so that locality is preserved but exact-bin
smoothness or local ordering is distorted.

## Addendum: Coordinate-Token Surface And Basin Geometry

I updated `coord_token_geometry_probe` so the route-row resolver can use
`target_next_token_text=<|coord_N|>` and `probe_position=pre_x1` when legacy
`target_y2_bin` fields are absent. The regression test is:

```text
tests/analysis/test_prefix_denoising_coord_token_geometry_probe.py::test_state_decomposition_rows_resolve_pre_x1_coord_from_token_and_bbox
```

Then I ran the selected-bin coordinate-token geometry probe on the same image
16228 microscope rows:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coord_token_geometry_probe/v9_16228_sorted_gt_neardup_prex1_neighbor_basin_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coord_token_geometry_probe/v10_16228_purece_gt_neardup_prex1_neighbor_basin_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coord_token_geometry_probe/v11_16228_random_generated_prex1_neighbor_basin_gpu2
```

All three runs completed with `error_count=0`. The two sorted/pure runs each
produced 2247 rows; the random-denoise run produced 2246 rows. The panel covers
surface geometry, state bin decomposition, adapter-head counterfactuals, and
effective-output readout surgery for selected bins around `0`, `120`, `133`,
`829`, and `862`.

Key decomposition table:

```text
model                    state target  target rank  coord0 rank  target adapter logit  coord0 adapter logit  ordinary top basin
sorted_denoise           120              87           1              0.648                 2.669             coord0
sorted_denoise           133             129           1              0.398                 2.611             coord0
pure_ce_sorted_natadj    120              14          67              1.678                 2.106             coord105
pure_ce_sorted_natadj    133              79          10              0.794                 2.477             coord100
random_denoise           829              13           1              0.475                 2.414             coord0
random_denoise           862              14           1              0.200                 2.568             coord0
```

This separates three phenomena that were previously tangled:

- The checked base-output coordinate rows are effectively identical between
  sorted-denoise and pure CE. For example, the selected base-output pair
  geometry for `coord133` vs `coord134` has the same distance and cosine in
  both runs (`euclidean=0.1998217`, `cosine=0.9885156`). The visible difference
  is therefore not explained by base coordinate-row drift on this panel.
- Pure CE has much stronger adapter-head support for the GT bins. For
  `coord133`, the adapter-head logit contribution is `0.794` in pure CE versus
  `0.398` in sorted-denoise; for `coord120`, it is `1.678` versus `0.648`.
- Coord0 is a strong readout competitor in all three families, but it only
  becomes dominant when the hidden/head state is still origin-aligned. Pure CE
  can move the matched-control state into a local x-basin (`coord105` top1 and
  target rank 14), while sorted-denoise remains at coord0 top1 for both GT
  states.

Adapter counterfactuals reinforce that the new coordinate-token surface matters
causally. Zeroing selected adapter-head offsets worsens the target ranks:

```text
model                    target  observed rank  zero-selected-adapter rank  target logit delta
sorted_denoise              120             87                         166              -0.648
sorted_denoise              133            129                         176              -0.398
pure_ce_sorted_natadj       120             14                          85              -1.678
pure_ce_sorted_natadj       133             79                         188              -0.794
random_denoise              829             13                         175              -0.475
random_denoise              862             14                          61              -0.200
```

So the model is not ignoring the coordinate-token adapter. The adapter helps,
but the help can be too weak relative to coord0/global competitors or can
prefer a neighboring bin inside the local basin.

The local adapter geometry differs from the effective output geometry. Around
`coord133`, pure CE's adapter-head offset surface is larger and rougher than
sorted-denoise:

```text
surface              pair       sorted distance/cosine   pure distance/cosine
adapter_head_offset  133-134    0.0334 / 0.8140          0.0736 / 0.6425
effective_output     133-134    0.2007 / 0.9885          0.2060 / 0.9879
```

This is consistent with the user's earlier warning: coordinate locality is
preserved at the effective-output level, but the learned adapter surface can be
locally rougher and more directional. The roughness is not automatically bad;
it may be the mechanism that lets pure CE escape coord0. But it also creates a
neighbor-attractor risk when the hidden state is not aligned precisely enough.

Effective-output surgery gives the cleanest basin picture. In the final-state
geometry probe, pushing away from only the observed top1 competitor shows
different thresholds:

```text
model                    target  observed-top1 antagonist  exact flip threshold
sorted_denoise              120  coord0                    alpha=0.2
pure_ce_sorted_natadj       120  coord105                  alpha=0.02
random_denoise              829  coord0                    alpha=0.02
random_denoise              862  coord0                    alpha=0.05
sorted_denoise              133  coord0                    no exact flip; snaps to coord134
pure_ce_sorted_natadj       133  coord100                  no exact flip in this handle; snaps to coord134
```

However, pushing away from the whole top-8 competitor centroid flips `coord133`
exactly for both sorted-denoise and pure CE at `alpha=0.1`. That is an important
mechanistic distinction: the exact `coord133` row is usable, but a single
top1-antagonist direction can enter the local basin through the wrong face and
land on `coord134`. Removing the multi-competitor basin as a whole lets the
same selected state land on the exact target.

Current mechanism update:

1. Prefix denoising does not simply destroy coordinate locality. The selected
   effective-output coordinate manifold remains smooth/local.
2. The failure is better described as weak or poorly synchronized basin entry:
   the hidden/head state stays too aligned with coord0/global competitors, and
   when forced toward the right local region it can cross into a neighboring
   coordinate attractor.
3. Pure CE is less muted. Its adapter offsets are larger and its hidden state
   is already closer to a local x-basin, so smaller readout edits can recover
   exact targets for some states.
4. Random-denoise false-positive coordinates are not hallucinated from nothing.
   They are rank-close latent basins with adapter support; coord0 blocks their
   ordinary emission until a small readout edit crosses the barrier.

Next probe direction: stop treating "coord0 vs target" as the only antagonist.
For false negatives and duplication bursts, record the full competitor simplex:
observed top1, local neighbors, repeated-object anchors, top-8 centroid, and
semantic-prefix competitors. The key origin mechanism may be the path through
that simplex: the same target vector can recover an object, snap to a nearby
duplicate basin, or terminate depending on which competitor face the state
crosses first.

## Addendum: Competitor Simplex Reducer

I added a reusable reducer for the geometry outputs:

```text
src/analysis/prefix_denoising_surgery_probing/coord_competitor_simplex_reduce.py
```

The first reduced artifact is:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coord_competitor_simplex_reduce/v1_16228_sorted_pure_random_prex1_neighbor_basin
```

It compresses the six image-16228 states into a state-level mechanism table:

```text
model                    target  observed top1  target rank  coord0 rank  pattern                                      top1 flip  top8 flip
pure_ce_sorted_natadj       133  100                    79          10    multi_competitor_escape_after_neighbor_snap  none       0.1
pure_ce_sorted_natadj       120  105                    14          67    single_top1_escape                           0.02       0.02
random_denoise              829  0                      13           1    single_top1_escape                           0.02       0.05
random_denoise              862  0                      14           1    single_top1_escape                           0.05       0.05
sorted_denoise              133  0                     129           1    multi_competitor_escape_after_neighbor_snap  none       0.1
sorted_denoise              120  0                      87           1    single_top1_escape                           0.2        0.05
```

This gives a concrete triage criterion for the next sample-base sweep:
prioritize states with `multi_competitor_escape_after_neighbor_snap`, high
coord0 rank, and weak target adapter support, because those states best expose
the difference between "object perceived but wrong basin path" and "object not
represented strongly enough."
