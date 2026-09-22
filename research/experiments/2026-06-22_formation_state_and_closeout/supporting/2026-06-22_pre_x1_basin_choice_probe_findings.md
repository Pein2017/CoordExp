# Pre-X1 Basin Choice Probe Findings

Date: 2026-06-22

Scope: tiny model-backed smoke over selected post-box descriptor-flip rows.
This is not a full validation set. It is a causal localization probe for the
question left open by the geometry-basis falsifier: whether the wrong
same-description coordinate basin is already selected before the first
coordinate token, or only after coordinate generation begins.

## Why This Probe

The previous score-traced continuation showed that descriptor-repairing patches
can still emit a wrong same-description box, and that the wrong coordinate bins
are already top-1 at the generated coordinate slots. That demoted late
descriptor-boundary coordinate side channels and promoted a sharper boundary:

```text
<|object_ref_start|>{target_desc}<|object_ref_end|><|box_start|>
```

At this prefix, the next token is x1. If a hidden delta from the clean baseline
descriptor-boundary state to the active flip state can move the x1 distribution
toward the target bin, then there is a separable pre-coordinate basin-choice
lever. If it cannot, the wrong object pointer is probably earlier, distributed,
or nonlinear.

## Implementation Added

New helper:

```text
src/analysis/autoregressive_binding_template_ablation/post_box_pre_x1_rows.py
scripts/analysis/run_autoregressive_binding_post_box_pre_x1_rows.py
tests/analysis/test_post_box_pre_x1_rows.py
```

The helper takes selected post-box descriptor-boundary rows and materializes
pre-x1 rows by appending the selected descriptor tokens, object-ref end, and
box-start marker. It preserves `post_box_boundary_case_id` and
`post_box_boundary_variant_role`, and sets:

```text
target_next_kind=coord
target_next_coord_slot=x1
target_next_coord_bin=target_bbox[0]
target_next_token_text=<|coord_x1|>
counterfactual_variant=post_box_pre_x1_{role}
case_id=case_id or post_box_boundary_case_id
```

This lets the existing `trajectory-hidden-causal-activation-patch` stage reuse
the `paired_post_box_baseline_minus_current` hidden-delta basis. Active flip
rows are selected by `counterfactual_variant=post_box_pre_x1_flip`, while the
baseline peer rows stay in the same input JSONL for paired lookup.

## Artifact Roots

Row materialization:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_box_pre_x1_row_builder/v1_desc_first_strict_selected_descriptor_flips_pre_x1_v1
```

Model-backed pre-x1 patch smoke:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_pre_x1_paired_baseline_minus_current_flip_m4_m1_smoke_v1
```

Post-hoc summaries:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_causal_patch_probe_support_analysis/post_box_pre_x1_paired_baseline_minus_current_flip_m4_m1_smoke_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_causal_patch_destination_basin_analysis/post_box_pre_x1_paired_baseline_minus_current_flip_m4_m1_smoke_v1
```

## Commands

Materialize pre-x1 rows:

```bash
python scripts/analysis/run_autoregressive_binding_post_box_pre_x1_rows.py \
  --selected-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_box_counterfactual_boundary_selection/v1_desc_first_strict_selected_descriptor_flips_v1/post_box_counterfactual_boundary_selected_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_box_pre_x1_row_builder/v1_desc_first_strict_selected_descriptor_flips_pre_x1_v1
```

Run the model-backed smoke:

```bash
python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage trajectory-hidden-causal-activation-patch \
  --prefix-greedy-trajectory-readout-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_box_pre_x1_row_builder/v1_desc_first_strict_selected_descriptor_flips_pre_x1_v1/post_box_pre_x1_rows.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_pre_x1_paired_baseline_minus_current_flip_m4_m1_smoke_v1 \
  --device cuda:0 \
  --allow-model-load \
  --hidden-layers=-4,-1 \
  --patch-strengths 0.25,0.5,1.0,2.0 \
  --patch-direction-bases paired_post_box_baseline_minus_current \
  --target-next-kinds coord \
  --counterfactual-variants post_box_pre_x1_flip \
  --probe-coord-bins 420,165,491,329,466,98,520,174,374,163,407,270,418,978,433,985,435,973,447,983,466,942,496,959
```

## Artifact Counters

Pre-x1 row builder:

```text
output_row_count=7
post_box_boundary_variant_role_counts={'baseline': 2, 'flip': 3, 'stable_counterfactual': 2}
counterfactual_variant_counts={'post_box_pre_x1_baseline': 2, 'post_box_pre_x1_flip': 3, 'post_box_pre_x1_stable_counterfactual': 2}
skipped_counts_by_reason={}
```

Model-backed smoke:

```text
row_count=30
state_row_count=3
case_count=2
source_hidden_layer_index=-4
target_hidden_layer_index=-1
direction_patch_status_counts={'realized': 30}
direction_patch_basis_key_counts={'paired_post_box_baseline_minus_current': 12}
probe_coord_bin_count=23
activation_patch_ran=true
training_ran=false
```

Post-hoc probe support:

```text
probe_support_row_count=690
strongest_row_count=30
any_offtarget_rank_le_10_rate=0.533333
any_offtarget_previsible_rank_le_10_rate=0.533333
```

Post-hoc destination basin:

```text
destination_basin_row_count=66
state_row_count=3
case_count=2
precursor_rank_le_5_before_exact_top1_count=6
exact_destination_top1_count=0
destination_basin_top1_count=1
```

## Key Smoke Table

The backpack case is `post_box_boundary_ec16feebe5d087b0`, target x1 bin 420.

| case | completed box variant | patch | target rank | target prob coord-only | top1 bin | distance |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| backpack | nearest_previous_same_desc_box | baseline_no_patch | 328 | 0.000619 | 874 | 454 |
| backpack | nearest_previous_same_desc_box | paired 0.5 | 107 | 0.002639 | 874 | 454 |
| backpack | nearest_previous_same_desc_box | paired 1.0 | 1 | 0.006080 | 420 | 0 |
| backpack | nearest_previous_same_desc_box | paired 2.0 | 5 | 0.008274 | 426 | 6 |
| backpack | next_generated_object_box | baseline_no_patch | 243 | 0.000919 | 874 | 454 |
| backpack | next_generated_object_box | paired 0.5 | 83 | 0.003042 | 874 | 454 |
| backpack | next_generated_object_box | paired 1.0 | 1 | 0.005591 | 387 | 33 |
| backpack | next_generated_object_box | paired 2.0 | 11 | 0.006967 | 426 | 6 |

The second case is `post_box_boundary_731c3fe5f9dce143`, target x1 bin 418.

| case | completed box variant | patch | target rank | target prob coord-only | top1 bin | distance |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| tiny object | previous_object_box | baseline_no_patch | 3 | 0.019166 | 414 | 4 |
| tiny object | previous_object_box | paired 0.5 | 5 | 0.016051 | 414 | 4 |
| tiny object | previous_object_box | paired 1.0 | 5 | 0.016628 | 414 | 4 |
| tiny object | previous_object_box | paired 2.0 | 8 | 0.013871 | 414 | 4 |

## Mechanistic Read

This is the first positive evidence that the wrong same-description coordinate
basin can be altered before x1 emission. On the backpack flip, the pre-x1
baseline-minus-current hidden delta is not merely descriptor repair; at strength
1.0 it makes the target x1 bin rank 1 for the nearest-previous-same-desc
counterfactual and also makes the target x1 rank 1 for the next-object-box
counterfactual, although the latter still has a nearby non-target top1 bin.

The same basis is not uniformly beneficial. On the tiny-object case, baseline
already has target rank 3 and top1 at nearby 414. The paired delta pushes target
rank down to 5/8 and leaves top1 at 414. This looks like a state-conditioned
basin-selection lever, not a globally corrective target-geometry direction.

Important distinction:

```text
pre-x1 basin can be shifted in at least one high-harm backpack state
but the shift is scale- and state-dependent
and strong patches can overshoot into a neighboring basin
```

This supports the roadmap pivot from aggregate template effects to local
coordinate-basin formation. It also warns against treating any hidden delta as a
universal rescue vector.

## Revised Next Directions

1. **Scale-local pre-x1 basin map.** Expand this exact pre-x1 probe over the
   selected descriptor-flip rows and layer pairs. Record target rank/prob,
   wrong-basin rank/prob, top1 bin, and overshoot bins. The immediate goal is to
   find whether the backpack rescue is a stable local window or a one-layer
   accident.

2. **Target-vs-wrong-same-desc coordinate basis.** The successful pre-x1 lever is
   a paired hidden delta, not an output-embedding target-bbox direction. Add a
   state-local target-vs-wrong-same-desc basis using known wrong bins
   (`420` vs `874/862` for the backpack case) and test whether it moves x1 with
   less overshoot than the broad paired hidden delta.

3. **Writer localization after a positive x1 handle.** Now that one pre-x1
   handle exists, localize which value/attention/head components write the
   target-vs-wrong shift. Use the backpack state first. The tiny-object state is
   a negative/control case because the same broad paired basis harms it.

4. **Boundary-gate mechanism remains separate.** Do not merge close-box rescue
   with coordinate-basin rescue. Boundary syntax patches can repair closure,
   while this pre-x1 probe is about choosing a spatial basin before the first
   coordinate.

5. **Cross-template/checkpoint transfer only after handles stabilize.** The next
   transfer target should be the same pre-x1 handle, not an aggregate metric.
   If the handle transfers across templates/checkpoints, it becomes a genuine
   mechanism candidate. If it does not, the mechanism is likely template- or
   checkpoint-specific.

## Stop/Demote For Now

```text
more late descriptor-gradient variants
more target-vs-completed-box output-embedding probes at post-box descriptor boundary
aggregate template comparisons without state-level basin traces
claims that wrong-basin emission is only downstream sampling
unconditional hidden-delta rescue narratives
```

## Falsifiers For The Next Round

If pre-x1 target-vs-wrong-same-desc bases fail to move target `420` above the
wrong `874/862` family without breaking the descriptor/schema, then the
successful paired-delta row may be broad state interpolation rather than a clean
coordinate-basin lever.

If component/head localization cannot reproduce even part of the pre-x1 rescue,
the basin choice may be distributed through a wider residual state or visual
prefix pathway.

If the same handle harms most states, then the deeper mechanism is likely not a
linear target-binding subspace but a nonlinear state-conditioned routing gate.

## Coordinate Surface Falsifier Follow-Up

After the positive pre-x1 paired-hidden-delta result, I tested whether a cleaner
output-embedding coordinate direction could reproduce the backpack rescue. This
uses existing bridge support, not new model code:

```text
coord_target_minus_bin:874
coord_target_minus_mean:874+862
```

The intended contrast is target x1 `420` versus the wrong-backpack x1 family
`874/862`.

I also fixed a small reproducibility footgun in the causal-patch stage while
running this: `--case-ids` was accepted by the shared CLI but was not wired into
`trajectory-hidden-causal-activation-patch`, so my first attempt silently ran all
three active flip rows. The stage now filters on either `case_id` or
`post_box_boundary_case_id`; the clean artifact below is case-filtered.

Case-filtered artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_pre_x1_coord_target_vs_wrong874_862_backpack_m4_m1_casefiltered_v1
```

Post-hoc summaries:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_causal_patch_probe_support_analysis/post_box_pre_x1_coord_target_vs_wrong874_862_backpack_m4_m1_casefiltered_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_causal_patch_destination_basin_analysis/post_box_pre_x1_coord_target_vs_wrong874_862_backpack_m4_m1_casefiltered_v1
```

Counters:

```text
row_count=32
state_row_count=2
case_count=1
case_ids=['post_box_boundary_ec16feebe5d087b0']
direction_patch_basis_key_counts={'coord_target_minus_bin:874': 10, 'coord_target_minus_mean:874+862': 10}
probe_support_row_count=448
destination_basin_row_count=52
exact_destination_top1_count=6
destination_basin_top1_count=7
```

Backpack results:

| completed box variant | basis | scale | target rank | target prob coord-only | top1 bin | distance | rank 874 | rank 862 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nearest_previous_same_desc_box | baseline | 0 | 328 | 0.000619 | 874 | 454 | 1 | 2 |
| nearest_previous_same_desc_box | coord_target_minus_bin:874 | 64 | 143 | 0.001490 | 862 | 442 | 3 | 1 |
| nearest_previous_same_desc_box | coord_target_minus_bin:874 | 128 | 47 | 0.003447 | 862 | 442 | 14 | 1 |
| nearest_previous_same_desc_box | coord_target_minus_mean:874+862 | 64 | 124 | 0.001755 | 874 | 454 | 1 | 3 |
| nearest_previous_same_desc_box | coord_target_minus_mean:874+862 | 128 | 30 | 0.004324 | 858 | 438 | 2 | 5 |
| next_generated_object_box | baseline | 0 | 243 | 0.000919 | 874 | 454 | 1 | 2 |
| next_generated_object_box | coord_target_minus_bin:874 | 64 | 99 | 0.002163 | 862 | 442 | 2 | 1 |
| next_generated_object_box | coord_target_minus_bin:874 | 128 | 22 | 0.004977 | 862 | 442 | 26 | 1 |
| next_generated_object_box | coord_target_minus_mean:874+862 | 64 | 85 | 0.002468 | 874 | 454 | 1 | 3 |
| next_generated_object_box | coord_target_minus_mean:874+862 | 128 | 5 | 0.006044 | 358 | 62 | 3 | 11 |

This is a useful negative result. The explicit output-embedding surface does
increase target `420` support, especially at large scale, but it does not
reproduce the earlier paired-hidden-delta rescue:

```text
paired baseline-minus-current, nearest_previous_same_desc_box, scale 1.0:
target rank 1, top1 420

coord_target_minus_bin/mean wrong-basin surfaces:
target rank improves but top1 remains 862/858/874 or overshoots to 358
```

Mechanistic update:

1. The positive paired-hidden-delta pre-x1 rescue is not explained by a simple
   target-coordinate output-embedding correction.
2. Suppressing one wrong bin can expose a neighboring wrong-basin ridge
   (`874 -> 862/858`) rather than selecting the target.
3. The useful signal likely includes richer state information: object pointer,
   descriptor/context binding, visual-anchor selection, or a nonlinear routing
   gate.
4. The next writer-localization pass should therefore use the paired hidden
   delta as the positive handle and decompose it, not replace it with a pure
   token-surface coordinate vector.

Revised next move:

```text
Use the paired-hidden-delta rescue state as the positive condition.
Measure which component/value/attention regions reproduce target-rank lift.
Use coord_target_minus_bin/mean as a negative or partial-control basis.
Track wrong-ridge substitution explicitly: 874 suppressed -> 862/858 exposed.
```

## Component Writer-Localization Follow-Up

The side-channel prompt reframed the next target correctly: this should not be
treated as a missing final descriptor logit. The live object is the post-box /
pre-x1 object-binding transition state: the row state has entered, or failed to
enter, the coordinate basin for the next object before the first coordinate is
emitted.

I therefore ran the smallest component localization probe on the positive
backpack handle:

```bash
python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage trajectory-hidden-causal-activation-patch \
  --prefix-greedy-trajectory-readout-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_box_pre_x1_row_builder/v1_desc_first_strict_selected_descriptor_flips_pre_x1_v1/post_box_pre_x1_rows.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_pre_x1_paired_component_sites_backpack_m4_m1_casefiltered_v1 \
  --device cuda:0 \
  --allow-model-load \
  --hidden-layers=-4,-1 \
  --patch-strengths 0.25,0.5,1.0,2.0 \
  --patch-direction-bases paired_post_box_baseline_minus_current \
  --patch-component-sites layer_input,self_attn,mlp \
  --target-next-kinds coord \
  --counterfactual-variants post_box_pre_x1_flip \
  --case-ids post_box_boundary_ec16feebe5d087b0 \
  --probe-coord-bins 420,165,491,329,466,98,520,174,374,163,407,270,874,862,858,426,387,358
```

Artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_pre_x1_paired_component_sites_backpack_m4_m1_casefiltered_v1
```

Post-hoc summaries:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_causal_patch_probe_support_analysis/post_box_pre_x1_paired_component_sites_backpack_m4_m1_casefiltered_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_causal_patch_destination_basin_analysis/post_box_pre_x1_paired_component_sites_backpack_m4_m1_casefiltered_v1
```

Counters:

```text
row_count=44
state_row_count=2
case_count=1
case_ids=['post_box_boundary_ec16feebe5d087b0']
patch_component_sites=['layer_input', 'self_attn', 'mlp']
counts_by_patch_space={
  'raw_decoder_layer_output': 20,
  'decoder_component_input:layer_input': 8,
  'decoder_component_output:self_attn': 8,
  'decoder_component_output:mlp': 8
}
probe_support_row_count=792
destination_basin_row_count=34
exact_destination_top1_count=8
destination_basin_top1_count=10
```

Best target-rank rows by patch family:

| completed variant | patch family | best scale | target rank | target p(coord) | top1 | rank874 | rank862 | rank858 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nearest_previous_same_desc_box | baseline | 0.0 | 328 | 0.000619 | 874 | 1 | 2 | 12 |
| nearest_previous_same_desc_box | full_layer_output_delta | 1.0 | 1 | 0.006080 | 420 | 24 | 103 | 45 |
| nearest_previous_same_desc_box | layer_input_delta | 1.0 | 1 | 0.005498 | 387 | 28 | 99 | 38 |
| nearest_previous_same_desc_box | self_attn_output_delta | 0.25 | 326 | 0.000630 | 874 | 1 | 2 | 9 |
| nearest_previous_same_desc_box | mlp_output_delta | 2.0 | 310 | 0.000445 | 893 | 9 | 12 | 58 |
| next_generated_object_box | baseline | 0.0 | 243 | 0.000919 | 874 | 1 | 2 | 13 |
| next_generated_object_box | full_layer_output_delta | 1.0 | 1 | 0.005591 | 387 | 27 | 109 | 49 |
| next_generated_object_box | layer_input_delta | 1.0 | 1 | 0.005617 | 387 | 28 | 108 | 50 |
| next_generated_object_box | self_attn_output_delta | 0.5 | 227 | 0.000951 | 874 | 1 | 2 | 8 |
| next_generated_object_box | mlp_output_delta | 0.25 | 227 | 0.000857 | 874 | 1 | 2 | 6 |

Mechanistic update:

1. The positive pre-x1 rescue is essentially a layer-input residual-state
   effect at this late block. The `layer_input` component patch reproduces the
   full hidden-delta target-rank rescue almost exactly for both active backpack
   variants.
2. The isolated `self_attn` output delta is not the writer of the rescue in
   this local intervention. It leaves the wrong same-description ridge at
   `874/862` and barely moves target rank.
3. The isolated `mlp` output delta is also not the writer. At large scale it can
   disturb the wrong ridge into another wrong bin (`893`) while leaving target
   far from top-1.
4. The distinction between rank rescue and exact emitted bin matters. For
   nearest-previous-same-desc, full layer-output delta at scale `1.0` gives
   exact top1 `420`; layer-input delta gives target rank 1 but top1 `387`.
   For next-generated-object-box, both full and layer-input deltas give target
   rank 1 with top1 `387`. So the layer-input carrier has most of the
   target-vs-wrong-family information, but exact-bin selection still depends on
   downstream transformation or local nonlinear geometry.

This demotes a simple "late self-attn writes the binding" or "late MLP writes
the binding" story. The better interpretation is:

```text
the useful binding/cursor state is already present at the late block input;
the block transforms it into the coordinate readout surface, and exact-bin
selection is partly decided by that downstream transformation rather than by
the isolated attention or MLP output delta alone.
```

## Critically Revised Near-Term Directions

1. **Immediate main line: formation-time layer-input map.** The next experiment
   should trace where the late `layer_input` carrier forms. Use the same
   backpack pre-x1 handle, but patch earlier meaningful token positions and
   replay the suffix so the later layer input is recomputed. The priority
   positions are descriptor onset, descriptor end, `object_ref_end`,
   `box_start`, pre-x1, post-x1, box close, and next-object onset.

2. **Path mediation, not isolated component blame.** The current component run
   says isolated late `self_attn` and `mlp` deltas are insufficient. The next
   path experiment should patch layer input and then clamp/recompute
   attention/MLP outputs, or use a small interventional Shapley over
   attention and MLP, to decide whether the block is a necessary transformer
   of an existing carrier rather than the original writer.

3. **Coordinate-family basin scoring.** The coordinate-output falsifier and
   this component run both expose family substitution (`874 -> 862/858/893`).
   Future readouts should score target `420` against a local wrong family, not
   just against a single wrong token. The local family should include observed
   exact and near-top bins from the artifact, e.g. `874,862,858,893,426,387`.

4. **Continuation taxonomy before broad transfer.** Rank-1 `420` at x1 is not
   automatically object-binding repair. The next continuation pass should
   classify descriptor span, full valid box, target-overlap box, transition to
   the next row, wrong-family continuation, and malformed span separately for
   full-layer and layer-input patches.

5. **Low-rank and cross-case only after regime labels.** The cases already show
   different regimes: backpack is layer-input steerable, tiny-object is harmed
   by the same broad delta, and other earlier coordinate cases include
   adapter-surface-local basin effects. A shared SVD now would probably average
   incompatible mechanisms. First label states by regime: layer-input
   steerable, descriptor-only repair, wrong-family substitution,
   local-simplex stubbornness, adapter-amplified cliff, and upstream
   weak-evidence/false-negative.

Deprioritize for the next few cycles:

```text
more static descriptor-axis accounting
more single wrong-bin coordinate directions as the main explanation
claims that isolated late attention or MLP output writes the binding state
broad template/checkpoint transfer before the pre-x1 basin labels stabilize
low-rank SVD over mixed regimes
```

## Continuation Taxonomy Follow-Up

The next gate was whether the pre-x1 rank/basin rescue actually repairs an
object span. I ran a continuation smoke using the same backpack case and the
same full/layer-input/self-attn/MLP component split, with both first-step-only
and all-steps patching.

While doing this, I found and fixed the same reproducibility footgun that had
already affected the non-continuation causal patch writer: the continuation
stage accepted the shared CLI `--case-ids` argument, but the writer did not pass
it into the selector or record it. The first continuation attempt therefore ran
three active states across two cases. The fixed `v2` artifact is the clean one.

Clean case-filtered artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_continuation_probe/post_box_pre_x1_paired_component_sites_backpack_m4_m1_casefiltered_continuation_v2
```

Command:

```bash
python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage trajectory-hidden-causal-activation-patch-continuation \
  --prefix-greedy-trajectory-readout-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_box_pre_x1_row_builder/v1_desc_first_strict_selected_descriptor_flips_pre_x1_v1/post_box_pre_x1_rows.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_continuation_probe/post_box_pre_x1_paired_component_sites_backpack_m4_m1_casefiltered_continuation_v2 \
  --device cuda:0 \
  --allow-model-load \
  --hidden-layers=-4,-1 \
  --patch-strengths 1.0 \
  --patch-direction-bases paired_post_box_baseline_minus_current \
  --patch-component-sites layer_input,self_attn,mlp \
  --patch-continuation-application-modes first_step,all_steps \
  --max-new-tokens 64 \
  --target-next-kinds coord \
  --counterfactual-variants post_box_pre_x1_flip \
  --case-ids post_box_boundary_ec16feebe5d087b0 \
  --probe-coord-bins 420,165,491,329,466,98,520,174,374,163,407,270,874,862,858,893,426,387,358
```

Bridge counters:

```text
row_count=40
state_row_count=2
case_count=1
case_ids=['post_box_boundary_ec16feebe5d087b0']
continuation_application_mode_counts={'first_step': 20, 'all_steps': 20}
generated_object_parse_status_counts={'complete_valid': 28, 'missing_object_ref_start': 12}
generated_desc_exact_match_target_counts={'true': 3, 'false': 37}
generated_emitted_basin_label_counts={
  'unmatched_valid': 26,
  'invalid_or_incomplete': 12,
  'same_desc_gt_basin': 1,
  'target_bbox_overlap': 1
}
mean_generated_bbox_iou_target=0.003729695245
mean_generated_bbox_iou_completed_box=0.003729695245
```

Important caveat: for this pre-x1 continuation prefix, the current object span
is already open and the prefix ends at `<|box_start|>`. The bridge
`generated_*` fields search for the next `<|object_ref_start|>`, so those fields
usually describe the following object, not the current object's box. I added a
small taxonomy reducer to classify both views explicitly:

```text
src/analysis/autoregressive_binding_template_ablation/continuation_taxonomy.py
scripts/analysis/run_autoregressive_binding_continuation_taxonomy.py
```

Taxonomy artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/continuation_taxonomy/post_box_pre_x1_paired_component_sites_backpack_m4_m1_casefiltered_continuation_v2
```

Taxonomy counters:

```text
row_count=40
case_count=1
state_count=2
primary_span_source_counts={'ongoing_box_from_prefix': 40}
taxonomy_label_counts={'descriptor_only': 12, 'descriptor_valid_box': 27, 'target_overlap': 1}
taxonomy_label_counts_by_mode={
  'all_steps': {'descriptor_only': 12, 'descriptor_valid_box': 8},
  'first_step': {'descriptor_valid_box': 19, 'target_overlap': 1}
}
taxonomy_label_counts_by_patch_family={
  'baseline': {'descriptor_valid_box': 4},
  'full_output_delta': {'descriptor_only': 2, 'descriptor_valid_box': 1, 'target_overlap': 1},
  'interpolation': {'descriptor_only': 8, 'descriptor_valid_box': 8},
  'layer_input_delta': {'descriptor_valid_box': 4},
  'mlp_delta': {'descriptor_valid_box': 4},
  'self_attn_delta': {'descriptor_valid_box': 4},
  'self_noop': {'descriptor_only': 2, 'descriptor_valid_box': 2}
}
generated_object_taxonomy_label_counts={
  'descriptor_valid_box': 1,
  'malformed_or_incomplete': 12,
  'target_overlap': 1,
  'wrong_desc_valid': 25,
  'wrong_same_desc_basin': 1
}
first_token_match_counts={'False': 38, 'True': 2}
desc_exact_match_counts={'True': 40}
object_complete_and_valid_counts={'False': 12, 'True': 28}
target_overlap_counts={'False': 39, 'True': 1}
mean_primary_bbox_iou_target=0.0102144306
mean_primary_bbox_iou_completed_box=0.001397019691
```

Key rows after separating current-open-box and next-object labels:

| completed variant | mode | patch | first token | current/open bbox | current IoU target | current label | next-object label | next desc | next bbox |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| nearest_previous_same_desc_box | first_step | baseline | 874 | [874,103,903,255] | 0.0 | descriptor_valid_box | wrong_desc_valid | person | [0,105,47,259] |
| nearest_previous_same_desc_box | first_step | full_output_delta | 420 | [420,129,480,259] | 0.408577 | target_overlap | wrong_desc_valid | person | [269,131,333,218] |
| nearest_previous_same_desc_box | all_steps | full_output_delta | 420 | [420,387,387,387] | 0.0 | descriptor_only | malformed_or_incomplete | none | none |
| nearest_previous_same_desc_box | first_step | layer_input_delta | 387 | [387,129,415,168] | 0.0 | descriptor_valid_box | wrong_same_desc_basin | backpack | [874,131,903,259] |
| nearest_previous_same_desc_box | all_steps | layer_input_delta | 387 | [387,100,415,165] | 0.0 | descriptor_valid_box | descriptor_valid_box | backpack | [387,104,406,163] |
| next_generated_object_box | first_step | baseline | 874 | [874,167,903,259] | 0.0 | descriptor_valid_box | wrong_desc_valid | person | [0,170,60,329] |
| next_generated_object_box | first_step | full_output_delta | 387 | [387,169,415,222] | 0.0 | descriptor_valid_box | wrong_desc_valid | person | [0,173,61,327] |
| next_generated_object_box | first_step | layer_input_delta | 387 | [387,169,415,222] | 0.0 | descriptor_valid_box | wrong_desc_valid | person | [0,173,61,327] |
| next_generated_object_box | all_steps | layer_input_delta | 387 | [387,197,419,270] | 0.0 | descriptor_valid_box | target_overlap | backpack | [420,201,436,277] |

Interpretation:

1. First-token or x1-rank rescue is not enough to claim object-binding repair,
   but the failure mode is more specific than the raw bridge counters suggested.
   The `full_output_delta` first-step patch repairs the current open box for one
   nearest-previous case enough to overlap the target (`[420,129,480,259]`,
   IoU `0.4086`), while the next object still routes to a wrong `person`.
2. The layer-input carrier does not locally repair the current open box in this
   smoke. Its stronger effect appears one transition later: it can route the
   following generated object into a backpack span, and in one all-steps
   `next_generated_object_box` row that next object overlaps the target
   (`[420,201,436,277]`, IoU `0.1044`).
3. Persistent full-output/interpolation patching can overdrive the coordinate
   basin and damage syntax. Rows such as `[420,387,387,387]` or
   `[387,387,387,387]` are descriptor-only/malformed, not valid object repair.
4. The corrected split separates four mechanisms that had been easy to blur:

```text
coordinate readout rescue: can make x1/rank/top1 look correct
current-box coordinate repair: can move the already open box toward target overlap
next-row routing rescue: can switch the following object route toward backpack/target
full instance-binding repair: requires coherent current box, correct route, and stable continuation
```

Updated mechanism picture:

```text
The late layer-input state looks more like a trajectory or row-transition
carrier than a complete current-box repair vector. The full layer output can
locally alter the current coordinate emission and sometimes produce target
overlap, but it does not necessarily fix the following object route. Layer input
can influence that following route under persistent patching, but it leaves the
current open box in a wrong valid basin in this smoke. The model therefore seems
to maintain separable but interacting surfaces for current coordinate selection,
object/row transition routing, and syntax stability.
```

Critically revised next experiment priority after this result:

1. Expand the continuation taxonomy across the existing selected pre-x1 cases
   before treating any local rank rescue as object repair. This is now cheap and
   should report current-open-box labels and next-object labels separately.
2. Build the formation-time layer-input map, but ask a sharper question: when
   does the carrier become a next-row routing state versus a current-coordinate
   state? The priority positions remain descriptor onset/end,
   `object_ref_end`, `box_start`, pre-x1, post-x1, box close, and next-object
   onset.
3. Run a bifurcation probe between `full_output_delta` and `layer_input_delta`:
   apply full-output only for the current coordinate step, then layer-input only
   after box close, and compare against the reverse schedule. This should test
   whether current-box repair and next-row routing can be causally composed.
4. Use persistent-vs-first-step contrast as a diagnostic, but interpret it by
   span. If all-steps is needed only for next-object repair, the mechanism is
   trajectory stabilization; if first-step repairs the current box, the mechanism
   is local coordinate-basin selection.
5. Deprioritize broader SVD or checkpoint-transfer claims until these span-level
   regimes are labeled. A mixed low-rank direction would likely average current
   coordinate repair, next-row routing, and malformed overdrive into one opaque
   axis.
