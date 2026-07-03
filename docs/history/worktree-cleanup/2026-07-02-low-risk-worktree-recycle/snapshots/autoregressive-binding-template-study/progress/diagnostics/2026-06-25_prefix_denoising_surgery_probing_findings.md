# Prefix-Denoising Surgery Probing Findings

Date: 2026-06-25

Scope: prefix-denoising SFT checkpoints, selected val200 image bases, position
rows from:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/position_plan/v1_selected_bases_treatment_only/position_rows.jsonl
```

This note records the first residual/readout-space surgery pass, the first
full-forward prefill activation-patching bridge, and short continuation probes
over selected high-value sample bases.

## Artifacts

Focus panel:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/focus_panel/v1_surgery_focus_no_sanity/focus_panel.jsonl
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/focus_panel/v1_surgery_focus_no_sanity/focus_panel_summary.md
```

Layerwise atlas evidence:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v1_image17899_sorted_prex1_term_layers0_8_16_last_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v1_image17899_random_prex1_term_layers0_8_16_last_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v1_focus_pairA_sorted_2685_12670_prex1_term_layers0_8_16_20_24_last_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v1_focus_pairA_random_2685_12670_prex1_term_layers0_8_16_20_24_last_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v1_focus_pairB_sorted_16228_2299_prex1_term_layers0_8_16_20_24_last_gpu2
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v1_focus_pairB_random_16228_2299_prex1_term_layers0_8_16_20_24_last_gpu3
```

Residual/readout-space surgery:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/residual_readout_surgery/v1_random_coord_focus_2685_16228_layers20_24_last_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/residual_readout_surgery/v1_random_terminal_focus_2685_16228_layers20_24_last_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/residual_readout_surgery/v1_terminal_17899_2685_layers24_last_gpu2
```

Full-forward prefill activation patching, true next-token logits:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v2_coord16228_bridge_true_logits_layers24_last_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v2_coord2685_bridge_true_logits_layers24_last_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v2_terminal_bridge_true_logits_layers24_last_gpu2
```

Short patched-continuation bridge, patched first token followed by unpatched
greedy continuation:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v4_coord16228_continuation_repair_labels_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v4_coord2685_continuation_repair_labels_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v4_terminal_continuation_repair_labels_gpu2
```

Direct tail-slot patched-continuation panels:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v5_tail_slots_16228_obj16_layers24_last_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v5_tail_slots_16228_first_token_only_layers24_last_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v5_tail_slots_2685_obj3_layers24_last_gpu2
```

Direct tail-slot all-layer atlas panels:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v6_tail_slots_16228_obj16_layers0_4_8_12_16_20_24_last_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v6_tail_slots_16228_first_token_only_layers0_4_8_12_16_20_24_last_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v6_tail_slots_2685_obj3_layers0_4_8_12_16_20_24_last_gpu2
```

Component-site activation/continuation panels:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v8_component_sites_16228_obj16_layers24_last_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v8_component_sites_16228_obj34_layers24_last_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v8_component_sites_16228_obj45_layers24_last_gpu2
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v8_component_sites_16228_obj51_layers24_last_gpu3
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v8_component_sites_2685_obj3_layers24_last_gpu0
```

Object-state selector and fresh high-value state probes:

```text
src/analysis/prefix_denoising_surgery_probing/object_state_selector.py
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/object_state_selection/v4_v8_v9_component_site_join_diverse
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v9_anchor_19432_random_obj12_prex1_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v9_anchor_12670_sorted_obj0_prex1_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v9_terminal_2299_random_obj27_gpu2
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v9_fn_2157_sorted_obj13_prex1_gpu3
```

Superseded diagnostic pass:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v1_coord16228_bridge_layers24_last_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v1_coord2685_bridge_layers24_last_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v1_terminal_bridge_layers24_last_gpu2
```

The `v1_*` activation-patch rows used reconstructed logits from returned
`hidden_states[-1]`. A follow-up diagnostic showed that a final-layer output
hook could change the consumed stream without changing returned
`hidden_states[-1]`; `v2_*` therefore reads `outputs.logits` at the prefix
position and is the activation-patch evidence to interpret.

## What Was Tested

The surgery runner reloads the checkpoint with the same tokenizer, prompt,
image geometry, `token_embeddings_adapter`, and template path used by the
layerwise atlas. For selected atlas-anomalous position rows, it recomputes
hidden states and perturbs the normalized readout vector before `lm_head` along:

```text
target token output direction - current antagonist output direction
```

For coordinate rows, the target is the emitted target coordinate bin and the
antagonist is the current coordinate top-1. For terminal rows, the target is
`<|im_end|>` and the antagonist is `<|object_ref_start|>`.

This is a local readout-space intervention. It tests basin movability, not
whether generation would remain repaired after the next token.

## Full-Forward Activation Patch Bridge

Activation-patch runner:

```text
src/analysis/prefix_denoising_surgery_probing/activation_patch_continuation.py
scripts/analysis/run_prefix_denoising_activation_patch.py
```

This bridge uses the strongest residual-surgery flips to select representative
source states, replays the same image and assistant prefix, injects a
target-minus-antagonist output-embedding direction into an actual decoder layer
output at the assistant-prefix decision token, and reads the true next-token
`outputs.logits`. It tests whether a local readout-space direction remains
causal when it is inserted into the real forward computation.

Coordinate scope:

```text
image 16228 random_denoise, 4 repeated-person coord_0-collapse pre_x1 rows
image 2685 random_denoise, 1 compact person/wine-glass anchor-collapse pre_x1 row
layers 24 and 27
alphas 0, 0.01, 0.02, 0.05
40 coordinate activation-patch rows
readout_status=ok for all rows
```

Coordinate effects:

```text
16228: 32 rows, 12 exact flips, 12 distance improvements, 23 rank improvements
2685: 8 rows, 5 exact flips, 6 distance improvements, 6 rank improvements
```

Representative thresholds:

```text
16228 object 16, target x1 coord_829:
  layer 24 first exact flip at alpha 0.05
  layer 27 first exact flip at alpha 0.02

16228 object 34, target x1 coord_893:
  layer 24 first exact flip at alpha 0.05
  layer 27 first exact flip at alpha 0.02

16228 object 45, target x1 coord_926:
  layer 24 first exact flip at alpha 0.05
  layer 27 first exact flip at alpha 0.02

16228 object 51, target x1 coord_862/893 ambiguity:
  layer 24 exact emitted-target-bin flip at alpha 0.05
  layer 27 exact next-token target flip at alpha 0.02

2685 object 3, target x1 coord_519:
  layer 24 first exact flip at alpha 0.01
  layer 27 first exact flip at alpha 0.01
```

Terminal scope:

```text
images 17899 and 2685
models sorted_denoise and random_denoise
4 terminal-pressure source states
layers 24 and 27
alphas 0, 0.005, 0.01, 0.02, 0.05
40 terminal activation-patch rows
readout_status=ok for all rows
```

Terminal effects:

```text
32 exact stop-router flips to target direction
32 terminal-margin improvements
all four selected states flip by alpha 0.005 at both layer 24 and layer 27
```

Examples:

```text
random 2685 object 60:
  baseline top1 <|object_ref_start|>, margin 0.0
  layer 24 alpha 0.005 top1 <|im_end|>, margin +0.1243
  layer 27 alpha 0.005 top1 <|im_end|>, margin +0.3026

sorted 17899 object 48:
  baseline top1 <|object_ref_start|>, margin -0.06236
  layer 24 alpha 0.005 top1 <|im_end|>, margin +0.06236
  layer 27 alpha 0.005 top1 <|im_end|>, margin +0.18518
```

Interpretation: the residual/readout-space result does survive real forward
patching for the selected bridge states. The coordinate failures are not simply
unreachable visual/coordinate representations; late-layer perturbations can
move the true next-token logits out of repeated-anchor or border-anchor basins.
The terminal router is even more fragile: a tiny layer-24 or layer-27 target
direction reliably crosses the `<|im_end|>` / `<|object_ref_start|>` boundary.

## Short Continuation Findings

The `v4_*` bridge uses the same full-forward patch but then lets the model
continue greedily for up to six tokens without further patching. This separates
three outcomes:

```text
coord_tail_coherent: patched x1 is followed by the original y1/x2/y2/box_end tail
first_token_only: patched x1 flips, but the remaining coordinate tail shifts
terminal_stop_repair: a non-stop baseline continuation is changed to <|im_end|>
terminal_already_stops: baseline greedy continuation already emits <|im_end|>
```

Coordinate continuation scope:

```text
16228 random_denoise: 4 repeated-person pre_x1 states, layers 24/27, 32 rows
2685 random_denoise: 1 person/wine-glass-anchor pre_x1 state, layers 24/27, 8 rows
continuation_steps: 6
readout_status=ok for all rows
```

Coordinate continuation effects:

```text
16228: 12 first-token target repairs, 3 full coordinate-tail coherent repairs
2685: 5 first-token target repairs, 0 full coordinate-tail coherent repairs
```

The successful full-tail case is specific:

```text
image 16228 object 16
baseline continuation after pre_x1: coord_0, coord_460, coord_40, coord_597, box_end, object_ref_start
patched layer 27 alpha 0.02: coord_829, coord_384, coord_861, coord_453, box_end, object_ref_start
expected tail after x1: coord_384, coord_861, coord_453, box_end
```

Other coordinate repairs are first-token-only. For example:

```text
image 2685 object 3
patched x1: coord_519
patched tail: coord_114, coord_999, coord_789, box_end
expected tail: coord_114, coord_999, coord_787, box_end
```

Interpretation: late coordinate patching can sometimes unlock a whole stored
object-span trajectory, but often the remaining coordinate slots fall into a
nearby basin. This is a deeper split than the v2 next-token result: a corrected
x1 does not automatically guarantee coherent y1/x2/y2. The model may carry a
partially coupled box plan where x1 is independently movable but later slots
remain attracted to local geometry basins.

Terminal continuation scope:

```text
17899 and 2685
sorted_denoise and random_denoise
4 terminal-pressure states, layers 24/27, 40 rows
continuation_steps: 6 with <|im_end|> as stop token
readout_status=ok for all rows
```

Terminal continuation effects:

```text
terminal_stop_repair: 16 rows
terminal_already_stops: 20 rows
first_token_not_repaired: 4 rows
```

The true repairs are the sorted-denoise terminal rows. Both sorted image `17899`
and sorted image `2685` continue with `<|object_ref_start|>` at alpha `0`, but
switch to a single-token `<|im_end|>` stop by alpha `0.005` at both layers 24
and 27. The random-denoise terminal rows mostly expose the exact-tie wrinkle:
their readout margins are at or near zero, and manual greedy already selects
`<|im_end|>` at alpha `0`, so they are `terminal_already_stops`, not true
repairs.

Interpretation: the sorted-denoise premature-continuation cases really are
late terminal-router failures that can be turned into clean stops with a tiny
target-direction patch. The random-denoise cases are even more knife-edge:
readout top-k/tie accounting may report continuation pressure, but greedy
decode can already fall to stop. Future terminal analysis must preserve tie
semantics instead of relying on top-1 text alone.

## Direct Tail-Slot Findings

The `v5_*` panels stop treating object-span repair as only a `pre_x1` event.
They re-enter the same selected sample bases at the direct tail-slot prefixes:

```text
post_x1_pre_y1
post_y1_pre_x2
post_x2_pre_y2
```

They were selected by exact `image_id` plus generated object index, so the
panel spends GPU time on mechanistically rich slices instead of normal learned
cases:

```text
16228 object 16: the prior full-tail coherent x1 repair case
16228 objects 34,45,51: prior first-token-only repeated-person repairs
2685 object 3: prior near-tail person/wine-glass-anchor repair
```

Artifact summaries:

```text
16228 object 16:
  selected_source_row_count: 3
  output_row_count: 30
  readout_status_counts: direction_error=30

16228 objects 34,45,51:
  selected_source_row_count: 9
  output_row_count: 90
  readout_status_counts: ok=60, direction_error=30
  continuation_repair_label_counts:
    coord_tail_coherent=6
    first_token_only=8
    first_token_not_repaired=46

2685 object 3:
  selected_source_row_count: 3
  output_row_count: 30
  readout_status_counts: ok=10, direction_error=20
  continuation_repair_label_counts:
    coord_tail_coherent=7
    first_token_not_repaired=3
```

The `direction_error` rows are not a generic runner failure. They line up with
baseline target rank `1`: for example, `16228` object 16 already places
`coord_384`, `coord_861`, and `coord_453` top-1 under the exact tail-slot
prefixes. In this readout-direction bridge, target and antagonist collapse to
the same coordinate token, producing a zero direction. Interpretation: once the
prefix has been guided into the correct object tail, these slots are locally
learned; their free-rollout failure is upstream in how the autoregressive
history enters or stays in that tail, not in direct local availability.

The non-saturated rows expose a sharper split:

```text
16228 object 34:
  y1 can be moved from coord_388 to coord_387 at one small-alpha layer-24 row,
  but x2/y2 remain at the old tail; this is first-token-only repair.

16228 object 45:
  x2 can be moved to coord_967, but the following y2 goes to coord_447 rather
  than the expected coord_433; this is first-token-only or nearby-basin repair.
  Its y2 target rank starts very weak at rank 26 and improves under patching,
  but still does not reach exact target in this panel.

16228 object 51:
  x2 is the clean positive case. Patching x2 to coord_899 is followed by the
  expected coord_437 and box_end in 6/10 rows, so a direct tail slot can unlock
  a coherent remaining tail even when the earlier pre_x1 repair was unstable.

2685 object 3:
  y2 is directly recoverable. Patching y2 from coord_789 to coord_787 is
  followed by box_end in 7/10 rows. The earlier y1 and x2 direct prefixes are
  already top-1 and therefore direction-saturated.
```

Interpretation: the coordinate tail is not one monolithic box program. Some
slots are locally saturated and only fail because the preceding autoregressive
history reaches the wrong state. Other slots are weak but independently
recoverable. Still others move only the immediate coordinate while the
following tail remains attached to the previous local basin. The deepest next
question is therefore a state-entry/coupling question: what upstream component
decides whether the model enters the correct tail manifold, and why do some
later slots remain bound to the old spatial anchor after the first coordinate
is repaired?

## Direct Tail-Slot Tensor-Flow Atlas

The `v6_*` atlas repeats the same representative sample bases over layers
`0,4,8,12,16,20,24,final` for `pre_x1` and the three direct coordinate-tail
slots. It uses exact generated-object-index filtering, so the rows are the same
high-value object bases as the `v5_*` continuation panel.

Scope:

```text
16228 object 16: 4 source rows, 32 atlas rows
16228 objects 34,45,51: 12 source rows, 96 atlas rows
2685 object 3: 4 source rows, 32 atlas rows
readout_status=ok for all 160 rows
```

Main tensor-flow pattern:

```text
pre_x1:
  layer 24 often has high coordinate-family mass and a nearby or plausible
  target rank, but the final layer can collapse to coord_0 or a repeated anchor.

direct tail slots:
  with the preceding coordinate(s) already in prefix, the final layer often
  snaps to exact or near-exact target with coordinate-family mass near 1.0,
  even when layers 0-24 looked diffuse, nearby, or wrong.
```

Representative traces:

```text
16228 object 16, pre_x1 target coord_829:
  L24: rank 41, top coord_853, distance 24, coord mass 0.476
  final: rank 6, top coord_0, distance 829, coord mass 0.994

16228 object 16, post_x1_pre_y1 target coord_384:
  L24: rank 48, top coord_371, distance 13, coord mass 0.001
  final: rank 1, top coord_384, distance 0, coord mass 1.000

16228 object 45, post_x2_pre_y2 target coord_433:
  L24: rank 2, top coord_432, distance 1, coord mass 0.000
  final: rank 26, top coord_447, distance 14, coord mass 1.000

2685 object 3, post_y1_pre_x2 target coord_999:
  L24: rank 1, top coord_999, distance 0, coord mass 0.000
  final: rank 1, top coord_999, distance 0, coord mass 0.997

2685 object 3, post_x2_pre_y2 target coord_787:
  L24: rank 8, top coord_404, distance 383, coord mass 0.007
  final: rank 2, top coord_789, distance 2, coord mass 0.999
```

Interpretation: layer 24 can carry a low-rank or nearby readout hint without
being in full coordinate-token mode. The final layer is the decisive router
into the coordinate-token simplex, but it is not always the correct object
state. For `pre_x1`, final-layer coordinate-mode saturation can amplify the
wrong anchor (`coord_0` or repeated-person anchor). For guided tail slots, the
same final-layer saturation often materializes the correct local coordinate.
This points to a two-stage mechanism: an upstream state-entry/anchor choice
before the final coordinate simplex, followed by a late coordinate-token
router that is powerful but not intrinsically object-correct.

## Component-Site Patch Findings

The `v8_*` panels extend the activation-patch bridge with four injection sites:

```text
layer_output
layer_input
self_attn
mlp
```

Scope:

```text
selected source states:
  16228 objects 16,34,45,51
  2685 object 3
positions:
  pre_x1, post_x1_pre_y1, post_y1_pre_x2, post_x2_pre_y2
layers:
  24, final
alphas:
  0, 0.005, 0.01, 0.02, 0.05
total rows:
  800
readout_status:
  ok=480
  direction_error=320
```

The `direction_error` rows are saturated-guidance cases: target is already the
current coordinate top-1, so target-minus-antagonist has zero norm. The 480
`ok` rows expose the component split:

```text
patched first token is target:
  layer_output: 38/120
  layer_input: 35/120
  mlp: 27/120
  self_attn: 11/120

full coordinate-tail coherent continuation:
  layer_output: 13/120
  layer_input: 13/120
  mlp: 12/120
  self_attn: 6/120
```

Layer split:

```text
layer 24:
  layer_output tail coherent: 6/60
  layer_input tail coherent: 5/60
  mlp tail coherent: 5/60
  self_attn tail coherent: 2/60

final layer:
  layer_input tail coherent: 8/60
  layer_output tail coherent: 7/60
  mlp tail coherent: 7/60
  self_attn tail coherent: 4/60
```

Strong positive examples:

```text
2685 object 3, post_x2_pre_y2, target coord_787:
  layer_output L24 repairs by alpha 0.005
  mlp L24 repairs by alpha 0.005
  layer_input L27 repairs by alpha 0.005
  self_attn is weaker: L24 needs alpha 0.05, L27 needs alpha 0.01

16228 object 51, post_y1_pre_x2, target coord_899:
  layer_output L27 repairs by alpha 0.005
  layer_input L27 repairs by alpha 0.005
  mlp L27 repairs by alpha 0.005
  self_attn only repairs by alpha 0.05 and less consistently
```

Negative but important examples:

```text
16228 objects 16,34,45,51, pre_x1:
  component patches can often repair the first x1 token,
  but none of these pre_x1 rows becomes full-tail coherent.

16228 object 45, post_x2_pre_y2, target coord_433:
  baseline rank is weak at 26;
  component patches improve nearby bins but do not reach exact coherent repair
  in this alpha range.
```

Interpretation: the late coordinate-basin router is not primarily an
attention-only effect. MLP and whole layer output are the most reliable direct
routes for turning a local target direction into a coordinate-token decision.
Layer input can also work, especially at the final layer, which suggests the
downstream layer can route an already-present residual hint into the coordinate
simplex. However, `pre_x1` state-entry failures remain first-token-only: a
single local component patch can move x1 but does not reconstruct the object
tail manifold. The next causal question is therefore not just "which component
knows the target coordinate?" but "which component binds the target coordinate
to the remaining object-tail trajectory?"

## Object-State Selection And V9 Probe Findings

The object-state selector turns the image-base registry and position plan into
a narrower surgical panel over exact `(image, model, object, position)` states.
It joins any available activation-patch rows post hoc, preserves target-token
semantics from the source artifacts, and caps selected rows per
image/model/family so repeated normal-looking states do not consume the deep
panel.

Current joined artifact:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/object_state_selection/v4_v8_v9_component_site_join_diverse
```

Scope:

```text
position rows: 11648
activation rows joined: 960
object-state candidates: 5376
selected states: 32
selected image bases: 2157, 2299, 2685, 12670, 14439, 16228, 19432
```

Selected family counts:

```text
same_image_tail_binding_contrast: 6
state_entry_first_token_only: 1
duplicate_anchor_basin: 19
termination_boundary: 3
false_negative_guidance: 3
```

The top selected states preserve the known high-value contrast:

```text
16228 random object 51 pre_x1:
  first-token-only x1 repair, no full tail binding

16228 random object 51 post_y1_pre_x2:
  tail-coherent repair exists, strongest at final-layer layer_input/layer_output/mlp

2685 random object 3 pre_x1:
  first-token-only x1 repair

2685 random object 3 post_x2_pre_y2:
  tail-coherent y2 repair exists
```

The fresh `v9_*` probes add three new mechanism handles:

```text
19432 random object 12 pre_x1, target coord_0:
  40/40 rows are direction_error across layer_output/layer_input/mlp/self_attn.

12670 sorted object 0 pre_x1, target coord_0:
  40/40 rows are direction_error across layer_output/layer_input/mlp/self_attn.

2299 random object 27 termination_pressure:
  40/40 rows already emit <|im_end|>; terminal is locally stable, not a repair need.

2157 sorted object 13 pre_x1, target coord_374:
  40/40 rows ok; 26 first-token-only repairs, 14 unrepaired;
  no component/site makes the y1/x2/y2 tail coherent.
```

Interpretation: not all duplicate-anchor states are merely movable wrong
basins. The selected `coord_0` border-anchor states are saturated in the local
target-minus-antagonist direction: the emitted bad coordinate is already the
coordinate top-1, so this perturbation family cannot define a corrective
direction. That makes them good candidates for an anti-anchor or alternate-GT
target probe, not another target-token reinforcement probe. By contrast,
`2157` shows the familiar state-entry split: x1 can be locally forced through
layer input/output or MLP at small alpha, but the tail follows the old basin.

## Coordinate Basin Findings

Coordinate surgery artifact:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/residual_readout_surgery/v1_random_coord_focus_2685_16228_layers20_24_last_gpu0/residual_surgery_summary.json
```

Scope:

```text
12 selected source states
layers 20, 24, final
alphas 0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5
252 surgery rows
readout_status=ok for all rows
```

Aggregate effects:

```text
flip_to_target_count: 89
coord_distance_improved_count: 167
rank_improved_count: 216
```

The strongest case is random-denoise image `16228`, repeated `person` spans.
The atlas showed many final-layer x1 rows collapsing to `coord_0` despite
target x1 values hundreds of bins away. Readout-space surgery shows that these
states are not visually or coordinate-type inaccessible. They are locally
movable:

```text
image 16228 random, 11 selected coordinate states
final layer alpha 0: mean x1 distance about 745.8 bins
final layer alpha 0.05: 8/11 exact target flips, mean distance about 1.36 bins
layer 24 alpha 0.1: 9/11 exact target flips, mean distance about 0.18 bins
```

For image `2685`, one selected random-denoise coordinate state is especially
compact:

```text
image 2685 random, object 3, desc=person
final layer alpha 0: x1 distance 519
final layer alpha 0.01: distance 250
final layer alpha 0.02: exact target
```

Interpretation: the failure looks less like "coordinate tokens unavailable" or
"visual evidence completely absent" and more like a late readout/basin
competition where repeated anchors or border anchors dominate an otherwise
recoverable target direction.

## Terminal Router Findings

Random terminal artifact:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/residual_readout_surgery/v1_random_terminal_focus_2685_16228_layers20_24_last_gpu1/residual_surgery_summary.json
```

Cross-family terminal artifact:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/residual_readout_surgery/v1_terminal_17899_2685_layers24_last_gpu2/residual_surgery_summary.json
```

Cross-family scope:

```text
4 selected terminal states
images 17899 and 2685
models sorted_denoise and random_denoise
layers 24 and final
56 surgery rows
readout_status=ok for all rows
```

Aggregate effects:

```text
flip_to_target_count: 36
terminal_margin_improved_count: 48
```

At final layer, the stop/continue router is extremely movable. The final-layer
margin flips toward `<|im_end|>` by alpha `0.005` for all selected terminal
states, including sorted-denoise cases whose baseline preferred continuation.
Layer 24 is less saturated but still close: alpha `0.05` generally flips the
margin positive.

Examples:

```text
sorted 17899 final: baseline im_end - object_ref_start = -0.06236
alpha 0.005: margin +0.18517

random 17899 final: baseline margin 0.0
alpha 0.005: margin +0.24478

sorted 2685 final: baseline margin -0.06237
alpha 0.005: margin +0.18520

random 2685 final: baseline margin 0.0
alpha 0.005: margin +0.24486
```

Interpretation: the terminal decision is plausibly a late router knife-edge,
not a hard unavailable stop representation. This matches the atlas observation
that terminal rows often sit at exact or near-exact `<|im_end|>` /
`<|object_ref_start|>` ties.

## Current Mechanism Picture

The first readout surgery supports two working mechanisms:

1. Coordinate duplication/basin failures are late, locally movable readout
   competitions. The model often enters high coordinate-token mass, but a wrong
   repeated anchor or border anchor wins. Small target-direction movement can
   restore the intended coordinate locally.

2. Premature continuation or termination is controlled by a late stop/continue
   router margin that can be tiny. The sorted/random differences appear partly
   as different terminal margins over similar `<|im_end|>` /
   `<|object_ref_start|>` directions.

The component-site bridge adds one refinement: local coordinate-token steering
is strongest through MLP/layer-output routes and weaker through attention-only
patches. That does not yet solve the autoregressive binding problem, because
`pre_x1` component patches still repair only the first coordinate rather than
the full object-tail trajectory.

The object-state selector and `v9_*` probes add a second refinement: the next
deep panel should separate state-entry failures into at least two subtypes.
Some non-anchor x1 states are locally steerable but not tail-bound (`2157`,
`16228`, `2685`). Some border-anchor duplicate states are already saturated at
their emitted `coord_0`, so target-token reinforcement is the wrong causal
handle (`12670`, `19432`). These require anti-anchor, alternate-GT, or
competitor-direction surgery.

## What This Does Not Prove

Residual/readout-space surgery alone does not prove that the model would emit a
repaired object span under normal autoregressive continuation. The v2
activation-patch bridge proves immediate next-token causal movability. The v4
short-continuation bridge proves clean terminal repairs for sorted-denoise
boundary cases and shows that coordinate repair can be either full-tail
coherent or first-token-only. The v5 tail-slot panel shows that some tail slots
are locally saturated under exact guidance, some are independently recoverable,
and some are only locally repaired while the remaining tail stays attached to a
nearby basin. The v6 tensor-flow atlas shows that the final layer is where
coordinate-mode saturation becomes decisive, but the saturated coordinate mode
can be either correct under guided tail prefixes or wrong under `pre_x1`
state-entry collapse. The v8 component-site panel localizes much of the
effective coordinate-token steering to MLP/layer-output routes, with
attention-only patches weaker. It still does not prove full free-rollout repair
over many objects, nor does it identify the component that binds a repaired x1
to the rest of the object-tail manifold. The `v9_*` anchor probes also do not
prove visual absence: `direction_error` only means the current target-direction
construction is saturated because the bad emitted coordinate is already the
local top-1.

## Recommended Next Probe

Next deepen the coordinate continuation split on:

```text
image 2685 random, object 3, pre_x1
image 16228 random, selected person coord-0-collapse rows
image 16228 object 16 full-tail coherent repair
image 16228 objects 34/45/51 first-token-only repairs
image 2685 object 3 first-token-only near-tail repair
```

Use image-base selection as a hard prioritization gate. The next round should
not spend its main budget on normal or well-learned scenarios. Pick
representative bases that compress multiple mechanism contrasts into one image:
duplication onset, false-negative candidates, terminal ambiguity, repeated
spatial anchors, and paired objects where one row repairs as a full tail while
another row repairs only the first coordinate.

Use the object-state selector as the deterministic entry point:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/object_state_selection/v4_v8_v9_component_site_join_diverse/object_state_selection.jsonl
```

Priority patch sites:

```text
layer 24
final layer
coordinate pre_x1
termination_pressure
```

Measure which components decide whether the post-x1 tail follows the patched
span or a nearby basin: compare layer input versus attention versus MLP output,
and patch y1/x2/y2 slots separately after x1 repair. The most valuable target
is now the transition into the final coordinate simplex: layer 24 carries
nearby or low-rank hints, while the final layer decides whether those hints
become the correct tail coordinate or the wrong repeated/border anchor. For
component surgery, prioritize `mlp` and `layer_output` when testing local
coordinate-token steering, but do not mistake first-token repair for object-tail
repair. For state-entry surgery, the open problem is to find the component that
couples x1 to y1/x2/y2 rather than only moving x1. For terminal rows, the next
probe should focus less on "can we stop" and more on tie semantics and why
sorted denoise keeps continuation pressure where random greedy already falls
to stop.

For saturated anchor rows such as `12670` sorted object 0 and `19432` random
object 12, do not spend another probe reinforcing the emitted `coord_0`.
Instead, construct an anti-anchor or alternate-target direction from a matched
neighbor, manually reviewed GT candidate, or competitor non-border coordinate
and ask whether the basin can be escaped before the final coordinate simplex.

## 2026-06-25 anchor-escape GT-candidate probe

The first anchor-escape implementation creates synthetic `pre_x1` source rows
for saturated border-anchor states. It copies the original position row,
preserves the emitted bad-anchor provenance, and replaces only the next x1
target with a same-description, non-border GT-candidate x1. The row is explicitly
marked as `next_token_basin_escape_only`; the original rollout tail is not
treated as a coherent expected continuation after this synthetic x1 rewrite.

Implementation:

```text
src/analysis/prefix_denoising_surgery_probing/anchor_escape_plan.py
tests/analysis/test_prefix_denoising_anchor_escape_plan.py
```

Activation-patch rows now preserve `anchor_escape_*` provenance fields so that
downstream reductions can retain the GT candidate identity directly:

```text
src/analysis/prefix_denoising_surgery_probing/activation_patch_continuation.py
tests/analysis/test_prefix_denoising_activation_patch.py
```

Strict plan artifact:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_plan/v2_saturated_direction_coord0_gt_candidates
```

Strict selection:

```text
candidate_count: 24
row_count: 4
row_counts_by_model: random_denoise=2, sorted_denoise=2
row_counts_by_image: 12670=2, 19432=2
row_counts_by_policy: same_desc_unmatched_gt=4
```

The four rows are:

```text
12670 sorted_denoise object 0 person: coord_0 -> coord_492, GT 15
12670 sorted_denoise object 0 person: coord_0 -> coord_344, GT 8
19432 random_denoise object 12 chair: coord_0 -> coord_537, GT 8
19432 random_denoise object 12 chair: coord_0 -> coord_351, GT 5
```

Pilot patch roots:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v11_anchor_escape_12670_sorted_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v11_anchor_escape_19432_random_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_reduce/v1_v11_pilot
```

Full layer-sweep roots:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v12_anchor_escape_12670_sorted_layers0_13_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v12_anchor_escape_12670_sorted_layers14_27_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v12_anchor_escape_19432_random_layers0_13_gpu2
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v12_anchor_escape_19432_random_layers14_27_gpu3
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_reduce/v2_v12_layer_sweep
```

Full sweep scope: all decoder layers `0..27`, patch sites
`layer_output,layer_input,mlp,self_attn`, alphas
`0,0.005,0.01,0.02,0.05,0.1`, `continuation_steps=0`, four strict
saturated anchor-escape rows, `2688` output rows total, all readout statuses
`ok`.

Full sweep reduction:

```text
12670 sorted object 0 target coord_344 GT 8:
  baseline top1/rank: coord_0 / 159
  best: rank 6, top1 coord_346, layer 27 layer_output alpha 0.1
  exact flips: 0
  near-top1<=4: 17
  first near layer: layer_output 22, layer_input 23, mlp 27
  self_attn best rank: 95, top1 remains coord_0

12670 sorted object 0 target coord_492 GT 15:
  baseline top1/rank: coord_0 / 237
  best: rank 22, top1 coord_429, layer 27 layer_output alpha 0.1
  exact flips: 0
  near-top1<=4: 0
  self_attn best rank: 160, top1 remains coord_0

19432 random object 12 target coord_351 GT 5:
  baseline top1/rank: coord_0 / 55
  best: rank 1, top1 coord_350, layer 27 layer_output alpha 0.1
  exact flips: 0
  near-top1<=4: 23
  first rank<=10 layer: layer_output 19, layer_input 20, mlp 20,
                        self_attn 25

19432 random object 12 target coord_537 GT 8:
  baseline top1/rank: coord_0 / 393
  best: rank 1, top1 coord_537, layer 27 layer_output alpha 0.1
  exact flips: 3
  first exact layer: layer_output 26, layer_input 27
  first rank<=10 layer: layer_output 24, layer_input 25, mlp 27
  self_attn best rank: 89, top1 remains coord_0
```

Interpretation:

1. The earlier `direction_error` result for the emitted `coord_0` target was
   a direction-construction circularity, not evidence that the state has no
   alternate-object support. When the target is changed to plausible
   same-description, non-border GT x1 candidates, the same bad states can be
   moved strongly away from the border anchor.

2. The escape signal is late and residual-stream/readout dominated. The best
   rows are all layer 27 `layer_output` at alpha `0.1`. `layer_input` is
   usually second-best, `mlp` can help but is weaker, and `self_attn` rarely
   leaves the `coord_0` basin. This supports a picture where the final
   coordinate simplex performs a basin selection over already available
   coordinate alternatives rather than self-attention directly rewriting the
   object identity at the probed layer.

3. The two sample bases split into useful subtypes. `19432/random/chair`
   contains a steerable alternate-chair coordinate basin: one target flips
   exactly to `coord_537`, and another reaches rank 1 with top1 `coord_350`,
   one bin from the target `coord_351`. `12670/sorted/person` contains a
   weaker or more confused alternate-person basin: target `coord_344` reaches
   a near top1 `coord_346`, but target `coord_492` is redirected toward
   `coord_429` instead of the requested candidate. This looks less like simple
   visual absence and more like a late competition among multiple same-class
   spatial anchors.

   A cheap post-hoc destination check supports that interpretation. In
   `12670`, the successful near target `coord_346` is adjacent to GT person 8
   at x1 bin `344`. The wrong `coord_414/427/429` destination region is not
   arbitrary: it sits near GT person 10 at bin `411`, GT backpack 11 at bin
   `447`, GT teddy bear 12 at bin `471`, and predicted person/backpack anchors
   at bins `414/439`. In `19432`, the near target `coord_350` is adjacent to
   GT chair 5 at bin `351`; the exact target `coord_537` is GT chair 8; and
   secondary destinations such as `coord_598` or `coord_414` align with other
   chair prediction/GT basins. The failure mode is therefore better described
   as destination-basin competition than as random coordinate noise.

4. The sample-base selection rule is validated. These two images carry more
   mechanistic information than a broad average over normal rows: the same
   surface symptom, `coord_0` duplicate-anchor onset, decomposes into exact
   escape, near escape, and destination-basin drift depending on the target
   candidate and image context.

Immediate next path:

- For `19432/random/chair`, use the exact/near escape rows as donor/receiver
  cases for tail binding: after x1 is moved to `coord_537` or near
  `coord_351`, test whether y1/x2/y2 can be bound to the same GT object or
  whether the continuation snaps back to the old chair-repeat manifold.
- For `12670/sorted/person`, inspect the destination bins `414, 427, 429,
  346` against GT/pred boxes and previous internal alternatives. The key
  question is whether the wrong destination is a real neighboring person basin,
  a mixed object/coordinate attractor, or a learned coordinate prior.
- For both bases, run visual-region/value-head attribution only on layers
  `22..27`, with primary focus on `layer_output/layer_input` and secondary
  focus on MLP. Attention-only patching is currently a low-priority route for
  this specific anchor-escape mechanism.

## 2026-06-25 anchor-escape tail-binding probe

The next probe asked whether an escaped x1 basin is enough to make the object
tail bind to the same GT object. A new prefix-denoising-specific runner builds
formation-span continuation plans from anchor-escape rows, then scores the
generated tail with the existing formation-span taxonomy plus stricter IoU
quality labels.

Implementation:

```text
src/analysis/prefix_denoising_surgery_probing/anchor_escape_tail_binding.py
tests/analysis/test_prefix_denoising_anchor_escape_tail_binding.py
```

Dry plan:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_tail_binding_plan/v1_strict4_all_modes
```

GPU run roots:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_tail_binding/v1_12670_sorted_all_modes_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_tail_binding/v1_19432_random_all_modes_gpu1
```

Thresholded reduction:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_tail_binding_reduce/v1_strict4_all_modes_iou_quality
```

Post-hoc y2 destination check:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_tail_binding_reduce/v1_strict4_all_modes_iou_quality/anchor_escape_tail_binding_y2_destination.md
```

Scope:

```text
4 anchor-escape targets
5 modes per target:
  pre_x1_free
  force_target_x1
  force_target_x1_y1
  force_target_x1_y1_x2
  force_target_full_box
20 scored continuations
0 errors
```

Quality counts by mode:

```text
pre_x1_free:
  original_anchor_replay: 2
  valid_non_target: 2

force_target_x1:
  target_bound_iou50: 2
  low_iou_target_overlap: 1
  valid_non_target: 1

force_target_x1_y1:
  target_bound_iou75: 2
  low_iou_target_overlap: 2

force_target_x1_y1_x2:
  target_bound_iou75: 2
  low_iou_target_overlap: 2

force_target_full_box:
  target_bound_iou75: 4
```

Per-base result:

```text
12670 sorted person target GT 15:
  pre_x1_free -> [0,0,999,999], original replay, target IoU 0.025
  force_x1 -> [492,49,528,166], valid non-target, target IoU 0.0
  force_x1_y1 -> [492,403,520,415], low target IoU 0.013
  force_x1_y1_x2 -> [492,403,615,415], low target IoU 0.059
  full_box -> exact closure

12670 sorted person target GT 8:
  pre_x1_free -> [0,0,999,999], original replay, target IoU 0.050
  force_x1 -> [344,0,403,999], low target IoU 0.301
  force_x1_y1 -> [344,365,426,415], low target IoU 0.082
  force_x1_y1_x2 -> [344,365,461,415], low target IoU 0.117
  full_box -> exact closure

19432 random chair target GT 8:
  pre_x1_free -> [0,0,73,270], valid non-target, target IoU 0.0
  force_x1 -> [537,0,651,342], target IoU 0.632
  force_x1_y1 -> [537,122,651,347], target IoU 0.996
  force_x1_y1_x2 -> [537,122,651,343], target IoU 0.978
  full_box -> exact closure

19432 random chair target GT 5:
  pre_x1_free -> [0,0,73,270], valid non-target, target IoU 0.0
  force_x1 -> [351,0,467,342], target IoU 0.584
  force_x1_y1 -> [351,122,467,342], target IoU 0.900
  force_x1_y1_x2 -> [351,122,458,342], target IoU 0.973
  full_box -> exact closure
```

Interpretation:

1. `19432/random/chair` has a real object-tail binding route behind the
   visible duplicate-anchor failure. Without guidance it emits a small
   non-target left-edge chair-like box. Once x1 is forced to the alternate GT
   candidate, the model recovers a target-overlapping chair box with IoU above
   `0.5`; once x1+y1 is forced, it nearly completes the GT box by itself. This
   makes `19432` the current best evidence that the object is not visually
   absent. The failure is an autoregressive entry/basin-selection failure.

2. `12670/sorted/person` is qualitatively different. Free continuation replays
   the original huge border box. Forcing x1 alone does not bind the tail to the
   target person. Even forcing x1+y1+x2 still produces a shallow y2 around
   `415`, not the target y2 (`608` or `793`). The tail is not merely waiting
   for the correct x1; it has a strong vertical-extent or local-height basin.
   This is the sharper subtype: x1 escape is possible or near-possible, but
   full object-tail binding is missing.

   The y2 destination check shows this shallow closure is not arbitrary. For
   both `12670` targets, generated y2 `415` is exactly or near predicted y2
   bins for existing `backpack/person` anchors (`415/416`) and near GT
   `cell phone/person` y2 bins (`402/396`). In contrast, the `19432` forced
   tails produce y2 bins `342/343/347/348`, which sit directly on chair
   GT/prediction basins. The vertical-extent failure is therefore another
   destination-basin choice, not random coordinate drift.

3. The permissive `target_bbox_overlap` label from the older formation-span
   taxonomy is not sufficient for this study. A full-canvas replay overlaps
   every target but is still an original-anchor failure. The new reduction
   therefore treats IoU `>=0.5` / `>=0.75` as the meaningful binding evidence
   and explicitly labels `original_anchor_replay`, `low_iou_target_overlap`,
   `target_bound_iou50`, and `target_bound_iou75`.

Mechanism update:

The anchor-escape study now separates three layers of failure:

```text
1. x1 basin selection:
   can the final coordinate simplex move away from coord_0?

2. y1/x2 local box scaffold:
   after x1 is supplied, does the model pick plausible y1/x2 for the same
   object?

3. y2 / vertical extent closure:
   does the box close around the target object, or snap to a shallow/large
   learned extent?
```

`19432/random/chair` passes stages 1-3 under modest language/coordinate
guidance. `12670/sorted/person` passes only a weak version of stage 1 and fails
stage 3 even under staged forcing. The next high-value probe should therefore
not ask a binary "can the model perceive the missing object?" It should compare
the internal visual/value routes that make chair tail closure available in
`19432` but leave person vertical extent unbound in `12670`.

## Y2 Closure Readout Panel

Artifact scope: representative sample-base hard contrasts selected from the
tail-binding pass/fail rows, not normal or well-learned population sampling.
This is deliberately narrow: the unit of study is the image-base plus model
checkpoint, and the result is a mechanism selector rather than a val200 rate
claim.

Artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_readout_plan/v1_force_x1_y1_x2
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_readout/v1_12670_sorted_layers22_27_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_readout/v1_19432_random_layers22_27_gpu1
```

Tooling:

```text
src/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_readout.py
tests/analysis/test_prefix_denoising_anchor_escape_y2_readout.py
```

Probe design:

```text
source rows:
  anchor_escape_tail_binding_quality_rows.jsonl

selected rows:
  continuation_application_mode=force_target_x1_y1_x2
  2 sorted_denoise / image 12670 / person / low_iou_target_overlap
  2 random_denoise / image 19432 / chair / target_bound_iou75

readout position:
  assistant prefix ending after forced target x1+y1+x2

target:
  next coord token is target y2

layers:
  22, 23, 24, 25, 26, 27
```

The selected row identities are:

```text
12670 sorted person GT15:
  target box [492,403,615,608]
  generated forced-tail box [492,403,615,415]
  original emitted y2 989

12670 sorted person GT8:
  target box [344,365,461,793]
  generated forced-tail box [344,365,461,415]
  original emitted y2 989

19432 random chair GT8:
  target box [537,122,651,348]
  generated forced-tail box [537,122,651,343]
  original emitted y2 300

19432 random chair GT5:
  target box [351,122,458,348]
  generated forced-tail box [351,122,458,342]
  original emitted y2 300
```

Layerwise result:

```text
12670 sorted target_y2=793 generated_y2=415:
  L22 top1=440 target_rank=969 generated_rank=370 target-generated_logit=-2.98
  L23 top1=440 target_rank=801 generated_rank=441 target-generated_logit=-1.62
  L24 top1=999 target_rank=591 generated_rank=34  target-generated_logit=-1.72
  L25 top1=697 target_rank=660 generated_rank=149 target-generated_logit=-1.44
  L26 top1=347 target_rank=683 generated_rank=88  target-generated_logit=-3.19
  L27 top1=415 target_rank=571 generated_rank=1   target-generated_logit=-2.44

12670 sorted target_y2=608 generated_y2=415:
  L22 top1=454 target_rank=715 generated_rank=200 target-generated_logit=-1.84
  L23 top1=440 target_rank=963 generated_rank=86  target-generated_logit=-4.09
  L24 top1=439 target_rank=729 generated_rank=35  target-generated_logit=-3.66
  L25 top1=454 target_rank=768 generated_rank=43  target-generated_logit=-3.44
  L26 top1=448 target_rank=523 generated_rank=26  target-generated_logit=-3.19
  L27 top1=415 target_rank=474 generated_rank=1   target-generated_logit=-2.69

19432 random target_y2=348 generated_y2=342:
  L22 top1=347 target_rank=2 generated_rank=18 target-generated_logit=+0.56
  L23 top1=348 target_rank=1 generated_rank=46 target-generated_logit=+1.62
  L24 top1=350 target_rank=4 generated_rank=3  target-generated_logit=-0.12
  L25 top1=344 target_rank=3 generated_rank=10 target-generated_logit=+0.50
  L26 top1=347 target_rank=2 generated_rank=11 target-generated_logit=+1.25
  L27 top1=342 target_rank=6 generated_rank=1  target-generated_logit=-0.50

19432 random target_y2=348 generated_y2=343:
  L22 top1=347 target_rank=2 generated_rank=24 target-generated_logit=+0.69
  L23 top1=348 target_rank=1 generated_rank=86 target-generated_logit=+1.75
  L24 top1=350 target_rank=3 generated_rank=6  target-generated_logit=+0.19
  L25 top1=347 target_rank=3 generated_rank=10 target-generated_logit=+0.25
  L26 top1=347 target_rank=2 generated_rank=12 target-generated_logit=+1.00
  L27 top1=343 target_rank=7 generated_rank=1  target-generated_logit=-0.38
```

Aggregate within this narrow panel:

```text
12670 sorted:
  target_y2 rank <= 10: 0 / 12 rows
  generated_y2 rank <= 10: 2 / 12 rows
  mean target_y2 rank: 703.9
  mean target_y2 minus generated_y2 logit: -2.69

19432 random:
  target_y2 rank <= 10: 12 / 12 rows
  generated_y2 rank <= 10: 6 / 12 rows
  mean target_y2 rank: 3.0
  mean target_y2 minus generated_y2 logit: +0.57
```

Interpretation:

1. `19432/random/chair` contains a strong target-y2 signal before final
   emission. At layer 23 the exact target y2 `348` is coordinate top-1 for both
   chair targets. Later layers keep the target neighborhood alive, but the
   final layer shifts top-1 to the actually emitted near-target bins `342/343`.
   This looks like a local closure/calibration choice inside the correct chair
   basin, not missing visual perception.

2. `12670/sorted/person` does not show an analogous hidden target-y2 reservoir.
   Across layers 22-27, target y2 is never top-10 and is usually hundreds of
   ranks down. By layer 27, the emitted shallow y2 `415` is coordinate rank 1
   for both targets. This supports the sharper diagnosis that forcing
   x1+y1+x2 leaves the state in a shallow vertical-extent basin rather than in
   a target-person closure basin.

3. The contrast suggests the next surgery handle. For `19432`, a final-layer or
   late-layer calibration patch could test why the state chooses a slightly
   low y2 despite earlier exact target evidence. For `12670`, a final-layer
   y2 patch alone is less likely to explain the mechanism; the deeper question
   is where the target-person vertical extent fails to enter the residual
   stream. The higher-value next probe is therefore a region/value-path
   comparison over the y2 decision state, using `19432` as the positive
   closure reference and `12670` as the missing-closure negative.

## Y2 Route Probe

Artifact scope: the same four forced `x1+y1+x2 -> y2` rows used by the y2
closure readout panel. This keeps the unit of study fixed at the hard
sample-base level instead of drifting into easy-image aggregate metrics.

Artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_plan/v1_force_x1_y1_x2
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route/v1_12670_sorted_layers22_27_allheads_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route/v1_19432_random_layers22_27_allheads_gpu1
```

Tooling:

```text
src/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_probe.py
scripts/analysis/run_prefix_denoising_anchor_escape_y2_route.py
tests/analysis/test_prefix_denoising_anchor_escape_y2_route_probe.py
```

Probe design:

```text
position:
  next token after forced target x1+y1+x2

direction:
  effective coord output(target_y2) - effective coord output(generated_y2)

effective coord output:
  lm_head coord row plus token_embeddings_adapter coord-row offset

layers:
  22, 23, 24, 25, 26, 27

heads:
  all attention heads

value-source regions:
  image_tokens
  prompt_non_image
  assistant_prefix_prior_objects
  current_object_ref
  current_box_start
  current_forced_coords
  current_partial_object
  recent_16
  all_prefix
```

Run health:

```text
12670 sorted:
  plan rows: 2
  route rows: 1728
  errors: 0

19432 random:
  plan rows: 2
  route rows: 1728
  errors: 0
```

Region-level route summary:

```text
12670 sorted mean target-minus-generated projection:
  image_tokens:              +0.0023
  current_object_ref:        -0.0073
  current_box_start:         -0.0110
  current_forced_coords:     -0.0587
  current_partial_object:    -0.0769
  recent_16:                 -0.0787
  prompt_non_image:          -0.1677
  all_prefix:                -0.2423

19432 random mean target-minus-generated projection:
  image_tokens:              +0.0007
  current_object_ref:        +0.0017
  current_box_start:         -0.0008
  prompt_non_image:          +0.0611
  all_prefix:                -0.1482
  recent_16:                 -0.2070
  current_partial_object:    -0.2074
  current_forced_coords:     -0.2084
```

The mean table is intentionally not the conclusion. It mostly says the final
state is dominated by prompt/template mass and local coordinate scaffold mass,
while direct image-token contribution at this exact y2 decision is close to
zero in both hard rows. The useful signal comes from layer/head-local routes.

Layer/head-local route findings:

```text
19432 random:
  L23 head 6 is the strongest positive current-object route.
  For the chair target that finally emits y2=342:
    current_forced_coords projection:  +4.16
    current_partial_object projection: +4.16
    recent_16 projection:              +4.70
  For the chair target that finally emits y2=343:
    current_forced_coords projection:  +4.67
    current_partial_object projection: +4.67
    recent_16 projection:              +5.38
    top source tokens include <|coord_651|>, <|coord_537|>, <|coord_122|>
  This coincides with the y2 readout result where L23 makes target y2=348
  coordinate top-1 for both selected chair rows.

19432 random:
  L27 head 14 is the strongest negative current-object route.
  For generated y2=343:
    current_forced_coords projection:  -13.17
    current_partial_object projection: -13.17
    recent_16 projection:              -13.18
  For generated y2=342:
    current_forced_coords projection:  -10.56
    current_partial_object projection: -10.56
    recent_16 projection:              -10.56
  This coincides with final-layer readout shifting top-1 away from target
  y2=348 and toward the actually emitted near-target bins 342/343.

12670 sorted:
  L23 head 6 has an isolated positive current-coordinate route for the
  deeper person target y2=793:
    current_forced_coords projection:  +5.85
    current_partial_object projection: +5.86
    recent_16 projection:              +5.86
  However the readout still ranks target y2 very poorly, and the mean
  current-object route over layers 22-27 is weak or negative. This is not a
  stable target-y2 reservoir comparable to 19432/random.
```

Mechanism interpretation:

1. `19432/random/chair` now has a concrete two-stage candidate mechanism. L23
   head 6 carries a positive target-y2 route from the forced coordinate
   scaffold/current object, but L27 head 14 carries a much stronger negative
   route from the same local scaffold. The likely failure is not "the chair is
   unseen"; it is a late local closure attractor overwriting an earlier exact
   y2 state.

2. `12670/sorted/person` remains the negative reference. The model can attend
   to the forced coordinate scaffold, and one head can produce an isolated
   positive target-y2 direction for one deeper person, but the target vertical
   extent does not become a layer-stable readout basin. The deeper question is
   upstream: why the person bottom boundary never becomes a robust residual
   feature under this sample-base/model pair.

3. Image-token routes are near zero at this late y2 decision position. This
   does not prove visual information is absent; it means the immediate y2
   closure state is mediated mostly through cached language/coordinate
   scaffold and residual basins. The visual-origin question should be pushed
   earlier in the tensor flow or tested by visual-region patch/masking, not
   inferred from this late attention route alone.

Immediate causal tests:

```text
19432/random:
  Patch or ablate L23 head 6 at the y2 decision.
  Prediction: weakening it should reduce target-y2 rank before final closure;
  strengthening it may preserve target y2 only if L27 head 14 does not
  overwrite the state.

19432/random:
  Patch or ablate L27 head 14 at the y2 decision.
  Prediction: subtracting the generated-y2 attractor route should move final
  y2 from 342/343 toward target y2=348 if this head is causal.

12670/sorted:
  Patch the isolated L23 head-6 positive route or transplant a matched
  target-y2 direction into the current-object route.
  Prediction: if target y2 still does not enter top ranks, the missing-closure
  mechanism lies upstream of this late route; if it improves sharply, the
  failure is late amplification/suppression rather than perception.
```

## Y2 Head Intervention Causal Panel

Artifact scope: the same four hard y2 route-plan rows, now with direct
attention-head scaling at the `self_attn.o_proj` input for the y2 decision
token. This is still a sample-base causal probe, not a val200 metric.

Artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_head_intervention/v1_smoke_one_row_l27h14_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_head_intervention/v1_12670_sorted_l23h6_l27h14_scales_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_head_intervention/v1_19432_random_l23h6_l27h14_scales_gpu1
```

Tooling:

```text
src/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_head_intervention.py
scripts/analysis/run_prefix_denoising_anchor_escape_y2_head_intervention.py
tests/analysis/test_prefix_denoising_anchor_escape_y2_head_intervention.py
```

Probe design:

```text
position:
  next token after forced target x1+y1+x2

intervention:
  scale one attention head slice before self_attn.o_proj at the y2 query token

heads:
  L23 head 6
  L27 head 14

scales:
  -1, 0, 0.5, 1, 1.5, 2

readout:
  true next-token logits at the y2 decision token
  target y2 rank and target-minus-generated-y2 logit margin
```

Run health:

```text
12670 sorted:
  plan rows: 2
  intervention rows: 24
  errors: 0

19432 random:
  plan rows: 2
  intervention rows: 24
  errors: 0
```

Key causal results:

```text
19432 random, L27 head 14:
  scale -1:
    target ranks 6/7 -> 2/2
    top1 y2 342/343 -> 347/347
    target-minus-generated margin delta +0.75 / +0.75
  scale 0:
    target ranks 6/7 -> 2/2
    top1 y2 342/343 -> 347/347
    margin delta +0.50 / +0.375
  scale 2:
    target ranks 6/7 -> 15/15
    top1 y2 342/343 -> 342/358
    margin delta -0.25 / -0.25

19432 random, L23 head 6:
  scale -1:
    target ranks 6/7 -> 5/2
    top1 y2 342/343 -> 347/347
    margin delta +0.375 / +0.50
  scale 0:
    target ranks 6/7 -> 5/5
    top1 y2 342/343 -> 342/347
    margin delta +0.375 / +0.25
  scale 2:
    target ranks 6/7 -> 15/14
    top1 y2 342/343 -> 342/342
    margin delta -0.25 / 0.0

12670 sorted, L23 head 6:
  scale -1:
    target ranks 571/474 -> 693/571
    top1 y2 415/415 -> 313/171
    mean margin delta -1.22
  scale 0:
    target ranks 571/474 -> 664/518
    top1 y2 415/415 -> 430/421
    mean margin delta -1.06
  scale 2:
    target ranks 571/474 -> 456/442
    top1 y2 415/415 -> 609/415
    mean margin delta +0.78

12670 sorted, L27 head 14:
  scaling has only small effects and never moves target y2 into a plausible
  high-rank basin.
```

Interpretation update:

1. `19432/random` L27 head 14 is now causal for the late shallow/near-y2
   attractor. Weakening or inverting this head moves both chair closures from
   emitted `342/343` to near-target `347` and improves target y2 to rank 2.
   Strengthening the same head worsens the target rank to 15. This supports a
   late overwrite mechanism: the model has the target neighborhood available,
   but one late attention head helps choose a lower/nearby closure basin.

2. `19432/random` L23 head 6 is more interesting than the route projection
   alone suggested. The route probe showed positive target-minus-generated
   value projection at L23, but causal scaling says reducing or inverting the
   whole head improves final y2, while amplifying it worsens the final rank.
   This means a one-step target-projection readout is insufficient: the head
   likely carries a mixed vector whose downstream effect helps organize the
   later closure attractor even though part of its local value contribution
   aligns with target y2.

3. `12670/sorted` L23 head 6 carries a real but insufficient person-height
   direction. Amplifying it improves target rank and, for target y2 `793`,
   moves top1 from `415` to `609`; however the true target remains hundreds of
   ranks down. This is exactly the negative-control behavior we wanted:
   a late coordinate head can push the state away from the shallow `415` basin,
   but it does not contain enough target-specific bottom-boundary information
   to solve the missing vertical extent.

4. The representative sample-base split is therefore justified. `19432/random`
   is a near-closure case with a causal late overwrite head. `12670/sorted` is
   a missing-extent case where the same intervention family exposes a generic
   height route but not the target boundary. These are not merely different
   metric examples; they instantiate different internal failure mechanisms.

Next high-value probe:

```text
Do not broaden to normal images yet.

For 19432/random:
  isolate L27 head 14 value-source subregions by patching/removing only the
  current_forced_coords/current_partial_object contribution. This should test
  whether the causal overwrite is specifically carried by the local coordinate
  scaffold rather than by prompt/template mass.

For 12670/sorted:
  move upstream and ask where target-specific person bottom boundary evidence
  should enter: visual-region patch/mask or earlier visual-to-language route
  probes are higher value than more late y2 rank nudges.
```

## Y2 Value-Region Intervention Panel

Artifact scope: the same four hard y2 route-plan rows, now with causal
subregion interventions inside selected attention-head value routes. For each
selected head, the probe captures the baseline attention-weighted value
contribution for a source region, then replays the same y2 decision after
scaling only that region's pre-`self_attn.o_proj` head slice.

Artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_value_region_intervention/v1_smoke_19432_random_l27h14_forced_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_value_region_intervention/v1_12670_sorted_l23h6_l27h14_regions_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_value_region_intervention/v1_19432_random_l23h6_l27h14_regions_gpu1
```

Tooling:

```text
src/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_value_region_intervention.py
scripts/analysis/run_prefix_denoising_anchor_escape_y2_value_region_intervention.py
tests/analysis/test_prefix_denoising_anchor_escape_y2_value_region_intervention.py
```

Probe design:

```text
position:
  next token after forced target x1+y1+x2

heads:
  L23 head 6
  L27 head 14

regions:
  current_forced_coords
  current_partial_object
  recent_16
  prompt_non_image
  all_prefix

region scales:
  0, 1, 2

readout:
  true next-token logits at the y2 decision token
  target y2 rank and target-minus-generated-y2 logit margin
```

Run health:

```text
12670 sorted:
  plan rows: 2
  intervention rows: 60
  errors: 0

19432 random:
  plan rows: 2
  intervention rows: 60
  errors: 0
```

Key causal results:

```text
19432 random, L27 head 14, current_forced_coords:
  scale 0:
    target ranks 6/7 -> 2/2
    top1 y2 342/343 -> 347/347
    mean margin delta +0.4375
  scale 2:
    target ranks 6/7 -> 17/17
    top1 y2 342/343 -> 342/358
    mean margin delta -0.3125
  mean attention mass: 0.8352

19432 random, L27 head 14, current_partial_object:
  scale 0:
    target ranks 6/7 -> 2/2
    top1 y2 342/343 -> 347/347
    mean margin delta +0.4375
  scale 2:
    target ranks 6/7 -> 12/16
    top1 y2 342/343 -> 342/358
    mean margin delta -0.25
  mean attention mass: 0.8364

19432 random, L27 head 14, recent_16:
  scale 0:
    target ranks 6/7 -> 2/2
    top1 y2 342/343 -> 347/347
    mean margin delta +0.4375
  scale 2:
    target ranks 6/7 -> 14/16
    top1 y2 342/343 -> 342/358
    mean margin delta -0.25
  mean attention mass: 0.8378

19432 random, L27 head 14, prompt_non_image:
  scale 0:
    target ranks 6/7 -> 6/5
    top1 y2 342/343 -> 342/347
    mean margin delta +0.0625
  scale 2:
    target ranks 6/7 -> 7/6
    top1 y2 342/343 -> 343/347
    mean margin delta +0.125
  mean attention mass: 0.1591

12670 sorted, L23 head 6, current_forced_coords:
  scale 0:
    target ranks 571/474 -> 660/517
    top1 y2 415/415 -> 430/421
    mean margin delta -1.03125
  scale 2:
    target ranks 571/474 -> 469/448
    top1 y2 415/415 -> 609/415
    mean margin delta +0.75
  mean attention mass: 0.8765

12670 sorted, L23 head 6, prompt_non_image:
  scale 0:
    target ranks 571/474 -> 566/480
    top1 y2 remains 415/415
    mean margin delta 0.0
  scale 2:
    target ranks 571/474 -> 574/472
    top1 y2 415/415 -> 415/418
    mean margin delta +0.03125
  mean attention mass: 0.1194
```

Interpretation update:

1. The `19432/random` L27H14 overwrite is not a vague whole-head or prompt
   artifact. Removing only the local forced-coordinate/current-object value
   contribution reproduces the whole-head repair: target y2 improves to rank
   2 and top1 moves to `347`. Amplifying the same local contribution drives
   the state away from the target and toward `342/358`. The causal carrier is
   therefore the local coordinate scaffold inside L27H14.

2. `prompt_non_image` is not the primary overwrite carrier for `19432/random`
   L27H14. It has much smaller attention mass and much weaker, partly
   non-monotone effects. This matters because the route probe saw large prompt
   mass in many heads; the causal subregion probe says the interesting y2
   collapse is carried by the object-local coordinate route.

3. `12670/sorted` L23H6 mirrors the negative-control story at subregion level.
   The current/recent coordinate scaffold carries a real generic height
   direction: removing it worsens target rank and amplifying it improves rank
   and can move the deeper target top1 from `415` to `609`. But even this
   amplified local route does not make the true target y2 (`793` or `608`) a
   high-rank closure. It is a generic vertical-extension handle, not a
   target-boundary representation.

4. The current mechanism picture has split into two cases:
   `19432/random` fails by late local coordinate-scaffold overwrite within an
   otherwise available target basin; `12670/sorted` fails because the target
   bottom boundary never becomes available as a strong coordinate-specific
   basin, even though a generic height-extension route exists.

## Y2 Value-Region Continuation Panel

Artifact scope: the value-region intervention now optionally uses the patched
y2 logits as the first greedy token and then continues unpatched for a short
tail. The continuation labels include exact target closure and near-coordinate
closure radii, so a one-bin y2 basin movement is not collapsed into the same
bucket as a far miss.

Artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_value_region_intervention/v3_continuation_19432_random_l27h14_regions_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_value_region_intervention/v3_continuation_12670_sorted_l23h6_regions_gpu1
```

Tooling update:

```text
src/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_value_region_intervention.py
scripts/analysis/run_prefix_denoising_anchor_escape_y2_value_region_intervention.py --continuation-steps 4
tests/analysis/test_prefix_denoising_anchor_escape_y2_value_region_intervention.py
```

Run health:

```text
19432 random, L27H14:
  plan rows: 2
  intervention rows: 30
  errors: 0

12670 sorted, L23H6:
  plan rows: 2
  intervention rows: 30
  errors: 0
```

Key continuation results:

```text
19432 random, L27H14, current_forced_coords:
  scale 0:
    baseline y2 tails: <|coord_342|><|box_end|>, <|coord_343|><|box_end|>
    patched y2 tails:  <|coord_347|><|box_end|>, <|coord_347|><|box_end|>
    target y2: 348
    distance to target: 6/5 -> 1/1
    target ranks: 6/7 -> 2/2
    mean margin delta: +0.4375
    exact target closures: 0/2
    radius-1 near-target closed tails: 2/2

19432 random, L27H14, current_forced_coords:
  scale 2:
    patched y2 tails: <|coord_342|><|box_end|>, <|coord_358|><|box_end|>
    distance to target: 6/10
    target ranks: 17/17
    mean margin delta: -0.3125

19432 random, L27H14:
  scale 0 for current_forced_coords/current_partial_object/recent_16/all_prefix:
    first token changed in 2/2 rows
    radius-1 near-target closed tails in 2/2 rows
  scale 0 for prompt_non_image:
    first token changed in 1/2 rows
    radius-1 near-target closed tails in 1/2 rows

12670 sorted, L23H6, current_forced_coords:
  scale 0:
    baseline y2 tails: <|coord_415|><|box_end|>, <|coord_415|><|box_end|>
    patched y2 tails:  <|coord_430|><|box_end|>, <|coord_421|><|box_end|>
    target y2: 793 / 608
    distance to target: 378/193 -> 363/187
    target ranks: 571/474 -> 660/517
    mean margin delta: -1.03125

12670 sorted, L23H6, current_forced_coords:
  scale 2:
    patched y2 tails: <|coord_609|><|box_end|>, <|coord_415|><|box_end|>
    distance to target: 184/193
    target ranks: 469/448
    mean margin delta: +0.75
    exact or radius-8 near-target closed tails: 0/2
```

Interpretation update:

1. `19432/random` is now an emission-level local-overwrite case, not only a
   first-logit rank effect. Removing the L27H14 local forced-coordinate/current
   object contribution changes the actual two-token tail from the duplicated
   shallow bottom (`342/343`) to a clean near-target closure
   `<|coord_347|><|box_end|>`. The exact supervised bin is `348`, so the
   remaining issue is a tight coordinate-basin calibration/smoothness question,
   not a missing box-tail grammar question.

2. The causal support for `19432/random` remains object-local. `recent_16`,
   `current_partial_object`, and `current_forced_coords` reproduce the same
   radius-1 near-target closure. `prompt_non_image` is weaker and only repairs
   one of the two rows at scale 0.

3. `12670/sorted` remains the complementary hard negative. L23H6 can push the
   y2 tail upward in a generic way and can emit `609` for the deeper person,
   but it never approaches the true bottom boundary `793`, and the second row
   with target `608` still closes at `415`. This is not an exact-bin
   calibration problem; it is a missing or unbound visual bottom-boundary
   reservoir at the y2 state.

## Y2 Coordinate-Basin Readout Panel

Artifact scope: the same two hard sample-base value-region panels were rerun
with richer coordinate-basin readouts attached to the exact same y2 logits:
coord-only entropy, mean/std coordinate bin, generated-basin radius mass, and
local windows around target/generated/top1 coordinate centers.

Artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_value_region_intervention/v4_basin_19432_random_l27h14_regions_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_value_region_intervention/v4_basin_12670_sorted_l23h6_regions_gpu1
```

Tooling update:

```text
src/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_head_intervention.py
src/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_value_region_intervention.py
tests/analysis/test_prefix_denoising_anchor_escape_y2_head_intervention.py
```

Run health:

```text
19432 random, L27H14:
  plan rows: 2
  intervention rows: 30
  errors: 0

12670 sorted, L23H6:
  plan rows: 2
  intervention rows: 30
  errors: 0
```

Key basin results:

```text
19432 random, L27H14, current_forced_coords scale 0:
  top1: 342/343 -> 347/347
  target bin: 348
  target radius-1 mass: +0.0120 mean
  target radius-8 mass: -0.0129 mean
  entropy: +0.0581 mean
  coord mean bin: +2.25 mean
  continuation: radius-1 near-target closed tails in 2/2 rows

19432 random, local window examples:
  row with baseline top1 342:
    baseline logits: 342=25.000 rank1, 347=24.875 rank2, 348=24.500 rank6
    scale0 logits:  347=24.625 rank1, 348=24.375 rank2, 342=24.375 rank2
  row with baseline top1 343:
    baseline logits: 343=22.750 rank1, 347=22.750 rank1, 348=22.375 rank7
    scale0 logits:  347=22.375 rank1, 348=22.000 rank2, 343=22.000 rank2
  scale2 worsens exact target:
    target 348 rank becomes 17/17, with first tokens 342/358

19432 random, region contrast:
  current_forced_coords/current_partial_object/recent_16/all_prefix scale0:
    radius-1 target mass increases about +0.011 to +0.012
    entropy increases about +0.058 to +0.063
    coord mean shifts upward about +2.2 bins
    first token becomes 347 in 2/2 rows
  prompt_non_image scale0:
    target radius-1 mass is essentially unchanged
    first token becomes 347 in only 1/2 rows

12670 sorted, L23H6, current_forced_coords scale 2:
  target 793 row:
    top1: 415 -> 609
    target radius-32 mass: 0.0242 -> 0.0487
    generated radius-32 mass: 0.1656 -> 0.1116
    coord mean bin: 517.1 -> 550.9
    target rank: 571 -> 469
    local top example: 609 logit 13.25 rank1; 793 logit 11.56 rank469
  target 608 row:
    top1 stays 415
    target radius-32 mass: 0.0290 -> 0.0365
    generated radius-32 mass: 0.2888 -> 0.2418
    coord mean bin: 405.6 -> 416.6
    target rank: 474 -> 448
    local target neighborhood remains buried: 608 rank448, 609 rank420

12670 sorted, region contrast:
  local/current/recent/all_prefix scale2:
    target radius-32 mass increases only about +0.016 mean
    generated radius-32 mass decreases about -0.051 mean
    coord mean shifts upward about +22 to +23 bins
    no exact or radius-8 near-target closure
  prompt_non_image scale2:
    near-zero target-basin movement
```

Interpretation update:

1. `19432/random` is not missing a target neighborhood. The target-near basin
   already exists, and removing the local L27H14 coordinate contribution
   reorders a tight neighborhood from `342/343` toward `347`. The exact target
   `348` rises to rank 2 in both rows but remains slightly below `347`. The
   remaining exact-bin problem is therefore local coordinate-neighbor
   preference or embedding/readout calibration, not absent perception or broken
   box-tail syntax.

2. The random-denoise L27H14 contribution behaves like a local attractor
   sharpening/tilting term. Removing it raises entropy slightly and shifts the
   coordinate mean upward by about two bins into the target-near region.
   Amplifying it sharpens the distribution away from exact `348` and can push
   one row to the farther `358` attractor.

3. `12670/sorted` is a different failure family. L23H6 scale2 moves the
   distribution upward and reduces mass around the duplicated shallow basin,
   but the true target neighborhoods remain low-probability islands. For the
   deep target `793`, even after scale2 the emitted top is `609` and the true
   target remains rank 469. For the target `608` row, even the neighboring
   `609` remains rank 420. This supports the bottom-boundary-reservoir
   hypothesis over a simple local coordinate calibration explanation.

## Y2 Coord-Token Geometry And Adapter Decomposition Panel

Artifact scope: selected-bin coordinate surface geometry and y2-state logit
decomposition for the same primary hard sample bases. The probe separates
static base rows, token-embeddings-adapter head offsets, effective output rows,
and the actual y2 hidden-state dot products.

Artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coord_token_geometry_probe/v2_19432_random_l27h14_bins_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coord_token_geometry_probe/v2_12670_sorted_l23h6_bins_gpu1
```

Tooling:

```text
src/analysis/prefix_denoising_surgery_probing/coord_token_geometry_probe.py
scripts/analysis/run_prefix_denoising_coord_token_geometry_probe.py
tests/analysis/test_prefix_denoising_coord_token_geometry_probe.py
```

Run health:

```text
19432 random:
  plan rows: 2
  rows: 500
  errors: 0

12670 sorted:
  plan rows: 2
  rows: 500
  errors: 0
```

Static geometry results:

```text
effective_output 347 vs 348:
  random: euclidean 0.2043, cosine 0.9880
  sorted: euclidean 0.2040, cosine 0.9881

effective_output 608 vs 609:
  random: euclidean 0.2017, cosine 0.9881
  sorted: euclidean 0.2025, cosine 0.9880

effective_output 415 vs 608:
  random: euclidean 0.9102, cosine 0.7597
  sorted: euclidean 0.9089, cosine 0.7603

adapter_head_offset 347 vs 348:
  random: euclidean 0.0378, cosine 0.7388
  sorted: euclidean 0.0353, cosine 0.7846

adapter_head_offset 608 vs 609:
  random: euclidean 0.0449, cosine 0.5729
  sorted: euclidean 0.0472, cosine 0.5678
```

Interpretation: the effective output rows still preserve strong adjacent
coordinate locality, because the base output surface dominates. The learned
adapter head-offset surface is much less smooth locally and can tilt neighboring
bins, but it is small in norm relative to the base rows. The exact y2 behavior
therefore depends on interaction between hidden-state/base-row preference and
adapter-row tilt.

Dynamic decomposition results:

```text
19432 random, target 348, generated 343 state:
  actual model logits:
    343 = 22.750 rank1
    347 = 22.750 rank2
    348 = 22.375 rank8
  base-output logits:
    343 = 22.065
    342 = 21.893
    347 = 21.260
    348 = 21.184
  adapter deltas:
    347 = +1.473
    348 = +1.218
    342 = +0.778
    343 = +0.581

19432 random, target 348, generated 342 state:
  actual model logits:
    342 = 25.000 rank1
    343 = 24.875 rank2
    347 = 24.875 rank4
    348 = 24.500 rank8
  base-output logits:
    343 = 24.267
    342 = 24.068
    358 = 23.717
    347 = 23.352
    348 = 23.209
  adapter deltas:
    347 = +1.479
    348 = +1.301
    342 = +0.812
    343 = +0.579
```

For `19432/random`, the adapter is actively trying to pull the state from the
342/343 shallow-anchor region into the 347/348 neighborhood, but it has a
local preference for `347` over exact `348` by about `0.18` to `0.25` logit.
The base hidden/readout component still favors 342/343, so the final state is
best understood as a tug-of-war: repeated-anchor hidden evidence plus a
nonsmooth adapter-row rescue that lands one bin short.

```text
12670 sorted, target 608, generated 415 state:
  actual model logits:
    415 = 14.375 rank1
    421 = 14.375 rank3
    608 = 11.688 rank476
    609 = 11.875 rank451
  base-output logits:
    421 = 14.312
    430 = 14.103
    415 = 13.998
    608 = 11.870
  adapter deltas:
    415 = +0.401
    421 = +0.033
    608 = -0.188
    609 = +0.036

12670 sorted, target 793, generated 415 state:
  actual model logits:
    415 = 13.312 rank1
    609 = 13.250 rank3
    608 = 13.000 rank53
    793 = 10.875 rank576
  base-output logits:
    430 = 13.463
    421 = 13.358
    415 = 13.177
    609 = 12.859
    608 = 12.730
    793 = 10.442
  adapter deltas:
    793 = +0.464
    609 = +0.371
    608 = +0.242
    415 = +0.113
```

For `12670/sorted`, the target-bottom bins are already far behind in the base
hidden/readout term. The adapter sometimes helps the far target (`793`) but not
nearly enough to overcome the hidden-state preference for the shallow/generated
region, and in the `608` row the adapter actually favors generated `415` over
target `608`. This strengthens the interpretation that the bottom-boundary
evidence is absent or unbound before the final coordinate surface, rather than
merely distorted by coordinate-token geometry.

Mechanism update:

1. `19432/random` is a local rescue-with-offset-bias case. The model has a
   viable 347/348 neighborhood; the token-embeddings-adapter surface gives a
   strong positive correction to both, but slightly more to `347`, while the
   base hidden/readout term still pulls toward 342/343.

2. `12670/sorted` is a state-formation failure, not a final-row geometry
   failure. The final coordinate surface preserves locality, and the adapter
   is too small or misdirected relative to the missing bottom-boundary hidden
   evidence.

Next high-value probe:

```text
Do not spend on normal images yet.

For 19432/random:
  run post-hoc output-surface counterfactuals at the same y2 hidden states:
  zero/swap/smooth adapter head offsets for 342/343/347/348/349/358 and ask
  whether exact 348 wins without changing the image or prefix.

For 12670/sorted:
  move upstream to visual-origin/bottom-boundary tracing. The local coordinate
  scaffold is already proven insufficient; the next question is whether a
  visual-region or earlier-layer signal for the person bottom exists but fails
  to bind, or never becomes available at all.
```

## Revised Sample-Base Panel For Next Round

Selection rule: prefer representative hard image-bases with mechanistic
contrast, not normal or well-learned cases. Easy images are useful only as
sanity controls after the mechanism handle is wired.

Primary panel:

```text
12670 / sorted_denoise:
  role: hard negative y2-closure / missing or unbound bottom-boundary reservoir
  evidence: forced x1+y1+x2 still emits y2=415; amplifying L23H6 local
    coordinate value can produce y2=609 but stays far from targets 793/608
  next use: upstream visual-origin and tensor-flow tracing for person bottom

19432 / random_denoise:
  role: hard positive near-closure / local y2 overwrite basin
  evidence: removing L27H14 local coordinate/current-object value turns
    duplicated y2=342/343 into radius-1 near-target y2=347 followed by box_end
  next use: coordinate-token basin and exact-bin calibration around 347/348

16228 / random_denoise:
  role: same-image tail-binding contrast
  evidence: previous object-state selector found both first-token-only and
    tail-coherent states in the same image/model
  next use: distinguish x1 repair from full tail binding without changing image

2685 / random_denoise:
  role: compact false-negative / anchor-collapse contrast
  evidence: previous notes show post_x2_pre_y2 can be tail-coherently repaired,
    while pre_x1 repair alone often stays first-token-only
  next use: test whether y2 route is available only after enough coordinate
    scaffold has been supplied

2157 / sorted_denoise:
  role: state-entry first-token-only failure outside the person/backpack pair
  evidence: object-state selector rank 6, sorted_denoise object 13 pre_x1
  next use: broaden negative subtype beyond 12670 without going broad-metric

2299 / random_denoise:
  role: termination/boundary contrast, not primary y2 closure
  evidence: random_denoise empty/termination fault while sorted_denoise predicts
  next use: keep separate from y2 closure; use for stop/continue router work
```

This panel should be treated as dynamic. If a new probe finds a particularly
informative route, it is acceptable to dive deeper on that image-base before
expanding breadth. The working criterion is influence on the final mechanism
picture, not equal coverage of every selected case.

## Adapter-Head Counterfactuals At Fixed Y2 States

New probe code:

```text
src/analysis/prefix_denoising_surgery_probing/coord_token_geometry_probe.py
scripts/analysis/run_prefix_denoising_coord_token_geometry_probe.py
```

Artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coord_token_geometry_probe/v3_counterfactual_19432_random_bins_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coord_token_geometry_probe/v3_counterfactual_12670_sorted_bins_gpu1
```

Scope:

```text
readout_only=true
training_ran=false
row_type=state_adapter_counterfactual
modes=observed,zero_selected_adapter,swap_target_best_neighbor_adapter,smooth_selected_adapter
```

This is a post-hoc output-surface counterfactual: image, assistant prefix,
hidden state, base output rows, and lm-head readout vector are fixed. Only the
local `token_embeddings_adapter` head-offset rows are changed. It asks whether
the exact target coordinate would win if the local coord-token adapter surface
were smoother, absent, or swapped with its best immediate neighbor.

`19432/random`, target `348`, generated `343` state:

```text
observed:
  top1=347
  target rank=9
  generated rank=3
  target_minus_generated=-0.2434

zero selected adapter rows:
  top1=361
  target rank=46
  generated rank=16
  target logit delta=-1.2178

swap target 348 with best neighbor 347:
  top1=342
  target rank=2
  generated rank=3
  target_minus_generated=+0.0113
  target logit delta=+0.2547

smooth selected adapter rows:
  top1=343
  target rank=4
  generated rank=1
  target logit delta=+0.0762
```

`19432/random`, target `348`, generated `342` state:

```text
observed:
  top1=342
  target rank=7
  generated rank=1
  target_minus_generated=-0.3696

zero selected adapter rows:
  top1=344
  target rank=36
  generated rank=14
  target logit delta=-1.3011

swap target 348 with best neighbor 347:
  top1=342
  target rank=4
  generated rank=1
  target_minus_generated=-0.1919
  target logit delta=+0.1777

smooth selected adapter rows:
  top1=343
  target rank=7
  generated rank=3
  target logit delta=+0.0538
```

Interpretation for `19432/random`: the local adapter surface matters, but it is
not the whole mechanism. Removing the local adapter rows makes the target much
worse, confirming that the adapter contributes real rescue mass around the
correct y2 neighborhood. Swapping `348` with neighbor `347` improves the exact
target substantially, but does not make it top-1; a base hidden/readout basin
around `342/343/344` still competes. This is therefore not merely an adapter-row
calibration problem. It is a coupled problem: the hidden state is close enough
that local adapter shape can move target rank, but still anchored enough that
exact closure needs upstream state movement.

`12670/sorted`, target `608`, generated `415` state:

```text
observed:
  top1=415
  target rank=479
  generated rank=1
  target_minus_generated=-2.7172

zero selected adapter rows:
  top1=432
  target rank=452
  generated rank=22
  target logit delta=+0.1877

swap target 608 with best neighbor 609:
  top1=415
  target rank=448
  generated rank=1
  target logit delta=+0.2238

smooth selected adapter rows:
  top1=432
  target rank=473
  generated rank=6
  target logit delta=+0.0521
```

`12670/sorted`, target `793`, generated `415` state:

```text
observed:
  top1=415
  target rank=572
  generated rank=1
  target_minus_generated=-2.3840

zero selected adapter rows:
  top1=430
  target rank=596
  generated rank=10
  target logit delta=-0.4639

swap target 793 with best neighbor 792:
  top1=415
  target rank=562
  generated rank=1
  target logit delta=+0.0770

smooth selected adapter rows:
  top1=612
  target rank=578
  generated rank=7
  target logit delta=-0.0599
```

Interpretation for `12670/sorted`: local coord-token adapter surgery cannot
explain the failure. The target bins stay hundreds of ranks below the shallow
generated/nearby bins. The adapter can nudge ranks, but there is no plausible
post-hoc local-row edit that turns the buried bottom-coordinate target into a
winner at the fixed hidden state. This strengthens the upstream state-formation
hypothesis: the model has not prepared the right bottom-boundary evidence, or
has failed to bind it to the current object, before the final coordinate
surface is consulted.

Revised next directions:

```text
19432/random:
  use as the narrow "near basin but exact miss" case.
  Next probe should move the hidden/readout vector itself: interpolate or patch
  late residual/layer streams toward exact 348 while keeping adapter rows fixed.
  The key question is how much upstream movement is required after local
  adapter calibration is made favorable.

12670/sorted:
  use as the narrow "bottom-boundary reservoir absent or unbound" case.
  Next probe should trace visual-region and object-binding tensor flow across
  earlier/middle layers, not spend further on final coord-row geometry.

16228/random and 2685/random:
  keep as second-stage controls for tail coherence and coordinate scaffold
  dependence after the 19432/12670 mechanisms are pinned down.
```

## Representative Sample-Base Panel

New selector:

```text
src/analysis/prefix_denoising_surgery_probing/representative_sample_base_panel.py
scripts/analysis/run_prefix_denoising_representative_sample_base_panel.py
tests/analysis/test_prefix_denoising_representative_sample_base_panel.py
```

Preferred current artifact:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/representative_sample_base_panel/v3_current_deep_panel_terminal_dense
```

Materialization command:

```text
python scripts/analysis/run_prefix_denoising_representative_sample_base_panel.py \
  --object-state-selection /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/object_state_selection/v4_v8_v9_component_site_join_diverse/object_state_selection.jsonl,/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/object_state_selection/v5_broad_image_coverage_unpatched/object_state_selection.jsonl \
  --bridge-sample-base-selection /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/bridge_sample_base_selector/v1_current_tomography_panel/bridge_sample_base_selection.jsonl \
  --tail-binding-quality /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_tail_binding_reduce/v1_strict4_all_modes_iou_quality/anchor_escape_tail_binding_quality_rows.jsonl \
  --fn-guidance-rows /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_guidance_probe/v4_contextual_all_selected_fn_layers0_8_16_20_23_24_27_gpu0/contextual_fn_guidance_probe_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/representative_sample_base_panel/v3_current_deep_panel_terminal_dense \
  --select-limit 8 \
  --per-role-limit 2
```

The selector intentionally ranks image/model bases by mechanistic leverage
rather than normal-image coverage. It also applies two interpretation guards:

```text
1. forced full-box closure is not counted as hidden tail recovery;
   max_non_full_box_tail_target_iou is the emission-relevant tail field

2. contextual FN guidance ranks are counted from true forward logits only,
   not intermediate layer-logit-lens rows
```

Current selected microscope panel:

```text
12670 / sorted_denoise:
  role: hard negative y2 closure
  evidence: bridge best rank 444, non-full-box max target IoU 0.301
  next: upstream visual-origin and bottom-boundary tensor-flow tracing

19432 / random_denoise:
  role: recoverable hidden tail binding
  evidence: bridge rank 1, non-full-box max target IoU 0.996
  next: late overwrite, coordinate-ridge, and adapter counterfactual controls

16228 / random_denoise:
  role: same-image tail-binding contrast
  evidence: paired first-token-only and tail-coherent object states
  next: paired state-entry/tail-binding surgery over pre_x1 and direct tail slots

2157 / random_denoise:
  role: false-negative guidance
  evidence: forward-logit guidance rank 2 plus duplicate-anchor evidence
  next: object-start scaffold/guidance plus visual-route check

2157 / sorted_denoise:
  role: state-entry first-token-only
  evidence: selected object-state family state_entry_first_token_only
  next: earlier-layer or multi-slot coupling from x1 to y1/x2/y2

19109 / random_denoise:
  role: false-negative guidance
  evidence: forward-logit guidance rank 1 outside the main person/backpack pair
  next: object-start scaffold/guidance plus visual-route check

2685 / random_denoise:
  role: same-image tail-binding contrast
  evidence: compact object case with first-token-only versus direct-tail repair
  next: coordinate-scaffold dependence after the main 19432/12670 pair

2299 / random_denoise:
  role: termination boundary
  evidence: densest terminal-boundary object-state selection in this panel
  next: stop/continue router tie semantics, separate from y2 closure
```

This panel should be treated as a dynamic microscope queue. If a new path
shows unusually high influence on the final mechanism picture, it is acceptable
to dive deeper on that image-base before expanding breadth. The purpose is to
avoid spending GPU time on normal or well-learned scenes while still keeping
positive, negative, state-entry, FN-guidance, and terminal-router controls in
the same deterministic queue.

## Effective Coord-Readout Surgery At Fixed Y2 States

New probe row type:

```text
row_type=state_effective_readout_surgery
surface=effective_output
adapter rows fixed=true
```

Code and artifacts:

```text
src/analysis/prefix_denoising_surgery_probing/coord_token_geometry_probe.py
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coord_token_geometry_probe/v4_effective_readout_surgery_19432_random_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coord_token_geometry_probe/v4_effective_readout_surgery_12670_sorted_gpu1
```

Scope:

```text
19432/random: 80 effective-readout surgery rows, readout_status=ok
12670/sorted: 80 effective-readout surgery rows, readout_status=ok
alphas=0,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5
modes=observed_top1,generated_bin,best_target_neighbor,top8_centroid
```

This differs from residual readout surgery. It perturbs the hidden readout
vector against the effective coordinate output surface after adding
`token_embeddings_adapter` head offsets. The hidden state and adapter rows are
fixed; only the readout vector is moved along a target-minus-antagonist
direction. This directly probes coordinate-basin attraction on the final
effective coord surface.

`19432/random`, target `348`, generated `343` state:

```text
observed:
  top1=347
  target rank=9
  generated rank=3
  target_minus_generated=-0.2434

observed_top1 / best_target_neighbor antagonist=347:
  no exact flip up to alpha=0.5
  best target rank=5 at alpha=0.02
  top ridge migrates 347 -> 342 -> 349 -> 350

generated_bin antagonist=343:
  first exact flip at alpha=0.1
  target rank=1, generated rank=106

top8_centroid antagonists=347,342,343,358,344,361,349,346:
  first exact flip at alpha=0.05
  target rank=1, generated rank=23
```

`19432/random`, target `348`, generated `342` state:

```text
observed:
  top1=342
  target rank=7
  generated rank=1
  target_minus_generated=-0.3696

observed_top1 / generated_bin antagonist=342:
  first exact flip at alpha=0.2
  before flip, mass goes through 347

best_target_neighbor antagonist=347:
  no exact flip up to alpha=0.5
  top ridge migrates toward 350

top8_centroid antagonists=342,343,347,344,346,340,349,341:
  first exact flip at alpha=0.05
  target rank=1, generated rank=21
```

Interpretation for `19432/random`: exact-bin failure is a ridge phenomenon,
not a single-token competition. Pushing against `347` alone can make the target
beat generated `342/343`, but it does not make exact `348` win because the
winner migrates along the local ridge to `349/350`. Broad suppression of the
top ridge is much more efficient and flips exact `348` at alpha `0.05`. This
clarifies the previous adapter-counterfactual result: local adapter shape
helps, but the hidden readout is sitting on a multi-bin coordinate attractor.

`12670/sorted`, target `608`, generated `415` state:

```text
observed:
  top1=415
  target rank=479
  generated rank=1
  target_minus_generated=-2.7173

best_target_neighbor antagonist=609:
  no exact flip up to alpha=0.5
  best target rank=276; top ridge migrates to 450

observed_top1 / generated_bin antagonist=415:
  first exact flip at alpha=0.2
  alpha=0.1 reaches rank=3 but top remains 483

top8_centroid antagonists=415,432,418,421,420,416,433,431:
  first exact flip at alpha=0.2
  alpha=0.1 reaches rank=2 with top=609
```

`12670/sorted`, target `793`, generated `415` state:

```text
observed:
  top1=415
  target rank=572
  generated rank=1
  target_minus_generated=-2.3840

best_target_neighbor antagonist=792:
  no exact flip up to alpha=0.5
  best target rank=335; top ridge stays near 420

observed_top1 / generated_bin antagonist=415:
  first exact flip at alpha=0.2
  alpha=0.1 reaches rank=27 with top=731

top8_centroid antagonists=415,612,430,609,425,664,428,484:
  first exact flip at alpha=0.1
  target rank=1, generated rank=201
```

Interpretation for `12670/sorted`: the target is not unreachable in the final
effective coordinate space. A sufficiently strong readout-vector movement can
make the exact bottom target win. However, compared with `19432`, the required
movement is larger and passes through a different broad ridge: shallow person
bottoms around `415/421/432` plus intermediate bins such as `609/612/731`.
This argues against a simple "model cannot see the missing object" explanation.
The more precise mechanism is weak or unbound bottom-boundary evidence hidden
behind a strong shallow-coordinate ridge. The image may contain the needed
visual reservoir, but the current object state does not bind it strongly enough
before y2 emission.

Mechanism update:

```text
1. A coordinate basin is better modeled as a ridge/manifold of mutually
   substitutable bins, not an individual wrong token.

2. Local neighbor antagonism is often the wrong intervention. It may suppress
   one bin while revealing the next ridge member.

3. Broad top-ridge suppression is a stronger diagnostic for whether the exact
   coordinate is latent in the readout state.

4. 19432/random has a near latent target: top8-ridge suppression flips exact
   y2=348 at alpha=0.05.

5. 12670/sorted has a weaker or less bound target: exact y2=608/793 needs
   alpha=0.1 to 0.2 depending on ridge mode, despite being hundreds of ranks
   down at baseline.
```

Next high-value probe:

```text
Move upstream from final readout to tensor-flow attribution:

19432/random:
  locate where the multi-bin ridge 342/343/347/349/350 is formed, and whether
  object-local visual/value regions can sharpen it to exact 348 before y2.

12670/sorted:
  trace whether bottom-boundary visual evidence for target 608/793 appears in
  image-token or object-prefix states at middle layers, and where it loses to
  shallow person-bottom ridge evidence.

Avoid broad normal-image sweeps until this ridge-vs-binding mechanism is
localized across layers/attention/value flow.
```

## Value-Region Tensor-Flow Panel For Ridge Formation

Artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/value_region_intervention/v10_ridge_flow_19432_random_heads23_27_regions_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/value_region_intervention/v10_ridge_flow_12670_sorted_heads23_27_regions_gpu1
```

Scope:

```text
route_plan_rows=v1_force_x1_y1_x2
images=19432/random_denoise and 12670/sorted_denoise
heads=23:6,27:14
regions=current_forced_coords,current_partial_object,recent_16,prompt_non_image,all_prefix
region_scales=0,0.5,1,2,4
continuation_steps=2
19432 rows=100, readout_status=ok, error_count=0
12670 rows=100, readout_status=ok, error_count=0
```

High-level counters:

```text
19432/random:
  target_rank_improved_non_noop_count=34
  target_margin_improved_non_noop_count=46
  top1_flipped_to_target_non_noop_count=0

12670/sorted:
  target_rank_improved_non_noop_count=46
  target_margin_improved_non_noop_count=28
  top1_flipped_to_target_non_noop_count=0
```

`19432/random` value-flow pattern:

```text
L27H14 attends heavily to current object/forced-coordinate regions:
  target348/generated342 state:
    current_forced_coords attention_mass=0.8769, contribution_norm=266.90
    current_partial_object attention_mass=0.8780, contribution_norm=266.99
    recent_16 attention_mass=0.8793, contribution_norm=267.17

Removing L27H14 current object/forced-coordinate contribution (scale=0):
  generated342 state:
    top1 342 -> 347
    target rank 6 -> 2
    target-generated margin -0.50 -> 0.00
    continuation first token <|coord_347|>, second token <|box_end|>

Amplifying the same contribution (scale=4):
  generated342 state:
    top1 -> 358
    target rank worsens to 21-23
    margin drops to about -1.125 to -1.25

For generated343 state:
  L27H14 scale=0 similarly moves top1 343 -> 347 and target rank 7 -> 2.
  L27H14 scale=4 pushes top1 to 358 and target rank to 22-23.
```

Interpretation: in `19432/random`, the implicated late head does not look like
a clean exact-target carrier. Its high-mass current-object/current-coordinate
value contribution sustains the local ridge. Removing it sharpens the output
toward near-target `347` with a valid box tail; amplifying it pushes the y2
ridge outward to `358`. This explains why final readout surgery needed broad
ridge suppression: part of the current-object value stream itself is feeding a
nearby but wrong coordinate manifold.

`12670/sorted` value-flow pattern:

```text
L23H6 is the active value-flow site:
  target793/generated415 state:
    current_forced_coords attention_mass=0.9007, contribution_norm=148.65
    current_partial_object attention_mass=0.9013, contribution_norm=148.70
    recent_16 attention_mass=0.9015, contribution_norm=148.72

Amplifying L23H6 current object/forced-coordinate contribution (scale=4):
  target793/generated415 state:
    target rank 571 -> 125-136
    target-generated margin -2.4375 -> +0.3125
    top1 becomes 999
    continuation label remains not_repaired

Removing the same contribution (scale=0):
  target793/generated415 state:
    target rank worsens to about 659-664
    margin drops to about -3.94 to -4.00
    top1 becomes 430

For target608/generated415:
  L23H6 scale=4 improves target rank 474 -> 343-351 and margin -2.6875 -> about -1.56.
  But top1 also becomes 999, and continuation remains not_repaired.

L27H14 is weak for 12670:
  current-object attention mass is only about 0.035-0.087 for the target793 row,
  with small rank/margin effects compared with L23H6.
```

Interpretation: in `12670/sorted`, L23H6 carries some bottom-boundary or
object-coordinate evidence: amplification helps the target margin and removal
hurts it. But the evidence is poorly calibrated or poorly bound. The same
amplification routes through border/bin-999 and intermediate shallow-bottom
ridges rather than exact y2 completion. This supports a "latent but misrouted"
account rather than visual absence.

Updated mechanism picture:

```text
19432/random:
  near target is available, but L27H14 current-object value flow reinforces a
  local 342/343/347/358 ridge. Removing that value source exposes a near-target
  valid tail, but exact 348 still needs broader readout-ridge suppression.

12670/sorted:
  L23H6 value flow contains useful target-direction evidence, especially for
  target793, but amplifying it expresses as border/intermediate ridge movement
  instead of exact bottom y2. This suggests binding/calibration failure rather
  than visual non-perception.

Both:
  no simple value-region scaling produced exact top1 target repair in true
  forward continuation, even when readout-only surgery could. The missing bridge
  is therefore not just "more of this head"; it is a composition/normalization
  problem across the ridge, adapter surface, and later residual readout.
```

Next probe refinement:

```text
Build a layer/head ridge-attribution panel that scores each captured head
contribution by dot products against:
  target effective coord row
  generated coord row
  top8 ridge centroid
  border/extreme bins such as 999

This should be cheaper than full patched forwards and should explain why
L27H14 removal helps 19432 while L23H6 amplification helps-but-misroutes 12670.
```

### Ridge-attribution route panel

Code/artifact update:

```text
route probe now records, for each layer/head/region value contribution:
  ridge_target_minus_generated_projection
  ridge_target_minus_observed_top1_projection
  ridge_target_minus_top8_centroid_projection
  ridge_target_minus_border_0_projection
  ridge_target_minus_border_999_projection

It uses the effective coord output surface, including token_embeddings_adapter
head offsets when present, and baseline next-token coord logits for the same
prefix/query position.
```

Artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route/v2_ridge_attribution_19432_random_layers22_27_allheads_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route/v2_ridge_attribution_12670_sorted_layers22_27_allheads_gpu1
```

Scope:

```text
19432/random_denoise:
  plan_row_count=2
  rows=1728
  ok_row_count=1728
  error_count=0
  target/generated y2 rows: 348<-342 and 348<-343

12670/sorted_denoise:
  plan_row_count=2
  rows=1728
  ok_row_count=1728
  error_count=0
  target/generated y2 rows: 608<-415 and 793<-415

layers=22,23,24,25,26,27
heads=all available attention heads
regions=image_tokens,prompt_non_image,assistant_prefix_prior_objects,
        current_object_ref,current_box_start,current_forced_coords,
        current_partial_object,recent_16,all_prefix
ridge_k=8
border_bins=0,999
```

`19432/random` passive route attribution:

```text
Observed coord ridges:
  generated342 row top8: 342,347,344,343,346,349,350,340
  generated343 row top8: 343,347,342,344,349,358,361,359

Strong positive target-vs-ridge routes:
  L23H6 all_prefix:
    target-minus-generated mean over two y2 rows: +9.133
    target-minus-top8-centroid mean: +7.783
    top source tokens include prior/current coord tokens such as
      <|coord_458|>, <|coord_651|>, <|coord_65|>, <|coord_101|>

  L26H5 prompt_non_image/all_prefix:
    target-minus-generated about +6.8 to +7.0
    target-minus-top8-centroid about +3.9 to +6.5
    source tokens are mostly grammar/schema words such as coord, bbox, coords.

Strong negative target-vs-ridge route:
  L27H14 current_forced_coords/current_partial_object/recent_16/all_prefix:
    target-minus-generated mean around -11.87 to -11.91
    target-minus-top8-centroid mean around -11.18 to -11.24
    current forced-coordinate attention mass around 0.835
    top source tokens include the forced coordinate tokens:
      <|coord_458|>, <|coord_651|>, <|coord_351|>, <|coord_537|>,
      <|coord_122|>
```

Interpretation: the passive route decomposition matches and sharpens the
previous causal value-region result. `L27H14` is not merely a large current-object
head; its value contribution points strongly away from exact `348` and toward
the observed local ridge. That explains why removing it improves rank/top1
toward near-target `347`, while amplifying it worsens into the wider ridge.
At the same time, `L23H6` and `L26H5` show that exact-target evidence is present
elsewhere; the failure is a conflict between target-evidence routes and a late
current-coordinate ridge stabilizer, not visual non-perception.

`12670/sorted` passive route attribution:

```text
Observed coord ridges:
  target608 row top8: 415,418,421,432,420,416,447,434
  target793 row top8: 415,609,664,425,612,430,426,484

Strong positive target-vs-ridge routes:
  target793 L23H6 recent_16/current_partial_object/current_forced_coords:
    target-minus-generated about +5.85
    target-minus-top8-centroid about +6.32
    current forced-coordinate attention mass about 0.901
    source tokens are forced/current coord tokens:
      <|coord_461|>, <|coord_365|>, <|coord_344|>

  target608 L24H10 prompt_non_image/all_prefix:
    target-minus-top8-centroid about +8.53
    source tokens are mostly delimiter/template fragments such as >< and >.

  target608 L26H9 current regions:
    target-minus-generated about +5.4
    target-minus-top8-centroid about +3.8
    source tokens include <|coord_615|>, <|coord_403|>, <|coord_492|>.

Strong negative target-vs-ridge routes:
  target793 L25H6 prompt_non_image/all_prefix:
    target-minus-generated about -12.4
    target-minus-top8-centroid about -15.5
    source tokens are wrapper/template tokens, especially <|box_end|>,
      <|box_start|>, <|object_ref_start|>

  target608 L23H1 prompt_non_image/all_prefix:
    target-minus-generated about -8.1
    target-minus-top8-centroid about -6.4
    source tokens are delimiter/newline/template fragments.

  target608 L27H15 current regions:
    target-minus-border999 about -8.0 to -11.0
    despite target-minus-generated being positive, this route points away from
    the high-border attractor, which helps separate "correct y2" evidence from
    generic bottom-boundary pressure.
```

Interpretation: `12670/sorted` is not a single-head story. `L23H6` does contain
target-direction current-coordinate evidence for the deeper `793` target, and
`L26H9` carries current-coordinate evidence for the `608` target. But the
strongest positive top8-centroid route for `608` is a prompt/template route
(`L24H10`), while several prompt/template heads push hard against the target.
This supports a split mechanism:

```text
current-coordinate routes preserve some visual/box evidence,
but prompt-template and wrapper routes dominate ridge calibration.
```

Bridge to the broader mechanism picture:

```text
19432/random:
  local near-target evidence exists, but a late current-coordinate head
  stabilizes the wrong local ridge.

12670/sorted:
  current-coordinate evidence exists, but grammar/template heads create a
  competing ridge and border-calibration field. The model can move toward the
  right semantic/geometry basin without selecting the exact coordinate.

Both:
  prefix-denoising failure is better described as miscalibrated ridge binding
  after partial perception, not absence of object perception. The next useful
  surgery should patch/add/remove whole route groups by sign and source type:
    target-positive current-coordinate routes
    target-negative current-coordinate routes
    target-positive template routes
    target-negative template routes
    border-999 routes
```

### Grouped route surgery

Implemented a grouped intervention runner that selects top signed route sites
from the ridge-attribution rows and patches several layer/head/region value
contributions in a single forward pass.

Artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_group_intervention/v1_ridge_group_19432_random_top1_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_group_intervention/v1_ridge_group_12670_sorted_top1_gpu1
```

Scope:

```text
group selection source:
  v2_ridge_attribution_19432_random_layers22_27_allheads_gpu0
  v2_ridge_attribution_12670_sorted_layers22_27_allheads_gpu1

route_plan_rows=v1_force_x1_y1_x2
top_n_per_group=1
group_metric=ridge_target_minus_top8_centroid_projection
positive_scale=2.0
negative_scale=0.0
include_combos=true
continuation_steps=2

19432/random:
  plan_row_count=2
  group_spec_count=14
  rows=16
  ok_row_count=16
  error_count=0
  target_rank_improved_non_noop_count=7
  target_margin_improved_non_noop_count=9
  top1_flipped_to_target_non_noop_count=0

12670/sorted:
  plan_row_count=2
  group_spec_count=14
  rows=16
  ok_row_count=16
  error_count=0
  target_rank_improved_non_noop_count=10
  target_margin_improved_non_noop_count=11
  top1_flipped_to_target_non_noop_count=0
```

`19432/random` causal route groups:

```text
target348/generated342:
  current_negative_suppress_top1:
    site: L27H14 current_forced_coords scale=0.0
    emitted tail: <|coord_342|>,<|box_end|> -> <|coord_347|>,<|box_end|>
    rank: 6 -> 2
    target-generated margin: -0.50 -> 0.00
    label: near_y2_box_tail_radius1

  current_positive_amplify_top1:
    site: L23H6 current_forced_coords scale=2.0
    emitted tail remains <|coord_342|>,<|box_end|>
    rank worsens: 6 -> 10

  template_positive_amplify_top1:
    site: L25H12 prompt_non_image scale=2.0
    emitted tail remains <|coord_342|>,<|box_end|>
    small margin improvement: -0.50 -> -0.375

  all_signed_combo_top1:
    sites: current positive + current negative + template positive + template negative
    emitted tail moves away to <|coord_340|>,<|box_end|>
    rank worsens: 6 -> 13

target348/generated343:
  current_negative_suppress_top1:
    site: L27H14 current_forced_coords scale=0.0
    emitted tail: <|coord_343|>,<|box_end|> -> <|coord_347|>,<|box_end|>
    rank: 7 -> 2
    margin: -0.375 -> 0.0

  template_signed_combo_top1:
    emitted tail also moves to <|coord_347|>,<|box_end|>
    rank: 7 -> 3
    margin: -0.375 -> 0.0

  all_signed_combo_top1:
    emitted tail moves to <|coord_347|>,<|box_end|>
    rank: 7 -> 2
    margin: -0.375 -> +0.125
```

Interpretation: `19432/random` now has a repeated causal signature across both
generated y2 states. Suppressing the late negative current-coordinate route
(`L27H14 current_forced_coords`) is sufficient to make the autoregressive tail
valid and near-exact (`347` then `<|box_end|>`), but not exact `348`. Amplifying
the strongest positive current route (`L23H6`) does not repair and can worsen
rank. Combining too many template/current interventions can oversteer the local
ridge to `340`. This strengthens the view that the missing exact coordinate is
not absent perception; it is local ridge calibration with a stubborn one-bin
attractor around 347/348.

`12670/sorted` causal route groups:

```text
target608/generated415:
  template_negative_suppress_top1:
    site: L23H1 prompt_non_image scale=0.0
    emitted tail: <|coord_415|>,<|box_end|> -> <|coord_420|>,<|box_end|>
    rank: 474 -> 444
    margin: -2.6875 -> -2.25

  template_signed_combo_top1:
    emitted tail: <|coord_415|>,<|box_end|> -> <|coord_421|>,<|box_end|>
    rank: 474 -> 409
    margin: -2.6875 -> -2.3125

  current_positive_amplify_top1:
    site: L26H9 current_forced_coords scale=2.0
    emitted tail moves to <|coord_432|>,<|box_end|>
    rank barely changes: 474 -> 472

  current_negative_suppress_top1:
    site: L23H6 current_forced_coords scale=0.0
    emitted tail moves to <|coord_421|>,<|box_end|>
    rank worsens: 474 -> 517

target793/generated415:
  current_positive_amplify_top1:
    site: L23H6 current_forced_coords scale=2.0
    emitted tail: <|coord_415|>,<|box_end|> -> <|coord_609|>,<|box_end|>
    rank: 571 -> 469
    margin: -2.4375 -> -1.3125

  current_signed_combo_top1:
    emitted tail moves to <|coord_600|>,<|box_end|>
    rank: 571 -> 406
    margin: -2.4375 -> -1.125

  all_signed_combo_top1:
    emitted tail moves to <|coord_605|>,<|box_end|>
    rank: 571 -> 264
    margin: -2.4375 -> -0.5625

  template_negative_suppress_top1:
    emitted tail stays <|coord_415|>,<|box_end|>
    rank improves: 571 -> 528
```

Interpretation: `12670/sorted` has a stronger intermediate-basin phenomenon
than `19432/random`. The grouped interventions can push the emitted y2 out of
the shallow `415` basin and into the `600/605/609` region, especially for the
deep `793` target, but none selects exact `793`. For `608`, suppressing a
negative template route helps more than current-coordinate amplification, but
the output still only moves to the shallow `420/421/432` ridge. This suggests
two separate barriers:

```text
1. escaping the shallow default y2 basin around 415;
2. resolving the remaining long-range bottom-coordinate calibration to exact
   608/793.
```

Current mechanistic update:

```text
The grouped surgery supports a staged mechanism:

perception/binding evidence:
  present in current-coordinate routes, especially L23H6 for target793 and
  L26H9 for target608.

autoregressive/ridge gate:
  late or template routes decide which coordinate basin becomes the emitted
  first y2 token.

exact-coordinate calibration:
  not solved by scaling the strongest signed routes. Exact repair likely needs
  either a readout-space ridge operation or a deeper residual/MLP normalization
  site that maps intermediate basin evidence to the final coordinate token.
```

Post-surgery coordinate-window check:

```text
19432/random, target348 after current_negative_suppress_top1:
  generated342 row:
    top1=347
    target348 rank=2
    target logit=24.375
    top1 logit=24.625
    top-target gap=0.25

  generated343 row:
    top1=347
    target348 rank=2
    target logit=22.000
    top1 logit=22.375
    top-target gap=0.375

  template_signed_combo on generated343:
    top1=347
    target348 rank=3
    top-target gap=0.125
```

Interpretation: for `19432/random`, grouped route surgery almost reaches exact
repair. The remaining failure is a very small local readout tie-break inside the
347/348 ridge, not a long-range missing-object or missing-geometry problem.

```text
12670/sorted, target793:
  current_positive_amplify_top1:
    top1=609
    target793 rank=469
    top-target gap=1.6875

  current_signed_combo_top1:
    top1=600
    target793 rank=406
    top-target gap=1.5

  all_signed_combo_top1:
    top1=605
    target793 rank=264
    top-target gap=1.5

12670/sorted, target608:
  template_negative_suppress_top1:
    top1=420
    target608 rank=444
    top-target gap=2.4375

  template_signed_combo_top1:
    top1=421
    target608 rank=409
    top-target gap=2.5

  all_signed_combo_top1:
    top1=421
    target608 rank=504
    top-target gap=3.125
```

Interpretation: for `12670/sorted`, grouped route surgery mostly solves only the
first barrier: escaping default y2 around `415`. It does not solve exact
long-range bottom-coordinate calibration. The model can be causally pushed into
a plausible intermediate bottom-region basin (`600/605/609` for the deeper
target), but exact `793` remains far away in both rank and logit. This is a
different mechanism from `19432/random`; treating both as one generic
duplication/coordinate failure would hide the distinction.

## Post-Group Readout Bridge Probe

Code/test increment:

```text
src/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_group_intervention.py
tests/analysis/test_prefix_denoising_anchor_escape_y2_route_group_intervention.py
```

New probe mode:

```text
--readout-bridge-alphas <csv>
```

The bridge is applied after a grouped route intervention. It extracts the
patched final hidden state at the y2 prediction position, applies the model's
final norm, and perturbs the lm-head input along an adapter-aware effective
output direction:

```text
target coord output row - post-group coord-top1 output row
```

This is deliberately not a proof of normal decoding behavior. It asks a narrower
causal question: after the route patch has moved the autoregressive state into a
near or intermediate coordinate basin, how much direct readout movement remains
before the exact target coordinate and `<|box_end|>` tail win?

Artifacts:

```text
v2 coarse random:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_group_intervention/v2_readout_bridge_19432_random_top1_gpu0

v3 wide random:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_group_intervention/v3_readout_bridge_wide_19432_random_top1_gpu0

v2 coarse sorted:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_group_intervention/v2_readout_bridge_12670_sorted_top1_gpu1

v3 fine sorted:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_group_intervention/v3_readout_bridge_fine_12670_sorted_top1_gpu1
```

Representative sample-base policy:

```text
Do not spend primary mechanistic budget on normal/well-learned cases.
Keep sample bases that create a sharp contrast between:
  - route-only basin motion,
  - exact post-route readout bridge threshold,
  - continuation tail validity,
  - and checkpoint/template behavior.
```

The current narrow sample bases remain high value:

```text
19432/random:
  duplicated chair sequence with adjacent local y2 ridge around 342/343/347/348.

12670/sorted:
  first-object person span with strong default 415 basin and long-range target
  y2 values 608/793.
```

Bridge findings:

```text
19432/random, target348/generated342:
  route-only best:
    current_negative_suppress_top1: 342 -> 347, rank 6 -> 2,
    emitted <|coord_347|><|box_end|>.

  exact bridge:
    min alpha = 0.2
    group = template_negative_suppress_top1
    route top1 = 342
    final top1 = 348
    final rank = 1
    emitted <|coord_348|><|box_end|>

19432/random, target348/generated343:
  route-only best:
    all_signed_combo_top1: 343 -> 347, rank 7 -> 2,
    emitted <|coord_347|><|box_end|>.

  exact bridge:
    min alpha = 0.3
    group = current_positive_amplify_top1
    route top1 = 342
    final top1 = 348
    final rank = 1
    emitted <|coord_348|><|box_end|>

12670/sorted, target608/generated415:
  route-only best:
    template_signed_combo_top1: 415 -> 421, rank 474 -> 409,
    emitted <|coord_421|><|box_end|>.

  exact bridge:
    min alpha = 0.125
    groups = current_positive_amplify_top1 or template_positive_amplify_top1
    route top1 = 432 or 415
    final top1 = 608
    final rank = 1
    emitted <|coord_608|><|box_end|>

12670/sorted, target793/generated415:
  route-only best:
    all_signed_combo_top1: 415 -> 605, rank 571 -> 264,
    emitted <|coord_605|><|box_end|>.

  exact bridge:
    min alpha = 0.075
    group = current_signed_combo_top1
    route top1 = 600
    final top1 = 793
    final rank = 1
    emitted <|coord_793|><|box_end|>
```

Mechanistic update:

```text
The sorted case is not simply missing visual perception or object binding.
Once route surgery moves the state out of the 415 default basin, the exact target
coordinate can be selected by a modest adapter-aware readout bridge. This points
to a staged failure:

  visual/object evidence exists,
  route dynamics partially expose it,
  but the normal autoregressive state does not place that evidence on the exact
  coordinate readout axis strongly enough.

The random chair case is more local and adjacent-ridge-like. It needs a larger
readout bridge despite already being one bin away after route-only surgery,
which suggests a stubborn 347/348 local tie-break/slot basin rather than a
long-range geometry absence.
```

Next promising direction:

```text
Trace where the post-route bridge direction naturally appears or disappears
across layers:
  1. final-norm input before and after route patch,
  2. residual stream projection onto effective target-minus-basin directions,
  3. MLP contribution at the last few layers,
  4. attention-vs-MLP split for converting route evidence into exact coord
     readout.

This should be done on selected hard sample bases first, then expanded to a
small panel of additional train/val images only after the layer/site signature is
stable.
```

## 2026-06-25 - Bridge-direction layer tomography

Implementation/artifacts:

```text
Probe:
src/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_bridge_tomography.py

Tests:
tests/analysis/test_prefix_denoising_anchor_escape_y2_bridge_tomography.py

19432/random output:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_bridge_tomography/v1_layers20_27_19432_random_gpu0

12670/sorted output:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_bridge_tomography/v1_layers20_27_12670_sorted_gpu1
```

Scope:

```text
readout_only=true
training_ran=false
layers=20,21,22,23,24,25,26,27
top_n_per_group=1
include_combos=false

Each run:
  plan rows = 2
  group specs = 8
  tomography rows = 128
  readout status = ok:128
  surfaces = baseline:64, route_patched:64
```

What the probe measures:

```text
For each selected sample-base and route group, run the existing route patch with
hidden_states enabled. Use the route-patched final coord top1 as the antagonist
unless the target is already top1, then use the top non-target bin. At each
selected layer, project the LM-head input onto:

  effective_output[target_coord] - effective_output[post_route_antagonist]

Then compare baseline vs route-patched projection, target-antagonist margin,
coord top1, and target rank.
```

Representative sample-base rule update:

```text
Keep the main budget on image bases that are diagnostically sharp, not merely
frequent:

  A. near-boundary cases where route surgery almost or actually reaches the
     target readout basin, useful as positive controls for the bridge axis;
  B. hard conversion failures where route surgery increases bridge projection
     and target margin but target rank remains very bad, useful for locating
     the route-to-readout conversion failure;
  C. false-negative or premature-stop cases only when the prefix can be aligned
     to a concrete missing object/span site.

Normal/well-learned images are secondary controls, not the center of the study.
```

Tomography findings:

```text
19432/random, target348/generated342:
  best group = template_negative_suppress_top1
  best layer = 23
  target-antagonist margin delta = +0.3082
  bridge projection delta = +0.5387
  coord target rank = 1
  coord top1 = 348
  antagonist = 342

19432/random, target348/generated343:
  best group = current_positive_amplify_top1
  best layer = 23
  target-antagonist margin delta = +0.2533
  bridge projection delta = +0.4428
  coord target rank = 1
  coord top1 = 348
  antagonist = 342

12670/sorted, target793/generated415:
  best group = current_positive_amplify_top1
  best layer = 26
  target-antagonist margin delta = +0.9003
  bridge projection delta = +1.0184
  coord target rank = 574
  coord top1 = 727
  antagonist = 609

12670/sorted, target608/generated415:
  best group = template_negative_suppress_top1
  best layer = 23
  target-antagonist margin delta = +0.6771
  bridge projection delta = +0.9140
  coord target rank = 970
  coord top1 = 440
  antagonist = 420
```

Mechanistic update:

```text
The bridge-direction tomography strengthens the split between local coordinate
ridge failures and deep conversion failures.

19432/random behaves like a near-boundary coordinate-basin case. Route surgery
injects enough target-minus-antagonist direction by layer 23 to make coord 348
rank 1 for both generated 342 and 343 variants. This is the useful positive
control: the bridge axis is a real local readout feature, not only an artificial
post-hoc vector.

12670/sorted is more revealing for the core failure. Route surgery creates much
larger bridge projection and margin gains than the random chair case, yet target
rank remains hundreds. So the failure is not "no relevant signal moved"; it is
that the moved signal is still too weak or too misaligned relative to many other
coordinate directions in the effective output geometry. The route state contains
some evidence but does not concentrate it into the exact coordinate-basin
readout axis.

This makes 12670/sorted a higher-value main-course sample-base than normal
images: it isolates the conversion gap between visual/route evidence and exact
coordinate readout, which is closer to the desired core mechanism.
```

Next action:

```text
Use the tomography-ranked cases to drive MLP/residual tensor-flow surgery:

  1. For 19432/random, treat layer 23 as a positive-control site where the
     correct bridge axis is recoverable.
  2. For 12670/sorted, inspect layers 23-27 for why large bridge-direction gains
     still leave the target behind hundreds of coordinate competitors.
  3. Add a sample-base selector over train/val rollouts that ranks candidates by:
       route patch moves bridge_projection_delta > threshold,
       target rank remains bad,
       or near-boundary rank becomes 1,
     so future narrow panels are selected by mechanistic leverage instead of
     aggregate metric frequency.
```

## 2026-06-25 - Bridge-tomography sample-base selector

Implementation/artifacts:

```text
Selector:
src/analysis/prefix_denoising_surgery_probing/bridge_sample_base_selector.py

Tests:
tests/analysis/test_prefix_denoising_bridge_sample_base_selector.py

Current panel output:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/bridge_sample_base_selector/v1_current_tomography_panel
```

Scope:

```text
Input tomography rows:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_bridge_tomography/v1_layers20_27_19432_random_gpu0/anchor_escape_y2_bridge_tomography_rows.jsonl
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_bridge_tomography/v1_layers20_27_12670_sorted_gpu1/anchor_escape_y2_bridge_tomography_rows.jsonl

candidate_count = 4
selected_count = 4
families:
  hard_conversion_failure = 2
  near_boundary_success_control = 2
readout_only = true
training_ran = false
```

Ranked queue:

```text
1. hard_conversion_failure
   sorted_denoise image=12670 target793/generated415
   score=134.7935
   best target rank=461
   max bridge projection delta=+1.0184
   max target margin delta=+0.9003
   next=MLP/residual tensor-flow surgery across the best margin/projection layers

2. hard_conversion_failure
   sorted_denoise image=12670 target608/generated415
   score=131.6783
   best target rank=444
   max bridge projection delta=+0.9140
   max target margin delta=+0.6771
   next=MLP/residual tensor-flow surgery across the best margin/projection layers

3. near_boundary_success_control
   random_denoise image=19432 target348/generated343
   score=58.2988
   best target rank=1
   max bridge projection delta=+0.6571
   max target margin delta=+0.2533
   next=positive-control bridge-axis replication and wrong-control patch

4. near_boundary_success_control
   random_denoise image=19432 target348/generated342
   score=57.9264
   best target rank=1
   max bridge projection delta=+0.5387
   max target margin delta=+0.3082
   next=positive-control bridge-axis replication and wrong-control patch
```

Mechanistic interpretation:

```text
The selector formalizes the sample-base policy:

  - 12670/sorted is the main-course hard conversion sample-base. It has large
    bridge/margin movement but still bad rank, so the next probe should dissect
    MLP/residual tensor flow, not simply search for stronger visual evidence.

  - 19432/random is the positive-control bridge-axis sample-base. It can reach
    rank 1 after route patch, so it is useful for wrong-control patching and for
    checking whether a proposed readout/tensor-flow mechanism actually traces a
    recoverable coordinate axis.

This keeps future exploration sample-base-centered while avoiding the trap of
spending attention on normal/well-learned images.
```

## 2026-06-25 - Tensor-flow bridge readout probe

Implementation/artifacts:

```text
Probe:
src/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_tensor_flow.py

Tests:
tests/analysis/test_prefix_denoising_anchor_escape_y2_tensor_flow.py

Sorted hard-conversion output:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_tensor_flow/v2_selector_layers23_27_12670_sorted_gpu0

Random positive-control output:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_tensor_flow/v2_selector_layers23_27_19432_random_gpu1
```

Scope:

```text
selector-filtered cases:
  sorted_denoise image=12670 target793/generated415 group=current_positive_amplify_top1
  sorted_denoise image=12670 target608/generated415 group=template_negative_suppress_top1
  random_denoise image=19432 target348/generated343 groups=current_positive_amplify_top1, template_negative_suppress_top1
  random_denoise image=19432 target348/generated342 group=template_negative_suppress_top1

layers = 23,24,25,26,27
components per layer:
  state_lens: layer_input_state, after_attention_state, layer_output_state
  update_vector: attention_update, mlp_update, layer_delta_update
readout_only = true
training_ran = false

sorted:
  plan rows = 2
  group specs = 2
  rows = 120
  errors = 0

random:
  plan rows = 2
  group specs = 3
  rows = 180
  errors = 0
```

Probe semantics:

```text
For each selected case/group, run baseline and route-patched forwards with
component hooks on the selected decoder layers. At the prefix token, capture:

  layer input,
  self-attention update,
  after-attention residual state,
  MLP update,
  layer output.

State rows use the same adapter-aware coordinate logit lens as bridge
tomography. Update rows are projected directly onto the same
effective_output[target] - effective_output[post-route antagonist] direction.
```

Tensor-flow findings:

```text
sorted_denoise image=12670 target793/generated415
  group=current_positive_amplify_top1

  L23:
    attention update delta = +8.697
    MLP update delta = +1.491
    layer delta update = +10.129
    readout-visible layer output delta = +0.870
    output rank = 722, top1 = 442

  L26:
    attention update delta = +0.871
    MLP update delta = +6.625
    layer delta update = +7.381
    readout-visible layer output delta = +1.020
    output rank = 572, top1 = 727

  L27:
    input state delta = +1.020
    MLP update delta = -18.359
    layer delta update = -16.539
    readout-visible layer output delta = +0.742
    output rank = 461, top1 = 612

sorted_denoise image=12670 target608/generated415
  group=template_negative_suppress_top1

  L23:
    attention update delta = +6.213
    MLP update delta = +1.540
    layer delta update = +7.591
    readout-visible layer output delta = +0.912
    output rank = 969, top1 = 440

  L24-L27:
    readout-visible layer output delta decays from +0.687 to +0.330
    final inspected rank improves only to 445, top1 = 421

random_denoise image=19432 target348/generated342
  group=template_negative_suppress_top1

  L23:
    attention update delta = +5.762
    MLP update delta = -0.940
    layer delta update = +4.853
    readout-visible layer output delta = +0.544
    output rank = 1, top1 = 348

  L24-L27:
    readout-visible layer output delta decays and becomes negative
    ranks/top1 drift back toward the local 347/342 basin.

random_denoise image=19432 target348/generated343
  group=current_positive_amplify_top1

  L23:
    attention update delta = +3.907
    MLP update delta = +0.676
    layer delta update = +4.565
    readout-visible layer output delta = +0.451
    output rank = 1, top1 = 348
```

Mechanistic update:

```text
The tensor-flow split strengthens the route-to-readout conversion picture.

Route surgery creates large bridge-aligned update vectors, especially through
attention at layer 23 and MLP at layer 26 for the hard sorted case. But those
large update vectors are compressed into much smaller readout-visible state
movement under the coordinate logit lens. For 12670/sorted, even the largest
state movement (+1.020 bridge projection, +0.902 margin) leaves the target rank
in the hundreds. The problem is therefore not the complete absence of a bridge
direction; it is weak/poorly concentrated readout alignment relative to many
coordinate competitors.

The random chair controls show the complementary pattern. The correct bridge
axis appears early enough to rank the target first at layer 23, but later layers
can erode or redirect that state back toward nearby coordinate basins. This
means near-duplication can be caused not only by missing evidence but by late
autoregressive cleanup/slot-basin dynamics that fail to preserve an already
available coordinate readout.

Working hypothesis after this probe:
  1. Route patches inject object/coordinate evidence mostly as attention-route
     updates.
  2. The MLP/residual stack decides whether that evidence becomes a stable exact
     coordinate readout basin or is dispersed/cancelled.
  3. Prefix denoising did not solve exposure bias because the model can move in
     the right internal direction without making that movement sufficiently
     rank-competitive at the coordinate output surface.
```

Next action:

```text
Use causal component patching next:
  - patch only the L23 attention update from route-patched into baseline,
  - patch only the L26 MLP update for 12670/sorted target793,
  - ablate or replace the negative L27 MLP update for target793,
  - use 19432/random L23 as a positive-control target-rank-1 patch.

This should distinguish whether the hard sorted case needs stronger early
attention evidence, a better MLP conversion, or removal of late cancellation.
```

## Component Patch: L23 Attention As The Causal Route Interface

Timestamp: 2026-06-25

Code added:

```text
src/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_component_patch.py
tests/analysis/test_prefix_denoising_anchor_escape_y2_component_patch.py
```

Primary outputs:

```text
single component checks:
  sorted:
    /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_component_patch/v1_replace_23attn_26mlp_12670_sorted_gpu0
  random:
    /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_component_patch/v1_replace_23attn_19432_random_gpu1

layer sweep:
  sorted:
    /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_component_patch/v2_layer_sweep_20_27_replace_12670_sorted_gpu0
  random:
    /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_component_patch/v2_layer_sweep_20_27_replace_19432_random_gpu1

L23 attention alpha sweep:
  sorted:
    /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_component_patch/v3_alpha_sweep_23attn_12670_sorted_gpu0
  random:
    /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_component_patch/v3_alpha_sweep_23attn_19432_random_gpu1
```

Scope:

```text
Readout-only causal patching, no training.
Selector-filtered high-value sample-bases only:
  sorted_denoise image=12670 target793/generated415
  sorted_denoise image=12670 target608/generated415
  random_denoise image=19432 target348/generated342
  random_denoise image=19432 target348/generated343

These are deliberately not normal/well-learned cases. They are representative
hard rows selected from bridge tomography as current best narrow sample-bases.
```

Probe semantics:

```text
For each selected route group:
  1. run the clean baseline forward,
  2. run the existing single-head route intervention,
  3. capture the route-patched component vector at the prefix token,
  4. transplant that component vector into a clean baseline forward at the same
     layer/component,
  5. compare clean baseline vs full route patch vs component-only patch.

This distinguishes "the route intervention as a whole changes logits" from
"this exact internal component output is sufficient to carry the effect."
```

Selected route specs:

```text
sorted image=12670 target608/generated415:
  group=template_negative_suppress_top1
  site=L23 head1 prompt_non_image scale=0

sorted image=12670 target793/generated415:
  group=current_positive_amplify_top1
  site=L23 head6 current_forced_coords scale=2

random image=19432 target348/generated342:
  group=template_negative_suppress_top1
  site=L23 head5 prompt_non_image scale=0

random image=19432 target348/generated343:
  group=current_positive_amplify_top1
  site=L23 head6 current_forced_coords scale=2
  group=template_negative_suppress_top1
  site=L23 head1 prompt_non_image scale=0
```

Main causal finding:

```text
The effective bridge intervention enters the residual stream through the L23
self-attention update. Replacing only the L23 attention_update in the clean
forward reproduces the full route-patched coordinate readout for the selected
single-head route interventions.

Sorted hard rows:
  target793/generated415 current_positive:
    baseline: top1=415 rank=571
    full route patch: top1=609 rank=469
    L23 attention-only replace: top1=609 rank=469

  target608/generated415 template_negative:
    baseline: top1=415 rank=474
    full route patch: top1=420 rank=444
    L23 attention-only replace: top1=420 rank=444

Random controls:
  target348/generated342 template_negative:
    baseline: top1=342 rank=6
    full route patch: top1=342 rank=11
    L23 attention-only replace: top1=342 rank=11

  target348/generated343 template_negative:
    baseline: top1=343 rank=7
    full route patch: top1=347 rank=5
    L23 attention-only replace: top1=347 rank=5
```

Layer/component sweep:

```text
Layers 20-22 are zero controls because they precede the selected L23 route site.

For sorted image=12670, L23 attention is the only component that exactly
matches the route-patched readout. Later attention/MLP components can nudge the
target rank or move top1 into nearby bins, but they do not reproduce the route
effect exactly.

Examples:
  target793/generated415:
    L23 attention: rank 469, top1 609, margin delta +1.125
    L26 MLP:       rank 511, top1 673, margin delta +0.500
    L27 MLP:       rank 536, top1 609, margin delta +0.5625

  target608/generated415:
    L23 attention: rank 444, top1 420, margin delta +0.4375
    L27 MLP:       rank 457, top1 418, margin delta +0.1875
```

Alpha sweep:

```text
Sorted image=12670 shows a smooth basin-response curve under L23 attention
delta scaling.

target793/generated415 current_positive:
  alpha=0.25 rank=538 margin +0.375 top1=609
  alpha=0.50 rank=516 margin +0.625 top1=609
  alpha=1.00 rank=464 margin +1.125 top1=609
  alpha=1.50 rank=345 margin +1.625 top1=609
  alpha=2.00 rank=241 margin +2.125 top1=609

target608/generated415 template_negative:
  alpha=0.25 rank=476 margin +0.0625
  alpha=0.50 rank=464 margin +0.250
  alpha=1.00 rank=444 margin +0.4375
  alpha=1.50 rank=431 margin +0.625
  alpha=2.00 rank=401 margin +0.8125

Random image=19432 is not smooth. Some L23 attention deltas improve nearby
rank, but other directions push into adjacent wrong coordinate basins. This
supports a "fragile local coordinate basin" reading rather than a simple missing
visual evidence reading.
```

Mechanistic update:

```text
The prior tensor-flow finding can now be sharpened:

  1. The route evidence is not just globally present somewhere in the residual
     stack; for these selected cases it is causally concentrated at the L23
     attention-update interface.
  2. The hard sorted failure is not fixed by a single realistic L23 route delta,
     even though the response is monotone. The model moves away from the
     duplicated/generated y2 basin but remains far from making the target
     coordinate rank-competitive.
  3. Later MLP/residual transformations can preserve, attenuate, or redirect
     the early attention evidence, but they are not the origin of the selected
     route intervention's causal effect.
  4. Prefix denoising appears to leave a bottleneck where visual/route evidence
     can enter correctly but lands in a broad nearby-coordinate attraction basin
     instead of an exact coordinate slot basin.

Working wording:
  L23 attention is the gate by which selected object/coordinate evidence enters
  the autoregressive coordinate basin; later layers decide whether that entered
  evidence becomes exact, remains a nearby anchor, or is eroded.
```

Next sample-base policy:

```text
Keep the current high-value sample-base set as a narrow microscope, but extend
selection next. The next selector should mine train and val prediction artifacts
for:
  - false negatives where the object may be visually perceived but language
    guidance/prefix state fails to recruit it,
  - duplication bursts with repeated coordinate anchors,
  - cases where L23 attention moves monotonically but cannot cross the exact
    coordinate basin threshold,
  - cases where train rows still fail despite direct exposure, to separate
    memorized visual recognition from autoregressive binding failure.
```

## Object-Level Probe Panel Selector

Selector code and test surface:

```text
src/analysis/prefix_denoising_surgery_probing/probe_panel_selector.py
tests/analysis/test_prefix_denoising_probe_panel_selector.py
```

Fresh object-level selector artifact:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/probe_panel_selector/v3_val200_object_panel_diverse_quota
```

Evidence scope:

```text
split: val
source: selected prefix-denoising val200 sample bases
candidate_count: 472
selected_count: 48
selected_kind_counts:
  duplication_cluster: 25
  false_negative_object: 16
  termination_or_empty: 7
selected_model_counts:
  random_denoise: 28
  sorted_denoise: 20
training_ran: false
```

The selector intentionally avoids normal/well-learned images. It turns the
image-level sample-base registry into object-level candidates:

```text
false_negative_object:
  unmatched GT object; rank higher when the paired model recovers the same GT
  and when same-description predictions exist elsewhere in the row

duplication_cluster:
  duplicate-guard cluster; rank higher for large repeated semantic anchors and
  suppressed autoregressive repeats

termination_or_empty:
  empty, parse-error, or missing-stop rows for stop-vs-continue pressure probes
```

Selected hard image bases:

```text
2157, 2299, 2685, 9590, 12639, 12670, 14439, 16228, 17899, 18380, 19109, 19432
```

High-value duplicate-basin lanes:

```text
random image=2685:  wine glass cluster size 53
random image=2157:  wine glass cluster size 39
random image=12670: person cluster size 30
random image=19432: chair cluster size 29
random image=16228: person cluster size 19
random image=17899: cake cluster size 18
random image=12639: person cluster size 42
sorted image=2685:  bottle cluster size 14
random image=19109: motorcycle cluster size 9
```

High-value false-negative guidance lanes:

```text
sorted image=2685:  GT bottle, same-desc pred count 17, paired model recovers
random image=12670: GT person, same-desc pred count 50, paired model recovers
sorted image=12670: GT person, same-desc pred count 18, paired model recovers
random image=16228: GT person, same-desc pred count 54, paired model recovers
sorted image=16228: GT person, same-desc pred count 15, paired model recovers
random image=2157:  GT wine glass, same-desc pred count 39, paired model recovers
sorted image=2157:  GT wine glass, same-desc pred count 10, paired model recovers
random image=19432: GT chair, same-desc pred count 40, paired model recovers
sorted image=19109: GT motorcycle, same-desc pred count 12, paired model recovers
sorted image=12670: GT handbag, same-desc pred count 2, paired model recovers
random image=2157:  GT knife, same-desc pred count 2, paired model recovers
```

Termination and boundary lanes:

```text
both models image=18380
both models image=9590
random image=2299
sorted image=14439
sorted image=17899
```

Current train-row status:

```text
No ckpt908 train rollout roots were found under /data/CoordExp/outputs/infer
with train/ckpt908/prefix-denoising path patterns. The selector already accepts
a split label and can consume future train roots through the same model-registry
contract once train prediction artifacts are materialized.
```

Recommended next surgery panel:

```text
1. random image=2685 wine-glass duplicate burst:
   massive repeated semantic anchor; compare onset attention and coordinate
   basin drift against sorted image=2685 bottle FN/dup behavior.

2. random image=19432 chair duplicate burst:
   known fragile coordinate basin case; test whether L23 attention route
   evidence creates a broad chair anchor instead of object-specific binding.

3. random/sorted image=12670 person conflict:
   paired model recovers several missed person GTs; inspect whether the failed
   model sees visual evidence but language/prefix state binds to existing
   person anchors.

4. images=18380 and 9590 termination/empty:
   compare stop pressure, first-object onset, and visual-to-language handoff
   before any object span is emitted.

5. image=2157 wine glass and knife:
   pair a large wine-glass repeat basin with a small-object FN to separate
   repeated common-object attractors from weaker object recruitment.
```

Mechanistic update:

```text
The next round should not spend compute on average val200 metrics. The useful
unit is now a sample(image)-base plus object/cluster/termination target. The
panel favors cases where there is evidence for perception or semantic
availability but the autoregressive binding/router fails: paired-model recovery,
same-description unbound predictions, repeated spatial/semantic anchors, or
empty/stop boundary collapse.
```

## False-Negative Forced Guidance Probe

Probe code and script:

```text
src/analysis/prefix_denoising_surgery_probing/fn_guidance_probe.py
scripts/analysis/run_prefix_denoising_fn_guidance_probe.py
tests/analysis/test_prefix_denoising_fn_guidance_probe.py
```

Artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_guidance_probe/v1_top4_layers0_8_16_20_23_24_27_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_guidance_probe/v2_wine_chair_knife_layers0_8_16_20_23_24_27_gpu0
```

Evidence scope:

```text
source panel: probe_panel_selector/v3_val200_object_panel_diverse_quota
v1 candidates: selection ranks 27, 28, 29, 30
v2 candidates: selection ranks 32, 33, 36, 44
source_count per run: 8
planned_forward_count per run: 40
row_count per run: 320
readout_status: ok for all 640 rows across both runs
layers: 0, 8, 16, 20, 23, 24, 27 plus true forward logits
training_ran: false
```

Probe design:

```text
For each selected false-negative object:
  1. run the failed checkpoint under an empty assistant prefix forced to the
     target object descriptor and target-box coordinate prefix;
  2. run the paired checkpoint that recovered the same GT under the same forced
     target descriptor/box prefix;
  3. compare a control descriptor chosen from that probe model's dominant
     non-target predictions;
  4. read coord-only rank, top bin, radius mass, coord-vocab mass, and layerwise
     logit-lens progression for x1, y1, x2, y2.

Forced target prefix examples:
  pre_x1: <|object_ref_start|>bottle<|object_ref_end|><|box_start|>
  pre_y1: <|object_ref_start|>bottle<|object_ref_end|><|box_start|><|coord_190|>
```

Strongest immediate pattern:

```text
Descriptor guidance helps, but it usually does not solve the first-anchor
problem. The target descriptor often improves x1 rank and radius-16 mass over a
control descriptor, yet the top x1 bin remains a border bin or an existing
anchor. Later slots often become much more target-like once earlier target
coordinates are forced.
```

Concrete examples:

```text
rank 27, sorted failed on image 2685 GT bottle:
  target-desc x1: rank 96, top bin 0, target 190, radius16 mass 0.096
  control-desc x1: rank 818, top bin 624, radius16 mass 0.000014
  target-desc x2: rank 6, top bin 263, target 283

rank 27, random paired model on the same GT bottle:
  target-desc x1: rank 9, top bin 197, target 190
  target-desc y1: rank 9, top bin 483, target 491
  target-desc y2: rank 26, top bin 592, target 607
  target-desc x2 remains weaker: rank 87, top bin 234, target 283

rank 28, random failed on image 12670 GT person:
  target-desc x1: rank 782, top bin 0, target 163
  target-desc y1: rank 10, top bin 280, target 276
  target-desc x2: rank 1, top bin 303, target 307
  target-desc y2: rank 165, top bin 636, target 698

rank 36, random failed on image 19432 GT chair:
  target-desc x1: rank 261, top bin 0, target 351
  target-desc y1: rank 12, top bin 0, target 122
  target-desc x2: rank 20, top bin 467, target 458
  target-desc y2: rank 4, top bin 347, target 348

rank 44, random failed on image 2157 GT knife:
  target-desc x1: rank 24, top bin 341, target 338
  target-desc y1: rank 113, top bin 670, target 729
  target-desc x2: rank 240, top bin 520, target 639
  target-desc y2: rank 7, top bin 987, target 991
```

Control descriptor effect:

```text
Most failed-model x1 probes improve materially under target descriptor versus
control descriptor, even when they still do not reach exact target x1:

sorted image 2685 bottle:
  control person -> target bottle improves x1 rank by 722 positions

random image 12670 person:
  control backpack -> target person improves x1 rank by 216 positions

sorted image 12670 person:
  control handbag -> target person improves x1 rank by 201 positions

random image 16228 person:
  control bench -> target person improves x1 rank by 422 positions

random image 19432 chair:
  control person -> target chair improves x1 rank by 359 positions

random image 2157 knife:
  control wine glass -> target knife improves x1 rank by 165 positions
```

Mechanistic update:

```text
The false-negative question should not be phrased as "does the model perceive
the object at all?" versus "does the language side guide it?" as a single
binary. The current evidence suggests a three-stage failure:

  1. The target descriptor can recruit some relevant coordinate basin mass.
  2. The first coordinate/anchor gate often remains captured by border,
     ordering, or repeated-object priors.
  3. Once the first one or two coordinates are forced, later slots frequently
     become locally coherent with the target object.

This points to object-onset and first-anchor binding as the likely core of many
false negatives, not pure visual invisibility and not pure schema/type loss.
The failure looks like a context-conditioned routing problem: visual/semantic
evidence exists but arrives too weakly or too late to choose the first spatial
anchor under the model's current autoregressive state.
```

Important caveat:

```text
These runs use empty-context forced object prefixes. Some paired-recovered
models still fail under the empty forced prefix even though they matched the GT
in free rollout. Therefore, paired recovery is not just "same image plus same
descriptor"; it likely depends on the preceding autoregressive object context.
The next probe should transfer natural paired-model prefixes before the matched
object span and test whether the failed checkpoint can follow that context into
the same object basin.
```

Next probe direction:

```text
Build a contextual FN guidance probe:
  - fetch paired model's natural pre_x1 prefix for matched_pred_idx,
  - replay that exact prefix into both checkpoints,
  - optionally replace only the target descriptor or first coordinate,
  - compare whether the failed model can recover under paired-model context,
  - contrast with its own nearest same-desc or duplicate-basin prefix.

This should separate "descriptor helps but first anchor is missing" from
"specific preceding object context is the missing key."
```

## Contextual False-Negative Prefix Transfer

Probe extension:

```text
src/analysis/prefix_denoising_surgery_probing/fn_guidance_probe.py
scripts/analysis/run_prefix_denoising_contextual_fn_guidance_probe.py
tests/analysis/test_prefix_denoising_fn_guidance_probe.py
```

Artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_guidance_probe/v3_contextual_ranks27_28_29_30_32_33_36_44_layers0_8_16_20_23_24_27_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_guidance_probe/v4_contextual_all_selected_fn_layers0_8_16_20_23_24_27_gpu0
```

Full selected-FN evidence scope:

```text
source panel: probe_panel_selector/v3_val200_object_panel_diverse_quota
position rows: position_plan/v1_selected_bases_treatment_only/position_rows.jsonl
context source: paired model's natural pre_x1 row for matched_pred_idx
source_count: 32
planned_forward_count: 66
row_count: 528
readout_status: ok for all rows
layers: 0, 8, 16, 20, 23, 24, 27 plus true forward logits
training_ran: false
```

Probe design:

```text
For each selected false-negative object:
  1. locate the paired model prediction that matched the same GT;
  2. fetch that paired model's natural pre_x1 prefix for matched_pred_idx;
  3. replay the exact prefix into both the failed checkpoint and the paired
     checkpoint;
  4. read x1 under two targets:
       target_gt: the selected GT x1 bin,
       paired_generated: the paired model's natural generated x1 bin;
  5. when the paired prefix descriptor disagrees with the GT descriptor, also
     retarget only the final descriptor while keeping the preceding context.
```

Aggregate result from all selected FN rows:

```text
failed model on paired context:
  natural_gt_x1:
    n=16, rank<=20: 1, top distance<=16: 2, median rank: 194
  natural_generated_x1:
    n=16, rank<=20: 7, top distance<=16: 4, median rank: 71

paired model on own context:
  natural_gt_x1:
    n=16, rank<=20: 3, top distance<=16: 5, median rank: 53
  natural_generated_x1:
    n=16, rank<=20: 16, top distance<=16: 8, median rank: 2
```

Interpretation:

```text
Natural paired context is not a universal repair for GT x1. It is much more
reliable at making both checkpoints favor the paired model's generated x1 than
the exact GT x1. This means the paired prefix carries an object-anchor
commitment, but that commitment is often a model-generated anchor rather than a
ground-truth-correct anchor.

For failed models, 7/16 selected rows put the paired generated x1 into rank<=20,
while only 1/16 put the exact GT x1 into rank<=20. The core bottleneck therefore
looks like anchor selection and anchor inheritance, not simply missing
descriptor guidance.
```

Concrete failed-model context-following cases:

```text
rank 27, sorted failed on image 2685 GT bottle:
  paired natural context came from random pred_idx=31, descriptor=wine glass
  GT x1=190, paired generated x1=209
  failed model under paired natural context:
    target generated x1: rank 14, top 216, distance 7
    target GT x1:        rank 75, top 216, distance 26
  retarget final descriptor wine glass -> bottle:
    target GT x1:        rank 22, top 197, distance 7

rank 30, random failed on image 16228 GT person:
  GT x1=13, paired generated x1=0
  failed model under paired natural context:
    target generated x1: rank 1, top 0, distance 0
    target GT x1:        rank 225, top 0, distance 13

rank 36, random failed on image 19432 GT chair:
  GT x1=351, paired generated x1=354
  failed model under paired natural context:
    target generated x1: rank 7, top 0, distance 354
    target GT x1:        rank 135, top 0, distance 351

rank 44, random failed on image 2157 GT knife:
  GT x1=338, paired generated x1=390
  failed model under paired natural context:
    target generated x1: rank 2, top 387, distance 3
    target GT x1:        rank 48, top 387, distance 49
```

Layerwise observations:

```text
rank 27, sorted failed, paired context retargeted to bottle:
  L20: rank 9, top 197, distance 7
  L23: rank 11, top 197, distance 7
  L27/final: rank 22, top 197, distance 7
  The target anchor becomes available by L20 and survives to final.

rank 27, sorted failed, paired natural descriptor wine glass:
  L20: rank 5, top 197, distance 7
  L23: rank 7, top 197, distance 7
  L27/final: rank 75, top 216, distance 26
  The model enters a useful nearby basin by L20, but late layers redirect toward
  the paired generated anchor.

rank 28, random failed, paired sorted context for person:
  L20: rank 207, top 197, distance 34
  L23: rank 570, top 0, distance 163
  L27/final: rank 676, top 0, distance 163
  A weak mid-layer improvement is overwritten by a late border-anchor collapse.

rank 32, random failed, paired sorted context for wine glass:
  L23: rank 27, top 3, distance 69
  L24: rank 21, top 3, distance 69
  L27/final: rank 21, top 0, distance 72
  The target rank improves late, but the top bin remains a border anchor.
```

Mechanistic update:

```text
The failure family should now be described as:

  descriptor guidance -> can raise target basin evidence,
  paired natural context -> can transfer an object-anchor commitment,
  late coordinate router -> often chooses the inherited/generated anchor or a
     border anchor instead of the exact GT anchor.

This explains why a paired model can "recover" a false-negative GT geometrically
while still carrying the wrong descriptor or an offset box, and why transferring
its natural prefix does not automatically recover the GT in the failed model.
The model is not merely deciding whether an object exists; it is inheriting and
committing to a specific autoregressive spatial anchor.
```

Next surgery direction:

```text
Run an anchor-substitution causal patch:
  - hold the paired natural context fixed,
  - compare residual/attention states for generated-x1 target versus GT-x1
    target at L20, L23, L24, L27,
  - inject a GT-x1 direction into the same prefix where generated-x1 is favored,
  - test whether later y1/x2/y2 follow the corrected anchor or snap back to the
    inherited generated anchor.

This is the direct test for whether x1 is a causal binding key that determines
the rest of the object span.
```

## 2026-06-25 continuation: contextual anchor substitution and strict x1 controls

New helper surfaces:

```text
src/analysis/prefix_denoising_surgery_probing/contextual_anchor_substitution.py
scripts/analysis/run_prefix_denoising_contextual_anchor_substitution.py

src/analysis/prefix_denoising_surgery_probing/anchor_escape_x1_controls.py
scripts/analysis/run_prefix_denoising_anchor_escape_x1_controls.py
```

All probes here are readout/surgery probes only. No training ran.

### Representative sample-base rule

The next phase should stay sample-base centered rather than metric centered.
Do not spend expensive hidden-state and tensor-flow analysis on normal,
well-learned images unless they serve as a direct negative control for a hard
case. Maintain a compact rotating case book and promote a new image only when
it exposes a distinct mechanistic pressure:

```text
1. anchor saturation or border-anchor collapse,
2. nearby same-class competitor binding,
3. valid-looking local coordinate strip without object scaffold,
4. y1/x2/y2 tail closure ridge after a plausible x1,
5. false negative that becomes recoverable under guided prefix,
6. premature termination or under-span after correct early coordinates.
```

Current high-value case bases:

```text
image 12670 / person:
  hard sorted-denoise person FN / coord_0 or full-image basin.
  Good for testing whether x1 is a binding key or only coordinate-slot entry.

image 19432 / chair:
  random-denoise chair duplication/anchor family with multiple same-class
  chairs. Good for testing same-class spatial selector versus y1/tail repair.

image 2685 / bottle-wine-glass:
  descriptor/context transfer case where x1+y1 repairs much more than x1 alone.

image 16228 / small person:
  boundary/small-object under-span case where x1/y1/x2 guidance still leaves y2
  short. Good for tail-closure and premature termination probes.
```

### Contextual false-negative anchor substitution

Artifact roots:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/contextual_anchor_substitution/v1_top4_dryrun
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/contextual_anchor_substitution/v1_top4_sorted_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/contextual_anchor_substitution/v1_top4_random_gpu1
```

Scope:

```text
source_count=4
plan_row_count=24
real rows=24
errors=0
models: sorted_denoise, random_denoise
selected source ranks: 27, 28, 29, 30
```

Summary semantics after the review patch:

```text
row_counts_by_forced_x1_reference includes explicit none buckets for free rows.
quality labels and threshold counts require generated_object_parse_status ==
complete_valid; complete_invalid rows are incomplete_or_invalid even if their
raw coordinate overlap would otherwise look high.
```

Key row-level outcomes:

```text
rank 27, sorted image 2685, GT bottle under paired wine-glass context:
  pre_x1_free:              [216,589,254,623], IoU target 0.060
  force_gt_x1:              [190,591,254,621], IoU target 0.088
  force_paired_generated_x1:[209,589,254,622], IoU target 0.071
  force_gt_x1_y1:           [190,491,246,622], IoU target 0.559
  force_gt_x1_y1_x2:        [190,491,283,622], IoU target 0.885

rank 29, sorted image 12670, GT person:
  pre_x1_free:              [460,999,553,999], invalid/degenerate
  force_gt_x1:              [344,999,468,796], invalid/degenerate
  force_paired_generated_x1:[354,999,480,792], invalid/degenerate
  force_gt_x1_y1:           [344,365,474,796], IoU target 0.894
  force_gt_x1_y1_x2:        [344,365,461,796], IoU target 0.993

rank 28, random image 12670, GT person:
  pre_x1_free:              [0,632,109,999], IoU target 0.000
  force_gt_x1:              [163,281,303,664], IoU target 0.882
  force_paired_generated_x1:[157,281,303,652], IoU target 0.825

rank 30, random image 16228, GT person:
  pre_x1_free:              [0,460,40,574], IoU target 0.307
  force_gt_x1:              [13,460,40,553], IoU target 0.282
  force_gt_x1_y1:           [13,506,40,540], IoU target 0.282
  force_gt_x1_y1_x2:        [13,506,48,526], IoU target 0.215
```

Interpretation:

```text
x1 alone is not a universal object-binding key.

Some false negatives are repaired once x1 and y1 are both forced; x1+y1+x2 can
then unlock the correct y2/tail. Other cases remain short or degenerate even
after x1/y1/x2, implying a later tail-closure or stop/height bottleneck.

The meaningful unit is therefore not "the model sees object or not"; it is the
route by which descriptor/context evidence becomes a stable coordinate scaffold
and then a closed object span.
```

### Strict anchor-escape x1 key-vs-control panel

Artifact roots:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_x1_controls/v1_strict4_dryrun
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_x1_controls/v1_strict4_sorted_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_x1_controls/v1_strict4_random_gpu1
```

Scope:

```text
input: /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_plan/v2_saturated_direction_coord0_gt_candidates/anchor_escape_rows.jsonl
source rows=4
planned rows=20
real rows=20
errors=0
modes: pre_x1_free, force_target_x1, force_alt_same_desc_x1,
       force_original_generated_x1, force_target_x1_plus16
```

Sorted-denoise, image 12670/person:

```text
GT15 target x1=492:
  pre_x1_free:          [0,0,999,999], target IoU 0.025
  force_target_x1:      [492,40,528,166], target IoU 0.000
  force_alt_same_desc:  [344,0,411,254], target IoU 0.000
  force_original_x1=0:  [0,0,999,999], original-control IoU 0.988
  force_target_x1+16:   [508,49,556,138], target IoU 0.000

GT8 target x1=344:
  pre_x1_free:          [0,0,999,999], target IoU 0.050
  force_target_x1:      [344,0,411,254], target IoU 0.000
  force_alt_same_desc:  [492,40,528,166], target IoU 0.000
  force_original_x1=0:  [0,0,999,999], original-control IoU 0.988
  force_target_x1+16:   [360,0,436,138], target IoU 0.000
```

Random-denoise, image 19432/chair:

```text
GT8 target x1=537:
  pre_x1_free:          [0,0,73,270], target IoU 0.000
  force_target_x1:      [537,0,651,347], target IoU 0.647
  force_alt_same_desc:  [351,0,467,342], control IoU 0.584
  force_original_x1=0:  [0,0,73,270], control IoU 0.066
  force_target_x1+16:   [553,0,651,347], target IoU 0.585

GT5 target x1=351:
  pre_x1_free:          [0,0,73,270], target IoU 0.000
  force_target_x1:      [351,0,467,342], target IoU 0.584
  force_alt_same_desc:  [537,0,651,347], control IoU 0.647
  force_original_x1=0:  [0,0,73,270], control IoU 0.066
  force_target_x1+16:   [367,0,480,342], target/control mixed IoU about 0.47/0.51
```

Mechanistic update:

```text
There are at least two distinct regimes.

1. x1-gated same-class spatial selector:
   In random-denoise image 19432/chair, forcing x1 selects the corresponding
   same-class chair column. Forcing the alternate same-desc x1 switches to the
   alternate chair basin. However y1 stays at 0 instead of the GT y1=122, so x1
   opens the right horizontal basin but does not repair the vertical scaffold.

2. coordinate-entry without object scaffold:
   In sorted-denoise image 12670/person, forcing target or alternate x1 does not
   bind either true person. The model emits a valid-looking shallow top strip at
   the forced x1. This means the coord token can steer local numeric emission
   while the object/visual scaffold remains absent or overwritten.

Therefore x1 should not be treated as the binding key by itself. A stronger
candidate mechanism is a staged scaffold:

  descriptor/object evidence
    -> x1 route entry or same-class spatial selector
    -> y1 vertical-anchor commitment
    -> x2 width/extent commitment
    -> y2 closure / stop decision.

The failure location differs by case within these prefix-denoising checkpoints.
This motivates a controlled baseline comparison before attributing the staged
recoverability pattern to prefix denoising itself.
```

Next focused probes:

```text
1. For image 12670/person, compare x1-only versus x1+y1 hidden-state patching:
   if x1+y1 repairs while x1-only produces shallow strips, y1 is the first true
   object-scaffold commitment in this case.

2. For image 19432/chair, patch y1 route after forced x1:
   if y1=122 converts [x1,0,x2,347] into [x1,122,x2,348], the missing mechanism
   is vertical-anchor repair rather than object selection.

3. For image 16228/person, patch y2/termination after x1+y1+x2:
   if y2 remains short, inspect stop/continue logits and y2 coordinate-basin
   attraction rather than x1/y1 object perception.

4. Expand the case book only by adding another image that cleanly represents a
   missing archetype, not by averaging over normal images.
```

### Slot-scaffold hidden-state readout

Artifact roots:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/slot_scaffold_readout/v1_anchor_escape_dryrun
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/slot_scaffold_readout/v1_12670_sorted_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/slot_scaffold_readout/v1_19432_random_gpu1
```

Implementation:

```text
src/analysis/prefix_denoising_surgery_probing/slot_scaffold_readout.py
scripts/analysis/run_prefix_denoising_slot_scaffold_readout.py
tests/analysis/test_prefix_denoising_slot_scaffold_readout.py
```

Scope:

```text
input: /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_plan/v2_saturated_direction_coord0_gt_candidates/anchor_escape_rows.jsonl
source rows=4
dry-run plan rows=20
real rows=200 total, 100 per checkpoint split
errors=0
layers: 0, 8, 16, 20, 22, 23, 24, 25, 26, 27
modes:
  force_target_x1_pre_y1
  force_alt_x1_pre_y1
  force_original_x1_pre_y1
  force_target_x1_y1_pre_x2
  force_target_x1_y1_x2_pre_y2
```

Representative sample-base rule:

```text
Do not spend the deep hidden-state/surgery budget on normal or already well
learned images. Use a small case book of high-influence image bases where one
coordinate or object-binding stage changes the visible failure. Expand the book
only when a new image represents a different failure archetype, for example:

1. same-class horizontal selector works but vertical scaffold is missing;
2. coordinate token steers local numeric emission but does not bind the object;
3. x1+y1+x2 are present but y2/termination collapses;
4. false negative is repaired by language-side guidance versus not visually
   represented;
5. train-set failure differs from val-set failure under the same probe.

Current narrow bases:

- sorted-denoise, image 12670/person pair: coordinate-entry without object
  scaffold; forcing x1 produces shallow strips instead of either true person.
- random-denoise, image 19432/chair pair: x1 selects the same-class chair
  column, while y1 remains fragile and x2 resolves late after x1+y1.
```

Layer readout findings:

```text
Sorted-denoise image 12670/person:

Final layer 27, GT15:
  force_target_x1_pre_y1:      target y1=403 rank 425, top_peak=40, tied
  force_target_x1_y1_pre_x2:   target x2=615 rank 155, top_peak=520, tied
  force_target_x1_y1_x2_pre_y2:target y2=608 rank 389, top_peak=415, tied

Final layer 27, GT8:
  force_target_x1_pre_y1:      target y1=365 rank 323, top_peak=0, tied
  force_target_x1_y1_pre_x2:   target x2=461 rank 82, top_peak=419, tied
  force_target_x1_y1_x2_pre_y2:target y2=793 rank 572, top_peak=415, tied

No scaffold mode produced a clean target winner in the strict posterior
taxonomy. The late y2 readout is especially informative: even after x1,y1,x2
are forced from the target GT, y2 is attracted toward about 415 or nearby
shallow-strip values, matching the bad decoded tails observed in tail-binding
rollouts. This is evidence against "the missing x1 alone caused the failure".
The failure is downstream: object scaffold or vertical/height closure is not
stable in the hidden state even when the prefix contains plausible coordinates.

Random-denoise image 19432/chair:

Final layer 27, GT8:
  force_target_x1_pre_y1:      target y1=122 rank 139, top_peak=0, tied
  force_target_x1_y1_pre_x2:   target x2=651 rank 1, top_peak=651, target
  force_target_x1_y1_x2_pre_y2:target y2=348 rank 4, top_peak=343, tied

Final layer 27, GT5:
  force_target_x1_pre_y1:      target y1=122 rank 32, top_peak=0, tied
  force_target_x1_y1_pre_x2:   target x2=458 rank 23, top_peak=467, background
  force_target_x1_y1_x2_pre_y2:target y2=348 rank 5, top_peak=342, tied

The chair pair shares y1=122 and y2=348, so y-slot "tied" labels are expected
and should not be over-interpreted as failure. The discriminating signal is the
x route. After x1,y1 are supplied, late layers can form a target-specific x2
basin, cleanly for GT8 at layer 27 and near-target for GT5. This matches the
decoded behavior where forcing x1 opened the right same-class column but y1
guidance was needed for complete object-span repair.
```

Mechanistic update:

```text
Prefix-denoising did not simply remove exposure bias. At least in these two
high-value bases, it exposes a staged scaffold mechanism:

1. Descriptor and image context choose an object-state basin.
2. x1 can act as a same-class horizontal selector, but only in some images.
3. y1 is the first vertical anchor commitment and may remain language/prefix
   fragile.
4. x2 can become target-specific late after x1+y1, especially in the random
   chair case.
5. y2/termination may collapse to a learned shallow strip or shared vertical
   closure even when earlier slots are forced.

The core question for the next surgery is therefore not "does the model perceive
the object", but "which residual route turns object evidence into a stable
coordinate scaffold, and at which slot does the route become overwritten by a
generic coordinate basin".
```

Next focused probes:

```text
1. Route patching for sorted-denoise image 12670/person:
   patch y2 or late residual components after force_target_x1_y1_x2 and ask
   whether the 415 shallow-strip basin can be replaced by target y2. If yes,
   y2 closure is recoverable but overwritten. If no, the object scaffold was
   never represented in the relevant residual path.

2. Route patching for random-denoise image 19432/chair:
   patch around the late x2 transition from layers 24-27 after x1+y1 and test
   whether GT5's near-target/background x2 can be sharpened into the same clean
   target basin observed for GT8.

3. Baseline bridge:
   rerun the slot-scaffold readout by remapping these same scaffold prefixes to
   a sorted pure-CE/full-wrapper baseline checkpoint. Use this only as a bridge
   comparison, not as a broad metric study.

4. Case-book expansion:
   add one train-set failure and one val-set failure only if they instantiate a
   new archetype beyond the current person/chair pair.
```

### Y2 route intervention bridge: causal replacement of the 415 basin

Artifact roots:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_plan/v1_force_x1_y1_x2
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_group_intervention/v3_readout_bridge_fine_12670_sorted_top1_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_route_group_intervention/v3_readout_bridge_wide_19432_random_top1_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_tensor_flow/v2_selector_layers23_27_12670_sorted_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_tensor_flow/v2_selector_layers23_27_19432_random_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_value_region_intervention/v4_basin_12670_sorted_l23h6_regions_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_value_region_intervention/v4_basin_19432_random_l27h14_regions_gpu0
```

Join check:

```text
The y2 route-plan prefixes in v1_force_x1_y1_x2 match the slot-scaffold
force_target_x1_y1_x2_pre_y2 prefixes for all four current case-book rows.
Therefore the route-group intervention evidence can be interpreted as the
causal follow-up to the slot-scaffold y2 readout.
```

Sorted-denoise, image 12670/person:

```text
Baseline route-plan state after forcing target x1,y1,x2:

GT8 target y2=793:
  baseline top1 y2 = 415
  baseline target rank = 571

GT15 target y2=608:
  baseline top1 y2 = 415
  baseline target rank = 474

Exact y2-box-tail repair exists under readout-bridge intervention:

GT8 target y2=793:
  current_negative_suppress_top1, alpha=0.2
    patched top1=793, target rank=1, first token=<|coord_793|>,
    second token=<|box_end|>, repair=y2_box_tail_complete
  template_negative_suppress_top1, alpha=0.15 or 0.2
    patched top1=793, target rank=1, first token=<|coord_793|>,
    second token=<|box_end|>, repair=y2_box_tail_complete
  current_signed_combo_top1, alpha=0.075 through 0.15
    patched top1=793, target rank=1, first token=<|coord_793|>,
    second token=<|box_end|>, repair=y2_box_tail_complete

GT15 target y2=608:
  current_positive_amplify_top1, alpha=0.125 through 0.2
    patched top1=608, target rank=1, first token=<|coord_608|>,
    second token=<|box_end|>, repair=y2_box_tail_complete
  template_positive_amplify_top1, alpha=0.125 through 0.2
    patched top1=608, target rank=1, first token=<|coord_608|>,
    second token=<|box_end|>, repair=y2_box_tail_complete
```

Contrast with narrower value-region scaling:

```text
The v4 single-head/value-region basin panel did not repair sorted image
12670/person. With layer 23 head 6 and region scales 0,1,2 over
current_forced_coords/current_partial_object/recent_16/prompt_non_image/all_prefix:

GT8 target y2=793:
  best observed top1 was 609, target rank about 452-469, not repaired.

GT15 target y2=608:
  best observed top1 stayed at 415 or moved to 418/421, not repaired.

Thus the repair is not a simple "scale one value source region" effect. It
requires a more global route/readout bridge direction or a combination of
current-coordinate/template components.
```

Random-denoise, image 19432/chair:

```text
Baseline y2 after forcing x1,y1,x2 is already near target:

target y2=348, generated y2=342 or 343.

Route-group intervention reaches exact y2=348 at broader and more stable alpha
ranges than sorted 12670/person. Examples:

generated y2=342:
  current_positive_amplify_top1, alpha=0.3 through 1.0 -> top1=348, rank=1,
  y2_box_tail_complete.
  template_positive_amplify_top1, alpha=0.25 through 1.0 -> top1=348, rank=1,
  y2_box_tail_complete.
  template_negative_suppress_top1, alpha=0.2 through 1.0 -> top1=348, rank=1,
  y2_box_tail_complete.

The chair case therefore has a shallower basin gap: the natural state is near
the correct y2, and route intervention stabilizes a nearby basin. The person
case has a much larger basin jump from 415 to 608/793 and requires finer bridge
direction/alphabet-specific signs.
```

Mechanistic update:

```text
The sorted 12670/person y2 failure is not proof that the target object or target
y2 is absent from the model. The target y2 closure is causally reachable and can
produce the correct immediate <|coord_y2|><|box_end|> tail.

The failure is better described as a route-selection or basin-selection failure:
after the correct x1,y1,x2 scaffold, the live residual state falls into a learned
generic/shallow y2 attractor at 415. A bridge direction can redirect the
coordinate lens to the true target y2, but a narrow single-head value-region
scale is insufficient.

This sharpens the staged scaffold model:

  x1,y1,x2 scaffold can be syntactically correct
    -> live y2 readout may still choose a generic closure basin
    -> readout bridge can restore exact y2 and stop
    -> the remaining question is whether the bridge direction is naturally
       computed somewhere and then suppressed, or whether it is only a linear
       readout artifact not used by natural autoregression.
```

Next focused probes:

```text
1. Natural-donor activation patch:
   Replace artificial readout-bridge directions with residual/attention/MLP
   donor states from a naturally successful or near-successful y2 case. Test
   whether donor patching reproduces exact y2 closure in sorted 12670/person.
   This separates "linear basin is reachable" from "the model naturally computes
   a transferable y2 route".

2. Bridge localization:
   For sorted 12670/person, localize the successful readout bridge by component
   and layer: attention update vs MLP update vs layer output, and current
   coordinate vs template directions. The current evidence points at route-level
   bridge directions, not one value-region scale.

3. Train/val case-book expansion:
   Add one train-set failure only after the natural-donor patch is scoped, so
   train-vs-val comparison asks whether the same bridge suppression happens on
   memorized images.
```

## 2026-06-25 Natural Donor Transplant Follow-Up

Implemented a sample-base-centered natural donor transplant probe:

```text
src/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant.py
scripts/analysis/run_prefix_denoising_y2_natural_donor_transplant.py
tests/analysis/test_prefix_denoising_y2_natural_donor_transplant.py
```

The probe builds explicit receiver/donor plan rows around selected image bases,
then injects natural donor component deltas into the receiver y2 decision token:

```text
delta = donor_component - receiver_component
patch sites = layer_output, self_attn, mlp
layers = 23..27
controls = baseline_no_patch, self_noop, same-image natural donor
readouts = receiver target y2, generated/stuck y2, donor y2
continuation = 2 greedy tokens to check <|coord_y2|><|box_end|>
```

Artifacts:

```text
Dry-run plan:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant/v1_selected_bases_dryrun

Initial matched-donor focused runs:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant/v1_focused_12670_sorted_layers23_27_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant/v1_focused_19432_random_layers23_27_gpu1

Near-target donor focused runs:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant/v2_near_y2_12670_sorted_layers23_27_gpu2
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant/v2_near_y2_19432_random_layers23_27_gpu3
```

Key result:

```text
No natural donor transplant flipped the receiver y2 top1 to the receiver target:
  sorted 12670 v1: receiver_target_top1_flip_count = 0 / 122
  random 19432 v1: receiver_target_top1_flip_count = 0 / 122
  sorted 12670 v2 near-y2: receiver_target_top1_flip_count = 0 / 122
  random 19432 v2 near-y2: receiver_target_top1_flip_count = 0 / 122

Natural donor transplants often moved the basin strongly:
  sorted 12670 v1: donor_y2_top1_transport_count = 10
  random 19432 v1: donor_y2_top1_transport_count = 15
  sorted 12670 v2 near-y2: donor_y2_top1_transport_count = 22
  random 19432 v2 near-y2: donor_y2_top1_transport_count = 10
```

The v2 donor selection was adjusted after seeing that evaluation-quality donors
were hiding stronger coordinate probes. It now prioritizes exact target-GT
donors if present, then nearest same-description y2 donors, then matched donors.
For sorted image 12670 this selected near-target person donors:

```text
receiver target y2=608:
  donor y2=605, false_positive, distance 3
  donor y2=609, false_positive, distance 1
  donor y2=617, false_positive, distance 9

receiver target y2=793:
  donor y2=796, false_positive, distance 3
  donor y2=617, false_positive, distance 176
  donor y2=648, matched, distance 145
```

Even the near-target donors did not repair the receiver target exactly. The
clearest layer-output rows:

```text
target y2=608, generated/stuck y2=415

donor y2=605:
  layer_output 23..27 -> patched top1=605, continuation=<|coord_605|><|box_end|>
  target rank improves from 474 to 20/15/9/9/12
  donor y2 rank becomes 1

donor y2=609:
  layer_output 23..27 -> patched top1=617, continuation=<|coord_617|><|box_end|>
  target rank improves from 474 to 34/27/27/18/18
  donor y2 rank becomes 3/4/2/4/2

donor y2=617:
  layer_output 23..27 -> patched top1=617, continuation=<|coord_617|><|box_end|>
  target rank improves from 474 to 18/18/17/17/16
  donor y2 rank becomes 1

target y2=793, generated/stuck y2=415

donor y2=796:
  layer_output 23..27 -> patched top1=796, continuation=<|coord_796|><|box_end|>
  target rank improves from 571 to 7/6/6/6/6
  donor y2 rank becomes 1
```

Mechanistic interpretation update:

```text
Natural donor component states are not inert. They can override the receiver's
generic 415 y2 basin and carry a local coordinate-closure attractor through the
causal forward path. However, the attractor follows the donor/local basin, not
the receiver's intended GT y2.

This separates two phenomena:

1. A coordinate closure route exists in natural hidden states and can be
   transplanted.
2. The receiver-specific binding from x1,y1,x2 to the correct y2 is still not
   recovered by naive donor transfer, even with near-target donors.

Together with the earlier readout-bridge repair, the current best hypothesis is:

  hidden state contains coordinate-basin content that is causally usable,
  but exact receiver-target y2 requires a binding/readout bridge that is more
  object-conditioned than raw donor layer output/self-attn/MLP deltas.

The donor 609 -> top1 617 snap is especially important: it suggests a jagged
coordinate attractor landscape. This matches the prior observation that CE
supervision preserves geometry locality but destroys smoothness. The model can
move to a nearby coordinate neighborhood, but the local basin winner is not a
smooth interpolation to the receiver target.
```

Next probes after this result:

```text
1. Donor-minus-receiver decomposition:
   Decompose the successful layer_output donor transport into attention vs MLP
   residual deltas and earlier/later accumulated residual state. Current rows
   show layer_output is the strongest exact donor-basin carrier.

2. Target-bridge vs donor-basin contrast:
   For the same receiver and donor rows, compare artificial readout-bridge
   direction against natural donor deltas in the same layer/site table. Ask what
   vector component is present in bridge repair but absent from natural donor
   transport.

3. Coordinate local smoothness probe at y2:
   Around donor y2 605/609/617 and target 608, inspect coord-token logit local
   surfaces and special-token embeddings. The 609 -> 617 snap is a concrete
   local-basin pathology to explain.

4. Visual/perception vs language guidance false-negative probe:
   Reuse the same sample-base planning machinery, but use missing-object/FN
   receivers and natural/teacher-forced guidance prefixes to test whether visual
   evidence exists but the language/object route fails to activate.
```

## 2026-06-25 Local Coordinate-Basin Geometry Follow-Up

After the natural donor transplant showed a concrete snap pattern
(`target 608`, `donor 609` often emitting `617`), ran the existing
`coord_token_geometry_probe` as a selected-bin local-basin panel instead of a
broad coord vocabulary survey.

Artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coord_token_geometry_probe/v2_local_basin_12670_sorted_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coord_token_geometry_probe/v2_local_basin_19432_random_gpu1
```

Probe scope:

```text
selected bins:
  342,343,347,348,354,362,415,432,605,608,609,617,793,796,999

selected states:
  sorted_denoise image 12670, forced y2 receivers for target 793 and 608
  random_denoise image 19432, forced y2 receivers for target 348

surfaces:
  base_input, adapter_embed_offset, effective_input
  base_output, adapter_head_offset, effective_output

state panels:
  state_bin_decomposition
  state_adapter_counterfactual
  state_effective_readout_surgery
```

Surface-level result:

```text
The effective coordinate-token output surface is locally smooth around the
selected bins in both checkpoints. Adjacent-bin distances are small and nearly
identical across sorted/random prefix-denoising checkpoints.

Examples:

sorted effective_output:
  608-609 distance 0.2025, cosine 0.9880
  793-796 distance 0.4931, cosine 0.9286
  342-343 distance 0.1980, cosine 0.9887
  347-348 distance 0.2040, cosine 0.9881

random effective_output:
  608-609 distance 0.2017, cosine 0.9881
  793-796 distance 0.4918, cosine 0.9289
  342-343 distance 0.1993, cosine 0.9885
  347-348 distance 0.2043, cosine 0.9880

Local second-difference norms are also ordinary, around 0.148-0.167 on the
selected bins.
```

This means the observed donor 609 -> top1 617 snap is not explained by an
obviously jagged coordinate-token vector surface. The surface preserves local
geometry well enough.

State-readout result:

```text
Sorted 12670, target y2=608, generated/stuck y2=415:
  415 rank=1
  608 rank=476
  605 rank=423
  609 rank=451
  617 rank=503

Sorted 12670, target y2=793, generated/stuck y2=415:
  415 rank=1
  793 rank=576
  796 rank=516
  609 rank=3
  617 rank=13

Random 19432, target y2=348:
  generated 342 case: 342 rank=1, target 348 rank=8
  generated 343 case: 343 rank=1 or 3, target 348 rank=8 or 9
```

So the jaggedness is in the hidden-state query/readout over a mostly smooth
coordinate surface. The same coordinate vectors can be locally smooth while the
current residual state scores nonlocal bins sharply.

Adapter contribution result:

```text
Sorted 12670, target 608 state:
  adapter_head_logit_delta at target 608 = -0.1877
  zero_selected_adapter improves target rank 479 -> 450
  smooth_selected_adapter improves target rank 479 -> 473
  swap target/neighbor adapter improves target rank 479 -> 448

Sorted 12670, target 793 state:
  adapter_head_logit_delta at target 793 = +0.4639
  zero_selected_adapter worsens target rank 572 -> 595
  smooth_selected_adapter slightly worsens target rank 572 -> 578
  swap target/neighbor adapter improves target rank 572 -> 562

Random 19432, target 348 states:
  adapter_head_logit_delta at target 348 = +1.30 or +1.22
  zero_selected_adapter worsens target rank 7 -> 35 or 9 -> 45
  smooth_selected_adapter keeps/improves target rank 7 -> 7 or 9 -> 5
  swap target/neighbor adapter improves target rank 7 -> 4 or 9 -> 2
```

The token embedding adapter is therefore not globally harmful. It helps the
random chair target and the sorted 793 target, but it actively hurts sorted 608
in this hidden state. This is a state-conditioned interaction, not a simple
"adapter destroyed coord geometry" story.

Readout-surgery result:

```text
Sorted 12670:
  target 793 flips to target at alpha=0.2 against observed/generated 415
  target 608 flips to target at alpha=0.2 against observed/generated 415
  top8_centroid can also flip to target at alpha=0.1 or 0.2 depending on case

Random 19432:
  target 348 from generated 342 flips at alpha=0.2 against observed/generated,
  or alpha=0.05 against top8_centroid.
  target 348 from generated 343 flips at alpha=0.1 against generated_bin,
  or alpha=0.05 against top8_centroid.
```

Mechanistic interpretation update:

```text
The coordinate vectors themselves look locally structured. The failure is not
"coord token geometry is globally scrambled."

The failure is a hidden-state basin query problem:
  the live y2 residual state projects a smooth coordinate surface into a jagged
  score landscape whose winners can be 415, 617, 999, or other basin bins.

Natural donor layer-output patches can carry a donor-local y2 attractor through
the causal path, but they do not install the receiver-specific binding vector.
Artificial readout surgery can still install the missing target-vs-antagonist
direction. Therefore the missing component is likely not coordinate-token
surface geometry; it is a state-conditioned bridge from the current object box
scaffold to the target coordinate readout.
```

Next probe refinement:

```text
Do target-bridge vs donor-basin vector comparison in the same vector space:
  donor_delta = natural donor layer_output - receiver layer_output
  bridge_delta = effective/readout target-antagonist direction

For each receiver/donor row, measure:
  projection(donor_delta, bridge_delta)
  projection(residual_state, bridge_delta)
  whether removing the donor-local coordinate component exposes target bridge
  whether adding bridge_delta to donor_delta converts donor-basin transport into
  receiver-target repair

This should distinguish "natural donor lacks target bridge" from "natural donor
contains bridge but it is dominated by donor-local y2 basin content."
```

## 2026-06-25 Target-Bridge vs Donor-Basin Projection Follow-Up

Extended `y2_natural_donor_transplant.py` to add readout-direction projection
fields for each natural donor delta:

```text
donor_delta = donor_component - receiver_component

projection directions:
  target_minus_generated
  donor_minus_generated
  target_minus_donor

new diagnostic gap:
  donor_delta_donor_vs_target_projection_gap
    = projection(donor_delta, donor_minus_generated)
      - projection(donor_delta, target_minus_generated)
```

Artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant/v3_projection_12670_sorted_layer_output_gpu2
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant/v3_projection_19432_random_layer_output_gpu3
```

These v3 runs used only `layer_output` over layers 23..27 because the previous
natural-donor panel showed layer output was the strongest exact donor-basin
carrier.

Projection summary:

```text
sorted_denoise image 12670, layer_output natural donors:
  donor rows = 30
  receiver_target_top1_flip_count = 0
  donor_y2_top1_transport_count = 21

  donor_y2_basin_transport rows:
    mean projection target_minus_generated = 54.12
    mean projection donor_minus_generated = 65.01
    mean donor_vs_target_projection_gap = +10.89
    mean projection target_minus_donor = -9.63

  target_and_donor_rank_both_improved rows:
    mean projection target_minus_generated = 58.19
    mean projection donor_minus_generated = 85.69
    mean donor_vs_target_projection_gap = +27.50
    mean projection target_minus_donor = -17.51

random_denoise image 19432, layer_output natural donors:
  donor rows = 30
  receiver_target_top1_flip_count = 0
  donor_y2_top1_transport_count = 4

  donor_y2_basin_transport rows:
    mean projection target_minus_generated = 19.95
    mean projection donor_minus_generated = 21.93
    mean donor_vs_target_projection_gap = +1.98
    mean projection target_minus_donor = -5.82
```

Interpretation:

```text
Natural donor deltas usually do contain positive target-minus-generated
projection: they are not totally orthogonal to the artificial target bridge.

However, in the sorted hard case, the donor-minus-generated projection is larger
and the target-minus-donor projection is negative on average. The donor delta
therefore lifts the donor/local y2 basin more than the receiver target basin.
This explains why natural transplant can strongly improve target rank while
still emitting donor y2 or nearby donor-local basin winners.
```

Important nuance:

```text
For very near donors, the scalar projection gap is not sufficient by itself.
Some rows have target_minus_generated projection comparable to or larger than
donor_minus_generated projection, yet still emit the donor/local bin. That means
rank competition after the state is lifted matters too: the projection can move
the neighborhood, but local basin ordering decides the winner.

So the failure is not "natural donor has no bridge component." It is closer to:

  natural donor delta contains a broad coordinate-closure lift,
  but its local basin component and rank ordering remain donor-centered.
```

Updated best mechanism picture:

```text
1. The coord-token vector surface is locally smooth enough.
2. The live receiver y2 state queries that surface in a jagged, basin-like way.
3. Natural donor layer-output deltas carry causal y2-closure content.
4. That content includes some target-bridge-aligned projection, but the stronger
   component is donor/local-basin-aligned.
5. Artificial bridge directions repair exact target y2 because they isolate the
   target-vs-antagonist readout axis instead of importing a full donor-local
   object-span state.
```

Next surgical move:

```text
Construct a two-component patch:

  donor_delta_projected = donor_delta - proj(donor_delta, donor_minus_generated)
  bridge_plus_donor = donor_delta + beta * target_minus_generated_unit

Test whether removing donor-local projection or adding target bridge converts
donor-basin transport into receiver-target repair on the same sorted 12670 rows.

This is the cleanest next causal separation between:
  "donor delta lacks exact target bridge"
and
  "donor delta has bridge but donor-local component dominates local rank."
```

## 2026-06-25 Donor-Basin Removal And Target-Bridge Surgery

Implemented explicit natural-donor patch transforms in
`src/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant.py`:

```text
donor_delta:
  original donor_vec - receiver_vec patch

remove_donor_minus_generated:
  donor_delta - projection(donor_delta, donor_token - generated_token)

bridge_plus_donor:beta:
  donor_delta + beta * ||donor_delta|| * unit(target_token - generated_token)

target_bridge:beta:
  beta * ||donor_delta|| * unit(target_token - generated_token)
```

Artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant/v4_bridge_transform_12670_sorted_layer_output_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant/v4_bridge_transform_19432_random_layer_output_gpu1

/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant/v5_layer27_bridge_scale_sweep_12670_sorted_gpu2
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant/v5_layer27_bridge_scale_sweep_19432_random_gpu3
```

Scope:

```text
v4:
  sample bases: sorted_denoise image 12670, random_denoise image 19432
  receivers per base: 2
  donors per receiver: 3
  layers: 23..27
  site: layer_output
  transforms:
    donor_delta
    remove_donor_minus_generated
    bridge_plus_donor:0.05
    bridge_plus_donor:0.1
    target_bridge:0.05
    target_bridge:0.1
  rows: 192 per checkpoint, zero errors

v5:
  same sample bases
  layer: 27 only
  transforms:
    donor_delta
    target_bridge:0.2
    target_bridge:0.5
    target_bridge:1.0
    bridge_plus_donor:0.2
    bridge_plus_donor:0.5
  rows: 40 per checkpoint, zero errors
```

Key observations:

```text
sorted_denoise image 12670, v4 donor rows:
  donor_delta:
    donor-y2 top1 transport = 21 / 30
    target top1 repair = 0 / 30
    mean target-minus-donor y2 logit delta = -1.0396

  remove_donor_minus_generated:
    donor-y2 top1 transport = 7 / 30
    target top1 repair = 0 / 30
    mean target-minus-donor y2 logit delta = -0.4667

  bridge_plus_donor:0.1:
    donor-y2 top1 transport = 5 / 30
    target top1 repair = 3 / 30
    all three repairs occur at layer 27
    mean target-minus-donor y2 logit delta = -0.2062

  target_bridge:0.1:
    donor-y2 top1 transport = 0 / 30
    target top1 repair = 0 / 30
    target rank improved = 30 / 30
    mean target-minus-donor y2 logit delta = +0.7354

random_denoise image 19432, v4 donor rows:
  donor_delta:
    donor-y2 top1 transport = 4 / 30
    target top1 repair = 0 / 30

  remove_donor_minus_generated:
    donor-y2 top1 transport = 0 / 30
    target top1 repair = 0 / 30

  target_bridge:0.1:
    donor-y2 top1 transport label = 10 / 30
    target top1 repair = 0 / 30
    row-level inspection shows this is mostly local-neighbor coordinate behavior:
      target_y2 = 348
      frequent patched top1 = 347
      target rank often = 2
```

Layer-27 scale sweep:

```text
sorted_denoise image 12670:
  target_bridge:0.2:
    target top1 repair = 4 / 6
    top1 bins: {608: 3, 793: 1, 731: 2}

  target_bridge:0.5 and target_bridge:1.0:
    target top1 repair = 3 / 6
    top1 bins: {608: 3, 794: 3}

  bridge_plus_donor:0.2 and bridge_plus_donor:0.5:
    target top1 repair = 3 / 6
    donor-y2 top1 transport = 0 / 6

random_denoise image 19432:
  target_bridge:0.2:
    target top1 repair = 0 / 6
    frequent top1 = 347 while target_y2 = 348

  target_bridge:0.5:
    target top1 repair = 4 / 6

  target_bridge:1.0:
    target top1 repair = 6 / 6

  bridge_plus_donor:0.5:
    target top1 repair = 4 / 6
```

Interpretation:

```text
1. The donor-minus-generated component is causal for donor/local y2 transport:
   removing that projection reduces sorted donor transport from 21/30 to 7/30
   and random donor transport from 4/30 to 0/30.

2. Removing donor-basin projection alone does not repair exact target y2. It
   exposes target-rank improvement but usually leaves the model in a broad
   coordinate-closure basin.

3. Small target bridges improve rank before they repair top1. Exact target
   repair appears when the target bridge is large enough or when donor_delta
   supplies broad coordinate-closure energy and the bridge retargets the basin.

4. The v5 scale sweep shows that target readout directions can be sufficient in
   some sample bases: random image 19432 reaches 6/6 target repairs with
   target_bridge:1.0.

5. The sorted image 12670 base is heterogeneous. The target_y2=608 receiver is
   repairable by target_bridge; the target_y2=793 receiver often falls into
   target-adjacent 794 or another attractor such as 731. This means the next
   relevant competitor is not always the original generated bin or donor bin.
```

Updated mechanism picture:

```text
The y2 emission state has at least two separable pieces:

  coordinate-closure energy:
    enough force to leave the original generated y2 basin and enter the right
    broad vertical region.

  local coordinate winner selection:
    a sharper basin/rank competition among target and nearby coordinate tokens.

Natural donor deltas mostly provide coordinate-closure energy but import their
own donor/local winner bias. Target bridges can retarget the closure, but
target-minus-generated is not always the right final-axis correction once the
state enters a local target-neighborhood basin. For hard cases such as sorted
12670 target_y2=793, the next surgical axis should be target-minus-local-winner
(for example target 793 vs emergent 794), not only target-minus-generated.
```

Representative sample-base policy:

```text
Current anchor bases:
  sorted_denoise / image 12670 / person:
    long-range y2 failure, strong donor-basin transport, heterogeneous target
    behavior (608 repairable, 793 local-neighbor/secondary attractor).

  random_denoise / image 19432 / chair:
    short-range y2 failure, adjacent-coordinate competition (347 vs 348), useful
    for separating donor-object transport from local coordinate-neighborhood
    smoothing.

Next bases should be added only when they represent a new mechanism class:
  - false-negative where visual evidence may exist but language/context guidance
    fails to initiate the object span;
  - duplication onset with repeated semantic anchor before repeated coordinates;
  - duplication onset with coordinate basin repetition before semantic repeat;
  - premature termination after apparently valid local object-span predictions;
  - hard coordinate-local winner failures where target and top competitor are
    adjacent or near-adjacent.
```

Immediate next probe:

```text
Add a local-winner bridge transform:
  target_minus_local_top1 or target_minus_observed_competitor

For sorted 12670 target_y2=793, explicitly test:
  target 793 - generated 415
  target 793 - local neighbor 794
  target 793 - secondary attractor 731

This should decide whether the residual failure is a generic target energy
deficit or a local coordinate-winner asymmetry after the broader y2 basin has
already moved.
```

## 2026-06-25 Local-Winner Bridge Surgery

Extended `y2_natural_donor_transplant.py` again so bridge transforms can use an
explicit coordinate-bin antagonist:

```text
target_bridge:scale
  target_token - generated_token

target_bridge:coord_bin:scale
  target_token - coord_bin_token

bridge_plus_donor:coord_bin:scale
  donor_delta + scale * ||donor_delta|| * unit(target_token - coord_bin_token)
```

This lets the same runner test whether the residual target-neighborhood failure
is caused by a specific local winner, for example target 793 versus observed
local winner 794.

Artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant/v6_local_winner_bridge_12670_sorted_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant/v6_local_winner_bridge_19432_random_gpu1
```

Scope:

```text
sample bases:
  sorted_denoise image 12670
  random_denoise image 19432

layers:
  27 only

site:
  layer_output

sorted transforms:
  donor_delta
  target_bridge:0.2
  target_bridge:794:{0.2,0.5,1.0}
  target_bridge:731:{0.2,0.5,1.0}
  bridge_plus_donor:794:0.2
  bridge_plus_donor:731:0.2

random transforms:
  donor_delta
  target_bridge:0.2
  target_bridge:347:{0.2,0.5,1.0}
  target_bridge:342:{0.2,0.5,1.0}
  bridge_plus_donor:347:0.2
  bridge_plus_donor:342:0.2

rows:
  64 per checkpoint, zero errors
```

Key row-group results:

```text
sorted_denoise image 12670, target_y2=608:
  target_bridge:0.2 (target - generated 415):
    target top1 repair = 3 / 3
    top1 bins = {608: 3}

  target_bridge:794:
    target top1 repair = 0 / 9
    top1 bins include 484 and 609

  target_bridge:731:
    target top1 repair = 0 / 9
    top1 bins include 391, 578, and 609

sorted_denoise image 12670, target_y2=793:
  target_bridge:0.2 (target - generated 415):
    target top1 repair = 1 / 3
    top1 bins = {793: 1, 731: 2}
    mean patched target rank = 8.0

  target_bridge:794:
    target top1 repair = 0 / 9
    top1 bins = {604, 666}
    mean patched target rank remains very poor:
      0.2 scale -> 386.67
      0.5 scale -> 318.67
      1.0 scale -> 311.67

  target_bridge:731:
    target top1 repair = 0 / 9
    top1 bins = {291, 540, 790}
    mean patched target rank improves with scale but stays non-top1:
      0.2 scale -> 120.33
      0.5 scale -> 34.67
      1.0 scale -> 19.0

random_denoise image 19432, target_y2=348:
  target_bridge:0.2 (target - generated):
    target top1 repair = 0 / 6
    top1 bins = {347: 6}
    mean target rank = 1.67

  target_bridge:342:
    target top1 repair rises with scale:
      0.2 -> 0 / 6, top1 {347: 6}
      0.5 -> 1 / 6, top1 {347: 5, 348: 1}
      1.0 -> 5 / 6, top1 {348: 5, 347: 1}

  target_bridge:347:
    target top1 repair = 0 / 18
    top1 bins move past the target to {349, 350}
    mean target rank gets worse with scale:
      0.2 scale -> 7.0
      0.5 scale -> 15.33
      1.0 scale -> 41.83
```

Interpretation:

```text
The naive "local winner bridge" hypothesis is false.

Pairwise target-minus-neighbor directions are not stable target attractors in
the live hidden state. They often behave like local tangent or repulsion
directions on the coordinate readout surface:

  target 348 - source 347 pushes the random state past 348 into 349/350.
  target 793 - source 794 pushes the sorted state down into 604/666.
  target 793 - source 731 pushes into 291/540/790.

This means local coordinate winner selection is not fixed by simply subtracting
the current top competitor row from the target row. The final y2 basin is not a
single linear "target beats competitor" margin problem.
```

Refined mechanism picture:

```text
There are at least three separable axes:

1. Broad closure / leave-generated-basin axis:
   target - generated is often useful because it moves the state from the old
   y2 basin toward the target vertical region.

2. Donor/local-basin import axis:
   donor deltas provide closure energy but import donor-centered local winner
   bias; removing donor-minus-generated reduces donor transport.

3. Coordinate manifold tangent axis:
   adjacent target-minus-neighbor row differences can push along the local
   coordinate surface rather than into the target attractor. The response is
   asymmetric and checkpoint/sample dependent.

The hard target_y2=793 case is therefore not solved by local pairwise readout
bridges. It needs a probe that estimates the local coordinate-basin Jacobian or
learns a small multi-direction correction from several nearby bins, rather than
using one antagonist row.
```

Next direction:

```text
Build a coordinate-neighborhood response atlas for one receiver state:

  for source bins around target and observed top winners:
    apply target_bridge:source_bin:scale over a small scale ladder
    record patched top1, target rank, and response direction

For sorted 12670 target_y2=793:
  source bins: 790..796, 731, 604, 666, generated 415

For random 19432 target_y2=348:
  source bins: 342..350

Then fit/inspect a local response map:
  source_bin -> induced top1_bin
  scale -> target_rank trajectory

This should reveal whether the coordinate readout surface around the live state
has a monotone basin, a fold/overshoot, or a discontinuous attractor switch.
```

## 2026-06-25 Coordinate-Neighborhood Response Atlas

Ran the local response atlas proposed above, using the explicit coordinate-bin
bridge source syntax.

Artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant/v7_coord_response_atlas_12670_sorted_gpu2
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant/v7_coord_response_atlas_19432_random_gpu3
```

Scope:

```text
sorted_denoise image 12670:
  layer = 27
  site = layer_output
  receivers = 2
  donors per receiver = 3
  target_bridge sources:
    generated
    415
    604
    666
    731
    790, 791, 792, 794, 795, 796
  scales:
    0.1, 0.2, 0.5, 1.0
  rows = 274, errors = 0

random_denoise image 19432:
  layer = 27
  site = layer_output
  receivers = 2
  donors per receiver = 3
  target_bridge sources:
    generated
    342, 343, 344, 345, 346, 347, 349, 350
  scales:
    0.1, 0.2, 0.5, 1.0
  rows = 226, errors = 0
```

Sorted response map:

```text
target_y2=608:
  generated/415 source:
    scale 0.2+ -> exact repair 3/3, top1 {608: 3}

  source 790 or 791:
    scale 0.5+ -> exact repair 3/3, top1 {608: 3}

  source 792:
    scale 0.5+ -> target rank 2, top1 {609: 3}

  source 794/795:
    scale 0.5+ -> top1 {609: 3}, target rank around 5-7

  source 796:
    scale 1.0 -> top1 {606: 3}, target rank 6

target_y2=793:
  generated/415 source:
    scale 0.2 -> exact repair 1/3, top1 {793: 1, 731: 2}
    scale 0.5+ -> target rank 2, top1 {794: 3}

  source 604:
    scale 1.0 -> top1 {794: 1, 795: 2}, target rank 5.33

  source 666:
    scale 1.0 -> top1 {794: 3}, target rank 8.67

  source 731:
    scale 1.0 -> top1 {790: 1, 291: 2}, target rank 19.0

  source 790:
    scale 1.0 -> top1 {357: 3}, target rank 83.0

  source 791:
    scale 1.0 -> top1 {357: 3}, target rank 182.67

  source 792:
    scale 1.0 -> top1 {420: 3}, target rank 326.67

  source 794:
    scale 1.0 -> top1 {604: 3}, target rank 311.67

  source 795:
    scale 1.0 -> top1 {604: 3}, target rank 156.33

  source 796:
    scale 1.0 -> top1 {605: 3}, target rank 67.67
```

Random response map:

```text
target_y2=348:
  generated source:
    scale 0.5 -> exact repair 4/6
    scale 1.0 -> exact repair 6/6

  source 342:
    scale 1.0 -> exact repair 5/6, top1 {348: 5, 347: 1}

  source 343:
    scale 0.5+ -> exact repair 6/6

  source 344:
    scale 0.2+ -> exact repair 6/6

  source 345:
    scale 0.2+ -> top1 {349: 6}, target rank stays near 2-3

  source 346:
    scale 0.2+ -> top1 {349: 6}, target rank worsens with scale

  source 347:
    scale 0.2 -> top1 {349: 3, 350: 3}
    scale 0.5+ -> top1 {350: 6}, target rank collapses

  source 349:
    scale 0.5+ -> top1 {346: 6}, target rank collapses

  source 350:
    scale 1.0 -> top1 {347: 6}, target rank 9.0
```

Interpretation:

```text
The random 348 base has a folded local response surface:
  - source bins below the target, especially 343/344, repair the target;
  - source bins just below the target, especially 347, push past target to 349/350;
  - source bins above the target push back below target to 346/347.

This is not a simple monotone coordinate axis. It looks like a curved local
coordinate manifold where pairwise row differences are directional derivatives:
depending on source location and scale, they can step through the target,
overshoot, or reflect to the opposite side.

The sorted 793 base is more discontinuous:
  - generated/415 is the best source but saturates at target rank 2 with top1
    794 for larger scales;
  - local-neighborhood sources 790..796 do not locally correct 793 and instead
    map to distant attractors such as 357, 420, 604, and 605;
  - previously observed attractors 604/666/731 can improve rank but still land
    on 794/795, 790/291, or other off-target winners.

So the hard sorted 793 problem is not only "target energy too low" and not a
single target-vs-local-competitor margin. It is a discontinuous basin routing
problem in the live hidden state.
```

Updated mechanism picture:

```text
Coordinate-token output rows preserve local geometry in static embedding/readout
space, but the live hidden state does not query them with a smooth coordinate
decoder. The same linear row-difference intervention can:
  - repair target exactly,
  - push into the adjacent neighbor,
  - overshoot across the local neighborhood,
  - or jump into a distant attractor.

The behavior depends on the receiver state, source row, scale, checkpoint, and
which broad y2 basin the state is already in.

This suggests the model stores object-coordinate emission as a basin-routing
state rather than as a clean continuous coordinate variable. Prefix denoising
may have improved some broad closure/retargeting directions, but it did not make
local coordinate winner selection globally smooth.
```

Next high-value bridge:

```text
Move from row-difference interventions to state-subspace interventions:

1. For one receiver state, build a response matrix:
     intervention vector -> top1/rank trajectory

2. Decompose intervention vectors into:
     generated-to-target bridge
     donor-delta closure direction
     local coordinate tangent directions
     residual directions

3. Search for a low-dimensional subspace that predicts:
     exact repair
     neighbor overshoot
     distant attractor jump

This should connect coordinate-basin formation with autoregressive onset
dynamics: the same state-subspace that routes y2 may be the state-subspace that
later causes duplicate object-span continuation or premature termination.
```

## 2026-06-25 Coordinate Response Matrix Reducer

Added a CPU-only response-matrix reducer:

```text
src/analysis/prefix_denoising_surgery_probing/coordinate_response_matrix.py
scripts/analysis/run_prefix_denoising_coordinate_response_matrix.py
tests/analysis/test_prefix_denoising_coordinate_response_matrix.py
```

The reducer converts surgery JSONL rows into grouped response cells keyed by:

```text
model_id
image_id
receiver_state_key
target_y2_bin
configured_layer
patch_site
patch_transform
effective_source_coord_bin
bridge_scale
```

It classifies each row as:

```text
exact_target_top1
adjacent_low / adjacent_high
near_low / near_high
overshoot_low / overshoot_high
distant_low / distant_high
donor_y2_top1
stayed_generated
missing_target_or_top1
```

Artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coordinate_response_matrix/v1_v7_atlas_only
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coordinate_response_matrix/v1_v6_v7_combined
```

v7-only matrix scope:

```text
inputs:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant/v7_coord_response_atlas_12670_sorted_gpu2/y2_natural_donor_transplant_rows.jsonl
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant/v7_coord_response_atlas_19432_random_gpu3/y2_natural_donor_transplant_rows.jsonl

input rows = 500
response rows = 492
matrix rows = 148
```

v7-only response class counts:

```text
exact_target_top1 = 83
adjacent_high = 87
adjacent_low = 62
overshoot_high = 20
overshoot_low = 17
distant_low = 180
distant_high = 1
donor_y2_top1 = 39
near_low = 1
stayed_generated = 2
```

Combined v6/v7 matrix scope:

```text
input rows = 628
response rows = 612
matrix rows = 156
```

Combined response class counts:

```text
exact_target_top1 = 93
adjacent_high = 106
adjacent_low = 84
overshoot_high = 36
overshoot_low = 18
distant_low = 214
distant_high = 2
donor_y2_top1 = 55
near_low = 2
stayed_generated = 2
```

Top exact-repair handles from the v7 matrix:

```text
random_denoise image 19432 target 348:
  source 342, scale 1.0:
    exact_target_top1_count = 6 / 6
    top1 = {348: 6}

  source 343, scale 0.5:
    exact_target_top1_count = 6 / 6
    top1 = {348: 6}

  source 343, scale 1.0:
    exact_target_top1_count = 6 / 6
    top1 = {348: 6}

sorted_denoise image 12670 target 608:
  source/generated 415, scale 0.2:
    exact_target_top1_count = 6 / 6
    top1 = {608: 6}

  source/generated 415, scale 0.5:
    exact_target_top1_count = 6 / 6
    top1 = {608: 6}

  source/generated 415, scale 1.0:
    exact_target_top1_count = 6 / 6
    top1 = {608: 6}
```

Top distant-jump handles from the v7 matrix:

```text
sorted_denoise image 12670 target 608:
  source/generated 415, scale 0.1:
    distant_jump_count = 6 / 6
    top1 = {483: 6}
    mean target rank = 9.33

sorted_denoise image 12670 target 793:
  source/generated 415, scale 0.1:
    distant_jump_count = 6 / 6
    top1 = {605: 4, 731: 2}
    mean target rank = 142.33

  source/generated 415, scale 0.2:
    distant_jump_count = 4 / 6
    top1 = {731: 4, 793: 2}
    mean target rank = 8.0

sorted_denoise image 12670 target 608:
  source 604, scale 0.1..1.0:
    distant_jump_count = 3 / 3 at every scale
    top1 = {421: 3}

  source 666, scale 0.2..1.0:
    distant_jump_count = 3 / 3
    top1 = {358: 3}
```

Interpretation:

```text
The response matrix makes the asymmetry sharper:

  random 348 has many exact-repair cells and localized overshoot cells.
  sorted 608 has clean exact-repair cells when source is generated/415.
  sorted 793 is dominated by distant_low cells, even when target rank improves.

So coordinate-basin routing has at least two regimes:

  folded local manifold:
    local source choices map to exact repair, adjacent winner, or overshoot.
    random 348 is the clearest example.

  discontinuous attractor routing:
    many row-difference directions jump to far bins that are not simple local
    neighbors of target, source, or generated. sorted 793 is the clearest
    example.
```

Next mechanism question:

```text
Which internal state subspace predicts folded-local versus discontinuous-jump
responses?

Candidate immediate probe:
  For the same receiver states, collect the perturbation vectors or their
  readout projections for each response matrix cell and fit a tiny linear
  classifier/regressor over:
    exact repair
    adjacent winner
    local overshoot
    distant jump

This does not need training the model. It can be a post-hoc probe over saved
row metadata plus regenerated direction vectors from the same checkpoint.
```

## Coordinate-Response Feature Reducer And Sample-Base Policy

Implementation:

```text
src/analysis/prefix_denoising_surgery_probing/coordinate_response_features.py
scripts/analysis/run_prefix_denoising_coordinate_response_features.py
tests/analysis/test_prefix_denoising_coordinate_response_features.py
```

Artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coordinate_response_features/v1_v7_atlas_only
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coordinate_response_features/v1_v6_v7_combined
```

The reducer is readout-only and CPU-only. It consumes natural-donor surgery
rows, reuses the coordinate-response classes from the response matrix, adds
scalar feature columns, aggregates feature stats by response family/class, and
ranks target-level cases plus image-level sample bases for deeper hidden-state
work. This is a triage tool, not a validation metric.

The v7 atlas-only run covered 492 response rows:

```text
exact: 83
adjacent: 149
overshoot: 37
distant_jump: 181
donor: 39
near: 1
stayed_generated: 2
```

Selected probed sample bases from the v7 atlas-only rows:

```text
image 12670, sorted_denoise:
  targets 608 and 793
  family discontinuous_jump_attractor
  tags include folded_local_basin, mixed_repair_failure, rank_instability,
  coordinate_source_sensitivity, multi_basin_destination, large_coordinate_jump

image 19432, random_denoise:
  target 348
  family discontinuous_jump_attractor
  tags include folded_local_basin, local_boundary_basin, mixed_repair_failure,
  coordinate_source_sensitivity, multi_basin_destination
```

Target-level handles from the v7 atlas-only reducer:

```text
sorted_denoise image 12670 target 793:
  response classes:
    distant_low 107, adjacent_high 17, overshoot_high 4,
    exact_target_top1 2, donor_y2_top1 2, near_low 1, stayed_generated 2
  top1 destinations include:
    291, 295, 296, 357, 358, 415, 416, 419, 420, 421,
    540, 552, 604, 605, 617, 666, 731, 790, 793, 794, 795, 796
  max |top1-target| = 502
  mean patched target rank = 184.85

sorted_denoise image 12670 target 608:
  response classes:
    distant_low 71, exact_target_top1 30, adjacent_high 16,
    donor_y2_top1 10, overshoot_low 7, distant_high 1
  top1 destinations include:
    358, 391, 420, 421, 451, 480, 481, 483, 484,
    578, 605, 606, 608, 609, 617
  max |top1-target| = 250
  mean patched target rank = 31.01

random_denoise image 19432 target 348:
  response classes:
    adjacent_low 62, adjacent_high 54, exact_target_top1 51,
    donor_y2_top1 27, overshoot_high 16, overshoot_low 10, distant_low 2
  top1 destinations:
    340, 346, 347, 348, 349, 350
  max |top1-target| = 8
  mean patched target rank = 5.97
```

Descriptive scalar contrast from v7 atlas-only rows, exact repair versus
distant jump:

```text
target_minus_generated:
  exact mean 82.24, distant_jump mean 300.29
baseline target rank:
  exact mean 189.08, distant_jump mean 526.18
patched target rank:
  exact mean 1.00, distant_jump mean 151.30
bridge_scale:
  exact mean 0.637, distant_jump mean 0.337
donor_delta_donor_vs_target_projection_gap:
  exact mean 6.61, distant_jump mean 31.90
```

Descriptive scalar contrast from v7 atlas-only rows, adjacent versus overshoot:

```text
patched target rank:
  adjacent mean 3.69, overshoot mean 20.32
bridge_scale:
  adjacent mean 0.439, overshoot mean 0.643
patch_transform_direction_norm:
  adjacent mean 0.520, overshoot mean 0.365
```

Interpretation:

```text
The atlas-only scalar probe supports two-stage sample-base selection:

1. Broad rollout registry chooses image bases where the model is under real
   pressure: false negatives, duplication bursts, termination faults, or strong
   sorted/random divergence.

2. Coordinate-response reducer chooses target slices inside those image bases
   where surgery responses expose basin mechanics: exact repair, adjacent
   boundary, local overshoot, or discontinuous jump.
```

The broad val200 sample-base registry already exists at:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/sample_base_registry/v1_denoise_sorted_random_val200
```

Its selected non-sanity image-base queue is:

```text
17899:
  boundary/termination plus false-negative pressure; sorted emits empty while
  random emits many objects.

16228:
  strong duplication burst and false-negative pressure; already has repeated
  person coordinate-collapse evidence.

2685:
  very strong duplication burst and same-image model divergence.

12670:
  false-negative pressure plus duplication burst; now also has sorted
  discontinuous coordinate-attractor evidence for y2 targets 608 and 793.

18380:
  heavy false-negative/empty-pred case in both models.

2157:
  false-negative plus random duplication burst; already used once for FN
  activation-patch state selection.

12639:
  false-negative plus random over-emission/duplication pressure.

19432:
  false-negative plus duplication burst; now also has random folded-local y2
  basin evidence for target 348.

19109:
  high-GT-count false-negative plus cross-model duplication pressure.

9590:
  heavy false-negative/termination-like case in both models.

2299:
  false-negative plus random empty-pred/termination fault and sorted
  duplication pressure.

14439:
  sorted empty-pred/termination fault versus random over-emission.
```

Operational policy for the next round:

```text
Do not spend primary GPU work on normal/well-learned images except for one or
two explicit positive controls. Prefer sample bases that expose at least one of:

  cross-model divergence on the same visual input
  false-negative pressure where guidance/perception can be separated
  duplication or repeated-anchor burst
  premature termination or wrapper-router fault
  coordinate-response basin ambiguity after surgery

For each admitted image base, pick target slices from the reducer/registry that
maximize mechanistic contrast, not metric representativeness.
```

## Broad Image-Coverage State Panel And Underexplored Atlas

Implementation:

```text
src/analysis/prefix_denoising_surgery_probing/object_state_position_subset.py
scripts/analysis/run_prefix_denoising_object_state_position_subset.py
```

The broad object-state selector was rerun in unpatched scouting mode:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/object_state_selection/v5_broad_image_coverage_unpatched
```

Scope:

```text
position rows:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/position_plan/v1_selected_bases_treatment_only/position_rows.jsonl

candidate_count = 5376
selected_count = 66
selected images = 2157, 2299, 2685, 9590, 9772, 12639, 12670,
                  14439, 16228, 17207, 17899, 18380, 19109, 19432
selected models = random_denoise 32, sorted_denoise 34
selected families:
  duplicate_anchor_basin 26
  false_negative_guidance 24
  termination_boundary 13
  matched_control 3
```

Per-image coverage:

```text
2157:  duplicate_anchor_basin 2, false_negative_guidance 2
2299:  duplicate_anchor_basin 2, termination_boundary 2, false_negative_guidance 2
2685:  duplicate_anchor_basin 2, false_negative_guidance 2
9590:  duplicate_anchor_basin 2, termination_boundary 2, false_negative_guidance 2
9772:  termination_boundary 2, matched_control 2
12639: duplicate_anchor_basin 2, termination_boundary 2, false_negative_guidance 2
12670: duplicate_anchor_basin 2, false_negative_guidance 2
14439: duplicate_anchor_basin 2, termination_boundary 1, false_negative_guidance 2
16228: duplicate_anchor_basin 2, termination_boundary 1, false_negative_guidance 2
17207: duplicate_anchor_basin 2, matched_control 1
17899: duplicate_anchor_basin 2, termination_boundary 1, false_negative_guidance 2
18380: duplicate_anchor_basin 2, termination_boundary 1, false_negative_guidance 2
19109: duplicate_anchor_basin 2, false_negative_guidance 2
19432: duplicate_anchor_basin 2, termination_boundary 1, false_negative_guidance 2
```

This panel is a corrective to earlier activation-evidence-biased selections:
it covers all selected image bases and should be the source for the next
sample-base-centered probes.

Exact position-row subsets:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/selected_position_subset/v1_broad_image_coverage_unpatched
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/selected_position_subset/v1_underexplored_termination_fn
```

The underexplored termination/FN subset selects 17 exact position rows:

```text
images = 2299, 9590, 14439, 17899, 18380
models = random_denoise 8, sorted_denoise 9
families = false_negative_guidance 10, termination_boundary 7
positions = pre_x1 10, box_end 7
```

Layerwise atlas artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v7_underexplored_termination_fn_dryrun
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v7_underexplored_termination_fn_shard0_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v7_underexplored_termination_fn_shard1_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v7_underexplored_termination_fn_shard2_gpu2
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v7_underexplored_termination_fn_shard3_gpu3
```

Combined reduction:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas_reduce/v1_underexplored_termination_fn
```

Run scope:

```text
source states = 17
layers = 0, 8, 16, 20, 24, 27
atlas rows = 102
readout_status = ok for all rows
families = false_negative_guidance 60 rows, termination_boundary 42 rows
models = random_denoise 48 rows, sorted_denoise 54 rows
```

Termination-boundary pattern:

```text
layer 0:
  mean target <|im_end|> rank 14625
  mean <|im_end|> prob 7.7e-7

layer 20:
  mean target <|im_end|> rank 3028.9
  mean <|im_end|> prob 2.8e-7
  top1 token is <|endoftext|> for all 7 states

layer 24:
  mean target <|im_end|> rank 77.7
  mean <|im_end|> prob 0.00169
  mean <|object_ref_start|> prob 0.01553

layer 27:
  mean target <|im_end|> rank 1.29
  mean <|im_end|> prob 0.54769
  mean <|object_ref_start|> prob 0.45169
  top1 <|im_end|> in 5/7 states
  top1 <|object_ref_start|> in 2/7 states
```

Final-layer termination rows:

```text
random 2299 person box_end:
  <|im_end|> rank 1, prob 0.53108
  <|object_ref_start|> prob 0.46868

sorted 2299 person box_end:
  <|im_end|> rank 1, prob 0.86674
  <|object_ref_start|> prob 0.13292

random 9590 bowl box_end:
  <|im_end|> rank 1, prob 0.53089
  <|object_ref_start|> prob 0.46851

sorted 9590 bowl box_end:
  <|im_end|> rank 2, prob 0.43749
  <|object_ref_start|> prob 0.56175

sorted 14439 backpack box_end:
  <|im_end|> rank 1, prob 0.49956
  <|object_ref_start|> prob 0.49956

sorted 17899 donut box_end:
  <|im_end|> rank 2, prob 0.46837
  <|object_ref_start|> prob 0.53074

random 18380 wine glass box_end:
  <|im_end|> rank 1, prob 0.49970
  <|object_ref_start|> prob 0.49970
```

Interpretation:

```text
Termination is a late binary router. Through layer 20 the hidden state is not
yet a clean stop/continue decision. Layer 24 starts to expose wrapper mass, but
the continuation token often dominates. Layer 27 collapses into an almost pure
<|im_end|> versus <|object_ref_start|> contest.

Several underexplored termination states are near exact knife-edges:
  sorted 14439: 0.49956 / 0.49956
  random 18380: 0.49970 / 0.49970
  random 2299 and random 9590: about 0.53 / 0.47

These are high-value surgery targets because a tiny late-layer direction should
flip stop/continue behavior, and because the evidence is not restricted to the
old person/backpack pair.
```

False-negative-guidance coordinate pattern:

```text
layer 0:
  mean target coord rank 475.7
  mean coord top1 distance to target 346.6
  mean coord vocab mass 0.1036

layer 20:
  mean target coord rank 471.6
  mean coord top1 distance to target 206.3
  mean coord vocab mass 0.0435

layer 24:
  mean target coord rank 278.9
  mean coord top1 distance to target 126.3
  mean coord vocab mass 0.0683

layer 27:
  mean target coord rank 1.3
  mean coord top1 distance to target 82.0
  mean coord vocab mass 0.9963
```

Final-layer FN-guidance rows:

```text
random 2299 tie pre_x1 target coord_696:
  target rank 2, coord top1 coord_0, distance 696

random 14439 person pre_x1 target coord_124:
  target rank 3, coord top1 coord_0, distance 124

all other selected FN-guidance rows:
  target rank 1 and coord top1 equals target, or tied with target.
```

Interpretation:

```text
For selected FN-guidance states, the model usually does perceive and represent
the next coordinate by the final layer: coord vocab mass rises to about 0.996
and most targets are rank 1. However, the two random-denoise failures show that
language-side guidance is not the only issue. Even when the coord-token basin is
fully active, border-anchor attraction can still win locally:

  random 2299 tie: coord_0 beats coord_696
  random 14439 person: coord_0 beats coord_124

So "false negative" should be decomposed into at least:
  missing/weak object availability before the coordinate slot
  available coordinate plan but fragile binding to the intended object/span
  available coordinate plan but coordinate basin captured by a border anchor
  stop/continue router deciding to terminate or continue at the wrong boundary
```

Caveat:

```text
Coordinate rows can contain exact probability ties among several coord bins.
For example, sorted 2299 person has target coord_414 tied with coord_519 and
coord_479 at the same displayed coord-only probability. Therefore
coord_target_rank=1 can coexist with a different full-vocab top1 token text.
Treat tied target-rank-1 rows as "target participates in the winning basin",
not always as unique target dominance.
```

## 2026-06-25 core sample-base causal patch panel

Scope:

```text
Panel:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/selected_position_subset/v3_sample_base_core_panel

Activation patch shards:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v13_core_coord_pathology_gpu0
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v14_core_coord_controls_non_target_antagonist_gpu1
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v13_core_terminal_samebase_gpu2
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v13_core_terminal_competition_gpu3

Reduction:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch_reduce/v1_core_sample_base_panel

Rows:
  9 source states
  378 activation-patch rows
  layers 24 and 27
  patch sites layer_output, self_attn, mlp
  continuation_steps=6
  alpha grid 0,0.005,0.01,0.02,0.05,0.1,0.2
```

The core panel follows the sample-base recommendation:

```text
Primary abnormal bases:
  image 2299
  image 14439

Secondary terminal-competition base:
  image 17899

Held out from the main pass:
  image 9590
  image 18380
```

The activation-patch runner was updated before this pass:

```text
src/analysis/prefix_denoising_surgery_probing/activation_patch_continuation.py
  adds exact --source-state-keys filtering for future narrow panels
  preserves sample_base_* and object_state_* provenance in new outputs

src/analysis/prefix_denoising_surgery_probing/residual_readout_surgery.py
  coordinate-basin antagonist is now the strongest non-target coord bin
  instead of allowing target==antagonist zero-direction controls

Tests:
  python -m pytest \
    tests/analysis/test_prefix_denoising_activation_patch.py \
    tests/analysis/test_prefix_denoising_residual_surgery.py -q
  result: 18 passed
```

Caveat:

```text
The v13 activation shards were generated before the sample_base_* provenance
fields were added to activation_patch_rows.jsonl, so their row-level case labels
must be recovered by joining source_state_key back to:

  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/selected_position_subset/v3_sample_base_core_panel/selection_summary.json

The v14 coordinate-control shard and later outputs contain the new row-level
sample_base_* fields directly. The reduction artifact above performs the join.
```

Core reduction:

```text
readout_status_counts:
  ok: 378

continuation repair labels:
  terminal_already_stops: 126
  terminal_stop_repair: 34
  coord_tail_coherent: 42
  first_token_only: 100
  first_token_not_repaired: 76
```

Terminal-router result:

```text
17899 sorted donut box_end:
  baseline top1: <|object_ref_start|>
  terminal_stop_repair_count: 34/42
  min repair alpha: 0.005

same-base terminal rows already stopping:
  2299 random person: terminal_already_stops 42/42
  2299 sorted person: terminal_already_stops 42/42
  14439 sorted backpack: terminal_already_stops 42/42
```

Interpretation:

```text
The stop/continue error behaves like a compact late binary router. The
over-continuing 17899 donut state can be turned into a real terminal stop by
tiny readout-direction pressure, and the patched continuation emits only
<|im_end|>. This is much cleaner than coordinate repair: the local decision is
causally exposed and behaviorally coherent.
```

Coordinate border-anchor result:

```text
2299 random tie pre_x1:
  baseline coord top1: coord_0
  target: coord_696
  baseline distance: 696
  baseline rank: 2
  flip_to_target_count: 17/42
  min flip alpha: 0.01
  coord_tail_complete_count: 0

14439 random person pre_x1:
  baseline coord top1: coord_0
  target: coord_124
  baseline distance: 124
  baseline rank: 3
  flip_to_target_count: 2/42
  min flip alpha: 0.05
  coord_tail_complete_count: 0
```

Best patched continuations:

```text
2299 random tie:
  target box: 696,195,684,259
  patched tokens:
    coord_696, coord_195, coord_684, coord_242, box_end, object_ref_start
  exact followup prefix after x1:
    y1 and x2 match; y2 misses by 17

14439 random person:
  target box: 124,114,198,230
  patched tokens:
    coord_124, coord_114, coord_198, coord_231, box_end, object_ref_start
  exact followup prefix after x1:
    y1 and x2 match; y2 misses by 1
```

Interpretation:

```text
Coordinate pathology is not just "the model cannot see the object" and not
just "the coordinate head is weak." A small late patch can move the first
coordinate out of the coord_0 anchor and even snaps the next two box
coordinates onto the intended object span. But the full object-span state is
still incomplete: y2 remains wrong before the box closes.

This separates at least two mechanisms:
  1. local coordinate basin escape, which readout-direction patching can do;
  2. full object-span binding, which needs more than first-slot coordinate
     pressure.
```

Coordinate controls after non-target antagonist fix:

```text
17899 sorted bowl pre_x1:
  baseline exact full box tail
  coord_tail_coherent: 42/42

14439 sorted person pre_x1:
  first coord is correct, but tail is not exact
  emitted 729,40,734,85 instead of 729,38,737,85

2299 sorted person pre_x1:
  first coord participates in a tied winning basin, but tail is not exact
  emitted 414,76,520,354 instead of 414,76,520,336
```

Interpretation:

```text
Even target-first coordinate states are not equivalent. The important hidden
object state is a coordinate bundle, not an independent x1 token. Some states
have the full bundle available (17899 bowl); others have only a fragile or
approximate bundle where the first coordinate is correct but later coordinates
drift.
```

## 2026-06-25 y2 slot follow-up

Scope:

```text
Panel:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/selected_position_subset/v4_core_y2_slot_panel

Activation patch shards:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v15_core_y2_random_gpu0
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v15_core_y2_sorted_gpu1

Reduction:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch_reduce/v2_core_y2_slot_panel

Rows:
  5 source states
  120 activation-patch rows
  layers 24 and 27
  patch sites layer_output, self_attn, mlp
  continuation_steps=3
  alpha grid 0,0.01,0.05,0.1
```

Question:

```text
When the prefix already contains the correct x1/y1/x2, does the model know y2?
This tests whether the failed y2 in patched pre_x1 continuations is a local y2
slot problem or a consequence of an artificial first-coordinate patch that does
not fully reconstruct the hidden object-span state.
```

Reduction:

```text
readout_status_counts:
  ok: 120

repair labels:
  coord_tail_coherent: 59
  first_token_not_repaired: 61
```

Case-level result:

```text
17899 sorted bowl y2:
  alpha0 emits coord_652, box_end, object_ref_start
  exact target; coherent without patch

2299 random tie y2:
  target coord_259 is tied for rank 1 in the readout
  greedy emits coord_242 because of exact/probability-tie behavior
  small patch can emit coord_259, box_end, object_ref_start

14439 random person y2:
  alpha0 emits coord_231 against target coord_230
  small patch can emit coord_230, box_end, object_ref_start

2299 sorted person y2 control:
  alpha0 emits coord_354 against target coord_336
  small patch can emit coord_336, box_end, object_ref_start

14439 sorted person y2 control:
  alpha0 emits coord_83 against target coord_85
  small patch can emit coord_85, box_end, object_ref_start
```

Interpretation:

```text
The model often has a local y2 basin near the target, and the correct y2 can be
recovered under teacher-forced correct x1/y1/x2. Therefore the failed y2 after
pre_x1 patching is probably not a pure visual-perception failure. It is more
consistent with an incomplete hidden object-span trajectory: the artificial x1
patch changes the visible first coordinate and partially snaps y1/x2, but it
does not fully transport the latent state that determines the final coordinate.

Exact or near-exact coordinate ties matter. In 2299 random tie y2, target
coord_259 and wrong coord_242 have equal displayed probability and target rank
1, while greedy continuation emits coord_242. Thus rank-1 evidence alone can
overstate functional correctness: "target is in the winning basin" is different
from "the autoregressive decoder will emit the target."
```

Current mechanism picture after this round:

```text
1. The observed 17899 terminal over-continuation case is compact and late under
   this intervention:
   a small stop-vs-continue direction can repair that state into a clean
   <|im_end|> emission. Broader terminal-error generalization still needs more
   terminal-error cases, because the other terminal rows in this panel were
   already stopping.

2. Coordinate mistakes are bundle/state mistakes:
   the coordinate vocabulary basin is active, and first-coordinate patching can
   escape coord_0, but object-span completion requires a hidden trajectory that
   carries the whole box, not just the next coordinate.

3. False negatives should be split into at least three mechanistic families:
   missing object availability,
   available first coordinate but weak/partial box-bundle binding,
   terminal-router or decode-tie behavior that hides otherwise available local
   evidence.

4. The next most promising direction is natural/causal donor transport:
   transplant hidden deltas from coherent same-slot states into the pathological
   pre_x1 and y2 states, then test whether the full coordinate bundle repairs
   rather than only the first token.
```

## 2026-06-25 natural y2 donor transport

Scope:

```text
Receiver plan:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant_plan/v1_core_y2_receivers

Donor pool:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/position_plan/v1_selected_bases_treatment_only/position_rows.jsonl

Main activation shards:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant/v2_core_y2_random_gpu0
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant/v2_core_y2_sorted_gpu1

Main reduction:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant_reduce/v1_core_y2_donor_transport

Projection-ablation shards:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant/v3_projection_ablation_random_gpu0
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant/v3_projection_ablation_sorted_gpu1

Projection-ablation reduction:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant_reduce/v2_projection_ablation
```

Receiver states:

```text
2299 random tie y2:
  receiver desc=tie, target=259, generated=242

14439 random person y2:
  receiver desc=person, target=230, generated=231

2299 sorted person y2 control:
  receiver desc=person, target=336, generated=354

14439 sorted person y2 control:
  receiver desc=person, target=85, generated=83

17899 sorted bowl y2 coherent control:
  receiver desc=bowl, target=652, generated=652
```

Main transport reduction:

```text
Rows:
  425 total
  425 readout ok

Effect labels:
  baseline_no_patch: 5
  self_noop_stable: 60
  no_clear_transfer: 142
  donor_y2_rank_improved: 134
  donor_y2_basin_transport: 18
  receiver_target_repair: 6
  receiver_target_rank_improved: 8
  target_and_donor_rank_both_improved: 2
  stayed_generated_y2_basin: 50
```

Case-level behavior:

```text
14439 random person y2:
  baseline top1=237, target=230, generated=231
  receiver-target repairs: 4
  donor transports: 1
  target-rank improvements: 15
  exact repair examples come from donor_delta at layer_output or mlp.

2299 random tie y2:
  baseline top1=259, target=259, generated=242
  receiver-target repairs: 0
  donor transports: 1
  interpretation: target is already top in readout, but visible greedy
  continuation can still choose the wrong tied/lower token.

14439 sorted person y2 control:
  baseline top1=83, target=85, generated=83
  receiver-target repairs: 2
  donor transports: 4

17899 sorted bowl coherent control:
  baseline top1=652, target=652, generated=652
  receiver-target repairs: 0
  donor transports: 12
  interpretation: even a coherent target state can be overwritten by a natural
  donor y2 basin.

2299 sorted person y2 control:
  baseline top1=336, target=336, generated=354
  receiver-target repairs: 0
  donor transports: 0
  many donor-rank improvements, but no exact repair under this grid.
```

Projection-ablation reduction:

```text
Rows:
  185 total
  185 readout ok

Effect labels:
  baseline_no_patch: 5
  self_noop_stable: 20
  no_clear_transfer: 64
  donor_y2_rank_improved: 61
  donor_y2_basin_transport: 19
  receiver_target_repair: 4
  receiver_target_rank_improved: 2
  target_and_donor_rank_both_improved: 2
  stayed_generated_y2_basin: 8

Transforms:
  donor_delta: 60
  remove_donor_minus_generated: 40
  target_bridge: 40
  bridge_plus_donor: 40
```

Projection-ablation interpretation:

```text
The strongest positive detail is 14439 random person y2. Removing the
donor-minus-generated projection does not destroy the exact receiver-target
repair. The best row is:

  donor desc=person, donor y2=246
  transform=remove_donor_minus_generated
  layer=27, site=mlp, strength=1.0
  patched top1=coord_230
  target rank=1

This means the useful repair component is not simply the raw donor-y2 axis.
Some same-desc donor hidden deltas appear to carry a more abstract object-span
correction or receiver-compatible bundle component, even after the explicit
donor-minus-generated projection is removed.

The negative/control detail is equally important. In sorted coherent controls,
natural donor states still transport donor y2 basins into the receiver. The
object-span hidden state is causally portable, but the portable content is
identity-specific geometry, not a neutral "be correct" direction.
```

Mechanism update:

```text
1. y2 is not merely a local coordinate-logit choice. Natural donor hidden states
   can move the receiver into the donor's y2 basin, so the late coordinate slot
   has a causally accessible object-specific geometry component.

2. Receiver-target repair is rare and conditional. It appears when donor and
   receiver are compatible enough that the transported component reinforces the
   receiver target instead of importing the donor coordinate.

3. The useful direction can survive removal of a simple donor-minus-generated
   projection in at least one high-value case. This argues for a factorized
   hidden-state picture:

     object identity / semantic route
     coordinate-basin geometry
     receiver-compatible box-bundle correction
     terminal or wrapper routing

   These are not cleanly separable yet, but they are also not collapsed into one
   scalar coordinate-token axis.

4. Coherent states being overwriteable is a strong caution against treating any
   successful donor patch as "visual rescue". A donor patch can repair, rank
   improve, or overwrite, depending on which component dominates.
```

Sample-base selection rule for the next round:

```text
Keep the panel deliberately small and high-signal. Do not spend model-load time
on normal or well-learned images unless they serve as controls for a specific
mechanistic contrast.

Active sample-base families:
  2299:
    rank/tie disagreement and visible y2 wrong-token behavior.

  14439:
    receiver-compatible target repair under same-desc donor transport.

  17899:
    terminal over-continuation plus coherent coordinate state that can be
    overwritten by donor transport.

Next additions should be selected only if they instantiate a missing mechanism:
  train-set analogue of a val failure,
  true false negative with no object span emitted,
  duplicate-onset state with repeated local spatial anchor,
  same-image crowded same-class ambiguity,
  or a prefix-denoising-specific divergence against a pure CE baseline.
```

Current next-step recommendation:

```text
Run a factorized donor/receiver panel over the selected bases:

1. same image, same desc, different object;
2. same image, different desc, nearby y2;
3. cross image, same desc, nearby y2;
4. receiver self-state with controlled coordinate bridge only;
5. donor state with donor-y2 projection removed and target bridge added.

Score exact coordinate emission, target rank, donor rank, wrapper routing, and
full three-token tail coherence. The aim is not broad metric evaluation. The
aim is to isolate which hidden-state component changes when prefix denoising
produces a different object-span behavior from the pure CE baseline.
```

## 2026-06-25 factorized y2 donor/receiver panel

Implementation:

```text
Extended reusable donor-transplant tooling:
  src/analysis/prefix_denoising_surgery_probing/y2_natural_donor_transplant.py

New launch wrapper:
  scripts/analysis/run_prefix_denoising_factorized_y2_donor_transplant.py

New reducer:
  scripts/analysis/run_prefix_denoising_factorized_y2_donor_reduce.py

Focused tests:
  tests/analysis/test_prefix_denoising_y2_natural_donor_transplant.py
```

The implementation adds two pieces that matter for interpretation:

```text
1. Explicit factorized donor lanes:
   same-image same-desc different-object donor
   same-image different-desc nearby-y2 donor
   cross-image same-desc nearby-y2 donor
   cross-image different-desc nearby-y2 donor

2. target_bridge_abs:
   a pure target-vs-generated coordinate bridge with an absolute hidden-state
   norm, so receiver-self controls are nonzero interventions instead of
   zero-delta noops.
```

Pre-launch gate:

```text
Audited dry-run:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_factorized_donor_transplant/v2_audited_dryrun

Plan rows:
  289

Plan audit:
  passed: true
  failures: []
  warnings: []

Plan counts:
  baseline_no_patch: 5
  self_noop: 20
  receiver_target_bridge_abs: 60
  factor_same_image_same_desc_donor: 36
  factor_same_image_diff_desc_near_y2_donor: 48
  factor_cross_image_same_desc_near_y2_donor: 60
  factor_cross_image_diff_desc_near_y2_donor: 60
```

The audit confirms that cross-image lanes are actually cross-image, same-image
lanes are actually same-image, same-desc and different-desc lanes do not leak
into each other, and the same-image same-desc lane excludes the receiver object
itself.

GPU shards:

```text
Random shard:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_factorized_donor_transplant/v3_random_gpu0
  rows: 118
  errors: 0
  plan audit passed: true

Sorted shard:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_factorized_donor_transplant/v3_sorted_gpu1
  rows: 171
  errors: 0
  plan audit passed: true

Combined reduction:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/y2_factorized_donor_transplant_reduce/v1_factorized_panel
  rows: 289
  cases: 5
  lane summaries: 15
```

Overall reduction:

```text
Readout status:
  ok: 289

Effect labels:
  baseline_no_patch: 5
  self_noop_stable: 20
  no_clear_transfer: 133
  donor_y2_rank_improved: 62
  donor_y2_basin_transport: 21
  stayed_generated_y2_basin: 39
  receiver_target_rank_improved: 2
  receiver_target_repair: 3
  target_and_donor_rank_both_improved: 4

Continuation-tail labels:
  receiver_target_y2_box_end: 51
  donor_y2_box_end: 21
  generated_y2_box_end: 17
  other_y2_box_end: 195
  no_continuation: 5
```

Lane-level behavior:

```text
donor_delta lanes:
  cross-image diff-desc nearby-y2:
    donor transports: 4 / 20
    target tails: 4 / 20

  cross-image same-desc nearby-y2:
    donor transports: 5 / 20
    target tails: 2 / 20

  same-image diff-desc nearby-y2:
    donor transports: 6 / 16
    target tails: 1 / 16

  same-image same-desc:
    donor transports: 2 / 12
    receiver-target repairs: 1 / 12
    target tails: 1 / 12

remove_donor_minus_generated lanes:
  cross-image diff-desc nearby-y2:
    donor transports: 0 / 20
    target tails: 4 / 20

  cross-image same-desc nearby-y2:
    donor transports: 0 / 20
    receiver-target repairs: 1 / 20
    target tails: 3 / 20

  same-image diff-desc nearby-y2:
    donor transports: 0 / 16
    target tails: 3 / 16

  same-image same-desc:
    donor transports: 2 / 12
    receiver-target repairs: 1 / 12
    target tails: 1 / 12

target_bridge / target_bridge_abs:
  relative target_bridge rarely creates exact top1 repair.
  absolute receiver target bridge creates 14 / 60 target-y2 + box_end tails,
  but usually does not create exact top1 repair.
```

Case-level highlights:

```text
14439 random person, target y2=230, generated y2=231:
  exact receiver-target repair remains concentrated in same-image same-desc
  donor states. The layer-27 layer_output donor_delta row emits:
    coord_230, box_end

  The same repair survives remove_donor_minus_generated:
    donor desc=person
    donor y2=231
    layer=27
    site=layer_output
    patched top1=230
    patched tail=coord_230, box_end

  Same-image different-desc donor_delta can instead import the chair donor y2:
    donor desc=chair
    donor y2=395
    patched top1=395
    patched tail=coord_395, box_end

2299 random tie, target y2=259, generated y2=242:
  readout repair count is 0 because local target top1 is already active.
  However, 14 rows emit coord_259, box_end under the local continuation probe.
  This reinforces that this case is a rank/tie/continuation discrepancy rather
  than absence of the target y2 basin.

14439 sorted person, target y2=85, generated y2=83:
  donor import dominates. Same-image same-desc, same-image diff-desc, cross-image
  same-desc, and cross-image diff-desc donor_delta rows can all transport donor
  y2 basins.

  The most interesting repair is cross-image same-desc after removing the
  donor-minus-generated projection:
    donor desc=person
    donor y2=210
    layer=24
    site=mlp
    patched top1=85
    patched tail=coord_85, box_end

17899 sorted bowl, target y2=652, generated y2=652:
  the coherent state remains stable under receiver-self and absolute target
  bridge controls. Donor states can still import donor y2, but removing the
  donor-minus-generated projection protects or restores the target tail.

2299 sorted person, target y2=336, generated y2=354:
  no exact target repair appears in this grid. Donor transports occur across
  same-image same-desc and cross-image lanes. The receiver-target absolute
  bridge moves the local readout near the target but typically lands at 335
  rather than exact 336.
```

Mechanism update:

```text
1. Donor-y2 basin transport is not primarily a same-desc phenomenon.
   It appears in same-image different-desc and cross-image different-desc lanes.
   Therefore y2 donor import can be driven by coordinate/box geometry carried in
   the hidden state, not only by semantic-class scaffold compatibility.

2. Same-desc compatibility still matters for receiver-target repair.
   The clearest exact repair remains the 14439 random person same-image
   same-desc donor, and the sorted 14439 exact repair comes from a cross-image
   same-desc donor after donor-y2 projection removal.

3. Removing the donor-minus-generated readout projection suppresses many donor
   transports, especially in different-desc lanes. This supports the idea that
   the raw donor coordinate axis is a major overwrite component.

4. Some useful receiver-compatible correction survives that removal. This is
   the strongest evidence so far that donor hidden state has a component beyond
   a scalar donor-y2 readout direction.

5. Pure target-coordinate bridges are weaker than donor-state transport for
   exact object-span repair. They can help local y2 + box_end tail closure,
   especially in tie/coherent states, but they usually do not reconstruct the
   full object bundle when the receiver is genuinely misbound.
```

Current interpretation:

```text
The y2 slot appears to be controlled by at least three partially separable
components:

  A. local coordinate readout basin
     can be steered by target-vs-generated bridge directions;
     explains tie-like and already-coherent target-tail closures.

  B. donor object/box geometry bundle
     can overwrite the receiver with donor y2 across same/different-desc and
     same/cross-image lanes;
     explains donor_y2_box_end imports.

  C. receiver-compatible correction bundle
     survives donor-y2 projection removal in selected same-desc cases;
     explains rare exact target repairs.

Prefix-denoising does not simply remove exposure bias. It shapes which of these
components dominates in a local object-span state. The sorted checkpoint often
looks more overwriteable by donor geometry, while the random checkpoint's
14439 person case has a fragile but accessible receiver-compatible correction.
```

Next experiment:

```text
For the next round, do not broaden to normal images. Add only one missing
mechanism at a time:

1. A duplicate-onset receiver where the donor has a repeated local spatial
   anchor, to test whether the same donor-geometry bundle causes duplication
   burst onset.

2. A true false-negative / no-object-span receiver, to test whether a same-desc
   donor can create object_ref_start + desc + box_start routing, not just y2.

3. A train-set analogue of 14439 or 2299, to separate unseen generalization from
   learned-sequence fragility.

Use the factorized plan audit as a required launch gate for all three.
```

## Current Operational Panel Lock

After the factorized donor pass, the current preferred sample-base queue is the
emission-strict representative panel:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/representative_sample_base_panel/v3_current_deep_panel_terminal_dense
```

It selects eight microscope cases from 28 candidates:

```text
1. 12670 / sorted_denoise:
   hard negative y2 closure; best bridge rank 444; non-full-box IoU 0.301

2. 19432 / random_denoise:
   recoverable hidden tail binding; bridge rank 1; non-full-box IoU 0.996

3. 16228 / random_denoise:
   same-image tail-binding contrast

4. 2157 / random_denoise:
   false-negative guidance outside the main y2 pair

5. 2157 / sorted_denoise:
   state-entry first-token-only

6. 19109 / random_denoise:
   false-negative guidance outside the main y2 pair

7. 2685 / random_denoise:
   compact same-image tail-binding contrast

8. 2299 / random_denoise:
   dense termination-boundary case
```

Use this queue for the next probe unless a newly discovered route has obviously
higher influence on the final mechanism picture. The first deterministic
execution target remains the `12670/sorted` versus `19432/random` y2 contrast:
trace why `19432` has a recoverable target-near y2 basin while `12670` lacks or
fails to bind the person bottom-boundary reservoir.

## First Post-Panel Tensor-Flow Read

Existing v2 tensor-flow artifacts already cover the first selected contrast:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_tensor_flow/v2_selector_layers23_27_12670_sorted_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_tensor_flow/v2_selector_layers23_27_19432_random_gpu1
```

Scope:

```text
12670 / sorted_denoise:
  plan rows: 2
  group specs: 2
  tensor-flow rows: 120
  errors: 0

19432 / random_denoise:
  plan rows: 2
  group specs: 3
  tensor-flow rows: 180
  errors: 0

layers:
  23, 24, 25, 26, 27

components:
  layer_input_state
  after_attention_state
  layer_output_state
  attention_update
  mlp_update
  layer_delta_update
```

Contrast reduction:

```text
12670 / sorted_denoise, target y2=608:
  route-patched state best rank: 445
  best state: layer 27 layer_output_state, top1=421
  strongest update vectors:
    layer 23 layer_delta_update +7.591
    layer 23 attention_update   +6.213
    layer 27 mlp_update         +4.640

12670 / sorted_denoise, target y2=793:
  route-patched state best rank: 461
  best state: layer 27 layer_output_state, top1=612
  strongest update vectors:
    layer 23 layer_delta_update +10.129
    layer 23 attention_update   +8.697
    layer 26 layer_delta_update +7.381
    layer 26 mlp_update         +6.625

19432 / random_denoise, target y2=348:
  route-patched state best rank: 1
  best states:
    layer 23 layer_output_state, top1=348
    layer 24 layer_input_state,  top1=348
    layer 24 after_attention_state, top1=348
  strongest update vectors:
    layer 27 layer_delta_update +11.428
    layer 27 mlp_update         +9.262
    layer 24 mlp_update         +7.614
    layer 23 attention_update   +7.365
```

Interpretation:

```text
1. 12670/sorted is not a no-signal case. Large attention/layer-delta/MLP
   bridge directions exist, but they do not become a target-y2 readout basin.
   The state moves toward intermediate or competing destinations such as
   421, 448, 612, or 999 while the true bottom targets remain rank 445/461.

2. 19432/random is a route-to-readout conversion success. The same tensor-flow
   family reaches exact target y2=348 at early state positions, and later
   layer-delta/MLP updates carry large bridge direction.

3. The first mechanistic split after sample-base selection is therefore:
   available target-neighborhood route plus late overwrite/calibration
   (`19432/random`) versus available generic height/bridge directions without
   target-specific bottom-boundary basin formation (`12670/sorted`).
```

Next deterministic action:

```text
Do not rerun this tensor-flow panel unless a code bug is found. Spend the next
GPU pass on the missing part: for 12670/sorted, push the visual-origin question
earlier than the forced x1+y1+x2 y2 state; for 19432/random, use it as the
positive reference where target y2 can already become rank 1.
```

## Sample-Base Microscope Policy Update

The next research loop should keep the sample/image base as the primary unit of
deep mechanism work. Broad metrics and val200 sweeps are useful only as sample
selectors and guardrails; they are not the main explanation surface.

Selection rule:

```text
1. Prefer hard contrastive sample bases where a failure and a near-success share
   enough prompt, object, slot, or geometry structure to make causal comparison
   meaningful.

2. Avoid spending GPU/time on normal or well-learned images unless they are a
   required positive control for a specific failure mechanism.

3. Keep a small current microscope queue, but allow dynamic replacement when a
   new route appears to have more influence on the final mechanism picture.

4. For each selected sample base, probe deeper across token positions, visual
   regions, layers, tensor-flow components, and continuation behavior before
   adding more images.
```

Current first-pair interpretation:

```text
12670 / sorted_denoise:
  hard negative. The forced x1/y1/x2 state strongly prefers the emitted y2=415
  over target lower-boundary bins 608/793. This is not a normal failure sample;
  it is a microscope case for why available visual/geometry evidence does not
  become the correct y2 basin.

19432 / random_denoise:
  positive reference. The target y2=348 is already near the local ridge
  (rank 6/7 while generated y2=342/343 is top-1), and earlier tensor-flow probes
  can reach exact target y2. Use it to identify what successful route-to-readout
  conversion looks like.
```

This means the next passes should center on why `12670/sorted` fails to bind or
route bottom-boundary evidence, using `19432/random` as the nearby success
reference, rather than broadening to ordinary images.

## Visual-Origin Patch Bridge

New runner:

```text
src/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_visual_origin_patch.py
scripts/analysis/run_prefix_denoising_anchor_escape_y2_visual_origin_patch.py
tests/analysis/test_prefix_denoising_anchor_escape_y2_visual_origin_patch.py
```

This probe builds target/object visual regions for selected hard y2 sample
bases, then patches decoder-layer residual streams at visual-token positions
before reading the forced x1/y1/x2 y2 decision state. The intervention surface
is visual-token residual patching, not raw image occlusion; interpret it as
causal evidence for these exact image/prefix states.

Artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_visual_origin_patch/v2_12670_sorted_layers0_27_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_visual_origin_patch/v2_19432_random_layers0_27_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_visual_origin_patch/v3_12670_sorted_bottom_amplify_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_visual_origin_patch/v3_19432_random_bottom_amplify_gpu1
```

Scope:

```text
12670 / sorted_denoise:
  plan rows: 2
  visual membership rows: 14
  v2 patch rows: 144
  v3 patch rows: 90
  errors: 0

19432 / random_denoise:
  plan rows: 2
  visual membership rows: 14
  v2 patch rows: 144
  v3 patch rows: 90
  errors: 0

layers:
  0, 8, 16, 20, 23, 24, 25, 26, 27
```

Region counts:

```text
12670 / sorted_denoise, target y2=608:
  target_full=30
  target_bottom_band=10
  target_upper_body=20
  context_ring=100
  generated_box=5
  original_anchor_box=1014
  far_background=884

12670 / sorted_denoise, target y2=793:
  target_full=60
  target_bottom_band=15
  target_upper_body=45
  context_ring=132
  generated_box=10
  original_anchor_box=1014
  far_background=822

19432 / random_denoise, target y2=348:
  target_full=18
  target_bottom_band=3
  target_upper_body=15
  context_ring=103
  generated_box=18
  original_anchor_box=288
  far_background=851
```

Key readout effects:

```text
12670 / sorted_denoise baseline:
  target y2=608, generated y2=415:
    rank=474, top1=415, target-minus-generated logit=-2.6875
  target y2=793, generated y2=415:
    rank=571, top1=415, target-minus-generated logit=-2.4375

12670 / sorted_denoise, strongest rescues:
  y2=793, layer 8, copy target_bottom_band -> target_upper_body:
    rank 571 -> 17, top1 415 -> 999, target logit +2.875
  y2=793, layer 8, zero target_upper_body:
    rank 571 -> 19, top1 415 -> 796, target logit +2.5
  y2=793, layer 8, zero context_ring:
    rank 571 -> 20, top1 415 -> 796, target logit +3.125
  y2=608, layer 8, zero original_anchor_box:
    rank 474 -> 203, top1 415 -> 999, target logit +3.3125

12670 / sorted_denoise, bottom evidence control:
  zero target_bottom_band has negative mean rank effect:
    mean rank delta -10.89, max +7, min -79, mean target logit -0.080
  zero target_full also has negative mean rank effect:
    mean rank delta -26.17, max +6, min -146, mean target logit -0.118

19432 / random_denoise baseline:
  target y2=348, generated y2=343:
    rank=7, top1=343, target-minus-generated logit=-0.375
  target y2=348, generated y2=342:
    rank=6, top1=342, target-minus-generated logit=-0.5

19432 / random_denoise visual-origin effects:
  effects are small around the already-near target ridge. Best v2 rank rescue:
    layer 16, zero original_anchor_box, rank 7 -> 2, top1 343 -> 347,
    target logit +3.625
  copying target_bottom_band -> context_ring is often harmful:
    v3 mean rank delta -18.00, min -198, mean target logit -1.597
```

Mechanism update:

```text
1. 12670/sorted is not a pure visual absence case. The target bottom-band and
   full target regions carry useful evidence: zeroing them does not rescue the
   target and usually makes the target y2 rank worse.

2. The suppressive or misrouting surface appears in target upper-body, context,
   and original-anchor regions. Zeroing target_upper_body or context_ring at
   early layers can uncover a target-near y2 basin around 796 for target y2=793.

3. The strongest new bridge is copy target_bottom_band -> target_upper_body at
   layer 8. It improves y2=793 rank from 571 to 17, which means bottom-boundary
   evidence is present and can be made influential when inserted into the
   upper-body channel that otherwise behaves like a suppressor/competitor.

4. The same intervention is not uniformly helpful. It can worsen the y2=608
   row, and context injection is harmful in 19432/random. Therefore the result
   is not a generic "more bottom features help" story. It is a sample-state and
   route-specific competition between visual evidence carriers.

5. Top-1 sometimes moves to 999 after rescue-like interventions. This suggests
   the rescued direction may partially enter a vertical-extension/canvas-height
   attractor rather than a clean object-boundary basin. The target-rank jump is
   still important, but it is not yet full object-span repair.

6. Compared with 19432/random, 12670/sorted has available bottom-boundary
   evidence but poor route-to-readout conversion. The positive reference has a
   target-near local ridge already available; patching mostly perturbs or
   calibrates that ridge.
```

Next deterministic action:

```text
Do not broaden to ordinary images. For the next microscope pass on 12670/sorted,
trace the layer-8 route that makes target_bottom_band usable when copied into
target_upper_body:

  A. split attention_update vs mlp_update at the same visual-token patch sites;
  B. test whether the layer-8 suppressor is query-side selection, value content,
     or residual-channel calibration;
  C. repeat only on 19432/random as the positive reference to identify which
     route-to-readout conversion is healthy;
  D. then move to one false-negative sample base from the locked panel, not to
     broad val200 sampling.
```

## Layer-8 Visual Component Split

Tooling update:

```text
src/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_visual_origin_patch.py
```

The visual-origin probe now accepts a `components` axis:

```text
layer_input
layer_output
attention_update
mlp_update
```

Default behavior remains `layer_output`, so previous visual-origin runs are
still reproduced by the default CLI. `layer_input` uses a decoder-layer
pre-hook; the other components use forward hooks.

Artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_visual_origin_patch/v4_12670_sorted_component_layers0_16_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_visual_origin_patch/v4_19432_random_component_layers0_16_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_visual_origin_patch/v5_12670_sorted_layer8_layer_input_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_visual_origin_patch/v5_19432_random_layer8_layer_input_gpu1
```

Scope:

```text
v4 12670 / sorted_denoise:
  layers: 0, 8, 16
  components: layer_output, attention_update, mlp_update
  patch rows: 90
  errors: 0

v4 19432 / random_denoise:
  layers: 0, 8, 16
  components: layer_output, attention_update, mlp_update
  patch rows: 90
  errors: 0

v5 12670 / sorted_denoise:
  layer: 8
  components: layer_input, layer_output, attention_update, mlp_update
  patch rows: 40
  errors: 0

v5 19432 / random_denoise:
  layer: 8
  components: layer_input, layer_output, attention_update, mlp_update
  patch rows: 40
  errors: 0
```

12670 / sorted_denoise layer-8 component split:

```text
target y2=793, zero target_upper_body:
  layer_input:      rank 571 -> 17, top1 415 -> 796, target logit +3.375
  layer_output:     rank 571 -> 19, top1 415 -> 796, target logit +2.5
  attention_update: rank 571 -> 518, top1 415 -> 679, target logit +0.9375
  mlp_update:       rank 571 -> 513, top1 415 -> 671, target logit +0.875

target y2=793, copy target_bottom_band -> target_upper_body:
  layer_input:      rank 571 -> 97, top1 415 -> 999, target logit +2.1875
  layer_output:     rank 571 -> 17, top1 415 -> 999, target logit +2.875
  attention_update: rank 571 -> 568, top1 415 -> 428, target logit +0.1875
  mlp_update:       rank 571 -> 567, top1 415 -> 415, target logit +0.25

target y2=793, zero context_ring:
  layer_input:      rank 571 -> 679, top1 415 -> 288, target logit -1.0625
  layer_output:     rank 571 -> 20, top1 415 -> 796, target logit +3.125
  attention_update: rank 571 -> 591, top1 415 -> 492, target logit -0.5625
  mlp_update:       rank 571 -> 553, top1 415 -> 415, target logit +0.4375

target y2=608, zero target_upper_body:
  layer_input:      rank 474 -> 186, top1 415 -> 421, target logit +0.9375
  layer_output:     rank 474 -> 448, top1 415 -> 415, target logit +0.125
  attention_update: rank 474 -> 405, top1 415 -> 415, target logit +0.1875
  mlp_update:       rank 474 -> 462, top1 415 -> 415, target logit +0.0625
```

19432 / random_denoise layer-8 component split:

```text
target y2=348, copy target_bottom_band -> context_ring:
  layer_input:      rank 7 -> 153, top1 343 -> 266, target logit -5.125
  layer_output:     rank 7 -> 205, top1 343 -> 270, target logit -5.125
  attention_update: rank 7 -> 9,   top1 343 -> 343, target logit +0.375
  mlp_update:       rank 7 -> 5,   top1 343 -> 343, target logit -0.25

target y2=348, zero context_ring:
  layer_input:      rank 7 -> 20, top1 343 -> 342, target logit -3.5
  layer_output:     rank 7 -> 14, top1 343 -> 364, target logit -2.625
  attention_update: rank 7 -> 5,  top1 343 -> 347, target logit +0.125
  mlp_update:       rank 7 -> 5,  top1 343 -> 347, target logit -0.75
```

Mechanism update:

```text
1. The strongest `12670/sorted` target_upper_body rescue is already present at
   layer input. Zeroing target_upper_body at layer-8 input exactly reaches rank
   17 for y2=793, matching the full layer-output rescue family. Therefore the
   upper-body suppressor is not created by layer-8 attention or MLP alone; it is
   already resident in the incoming residual stream.

2. Copy target_bottom_band -> target_upper_body is also mostly a layer-input
   phenomenon for y2=793. Input-copy reaches rank 97 and layer-output-copy
   reaches rank 17, while attention_update and mlp_update copies barely move
   the rank. The bottom-boundary reservoir is therefore usable as residual
   content, but its readout conversion still depends on the full downstream
   layer state.

3. Context_ring has the opposite sign at layer input versus layer output for
   y2=793. Zeroing context at layer input hurts badly, but zeroing it at layer
   output rescues to rank 20. This suggests context evidence is useful inside
   layer-8 computation, likely as keys/values or calibration evidence, while
   its post-layer residual representation becomes a downstream suppressor or
   competing basin carrier.

4. The positive `19432/random` reference shows why this is not a generic patch
   artifact. Layer-input and layer-output context perturbations destroy the
   already-near target ridge, while isolated attention/MLP changes are small.
   Healthy route-to-readout conversion appears to depend on preserving the
   context residual scaffold, not simply suppressing it.

5. The current best hypothesis is a two-stage visual binding failure for
   `12670/sorted`: lower-boundary evidence exists, but target upper-body
   residual content already carries a suppressive or wrong-owner basin by layer
   8; context evidence is needed during layer computation but harmful as a
   downstream residual carrier. This is deeper than "visual blindness" and also
   deeper than a single attention-head routing story.
```

Next deterministic action:

```text
Stay sample-base centered. The next new experiment should not add ordinary
images. Choose one of two high-value continuations:

  A. For 12670/sorted, test whether the layer-8 target_upper_body input
     suppressor is already present at earlier layers by running layer_input
     patches at layers 0, 4, 8, 12, 16.

  B. Move to one locked false-negative sample base and use the same
     layer_input/layer_output split to ask whether missing objects are visually
     absent or present but suppressed before object-start routing.

Given the current evidence, A is the cleaner causal bridge; B is the better
coverage bridge toward the user's false-negative question.
```

## Early-Layer Suppressor-Origin Panel

Artifact roots:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_visual_origin_patch/v6_12670_sorted_early_layer_input_output_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/anchor_escape_y2_visual_origin_patch/v6_19432_random_early_layer_input_output_gpu1
```

Scope:

```text
12670 / sorted_denoise:
  layers: 0, 4, 8, 12, 16
  components: layer_input, layer_output
  patch rows: 100
  errors: 0

19432 / random_denoise:
  layers: 0, 4, 8, 12, 16
  components: layer_input, layer_output
  patch rows: 100
  errors: 0
```

12670 / sorted_denoise target y2=793 trajectory:

```text
zero target_upper_body, layer_input:
  L0:  rank 571 -> 162, top1 553, target logit +1.625
  L4:  rank 571 -> 18,  top1 796, target logit +2.125
  L8:  rank 571 -> 17,  top1 796, target logit +3.375
  L12: rank 571 -> 20,  top1 796, target logit +3.125
  L16: rank 571 -> 617, top1 602, target logit -0.8125

zero target_upper_body, layer_output:
  L0:  rank 571 -> 25,  top1 999, target logit +2.25
  L4:  rank 571 -> 23,  top1 796, target logit +3.5625
  L8:  rank 571 -> 19,  top1 796, target logit +2.5
  L12: rank 571 -> 312, top1 552, target logit +1.125
  L16: rank 571 -> 591, top1 597, target logit -0.9375

copy target_bottom_band -> target_upper_body, layer_input:
  L0:  rank 571 -> 333, top1 553, target logit +1.3125
  L4:  rank 571 -> 15,  top1 999, target logit +2.6875
  L8:  rank 571 -> 97,  top1 999, target logit +2.1875
  L12: rank 571 -> 297, top1 999, target logit +1.625
  L16: rank 571 -> 602, top1 600, target logit -0.8125

copy target_bottom_band -> target_upper_body, layer_output:
  L0:  rank 571 -> 396, top1 547, target logit +1.1875
  L4:  rank 571 -> 25,  top1 999, target logit +2.375
  L8:  rank 571 -> 17,  top1 999, target logit +2.875
  L12: rank 571 -> 612, top1 600, target logit -0.9375
  L16: rank 571 -> 592, top1 597, target logit -0.75
```

Context-ring trajectory for the same y2=793 row:

```text
zero context_ring, layer_input:
  L0:  rank 571 -> 548, target logit +1.4375
  L4:  rank 571 -> 352, target logit +2.0
  L8:  rank 571 -> 679, target logit -1.0625
  L12: rank 571 -> 568, target logit +1.1875
  L16: rank 571 -> 621, target logit +0.5

zero context_ring, layer_output:
  L0:  rank 571 -> 598, target logit +0.8125
  L4:  rank 571 -> 13,  target logit +1.8125
  L8:  rank 571 -> 20,  target logit +3.125
  L12: rank 571 -> 699, target logit +0.5625
  L16: rank 571 -> 551, target logit +0.625
```

19432 / random_denoise contrast:

```text
The positive reference has no analogous target_upper_body rescue. For target
y2=348, zero target_upper_body stays near-neutral across layers, while
copy target_bottom_band -> target_upper_body mostly damages the ridge:

  layer_input L4:  rank 7 -> 124, top1 93, target logit -3.5
  layer_output L4: rank 7 -> 95,  top1 93, target logit -3.75
  layer_input L8:  rank 7 -> 48,  top1 93, target logit -3.5
  layer_output L8: rank 7 -> 12,  top1 342, target logit -3.125

copy target_bottom_band -> context_ring is even more destructive:

  layer_input L4:  rank 7 -> 259, top1 171, target logit -5.5
  layer_output L4: rank 7 -> 235, top1 266, target logit -5.375
  layer_input L8:  rank 7 -> 153, top1 266, target logit -5.125
  layer_output L8: rank 7 -> 205, top1 270, target logit -5.125
```

Mechanism update:

```text
1. The 12670/sorted target_upper_body suppressor is already visible at the
   visual/input side. Zeroing target_upper_body at layer input helps at L0 and
   becomes a near-complete rescue by L4. It remains strong through L12, then
   no longer helps by L16. This means the wrong-owner/suppressive basin is not
   a late decoder-only phenomenon.

2. The strongest bottom-band-to-upper-body copy occurs earlier than the first
   layer-8 pass suggested. Copying target_bottom_band into target_upper_body at
   L4 layer input reaches rank 15 for y2=793. That is stronger than the L8
   layer-input copy and comparable to the L8 layer-output copy. The bottom
   evidence reservoir is available early enough to alter the basin before the
   late object-span decision.

3. The suppressor window is phase-specific. L4-L12 are the high-leverage visual
   residual layers for this sample. L16 interventions often reverse or collapse
   toward unrelated bins, suggesting the state has moved from visual residual
   routing into later autoregressive/readout calibration.

4. Context_ring remains dual-role. Removing its post-layer residual at L4/L8
   rescues y2=793, but removing it at layer input does not consistently rescue
   and can hurt badly at L8. This supports a two-role view: context participates
   usefully in local computation but leaves a downstream residual carrier that
   can suppress or misroute the target boundary.

5. The 19432/random reference rejects a generic "target_upper_body/context are
   bad" interpretation. The same perturbations mostly damage a healthy
   target-near ridge. The phenomenon is tied to the hard sample's basin
   ownership and visual residual geometry.
```

Next deterministic action:

```text
The next most valuable step is to move from y2 closure to false-negative
object-start guidance on one locked FN sample base. Reuse the same
layer_input/layer_output split, but probe object_ref_start / description /
box_start routing rather than y2. The working question becomes:

  Is the missing object visually absent, or is its visual residual evidence
  present but suppressed before the autoregressive object-start scaffold forms?

Do not broaden to normal images before this FN bridge.
```

## False-Negative Visual-Scaffold Bridge

Implementation:

```text
src/analysis/prefix_denoising_surgery_probing/fn_visual_scaffold_probe.py
scripts/analysis/run_prefix_denoising_fn_visual_scaffold_probe.py
tests/analysis/test_prefix_denoising_fn_visual_scaffold_probe.py
```

The probe consumes the existing contextual FN guidance plan and splits a paired
natural pre-x1 prefix into four scaffold sites:

```text
pre_object_ref_start: predict <|object_ref_start|>
pre_desc_first:       predict first descriptor token
pre_box_start:        predict <|box_start|>
pre_x1:               predict the target x1 coord token
```

It then applies visual-token residual patches over the target and context
regions at `layer_input` and `layer_output`. The second pass adds derived
non-overlap regions so target/generated overlap does not confound the
interpretation:

```text
target_unique
generated_unique
target_generated_overlap
context_unique
context_generated_overlap
```

### 2157 / rank32 / wine glass

Artifact roots:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_scaffold_probe/v1_2157_wineglass_rank32_random_failed_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_scaffold_probe/v1_2157_wineglass_rank32_sorted_paired_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_scaffold_probe/v2_2157_wineglass_rank32_random_failed_derived_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_scaffold_probe/v2_2157_wineglass_rank32_sorted_paired_derived_gpu1
```

Context:

```text
failed model: random_denoise
paired context/model: sorted_denoise
target desc: wine glass
target bbox bins: [72, 40, 178, 377]
paired generated bbox bins: [72, 121, 178, 404]
```

Scaffold baseline:

```text
random_denoise on paired sorted context:
  pre_object_ref_start: rank 1, top1 <|object_ref_start|>
  pre_desc_first:       rank 1, top1 wine
  pre_box_start:        rank 1, top1 <|box_start|>
  pre_x1 target 72:     rank 27, coord top1 0, top1 <|coord_0|>

sorted_denoise on own context:
  pre_object_ref_start: rank 1, top1 <|object_ref_start|>
  pre_desc_first:       rank 1, top1 wine
  pre_box_start:        rank 1, top1 <|box_start|>
  pre_x1 target 72:     coord rank 1, coord top1 72
```

This rejects the simple explanation "the failed model cannot start the object
span" for this sample. The object-start, descriptor, and box-start scaffold are
all locally healthy; the failure is the conversion from a valid object scaffold
into the first coordinate anchor.

Derived visual membership for the pre-x1 site:

```text
target_full:               36 visual tokens
target_unique:              8
target_generated_overlap:  28
generated_unique:           4
context_ring:              96
context_unique:            92
context_generated_overlap:  4
```

Random failed model, pre-x1, selected layer_input effects:

```text
baseline: target x1=72 rank 27, coord top1 0, target logit 19.5

zero target_full @ L8:
  rank 27 -> 559, target logit -5.5625

zero target_generated_overlap @ L8:
  rank 27 -> 243, target logit -3.0

zero target_unique @ L8:
  rank 27 -> 33, target logit +0.0

zero context_ring @ L0:
  rank 27 -> 1, target logit +2.125, top1 token <|coord_72|>

zero context_unique @ L0:
  rank 27 -> 1, target logit +2.25, top1 token <|coord_72|>

zero context_generated_overlap @ L0:
  rank 27 -> 18, target logit +0.0, top1 remains <|coord_0|>

zero context_ring @ L16:
  rank 27 -> 1, target logit +1.375, coord top1 72

zero context_unique @ L16:
  rank 27 -> 1, target logit +1.25, coord top1 72
```

Sorted paired model, same site, selected layer_input effects:

```text
baseline: target x1=72 coord rank 1, coord top1 72

zero target_full @ L8:
  rank 1 -> 598, target logit -5.6875

zero target_generated_overlap @ L8:
  rank 1 -> 503, target logit -6.1875

zero target_unique @ L8:
  rank 1 -> 2, target logit -0.5

zero context_unique @ L0:
  rank 1 -> 4, target logit +1.5

zero context_unique @ L16:
  rank 1 -> 5, target logit +1.375
```

Interpretation:

```text
1. The missing wine glass is not visually absent. Removing target visual tokens
   strongly damages the target x1 in both the failed and paired models.

2. The positive x1 evidence is concentrated in the target/generated overlap
   core, not the small target_unique strip. This is a geometry-local evidence
   reservoir shared by the GT target and the paired generated box.

3. The failed random model is specifically held in a border/early-anchor basin.
   Removing broad context_unique visual tokens moves target x1 from rank 27 to
   rank 1 at L0 and L16. Removing only the 4-token context/generated overlap
   does not explain the rescue.

4. The paired sorted model uses the same target/generated overlap evidence but
   is not trapped by the context carrier. Context removal perturbs it, but does
   not create the same dramatic repair because the exact x1 anchor is already
   internally available.

This is the cleanest current evidence for a false-negative subtype where the
model sees the object and can form the object span, but context-side visual
residuals suppress or misroute the first coordinate anchor before rollout.
```

### 19109 / rank37 / person

Artifact roots:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_scaffold_probe/v3_19109_person_rank37_random_failed_derived_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_scaffold_probe/v3_19109_person_rank37_sorted_paired_derived_gpu1
```

Context:

```text
failed model: random_denoise
paired context/model: sorted_denoise
target desc: person
target bbox bins: [572, 531, 610, 658]
paired generated bbox bins: [584, 527, 606, 657]
```

Derived visual membership:

```text
target_full:               6 visual tokens
target_unique:             3
target_generated_overlap:  3
generated_unique:          0
context_ring:             75
context_unique:           75
context_generated_overlap: 0
```

Pre-x1 baseline and selected effects:

```text
random_denoise failed:
  baseline target x1=572: rank 22, coord top1 0
  zero target_full @ L8:              rank 22 -> 300, target logit -0.75
  zero target_generated_overlap @ L8: rank 22 -> 87,  target logit -0.25
  zero target_unique @ L8:            rank 22 -> 67,  target logit -0.375
  zero context_unique @ L0:           rank 22 -> 98,  target logit +0.0
  zero context_unique @ L16:          rank 22 -> 27,  target logit +0.25

sorted_denoise paired:
  baseline target x1=572: rank 39, coord top1 0
  zero target_full @ L8:              rank 39 -> 349, target logit -0.9375
  zero target_generated_overlap @ L8: rank 39 -> 225, target logit -0.375
  zero target_unique @ L8:            rank 39 -> 142, target logit -0.25
  zero context_unique @ L0:           rank 39 -> 232, target logit -0.4375
  zero context_unique @ L16:          rank 39 -> 10,  target logit +0.1875
```

Interpretation:

```text
19109/rank37 does not replicate the 2157 context-suppression rescue. The exact
GT x1 is weak in both models under the paired context, even though the paired
model matched the object in rollout. The target covers only six visual tokens,
and removing those tokens damages the x1 readout, but removing context does not
repair the border basin.

This is a second false-negative subtype: weak or tiny-target anchor evidence,
plus exact-GT-vs-generated-anchor mismatch. It warns that "paired model matched
the object" is not always a clean positive control for exact GT coordinate
readout.
```

Mechanism update:

```text
The FN bridge now separates at least two sample-base mechanisms:

  A. Visual-present / context-suppressed anchor conversion.
     Example: 2157/rank32 wine glass. Object scaffold is healthy; target visual
     evidence is causally present; broad context_unique visual residuals pull
     the x1 site into a border/early-anchor basin.

  B. Weak tiny-target / exact-anchor mismatch.
     Example: 19109/rank37 person. Target visual evidence exists but is sparse;
     both models prefer border/nearby generated anchors for the exact x1; broad
     context removal does not rescue.

This supports a sample-base research strategy: do not average these mechanisms
together too early. Pick high-value images by failure role, target token
coverage, paired-context exact-rank, and whether context removal has a causal
effect.
```

Next deterministic action:

```text
Keep the next round sample-base centered. Recommended branches:

1. Replicate subtype A on another non-border FN where the paired model has exact
   target x1 rank <= 5 under its own context, then test whether context_unique
   removal again rescues the failed model.

2. For subtype A, move from zeroing to transport/copy surgery:
   copy context_unique -> target_generated_overlap and
   copy target_generated_overlap -> context_unique.
   This asks whether the context carrier is merely suppressive when removed or
   whether it actively encodes a competing anchor that can be transplanted.

3. For subtype B, inspect visual-token saliency/attention for tiny targets and
   compare exact GT x1 versus paired generated x1. Do not treat paired rollout
   matching as exact-coordinate evidence.
```

## FN Context-Target Transport Surgery

Candidate selection from the existing contextual FN guidance rows:

```text
source:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_guidance_probe/v4_contextual_all_selected_fn_layers0_8_16_20_23_24_27_gpu0/contextual_fn_guidance_probe_rows.jsonl

filter:
  readout_source = forward_logits
  prefix_variant = paired_context_natural_gt_x1
```

Only one non-border sample in the current contextual FN panel has the desired
subtype-A signature:

```text
2157 / rank32 / wine glass:
  failed random_denoise exact x1 rank: 21
  paired sorted_denoise exact x1 rank: 1
  failed top coord: 0
  paired top coord: 72

19109 / rank38 / person is exact-rank-positive, but target x1 is 0; it is a
border-anchor control rather than a useful non-border subtype-A replication.
```

Transport artifact roots:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_scaffold_probe/v4_2157_wineglass_rank32_random_failed_transport_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_scaffold_probe/v4_2157_wineglass_rank32_sorted_paired_transport_gpu1
```

Transport interventions:

```text
zero:context_unique
zero:target_generated_overlap
copy:context_unique->target_generated_overlap
copy:target_generated_overlap->context_unique
```

### Random Failed Model

Baseline:

```text
target x1=72, rank 27, coord top1 0, target logit 19.5
```

Layer-input selected effects:

```text
L0:
  zero context_unique:
    rank 27 -> 1, top1 token <|coord_72|>, target logit +2.25
  zero target_generated_overlap:
    rank 27 -> 249, target logit -3.25
  copy context_unique -> target_generated_overlap:
    rank 27 -> 216, target logit -1.875
  copy target_generated_overlap -> context_unique:
    rank 27 -> 38, target logit +0.0

L8:
  zero context_unique:
    rank 27 -> 60, target logit -0.125
  zero target_generated_overlap:
    rank 27 -> 243, target logit -3.0
  copy context_unique -> target_generated_overlap:
    rank 27 -> 243, target logit -3.25
  copy target_generated_overlap -> context_unique:
    rank 27 -> 123, target logit -1.125

L16:
  zero context_unique:
    rank 27 -> 1, coord top1 72, target logit +1.25
  zero target_generated_overlap:
    rank 27 -> 216, target logit -2.25
  copy context_unique -> target_generated_overlap:
    rank 27 -> 206, target logit -1.5
  copy target_generated_overlap -> context_unique:
    rank 27 -> 2, coord top1 still 0, target logit +1.125
```

Layer-output selected effects:

```text
L0:
  zero context_unique:
    rank 27 -> 2, target logit +2.125
  copy context_unique -> target_generated_overlap:
    rank 27 -> 216, target logit -1.875

L16:
  zero context_unique:
    rank 27 -> 3, target logit +0.875
  copy target_generated_overlap -> context_unique:
    rank 27 -> 31, coord top1 92, target logit +2.25
```

### Sorted Paired Model

Baseline:

```text
target x1=72, rank 1, coord top1 72, target logit 19.125
```

Layer-input selected effects:

```text
L0:
  zero context_unique:
    rank 1 -> 4, coord top1 84, target logit +1.5
  zero target_generated_overlap:
    rank 1 -> 464, target logit -4.875
  copy context_unique -> target_generated_overlap:
    rank 1 -> 391, target logit -4.3125
  copy target_generated_overlap -> context_unique:
    rank 1 -> 10, target logit -0.125

L8:
  zero context_unique:
    rank 1 -> 54, target logit -0.875
  zero target_generated_overlap:
    rank 1 -> 503, target logit -6.1875
  copy context_unique -> target_generated_overlap:
    rank 1 -> 456, target logit -5.125
  copy target_generated_overlap -> context_unique:
    rank 1 -> 212, target logit -1.625

L16:
  zero context_unique:
    rank 1 -> 5, coord top1 84, target logit +1.375
  zero target_generated_overlap:
    rank 1 -> 466, target logit -5.8125
  copy context_unique -> target_generated_overlap:
    rank 1 -> 462, target logit -5.125
  copy target_generated_overlap -> context_unique:
    rank 1 -> 10, coord top1 78, target logit +1.375
```

Interpretation:

```text
1. Context_unique is not neutral background. Copying its mean activation into
   the target_generated_overlap evidence core is strongly destructive in both
   the failed and paired models. It behaves much closer to zeroing or corrupting
   the positive target core than to a harmless region swap.

2. Target_generated_overlap is a genuine positive evidence reservoir. Zeroing
   it badly damages exact x1 in both models, and context_unique -> target-core
   transport also collapses the target rank.

3. Removing the broad context carrier and overwriting it with target evidence
   are not equivalent. In the failed model, zeroing context_unique gives the
   cleanest rescue at L0 and L16. Copying target_generated_overlap into
   context_unique gives only a partial late rescue at L16 layer input
   (rank 27 -> 2, but coord top1 remains 0) and can still damage the state at
   L4/L8.

4. This supports an active-carrier view: the context_unique residual carries
   information that can suppress or redirect the coordinate anchor, but the
   harmful behavior is not simply a missing target vector in that region. The
   target coordinate basin appears to depend on removing/neutralizing a broad
   context-side competitor while preserving the localized target/generated
   overlap evidence core.

5. The paired model provides the control: target core corruption is destructive
   even when the model is healthy, while context removal perturbs but does not
   create the failed model's dramatic repair pattern because the paired state
   already owns the correct x1 basin.
```

Next deterministic action:

```text
The best next subtype-A probe is not a broader FN metric sweep. It is a
same-sample attention/logit-flow bridge:

  - Track attention into target_generated_overlap versus context_unique at the
    pre_x1 site across layers.
  - Compare failed random_denoise baseline, zero context_unique rescue, and
    context_unique -> target_generated_overlap corruption.
  - Ask whether the border/coord_0 basin is born from attention allocation,
    residual carrier geometry, or LM-head readout calibration after the visual
    carrier has already entered the language stream.

For subtype replication, the current contextual FN panel has no second clean
non-border exact-positive candidate. Additional candidates should be selected
from a larger panel by requiring paired exact x1 rank <= 5 and failed exact x1
rank >> 1 before running surgery.
```

## Attention-Flow Feasibility Smoke

Smoke scope:

```text
sample: 2157 / rank32 / wine glass / pre_x1
models:
  random_denoise failed on paired sorted context
  sorted_denoise paired on own context
attn_implementation: eager
query index: prefix_hidden_index = 1449
attention tensor shape: (1, 16, 1450, 1450)
layers inspected: 0, 4, 8, 12, 16, 27
```

This was a feasibility smoke, not yet a full artifactized probe. It confirms
that `output_attentions=True` is available and that attention mass into derived
visual regions can be computed directly for the next bridge.

Failed random_denoise baseline attention mass, averaged over heads:

```text
layer 0:
  target_generated_overlap: 0.000159
  target_unique:            0.000125
  context_unique:           0.002693
  context_generated_overlap:0.0000247

layer 8:
  target_generated_overlap: 0.001021
  target_unique:            0.000343
  context_unique:           0.008136
  context_generated_overlap:0.000230

layer 12:
  target_generated_overlap: 0.003226
  target_unique:            0.001379
  context_unique:           0.014476
  context_generated_overlap:0.000523

layer 16:
  target_generated_overlap: 0.057500
  target_unique:            0.010504
  context_unique:           0.069690
  context_generated_overlap:0.001492
```

Paired sorted_denoise baseline attention mass, averaged over heads:

```text
layer 0:
  target_generated_overlap: 0.000169
  target_unique:            0.000130
  context_unique:           0.002857
  context_generated_overlap:0.0000242

layer 8:
  target_generated_overlap: 0.001086
  target_unique:            0.000471
  context_unique:           0.007949
  context_generated_overlap:0.000150

layer 12:
  target_generated_overlap: 0.004887
  target_unique:            0.002067
  context_unique:           0.023111
  context_generated_overlap:0.000941

layer 16:
  target_generated_overlap: 0.102286
  target_unique:            0.001012
  context_unique:           0.045366
  context_generated_overlap:0.000262
```

Preliminary interpretation:

```text
Attention extraction is practical and likely high-value. By L16, the paired
model strongly concentrates query attention on the target/generated overlap
core, while the failed model keeps comparable or larger mass on context_unique.
This matches the surgery picture: target_generated_overlap is the positive
anchor evidence reservoir, and context_unique is the broad competing carrier.

The next implementation should artifactize this into an attention-flow probe
with baseline and patched conditions:

  baseline failed random_denoise
  zero context_unique rescue
  context_unique -> target_generated_overlap corruption
  paired sorted_denoise baseline control

The key question becomes whether the border/coord_0 basin is visible as
attention allocation before it appears in the final coordinate logits, or
whether the attention pattern is downstream of an already-formed residual
carrier geometry.
```

## Artifactized FN Attention-Flow Panel

Implementation:

```text
src/analysis/prefix_denoising_surgery_probing/fn_visual_attention_flow_probe.py
scripts/analysis/run_prefix_denoising_fn_visual_attention_flow_probe.py
tests/analysis/test_prefix_denoising_fn_visual_attention_flow_probe.py
```

Final artifact roots:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_attention_flow_probe/v2_2157_random_failed_full_transport_attention_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_attention_flow_probe/v2_2157_sorted_paired_full_transport_attention_gpu1
```

Scope:

```text
sample: 2157 / rank32 / wine glass / pre_x1
attention layers: 0, 4, 8, 12, 16, 27
patch layers: 0, 16
component: layer_input
regions:
  target_generated_overlap
  target_unique
  context_unique
  context_generated_overlap
  target_full
  context_ring
conditions per model: 7
attention rows per model: 252
errors: 0
```

Conditions:

```text
baseline
L0:layer_input:zero_context_unique
L16:layer_input:zero_context_unique
L0:layer_input:copy_context_unique_to_target_generated_overlap
L16:layer_input:copy_context_unique_to_target_generated_overlap
L0:layer_input:copy_target_generated_overlap_to_context_unique
L16:layer_input:copy_target_generated_overlap_to_context_unique
```

### Random failed model

Condition readout:

```text
baseline:
  rank 27, coord top1 0, top1 <|coord_0|>, target logit 19.5

L0 zero context_unique:
  rank 1, coord top1 0, top1 <|coord_72|>, target logit 21.75

L16 zero context_unique:
  rank 1, coord top1 72, top1 <|coord_72|>, target logit 20.75

L0 copy context_unique -> target_generated_overlap:
  rank 216, coord top1 0, top1 <|coord_0|>, target logit 17.625

L16 copy context_unique -> target_generated_overlap:
  rank 206, coord top1 0, top1 <|coord_0|>, target logit 18.0

L0 copy target_generated_overlap -> context_unique:
  rank 38, coord top1 0, top1 <|coord_0|>, target logit 19.5

L16 copy target_generated_overlap -> context_unique:
  rank 2, coord top1 0, top1 <|coord_0|>, target logit 20.625
```

L16 attention mass into target/core versus context/carrier:

```text
baseline:
  target_generated_overlap: 0.057500
  context_unique:           0.069690
  target share:             0.452
  target - context:        -0.012190

L0 zero context_unique:
  target_generated_overlap: 0.092535
  context_unique:           0.024189
  target share:             0.793
  target - context:         0.068346

L16 zero context_unique:
  target_generated_overlap: 0.070119
  context_unique:           0.000151
  target share:             0.998
  target - context:         0.069968

L0 copy context_unique -> target_generated_overlap:
  target_generated_overlap: 0.024156
  context_unique:           0.083015
  target share:             0.225
  target - context:        -0.058858

L16 copy context_unique -> target_generated_overlap:
  target_generated_overlap: 0.083092
  context_unique:           0.064035
  target share:             0.565
  target - context:         0.019057

L0 copy target_generated_overlap -> context_unique:
  target_generated_overlap: 0.050968
  context_unique:           0.123109
  target share:             0.293
  target - context:        -0.072141

L16 copy target_generated_overlap -> context_unique:
  target_generated_overlap: 0.029931
  context_unique:           0.191864
  target share:             0.135
  target - context:        -0.161933
```

### Sorted paired model

Condition readout:

```text
baseline:
  rank 1, coord top1 72, top1 <|coord_84|>, target logit 19.125

L0 zero context_unique:
  rank 4, coord top1 84, top1 <|coord_84|>, target logit 20.625

L16 zero context_unique:
  rank 5, coord top1 84, top1 <|coord_84|>, target logit 20.5

L0 copy context_unique -> target_generated_overlap:
  rank 391, coord top1 0, top1 <|coord_0|>, target logit 14.8125

L16 copy context_unique -> target_generated_overlap:
  rank 462, coord top1 0, top1 <|coord_0|>, target logit 14.0

L0 copy target_generated_overlap -> context_unique:
  rank 10, coord top1 0, top1 <|coord_0|>, target logit 19.0

L16 copy target_generated_overlap -> context_unique:
  rank 10, coord top1 78, top1 <|coord_78|>, target logit 20.5
```

L16 attention mass into target/core versus context/carrier:

```text
baseline:
  target_generated_overlap: 0.102286
  context_unique:           0.045366
  target share:             0.693
  target - context:         0.056920

L0 zero context_unique:
  target_generated_overlap: 0.143668
  context_unique:           0.012309
  target share:             0.921
  target - context:         0.131359

L16 zero context_unique:
  target_generated_overlap: 0.116212
  context_unique:           0.000055
  target share:             1.000
  target - context:         0.116158

L0 copy context_unique -> target_generated_overlap:
  target_generated_overlap: 0.040940
  context_unique:           0.059849
  target share:             0.406
  target - context:        -0.018909

L16 copy context_unique -> target_generated_overlap:
  target_generated_overlap: 0.024645
  context_unique:           0.061140
  target share:             0.287
  target - context:        -0.036495

L0 copy target_generated_overlap -> context_unique:
  target_generated_overlap: 0.099392
  context_unique:           0.091626
  target share:             0.520
  target - context:         0.007766

L16 copy target_generated_overlap -> context_unique:
  target_generated_overlap: 0.057101
  context_unique:           0.158128
  target share:             0.265
  target - context:        -0.101027
```

Head-level L16 clue:

```text
The competition is concentrated in a small recurring set of heads.

Failed baseline:
  target_generated_overlap top heads:
    h8=0.288873, h13=0.209161, h12=0.187104
  context_unique top heads:
    h7=0.209063, h13=0.207696, h9=0.204919, h8=0.196952

Paired baseline:
  target_generated_overlap top heads:
    h8=0.571306, h13=0.381191, h12=0.367594
  context_unique top heads:
    h12=0.224682, h7=0.121025, h6=0.115967

Failed L16 zero context_unique:
  target_generated_overlap top heads:
    h8=0.359534, h13=0.263998, h12=0.216856
  context_unique is effectively erased:
    max head h3=0.001437

Failed L16 target_generated_overlap -> context_unique:
  context_unique top heads:
    h8=0.698242, h13=0.684784, h12=0.548096
  target_generated_overlap top heads:
    h8=0.108077, h12=0.098218, h13=0.083221
```

Mechanism update:

```text
The attention bridge supports the active-carrier picture.

1. The failed baseline does not merely lack attention to the target core. It
   attends to target_generated_overlap, but context_unique remains equally or
   more competitive at the decisive L16 site. This matches the final x1 basin:
   target rank 27, coord top1 0.

2. Successful zero-context rescue is visible in attention before the final
   logit readout. Zeroing context_unique shifts L16 attention mass toward the
   target core and away from the context carrier:
     failed target share 0.452 -> 0.998 for L16 zero context_unique.
   This says the repair is not only LM-head calibration; it changes the
   attention allocation state at the same layer where coordinate ownership is
   decided.

3. Context-to-target corruption reverses the effect. Copying context_unique
   into target_generated_overlap collapses exact x1 in both models and yields
   a low target share / high context mass pattern. The paired model is a strong
   control: even when baseline ownership is healthy, corrupting the target core
   with context vectors sends it to the border basin.

4. Target-to-context copy explains why transport was only a partial late rescue.
   It can improve target logit/rank, but it also makes context_unique a massive
   attention attractor. In the failed model at L16, target->context copy leaves
   target rank 2 but coord top1 still 0; attention mass is dominated by
   context_unique (0.191864) rather than target_generated_overlap (0.029931).

5. A few recurring heads appear to mediate the basin competition, especially
   heads 8, 12, and 13. In paired baseline these heads concentrate strongly on
   target_generated_overlap; in failed/transport-corrupted states they also
   load context_unique heavily. This is a promising target for a head-level
   intervention or attribution probe.
```

Next deterministic action:

```text
Move from region-level attention mass to head-level causal surgery on the
same 2157 subtype-A state:

  - Intervene on heads 8, 12, 13 at L16.
  - Compare suppressing attention to context_unique versus preserving or
    boosting attention to target_generated_overlap.
  - Keep paired sorted_denoise as the control to avoid mistaking generic
    attention damage for causal rescue.

This is now a concrete path toward the inner mechanism of the false-negative
anchor basin rather than another surface metric.
```

## 2026-06-25 Addendum: L16 Head-Output Route Surgery For 2157 Wine Glass

Question:

```text
Do the L16 heads that carry target/context attention mass merely correlate with
the x1 basin, or do their query-token outputs mediate the repaired coordinate
emission after context_unique is neutralized?
```

New committed probe surface:

```text
src/analysis/prefix_denoising_surgery_probing/fn_visual_head_output_probe.py
scripts/analysis/run_prefix_denoising_fn_visual_head_output_probe.py
tests/analysis/test_prefix_denoising_fn_visual_head_output_probe.py
```

The probe patches the concatenated attention-head input to `self_attn.o_proj`
at the pre-x1 query token. It supports head-only dose response and a combined
visual-region + head-output condition, so the same run can ask whether a visual
repair still works when a selected head group is removed.

Artifact roots:

```text
Head-removal panel:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_head_output_probe/v1_2157_random_failed_l16_heads_gpu0
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_head_output_probe/v1_2157_sorted_paired_l16_heads_gpu1

Head dose-response panel:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_head_output_probe/v2_2157_random_failed_l16_head_dose_gpu0
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_head_output_probe/v2_2157_sorted_paired_l16_head_dose_gpu1

Combined context_unique zero + head-removal panel:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_head_output_probe/v3_2157_random_failed_l16_context_head_combo_gpu0
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_head_output_probe/v3_2157_sorted_paired_l16_context_head_combo_gpu1
```

Verification:

```text
pytest tests/analysis/test_prefix_denoising_fn_visual_head_output_probe.py -q
  -> 6 passed

python -m py_compile \
  src/analysis/prefix_denoising_surgery_probing/fn_visual_head_output_probe.py \
  scripts/analysis/run_prefix_denoising_fn_visual_head_output_probe.py
  -> passed

All six GPU panels above:
  error_count=0
  training_ran=false
  readout_only=true
```

Head-only removal result:

```text
random_denoise failed state, baseline:
  target x1=72 rank 27, coord top1 0, target-minus-top1 logit -3.125

Remove L16 h8:
  rank 27 -> 306, top1 stays 0, margin -3.125 -> -4.5

Remove L16 h12:
  rank 27 -> 3, top1 stays 0, margin -3.125 -> -1.75

Remove L16 h13:
  rank 27 -> 58, top1 stays 0, margin -3.125 -> -3.875

Remove L16 h0,h1,h2:
  rank 27 -> 8, top1 stays 0, margin -3.125 -> -2.875

sorted_denoise paired-success state, baseline:
  target x1=72 rank 1, coord top1 72

Remove L16 h8:
  rank 1 -> 327, top1 72 -> 0

Remove L16 h12:
  rank 1 -> 9, top1 72 -> 84

Remove L16 h13:
  rank 1 -> 4, top1 72 -> 0

Remove L16 h0,h1,h2:
  rank 1 -> 1, top1 stays 72
```

Dose-response result:

```text
random_denoise failed state:
  h8 has positive target-x1 polarity:
    scale -1.0 -> rank 691 / top1 335
    scale  0.0 -> rank 306 / top1 0
    scale  1.0 -> rank 27  / top1 0
    scale  1.5 -> rank 4   / top1 0
    scale  2.0 -> rank 6   / top1 0

  h12 has opposite failed-state polarity:
    scale -1.0 -> rank 4 / top1 0
    scale  0.0 -> rank 3 / top1 0
    scale  1.0 -> rank 27 / top1 0
    scale  2.0 -> rank 56 / top1 0

  h13 is partly target-supporting when amplified:
    scale -1.0 -> rank 71 / top1 0
    scale  0.0 -> rank 58 / top1 0
    scale  1.5 -> rank 8  / top1 0
    scale  2.0 -> rank 8  / top1 0

sorted_denoise paired-success state:
  h8 is necessary and strongly target-supporting:
    scale -1.0 -> rank 735 / top1 0
    scale  0.0 -> rank 327 / top1 0
    scale  0.5 -> rank 25  / top1 0
    scale  1.0 -> rank 1   / top1 72

  h12 is calibrated, not simply helpful or harmful:
    scale  0.0 -> rank 9   / top1 84
    scale  1.0 -> rank 1   / top1 72
    scale  2.0 -> rank 762 / top1 0
```

Combined context-zero + head-removal result:

```text
random_denoise failed state:
  baseline:
    rank 27, top1 0

  L16 layer_input zero:context_unique:
    rank 27 -> 1, top1 0 -> 72

  zero:context_unique + remove h8:
    rank 27 -> 48, top1 remains 0

  zero:context_unique + remove h12:
    rank 27 -> 1, top1 0 -> 72

  zero:context_unique + remove h13:
    rank 27 -> 2, top1 remains 0

  zero:context_unique + remove h0,h1,h2:
    rank 27 -> 1, top1 0 -> 72

sorted_denoise paired-success state:
  baseline:
    rank 1, top1 72

  L16 layer_input zero:context_unique:
    rank 1 -> 5, top1 72 -> 84

  zero:context_unique + remove h8:
    rank 1 -> 219, top1 72 -> 0

  zero:context_unique + remove h12:
    rank 1 -> 16, top1 72 -> 80

  zero:context_unique + remove h13:
    rank 1 -> 1, top1 stays 72
```

Mechanism update:

```text
The 2157 wine-glass false negative is no longer best described as "model cannot
see the object" or even "target attention is weak" in isolation.

The emerging mechanism is a carrier-to-head route:

1. A context_unique visual carrier competes with the target visual carrier in
   the failed random_denoise state. Removing that carrier at L16 layer_input is
   sufficient to flip x1 from the border basin to the correct coordinate.

2. The successful repaired emission requires L16 head 8. When context_unique is
   zeroed, the failed model reaches target rank 1/top1 72; if h8 output is also
   removed, the repair collapses to rank 48/top1 0. Therefore h8 is downstream
   of the repaired carrier route, not merely a correlational attention head.

3. Head 12 is not the same kind of target route. In the failed baseline, removing
   or inverting h12 improves target rank without changing top1, suggesting it
   carries a competing or miscalibrated basin component under this context. But
   when context_unique is already neutralized, h12 is not required for the
   rank-1/top1-72 repair.

4. Head 13 is intermediate. Removing it weakens the context-zero repair from
   rank 1/top1 72 to rank 2/top1 0, so it helps stabilize the final top1 decision
   but is less decisive than h8.

5. The sorted_denoise paired model uses the same h8-dependent target route but
   starts from a healthier carrier allocation. Zeroing context_unique slightly
   damages sorted_denoise instead of repairing it, while removing h8 is
   catastrophic with or without context_unique zeroing.

This gives a more precise prefix-denoising comparison: the difference is not
that one model has a unique coordinate head and the other lacks it. Both can use
the h8 route. The failure is that random_denoise routes the pre-x1 state through
an interfering context carrier, so h8 cannot select the true coordinate until
that carrier is neutralized.
```

Next deterministic action:

```text
Promote the h8 route into the next probe family:

  - For subtype-A false negatives, test whether h8 necessity generalizes beyond
    image 2157 or whether each missing object has its own carrier/head route.
  - Add a head-region attribution panel for h8/h12/h13: target_generated_overlap
    versus context_unique value/readout contribution before and after
    context_unique zeroing.
  - Compare train versus val high-value sample bases to see whether trained
    failures are also carrier misrouting, or whether train failures are mostly
    coordinate-basin calibration and val failures add perception/grounding
    fragility.
```

## 2026-06-25 Addendum: H8 Route Generalization Panel

Question:

```text
Does the h8/context-carrier mechanism from 2157/rank32 generalize to other
false-negative sample bases, or is it a local repair for one subtype?
```

Selected panel:

```text
Source contextual FN plan:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_guidance_probe/v4_contextual_all_selected_fn_layers0_8_16_20_23_24_27_gpu0/contextual_fn_guidance_probe_plan.jsonl

Scaffold dry-run:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_scaffold_probe/v5_generalization_ranks27_28_37_prex1_dryrun

Rows:
  rank 27 / image 2685 / bottle:
    descriptor/context-confusion case. Target x1=190, context generated x1=209,
    context generated desc=wine glass, target desc=bottle.

  rank 28 / image 12670 / person:
    same-desc crowded-person case. Target x1=163, generated x1=157, paired
    model has much better target rank but still border/top1 competition.

  rank 37 / image 19109 / person:
    tiny/weak-target contrast. Target x1=572, generated x1=584, target visual
    region has only a few tokens.
```

Artifact roots:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_head_output_probe/v4_generalization_ranks27_28_37_random_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_head_output_probe/v4_generalization_ranks27_28_37_sorted_gpu1
```

Verification:

```text
random_denoise panel:
  plan_row_count=3
  row_count=30
  error_count=0
  training_ran=false

sorted_denoise panel:
  plan_row_count=3
  row_count=30
  error_count=0
  training_ran=false
```

Compact result table:

```text
2685 / rank27 / bottle / random_denoise paired context:
  baseline:
    target rank 95, top1 210
  zero context_unique:
    rank 95 -> 105, top1 stays 210
  remove h8:
    rank 95 -> 173, top1 stays 210
  zero context_unique + remove h8:
    rank 95 -> 183, top1 stays 210
  remove h12:
    rank 95 -> 67, top1 210 -> 197

2685 / rank27 / bottle / sorted_denoise failed context:
  baseline:
    target rank 71, top1 216
  zero context_unique:
    rank unchanged at 71, top1 stays 216
  remove h8:
    rank 71 -> 56, top1 216 -> 0
  zero context_unique + remove h8:
    rank 71 -> 60, top1 216 -> 0

12670 / rank28 / person / random_denoise failed context:
  baseline:
    target rank 613, top1 0
  zero context_unique:
    rank 613 -> 774, top1 stays 0
  remove h8:
    rank 613 -> 770, top1 stays 0
  zero context_unique + remove h8:
    rank 613 -> 720, top1 stays 0

12670 / rank28 / person / sorted_denoise paired context:
  baseline:
    target rank 11, top1 0
  zero context_unique:
    rank unchanged at 11, top1 stays 0
  remove h8:
    rank 11 -> 614, top1 stays 0
  zero context_unique + remove h8:
    rank 11 -> 959, top1 stays 0
  remove h12:
    rank 11 -> 4, top1 0 -> 157

19109 / rank37 / person / random_denoise failed context:
  baseline:
    target rank 22, top1 0
  zero context_unique:
    rank 22 -> 27, top1 stays 0
  remove h8:
    rank 22 -> 269, top1 stays 0
  zero context_unique + remove h8:
    rank 22 -> 341, top1 stays 0
  remove h12:
    rank 22 -> 15, top1 stays 0

19109 / rank37 / person / sorted_denoise paired context:
  baseline:
    target rank 39, top1 0
  zero context_unique:
    rank 39 -> 10, top1 stays 0
  remove h8:
    rank 39 -> 349, top1 stays 0
  zero context_unique + remove h8:
    rank 39 -> 254, top1 stays 0
  remove h12:
    rank 39 -> 11, top1 stays 0
```

Mechanism update:

```text
The 2157 result does generalize in one way and fails to generalize in another.

Generalizes:
  h8 behaves like a recurring coordinate-route bottleneck. Removing h8 is
  destructive in all three non-2157 sample bases whenever the target coordinate
  is already in a moderately accessible basin:
    12670 sorted: rank 11 -> 614
    19109 random: rank 22 -> 269
    19109 sorted: rank 39 -> 349
  This supports treating h8 as a reusable coordinate-grounding/output route,
  not a wine-glass-only artifact.

Does not generalize:
  L16 zero:context_unique is not a universal FN repair. It slightly damages or
  does little on 2685 and 12670, and it improves 19109 sorted rank only from 39
  to 10 without moving top1 from border 0. The dramatic 2157 repair
  (rank 27 -> 1, top1 0 -> 72) is therefore subtype-specific.

Interpretation:
  Prefix-denoising FN failures split into at least two separable layers:

  1. A shared h8-mediated coordinate route. Many states depend on h8 for
     preserving or improving the target coordinate rank.

  2. A sample-specific carrier allocation problem. In 2157, context_unique is
     the wrong carrier suppressing the h8 route; neutralizing it is sufficient
     to unlock h8. In 12670 and 19109, the bottleneck is not solved by
     removing context_unique, so the failure likely sits in border-basin
     attraction, sparse/tiny target evidence, descriptor confusion, or another
     carrier not captured by the context_unique mask.

  Head 12 appears repeatedly as a counter-route or basin-shaping component:
     removing h12 improves target rank in 2685 random, 19109 random, 19109
     sorted, and 12670 sorted, sometimes moving top1 from 0 to a near-target
     generated x1 (12670 sorted top1 157). This is not a full rescue, but it is
     a consistent sign that h12 can carry competing basin geometry.
```

Next deterministic action:

```text
Move from "is h8 necessary?" to "what does h8 read from?"

The next useful probe is head-region value/readout attribution for h8/h12/h13:

  - region split: target_generated_overlap, target_unique, generated_unique,
    context_unique, context_generated_overlap, far background.
  - samples: 2157/rank32, 12670/rank28, 19109/rank37.
  - question: does h8 route target-region value in all cases, and does h12 route
    generated/context/border value when it behaves antagonistically?

This should separate three possibilities:
  1. h8 always carries target evidence but top1 is lost later;
  2. h8 itself reads the wrong region in some samples;
  3. h8 is target-supporting, while h12/context/border routes dominate the
     coordinate basin downstream.
```

## 2026-06-25 FN visual value-region probe, L16 h8/h12/h13

Implementation:

```text
script:
  scripts/analysis/run_prefix_denoising_fn_visual_value_region_probe.py

module:
  src/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe.py

tests:
  tests/analysis/test_prefix_denoising_fn_visual_value_region_probe.py

input plan rows:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/inputs/combined_2157_2685_12670_19109_prex1_plan_rows.jsonl

random_denoise artifact:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v2_combined_h8_h12_h13_random_gpu0

sorted_denoise artifact:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v2_combined_h8_h12_h13_sorted_gpu1
```

Verification:

```text
pytest tests/analysis/test_prefix_denoising_fn_visual_value_region_probe.py -q
  Pytest: 2 passed

python -m py_compile \
  src/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe.py \
  scripts/analysis/run_prefix_denoising_fn_visual_value_region_probe.py
  passed

random_denoise full run:
  plan_row_count=4
  condition_plan_row_count=148
  membership_row_count=44
  row_count=148
  error_count=0

sorted_denoise full run:
  plan_row_count=4
  condition_plan_row_count=148
  membership_row_count=44
  row_count=148
  error_count=0

identity controls:
  region_scale=1.0 rows have exactly zero rank/logit movement for both models
```

Sample-base policy:

```text
This round should stay sample-base-centered. The goal is not to spend compute on
normal/well-learned images, but to repeatedly cut into a small set of
high-information image bases where the model state is fragile, asymmetric, or
contradictory across interventions/checkpoints.

Current representative panel:
  2157  / wine glass / subtype-A carrier repairable case
  2685  / wine glass vs bottle descriptor/context confusion
  12670 / person / crowded same-desc false negative
  19109 / person / tiny weak-target / subtype-B border-basin case

Future additions should satisfy at least one of:
  - sorted and random denoising differ sharply on the same image/context;
  - a false negative has moderate target rank but wrong top1 basin;
  - a duplication burst has a measurable pre-onset route or basin signal;
  - train and val versions of the same surface behavior diverge;
  - a head/region patch causes a large causal movement despite weak raw metric
    intuition.
```

Baseline readouts for this value-region pass:

```text
random_denoise:
  2685:  target <|coord_209|>, rank 95,  top1 210, dist 20
  12670: target <|coord_157|>, rank 613, top1 0,   dist 163
  2157:  target <|coord_72|>,  rank 27,  top1 0,   dist 72
  19109: target <|coord_584|>, rank 22,  top1 0,   dist 572

sorted_denoise:
  2685:  target <|coord_209|>, rank 71, top1 216, dist 26
  12670: target <|coord_157|>, rank 11, top1 0,   dist 163
  2157:  target <|coord_72|>,  rank 1,  top1 72,  dist 0
  19109: target <|coord_584|>, rank 39, top1 0,   dist 572
```

Strongest causal movements from zeroing one L16 head value route by visual
region:

```text
Destructive/supporting target route:
  sorted 2157  h8  target_generated_overlap:
    rank 1 -> 486, target logit -5.688, top1 72 -> 0
    source attention 0.5713, contribution norm 20.70

  random 2157  h8  target_generated_overlap:
    rank 27 -> 244, target logit -2.375
    source attention 0.2889, contribution norm 9.92

  sorted 12670 h8  target_generated_overlap:
    rank 11 -> 105, target logit -2.188
    source attention 0.3384, contribution norm 10.49

Counter-route / basin-shaping signals:
  random 12670 h12 context_unique:
    rank 613 -> 538, target logit +0.188
    source attention 0.0597, contribution norm 1.19

  random 12670 h12 target_generated_overlap:
    rank 613 -> 560, target logit +0.188
    source attention 0.0208, contribution norm 0.60

  sorted 12670 h12 target_generated_overlap:
    rank 11 -> 5, target logit +1.125, top1 0 -> 157
    source attention 0.1927, contribution norm 5.66

  sorted 19109 h12 target_generated_overlap:
    rank 39 -> 7, target logit +0.062
    source attention 0.0302, contribution norm 0.98

  sorted 19109 h12 target_unique:
    rank 39 -> 7, target logit +0.125
    source attention 0.0355, contribution norm 1.10

Background is not neutral:
  random 12670 h8 far_background:
    rank 613 -> 566, target logit -0.062
    source attention 0.3919, contribution norm 7.12

  sorted 19109 h12 far_background:
    rank 39 -> 129, target logit -0.625, top1 0 -> 841
    source attention 0.4314, contribution norm 9.54

  sorted 19109 h13 far_background:
    rank 39 -> 15, target logit +0.125
    source attention 0.5854, contribution norm 13.92
```

Mechanism update:

```text
1. h8 is now more specifically localized as a target-overlap value carrier,
   not merely a generic "important head." On 2157, especially sorted_denoise,
   h8 routes value from target_generated_overlap with enormous source
   attention/contribution. Removing that route destroys an otherwise correct
   coordinate basin.

2. Prefix denoising appears to sharpen the same h8 target-overlap route when
   the image/context is solvable. The sorted 2157 route is much stronger than
   random 2157 and is causally necessary for top1 correctness. This supports a
   positive capability-shaping effect, even though the method did not remove
   the broader exposure-bias/duplication problem.

3. The same h8 route can be necessary but insufficient. In 12670 and 19109,
   h8 removal is destructive in earlier head-removal probes, but preserving h8
   does not guarantee top1 correctness. These cases look less like "cannot
   perceive object" and more like target evidence being present but losing the
   basin contest against border/background/context routes.

4. h12 remains the main antagonist/counter-route candidate. Zeroing selected
   h12 visual-region value routes improves target rank on 12670 and 19109,
   sometimes moving top1 from the border token 0 to the target token 157. The
   effect is not uniform enough to call h12 simply "bad"; it is better modeled
   as a basin-shaping route that can either support or oppose target x1
   depending on visual region and sample state.

5. Far-background routing is mechanistically active. It often has the largest
   attention mass/contribution norm and can either repair or damage target
   rank. Treating "background" as null evidence is wrong for these coordinate
   basins; it may carry image-border, canvas-prior, stop/continuation, or
   object-layout evidence.

6. The representative sample-base strategy is paying off. A normal aggregate
   sweep would likely blur the split between:
     - 2157 subtype-A: target carrier exists and can be unlocked/sharpened;
     - 12670 crowded subtype: target evidence competes with same-desc/context
       and border basin;
     - 19109 tiny subtype-B: target evidence is weak and far-background/border
       routes dominate the emitted coordinate.
```

Next deterministic action:

```text
Build a deeper tensor-flow/surgery stack on the same sample bases:

  A. For 2157, trace target_generated_overlap -> L16 h8 -> coordinate readout
     across adjacent layers and test whether sorted_denoise creates a stronger
     bridge than random_denoise.

  B. For 12670 and 19109, isolate the h12 and far-background routes that improve
     rank when removed. Test whether they inject border-token attraction,
     context-object carryover, or termination/continuation bias.

  C. Add one train-set analogue for each subtype before making any val-only
     claim. The point is not metric evaluation; it is to check whether these
     route/basin mechanics exist even on trained sequences or are mainly
     unseen-image generalization failures.
```

## 2026-06-25 FN visual value-region layer sweep, L12-L20

Artifacts:

```text
random_denoise, 2157+2685:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v3_layer_sweep_2157_2685_random_gpu0

sorted_denoise, 2157+2685:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v3_layer_sweep_2157_2685_sorted_gpu1

random_denoise, 12670+19109:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v3_layer_sweep_12670_19109_random_gpu2

sorted_denoise, 12670+19109:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v3_layer_sweep_12670_19109_sorted_gpu3
```

Scope:

```text
layers:
  12,13,14,15,16,17,18,19,20

heads:
  8,12,13

regions:
  target_generated_overlap,target_unique,context_unique,far_background

scale controls:
  0.0 causal zeroing
  1.0 identity

result:
  each artifact has plan_row_count=2, condition_plan_row_count=434,
  membership_row_count=22, row_count=434, error_count=0

identity controls:
  all four artifacts have max_abs_rank_delta=0 and max_abs_logit_delta=0.0 for
  region_scale=1.0 rows
```

Strongest layer-sweep causal effects:

```text
Top harms from zeroing one value-region route:
  sorted 2157  L16 h8  target_generated_overlap:
    rank 1 -> 486, target logit -5.688, top1 72 -> 0,
    attention 0.571, contribution norm 20.70

  random 2157  L16 h8  target_generated_overlap:
    rank 27 -> 244, target logit -2.375,
    attention 0.289, contribution norm 9.92

  sorted 19109 L17 h8  target_generated_overlap:
    rank 39 -> 214, target logit -0.375,
    attention 0.425, contribution norm 22.40

  random 12670 L16 h8  context_unique:
    rank 613 -> 767, target logit -0.312,
    attention 0.101, contribution norm 2.25

  random 12670 L16 h12 far_background:
    rank 613 -> 734, target logit -0.125,
    attention 0.334, contribution norm 5.67

  sorted 12670 L16 h8  target_generated_overlap:
    rank 11 -> 105, target logit -2.188,
    attention 0.338, contribution norm 10.49

  sorted 19109 L16 h12 far_background:
    rank 39 -> 129, target logit -0.625, top1 0 -> 841,
    attention 0.431, contribution norm 9.54

Top improvements from zeroing one value-region route:
  random 12670 L14 h13 far_background:
    rank 613 -> 526, target logit +0.188,
    attention 0.271, contribution norm 2.73

  random 12670 L16 h12 context_unique:
    rank 613 -> 538, target logit +0.188,
    attention 0.060, contribution norm 1.19

  random 12670 L17 h8 far_background:
    rank 613 -> 542, target logit +0.125,
    attention 0.788, contribution norm 25.09

  random 12670 L16 h12 target_generated_overlap:
    rank 613 -> 560, target logit +0.188,
    attention 0.021, contribution norm 0.60

  sorted 19109 L16 h12 target_generated_overlap:
    rank 39 -> 7, target logit +0.062,
    attention 0.030, contribution norm 0.98

  sorted 19109 L16 h12 target_unique:
    rank 39 -> 7, target logit +0.125,
    attention 0.035, contribution norm 1.10

  sorted 19109 L16 h13 far_background:
    rank 39 -> 15, target logit +0.125,
    attention 0.585, contribution norm 13.92
```

Layer trajectory for the h8 target-overlap route:

```text
2157 / wine glass:
  random:
    L12 +6, L13 0, L14 0, L15 +3,
    L16 +217 with logit -2.38,
    L17 -5, L18 0, L19 0, L20 -4

  sorted:
    L12 0, L13 0, L14 0, L15 0,
    L16 +485 with logit -5.69 and top1 72 -> 0,
    L17 0, L18 0, L19 0, L20 0

2685 / descriptor-context confusion:
  random:
    mostly small, but L17 has attention 0.253 and contribution norm 10.88 with
    little rank effect

  sorted:
    L16 -3 with logit +0.25, L17 -10 with logit +0.38 and very large
    attention/contribution (0.936 / 46.11), but top1 remains descriptor/context
    confused

12670 / crowded same-desc person:
  random:
    L16 +108 with logit -0.25, surrounded by mixed small effects; later L19/L20
    zeroing improves rank despite almost no target-overlap mass

  sorted:
    L16 +94 with logit -2.19, while adjacent layers are small; target evidence
    is concentrated around L16 but still loses top1 to border 0

19109 / tiny weak-target person:
  random:
    L17 has large h8 target-overlap contribution norm (12.27) but almost no
    rank movement, suggesting target evidence can be present without readout
    control

  sorted:
    L16 +59 and L17 +175; the h8 target route becomes most necessary at L17,
    not L16, but top1 still remains border 0
```

Mechanism update:

```text
1. L16 is a genuine basin-injection layer for several sample bases, not an
   arbitrary probe site. It is the sharpest h8 target-overlap dependency for
   2157 and 12670, and it is also where h12/far-background has large causal
   effects.

2. Prefix denoising sharpens a sparse target-overlap bridge when the state is
   already close to solvable. The sorted 2157 bridge is almost all-or-nothing:
   L16 h8 target-overlap zeroing alone destroys top1 correctness. Adjacent
   layers have value mass but no comparable causal readout effect.

3. False negatives are not explained by absence of visual target evidence.
   In 12670 and 19109, h8 target-overlap routes can be causally necessary, yet
   the emitted top1 remains a border or wrong basin. The failure is better
   modeled as a basin competition problem after perception, not a pure visual
   blindness problem.

4. h12 is not simply an antagonist. In the same L16 neighborhood, h12
   target/context routes can oppose the target coordinate, while h12
   far-background can support the target. The meaningful unit is therefore
   "head x region x layer," not just "head."

5. Far-background should be treated as an active coordinate prior route. The
   largest attention/contribution values often come from far_background, and
   zeroing those values can either repair or damage target rank. This is a
   plausible carrier for border/canvas priors, object-layout priors, and
   continuation/termination pressure.

6. 2685 is a useful non-person sample-base because it shows a different failure
   mode: huge h8 target-overlap value mass at L17 in sorted_denoise, but little
   rank/top1 repair. That points toward descriptor/context binding rather than
   raw target-region routing as the bottleneck.
```

Next deterministic action:

```text
Do not broaden to normal images. Build the next surgery around three narrow
questions:

  1. Bridge tomography:
     For 2157 and 12670, patch L16 h8 target-overlap value/output from sorted
     into random and random into sorted. Test whether the prefix-denoising
     capability gain is the route vector itself, the downstream readout state,
     or their compatibility.

  2. Basin-prior decomposition:
     For 12670 and 19109, decompose h12 and far-background effects by coord
     token family: target bin, border 0, far-border/high-bin modes such as 841,
     and local radius mass. The current rank movement is not enough; we need
     to know which basin is being suppressed or amplified.

  3. Train-set analogue:
     Select one train image base matching each subtype before making a val-only
     mechanism claim:
       - subtype-A carrier repairable target-overlap bridge;
       - crowded same-desc target-vs-context basin competition;
       - tiny weak-target border-prior basin;
       - descriptor/context confusion.
```

## 2026-06-25 FN visual value-region bridge transplant

Implementation:

```text
script:
  scripts/analysis/run_prefix_denoising_fn_visual_value_region_bridge_probe.py

module:
  src/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_bridge_probe.py

tests:
  tests/analysis/test_prefix_denoising_fn_visual_value_region_bridge_probe.py
```

Bridge semantics:

```text
For the same image base / pre-x1 state family:
  receiver_contribution = receiver checkpoint value-region contribution
  donor_contribution    = donor checkpoint value-region contribution

Transforms:
  donor_replace:
    receiver head slice += donor_contribution - receiver_contribution

  donor_add:
    receiver head slice += donor_contribution

  receiver_remove:
    receiver head slice -= receiver_contribution

This tests whether the prefix-denoising effect lives in the route vector, in the
receiver downstream basin/readout state, or in their compatibility.
```

Verification:

```text
pytest tests/analysis/test_prefix_denoising_fn_visual_value_region_bridge_probe.py -q
  Pytest: 4 passed

python -m py_compile \
  src/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_bridge_probe.py \
  scripts/analysis/run_prefix_denoising_fn_visual_value_region_bridge_probe.py
  passed

smoke:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_bridge_probe/v2_smoke_2157_sorted_to_random_h8_gpu0
  plan_row_count=2
  condition_plan_row_count=3
  membership_row_count=33
  row_count=3
  error_count=0

full bridge panel:
  each shard has plan_row_count=4, condition_plan_row_count=146,
  membership_row_count=66, row_count=146, error_count=0
```

Artifacts:

```text
2157 + 12670, sorted_denoise -> random_denoise:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_bridge_probe/v1_bridge_2157_12670_sorted_to_random_gpu0

2157 + 12670, random_denoise -> sorted_denoise:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_bridge_probe/v1_bridge_2157_12670_random_to_sorted_gpu1

2685 + 19109, sorted_denoise -> random_denoise:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_bridge_probe/v1_bridge_2685_19109_sorted_to_random_gpu2

2685 + 19109, random_denoise -> sorted_denoise:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_bridge_probe/v1_bridge_2685_19109_random_to_sorted_gpu3
```

Selected bridge results:

```text
2157 / wine glass / subtype-A:
  sorted -> random, L16 h8 target_generated_overlap donor_replace:
    target 72, rank 27 -> 2, top1 0 -> 0, distance 72 -> 72,
    target logit +2.750, bridge delta norm 12.299

  sorted -> random, L16 h8 target_generated_overlap donor_add:
    target 72, rank 27 -> 2, top1 0 -> 78, distance 72 -> 6,
    target logit +3.625, bridge delta norm 20.700

  random -> sorted, L16 h8 target_generated_overlap donor_replace:
    target 72, rank 1 -> 302, top1 72 -> 0, distance 0 -> 72,
    target logit -2.625, bridge delta norm 12.299

12670 / crowded person:
  sorted -> random, L16 h8 target_generated_overlap donor_replace:
    target 163, rank 613 -> 59, top1 stays 0,
    target logit +1.938, bridge delta norm 9.517

  sorted -> random, L16 h8 target_generated_overlap donor_add:
    target 163, rank 613 -> 34, top1 stays 0,
    target logit +2.188, bridge delta norm 10.493

  random -> sorted, L16 h12 target_generated_overlap donor_replace:
    target 163, rank 11 -> 4, top1 0 -> 157, distance 163 -> 6,
    target logit +1.000, bridge delta norm 5.100

19109 / tiny weak-target person:
  random -> sorted, L17 h8 target_generated_overlap donor_add:
    target 572, rank 39 -> 14, top1 0 -> 584, distance 572 -> 12,
    target logit +0.125, bridge delta norm 12.262

  random -> sorted, L17 h8 context_unique donor_add:
    target 572, rank 39 -> 17, top1 0 -> 584, distance 572 -> 12,
    target logit +0.188, bridge delta norm 19.171

  random -> sorted, L16 h12 far_background receiver_remove:
    target 572, rank 39 -> 129, top1 0 -> 841, distance 572 -> 269,
    target logit -0.625, bridge delta norm 9.548

2685 / descriptor-context confusion:
  random -> sorted, L17 h8 target_generated_overlap donor_replace:
    target 190, rank 71 -> 63, top1 remains 216, distance stays 26,
    target logit +0.375, bridge delta norm 36.284

  random -> sorted, L17 h8 target_generated_overlap receiver_remove:
    target 190, rank 71 -> 61, top1 remains 216, distance stays 26,
    target logit +0.375, bridge delta norm 46.120
```

Mechanism update:

```text
1. Prefix denoising changes a transferable target-overlap route vector. The
   clearest evidence is sorted -> random at 2157 and 12670:
     - 2157 L16 h8 target-overlap moves random rank 27 -> 2.
     - 12670 L16 h8 target-overlap moves random rank 613 -> 59 or 34.
   This is stronger than simple removal/ablation evidence because the donor
   vector carries capability into a weaker receiver.

2. The donor route vector is not sufficient for exact top1 by itself. On 2157,
   sorted -> random donor_replace makes the target rank 2 but leaves top1 at
   border 0. donor_add moves top1 near target (78, distance 6) but not exact
   target 72. This separates "route contains target evidence" from "receiver
   readout state collapses onto exact coordinate."

3. The route/readout compatibility is asymmetric. Replacing sorted_denoise's
   correct 2157 h8 target-overlap contribution with random_denoise's
   contribution destroys the correct top1:
     rank 1 -> 302, top1 72 -> 0.
   Therefore sorted_denoise is not merely using a generic h8 slot; its L16 h8
   target-overlap vector and downstream state are tuned together.

4. 12670 shows that sorted_denoise owns a much stronger h8 target-overlap route
   than random_denoise, but the exact border-vs-target readout remains hard.
   The target rank can be largely rescued in random, yet top1 remains 0. This
   reinforces the basin-competition story rather than a pure perception story.

5. 19109's tiny-target failure is later and fuzzier. The useful bridge is L17
   h8 and it moves sorted top1 from border 0 to near-target 584 while the target
   is 572. This is not exact correction, but it is strong evidence that the
   model can be pushed into the local coordinate neighborhood.

6. h12/far_background is still an active basin route. Removing sorted 19109
   L16 h12 far_background moves top1 from 0 to 841 and damages target rank,
   while h12 target-overlap manipulations can improve 12670. This is a
   region-specific basin-shaping circuit, not a globally bad head.

7. 2685 behaves differently from the person FNs. Even large L17 h8
   target-overlap bridge deltas do not fix top1; target rank moves only modestly
   and the top1 remains around the descriptor/context-confused coordinate. This
   strengthens the split between visual-route availability and descriptor/context
   binding failure.
```

Working theory after bridge transplant:

```text
Prefix-denoising SFT appears to sharpen a sparse mid-layer visual-to-coordinate
route, especially L16 h8 target-overlap. That route is partially transferable
across checkpoints, so the capability difference is not only in late lm_head
readout. But exact coordinate emission requires compatibility between:

  visual region contribution vector,
  head/query receiving state,
  downstream coordinate-basin state,
  competing background/context/head routes,
  and the special-token coordinate embedding basin.

False negatives are therefore better modeled as failures of basin selection and
binding under autoregressive context, not as absence of object perception.
```

Next deterministic action:

```text
Use the bridge rows to build a basin-prior decomposition rather than another
broad metric:

  - track target bin, patched top1 bin, border 0, near-target radius mass, and
    high-border modes such as 841;
  - compare bridge transforms that improve rank but leave top1 at border;
  - isolate whether the final obstruction is lm_head/coord embedding geometry,
    downstream residual competition, or a specific h12/background route.

Then add one train-set analogue per subtype to separate memorized-sequence
mechanics from unseen-val mechanics.
```

## 2026-06-25 Coordinate-basin decomposition over bridge panel

Implementation:

```text
script:
  scripts/analysis/run_prefix_denoising_coord_basin_decomposition.py

module:
  src/analysis/prefix_denoising_surgery_probing/coord_basin_decomposition.py

tests:
  tests/analysis/test_prefix_denoising_coord_basin_decomposition.py
```

Scope:

```text
Input rows:
  bridge-transplant panel over image bases 2157, 12670, 2685, 19109

Total bridge rows:
  584

Output:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coord_basin_decomposition/v1_bridge_panel

Boundary:
  This is a readout decomposition over selected mechanistic states, not a
  population metric and not a new model evaluation.
```

Reducer semantics:

```text
For each bridge row, compare baseline and patched top-1 coordinate bins against
the intended target coordinate bin. Label the top-1 coordinate basin as:

  exact_target
  near_target_radius1 / 4 / 8 / 16 / 32
  low_border
  high_border
  far_from_target
  missing_target_or_top1

Track both rank movement and top-1 basin movement:

  target rank improved/worsened
  top-1 moved toward/away from target
  entered exact target
  entered near-target radius16
  entered or left border basin
```

Aggregate result:

```text
baseline_basin_family_counts:
  exact_target: 73
  low_border: 365
  near_target_radius32: 146

patched_basin_family_counts:
  exact_target: 40
  far_from_target: 3
  low_border: 375
  near_target_radius8: 7
  near_target_radius16: 18
  near_target_radius32: 141

entered_near_target_radius16_count: 9
entered_border_count: 17
left_border_count: 7
target_rank_improved_count: 245
target_rank_worsened_count: 249
```

Near-target entries from outside radius16:

```text
2157 / sorted -> random / L16 h8 / target_generated_overlap / donor_add:
  top1 0 -> 78, target 72, basin low_border -> near_target_radius8,
  rank 27 -> 2

12670 / random -> sorted / L16 h12 / target_generated_overlap / donor_replace:
  top1 0 -> 157, target 163, basin low_border -> near_target_radius8,
  rank 11 -> 4

12670 / random -> sorted / L16 h12 / target_generated_overlap / receiver_remove:
  top1 0 -> 157, target 163, basin low_border -> near_target_radius8,
  rank 11 -> 5

2685 / sorted -> random / L16 h8 / target_generated_overlap / donor_add:
  top1 210 -> 197, target 190, basin near_target_radius32 -> near_target_radius8,
  rank 95 -> 98

2685 / sorted -> random / L17 h8 / target_generated_overlap / donor_add:
  top1 210 -> 197, target 190, basin near_target_radius32 -> near_target_radius8,
  rank 95 -> 112

2685 / sorted -> random / L17 h8 / target_generated_overlap / receiver_remove:
  top1 210 -> 177, target 190, basin near_target_radius32 -> near_target_radius16,
  rank 95 -> 96

2685 / sorted -> random / L17 h8 / context_unique / receiver_remove:
  top1 210 -> 177, target 190, basin near_target_radius32 -> near_target_radius16,
  rank 95 -> 99

19109 / random -> sorted / L17 h8 / target_generated_overlap / donor_add:
  top1 0 -> 584, target 572, basin low_border -> near_target_radius16,
  rank 39 -> 14

19109 / random -> sorted / L17 h8 / context_unique / donor_add:
  top1 0 -> 584, target 572, basin low_border -> near_target_radius16,
  rank 39 -> 17
```

Mechanism update:

```text
1. The bridge panel cleanly separates target-evidence movement from final
   coordinate-basin capture. Many rows improve the target coordinate rank, but
   only 9 / 584 rows enter the near-target radius16 basin from outside it.

2. The dominant obstruction is often a low-border coordinate attractor rather
   than absence of target evidence. This is clearest for 12670: sorted -> random
   h8 target-overlap bridge rows rescue the target rank substantially, but top1
   remains at 0.

3. 2157 is the strongest transferable route case. sorted -> random L16 h8
   target-overlap donor_add moves the random receiver from border 0 to near
   target 78 for target 72. donor_replace improves rank without escaping the
   border basin, suggesting that route-vector magnitude and downstream basin
   state jointly decide top1 capture.

4. 19109 is a weak/tiny-object case where the useful movement is later and less
   exact. random -> sorted L17 h8 donor_add reaches local neighborhood 584 for
   target 572 but does not land exactly. This looks like visual evidence exists
   but is weakly synchronized with the exact coordinate basin.

5. 2685 remains a descriptor/context-binding case. Several surgeries move the
   top1 coordinate closer, but the rank improvements are modest or inconsistent
   and the model stays near a confused local coordinate rather than becoming an
   exact target emitter.
```

Sample-base rule for next research loop:

```text
Use image-base-centered panels, not broad normal-image sweeps. Each selected
sample base must earn its slot by exposing a mechanism:

  - strong same-image difference between checkpoints;
  - false negative with evidence/rank/top1 dissociation;
  - duplication burst or repeated spatial anchor;
  - premature stop or continuation collapse;
  - coordinate-basin transition under causal surgery;
  - train-vs-val counterpart for the same subtype.

Normal or well-learned images are controls only. They should be sampled when a
specific counterfactual is needed, e.g. whether a route exists in healthy
emission, whether border attraction is special to failures, or whether a
train-set sequence uses the same basin mechanics as an unseen val image.
```

Next deterministic action:

```text
Promote the study unit from "metric slice" to:

  sample_base_id + target object span + checkpoint pair + failure subtype.

The seed panel remains:

  2157: transferable h8 target-overlap route, border escape under donor_add
  12670: rank-rescue without top1 basin escape
  19109: tiny weak target, later fuzzy local-basin entry
  2685: descriptor/context binding failure around near but wrong coordinate

Add only high-value bases after this, preferably by selecting the most extreme
same-image contrast cases from existing val200 and then one matched training
analogue per subtype.
```

## 2026-06-26 update - sample-base contrast matrix and duplicate-onset atlas

Scope:

```text
Question:
  Compare how prefix-denoising changes object perception, binding, coordinate
  basin formation, and object-span emission. Avoid broad normal-image sweeps;
  select representative sample(image)-bases that expose a mechanism.

Primary checkpoints:
  sorted prefix-denoising:
    /data/CoordExp/outputs/stage1_2b/detection_teacher_forcing/compact_object_box_closed_desc_first_prefix_denoising_kl_w0p05_k2_2b_base_sorted_bsz1x128_4epoch/compact-object-box-closed-desc-first-prefix-denoising-kl-w0p05-k2-2b-base-sorted-bsz1x128-4epoch/v5-20260623-133125/checkpoint-908

  random prefix-denoising:
    /data/CoordExp/outputs/stage1_2b/detection_teacher_forcing/compact_object_box_closed_desc_first_prefix_denoising_kl_w0p05_k2_2b_base_random_bsz1x128_4epoch/compact-object-box-closed-desc-first-prefix-denoising-kl-w0p05-k2-2b-base-random-bsz1x128-4epoch/v2-20260623-133126/checkpoint-908

Weak rollout control:
  pure-CE sorted natural-adjacent checkpoint-928, from:
    /data/CoordExp/outputs/infer/natadj_len12000_free_val200/natadj_sorted_ckpt928_val200_free_temp0_rp1p10_max3084_bsz1_8gpu_symlinkbase
```

Artifact handles:

```text
sample-base registry:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/sample_base_registry/v3_denoise_sorted_random_purece_val200_canonical

object-level probe panel:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/probe_panel_selector/v5_val200_object_panel_purece_control_canonical

cross-model contrast case matrix:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/contrast_case_matrix/v4_val200_purece_control_candidates_canonical

denoise-only duplicate-onset atlases:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v10_duplicate_onset_2685_wineglass_denoise_gpu0
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v10_duplicate_onset_2157_wineglass_denoise_gpu1
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v10_duplicate_onset_12670_person_denoise_gpu2
```

Implementation update:

```text
1. Added contrast_case_matrix.py to reduce existing probe-panel rows into
   surgery-ready cross-model cases:
     - false_negative_object_contrast
     - duplicate_loop_contrast
     - termination_contrast

2. Added checkpoint-path canonicalization to sample_base_registry.py. This
   preserves the requested checkpoint path but canonicalizes stale
   /data/CoordExp/output/... handles to existing /data/CoordExp/outputs/...
   handles when needed.

3. Regression coverage now checks both the contrast reducer and the
   singular-output to plural-outputs path canonicalization.
```

Pure-CE control boundary:

```text
The pure-CE checkpoint is usable as a rollout-level weak control for sample
selection and contrast grouping.

It is not load-compatible with the current inner-state atlas surface without a
separate compatibility plan:

  pure-CE checkpoint modules_to_save:
    ["coord_offset_adapter"]

  prefix-denoising checkpoint modules_to_save:
    ["token_embeddings_adapter"]

Therefore pure-CE evidence in this round should be interpreted as:
  "surface rollout control and selection comparator"
not:
  "matched hidden-state causal comparator".
```

Contrast-matrix summary:

```text
Full candidate matrix:
  input rows: 886
  cases: 357

case_kind_counts:
  duplicate_loop_contrast: 30
  false_negative_object_contrast: 319
  termination_contrast: 8

contrast_family_counts:
  random_denoise_amplified_loop: 8
  control_amplified_loop: 4
  shared_duplicate_loop: 5
  denoise_both_miss_control_recovers: 96
  denoise_split_object_binding: 62
  control_misses_denoise_recovers: 4
  denoise_empty_control_not_flagged: 5
  denoise_termination_fault: 3
  all_models_miss_unbound_same_desc: 157
  duplicate_loop_contrast: 13
```

Highest-value sample-base seeds:

```text
Duplication onset:
  2685 / wine glass:
    random_denoise duplicate cluster 53 vs pure-CE 10
    This is the strongest random-denoise amplified loop.

  2157 / wine glass:
    random_denoise 39 vs sorted_denoise 8 vs pure-CE 4
    This is the strongest bridge/border-basin case already seen in surgery.

  12670 / person:
    random_denoise 30 vs sorted_denoise 8 vs pure-CE 3
    This is a repeated person/start/border-anchor case.

  19432 / chair:
    random_denoise 29 vs pure-CE 7 vs sorted_denoise 3
    This is the next chair analogue for replication after the three seeds.

False-negative and guidance bridge:
  12670 / gt15 person [492,403,615,608]:
    random+sorted denoise miss, pure-CE matches, same-desc unbound predictions
    are abundant.

  12670 / gt2 person [64,539,242,999]:
    random+sorted denoise miss, pure-CE matches.

  16228 / gt13 person [490,406,549,531]:
    random+sorted denoise miss, pure-CE matches.

  2157 / gt13 wine glass [370,113,469,420]:
    random+sorted denoise miss, pure-CE matches.

  2157 / gt4 wine glass [72,40,178,377]:
    denoise split-object binding case; random+pure miss, sorted matches.

Termination or empty continuation:
  12120, 18380, 4134:
    sorted/random denoise emit empty predictions while pure-CE emits many
    objects. These are high-value wrapper/continuation collapse bases, not
    ordinary false-negative examples.
```

Duplicate-onset atlas readout, denoise-only:

```text
2685 / wine glass:
  rows: 168
  model rows: random=84, sorted=84
  readout_status: ok=168

  random L20 -> L27:
    median target rank 446.5 -> 2.0
    median top1 distance 52.5 -> 6.5
    median coord-vocab mass 0.4551 -> 0.9973
    L27 top bins include 138, 131, 105, 624

  sorted L20 -> L27:
    median target rank 89.0 -> 2.0
    median top1 distance 14.0 -> 3.0
    median coord-vocab mass 0.2112 -> 0.9981
    L27 top bins include 197, 414, 236, 229

2157 / wine glass:
  rows: 196
  model rows: random=98, sorted=98
  readout_status: ok=196

  random L20 -> L27:
    median target rank 121.0 -> 2.0
    median top1 distance 95.0 -> 95.0
    median coord-vocab mass 0.1507 -> 0.9962
    L27 top bin is mostly 0

  sorted L20 -> L27:
    median target rank 91.5 -> 1.0
    median top1 distance 105.5 -> 0.0
    median coord-vocab mass 0.2528 -> 0.9967
    L27 top bin often 0 but target rank and exact capture are better.

12670 / person:
  rows: 238
  model rows: random=126, sorted=112
  readout_status: ok=238

  random L20 -> L27:
    median target rank 29.0 -> 1.0
    median top1 distance 197.0 -> 0.0
    median coord-vocab mass 0.1393 -> 0.9922
    L27 top bin is always 0 for selected pre-x1 states.

  sorted L20 -> L27:
    median target rank 460.0 -> 2.0
    median top1 distance 141.0 -> 213.0
    median coord-vocab mass 0.1237 -> 0.9941
    L27 top bins are mostly 0, with smaller alternatives 203, 414, 531.
```

Mechanism update:

```text
1. Duplication onset is not explained by absence of coordinate knowledge. In
   the selected duplicate states, both denoise models often form strong
   coordinate-token mass by the final layer.

2. The visible burst is better modeled as repeated re-entry into an
   object-start/pre-x1 state manifold, followed by a late coordinate-basin
   capture. The first major coordinate transition appears around L20, L23/L24
   decide local-vs-border basin behavior, and L27 locks the final coordinate
   distribution.

3. Random-denoise amplified loops often contain target-rank evidence, but that
   evidence can dissociate from top1 basin capture. 2157 is the cleanest case:
   random reaches median target rank 2.0 at L27 while top1 remains mostly in
   the border basin.

4. 12670 shows a more severe repeated border-anchor onset circuit. Random's
   selected person pre-x1 states collapse to coord 0 at L27 for every selected
   row. That is not healthy perception; it is repeated state entry plus a
   border/start anchor.

5. The next probe should not ask "which model has better metrics?" It should
   ask how a given sample-base routes from visual object evidence to semantic
   descriptor binding, then to coordinate-basin entry, then to object-span
   emission or collapse.
```

Sample-base selection rule, tightened:

```text
Representative image bases are the primary experimental unit:

  sample_base_id + target object span + checkpoint pair + failure subtype

Normal or well-learned images are controls only. They should not consume the
main GPU budget unless they answer a specific counterfactual:

  - Is the same route present during healthy object emission?
  - Is border attraction special to failure cases?
  - Does a train-set analogue use the same basin mechanics as an unseen val
    sample?
  - Does a prompt/guidance intervention rescue a false negative without
    changing visual evidence?

Default narrow-deep queue:
  1. duplicate onset: 2685/wine_glass, 2157/wine_glass, 12670/person
  2. false-negative guidance bridge: 12670/gt15, 12670/gt2, 16228/gt13,
     2157/gt13, 2157/gt4
  3. termination collapse: 12120, 18380, 4134
  4. replication families only after these mechanisms sharpen:
     19432/chair, 12639/person, 15254/broccoli, 19109/person-or-motorcycle
```

Next deterministic implementation action:

```text
Create a compact atlas reducer that turns atlas_rows.jsonl into per-case
layer-transition features:

  - target-rank trajectory by model/layer
  - top1 distance and basin label by model/layer
  - coord-vocab mass and radius mass by model/layer
  - duplicate_guard_action / keep-vs-suppress stratification
  - object descriptor and generated span anchors

Then use that reducer to select L20-L24 causal patch/attention probes for the
top three duplicate-onset bases, followed by the false-negative guidance bridge
cases.
```

## 2026-06-26 update - atlas transition reducer and next surgery queue

Implemented the compact atlas reducer proposed above:

```text
module:
  src/analysis/prefix_denoising_surgery_probing/atlas_transition_features.py

test:
  tests/analysis/test_prefix_denoising_atlas_transition_features.py

artifact:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/atlas_transition_features/v3_duplicate_onset_2685_2157_12670_denoise_dominant_sample_base
```

The reducer writes:

```text
atlas_transition_layer_rows.jsonl
  one row per atlas state/layer, with basin family, rank-good flag,
  high-coordinate-mass flag, duplicate guard action, and coordinate readout
  features.

atlas_transition_case_rows.jsonl
  one row per generated object coordinate state, summarizing layer trajectory
  and recommending L20/L23/L24 patch sites.

atlas_transition_sample_base_rows.jsonl
  one row per image base, using dominant transition family per model so a
  minority failure row does not hide the actual cross-model contrast.
```

Aggregate result over the three duplicate-onset atlases:

```text
input atlas rows: 602
transition layer rows: 602
transition case rows: 86
sample-base rows: 3

transition_family_counts:
  late_exact_or_near_lock: 63
  rank_evidence_basin_dissociation: 21
  border_locked_without_rank_evidence: 2

final_basin_family_counts:
  exact_target: 42
  low_border: 21
  near_target_radius1: 5
  near_target_radius4: 5
  near_target_radius8: 6
  near_target_radius16: 5
  near_target_radius32: 1
  far_from_target: 1
```

Sample-base read:

```text
1. image 2157 / wine glass
   sample_family: cross_model_basin_dissociation
   dominant_transition_family_by_model:
     random_denoise: rank_evidence_basin_dissociation
     sorted_denoise: late_exact_or_near_lock
   final_basin_family_counts_by_model:
     random_denoise: exact=4, low_border=8, near=2
     sorted_denoise: exact=8, low_border=4, near=2
   next_probe:
     paired_l20_l24_cross_model_basin_patch

2. image 12670 / person
   sample_family: cross_model_basin_dissociation
   dominant_transition_family_by_model:
     random_denoise: late_exact_or_near_lock
     sorted_denoise: rank_evidence_basin_dissociation
   final_basin_family_counts_by_model:
     random_denoise: exact=18
     sorted_denoise: exact=5, low_border=9, near=2
   next_probe:
     paired_l20_l24_cross_model_basin_patch

3. image 2685 / mixed wine-glass/person/bottle states
   sample_family: minority_basin_dissociation
   dominant_transition_family_by_model:
     random_denoise: late_exact_or_near_lock
     sorted_denoise: late_exact_or_near_lock
   interpretation:
     still important for duplicate burst and semantic-anchor analysis, but no
     longer the first target for coordinate-basin escape surgery because the
     selected pre-x1 atlas states are mostly late-lock states.
```

Mechanism refinement:

```text
The strongest next causal question is no longer "does the final layer know the
coordinate?" The answer is often yes by rank.

The sharper question is:

  Which L20/L23/L24 route components decide whether high-rank coordinate
  evidence is captured by the intended local basin, or is instead swallowed by
  a low-border/border-start basin?

2157 and 12670 are the first paired sample bases because they reverse the
dominant model roles:

  2157:
    random is mostly rank-evidence/basin-dissociation;
    sorted is mostly late-lock.

  12670:
    sorted is mostly rank-evidence/basin-dissociation;
    random is mostly late-lock.

That reversal is high value: it can separate "random ordering is bad" from a
more general route/basin mechanism shaped by prefix-denoising plus sample
context.
```

Next deterministic implementation action:

```text
Run paired L20/L23/L24 cross-model basin patching for:

  2157:
    receiver=random_denoise dissociation states;
    donor=sorted_denoise late-lock states.

  12670:
    receiver=sorted_denoise dissociation states;
    donor=random_denoise late-lock states.

The immediate readouts should be:
  - target rank movement;
  - top1 coordinate movement;
  - basin family movement;
  - whether border/low-border top1 is escaped;
  - whether the same route works for suppress vs keep duplicate states.
```

## 2026-06-26 update - exact-state residual surgery and sample-base gate

Added exact source-state filtering to the residual/readout surgery runner:

```text
module:
  src/analysis/prefix_denoising_surgery_probing/residual_readout_surgery.py

new selector:
  --source-state-keys

test:
  tests/analysis/test_prefix_denoising_residual_surgery.py
```

Also extended the coordinate-basin reducer so it can consume residual-surgery
rows directly:

```text
module:
  src/analysis/prefix_denoising_surgery_probing/coord_basin_decomposition.py

accepted residual fields:
  coord_top1_bin -> patched_coord_top1_bin
  coord_target_rank -> patched_coord_target_rank
  coord_target_logit -> patched_coord_target_logit
  alpha -> bridge_strength

test:
  tests/analysis/test_prefix_denoising_coord_basin_decomposition.py
```

Selector lesson:

```text
Exact source keys selected from atlas-transition top cases often do not look
good at L20/L23/L24 yet. Selecting source rows by candidate layers 20,23,24
missed most of the intended top-six cases. Selecting by L27 atlas evidence and
then probing L20/L23/L24 selected all six target rows for both sample bases.
```

Readout-surgery artifacts:

```text
2157 random_denoise top-six dissociation rows:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/residual_readout_surgery/v4_2157_random_top6_select_l27_probe_l20_l23_l24_gpu0

12670 sorted_denoise top-six dissociation rows:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/residual_readout_surgery/v4_12670_sorted_top6_select_l27_probe_l20_l23_l24_gpu1
```

Both jobs used:

```text
selector evidence layer: 27
probe layers: 20,23,24
alphas: 0,0.005,0.01,0.02,0.05,0.1,0.2
surgery mode: normalized coordinate-basin readout direction
training ran: false
scope: readout-space intervention, not full activation-patch generation
```

Raw residual-surgery summary:

```text
2157 random_denoise:
  selected_source_row_count: 6
  output_row_count: 126
  readout_status_counts: ok=126
  flip_to_target_count: 25
  coord_distance_improved_count: 56
  rank_improved_count: 102

12670 sorted_denoise:
  selected_source_row_count: 6
  output_row_count: 126
  readout_status_counts: ok=126
  flip_to_target_count: 37
  coord_distance_improved_count: 71
  rank_improved_count: 108
```

Coordinate-basin decomposition artifacts:

```text
2157 random_denoise:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coord_basin_decomposition/v5_2157_random_residual_surgery_l20_l23_l24

12670 sorted_denoise:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coord_basin_decomposition/v5_12670_sorted_residual_surgery_l20_l23_l24
```

Coordinate-basin read:

```text
2157 random_denoise, 126 rows:
  baseline families:
    exact_target: 7
    far_from_target: 56
    low_border: 42
    near_target_radius32: 21
  patched families:
    exact_target: 32
    far_from_target: 35
    low_border: 33
    near_target_radius1: 5
    near_target_radius4: 7
    near_target_radius8: 2
    near_target_radius16: 2
    near_target_radius32: 10
  entered_near_target_radius16_count: 41
  left_border_count: 30
  target_rank_improved_count: 102

12670 sorted_denoise, 126 rows:
  baseline families:
    far_from_target: 91
    low_border: 35
  patched families:
    exact_target: 37
    far_from_target: 66
    low_border: 16
    near_target_radius1: 1
    near_target_radius4: 2
    near_target_radius8: 3
    near_target_radius16: 1
  entered_near_target_radius16_count: 44
  left_border_count: 27
  target_rank_improved_count: 108
```

Layer read:

```text
2157 random_denoise:
  L20: exact=11, entered_near16=13, left_border=0, rank_improved=36
  L23: exact=11, entered_near16=11, left_border=11, rank_improved=30
  L24: exact=10, entered_near16=17, left_border=19, rank_improved=36

12670 sorted_denoise:
  L20: exact=11, entered_near16=13, left_border=0, rank_improved=36
  L23: exact=11, entered_near16=14, left_border=15, rank_improved=36
  L24: exact=15, entered_near16=17, left_border=12, rank_improved=36
```

Alpha read:

```text
2157 random_denoise:
  alpha 0.0: exact=1, near16=0, left_border=0
  alpha 0.005: exact=1, near16=0, left_border=2
  alpha 0.01: exact=2, near16=1, left_border=4
  alpha 0.02: exact=2, near16=4, left_border=6
  alpha 0.05: exact=4, near16=7, left_border=6
  alpha 0.1: exact=10, near16=14, left_border=6
  alpha 0.2: exact=12, near16=15, left_border=6

12670 sorted_denoise:
  alpha 0.0: exact=0, near16=0, left_border=0
  alpha 0.005: exact=0, near16=0, left_border=2
  alpha 0.01: exact=0, near16=2, left_border=5
  alpha 0.02: exact=2, near16=3, left_border=5
  alpha 0.05: exact=4, near16=7, left_border=5
  alpha 0.1: exact=16, near16=16, left_border=5
  alpha 0.2: exact=15, near16=16, left_border=5
```

Representative flips:

```text
2157 random_denoise:
  L20 object 6 target coord_170 alpha 0.1:
    top1 197 -> 170, rank 634 -> 1
  L20 object 5 target coord_249 alpha 0.1:
    top1 221 -> 249, rank 140 -> 1
  L24 object 11 target coord_302 alpha 0.01:
    top1 354 -> 302, rank 7 -> 1

12670 sorted_denoise:
  L23 object 13 target coord_269 alpha 0.05:
    top1 308 -> 269, rank 114 -> 1
  L20 object 12 target coord_519 alpha 0.1:
    top1 446 -> 519, rank 658 -> 1
  L24 object 13 target coord_269 alpha 0.02:
    top1 354 -> 269, rank 19 -> 1
```

Mechanism update:

```text
The exact top-six rows for 2157 and 12670 are locally movable at L20/L23/L24.
This strengthens the interpretation that many false negatives and duplicate
onsets are not caused by complete absence of visual coordinate evidence. The
more likely failure is route or basin capture: the hidden state carries enough
recoverable coordinate information, but the next-token readout is attracted to
border/default, repeated-anchor, or stale local-history basins.

12670 sorted_denoise is especially important because it is highly movable by
readout surgery despite being the denoising checkpoint's failure side for that
sample base. That makes it a strong candidate for full activation patching and
component localization: if the real forward stream can realize the same
direction, then prefix denoising shaped a fragile coordinate-basin router rather
than simply improving or degrading object perception.
```

Sample-base gate for the next round:

```text
Do not spend primary time on normal or well-learned images. Broad metrics and
population scans are only a selector, not the main research object.

Primary sample bases should satisfy most of:
  - same image has divergent behavior across sorted_denoise/random_denoise or
    against a prior CE baseline;
  - atlas transition families reverse or separate by model;
  - rows include false-negative, duplication-burst, over-emission, or premature
    stop pressure;
  - L20/L23/L24/L27 show rank-good but basin-wrong states, border escape, or
    late-lock disagreement;
  - there are enough repeated states within the image to test local-history and
    duplicate-basin hypotheses;
  - manual visual review would plausibly teach us something about perception,
    binding, grounding preparation, span emission, or transition to next object.

Current primary sample bases:
  2157: random_denoise coordinate-basin dissociation vs sorted_denoise late lock.
  12670: sorted_denoise coordinate-basin dissociation vs random_denoise late lock.

Current secondary sample base:
  2685: valuable for duplicate burst, semantic-anchor, and coordinate-tail
  coherence, but less clean for the first coordinate-basin escape surgery.
```

Next deterministic action:

```text
Run full-forward activation patching, not only readout-space surgery, on the
exact top-six rows above. Start with 2157 and 12670 at L20/L23/L24, then split
by component site if the full forward patch reproduces the basin escape:

  1. residual stream patch at decision token;
  2. attention output patch;
  3. MLP output patch;
  4. selected attention-head value-region patch if attention output is causal.

Readouts:
  - true next-token logits;
  - coordinate top1 and target rank;
  - basin family;
  - whether border/default top1 is escaped;
  - patched short continuation tail coherence;
  - whether duplicate suppression and object advancement improve or merely move
    the first coordinate token.
```

## 2026-06-26 update - full-forward top-six patching and tail tolerance

Full-forward activation patching was run on the exact top-six coordinate-basin
dissociation rows selected above. This is the real forward path: perturbations
are inserted into decoder layer outputs at the assistant-prefix decision token,
then true next-token logits and short greedy continuations are read.

Activation-patch artifacts:

```text
2157 random_denoise top-six:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v16_2157_random_top6_l20_l23_l24_gpu0

12670 sorted_denoise top-six:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v16_12670_sorted_top6_l20_l23_l24_gpu1
```

Both jobs used:

```text
source rows: exact top-six source_state_keys
probe layers: 20,23,24
patch site: layer_output
alphas: 0,0.01,0.02,0.05,0.1
continuation_steps: 6
training ran: false
```

Coordinate-basin decomposition artifacts:

```text
2157 random_denoise:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coord_basin_decomposition/v6_2157_random_activation_patch_l20_l23_l24

12670 sorted_denoise:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coord_basin_decomposition/v6_12670_sorted_activation_patch_l20_l23_l24
```

Full-forward result:

```text
2157 random_denoise, 90 rows:
  readout_status_counts: ok=90
  baseline basin families: low_border=90
  patched basin families:
    exact_target: 19
    low_border: 64
    near_target_radius1: 3
    near_target_radius8: 1
    near_target_radius16: 2
    far_from_target: 1
  entered_near_target_radius16_count: 25
  left_border_count: 26
  rank_improved_count: 51
  continuation labels:
    coord_tail_coherent: 3
    first_token_only: 16
    first_token_not_repaired: 71

12670 sorted_denoise, 90 rows:
  readout_status_counts: ok=90
  baseline basin families: low_border=90
  patched basin families:
    exact_target: 25
    low_border: 60
    near_target_radius8: 3
    near_target_radius16: 1
    far_from_target: 1
  entered_near_target_radius16_count: 29
  left_border_count: 30
  rank_improved_count: 44
  continuation labels:
    coord_tail_coherent: 5
    first_token_only: 20
    first_token_not_repaired: 65
```

Layer and alpha read:

```text
2157 random_denoise:
  L20: exact_first=3, exact_tail=0, rank_improved=15
  L23: exact_first=8, exact_tail=1, rank_improved=18
  L24: exact_first=8, exact_tail=2, rank_improved=18
  alpha 0.0/0.01/0.02: exact_first=0
  alpha 0.05: exact_first=7, exact_tail=1
  alpha 0.1: exact_first=12, exact_tail=2

12670 sorted_denoise:
  L20: exact_first=4, exact_tail=1, rank_improved=11
  L23: exact_first=11, exact_tail=2, rank_improved=17
  L24: exact_first=10, exact_tail=2, rank_improved=16
  alpha 0.0/0.01/0.02: exact_first=0
  alpha 0.05: exact_first=11, exact_tail=2
  alpha 0.1: exact_first=14, exact_tail=3
```

Tail-tolerance reducer:

```text
module:
  src/analysis/prefix_denoising_surgery_probing/activation_tail_tolerance.py

test:
  tests/analysis/test_prefix_denoising_activation_tail_tolerance.py

artifacts:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_tail_tolerance/v1_2157_random_activation_patch_l20_l23_l24
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_tail_tolerance/v1_12670_sorted_activation_patch_l20_l23_l24
```

Why the reducer matters:

```text
The strict continuation label only counts exact y1/x2/y2/box_end tails. Some
"first_token_only" rows actually emit the correct x1 and then a near-matching
y1/x2/y2 tail, for example within 4 or 8 coordinate bins. The tolerance reducer
therefore separates:

  - first coordinate can be moved;
  - full box-span program is exactly coherent;
  - full box-span program is near coherent;
  - first token moves but tail remains a different object/anchor program.
```

Tail-tolerance result:

```text
2157 random_denoise:
  first_token_target_count: 19
  exact_tail_complete_count: 3
  tail_complete_tol4: 8
  tail_complete_tol8: 9
  tail_complete_tol16: 9
  object read:
    obj6: exact/tol8 = 3/3
    obj8: exact/tol8 = 0/5
    obj5: exact/tol8 = 0/1
    obj11: exact/tol8 = 0/0
    obj12: exact/tol8 = 0/0
    obj9: exact/tol8 = 0/0

12670 sorted_denoise:
  first_token_target_count: 25
  exact_tail_complete_count: 5
  tail_complete_tol4: 10
  tail_complete_tol8: 19
  tail_complete_tol16: 20
  object read:
    obj12: exact/tol8 = 5/5
    obj8: exact/tol8 = 0/5
    obj4: exact/tol8 = 0/5
    obj7: exact/tol8 = 0/4
    obj18: exact/tol8 = 0/0
    obj13: exact/tol8 = 0/0
```

Representative continuation examples:

```text
2157 object 6, L24 alpha 0.05:
  baseline:
    coord_0, coord_157, coord_93, coord_295, box_end, object_ref_start
  patched:
    coord_170, coord_38, coord_263, coord_388, box_end, object_ref_start
  expected followup:
    coord_38, coord_263, coord_388, box_end

2157 object 8, L23 alpha 0.05:
  baseline:
    coord_0, coord_165, coord_93, coord_296, box_end, object_ref_start
  patched:
    coord_92, coord_119, coord_170, coord_395, box_end, object_ref_start
  expected followup:
    coord_116, coord_171, coord_394, box_end
  interpretation:
    x1 is repaired and the remaining box is near-coherent, not exact.

12670 object 12, L23 alpha 0.05:
  baseline:
    coord_0, coord_335, coord_105, coord_648, box_end, object_ref_start
  patched:
    coord_519, coord_296, coord_641, coord_546, box_end, object_ref_start
  expected followup:
    coord_296, coord_641, coord_546, box_end

12670 object 4, L23 alpha 0.05:
  baseline:
    coord_0, coord_187, coord_73, coord_415, box_end, object_ref_start
  patched:
    coord_394, coord_128, coord_528, coord_558, box_end, object_ref_start
  expected followup:
    coord_128, coord_526, coord_552, box_end
  interpretation:
    x1 is repaired and the remaining box is near-coherent within 8 bins.
```

Mechanism update:

```text
Readout-space surgery overstated how much of the repair is available inside the
real forward stream. Full-forward layer-output patching confirms that the
border basin can be causally escaped, but it usually repairs the next coordinate
token before it repairs the whole object-span program.

The mechanism is therefore at least two-stage:

  1. coordinate-basin escape / first-token router:
     L23/L24, alpha >= 0.05, can move many low-border x1 decisions to the
     target coordinate.

  2. object-span continuation program:
     only some object states have y1/x2/y2/box_end already aligned behind the
     repaired x1. Others become near-coherent, and some remain tied to a stale
     anchor/duplicate basin.

This is a stronger and deeper picture than "model sees or does not see the
object": perception/grounding evidence can be enough to move the first slot,
while autoregressive binding of the remaining box span is separately fragile.
```

Next deterministic action:

```text
Run component-site decomposition over the same high-value source states, but
start at L23/L24 and alpha 0.05/0.1 where full-forward layer-output patching
actually crosses the first-token basin:

  - layer_input;
  - self_attn;
  - mlp.

Split the read by object classes:
  - exact coherent object programs: 2157 obj6, 12670 obj12;
  - near-coherent object programs: 2157 obj8, 12670 obj4/7/8;
  - first-token repair without near tail: 2157 obj11/12, 12670 obj13;
  - hard non-repair rows: 2157 obj9, 12670 obj18.

The question is whether attention, MLP, or residual-input sites own:
  - border escape;
  - exact target x1;
  - near-tail object-span coherence;
  - stale-anchor tail persistence.
```

## 2026-06-26 update - component-site decomposition at crossing layers

Component-site activation patching was run on the same exact top-six source
states, restricted to the regime where full layer-output patching already
crossed the first-token basin:

```text
layers: 23,24
alphas: 0.05,0.1
patch sites: layer_input,self_attn,mlp
continuation_steps: 6
training ran: false
```

Artifacts:

```text
activation patch:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v17_2157_random_component_sites_l23_l24_gpu2
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v17_12670_sorted_component_sites_l23_l24_gpu3

coordinate-basin decomposition:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coord_basin_decomposition/v7_2157_random_component_sites_l23_l24
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/coord_basin_decomposition/v7_12670_sorted_component_sites_l23_l24

tail tolerance:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_tail_tolerance/v2_2157_random_component_sites_l23_l24
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_tail_tolerance/v2_12670_sorted_component_sites_l23_l24
```

Matched layer-output reference from `v16`:

```text
2157 random_denoise, L23/L24 alpha 0.05/0.1:
  rows: 24
  exact_first: 16
  exact_tail: 3
  rank_improved: 22

12670 sorted_denoise, L23/L24 alpha 0.05/0.1:
  rows: 24
  exact_first: 21
  exact_tail: 4
  rank_improved: 24
```

Component-site result:

```text
2157 random_denoise, 72 rows:
  baseline basin families: low_border=72
  patched basin families:
    exact_target: 22
    low_border: 44
    near_target_radius1: 4
    near_target_radius16: 2
  entered_near_target_radius16_count: 28
  left_border_count: 28
  rank_improved_count: 51
  first_token_target_count: 22
  exact_tail_count: 2
  tail_tol4_count: 8
  tail_tol8_count: 10

12670 sorted_denoise, 72 rows:
  baseline basin families: low_border=72
  patched basin families:
    exact_target: 33
    low_border: 37
    near_target_radius8: 2
  entered_near_target_radius16_count: 35
  left_border_count: 35
  rank_improved_count: 46
  first_token_target_count: 33
  exact_tail_count: 5
  tail_tol4_count: 13
  tail_tol8_count: 24
```

Patch-site split:

```text
2157 random_denoise:
  layer_input:
    rows=24, exact_first=14, exact_tail=1, rank_improved=21
  mlp:
    rows=24, exact_first=8, exact_tail=1, rank_improved=19
  self_attn:
    rows=24, exact_first=0, exact_tail=0, rank_improved=11

12670 sorted_denoise:
  layer_input:
    rows=24, exact_first=20, exact_tail=3, rank_improved=23
  mlp:
    rows=24, exact_first=13, exact_tail=2, rank_improved=17
  self_attn:
    rows=24, exact_first=0, exact_tail=0, rank_improved=6
```

Layer split:

```text
2157 random_denoise:
  layer_input L23/L24: exact_first=6/8, exact_tail=0/1
  mlp L23/L24: exact_first=4/4, exact_tail=1/0
  self_attn L23/L24: exact_first=0/0, exact_tail=0/0

12670 sorted_denoise:
  layer_input L23/L24: exact_first=9/11, exact_tail=1/2
  mlp L23/L24: exact_first=7/6, exact_tail=1/1
  self_attn L23/L24: exact_first=0/0, exact_tail=0/0
```

Object/tail read:

```text
2157 random_denoise:
  obj6: exact_tail=2, tol8=2
  obj8: exact_tail=0, tol8=6
  obj5: exact_tail=0, tol8=2
  obj11/obj12: exact_first without near-tail
  obj9: hard non-repair

12670 sorted_denoise:
  obj12: exact_tail=5, tol8=5
  obj8: exact_tail=0, tol8=8
  obj4: exact_tail=0, tol8=5
  obj7: exact_tail=0, tol8=6
  obj13: exact_first without near-tail
  obj18: weak/hard tail
```

Mechanism update:

```text
The crossing signal is not carried by isolated self-attention output at these
sites. `self_attn` patching improves target rank in some rows, but never flips
the low-border x1 token to the target coordinate. The first-token border escape
is strongest when patching the layer input, and partially reproduced by the MLP
output.

This suggests a residual/MLP-mediated coordinate-basin router at L23/L24:

  - the incoming residual state already contains the actionable direction;
  - MLP output can amplify or project part of that direction into the readout
    basin;
  - attention output alone is not sufficient to cross the border basin at the
    decision token, although it may still contribute upstream or through
    earlier value routing.

The full object-span program remains separate. `layer_input` and `mlp` can move
many x1 decisions, but exact y1/x2/y2/box_end coherence is still concentrated
in a few object states. Near-tail coherence is broader, especially for 12670,
which means the model often has a nearby box program but not always the exact
bound object program.
```

Next deterministic action:

```text
Focus on tensor-flow around L20->L24 rather than only component output at one
layer:

  1. compare incoming residual directions for exact-tail, near-tail, and
     hard-nonrepair objects;
  2. measure whether MLP transforms increase target-coordinate margin more than
     they increase stale-anchor tail compatibility;
  3. inspect earlier attention/value routing only as upstream preparation,
     because isolated L23/L24 self-attention output did not cross the basin;
  4. use 2157 obj6 vs obj8 and 12670 obj12 vs obj4/8 as paired exact-tail vs
     near-tail contrasts inside the same image.
```

## 2026-06-26 update - x1 component-flow contrast reducer

Added a matched reducer that joins layer-output and component-site activation
patch rows by sample/object/state/layer/alpha. It is x1-specific and designed
for the current high-value sample bases rather than the older y2 route stack.

```text
module:
  src/analysis/prefix_denoising_surgery_probing/x1_component_flow_contrast.py

test:
  tests/analysis/test_prefix_denoising_x1_component_flow_contrast.py

input layer-output rows:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v16_2157_random_top6_l20_l23_l24_gpu0/activation_patch_rows.jsonl
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v16_12670_sorted_top6_l20_l23_l24_gpu1/activation_patch_rows.jsonl

input component rows:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v17_2157_random_component_sites_l23_l24_gpu2/activation_patch_rows.jsonl
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v17_12670_sorted_component_sites_l23_l24_gpu3/activation_patch_rows.jsonl

matched output artifacts:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_component_flow_contrast/v2_2157_random_l23_l24_alpha005_01_matched
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_component_flow_contrast/v2_12670_sorted_l23_l24_alpha005_01_matched
```

The reducer classifies each matched row by the layer-output result:

```text
exact_tail_program:
  x1 repaired and y1/x2/y2/box_end exactly match the expected continuation.

near_tail_program:
  x1 repaired and y1/x2/y2/box_end are within the configured tolerance
  (8 coordinate bins here), but not exact.

x1_only_stale_tail:
  x1 repaired, but the continuation tail remains outside the near-tail basin.

hard_nonrepair:
  layer-output patch does not repair x1.
```

2157 random_denoise matched flow read:

```text
rows: 24
object_flow_label counts:
  exact_tail_program: 3
  near_tail_program: 5
  x1_only_stale_tail: 8
  hard_nonrepair: 8

component_owner_label counts:
  residual_and_mlp_crossing: 7
  residual_input_only_crossing: 7
  layer_output_only_crossing: 2
  mlp_only_crossing: 1
  no_component_crossing: 7

site first-token target counts:
  layer_output: 16
  layer_input: 14
  mlp: 8
  self_attn: 0

site near-tail counts:
  layer_output: 8
  layer_input: 7
  mlp: 3
  self_attn: 0
```

Object-level read for 2157:

```text
obj6:
  exact_tail rows: 3/4
  layer_output_first: 3
  layer_input_first: 1
  mlp_first: 1
  owner mix:
    layer_output_only, mlp_only, residual_input_only, no_component

obj8:
  near_tail rows: 4/4
  layer_output_first: 4
  layer_input_first: 4
  mlp_first: 2
  owner mix:
    residual_input_only=2, residual_and_mlp=2

obj11/obj12:
  x1_only_stale_tail rows: 8/8 combined
  layer_input and MLP often cross x1, but tail remains stale.

obj9:
  hard_nonrepair rows: 4/4
  no component crossing.
```

12670 sorted_denoise matched flow read:

```text
rows: 24
object_flow_label counts:
  exact_tail_program: 4
  near_tail_program: 12
  x1_only_stale_tail: 5
  hard_nonrepair: 3

component_owner_label counts:
  residual_and_mlp_crossing: 13
  residual_input_only_crossing: 7
  layer_output_only_crossing: 3
  no_component_crossing: 1

site first-token target counts:
  layer_output: 21
  layer_input: 20
  mlp: 13
  self_attn: 0

site near-tail counts:
  layer_output: 16
  layer_input: 14
  mlp: 10
  self_attn: 0
```

Object-level read for 12670:

```text
obj12:
  exact_tail rows: 4/4
  layer_output_first: 4
  layer_input_first: 3
  mlp_first: 2

obj4/obj7/obj8:
  near_tail rows: 12/12 combined
  obj8 is strongest: layer_input_first=4 and mlp_first=4.

obj13:
  x1_only_stale_tail rows: 4/4
  residual and MLP can cross x1, but tail remains unbound.

obj18:
  mixed weak/hard rows; useful for hard-row contrast but not as clean as obj13.
```

Mechanism update:

```text
The x1 component-flow contrast makes the residual/MLP split more precise:

  - self_attn is still zero for first-token crossing in the matched panel;
  - layer_input is nearly sufficient for most x1 basin escapes;
  - MLP is a partial amplifier/projector, especially in 12670 near-tail rows;
  - exact-tail vs near-tail vs stale-tail is not explained by x1 crossing alone.

The deepest current picture is a three-part program:

  1. incoming residual holds an actionable coordinate-basin direction;
  2. MLP can amplify that direction into the x1 readout basin;
  3. the object-span tail program is separately bound or stale.

This directly connects false-negative/duplication-burst mechanics: duplicated
or missing objects can arise when the model has enough x1 coordinate evidence
to leave the border basin, but the autoregressive tail remains attached to a
nearby or stale object program instead of the intended object.
```

Next deterministic action:

```text
Do a narrow hidden-vector/tensor-flow capture, not just row reduction, for four
paired object states:

  2157 obj6 exact-tail vs obj8 near-tail;
  12670 obj12 exact-tail vs obj8 near-tail;
  optionally add 2157 obj11 and 12670 obj13 as stale-tail controls.

For each state, capture L20/L23/L24:
  - layer input;
  - MLP output/update;
  - layer output;
  - projection onto target-vs-border direction;
  - projection onto expected-tail continuation directions if cheap.

The specific question:
  what vector feature separates "x1 repaired and tail exact" from "x1 repaired
  but tail only near/stale" inside the same image and checkpoint?
```

## 2026-06-26 update - x1 hidden-vector flow lens

Added a narrow hidden-vector lens for the exact-vs-near object pairs. The runner
loads the checkpoint, replays the original prefix with the existing atlas
forward path, captures component tensors at the decision token, then projects
each captured vector through the final language norm / lm-head direction:

```text
target_coord_token - coord_0_token
```

This is not itself causal surgery. It is a selected-state lens over the same
states where causal activation patching already established border-basin escape.

```text
module:
  src/analysis/prefix_denoising_surgery_probing/x1_hidden_vector_flow.py

test:
  tests/analysis/test_prefix_denoising_x1_hidden_vector_flow.py
```

Primary exact-vs-near capture artifacts:

```text
2157 random_denoise:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_hidden_vector_flow/v1_2157_obj6_obj8_l20_l23_l24_gpu0

12670 sorted_denoise:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_hidden_vector_flow/v1_12670_obj12_obj8_l20_l23_l24_gpu1
```

Both jobs used:

```text
selected source rows: 2
layers: 20,23,24
components:
  layer_input_state
  attention_update
  after_attention_state
  mlp_update
  layer_output_state
  layer_delta_update
rows per job: 36
flow_status_counts: ok=36
training ran: false
```

2157 random_denoise, obj6 exact-tail vs obj8 near-tail:

```text
obj6 exact_tail_program:
  layer_input_state margin:
    L20 -0.351668, L23 +0.046110, L24 +0.592698
  attention_update margin:
    L20 -2.924863, L23 +2.103571, L24 -0.693120
  mlp_update margin:
    L20 -1.637420, L23 +0.583879, L24 -2.966684
  layer_output_state margin:
    L20 -1.292117, L23 +0.592698, L24 -1.019776
  layer_delta_update margin:
    L20 -2.283001, L23 +1.061890, L24 -3.035031

obj8 near_tail_program:
  layer_input_state margin:
    L20 -0.144572, L23 -0.606819, L24 -0.045186
  attention_update margin:
    L20 -3.310044, L23 +0.152483, L24 +0.885257
  mlp_update margin:
    L20 -1.899554, L23 +0.868070, L24 -0.253843
  layer_output_state margin:
    L20 -1.251953, L23 -0.045186, L24 -0.015256
  layer_delta_update margin:
    L20 -2.620424, L23 +0.883603, L24 +0.041660
```

2157 interpretation:

```text
The exact-tail object does not have a uniformly stronger target-vs-border
margin than the near-tail object. Both show positive mid-layer updates, and the
L24 layer-output margin is not a simple exact-tail separator. The exact-vs-near
difference for 2157 likely depends on tail-binding geometry or stale-tail
compatibility, not only the x1 target-vs-border direction.
```

12670 sorted_denoise, obj12 exact-tail vs obj8 near-tail:

```text
obj12 exact_tail_program:
  layer_input_state margin:
    L20 -1.531821, L23 +0.215327, L24 +0.554123
  attention_update margin:
    L20 -1.738663, L23 +2.742114, L24 +4.378156
  mlp_update margin:
    L20 +2.240943, L23 +0.073836, L24 +3.191598
  layer_output_state margin:
    L20 -0.573669, L23 +0.554123, L24 +2.529778
  layer_delta_update margin:
    L20 +1.695747, L23 +0.680660, L24 +4.401669

obj8 near_tail_program:
  layer_input_state margin:
    L20 -0.552856, L23 -0.427061, L24 +0.653851
  attention_update margin:
    L20 -2.562607, L23 +2.986615, L24 +0.637557
  mlp_update margin:
    L20 -1.952716, L23 +1.148055, L24 +0.080970
  layer_output_state margin:
    L20 -1.597056, L23 +0.653851, L24 +0.712237
  layer_delta_update margin:
    L20 -2.613567, L23 +1.802028, L24 +0.310241
```

12670 interpretation:

```text
Here the exact-tail object has a clear late positive L24 amplification:

  obj12 exact:
    L24 attention_update +4.378156
    L24 mlp_update +3.191598
    L24 layer_output_state +2.529778
    L24 layer_delta_update +4.401669

  obj8 near:
    L24 attention_update +0.637557
    L24 mlp_update +0.080970
    L24 layer_output_state +0.712237
    L24 layer_delta_update +0.310241

For 12670, exact-tail coherence appears aligned with a much stronger late
positive target-vs-border flow, especially at L24. The near-tail object has
enough margin to escape the border basin but not the same late amplification.
```

Mechanism update:

```text
The hidden-vector lens supports a split mechanism, but not a one-size-fits-all
scalar rule:

  - x1 border escape can occur with modest positive margin;
  - exact tail coherence may require stronger late positive flow in some
    samples, especially 12670;
  - 2157 shows that exact-vs-near tail quality can depend on something beyond
    the target-vs-border x1 direction, likely the compatibility between the
    repaired x1 and the bound y1/x2/y2 continuation program.

Current best hypothesis:

  coordinate-basin formation is residual/MLP-mediated, while object-span
  binding is an additional program whose quality is only partially predicted by
  x1 target-vs-border margin. The missing core variable is probably a
  multi-slot/tail-direction alignment feature, not only the x1 coordinate
  direction.
```

Next deterministic action:

```text
Extend the hidden-vector lens from x1 target-vs-border to tail-direction
alignment:

  - compute projections for y1, x2, y2 expected continuation tokens;
  - compare exact-tail obj6/obj12 to near-tail obj8 within the same samples;
  - add stale-tail controls obj11/obj13 if the tail-direction lens is cheap.

This should test whether exact-tail states already carry a coherent multi-slot
box program before x1 emission, while near/stale states only carry the first
coordinate escape direction.
```

## 2026-06-26: tail-slot hidden vector flow at pre-x1

Implementation slice:

```text
src/analysis/prefix_denoising_surgery_probing/x1_hidden_vector_flow.py
tests/analysis/test_prefix_denoising_x1_hidden_vector_flow.py
```

The x1 hidden-vector flow lens now has an optional multi-slot projection mode:

```text
--projection-slots tail        # y1,x2,y2
--projection-slots all         # x1,y1,x2,y2
--projection-bbox-field generated_bbox_bins | matched_gt_bbox_bins
--antagonist-coord-bin 0
```

The legacy no-flag behavior is preserved: project only the selected row's
`target_next_coord_bin` against the antagonist coordinate.

Verification:

```text
pytest tests/analysis/test_prefix_denoising_x1_hidden_vector_flow.py -q
  8 passed for the first tail-slot slice; later trajectory/competitor coverage
  extended this file to 10 focused tests.
python -m py_compile src/analysis/prefix_denoising_surgery_probing/x1_hidden_vector_flow.py
  pass
```

Dry-run selection:

```text
position rows:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/position_plan/v1_selected_bases_treatment_only/position_rows.jsonl

2157 random component-flow rows:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_component_flow_contrast/v2_2157_random_l23_l24_alpha005_01_matched/x1_component_flow_rows.jsonl
selected source rows: 3
objects: 5,6,8

12670 sorted component-flow rows:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_component_flow_contrast/v2_12670_sorted_l23_l24_alpha005_01_matched/x1_component_flow_rows.jsonl
selected source rows: 4
objects: 4,7,8,12
```

Real run artifacts:

```text
2157 random_denoise:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_hidden_vector_flow/v2_2157_random_tail_slots_l20_l23_l24_gpu0
  rows: 162
  ok: 162
  objects: obj5 near, obj6 exact, obj8 near
  slots: y1,x2,y2
  layers: 20,23,24

12670 sorted_denoise:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_hidden_vector_flow/v2_12670_sorted_tail_slots_l20_l23_l24_gpu1
  rows: 216
  ok: 216
  objects: obj4 near, obj7 near, obj8 near, obj12 exact
  slots: y1,x2,y2
  layers: 20,23,24
```

Key readout, L24 `layer_output_state` margins against coord_0:

```text
2157 random_denoise:
  obj5 near_tail_program:
    y1 +0.791696, x2 +2.496394, y2 +1.795105
  obj6 exact_tail_program:
    y1 +0.704001, x2 +0.786506, y2 -0.747578
  obj8 near_tail_program:
    y1 +0.330829, x2 -0.892786, y2 +0.106454

12670 sorted_denoise:
  obj4 near_tail_program:
    y1 +1.704335, x2 +1.337962, y2 +0.499472
  obj7 near_tail_program:
    y1 +1.115650, x2 -0.840649, y2 -0.149995
  obj8 near_tail_program:
    y1 +1.071336, x2 +0.387738, y2 -1.493864
  obj12 exact_tail_program:
    y1 +1.234072, x2 +0.478460, y2 +0.006135
```

Mechanism update:

```text
The pre-x1 tail-slot lens falsifies a simple static-box-plan hypothesis. Exact
full-tail continuation is not equivalent to "all y1/x2/y2 target directions are
already strongly positive at pre-x1 against coord_0".

Evidence:
  - 2157 obj6 is the exact-tail case but has negative pre-x1 y2 margin at L24.
  - 2157 obj5 is only near-tail but has strong positive L24 margins for all
    tail slots under this target-vs-border lens.
  - 12670 obj12 is exact-tail, but obj4 near-tail has stronger L24 y1/x2/y2
    layer-output margins than obj12 under this same readout.

Interpretation:
  - pre-x1 state may carry enough evidence for first-coordinate basin escape,
    but not a fully settled four-slot object program;
  - later slots are probably formed autoregressively after x1/y1/x2 emissions,
    via prefix-conditioned tensor-flow rather than a single static vector at
    object entry;
  - target-vs-border is also an incomplete antagonist choice for tail quality,
    because near-tail failures may be target-positive against coord_0 while
    still losing to a local competitor, stale anchor, or wrong slot basin.
```

Representative sample-base doctrine from the parallel diagnostician:

```text
Core next batch:
  2157, 12670, 19432, 16228, 2685, 2299, 19109, 17899
Alternates:
  14439, 18380

Reason:
  this panel covers FN guidance, duplication bursts, stale/first-token-only tail
  failures, termination/router cases, recoverable hidden-tail controls, and
  sorted/random divergence without collapsing to person-backpack only.
```

Next deterministic probe:

```text
Build a slot-position hidden-flow trajectory lens:
  - select the same object spans at post_x1_pre_y1, post_y1_pre_x2,
    post_x2_pre_y2 positions;
  - project only the actual next slot at each position;
  - compare whether exact-tail objects form positive slot directions only after
    prior coordinate emissions, while near/stale states lose to local/stale
    competitor basins;
  - add competitor-specific antagonists from observed near/stale generated bins
    rather than only coord_0.

This is more likely to expose the tensor-flow mechanism than additional pre-x1
aggregate target-vs-border margins.
```

## 2026-06-26: slot trajectory and local-competitor basin readout

Implementation extension:

```text
src/analysis/prefix_denoising_surgery_probing/x1_hidden_vector_flow.py
  added --trajectory-probe-positions
  added --antagonist-coord-bin-field

tests/analysis/test_prefix_denoising_x1_hidden_vector_flow.py
  focused coverage: 10 passed
```

Slot trajectory mode keeps the same pre-x1 object selection from x1 component
flow labels, but expands selected objects to later autoregressive positions:

```text
post_x1_pre_y1
post_y1_pre_x2
post_x2_pre_y2
```

Artifacts:

```text
Target-vs-border, slot-at-due-prefix:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_hidden_vector_flow/v3_2157_random_slot_trajectory_l20_l23_l24_gpu0
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_hidden_vector_flow/v3_12670_sorted_slot_trajectory_l20_l23_l24_gpu1

Local competitor position rows:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/local_competitor_position_rows/v1_2157_random_from_v16_layer_output
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/local_competitor_position_rows/v1_12670_sorted_from_v16_layer_output

Target-vs-local-competitor, slot-at-due-prefix:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_hidden_vector_flow/v4_2157_random_slot_local_competitor_l20_l23_l24_gpu0
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_hidden_vector_flow/v4_12670_sorted_slot_local_competitor_l20_l23_l24_gpu1
```

Slot trajectory result:

```text
The actual native trajectory strongly forms each due slot by L24. Examples:

2157 random_denoise, L24 layer_output_state target-vs-border:
  obj6 exact:
    y1 +2.133102, x2 +4.010159, y2 +4.833184
  obj8 near:
    y1 +2.275681, x2 +4.386784, y2 +3.956804

12670 sorted_denoise, L24 layer_output_state target-vs-border:
  obj12 exact:
    y1 +3.625536, x2 +3.438668, y2 +1.141093
  obj8 near:
    y1 +1.782376, x2 +7.724751, y2 +2.986004
```

Interpretation:

```text
The model's native emitted object span is internally self-consistent by the time
each coordinate slot is due. This supports a sequential slot-formation story:
later coordinates are not fully present at pre-x1, but they are built after
previous coordinate emissions.

However, target-vs-border at the due slot still does not separate exact-tail
from near-tail. Near-tail cases can have very strong positive exact-slot margin
against coord_0.
```

Local competitor result:

```text
The patched continuation rows expose the actual near/stale competitor tokens.
Examples:

2157 random:
  obj8 expected tail:
    y1/x2/y2 = 116/171/394
  patched near tail:
    y1/x2/y2 = 119/170/395

12670 sorted:
  obj4 expected tail:
    y1/x2/y2 = 128/526/552
  patched near tail:
    y1/x2/y2 = 128/528/558
```

Target-vs-local-competitor margins at L24 layer_output_state:

```text
2157 random_denoise:
  obj5 near, y2 414 vs competitor 421: +0.232666
  obj8 near, y1 116 vs competitor 119: +0.103345
  obj8 near, x2 171 vs competitor 170: -0.523640
  obj8 near, y2 394 vs competitor 395: -0.258501

12670 sorted_denoise:
  obj4 near, x2 526 vs competitor 528: +0.263398
  obj4 near, y2 552 vs competitor 558: -1.987744
  obj7 near, x2 421 vs competitor 415: -0.264581
  obj7 near, y2 605 vs competitor 602: -0.421293
  obj8 near, y1 167 vs competitor 165: -0.475112
  obj8 near, x2 419 vs competitor 415: +0.831455
  obj8 near, y2 609 vs competitor 605: +0.274783
```

Mechanism update:

```text
This is the strongest lens in this slice.

Near-tail failure is not well explained by target-vs-border weakness. The same
states can be target-positive against coord_0 while only weakly positive or
negative against the locally realized competitor token. The collapse basin is a
local coordinate-neighbor/stale-tail basin, not a broad invalid-coordinate or
border-basin failure.

The current best picture:
  1. x1 repair can move the object entry into a plausible coordinate basin.
  2. Subsequent autoregressive slots are generated by slot-local dynamics.
  3. Exact binding succeeds only when the slot-local state beats the nearest
     competitor/stale anchor, not merely when it beats coord_0.
  4. Prefix denoising appears to make native slot formation strong, but does not
     guarantee counterfactual tail rebinding after x1 surgery.
```

Next deterministic probe:

```text
Move from readout to counterfactual hidden-state trajectory:
  - construct patched-prefix rows after the repaired x1 token;
  - capture hidden states at y1/x2/y2 under the patched prefix, not only under
    the native rollout prefix;
  - compare exact-tail obj6/obj12 against near-tail obj5/obj8/obj4/obj7 using
    target-vs-local-competitor margins.

This should directly test whether the x1 surgery causes the hidden state to
enter the neighbor basin before y1/x2/y2 emission.
```

## 2026-06-26: patched-prefix replay versus native slot trajectory

Implementation slice:

```text
src/analysis/prefix_denoising_surgery_probing/patched_prefix_trajectory_rows.py
tests/analysis/test_prefix_denoising_patched_prefix_trajectory_rows.py
```

The new row builder materializes counterfactual position rows from existing
activation-patch continuation outputs:

```text
pre_x1 native prefix + patched x1 token -> post_x1_pre_y1
pre_x1 native prefix + patched x1/y1 tokens -> post_y1_pre_x2
pre_x1 native prefix + patched x1/y1/x2 tokens -> post_x2_pre_y2
```

It also annotates each due slot with the locally emitted competitor token when
the patched continuation differs from the expected target token.

Generated row artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/patched_prefix_trajectory_rows/v1_2157_random_layer_output_from_v16
  rows: 24
  counterfactual rows: 18
  local competitor annotations: 9

/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/patched_prefix_trajectory_rows/v1_12670_sorted_layer_output_from_v16
  rows: 24
  counterfactual rows: 18
  local competitor annotations: 9
```

Patched-prefix hidden-flow artifacts:

```text
L20/L23/L24:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_hidden_vector_flow/v5_2157_random_patched_prefix_local_competitor_l20_l23_l24_gpu0
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_hidden_vector_flow/v5_12670_sorted_patched_prefix_local_competitor_l20_l23_l24_gpu1

L24/L25/L26/L27, native-prefix control:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_hidden_vector_flow/v6_2157_random_native_local_competitor_l24_l25_l26_l27_gpu0
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_hidden_vector_flow/v6_12670_sorted_native_local_competitor_l24_l25_l26_l27_gpu2

L24/L25/L26/L27, patched-prefix replay:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_hidden_vector_flow/v6_2157_random_patched_local_competitor_l24_l25_l26_l27_gpu1
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/x1_hidden_vector_flow/v6_12670_sorted_patched_local_competitor_l24_l25_l26_l27_gpu3
```

Key result:

```text
Text-replayed patched prefixes barely move target-vs-local-competitor margins
relative to native prefixes.

L27 layer_output_state patched-minus-native:
  2157 obj8 x2 171 vs 170: -0.010777
  2157 obj8 y2 394 vs 395: +0.002466
  12670 obj7 y2 605 vs 602: -0.049824
  12670 obj8 x2 419 vs 415: -0.038389
  12670 obj8 y2 609 vs 605: -0.075252
```

Late-layer local-competitor margins remain narrow:

```text
2157 native L27 layer_output_state:
  obj5 y2 414 vs 421: -0.099348
  obj8 y1 116 vs 119: +0.067398
  obj8 x2 171 vs 170: +0.164947
  obj8 y2 394 vs 395: -0.084455

12670 native L27 layer_output_state:
  obj4 x2 526 vs 528: -0.106932
  obj4 y2 552 vs 558: -0.317910
  obj7 x2 421 vs 415: +0.198397
  obj7 y2 605 vs 602: -0.408046
  obj8 y1 167 vs 165: +0.347246
  obj8 x2 419 vs 415: +0.123949
  obj8 y2 609 vs 605: +0.443189
```

Mechanism update:

```text
The neighbor-basin picture survives the late-layer read: exact target and local
competitor are separated by small margins, often with sign changes across late
layers. This is a narrow local decision, not a broad target-vs-border failure.

But patched-prefix text replay does not reproduce the full causal path of the
manual incremental continuation. Some text-replayed L27 margins favor the exact
target even though the original patched greedy continuation emitted the local
competitor. Therefore, the current replay lens is useful as a compatibility
readout, but not sufficient as the final causal trace.
```

Next deterministic probe:

```text
Capture hidden states inside the manual incremental continuation loop used by
activation_patch_continuation.py:
  - after patched x1 logits select the first token;
  - before y1, x2, and y2 emissions;
  - with the same appended-token model_inputs path as the actual greedy
    continuation, not re-rendered assistant text;
  - project those incremental hidden states against exact slot tokens and the
    realized local competitors.

This should resolve whether the residual stream actually favors the local
competitor at generation time, or whether the discrepancy is caused by another
piece of decoding state/logit computation.
```

## 2026-06-26: exact appended-token continuation trace

Implementation slice:

```text
src/analysis/prefix_denoising_surgery_probing/appended_continuation_hidden_flow.py
tests/analysis/test_prefix_denoising_appended_continuation_hidden_flow.py
```

The new helper traces the same post-x1 path used by
`manual_greedy_continuation`: start from the pre-x1 model inputs, truncate at
the decision token, append the recorded activation-patch token ids, and then
capture hidden/component vectors at the token that predicts the next slot.
This avoids the text-detokenization/re-rendering ambiguity in the previous
patched-prefix replay.

Representative sample-base selection:

```text
2157 random_denoise:
  local-competitor slots selected: 4
  objects: 5, 8
  object 8 gives a full y1/x2/y2 local-neighbor tail collapse

12670 sorted_denoise:
  local-competitor slots selected: 7
  objects: 4, 7, 8
  object 8 gives a full y1/x2/y2 local-neighbor tail collapse
```

Artifacts:

```text
Dry runs:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/appended_continuation_hidden_flow/dryrun_2157_random_local_competitors
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/appended_continuation_hidden_flow/dryrun_12670_sorted_local_competitors

Exact appended-token traces, raw projection only:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/appended_continuation_hidden_flow/v1_2157_random_local_competitors_l24_l25_l26_l27_gpu0
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/appended_continuation_hidden_flow/v1_12670_sorted_local_competitors_l24_l25_l26_l27_gpu1

Exact appended-token traces, with effective-lm-head margins:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/appended_continuation_hidden_flow/v2_2157_random_local_competitors_effective_head_l24_l25_l26_l27_gpu0
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/appended_continuation_hidden_flow/v2_12670_sorted_local_competitors_effective_head_l24_l25_l26_l27_gpu1
```

Route validation:

```text
Every selected exact appended-token slot reproduced the recorded patched
continuation token as greedy next token.

2157 random_denoise:
  obj5 y2 414 vs local 421: greedy 421, target-local logit -0.250
  obj8 y1 116 vs local 119: greedy 119, target-local logit -0.250
  obj8 x2 171 vs local 170: greedy 170, target-local logit  0.000
  obj8 y2 394 vs local 395: greedy 395, target-local logit -0.125

12670 sorted_denoise:
  obj4 x2 526 vs local 528: greedy 528, target-local logit -0.125
  obj4 y2 552 vs local 558: greedy 558, target-local logit -0.125
  obj7 x2 421 vs local 415: greedy 415, target-local logit -0.125
  obj7 y2 605 vs local 602: greedy 602, target-local logit  0.000
  obj8 y1 167 vs local 165: greedy 165, target-local logit -0.125
  obj8 x2 419 vs local 415: greedy 415, target-local logit -0.125
  obj8 y2 609 vs local 605: greedy 605, target-local logit -0.125
```

Mechanism update:

```text
The text-replay ambiguity is resolved. Under the exact appended-token path, the
post-x1/y1/x2 continuation states really do decode to the recorded local
competitor. This strongly supports the local coordinate-neighbor basin picture:
after x1 repair, the later slot state can remain in a plausible but locally
wrong coordinate basin.

The failures are not broad invalid-token failures. They are tiny target-vs-near
competitor choices, sometimes exact ties by the rank convention but with greedy
tie/ordering landing on the locally emitted competitor.
```

Important lens correction:

```text
Raw lm-head-weight projections are not always exact final-logit explanations.
At L27 layer_output_state, raw target-vs-competitor projection can be positive
while the actual next-token target-local logit is negative.

Example:
  12670 obj8 y2 609 vs 605:
    raw L27 projection margin: +0.367937
    effective-lm-head margin:  -0.125000
    actual next-token margin:  -0.125000

The new v2 artifacts therefore include `effective_target_minus_antagonist_logit`
computed by calling the actual lm_head on the normalized captured vector. These
effective-head margins align with the actual final next-token margin for
L27 layer_output_state. Older raw projection rows remain useful as directional
diagnostics, but should not be treated as exact final-logit evidence under the
token-embeddings-adapter/effective-head surface.
```

Next deterministic probes:

```text
1. Add effective-lm-head margins to earlier hidden-flow lenses or regenerate
   the critical native/text-replay rows when making final claims.
2. Run the same exact appended-token trace on false-negative/termination sample
   bases to separate visual non-perception from language-side prefix fragility.
3. Expand representative sample-base selection beyond 2157 and 12670, including
   train-set failures, but keep the unit of deep analysis as image-base plus
   object trajectory rather than broad normal-case averaging.
```

## 2026-06-26: false-negative visual route versus language guidance reduction

Implementation slice:

```text
src/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_reduce.py
tests/analysis/test_prefix_denoising_fn_visual_value_region_reduce.py
```

The reducer is a post-hoc selector over existing FN visual value-region probe
rows. It groups by `(probe_model_id, image_id, target_desc)` and assigns a
conservative sample-base signature:

```text
language_guidance_likely:
  target coordinate is already high-rank under the contextual/prefix probe.

visual_route_sensitive:
  suppressing target-overlap visual-value regions strongly changes target rank.

competing_context_suppression_sensitive:
  suppressing non-target context/background strongly improves target rank.

weak_or_hidden_perception / mixed_or_inconclusive:
  weak or non-specific intervention evidence.
```

Important schema correction:

```text
For v3 FN rows, `failed_model_id` is not the executed probe model. The reducer
must prefer `probe_model_id`; otherwise random/sorted rows silently merge in
combined reductions. This is covered by a regression test.
```

Reduction artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_reduce/v2_v3_layer_sweep_random
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_reduce/v2_v3_layer_sweep_sorted
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_reduce/v2_v3_layer_sweep_combined
```

Combined v3 reduction:

```text
source rows: 1736
reduced rows: 8

signatures:
  language_guidance_likely: 1
  visual_route_sensitive: 4
  mixed_or_inconclusive: 3
```

High-value FN sample-base signatures:

```text
2157 / wine glass:
  sorted_denoise:
    baseline x1 target rank = 1
    target-overlap suppression worst delta = +485
    signature = language_guidance_likely
  random_denoise:
    baseline x1 target rank = 27
    target-overlap suppression worst delta = +217
    signature = visual_route_sensitive

12670 / person:
  sorted_denoise:
    baseline x1 target rank = 11
    target-overlap suppression worst delta = +94
    signature = visual_route_sensitive
  random_denoise:
    baseline x1 target rank = 613
    strongest non-target improvement = -87
    target/context sensitivity max abs delta = 154
    signature = visual_route_sensitive

2685 / bottle:
  random rank = 95, sorted rank = 71, max abs deltas only 11/10.
  This is a lower-priority weak/mixed case, not a good first deep slice.
```

Cross-checkpoint bridge evidence:

```text
Existing 2157 sorted->random bridge:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_bridge_probe/v2_smoke_2157_sorted_to_random_h8_gpu0

  random receiver baseline: target rank 27
  sorted donor replace at L16 h8 target_generated_overlap: 27 -> 2
  receiver remove at L16 h8 target_generated_overlap: 27 -> 244

New 12670 sorted->random bridge:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_bridge_probe/v3_12670_sorted_to_random_h8_target_overlap_gpu0

  random receiver baseline: target rank 613
  sorted donor replace at L16 h8 target_generated_overlap: 613 -> 59
  receiver remove at L16 h8 target_generated_overlap: 613 -> 721
```

Mechanism update:

```text
False negatives in these high-value cases are not simply "the model cannot see
the object." Target-overlap visual-value pathways are causal: removing the
receiver's own target-overlap pathway hurts, and replacing it with a stronger
sorted donor pathway improves the target coordinate.

The 2157 case is closest to a language/prefix-guidance fragility: sorted already
places the target x1 coordinate at rank 1 under the contextual probe, and the
donor value path makes random nearly recover the coordinate rank (27 -> 2).

The 12670 case is deeper: random's target coordinate is buried at rank 613, but
the sorted donor target-overlap value path still moves it to rank 59. This
suggests visual evidence exists but is too weakly synchronized or badly bound in
the receiver; it is not a pure visual non-perception failure.

The emerging contrast with duplication:
  - duplication tail failures collapse into local coordinate-neighbor basins
    after object-span onset;
  - false negatives can have latent visual evidence, but the object-start state
    lacks enough language-side/prefix-side guidance and visual-value binding to
    promote the missing object's coordinate basin to top rank.
```

Next deterministic probes:

```text
1. For 2157 wine-glass and 12670 person, capture effective-head hidden/component
   flow at the false-negative pre-x1 state across random/sorted, centered on
   L16 h8 target-overlap and later layers.
2. Add a small bridge sweep for 19109 person because it has sorted visual-route
   sensitivity and random mixed behavior.
3. For the underexplored termination rows, reduce im_end-vs-object_ref_start or
   im_end-vs-next-object x1 margins at box_end states to test premature stop as
   a boundary basin rather than object perception.
```

## 2026-06-26 termination boundary reduction

Implementation slice:

```text
src/analysis/prefix_denoising_surgery_probing/termination_boundary_reduce.py
tests/analysis/test_prefix_denoising_termination_boundary_reduce.py
```

Input atlas shards:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v7_underexplored_termination_fn_shard0_gpu0/atlas_rows.jsonl
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v7_underexplored_termination_fn_shard1_gpu1/atlas_rows.jsonl
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v7_underexplored_termination_fn_shard2_gpu2/atlas_rows.jsonl
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v7_underexplored_termination_fn_shard3_gpu3/atlas_rows.jsonl
```

Reduction artifact:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/termination_boundary_reduce/v1_underexplored_termination_fn_box_end
```

Scope:

```text
source rows: 102
reduced termination-boundary box_end rows: 7
margin definition: log(P(<|im_end|>) / P(<|object_ref_start|>))
penultimate layer used for late-boundary contrast: 24

signature counts:
  continue_basin: 2
  late_stop_flip: 3
  knife_edge_boundary: 2
```

Compact row table:

```text
model           image  obj  desc        final_top1             penult_margin  final_margin  gain    signature
sorted_denoise   9590   33  bowl        <|object_ref_start|>          -2.500        -0.250   2.250  continue_basin
sorted_denoise  17899   48  donut       <|object_ref_start|>          -3.188        -0.125   3.062  continue_basin
random_denoise   2299   27  person      <|im_end|>                    -1.938         0.125   2.062  late_stop_flip
random_denoise   9590   49  bowl        <|im_end|>                    -2.312         0.125   2.438  late_stop_flip
random_denoise  18380   85  wine glass  <|im_end|>                    -3.125         0.000   3.125  knife_edge_boundary
sorted_denoise   2299   20  person      <|im_end|>                    -0.250         1.875   2.125  late_stop_flip
sorted_denoise  14439   22  backpack    <|im_end|>                    -2.062         0.000   2.062  knife_edge_boundary
```

Mechanism update:

```text
The underexplored termination rows do not look like a simple absence of
continuation pressure. Every reduced state has a negative penultimate-layer
stop/continue margin: before the final layer, <|object_ref_start|> is still
favored over <|im_end|>. The final layer then adds a large positive stop shift
(gain 2.062 to 3.125 log units in these rows), producing three late stop flips,
two exact knife-edge ties, and two rows that still continue.

This makes premature or fragile termination a late boundary-basin phenomenon:
the model has an object-continuation route alive near the end of the object span,
but the last-layer wrapper decision can snap toward stop, tie with continue, or
remain in continue. The two knife-edge rows are especially high-value because a
tiny perturbation or tie-breaking rule can decide whether the rollout terminates
or begins another object span.

The current evidence is post-hoc and probability-based because the layerwise
atlas stores wrapper probabilities rather than wrapper logits. It should be
treated as a candidate mechanism, not a proof. The next direct test is to run
causal surgery at box_end states that patches/removes late-layer residual or
effective-head components and asks whether the final im_end/object_ref_start
margin can be moved across zero without disrupting the already-emitted object.
```

High-value next sample bases:

```text
1. 18380 random_denoise wine glass obj85:
   exact final tie after a very negative penultimate margin; best knife-edge
   termination probe.
2. 14439 sorted_denoise backpack obj22:
   exact final tie on a different model/sample; good replication target.
3. 2299 random_denoise person obj27 and 9590 random_denoise bowl obj49:
   shallow late-stop flips with only +0.125 final margin; good causal-patch
   targets because a small late-layer intervention may reverse the decision.
4. 9590 sorted_denoise bowl obj33 and 17899 sorted_denoise donut obj48:
   same late stop gain exists, but continuation survives; useful controls for
   why some states resist the stop basin.
```

## 2026-06-26 termination counter-boundary surgery

Implementation slice:

```text
src/analysis/prefix_denoising_surgery_probing/termination_boundary_surgery_plan.py
src/analysis/prefix_denoising_surgery_probing/termination_boundary_surgery_reduce.py
src/analysis/prefix_denoising_surgery_probing/residual_readout_surgery.py
src/analysis/prefix_denoising_surgery_probing/activation_patch_continuation.py
tests/analysis/test_prefix_denoising_termination_boundary_surgery_plan.py
tests/analysis/test_prefix_denoising_termination_boundary_surgery_reduce.py
```

Why this slice was needed:

```text
The previous boundary reduction identified seven stop/continue boundary states,
but it was still post-hoc. This slice turned those states into explicit causal
surgery rows:

  continue_against_stop:
    target = <|object_ref_start|>, antagonist = <|im_end|>

  stop_against_continue:
    target = <|im_end|>, antagonist = <|object_ref_start|>

Two code-surface fixes were required:
  1. residual terminal-router surgery now accepts an explicit antagonist token
     instead of always treating <|object_ref_start|> as the antagonist.
  2. activation patch rows now preserve boundary_surgery_direction,
     boundary_signature, and source_boundary_state_key so downstream reducers do
     not parse semantics out of state-key suffixes.
```

Plan artifact:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/termination_boundary_surgery_plan/v1_counter_boundary
```

Residual readout-space surgery artifact:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/residual_readout_surgery/v6_termination_counter_boundary_l24_l27_gpu0
```

Full-forward activation patch artifact:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v11_termination_counter_boundary_all7_l24_l27_gpu1
```

Reduced case artifact:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/termination_boundary_surgery_reduce/v1_activation_counter_boundary_all7
```

Activation patch scope:

```text
source boundary cases: 7
activation patch rows: 336
layers: 24, 27
patch sites: layer_input, layer_output, self_attn, mlp
alphas: 0, 0.001, 0.002, 0.005, 0.01, 0.02
continuation_steps: 1

activation_boundary_crossed_count: 7 / 7
directions:
  continue_against_stop: 5
  stop_against_continue: 2
signatures:
  late_stop_flip: 3
  knife_edge_boundary: 2
  continue_basin: 2
first_flip_alpha_counts:
  0.001: 2
  0.002: 2
  0.005: 2
  0.02: 1
```

Case summary:

```text
model           image  obj  desc        signature            direction                first flip
random_denoise  18380   85  wine glass  knife_edge_boundary  continue_against_stop   L24 layer_input  alpha=0.001 -> <|object_ref_start|>
sorted_denoise  14439   22  backpack    knife_edge_boundary  continue_against_stop   L24 layer_output alpha=0.001 -> <|object_ref_start|>
random_denoise   2299   27  person      late_stop_flip       continue_against_stop   L27 layer_input  alpha=0.002 -> <|object_ref_start|>
sorted_denoise  17899   48  donut       continue_basin       stop_against_continue   L27 layer_input  alpha=0.002 -> <|im_end|>
random_denoise   9590   49  bowl        late_stop_flip       continue_against_stop   L27 layer_input  alpha=0.005 -> <|object_ref_start|>
sorted_denoise   9590   33  bowl        continue_basin       stop_against_continue   L24 layer_input  alpha=0.005 -> <|im_end|>
sorted_denoise   2299   20  person      late_stop_flip       continue_against_stop   L27 layer_input  alpha=0.020 -> <|object_ref_start|>
```

Mechanism update:

```text
The stop/continue decision at these box_end states is not a hard semantic
decision about "no more objects" versus "more objects exist." It is a small,
late, bidirectionally movable boundary basin.

Evidence:
  - residual readout-space surgery crossed all seven boundaries locally;
  - full-forward activation patch also crossed all seven boundaries, so the
    effect survives the remaining network computation and changes the patched
    first token;
  - the two knife-edge rows crossed at alpha=0.001, confirming that exact ties
    are extremely fragile boundary states;
  - the shallow late-stop rows crossed at alpha=0.002-0.005;
  - the strong sorted_denoise 2299 person stop basin required alpha=0.02, making
    it a useful harder control rather than a tie-like case.

Layer/component signal:
  - layer_input is the most frequent first-success site in this sweep, but
    layer_output, mlp, and self_attn also cross many cases at slightly larger
    alpha.
  - This pattern suggests the boundary direction is already linearly available
    near the layer input/readout stream, while component-local routes can still
    nudge it across the final decision.

Interpretation boundary:
  - Crossing here means the patched first token becomes the explicit
    counter-boundary target. It does not by itself prove whether the dataset
    label says the model should stop or continue. The value is mechanistic:
    termination is a reversible late router, not an irrevocable failure to
    perceive possible continuation.
```

Next deterministic probes:

```text
1. For 18380 random wine glass and 14439 sorted backpack, inspect exact
   knife-edge boundary flow at smaller alpha around 0.00025-0.001 and compare
   whether ties come from shared wrapper-token geometry or sample-specific
   hidden-state alignment.
2. For 2299 sorted person obj20, trace why the stop basin is much stronger than
   the other late-stop cases despite the penultimate continue preference.
3. Bridge this stop/continue router to false-negative guidance cases: test
   whether missing-object recovery fails because object_ref_start is suppressed
   at the same boundary router, or because the subsequent descriptor/coordinate
   basin is weak after object_ref_start is forced.
```

## 2026-06-26 knife-edge tiny-alpha refinement

Activation patch artifact:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v12_knife_edge_tiny_alpha_18380_14439_l24_l27_gpu2
```

Reduced artifact:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/termination_boundary_surgery_reduce/v2_knife_edge_tiny_alpha_18380_14439
```

Scope:

```text
source boundary cases: 2
activation patch rows: 112
cases:
  random_denoise 18380 wine glass obj85
  sorted_denoise 14439 backpack obj22
layers: 24, 27
patch sites: layer_input, layer_output, self_attn, mlp
alphas: 0, 0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002

activation_boundary_crossed_count: 2 / 2
first_flip_alpha_counts:
  0.00025: 2
```

Case detail:

```text
random_denoise 18380 wine glass obj85:
  first flip: L24 layer_input alpha=0.00025 -> <|object_ref_start|>
  site thresholds:
    layer_input: 0.00025
    layer_output: 0.00075
    mlp: 0.0015
    self_attn: 0.002

sorted_denoise 14439 backpack obj22:
  first flip: L24 layer_input alpha=0.00025 -> <|object_ref_start|>
  site thresholds:
    layer_input: 0.00025
    layer_output: 0.001
    mlp/self_attn: no crossing by 0.002 in this sweep
```

Mechanism refinement:

```text
The exact-tie termination states are even shallower than the all-case sweep
showed. A layer-input perturbation at alpha=0.00025 is sufficient to turn both
from <|im_end|> to <|object_ref_start|> under full-forward activation patch.

This strongly supports the view that these are genuine knife-edge router states:
the model is not robustly deciding to stop; it is balanced almost exactly on a
wrapper-token boundary. Layer input crosses first in both cases, which suggests
that the continuation/stop direction is already present before the layer's
component computation and only needs a tiny displacement in the residual stream.
```

## 2026-06-26 forced object-start bridge to FN coordinates

Implementation slice:

```text
src/analysis/prefix_denoising_surgery_probing/forced_object_start_bridge_plan.py
src/analysis/prefix_denoising_surgery_probing/forced_object_start_bridge_reduce.py
tests/analysis/test_prefix_denoising_forced_object_start_bridge_plan.py
tests/analysis/test_prefix_denoising_forced_object_start_bridge_reduce.py
```

Why this slice was needed:

```text
The termination-boundary surgery showed that stop/continue is a shallow late
router. The next question was whether selected false negatives are mainly
missing because the model fails to open a new object span, or because the
coordinate/grounding basin remains weak even after language-side guidance is
provided.

This bridge constructs counterfactual rows from same image/model pairs:

  terminal box_end prefix
  + <|object_ref_start|>{false-negative desc}<|object_ref_end|><|box_start|>

The target next token is the false-negative object's x1 coordinate. This
bypasses the object-start router by construction and tests the downstream
coordinate basin directly.
```

Plan, atlas, reduce, and activation-patch artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/forced_object_start_bridge_plan/v1_underexplored_termination_fn_same_model
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v8_forced_object_start_bridge_same_model_l24_l27_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/forced_object_start_bridge_reduce/v1_same_model_l24_l27
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/residual_readout_surgery/v7_forced_object_start_bridge_coord_l24_l27_gpu1
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v13_forced_object_start_bridge_coord_l24_l27_gpu2
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/activation_patch/v14_forced_object_start_bridge_span6_l24_l27_gpu3
```

Scope:

```text
forced bridge cases: 7
models:
  random_denoise: 3
  sorted_denoise: 4
image bases:
  2299: 2
  9590: 2
  18380: 1
  14439: 1
  17899: 1
forced descriptors:
  person: 2
  bowl: 2
  tie: 1
  fork: 1
  clock: 1
```

Layerwise forced-bridge signatures:

```text
forced_coord_local_recovered: 2
forced_coord_border_collapse: 3
forced_coord_misaligned_basin: 1
forced_coord_hidden_or_weak: 1
```

Case table:

```text
model           image  terminal source    forced FN target  atlas signature              atlas final readout
random_denoise   2299  person obj27       tie obj1    696   local_recovered             top coord_664, rank 1, dist 32
random_denoise  18380  wine glass obj85   fork obj1   640   local_recovered             top coord_696, rank 3, dist 32
random_denoise   9590  bowl obj49         bowl obj0   644   hidden_or_weak              top coord_420, rank 199, dist 230
sorted_denoise   9590  bowl obj33         clock obj1  273   misaligned_basin            top coord_144, rank 78, dist 129
sorted_denoise   2299  person obj20       person obj1 414   border_collapse             top coord_0, rank 236, dist 414
sorted_denoise  17899  donut obj48        bowl obj5   276   border_collapse             top coord_0, rank 265, dist 276
sorted_denoise  14439  backpack obj22     person obj0 729   border_collapse             top coord_0, rank 465, dist 729
```

Full-forward coordinate activation patch:

```text
selected source cases: 6
activation patch rows: 336
readout_status=ok: 336
layers: 24, 27
patch sites: layer_input, layer_output, self_attn, mlp
alphas: 0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1
continuation_steps: 1

continuation_repair_label_counts:
  first_token_only: 64
  first_token_not_repaired: 272
```

First successful patched x1 token by case:

```text
random_denoise 2299 tie target coord_696:
  30/56 rows patched first token to target
  first success: L24 layer_output alpha=0.005

random_denoise 18380 fork target coord_640:
  13/56 rows patched first token to target
  first success: L27 layer_output alpha=0.002

random_denoise 9590 bowl target coord_644:
  9/56 rows patched first token to target
  first success: L24 layer_output alpha=0.05

sorted_denoise 2299 person target coord_414:
  5/56 rows patched first token to target
  first success: L27 layer_output alpha=0.05

sorted_denoise 9590 clock target coord_273:
  6/56 rows patched first token to target
  first success: L27 layer_output alpha=0.05

sorted_denoise 17899 bowl target coord_276:
  1/56 rows patched first token to target
  first success: L27 layer_output alpha=0.1

sorted_denoise 14439 person target coord_729:
  not selected for activation patch from the residual evidence set
  atlas readout remains the hardest bridge case: coord_0 border collapse,
  target rank 465, distance 729
```

Six-token span continuation:

```text
selected source cases: 6
activation patch rows: 336
readout_status=ok: 336
layers: 24, 27
patch sites: layer_input, layer_output, self_attn, mlp
alphas: 0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1
continuation_steps: 6

continuation_repair_label_counts:
  first_token_only: 64
  first_token_not_repaired: 272
  coord_tail_coherent: 0
```

Best span-continuation examples:

```text
random_denoise 2299 tie target coord_696:
  best patched continuation:
    coord_696, coord_195, coord_677, coord_263, box_end, object_ref_start
  expected tail after x1:
    coord_195, coord_684, coord_259, box_end
  match count after x1: 1/4

random_denoise 9590 bowl target coord_644:
  best patched continuation:
    coord_644, coord_601, coord_677, coord_641, box_end, im_end
  expected tail after x1:
    coord_604, coord_673, coord_633, box_end
  match count after x1: 0/4

random_denoise 18380 fork target coord_640:
  best patched continuation:
    coord_640, coord_592, coord_651, coord_641, box_end, object_ref_start
  expected tail after x1:
    coord_814, coord_754, coord_874, box_end
  match count after x1: 0/4

sorted_denoise 2299 person target coord_414:
  best patched continuation:
    coord_414, coord_652, coord_520, coord_967, box_end, im_end
  expected tail after x1:
    coord_76, coord_520, coord_336, box_end
  match count after x1: 0/4

sorted_denoise 9590 clock target coord_273:
  best patched continuation:
    coord_273, coord_710, coord_283, coord_715, box_end, im_end
  expected tail after x1:
    coord_217, coord_314, coord_295, box_end
  match count after x1: 0/4

sorted_denoise 17899 bowl target coord_276:
  best patched continuation:
    coord_276, coord_922, coord_484, coord_999, box_end, im_end
  expected tail after x1:
    coord_548, coord_454, coord_652, box_end
  match count after x1: 0/4
```

Representative sample-base roster for deeper work:

```text
Primary recoverable bridges:
  2299 random_denoise person->tie:
    object-start forcing nearly recovers the coordinate basin naturally, and
    activation patch crosses at small alpha. Use as the positive bridge case.

  18380 random_denoise wine-glass->fork:
    paired with the earlier knife-edge termination state; object-start forcing
    makes x1 local but not exact. Use as the stop/router-to-coordinate bridge.

Medium / fragile bridges:
  9590 random_denoise bowl->bowl:
    target is hidden or weak in atlas, but full-forward patch can force x1 at
    alpha 0.05. Use as the weak visual/coordinate basin case.

  9590 sorted_denoise bowl->clock:
    atlas chooses a wrong local basin rather than coord_0; patch can force x1
    at alpha 0.05. Use as the misaligned-anchor case.

Hard negative:
  14439 sorted_denoise backpack->person:
    object-start guidance plus descriptor scaffold still collapses to coord_0,
    and this case does not enter the activation-patch selected set. Use as the
    strongest evidence that some false negatives are not merely object-start
    suppression.

Border-collapse controls:
  2299 sorted_denoise person->person:
    border collapse but patchable at alpha 0.05; useful contrast against the
    random 2299 tie-positive bridge under the same image base.

  17899 sorted_denoise donut->bowl:
    border collapse with only one target patch success at alpha 0.1; useful
    as a near-hard sorted-denoise control.
```

Mechanism update:

```text
Object-start suppression is not the only false-negative mechanism. After
manually supplying object_ref_start, descriptor text, object_ref_end, and
box_start, only 2/7 forced bridges naturally land in a local coordinate basin.
The other five remain weak, misaligned, or collapsed to coord_0 at the final
coordinate readout.

The activation-patch result refines this:
  - six cases can be made to emit the intended x1 under full-forward patching,
    so many missing objects are not visually impossible for the model;
  - however, the alpha/site thresholds are much larger than the knife-edge
    termination router, especially for weak and sorted-denoise cases;
  - the six-token continuation probe found zero full-tail coherent repairs,
    so repaired x1 is usually an isolated coordinate intervention rather than
    entry into the expected missing-object span;
  - 14439 remains a hard counterexample where the forced descriptor scaffold
    still does not expose a usable x1 direction in this selected source set.

The emerging picture is a two-gate failure:
  1. a shallow stop/continue router can prematurely suppress a new object span;
  2. even after that router is bypassed, a separate coordinate/grounding basin
     must assemble the right spatial anchor. This second gate is sample- and
     checkpoint-dependent and can fail as coord_0 border collapse, wrong local
     anchor, or diffuse hidden target;
  3. after x1 is forced, the remaining y1/x2/y2 tail can stay attached to a
     different latent box program. This makes object-span emission a coupled
     manifold-entry problem, not a sequence of independently fixable local
     coordinate logits.
```

Next deterministic probes:

```text
1. Keep the above seven image-base cases as the narrow deep panel and avoid
   spending GPU on normal/well-learned rows.
2. For 14439 sorted_denoise, trace whether coord_729 is absent already in
   visual-to-language binding layers, or whether it is present but overwritten
   by a final coord_0 attractor.
3. For paired image 2299, compare random tie-positive versus sorted
   person-border-collapse under the same visual input to isolate checkpoint
   shaping rather than image content.
4. For image 9590, compare random hidden/weak bowl versus sorted misaligned
   clock to separate weak target evidence from wrong-anchor attraction.
5. For the six patchable forced bridges, move from x1-only patching to
   multi-slot or state-entry surgery: patch the x1 state and then inspect the
   next-token hidden state before y1, rather than assuming the expected tail
   will follow from a corrected first coordinate.
```

## 2026-06-26 forced tail-entry state probe

Implementation slice:

```text
src/analysis/prefix_denoising_surgery_probing/forced_tail_entry_plan.py
src/analysis/prefix_denoising_surgery_probing/forced_tail_entry_reduce.py
tests/analysis/test_prefix_denoising_forced_tail_entry_plan.py
tests/analysis/test_prefix_denoising_forced_tail_entry_reduce.py
```

Why this slice was needed:

```text
The span6 forced-object-start bridge showed zero full-tail coherent repairs:
patched x1 could be produced, but y1/x2/y2 usually followed a different box
program. The next deterministic question was whether this is simply because
the wrong continuation tokens poison later slots, or whether the expected tail
is already unavailable even under ideal textual previous-coordinate prefixes.

This slice materializes two counterfactual prefix families:

  expected_target_tail_prefix:
    forced object prefix + intended previous bbox coordinates

  patched_actual_tail_prefix:
    forced object prefix + the model's own successful-x1 patched continuation

Both families are then read with the same layerwise atlas at y1, x2, and y2.
```

Plan, atlas, and reduction artifacts:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/forced_tail_entry_plan/v1_expected_vs_patched_actual_span6_layer_output
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/layerwise_atlas/v9_forced_tail_entry_expected_vs_patched_actual_layers0_4_8_12_16_20_24_27_gpu0
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/forced_tail_entry_reduce/v1_expected_vs_patched_actual_layers0_4_8_12_16_20_24_27
```

Scope:

```text
position rows: 39
atlas rows: 312
layers: 0, 4, 8, 12, 16, 20, 24, 27

expected_target_tail_prefix: 21 rows
patched_actual_tail_prefix: 18 rows

positions:
  post_x1_pre_y1: 13
  post_y1_pre_x2: 13
  post_x2_pre_y2: 13

models:
  random_denoise: 18
  sorted_denoise: 21
```

Final-layer signatures:

```text
tail_entry_exact: 4
tail_entry_near: 11
tail_entry_wrong_or_weak: 24
```

By prefix kind and slot:

```text
expected_target_tail_prefix:
  post_x1_pre_y1:
    exact: 1
    near: 1
    wrong_or_weak: 5
  post_y1_pre_x2:
    exact: 1
    near: 3
    wrong_or_weak: 3
  post_x2_pre_y2:
    near: 2
    wrong_or_weak: 5

patched_actual_tail_prefix:
  post_x1_pre_y1:
    exact: 1
    near: 1
    wrong_or_weak: 4
  post_y1_pre_x2:
    exact: 1
    near: 2
    wrong_or_weak: 3
  post_x2_pre_y2:
    near: 2
    wrong_or_weak: 4
```

Representative cases:

```text
random_denoise 2299 tie:
  expected y1 is exact:
    target coord_195, final top coord_195, rank 1, dist 0
  expected x2 is near:
    target coord_684, final top coord_677, rank 4, dist 7
  expected y2 is near:
    target coord_259, final top coord_263, rank 2, dist 4

random_denoise 9590 bowl:
  expected y1/x2/y2 are all near:
    y1 target coord_604, final top coord_604/coord-bin 601, rank 1, dist 3
    x2 target coord_673, final top coord_677, rank 2, dist 4
    y2 target coord_633, final top coord_641, rank 3, dist 8

random_denoise 18380 fork:
  expected textual tail does not recover the object:
    y1 target coord_814 -> final top coord_608/coord-bin 592, rank 329, dist 222
    x2 target coord_754 -> final top coord_651, rank 355, dist 103
    y2 target coord_874 -> final top coord_999, rank 191, dist 125

sorted_denoise 14439 person:
  hard negative remains hard even under ideal previous-coordinate text:
    y1 target coord_38 -> final coord-bin 385, rank 920, dist 347
    x2 target coord_737 -> final coord_932, rank 591, dist 195
    y2 target coord_85 -> final coord-bin 664, rank 944, dist 579

sorted_denoise 2299 person:
  mixed internal tail:
    y1 wrong: target coord_76 -> final coord_652, rank 842, dist 576
    x2 near/exact depending prefix family: target coord_520
    y2 wrong: target coord_336 -> final coord_999/coord_967, rank 724/814

sorted_denoise 17899 bowl:
  x2 can be exact under expected textual y1:
    target coord_454 -> final coord_454, rank 1, dist 0
  but y1 and y2 stay wrong:
    y1 target coord_548 -> final coord_922, rank 407, dist 374
    y2 target coord_652 -> final coord_671/coord_999, rank 29/271
```

Mechanism update:

```text
This probe separates two mechanisms that the span6 continuation alone could
not separate.

Some forced false negatives are missing mainly at the state-entry boundary:
  - random 2299 tie and random 9590 bowl have expected-tail slots that are
    exact or near once the intended previous coordinates are present in text.
  - For these cases, the model has a usable local tail manifold, but the
    autoregressive rollout often enters it imprecisely or drifts to nearby
    local anchors.

Other forced false negatives are missing because the tail manifold itself is
not locally available:
  - random 18380 fork remains wrong/weak at y1/x2/y2 even with the ideal
    previous-coordinate prefix.
  - sorted 14439 person remains the hard negative at every tail slot, so this
    is not merely a failure to emit object_ref_start or x1.
  - sorted 9590 clock remains wrong/weak across the tail despite x1 patchability.

The final coordinate router is highly saturated in nearly every row, often
with coord_vocab_mass near 0.999. The failure is therefore not "not in coord
mode"; it is selection of the wrong coordinate attractor inside coord mode.

The emerging object-emission picture is now three-stage:
  1. stop/continue router: shallow and reversible;
  2. x1 basin: sometimes patchable even when not naturally selected;
  3. tail manifold: either locally available under correct history, or absent /
     dominated by another coordinate program even under ideal previous tokens.
```

Next deterministic probes:

```text
1. Treat 14439 sorted person as the hard-negative anchor and inspect visual/
   region evidence, because language guidance and ideal coordinate history are
   both insufficient.
2. Treat 2299 random tie and 9590 random bowl as positive controls where the
   local tail manifold exists; use them to identify what successful tail-entry
   hidden states look like before final coordinate routing.
3. Compare 18380 random fork against 2299/9590 positives: it was x1-local but
   tail-unavailable, making it the cleanest bridge between terminal knife-edge
   and downstream grounding failure.
4. For sorted 2299 person, inspect why x2 can recover while y1/y2 fail; this
   is a non-monolithic box-program case and may reveal slot-specific binding.
```

## Forced pre-x1 visual-region value surgery

Date: 2026-06-26.

Purpose: test whether forced false-negative objects have direct local visual
evidence available at the pre-x1 state once the language-side object scaffold
has already been supplied. This is deliberately not a broad metric pass; it is
a narrow mechanism pass over the previously selected hard/control image bases.

Adapter and artifacts:

```text
adapter module:
  src/analysis/prefix_denoising_surgery_probing/forced_visual_plan_adapter.py

forced visual plan:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/forced_visual_plan_adapter/v1_target_region_only_forced_bridge/forced_visual_plan_rows.jsonl
  rows: 5

overlap-derived value-region probe:
  sorted:
    /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v4_forced_bridge_target_region_l16_h8h12h13_sorted_gpu0
  random:
    /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v4_forced_bridge_target_region_l16_h8h12h13_random_gpu1
  reduced:
    /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_reduce/v3_forced_bridge_target_region_l16_h8h12h13

primitive-region value-region probe:
  sorted:
    /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v5_forced_bridge_primitive_regions_l16_h8h12h13_sorted_gpu0
  random:
    /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v5_forced_bridge_primitive_regions_l16_h8h12h13_random_gpu1
  reduced:
    /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_reduce/v4_forced_bridge_primitive_regions_l16_h8h12h13
```

Plan-row scope:

```text
sorted_denoise 14439 person:
  target bbox bins [729, 38, 737, 85], target x1 coord_729

sorted_denoise 2299 person:
  target bbox bins [414, 76, 520, 336], target x1 coord_414

random_denoise 9590 bowl:
  target bbox bins [644, 604, 673, 633], target x1 coord_644

sorted_denoise 9590 clock:
  target bbox bins [273, 217, 314, 295], target x1 coord_273

random_denoise 18380 fork:
  target bbox bins [640, 814, 754, 874], target x1 coord_640
```

Important exclusion: the expected `random_denoise 2299 tie` positive control
from tail-entry probing was not included in this visual-region pass because the
forced row exposes invalid xyxy target bins `[696, 195, 684, 259]` for visual
membership. The same source row also carries a valid pixel box, but resolving
that conflict requires a separate GT/pixel-to-bin provenance check rather than
silently fabricating a region.

Baseline pre-x1 coordinate ranks under the forced object scaffold:

```text
sorted_denoise 14439 person:
  target coord_729, top1 coord_0, rank 441, target-minus-top1 logit -5.5625

sorted_denoise 2299 person:
  target coord_414, top1 coord_0, rank 231, target-minus-top1 logit -1.75

sorted_denoise 9590 clock:
  target coord_273, top1 coord_144, rank 77, target-minus-top1 logit -2.0

random_denoise 9590 bowl:
  target coord_644, top1 coord_414, rank 242, target-minus-top1 logit -1.125

random_denoise 18380 fork:
  target coord_640, top1 coord_670/decoded coord_672, rank 3,
  target-minus-top1 logit -0.125
```

Visual-token membership, primitive regions:

```text
sorted_denoise 14439 person:
  target_full 0, target_bottom_band 0, target_upper_body 0,
  context_ring 32, far_background 880

sorted_denoise 2299 person:
  target_full 24, target_bottom_band 8, target_upper_body 16,
  context_ring 86, far_background 764

sorted_denoise 9590 clock:
  target_full 2, target_bottom_band 1, target_upper_body 1,
  context_ring 61, far_background 951

random_denoise 9590 bowl:
  target_full 0, target_bottom_band 0, target_upper_body 0,
  context_ring 54, far_background 960

random_denoise 18380 fork:
  target_full 8, target_bottom_band 4, target_upper_body 4,
  context_ring 64, far_background 942
```

Layer-16 head intervention highlights, region scale 0:

```text
sorted_denoise 14439 person:
  target regions are empty and no-op.
  context_ring removal hurts rank mildly:
    h8 441 -> 462, h12 441 -> 451, h13 441 -> 466.
  far_background removal is head-opposed:
    h12 improves 441 -> 293,
    h13 worsens 441 -> 487,
    h8 is near-neutral 441 -> 447.

sorted_denoise 2299 person:
  target_full/target bands are no-op despite 24 target tokens.
  context_ring is also no-op.
  far_background is strongly head-opposed:
    h8 improves 231 -> 8,
    h13 improves 231 -> 9,
    h12 worsens 231 -> 645 and changes top1 to coord_742.

sorted_denoise 9590 clock:
  target_full removal weakly hurts:
    h8 77 -> 90, h12 77 -> 90, h13 77 -> 89.
  far_background removal hurts more:
    h8 77 -> 147, h13 77 -> 134.
  This is the only slice here with a small but visible direct target-region
  value route at layer 16.

random_denoise 9590 bowl:
  target regions are empty and no-op.
  context_ring removal strongly hurts:
    h8 242 -> 469, h13 242 -> 352, h12 242 -> 296.
  far_background removal also hurts:
    h8 242 -> 375, h13 242 -> 363.

random_denoise 18380 fork:
  baseline x1 is already near/top-3.
  target_full removal hurts mildly:
    h8 3 -> 7, h12 3 -> 9.
  target_upper_body and context/far-background can slightly improve to rank 1
  depending head, so the x1 basin is language-guided and locally fragile rather
  than strongly target-visual dominated.
```

Mechanism update:

```text
The hard 14439 sorted/person false negative is not explained by "target visual
tokens are present but ignored" at this probe granularity. The target box maps
to zero visual tokens under the current merged visual grid. The local
context_ring has tokens, but removing them hurts rather than rescues the target
x1. The only rescue-like operation in this layer-16 pass is far_background
head-12 removal, which improves rank 441 -> 293 while head-13 far_background
removal worsens it. This suggests a distributed/background-mediated coordinate
competition or suppressor, not a simple absent language cue.

Sorted 2299 person is a sharper head-opposition case: target and context
regions are no-op, but far_background h8/h13 removal almost rescues x1 to top
10 while far_background h12 removal catastrophically worsens it. This makes it
a promising sample for head-level sign/opposition analysis rather than just
visual locality analysis.

Random 9590 bowl is not target-token local in this visual pass because the
target box has zero visual tokens. Unlike 14439, its context_ring/far_background
carry supportive evidence: removing them worsens the target x1. This makes it
a useful contrast for "tiny/unresolved object whose surrounding visual context
helps" versus "tiny/unresolved object whose coordinate basin is suppressed by
global/background streams."

Random 18380 fork confirms the earlier bridge interpretation. Its x1 is already
available under the forced scaffold, but later tail slots failed in the forced
tail-entry probe. This separates x1 grounding from tail-manifold availability:
good x1 rank does not guarantee a coherent object-span box program.
```

Next deterministic probes:

```text
1. For 14439 sorted/person and 2299 sorted/person, run head-sign/opposition
   decomposition around far_background h8/h12/h13 at layer 16. The question is
   whether these heads write competing coordinate attractors into the residual
   stream or alter the coord_0 / coord_target margin through a shared router.

2. Add a principled region-resolver for invalid or sub-token boxes:
   distinguish target box with zero visual-token membership, invalid xyxy bins,
   valid pixel box, and GT annotation bbox. Do not mix these silently.

3. For 9590 random/bowl, probe whether the context_ring support is descriptor-
   specific or merely location prior by replacing the forced descriptor while
   preserving the same prefix and target x1.

4. For 18380 random/fork, move from x1 visual grounding to tail manifold:
   patch the x1-good hidden state into post-x1/post-y1 positions and test
   whether y1/x2/y2 failure is slot-specific, descriptor-specific, or a loss of
   object identity after the first coordinate.
```

## Far-background head-opposition scale sweep

Date: 2026-06-26.

Purpose: refine the forced pre-x1 visual-region result for the two sorted/person
cases. The previous pass showed that target/context regions were mostly no-op
and far-background head removal could either improve or damage the x1 basin. The
scale sweep tests whether these are sign-stable contributions by scaling the
same layer-16 head/region value contribution through:

```text
region_scale = -1, 0, 0.5, 1, 1.5, 2
```

where `1` is the unpatched/no-op contribution, `0` removes the region
contribution, and `2` amplifies it by one extra copy.

Artifacts:

```text
scale sweep probes:
  14439 sorted/person:
    /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v6_forced_bridge_far_background_scale_sweep_sorted_14439_l16_h8h12h13_gpu0
  2299 sorted/person:
    /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v6_forced_bridge_far_background_scale_sweep_sorted_2299_l16_h8h12h13_gpu1

scale sweep reducer:
  module:
    src/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_scale_sweep_reduce.py
  reduced artifact:
    /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_scale_sweep_reduce/v1_forced_bridge_far_background_scale_sweep_sorted_person_l16_h8h12h13
```

Summary signature counts over 18 curves
`2 images x 3 heads x 3 regions`:

```text
mostly_neutral: 9
mixed_or_nonlinear: 4
region_contribution_suppresses_target: 3
region_contribution_supports_target: 2
```

All five strong supportive/suppressive signatures are from `far_background`.
`target_full` is mostly neutral, and `context_ring` is neutral or weak/mixed.
This reinforces that the decisive signal is not local target-box evidence in
these two sorted/person slices.

Far-background curves:

```text
sorted_denoise 14439 person, baseline x1 rank 441:
  h12 is target-suppressive:
    scale 0:   rank 293, delta -148
    scale 0.5: rank 366, delta -75
    scale 1:   rank 441, delta 0
    scale 1.5: rank 538, delta +97
    scale 2:   rank 584, delta +143

  h13 is target-supportive:
    scale 0: rank 487, delta +46
    scale 1: rank 441, delta 0
    scale 2: rank 416, delta -25

  h8 is mixed/nonlinear:
    scale -1 gives the best rank 418, but scale 2 worsens to 646.

sorted_denoise 2299 person, baseline x1 rank 231:
  h8 is strongly target-suppressive:
    scale 0:   rank 8, delta -223
    scale 0.5: rank 2, delta -229
    scale 1:   rank 231, delta 0
    scale 1.5: rank 787, delta +556
    scale 2:   rank 871, delta +640

  h13 is target-suppressive:
    scale 0: rank 9, delta -222
    scale 1: rank 231, delta 0
    scale 2: rank 617, delta +386

  h12 is target-supportive:
    scale 0: rank 645, delta +414
    scale 1: rank 231, delta 0
    scale 2: rank 5, delta -226
```

Important readout detail:

```text
2299 h8 far_background scale 2:
  target logit rises slightly relative to baseline:
    target logit 15.875 -> 16.375
  but a competing coordinate attractor explodes:
    top1 bin 0 -> 768
    top1 logit 17.625 -> 24.625
  target rank worsens 231 -> 871.

2299 h12 far_background scale 2:
  target rank improves 231 -> 5.
  top1 remains coord_0 and target margin improves modestly:
    target-minus-top1 -1.75 -> -1.5.

14439 h12 far_background scale 0:
  target rank improves 441 -> 293.
  target logit rises 14.4375 -> 14.75 and coord_0 top1 logit falls
  20.0 -> 19.5.
```

Mechanism update:

```text
Layer-16 far-background heads are not merely "visual support" or "visual
noise." They act like sample-state-dependent coordinate attractor modulators.
The same head can be supportive in one sorted/person image and suppressive in
another:
  - h12 suppresses 14439/person but supports 2299/person.
  - h13 supports 14439/person but suppresses 2299/person.

The strongest suppressive curves do not always damage the target by lowering
the target coordinate logit. In 2299 h8, amplification raises the target logit
but raises a wrong coordinate basin much more strongly, flipping top1 to a
distant coordinate around 768. So the operative mechanism is competition among
coordinate attractors inside an already-stable coord-token mode, not failure to
enter coord mode.

This makes "far-background" an imperfect spatial name for the causal route.
At this layer/head/query site, it behaves as a large distributed value source
that can write global coordinate priors or competing anchors. The route may
contain object-layout context, border priors, repeated-person priors, or
residualized image/prefix statistics rather than literal object pixels.
```

Next deterministic probes:

```text
1. Decompose the far-background value contribution for 2299/person h8/h12/h13
   into token subsets or top-attended visual tokens. The goal is to find which
   visual-token groups drive the wrong coord_768/742 attractor versus the target
   coord_414 basin.

2. For 14439/person, inspect whether the zero-token target box is a resolution
   artifact by using expanded target boxes, pixel/GT-backed bbox provenance, and
   top-attended-token visualization. The hard negative may live below visual
   token granularity.

3. Run the same scale-sweep reducer on random/sorted paired cases beyond
   person, especially 9590 bowl/clock and 18380 fork, to see whether the
   sample-state-dependent sign flip is specific to crowded-person layouts or a
   general coordinate-router feature.
```

## Far-background top-token decomposition

Date: 2026-06-26.

Purpose: localize which visual tokens inside the `far_background` value source
drive the `sorted_denoise 2299/person` coordinate-attractor split found in the
scale sweep. This pass projects each selected visual token's attention-weighted
value contribution onto target-minus-contrast coord-output directions:

```text
target x1: coord_414
contrasts: coord_0, coord_742, coord_768
layer/head: L16 h8/h12/h13
region: far_background
top_k: 25 per head/contrast by absolute projection
```

Artifacts:

```text
token probe module:
  src/analysis/prefix_denoising_surgery_probing/fn_visual_value_token_probe.py

token probe artifact:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_token_probe/v1_sorted_2299_far_background_l16_h8h12h13_contrast0_742_768_gpu0

rows:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_token_probe/v1_sorted_2299_far_background_l16_h8h12h13_contrast0_742_768_gpu0/fn_visual_value_token_probe_rows.jsonl
summary:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_token_probe/v1_sorted_2299_far_background_l16_h8h12h13_contrast0_742_768_gpu0/fn_visual_value_token_probe_summary.json
```

Probe health:

```text
plan rows: 1
membership rows: 11
token rows: 225
errors: 0
projection role counts over top-k rows:
  supports_contrast: 123
  supports_target: 102
```

Important implementation note: the first token-probe run exposed a runtime
bug in the naive implementation. It built all 1000 coord-output rows through a
helper that copied the full LM-head weight for each coord token. The final
module caches only the needed output rows: target coord_414 plus contrasts
coord_0/742/768. The optimized run finished in seconds.

High-level spatial pattern:

```text
The strongest top-token projections concentrate in a lower-right visual band:
  x roughly 776-855
  y roughly 760-935

This band is outside the target bbox region in this forced visual plan, but it
is not diffuse "all background." It looks like a localized lower/right layout
zone that the layer-16 heads use as a coordinate-attractor source.
```

Per-head/contrast summary over top-25 token rows:

```text
h8, target coord_414 vs coord_0:
  25/25 top rows support target.
  sum projection +0.8043.
  strongest token:
    row 17 col 30, x 802.6 y 760.9, attention 0.0835,
    projection +0.1130.

h8, target coord_414 vs coord_768:
  21/25 top rows support contrast.
  sum projection -0.1573.
  strongest contrast token:
    row 20 col 31, x 828.9 y 891.3, attention 0.0447,
    projection -0.0234.
  strongest target token:
    row 16 col 31, x 828.9 y 717.4, attention 0.0347,
    projection +0.0240.

h12, target coord_414 vs coord_742:
  25/25 top rows support target.
  sum projection +0.2767.
  strongest token:
    row 17 col 30, x 802.6 y 760.9, attention 0.0170,
    projection +0.0257.

h12, target coord_414 vs coord_768:
  18/25 top rows support contrast.
  sum projection -0.0708.
  strongest contrast token:
    row 16 col 31, x 828.9 y 717.4, attention 0.0219,
    projection -0.0152.

h13, target coord_414 vs coord_742:
  24/25 top rows support contrast.
  sum projection -0.4420.
  strongest token:
    row 20 col 31, x 828.9 y 891.3, attention 0.1055,
    projection -0.1088.

h13, target coord_414 vs coord_768:
  18/25 top rows support contrast.
  sum projection -0.2065.
  strongest token:
    row 20 col 31, x 828.9 y 891.3, attention 0.1055,
    projection -0.0645.
```

Mechanism update:

```text
The scale-sweep sign split now has a token-level candidate substrate.

For 2299/person, h13's suppressive far-background route is dominated by a few
high-attention lower-right visual tokens, especially row 20 col 31
(norm x 828.9, y 891.3). Those tokens project strongly toward the observed
wrong-attractor directions coord_742/coord_768 rather than target coord_414.

h12 is not simply attending a different part of the image. It uses many of the
same lower-right coordinates, but the contribution direction flips for
coord_742: the top-25 tokens all support target over coord_742. This suggests
head-specific value/output projection semantics over a shared spatial evidence
band, rather than one head looking at "right pixels" and another at "wrong
pixels."

h8 is contrast-dependent. Against coord_0 it strongly supports target, but
against coord_768 most top rows support contrast. This matches the scale-sweep
observation that h8 amplification makes a distant coord_768 basin explode even
while the target logit can rise: the h8 value stream contains mixed target-vs-
wrong-attractor components, and amplification can preferentially grow the wrong
basin.
```

Interpretation boundary:

```text
These rows are top-k token projections, not a complete additive decomposition
of all 764 far-background tokens. They localize candidate contributors and
explain the qualitative head-opposition pattern, but the next causal step is to
patch or scale token subsets by spatial band.
```

Next deterministic probes:

```text
1. Add a token-subset intervention over the lower-right band found here:
   approximately x 776-855 and y 760-935. Compare removing this band versus
   removing the rest of far_background for h8/h12/h13.

2. Run the same token probe for 14439/person h12/h13 to see whether its
   far-background sign flip uses the same lower-right band or a different
   distributed visual source.

3. Build a visualization overlay for the top projected tokens so manual review
   can connect this lower-right band to actual image content and annotation
   geometry.
```

## Lower-right band causal partition

Date: 2026-06-26.

Purpose: test whether the lower-right top-token band from the previous
decomposition is causal, or merely an artifact of ranking individual token
projections. This probe adds opt-in custom box regions to the value-region
intervention runner, then partitions `far_background` into:

```text
lower_right_band bbox: [760, 650, 880, 950]
far_background_intersect_lower_right_band
far_background_minus_lower_right_band
```

The partition is over the same visual-token grid membership used by the
existing visual-origin probes; no new geometry convention was introduced.

Artifacts:

```text
probe code:
  src/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe.py

artifact root:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v7_sorted_2299_lower_right_band_l16_h8h12h13_scales_gpu0

rows:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v7_sorted_2299_lower_right_band_l16_h8h12h13_scales_gpu0/fn_visual_value_region_probe_rows.jsonl
membership:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v7_sorted_2299_lower_right_band_l16_h8h12h13_scales_gpu0/fn_visual_value_region_probe_membership_rows.jsonl
summary:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v7_sorted_2299_lower_right_band_l16_h8h12h13_scales_gpu0/fn_visual_value_region_probe_summary.json
```

Probe command:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.analysis.prefix_denoising_surgery_probing.fn_visual_value_region_probe \
  --plan-rows /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/forced_visual_plan_adapter/v1_target_region_only_forced_bridge/forced_visual_plan_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v7_sorted_2299_lower_right_band_l16_h8h12h13_scales_gpu0 \
  --image-ids 2299 \
  --probe-model-ids sorted_denoise \
  --scaffold-stages pre_x1 \
  --layers 16 \
  --heads 8,12,13 \
  --regions lower_right_band,far_background_intersect_lower_right_band,far_background_minus_lower_right_band,far_background \
  --region-scales 0,0.5,1,1.5,2 \
  --custom-box-regions lower_right_band:760,650,880,950:far_background \
  --device cuda:0 \
  --attn-implementation eager \
  --torch-dtype bfloat16 \
  --top-k 8
```

Probe health:

```text
plan rows: 1
condition rows: 61
membership rows: 14
probe rows: 61
errors: 0

baseline target x1: coord_414
baseline target rank: 231
baseline top1: coord_0
baseline target logit: 15.875
baseline coord_0 logit: 17.625
```

Membership:

```text
target_full: 24 tokens
context_ring: 86 tokens
far_background: 764 tokens
lower_right_band: 28 tokens
far_background_intersect_lower_right_band: 28 tokens
far_background_minus_lower_right_band: 736 tokens
```

The custom band is exactly inside `far_background` for this sample-base:
`lower_right_band` and `far_background_intersect_lower_right_band` have the
same 28 visual tokens.

Scale response table, target rank lower is better:

```text
h8:
  lower_right/intersect scale 0.0 -> rank 2, top1 coord_0
  lower_right/intersect scale 0.5 -> rank 2, top1 coord_0
  lower_right/intersect scale 1.5 -> rank 751, top1 coord_768
  lower_right/intersect scale 2.0 -> rank 821, top1 coord_768
  far_background_minus scale 0.0 -> rank 56, top1 coord_0
  far_background_minus scale 2.0 -> rank 411, top1 coord_742
  full far_background scale 2.0 -> rank 871, top1 coord_768

h12:
  lower_right/intersect scale 0.0 -> rank 554, top1 coord_775
  lower_right/intersect scale 0.5 -> rank 386, top1 coord_0
  lower_right/intersect scale 1.5 -> rank 42, top1 coord_0
  lower_right/intersect scale 2.0 -> rank 5, top1 coord_0
  far_background_minus scale 0.0 -> rank 298, top1 coord_0
  far_background_minus scale 2.0 -> rank 193, top1 coord_0
  full far_background scale 0.0 -> rank 645, top1 coord_742
  full far_background scale 2.0 -> rank 5, top1 coord_0

h13:
  lower_right/intersect scale 0.0 -> rank 18, top1 coord_0
  lower_right/intersect scale 0.5 -> rank 22, top1 coord_0
  lower_right/intersect scale 1.5 -> rank 440, top1 coord_775
  lower_right/intersect scale 2.0 -> rank 584, top1 coord_768/775 basin
  far_background_minus scale 0.0 -> rank 174, top1 coord_0
  far_background_minus scale 2.0 -> rank 328, top1 coord_0
  full far_background scale 0.0 -> rank 9, top1 coord_0
  full far_background scale 2.0 -> rank 617, top1 coord_768/775 basin
```

Attention/contribution concentration:

```text
h8 far_background attention mass: 0.7905
  lower-right partition: 0.5327 over 28 tokens
  remainder: 0.2578 over 736 tokens

h12 far_background attention mass: 0.3941
  lower-right partition: 0.1538 over 28 tokens
  remainder: 0.2403 over 736 tokens

h13 far_background attention mass: 0.7929
  lower-right partition: 0.5672 over 28 tokens
  remainder: 0.2257 over 736 tokens
```

Mechanism update:

```text
The lower-right band is causal for the 2299/person x1 basin split.

h8 and h13: the 28-token lower-right partition carries most of the wrong-
attractor route. Removing or halving it sharply improves target rank; amplifying
it drives the state into the coord_768/775 basin. The much larger 736-token
remainder still matters, but its effect is weaker and tends toward coord_742
or mild rank drift rather than the full coord_768 explosion.

h12: the same 28-token band has opposite sign. Removing it hurts the target
badly, while amplifying it improves target rank to 5. The band is not simply
"bad background"; the head-specific value/output transformation determines
whether the shared visual evidence supports target coord_414 or a wrong
lower-right coordinate basin.

This supports a more precise mechanism than diffuse background distraction:
the model contains a localized coordinate-attractor source in visual-token
space, and different attention heads route the same local visual evidence into
opposite coordinate-output directions. The autoregressive pre-x1 false-negative
state is therefore partly a competition between head-specific value semantics
over a shared spatial substrate, not a simple failure to perceive the missing
person.
```

Important logit detail:

```text
h8 lower-right scale 2.0:
  target coord_414 logit rises slightly from 15.875 to 16.000,
  but coord_768 logit rises to 24.25 and target rank collapses to 821.

h12 lower-right scale 2.0:
  target coord_414 logit rises to 16.375,
  local mass near target increases,
  target rank improves to 5.

h13 lower-right scale 2.0:
  target coord_414 logit falls to 15.625,
  coord_768/775 basin dominates with top probabilities around 0.0147.
```

Interpretation boundary:

```text
Evidence scope is one curated false-negative sample-base:
  sorted_denoise, image 2299, target person, forced pre-x1 state, layer 16.

This proves a causal lower-right-band partition for this state. It does not
yet prove that all false negatives or all prefix-denoising failures use the
same spatial band.
```

Next deterministic probes:

```text
1. Repeat the custom band partition on 14439/person to check whether its
   far-background h12/h13 split uses the same lower-right band or a different
   visual substrate.

2. Run a narrow visual overlay for image 2299 marking the 28 lower-right tokens,
   target bbox, context ring, and wrong coord_742/768/775 attractor bins.

3. Test whether this band remains causal at later object-span positions
   (x2/y2/end) or whether the wrong basin is primarily pre-x1 onset dynamics.

4. Add a contrast-specific readout reducer for coord_414 vs coord_742/768/775
   so future runs can directly report wrong-basin logit margins, not only rank.
```

## 14439 Replication: Different Far-background Substrate

Date: 2026-06-26.

Purpose: test whether the 2299/person lower-right-band mechanism generalizes
to the other sorted-denoise forced false-negative case, `14439/person`.

Artifacts:

```text
same lower-right partition:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v8_sorted_14439_lower_right_band_l16_h8h12h13_scales_gpu0

token decomposition:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_token_probe/v2_sorted_14439_far_background_l16_h8h12h13_contrast0_1_249_236_gpu0

candidate-band causal partition:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v9_sorted_14439_candidate_bands_l16_h8h12h13_scales_gpu0
```

Baseline:

```text
target x1: coord_729
baseline target rank: 441
baseline top1: coord_0
baseline target logit: 14.4375
baseline coord_0 logit: 20.0
```

Same lower-right partition result:

```text
far_background: 880 tokens
lower_right_band [760,650,880,950]: 28 tokens
far_background_minus_lower_right_band: 852 tokens

h8 lower-right attention mass: 0.0150
h12 lower-right attention mass: 0.0243
h13 lower-right attention mass: 0.0169
```

Unlike 2299/person, this 28-token lower-right band is nearly inert:

```text
h8 lower-right scale 0/1/2 target ranks: 435 / 441 / 440
h12 lower-right scale 0/1/2 target ranks: 425 / 441 / 472
h13 lower-right scale 0/1/2 target ranks: 447 / 441 / 437
```

The full far-background route still matters:

```text
h8 full far_background scale 0/1/2 ranks: 447 / 441 / 646
h12 full far_background scale 0/1/2 ranks: 293 / 441 / 584
h13 full far_background scale 0/1/2 ranks: 487 / 441 / 416
```

Token decomposition findings:

```text
contrasts: coord_0, coord_1, coord_249, coord_236
token rows: 300
errors: 0

h12 is consistently contrast-supporting against coord_0/1/236:
  coord_0: 25/25 top rows support contrast, sum projection -0.1911
  coord_1: 25/25 top rows support contrast, sum projection -0.2392
  coord_236: 25/25 top rows support contrast, sum projection -0.2170

h13 is mostly target-supporting:
  coord_0: 25/25 top rows support target, sum projection +0.1600
  coord_1: 25/25 top rows support target, sum projection +0.1762
  coord_249: 24/25 top rows support target, sum projection +0.1593

h8 is target-supporting against coord_0/1 but mixed against coord_236/249.
```

Candidate spatial boxes from token rows:

```text
central_left_band: [190, 280, 310, 380] -> 10 tokens
right_mid_band: [760, 470, 950, 700] -> 42 tokens
lower_left_band: [0, 560, 320, 1000] -> 132 tokens
```

Candidate-band causal partition:

```text
h12:
  right_mid scale 0/1/2 ranks: 354 / 441 / 544
  full far_background scale 0/1/2 ranks: 293 / 441 / 584
  far_background_minus_right_mid scale 0/1/2 ranks: 402 / 441 / 524

h13:
  right_mid scale 0/1/2 ranks: 477 / 441 / 468
  full far_background scale 0/1/2 ranks: 487 / 441 / 416
  far_background_minus_right_mid scale 0/1/2 ranks: 498 / 441 / 408

h8:
  full far_background scale 0/1/2 ranks: 447 / 441 / 646
  far_background_minus_central_left scale 0/1/2 ranks: 468 / 441 / 707
  far_background_minus_lower_left scale 0/1/2 ranks: 502 / 441 / 655
```

Mechanism update:

```text
The 2299 lower-right attractor band does not generalize as a fixed spatial
source. For 14439/person, the same box carries negligible attention and almost
no causal effect.

The h12 far-background route is still strongly suppressive: removing it
improves target rank from 441 to 293, while amplifying it worsens rank to 584.
Token projections and candidate-band surgery both implicate a right-middle
patch as part of this h12 suppressive route, but not the whole route. The
remaining far-background complement still carries substantial effect.

The h13 route has the opposite sign at full far_background scale: removing it
worsens rank to 487, amplifying it improves rank to 416. However, the tested
candidate boxes do not explain most of h13's helpful route; the best candidate
partition is actually the complement after removing right_mid. This suggests a
more distributed or differently shaped h13 target-supporting substrate.

Current picture: prefix-denoising false-negative onset can depend on localized
visual attractor patches, but the patch location is sample-state-specific and
head-specific. The deeper invariant is not "lower-right background"; it is a
competition between head-specific value semantics over whichever visual-token
regions a given image/prefix state routes into coordinate-output space.
```

Interpretation boundary:

```text
Evidence scope is still two curated sorted_denoise forced false-negative
sample-bases: 2299/person and 14439/person, both at pre-x1 layer 16. The
findings support a mechanism family, not a population law.
```

Next deterministic probes:

```text
1. Build a contrast-specific wrong-basin reducer before more sample expansion;
   rank alone hides which competitor basin is changing.

2. Add custom union/difference regions so multi-patch substrates can be tested
   as one causal set instead of one box at a time.

3. For 14439, find the h13 helpful substrate by ranking all far-background
   tokens by target-support projection and testing the top-k union directly.

4. Run these localized/union probes on a small train-vs-val sample panel to
   separate memorized object perception from unseen-prefix basin routing.
```

## Contrast-bin margin readout

Date: 2026-06-26.

Purpose: rank alone was hiding which wrong coordinate basin moved. The
value-region probe now accepts `--contrast-bins` and records, for each contrast
bin:

```text
baseline_contrast_coord_<bin>_logit
patched_contrast_coord_<bin>_logit
baseline_target_minus_contrast_coord_<bin>_logit
patched_target_minus_contrast_coord_<bin>_logit
target_minus_contrast_coord_<bin>_logit_delta
```

Probe-code change:

```text
src/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe.py
```

Artifacts:

```text
2299/person lower-right contrast rerun:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v10_sorted_2299_lower_right_band_l16_h8h12h13_scales_contrast0_742_768_775_gpu0

14439/person candidate-band contrast rerun:
  /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v10_sorted_14439_candidate_bands_l16_h8h12h13_scales_contrast0_1_236_249_gpu1
```

Probe health:

```text
2299 rows: 37, errors: 0
14439 rows: 64, errors: 0
```

2299/person contrast margins:

```text
target: coord_414
baseline margins:
  target - coord_0:   -1.750
  target - coord_742: -1.375
  target - coord_768: -0.750
  target - coord_775: -1.125

h8 lower-right/intersect:
  scale 0, rank 2:
    margin deltas vs 0/742/768/775: +0.875 / +4.0625 / +4.25 / +4.1875
  scale 2, rank 821:
    margin deltas vs 0/742/768/775: -3.875 / -5.375 / -7.5 / -6.625

h12 lower-right/intersect:
  scale 0, rank 554:
    margin deltas vs 0/742/768/775: -1.25 / -1.875 / -2.75 / -2.5
  scale 2, rank 5:
    margin deltas vs 0/742/768/775: +0.625 / +1.8125 / +2.125 / +2.3125

h13 lower-right/intersect:
  scale 0, rank 18:
    margin deltas vs 0/742/768/775: +0.375 / +2.25 / +2.5 / +2.8125
  scale 2, rank 584:
    margin deltas vs 0/742/768/775: -1.375 / -2.125 / -3.375 / -3.0
```

Interpretation for 2299:

```text
The lower-right partition is not merely moving target rank indirectly. It
directly controls target-vs-wrong-basin margins.

h8 and h13 lower-right amplification sharply worsens the target margin against
the wrong coord_768/775 basin. h12 lower-right amplification does the opposite,
turning the same visual band into target-supporting margin against 742/768/775.
This is stronger evidence for head-specific value semantics over a shared
visual substrate than the rank-only table.
```

14439/person contrast margins:

```text
target: coord_729
baseline margins:
  target - coord_0:   -5.5625
  target - coord_1:   -3.9375
  target - coord_236: -2.4375
  target - coord_249: -2.4375

h12 right_mid:
  scale 0, rank 354:
    margin deltas vs 0/1/236/249: +0.8125 / +0.6875 / +0.5625 / +0.5625
  scale 2, rank 544:
    margin deltas vs 0/1/236/249: -0.5625 / -0.5625 / -0.3125 / -0.3125

h12 full far_background:
  scale 0, rank 293:
    margin deltas vs 0/1/236/249: +0.8125 / +0.8125 / +0.1875 / +0.3125
  scale 2, rank 584:
    margin deltas vs 0/1/236/249: -0.75 / -0.625 / 0.0 / -0.25

h13 full far_background:
  scale 0, rank 487:
    margin deltas vs 0/1/236/249: -0.375 / -0.375 / +0.125 / 0.0
  scale 2, rank 416:
    margin deltas vs 0/1/236/249: +0.25 / +0.25 / 0.0 / 0.0
```

Interpretation for 14439:

```text
h12's right-middle patch is a real wrong-basin margin controller: removing it
improves target margin against every tracked contrast, and amplifying it
worsens every tracked contrast margin.

h13's full far-background improvement is weaker and more selective. It mainly
improves target margin against coord_0/1, while margins against coord_236/249
barely move. This explains why h13 rank can improve without a dramatic
wrong-basin signature like 2299: the competitor structure is broader and the
helpful h13 route is not captured by a single tested candidate patch.
```

Mechanism update:

```text
Contrast margins sharpen the current model:

1. 2299/person is a localized wrong-attractor-basin case. The 28-token
   lower-right visual band controls large target-vs-742/768/775 margins, with
   h8/h13 and h12 having opposite signs over the same spatial evidence.

2. 14439/person is a weaker, more distributed coord_0-collapse case. Its h12
   right-middle patch affects margins, but the full far-background effect is
   not reducible to that patch; h13 support is broad and mostly improves
   target-vs-origin margins.

So the likely core mechanism is not a universal visual location, but a
head-specific conversion from attended visual-token subsets into coordinate
basin margins. The observed false-negative onset depends on which subsets the
prefix state routes into the coordinate readout at the next object-span slot.
```

## 14439 h13 Token-union Causal Probe

Date: 2026-06-26.

Purpose: replace hand-drawn candidate boxes with token-decomposition-derived
causal sets. The value-region probe now accepts `--custom-token-regions` so a
token union from the token-decomposition rows can be tested directly:

```text
region_kind:prompt_token_index,prompt_token_index[:base_region]
```

As with custom boxes, a base region derives:

```text
base_region_intersect_region_kind
base_region_minus_region_kind
```

Probe-code change:

```text
src/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe.py
```

Artifact:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v11_sorted_14439_h13_target_support_token_unions_l16_contrast0_1_236_249_gpu0
```

Token-union source:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_token_probe/v2_sorted_14439_far_background_l16_h8h12h13_contrast0_1_249_236_gpu0
```

Probe health:

```text
condition rows: 28
membership rows: 18
errors: 0
target: coord_729
baseline rank: 441
baseline top1: coord_0
baseline margins vs 0/1/236/249: -5.5625 / -3.9375 / -2.4375 / -2.4375
```

Token unions tested:

```text
top5-per-contrast union:  11 unique tokens
top10-per-contrast union: 20 unique tokens
top15-per-contrast union: 29 unique tokens
top25-per-contrast union: 37 unique tokens
```

The unions are derived from h13 `supports_target` top projection rows across
contrast bins `0,1,236,249`, then intersected with `far_background`.

Membership and attention:

```text
far_background: 880 tokens, attention mass 0.6705

top5 union:  11 tokens, attention mass 0.1781
top10 union: 20 tokens, attention mass 0.2307
top15 union: 29 tokens, attention mass 0.2782
top25 union: 37 tokens, attention mass 0.3111
```

Causal response, h13 only:

```text
full far_background:
  scale 0 -> rank 487
  scale 2 -> rank 416
  margin delta vs coord_0/1 at scale 2: +0.25 / +0.25

top15 union:
  scale 0 -> rank 491
  scale 2 -> rank 428
  margin delta vs coord_0/1 at scale 2: +0.0625 / -0.0625

top25 union:
  scale 0 -> rank 493
  scale 2 -> rank 425
  margin delta vs coord_0/1 at scale 2: +0.1875 / +0.0625

top25 complement:
  scale 0 -> rank 438
  scale 2 -> rank 458
```

Mechanism update:

```text
The h13 helpful route for 14439 is not captured by the earlier hand-drawn
candidate boxes, but it is captured by a token-decomposition-derived union.

Removing only the top15/top25 h13 target-support union worsens target rank to
491/493, matching or slightly exceeding the full far-background removal rank
487. Amplifying the same union improves rank to 428/425, recovering most of
the full far-background amplification benefit at rank 416.

The complement does not carry the same helpful effect: top25 complement
amplification worsens rank to 458. Thus, for h13, the helpful route is a sparse
multi-site token union, not a contiguous box and not diffuse background.
```

Interpretation:

```text
This closes a small but important gap from the box-based probe. The earlier
candidate boxes failed because the h13 substrate is spatially discontinuous:
central-left, lower-left, right-mid, and edge tokens all contribute. The
attention/value route is sparse in token space but not compact in image space.

Compared to 2299:
  2299 h8/h13 wrong-attractor route: compact lower-right visual band.
  14439 h13 helpful route: sparse multi-site target-support token union.

The common mechanism remains head-specific value conversion into coordinate
basin margins, but the spatial basis can be compact or discontinuous depending
on image/prefix state.
```

Next deterministic probes:

```text
1. Build a small val/train sample panel and compare whether trained images show
   more compact, lower-entropy token unions than held-out val images.

2. Extend token-union causal probes to x2/y2/end slots to test whether sparse
   visual unions mainly govern onset or continue to shape the object span.
```

## 14439 h12 Suppressive Token-union Probe

Date: 2026-06-26.

Purpose: run the symmetric token-union test for h12 using `supports_contrast`
token-decomposition rows. This asks whether the h12 suppressive route is a
contiguous patch, diffuse background, or a sparse multi-site union like h13.

Artifact:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v12_sorted_14439_h12_contrast_support_token_unions_l16_contrast0_1_236_249_gpu0
```

Token-union source:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_token_probe/v2_sorted_14439_far_background_l16_h8h12h13_contrast0_1_249_236_gpu0
```

Probe health:

```text
condition rows: 28
membership rows: 18
errors: 0
target: coord_729
baseline rank: 441
baseline top1: coord_0
```

Token unions tested:

```text
top5-per-contrast union:  12 unique tokens
top10-per-contrast union: 17 unique tokens
top15-per-contrast union: 25 unique tokens
top25-per-contrast union: 46 unique tokens
```

Membership and attention:

```text
far_background: 880 tokens, attention mass 0.5064

top5 union:  12 tokens, attention mass 0.1444
top10 union: 17 tokens, attention mass 0.1646
top15 union: 25 tokens, attention mass 0.2063
top25 union: 46 tokens, attention mass 0.2730
```

Causal response, h12 only:

```text
full far_background:
  scale 0 -> rank 293
  scale 2 -> rank 584
  margin delta vs coord_0/1/236/249:
    scale 0: +0.8125 / +0.8125 / +0.1875 / +0.3125
    scale 2: -0.75 / -0.625 / 0.0 / -0.25

top25 union:
  scale 0 -> rank 311
  scale 2 -> rank 562
  margin delta vs coord_0/1/236/249:
    scale 0: +0.5625 / +0.5625 / +0.1875 / +0.3125
    scale 2: -0.5625 / -0.5625 / +0.0625 / -0.1875

top25 complement:
  scale 0 -> rank 426
  scale 2 -> rank 479
  margin delta vs coord_0/1/236/249:
    scale 0: +0.1875 / +0.0625 / -0.0625 / -0.0625
    scale 2: -0.3125 / -0.3125 / -0.0625 / -0.0625
```

Mechanism update:

```text
h12's suppressive route for 14439 is also sparse multi-site, but less compact
than h13's helpful route.

The h12 top25 support-contrast union captures most of the full far-background
effect: removing it improves rank to 311, close to full removal at 293;
amplifying it worsens rank to 562, close to full amplification at 584. The
top25 complement is much weaker: scale 0 only reaches rank 426 and scale 2
only rank 479.

The h12 route is therefore not purely diffuse background. It is a sparse,
multi-site contrast-supporting value route with enough remaining mass outside
the top union to make the full far-background effect slightly stronger.
```

Combined h12/h13 picture for 14439:

```text
h13 target-supporting route:
  top25 target-support union: 37 tokens, attention mass 0.3111
  scale 0 rank: 493
  scale 2 rank: 425
  full far_background scale 0/2 ranks: 487 / 416

h12 contrast-supporting route:
  top25 contrast-support union: 46 tokens, attention mass 0.2730
  scale 0 rank: 311
  scale 2 rank: 562
  full far_background scale 0/2 ranks: 293 / 584

Both routes are sparse in token space and discontinuous in image space. The
difference is semantic sign: h13's top union is target-supporting against
coord_0/1, while h12's top union is contrast-supporting and suppresses the
target basin. This is the cleanest 14439 evidence so far for head-specific
value semantics over sparse visual-token unions.
```

Next deterministic probes:

```text
1. Build a small val/train sample panel and compare token-union sparsity,
   spatial continuity, and margin sign across seen versus held-out images.

2. Extend token-union probes from pre-x1 to x2/y2/end slots for 2299 and 14439.

3. Add a reducer that computes union compactness metrics: token count,
   attention mass, connected components on visual grid, bbox area, and margin
   capture ratio versus full far_background.
```

## Token-union Compactness Reducer

Implementation:

```text
src/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_compactness_reduce.py
tests/analysis/test_prefix_denoising_fn_visual_value_region_compactness_reduce.py
```

Verification:

```text
python -m pytest -q tests/analysis/test_prefix_denoising_fn_visual_value_region_compactness_reduce.py
python -m pytest -q \
  tests/analysis/test_prefix_denoising_fn_visual_value_region_compactness_reduce.py \
  tests/analysis/test_prefix_denoising_fn_visual_value_region_probe.py \
  tests/analysis/test_prefix_denoising_fn_visual_value_token_probe.py \
  tests/analysis/test_prefix_denoising_fn_visual_value_region_scale_sweep_reduce.py
```

Artifact outputs:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_compactness_reduce/v1_sorted_14439_h13_token_union
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_compactness_reduce/v1_sorted_14439_h12_token_union
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_compactness_reduce/v1_sorted_2299_lower_right_band
```

Compactness readout:

```text
14439/person h13 target-support top25 union:
  tokens: 37 / 880 far_background tokens
  token fraction: 0.0420
  connected components: 23
  bbox grid: [0, 5, 36, 23]
  bbox area fraction: 0.7708
  density in bbox: 0.0526
  scale 2 rank delta: -16

14439/person h12 contrast-support top25 union:
  tokens: 46 / 880 far_background tokens
  token fraction: 0.0523
  connected components: 27
  bbox grid: [0, 5, 35, 23]
  bbox area fraction: 0.7500
  density in bbox: 0.0673
  scale 2 rank delta: +121

2299/person lower-right band:
  tokens: 28 / 764 far_background tokens
  token fraction: 0.0366
  connected components: 1
  bbox area fraction: 0.0320
  density in bbox: 1.0000
  h12 scale 2 rank delta: -226
  h13 scale 2 rank delta: +353
  h8  scale 2 rank delta: +590
```

Interpretation:

```text
The compactness reducer separates two different mechanism families that looked
similar under the earlier "far_background matters" label.

2299/person is a compact spatial-attractor case. A dense 28-token lower-right
patch is sufficient to carry large and opposite-signed value effects across
heads: h12 helps the target basin, while h8/h13 push toward the wrong attractor.

14439/person is not a compact patch case. Both the helpful h13 route and the
suppressive h12 route are sparse multi-site token unions. They cover only about
4-5% of far_background tokens, but their visual-grid bounding boxes span about
75-77% of the image with very low density. This says the route is distributed
over selected visual anchors, not a single object-shaped crop or a diffuse
background average.

Therefore the current false-negative mechanism split is:
  - compact wrong-attractor basin: 2299/person;
  - distributed sparse binding/contrast route: 14439/person.
```

Representative sample-base policy update:

```text
Do not spend the next deep probes on normal/well-learned images just because
they are common. Keep a narrow microscope panel whose samples expose different
mechanism families.

Locked high-value bases:
  1. 14439/person/sorted_denoise:
     distributed sparse visual-token substrate; h13 target-supporting and h12
     contrast-supporting routes have opposite semantic sign without compact
     spatial localization.

  2. 2299/person/sorted_denoise:
     compact lower-right attractor patch; h12/h13/h8 reveal strong
     opposite-signed head effects over the same dense local visual region.

Positive/near-success controls remain useful only when they explain a failed
base by contrast. They should not replace the hard bases as the main microscope.

Next sample additions should be selected by mechanism contrast:
  - one train-set analogue of 14439, if available, to test whether distributed
    sparse routes are generalization failures or also exist on trained images;
  - one train-set analogue of 2299, if available, to test whether compact
    wrong-attractor patches are learned basin artifacts or held-out artifacts;
  - one termination-boundary case only if it can be tied to the same hidden
    object-state or value-route readouts.
```

## Train-set Analogue Selector and First Guided Probe

Purpose: move beyond held-out `val200` without spending effort on normal images.
The selector ranks train-set GT object rows as analogues of the two locked
archetypes:

```text
14439/person/sorted_denoise:
  distributed_sparse_binding
  anchor target: coord_729 at forced pre-x1
  anchor image structure: 20 objects, 13 persons, 7 small person boxes, kite/backpack/chair context

2299/person/sorted_denoise:
  compact_patch_attractor
  anchor target: coord_414 at forced pre-x1
  anchor image structure: 22 objects, 13 persons, 9 ties, lower-right compact patch
```

Implementation:

```text
src/analysis/prefix_denoising_surgery_probing/gt_analogue_selector.py
src/analysis/prefix_denoising_surgery_probing/gt_analogue_probe_plan.py
tests/analysis/test_prefix_denoising_gt_analogue_selector.py
tests/analysis/test_prefix_denoising_gt_analogue_probe_plan.py
```

Important implementation correction:

```text
The first selector draft over-read target/context helper regions as compact
patches. This was wrong for 14439: target_bottom_band and context_ring are
not independent compact attractor regions. The corrected selector treats
14439 as distributed_sparse_binding and only explicit custom focus regions
such as lower_right_band as compact patches.
```

Current artifacts:

```text
GT analogue selector, current:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/gt_analogue_selector/v3_train_bbox_len12000_14439_2299

GT forced pre-x1 probe plan, current:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/gt_analogue_probe_plan/v2_train_bbox_len12000_14439_2299

First two-row GPU value-region probe:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v13_train_analogue_compact_distributed_l16_h8h12h13_gpu0

Compactness reduction:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_compactness_reduce/v2_train_analogue_compact_distributed_l16_h8h12h13
```

Commands:

```text
python -m src.analysis.prefix_denoising_surgery_probing.gt_analogue_selector \
  --anchor sorted_14439_person_distributed=/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v11_sorted_14439_h13_target_support_token_unions_l16_contrast0_1_236_249_gpu0/fn_visual_value_region_probe_membership_rows.jsonl \
  --anchor sorted_2299_person_compact=/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v10_sorted_2299_lower_right_band_l16_h8h12h13_scales_contrast0_742_768_775_gpu0/fn_visual_value_region_probe_membership_rows.jsonl \
  --anchor-gt-jsonl /data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl \
  --candidate-gt-jsonl /data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl \
  --output-root /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/gt_analogue_selector/v3_train_bbox_len12000_14439_2299 \
  --top-k 30

python -m src.analysis.prefix_denoising_surgery_probing.gt_analogue_probe_plan \
  --analogue-rows /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/gt_analogue_selector/v3_train_bbox_len12000_14439_2299/gt_analogue_candidates.jsonl \
  --gt-jsonl /data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl \
  --output-root /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/gt_analogue_probe_plan/v2_train_bbox_len12000_14439_2299 \
  --checkpoint-path /data/CoordExp/outputs/stage1_2b/detection_teacher_forcing/compact_object_box_closed_desc_first_prefix_denoising_kl_w0p05_k2_2b_base_sorted_bsz1x128_4epoch/compact-object-box-closed-desc-first-prefix-denoising-kl-w0p05-k2-2b-base-sorted-bsz1x128-4epoch/v5-20260623-133125/checkpoint-908 \
  --probe-model-id sorted_denoise

CUDA_VISIBLE_DEVICES=0 python -m src.analysis.prefix_denoising_surgery_probing.fn_visual_value_region_probe \
  --plan-rows /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/gt_analogue_probe_plan/v2_train_bbox_len12000_14439_2299/gt_analogue_value_probe_plan_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v13_train_analogue_compact_distributed_l16_h8h12h13_gpu0 \
  --image-root /data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000 \
  --selection-ranks 0,30 \
  --layers 16 \
  --heads 8,12,13 \
  --regions target_full,far_background,lower_right_band,far_background_intersect_lower_right_band,far_background_minus_lower_right_band \
  --custom-box-regions lower_right_band:760,650,880,950:far_background \
  --region-scales 0,1,2 \
  --contrast-bins 0,1,236,249 \
  --device cuda:0 \
  --attn-implementation eager \
  --torch-dtype bfloat16
```

Probe health:

```text
selector candidates: 60 total, 30 per anchor
probe plan rows: 60
GPU probe rows: 92
membership rows: 23
errors: 0
```

First selected train analogue rows:

```text
compact-patch analogue:
  image 546097, object 0, person, bbox [258, 66, 355, 361]
  anchor: sorted_2299_person_compact
  forced pre-x1 baseline target coord_258 rank: 18
  baseline top1: coord_249
  target-minus-top1 logit: -0.75

distributed analogue:
  image 418535, object 8, person, bbox [795, 48, 806, 83]
  anchor: sorted_14439_person_distributed
  forced pre-x1 baseline target coord_795 rank: 127
  baseline top1: coord_302
  target-minus-top1 logit: -2.125
```

Key intervention readout:

```text
image 418535, distributed train analogue:
  target_full has 0 visual tokens, matching the zero-token target problem.

  h12 far_background:
    scale 0 rank 127 -> 107
    scale 2 rank 127 -> 522
    attention mass 0.893

  h12 far_background_minus_lower_right_band:
    scale 0 rank 127 -> 121
    scale 2 rank 127 -> 557
    attention mass 0.891

  h13 far_background:
    scale 0 rank 127 -> 138
    scale 2 rank 127 -> 103
    attention mass 0.910

  lower_right_band:
    40 tokens, dense compact patch, but only small effects
    h8 scale 2 rank 127 -> 146
    h12 scale 2 rank 127 -> 132
    h13 scale 2 rank 127 -> 137

image 546097, compact-patch train analogue:
  target_full has 21 visual tokens.
  baseline rank is already much healthier: 18.
  lower_right_band has 32 visual tokens but near-zero attention mass for h8/h12/h13.

  h12 target_full:
    scale 0 rank 18 -> 9
    scale 2 rank 18 -> 17
    attention mass 0.103

  h13 target_full:
    scale 0 rank 18 -> 16
    scale 2 rank 18 -> 15
    attention mass 0.263

  h8 far_background:
    scale 2 rank 18 -> 14
    attention mass 0.346
```

Mechanism update:

```text
The first train-set check supports the sample-base split.

The distributed/zero-target-token family transfers to train: image 418535 is a
trained-image analogue where the guided pre-x1 target is still rank 127, target
visual membership is zero, and h12 far_background amplification catastrophically
pushes the model away from the target basin. This is very close in sign to
14439's h12 suppressive route, though the current probe did not yet decompose
the sparse token union for 418535.

The compact lower-right patch does not simply transfer by GT geometry alone:
image 546097 is a strong GT analogue of 2299 by person crowding and a compact
lower-right region, but under the forced pre-x1 prefix the target is already
rank 18, the lower-right patch receives near-zero attention, and most effects
come from target_full or broad background. This suggests the compact-patch
failure is not just a geometric motif; it likely requires the autoregressive
context state / generated prefix to bind the patch into the wrong coordinate
basin.

Immediate implication:
  - For distributed cases, GT-guided train probes can reveal the same suppressive
    value route without needing a free-rollout failure first.
  - For compact-patch cases, train GT analogues need either rollout-derived bad
    prefixes or contextual corruption/transplant. Geometry alone is too weak.
```

Next deterministic probes:

```text
1. For train image 418535, run token-decomposition on far_background h12/h13,
   then test whether sparse multi-site token unions recapitulate 14439.

2. For compact image 546097, do not keep scaling the lower-right patch under GT
   guidance. Instead, transplant the 2299 bad autoregressive context or search
   train free-rollout prefixes for compact-patch onset states.

3. Add a balanced selection option to gt_analogue_probe_plan if future probes
   should automatically take top-k per anchor instead of global-score ranks.
```

## Train Analogue 418535 Token-Union Recapitulation

Date: 2026-06-26

Purpose: follow the first train-analogue result above instead of stopping at a
broad `far_background` statement. The question was whether image 418535's
distributed false-negative route is carried by a small set of high-leverage
visual tokens, by the whole background route, or by a mixed-sign basin where
token decomposition and causal scaling disagree in useful ways.

Artifacts:

```text
Token decomposition:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_token_probe/v3_train_418535_far_background_l16_h12h13_contrast0_302_gpu0

Custom-token value-region recapitulation:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v14_train_418535_token_union_l16_h12h13_gpu0

Token sign-consistency reducer:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_token_sign_reduce/v1_train_418535_l16_h12h13_contrast0_302_gpu0

Compactness reduction:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_compactness_reduce/v3_train_418535_token_union_l16_h12h13
```

Commands:

```text
CUDA_VISIBLE_DEVICES=0 python -m src.analysis.prefix_denoising_surgery_probing.fn_visual_value_token_probe \
  --plan-rows /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/gt_analogue_probe_plan/v2_train_bbox_len12000_14439_2299/gt_analogue_value_probe_plan_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_token_probe/v3_train_418535_far_background_l16_h12h13_contrast0_302_gpu0 \
  --image-root /data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000 \
  --selection-ranks 30 \
  --layers 16 \
  --heads 12,13 \
  --regions far_background \
  --contrast-bins 0,1,302,663,696 \
  --top-k 40 \
  --device cuda:0 \
  --attn-implementation eager \
  --torch-dtype bfloat16

CUDA_VISIBLE_DEVICES=0 python -m src.analysis.prefix_denoising_surgery_probing.fn_visual_value_region_probe \
  --plan-rows /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/gt_analogue_probe_plan/v2_train_bbox_len12000_14439_2299/gt_analogue_value_probe_plan_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v14_train_418535_token_union_l16_h12h13_gpu0 \
  --image-root /data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000 \
  --selection-ranks 30 \
  --layers 16 \
  --heads 12,13 \
  --regions far_background,h12_pos_core,far_background_minus_h12_pos_core,h12_neg_core,far_background_minus_h12_neg_core,h13_pos_core,far_background_minus_h13_pos_core,h13_neg_core,far_background_minus_h13_neg_core,shared_top_abs,far_background_minus_shared_top_abs \
  --custom-token-regions h12_pos_core:231,266,196,195,267:far_background\;h12_neg_core:155,190,191,227,226,262:far_background\;h13_pos_core:191,155,190,335,159,264,226:far_background\;h13_neg_core:231,160,196,266,195:far_background\;shared_top_abs:191,231,155,196,190,266,195,160:far_background \
  --region-scales 0,1,2 \
  --contrast-bins 0,1,302,663,696 \
  --device cuda:0 \
  --attn-implementation eager \
  --torch-dtype bfloat16

python -m src.analysis.prefix_denoising_surgery_probing.fn_visual_value_region_compactness_reduce \
  --membership-rows /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v14_train_418535_token_union_l16_h12h13_gpu0/fn_visual_value_region_probe_membership_rows.jsonl \
  --rows /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_probe/v14_train_418535_token_union_l16_h12h13_gpu0/fn_visual_value_region_probe_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_region_compactness_reduce/v3_train_418535_token_union_l16_h12h13

python -m src.analysis.prefix_denoising_surgery_probing.fn_visual_value_token_sign_reduce \
  --rows /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_token_probe/v3_train_418535_far_background_l16_h12h13_contrast0_302_gpu0/fn_visual_value_token_probe_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_token_sign_reduce/v1_train_418535_l16_h12h13_contrast0_302_gpu0
```

Probe health:

```text
token probe: 1 plan row, 400 token rows, 6 membership rows, errors 0
custom-token causal probe: 1 plan row, 67 condition rows, 67 rows,
  21 membership rows, errors 0
compactness reduce: 66 non-baseline rows
sign-consistency reduce: 111 token groups, 30 mixed-sign tokens, errors 0
```

Token-decomposition evidence:

```text
Baseline state:
  image 418535, sorted_denoise, object 8 person
  target x1 coord_795
  baseline top1 coord_302
  target rank 127
  target-minus-coord_302 logit -2.125

h12 strongest positive target-minus-contrast contributors:
  token 231, visual (col=15,row=3), center (430.6,140.0)
    summed positive projection 1.2699 across contrast bins
  token 266, visual (14,4), center (402.8,180.0)
    summed positive projection 0.3036
  tokens 196/195/267 form the same upper-mid compact support.

h12 strongest mixed/contrast contributors:
  token 155, visual (11,1), center (319.4,60.0)
    positive 0.0357, negative -0.1264
  tokens 190/191/227/226/262 are neighboring upper-left sites with
    negative mass against most contrasts and positive mass against coord_302.

h13 strongest contributors:
  token 191, visual (11,2), center (319.4,100.0)
    positive 0.4894, negative -0.2633
  token 155, visual (11,1), center (319.4,60.0)
    positive 0.2262, negative -0.0098
  h13 also gives token 231 the opposite sign: negative -0.1150.

Sign-consistency reducer:
  111 token groups from 400 token rows
  dominant roles:
    supports_target: 59
    supports_contrast: 22
    mixed_supports_contrast: 20
    mixed_supports_target: 10
  mixed signs:
    h12: 15/51 token groups
    h13: 15/60 token groups
  top absolute token:
    h12 token 231, abs 1.2699, supports_target, sign consistency 1.0
  top mixed token:
    h13 token 191, abs 0.7527, mixed_supports_target, sign consistency 0.6
```

Causal token-union readout:

```text
h12 full far_background:
  scale 0 rank 127 -> 107
  scale 2 rank 127 -> 522, top1 coord_663
  interpretation: h12's full background contribution is suppressive for
  the target basin when amplified.

h12 shared_top_abs, 8 tokens only:
  scale 2 rank 127 -> 562, top1 coord_663
  bbox area fraction 0.0311, 3 connected components
  interpretation: a tiny high-leverage token union recapitulates and even
  exceeds the full h12 amplification failure.

h12 h12_pos_core, 5 tokens:
  scale 0 rank 127 -> 162
  scale 2 rank 127 -> 230
  interpretation: tokens that project positively onto target-minus-contrast
  directions are not a rescue route under actual causal scaling. They behave
  as part of the bad local basin for h12.

h12 h12_neg_core, 6 tokens:
  scale 0 rank 127 -> 115
  scale 2 rank 127 -> 205
  interpretation: removing this compact upper-left contrast core helps, but
  amplifying it hurts.

h13 full far_background:
  scale 0 rank 127 -> 138
  scale 2 rank 127 -> 103
  interpretation: h13 has the opposite sign from h12 at the full-route level.

h13 h13_pos_core, 7 tokens:
  scale 0 rank 127 -> 109
  scale 2 rank 127 -> 114
  bbox area fraction 0.0400, 4 connected components

h13 shared_top_abs, 8 tokens:
  scale 0 rank 127 -> 102
  scale 2 rank 127 -> 121
  interpretation: the same high-absolute support that is harmful when
  amplified through h12 is helpful when removed through h13, again pointing to
  mixed-sign routing rather than a single object-evidence vector.
```

Mechanism update:

```text
The 418535 train analogue is not just a diffuse far-background phenomenon.
A very small set of upper-image visual tokens carries a large fraction of the
causal effect, but the sign depends on head and intervention direction.

This should not be reported as a clean sparse positive target route. No v14
non-baseline intervention flips top1 to coord_795, and h12's strongest sparse
routes are anti-target under causal scaling. The safer verdict is that v14
exposes head-specific value routes that move the coordinate basin, with h12
acting as a strong wrong-basin amplifier and h13 providing only weak target
pressure.

The most important correction is that "token projects toward target" is not
equivalent to "amplifying this token-region rescues the target." The readout
projection is local to target-minus-contrast output rows, while causal scaling
changes the whole head contribution before downstream residual computation.
For h12, the compact high-leverage token union is a bad-basin amplifier. For
h13, the broader route is comparatively target-supporting, but the compact
shared tokens are still mixed.

This supports a mixed-sign coordinate-basin picture:
  - the visual evidence is not absent;
  - the evidence is routed through a small upper-image support, not through
    the zero-token target box;
  - h12 and h13 disagree over how that support moves the coordinate basin;
  - false negatives can arise when the correct object has no direct token
    membership and target evidence is present only through fragile,
    contrast-dependent, head-specific routes.
```

Updated next deterministic probes:

```text
1. For 418535/14439-like distributed cases, run a layer/head tomography over
   the same token unions. The key question is where the sign flip first appears:
   visual value projection, o_proj head output, MLP/residual mixing, or final
   coord-output readout.

2. Before broadening samples, run a signed additive value-component
   recomposition panel at the same 418535 L16 h12/h13 sites:
     - mean-preserving region ablation;
     - target-direction component only;
     - orthogonal remainder only;
     - h13 positive component plus h12 anti-component ablation;
     - matched random far-background controls by token count and attention mass.
   This directly separates "selected tokens are wrong", "sign is mixed", and
   "region scaling amplifies the wrong subspace."

3. For compact-patch cases, stop using GT geometry alone. Use
   contextual_anchor_substitution or a new train-prefix source to create bad
   autoregressive contexts for 546097-like train images, then probe whether
   the lower-right compact patch gains attention/value mass only after the
   prefix has entered the wrong basin.

4. Use the new sign-consistency reducer before comparing many cases, so
   contrast-dependent sign flips are explicit row-level artifacts rather than
   hand-derived snippets.
```

## Signed Value-Component Recomposition Panel

Status: completed narrow single-state probe for the highest-leverage train
analogue slice from the token-union study.

Artifact:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_component_recompose_probe/v1_train_418535_l16_h12h13_modes6_gpu0
```

Command:

```text
CUSTOM='h12_pos_core:231,266,196,195,267:far_background;h12_neg_core:155,190,191,227,226,262:far_background;h13_pos_core:191,155,190,335,159,264,226:far_background;h13_neg_core:231,160,196,266,195:far_background;shared_top_abs:191,231,155,196,190,266,195,160:far_background'
REGIONS='h12_pos_core,h12_neg_core,h13_pos_core,h13_neg_core,shared_top_abs'
MODES='remove_full,add_full,target_component_only,orthogonal_remainder_only,add_target_component,remove_orthogonal_remainder'

CUDA_VISIBLE_DEVICES=0 python -m src.analysis.prefix_denoising_surgery_probing.fn_visual_value_component_recompose_probe \
  --plan-rows /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/gt_analogue_probe_plan/v2_train_bbox_len12000_14439_2299/gt_analogue_value_probe_plan_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_component_recompose_probe/v1_train_418535_l16_h12h13_modes6_gpu0 \
  --image-root /data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000 \
  --selection-ranks 30 \
  --layers 16 \
  --heads 12,13 \
  --regions "$REGIONS" \
  --custom-token-regions "$CUSTOM" \
  --contrast-bins 302,663 \
  --component-modes "$MODES" \
  --device cuda:0 \
  --attn-implementation eager \
  --torch-dtype bfloat16
```

Probe health:

```text
plan rows: 1
condition rows: 121
membership rows: 21
readout rows: 121
ok rows: 121
errors: 0
baseline: target coord_795, top1 coord_302, target rank 127
baseline margins: target-minus-302 = -2.125, target-minus-663 = 0.625
```

Best rank improvements:

```text
h13 shared_top_abs, contrast 302:
  target_component_only: rank 127 -> 86, top1 stays coord_302
  remove_orthogonal_remainder: rank 127 -> 86, top1 stays coord_302
  target logit +0.875, target-minus-302 +1.0625
  full norm 17.2280, target component norm 0.9188,
  orthogonal remainder norm 17.2035

h13 shared_top_abs, contrast 663:
  target_component_only: rank 127 -> 101, top1 stays coord_302
  remove_orthogonal_remainder: rank 127 -> 101, top1 stays coord_302
  target logit +0.8125, target-minus-302 +0.9375,
  target-minus-663 -0.375

h13 shared_top_abs, remove_full:
  rank 127 -> 102 for both contrast pulls, top1 stays coord_302
```

Worst basin failures:

```text
h12 shared_top_abs, add_full:
  rank 127 -> 562, top1 coord_302 -> coord_663
  target logit +0.6875, target-minus-302 +1.125,
  target-minus-663 -3.0625

h12 h12_pos_core, add_full:
  rank 127 -> 230, top1 coord_302 -> coord_328
  target-minus-302 +0.875, target-minus-663 -1.3125

h12 h13_neg_core, add_full:
  rank 127 -> 225, top1 coord_302 -> coord_335
  target-minus-302 +0.9375, target-minus-663 -1.375

h12 h12_neg_core, add_full:
  rank 127 -> 205, top1 coord_302 -> coord_335
```

Important control symmetry:

```text
For a fixed head-direction decomposition, target_component_only and
remove_orthogonal_remainder are intentionally equivalent:
  new vector = old vector - orthogonal remainder = target component.

The duplicate results in this panel are therefore a useful implementation
sanity check, not independent evidence.
```

Mechanism update:

```text
This panel strengthens the mixed-sign coordinate-basin account.

The clearest partial rescue is h13 shared_top_abs target-direction isolation:
it improves coord_795 rank from 127 to 86 and improves the target-minus-302
margin by +1.0625. However, it does not flip top1 from coord_302 to coord_795.
So the object evidence is present and causally useful, but it is still too weak
or too late to dominate the coordinate basin.

The strongest failure is h12 shared_top_abs add_full. It improves
target-minus-302 while catastrophically flipping the local coordinate basin
toward coord_663 and pushing target rank to 562. This is direct evidence that
single-contrast margins can be misleading: a patch can make the target look
better against one wrong coordinate while moving the full coordinate
distribution into an even worse basin.

h12 target components are not clean rescue vectors. h12 shared_top_abs
target_component_only makes target-minus-302 worse and slightly worsens rank;
h12 h12_pos_core target_component_only also worsens rank. That means the
pulled-back target direction is head- and downstream-context dependent, not a
global "correct object" axis that can be reused across heads.

h13 is different: its shared high-absolute visual support contains a small
target-direction component surrounded by a much larger orthogonal remainder.
Removing that remainder is enough for the best rank improvement in the panel.
This points to a plausible internal failure mode: correct object pressure is
encoded as a small component inside a high-norm visual value route, but the
orthogonal route and competing heads dominate the autoregressive coordinate
decision.
```

Revised next deterministic probes after recomposition:

```text
1. Run the same component-recomposition panel on at least two more selected
   image-bases: one val false-negative case and one duplication-burst onset
   case. Use representative sample-base selection rather than normal/well
   learnt images.

2. Add full-coordinate-distribution reducers for component surgery. Rank and
   one contrast margin are insufficient; the h12 shared_top_abs failure shows
   that top-k basin migration, expected distance, radius mass, and top1 flip
   categories are first-class evidence.

3. Run layer tomography for the same token unions and component modes. The
   open question is where h13's small useful target component appears and where
   h12's wrong-basin route becomes dominant: value vector, o_proj output,
   residual stream, MLP, or final coordinate readout.

4. Add paired-checkpoint comparison for prefix-denoising versus the strongest
   pure-CE sorted baseline on the same selected sample-bases. Do not average
   away the case structure; compare the tensor-flow routes that lead to
   different object-span emissions.

5. Keep sample-base selection aggressive. Avoid spending GPU budget on normal
   images unless they are used as matched controls for the same class,
   geometry, object count, or prefix context.
```

## Full-Basin Reducer And Val False-Negative Replication

Status: completed post-hoc reducer plus one val sample-base recomposition panel.

New reducer:

```text
src/analysis/prefix_denoising_surgery_probing/fn_visual_value_component_basin_reduce.py
tests/analysis/test_prefix_denoising_fn_visual_value_component_basin_reduce.py
```

Reducer purpose:

```text
Rank and one contrast margin are insufficient for component surgery. The new
reducer classifies every non-baseline recomposition row by:
  - target rank movement;
  - top1 coordinate movement;
  - expected absolute distance movement;
  - radius-16 target mass movement;
  - positive/negative target-minus-contrast margin deltas;
  - margin/basin conflict labels.

It is a post-hoc artifact reducer. It does not load a model and does not create
new causal evidence by itself.
```

Reducer artifact for the earlier train analogue:

```text
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_component_basin_reduce/v1_train_418535_l16_h12h13_modes6_gpu0
```

Train analogue reducer read:

```text
source rows: 121
non-baseline rows: 120
baseline: target coord_795, top1 coord_302, target rank 127

effect labels:
  partial_rescue_no_top1_flip: 36
  target_rank_regression: 46
  wrong_basin_amplifier: 38

rank movement:
  improved: 36
  regressed: 84

top1 movement:
  changed_wrong_top1: 10
  unchanged_wrong_top1: 110

margin/basin conflicts: 58
top1 flips: 10
```

This formalizes the previous hand-read: `418535` has real h13 partial rescue
routes, but it also has many h12-style interventions where a local margin
improves while the coordinate basin worsens.

Val replication artifact:

```text
Recomposition:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_component_recompose_probe/v2_val_14439_l16_h8h12h13_modes6_gpu0

Full-basin reduction:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_component_basin_reduce/v2_val_14439_l16_h8h12h13_modes6_gpu0

Token sign reduction:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_token_sign_reduce/v2_val_14439_l16_h8h12h13_contrast0_1_236_249
```

Val command:

```text
CUSTOM='h8_target_core:594,634,756,383,421,420,597,632:far_background;h12_contrast_core:708,383,560,374,566,421,745,604,599,643:far_background;h13_target_core:643,634,421,383,992,594,408,878,756,422:far_background;shared_target_core:594,634,756,383,421,643,422,408,992:far_background;opponent_overlap:383,421,643:far_background'
REGIONS='h8_target_core,h12_contrast_core,h13_target_core,shared_target_core,opponent_overlap'
MODES='remove_full,add_full,target_component_only,orthogonal_remainder_only,add_target_component,remove_orthogonal_remainder'

CUDA_VISIBLE_DEVICES=0 python -m src.analysis.prefix_denoising_surgery_probing.fn_visual_value_component_recompose_probe \
  --plan-rows /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_token_probe/v2_sorted_14439_far_background_l16_h8h12h13_contrast0_1_249_236_gpu0/fn_visual_value_token_probe_plan_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_component_recompose_probe/v2_val_14439_l16_h8h12h13_modes6_gpu0 \
  --image-root /data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60 \
  --selection-ranks 2 \
  --layers 16 \
  --heads 8,12,13 \
  --regions "$REGIONS" \
  --custom-token-regions "$CUSTOM" \
  --contrast-bins 0,1,236,249 \
  --component-modes "$MODES" \
  --device cuda:0 \
  --attn-implementation eager \
  --torch-dtype bfloat16
```

Val probe health:

```text
plan rows: 1
condition rows: 361
membership rows: 21
readout rows: 361
errors: 0
baseline: image 14439, sorted_denoise, target person coord_729
baseline top1: coord_0
baseline target rank: 441
```

Val full-basin reducer read:

```text
non-baseline rows: 360

effect labels:
  partial_rescue_no_top1_flip: 96
  target_rank_regression: 164
  wrong_basin_amplifier: 98
  margin_basin_conflict: 1
  target_logit_up_rank_down: 1

rank movement:
  improved: 96
  regressed: 263
  unchanged: 1

top1 movement:
  unchanged_wrong_top1: 360

expected distance:
  toward_target: 95
  away_from_target: 265

radius-16 target mass:
  toward_target: 110
  away_from_target: 250

margin/basin conflicts: 105
top1 flips: 0
```

Best val interventions:

```text
h12 h12_contrast_core, contrast 236:
  target_component_only: rank 441 -> 321
  remove_orthogonal_remainder: rank 441 -> 321
  top1 remains coord_0
  target logit +0.25
  all 4 tracked contrast margins improve
  expected distance and radius-16 mass move toward target

h12 h12_contrast_core, contrast 249:
  target_component_only: rank 441 -> 342
  remove_orthogonal_remainder: rank 441 -> 342

h12 h12_contrast_core, remove_full:
  rank 441 -> 348

h8 h8_target_core, add_full:
  rank 441 -> 391
  top1 remains coord_0
  target logit +0.1875
```

Worst val interventions:

```text
h12 h12_contrast_core, add_full:
  rank 441 -> 545
  top1 remains coord_0
  target logit -0.375
  all 4 tracked contrast margins worsen

h8 h8_target_core, remove_full:
  rank 441 -> 544
  top1 remains coord_0
  target logit -0.3125

h13 h13_target_core, target_component_only/remove_orthogonal_remainder:
  can regress to rank 506 at contrast 1
```

Token sign support for the val case:

```text
source token rows: 300
token groups: 127
mixed-sign tokens: 6

dominant roles:
  supports_target: 65
  supports_contrast: 56
  mixed_supports_target: 4
  mixed_supports_contrast: 2

top readout-supporting tokens are h8 and sign-consistent:
  h8 token 594 abs 0.1244, supports_target, sign consistency 1.0
  h8 token 634 abs 0.1002, supports_target, sign consistency 1.0
  h8 token 756 abs 0.0945, supports_target, sign consistency 1.0
  h8 token 383 abs 0.0906, supports_target, sign consistency 1.0
  h8 token 421 abs 0.0888, supports_target, sign consistency 1.0

h12 token 708 is the strongest mixed token:
  abs 0.0766
  positive 0.0166, negative -0.0600
  dominant role mixed_supports_contrast

h12 token 383 is strongly contrast-supporting:
  abs 0.0728, negative on all 4 contrast directions
```

Mechanism update from the val replication:

```text
The val case does not simply repeat the 418535 train analogue.

418535:
  - baseline is bad but not catastrophic: target rank 127, top1 coord_302;
  - the best clean partial rescue is h13 shared_top_abs target-component
    isolation, rank 127 -> 86;
  - h12 can become a severe wrong-basin amplifier and even flip top1.

14439:
  - baseline is much deeper in a border basin: target rank 441, top1 coord_0;
  - no tested intervention flips top1 away from coord_0;
  - h8 has the clearest sign-consistent readout target support and full-route
    addition helps modestly, rank 441 -> 391;
  - h12 contrast-core full addition damages the target, but isolating its
    target-direction component is the strongest partial rescue, rank 441 -> 321.

This suggests at least two false-negative submechanisms:
  1. mixed-head coordinate-basin competition, where a useful component can be
     isolated from h13 or h12 and partially rescue rank;
  2. deep border-basin lock, where visual target evidence exists but the top1
     coordinate basin is so stable that even strong component surgery only moves
     rank inside the long tail.

The surprising point is h12 in 14439. Its token readout labels are mostly
contrast-supporting, yet the h12 contrast-core contains the strongest isolated
target component. This reinforces the warning that "token supports contrast"
and "the same token has no useful target subcomponent" are different claims.
The relevant object signal can live as a small target-direction component inside
an otherwise suppressive visual route.
```

Next deterministic probes after the val replication:

```text
1. Run layer tomography for h12_contrast_core and h8_target_core on 14439.
   The key question is where the h12 target component is created or suppressed:
   value vector, o_proj head output, residual stream, MLP, or final coord readout.

2. Run a matched-control recomposition panel from the same representative
   selection family, preferably a simple matched control or same-image matched
   row, so we can tell whether deep coord_0 border lock is specific to false
   negatives or common in normal learned rows.

3. Run the same reducer on future val and train panels before interpreting any
   rank movement. Top1 and full-distribution movement must stay first-class.

4. Keep 2299 separate as a termination-boundary case. Its next probe should
   inspect stop/continue router tie semantics rather than reusing x1 visual-value
   surgery as if it were another ordinary false-negative coordinate basin.
```

## 14439 Layer-Tomography Recomposition

Status: completed focused layer sweep after the val replication.

Artifact:

```text
Recomposition:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_component_recompose_probe/v3_val_14439_layer_tomography_h8h12_gpu0

Full-basin reduction:
/data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_component_basin_reduce/v3_val_14439_layer_tomography_h8h12_gpu0
```

Command:

```text
CUSTOM='h8_target_core:594,634,756,383,421,420,597,632:far_background;h12_contrast_core:708,383,560,374,566,421,745,604,599,643:far_background'
REGIONS='h8_target_core,h12_contrast_core'
MODES='remove_full,add_full,target_component_only,remove_orthogonal_remainder'

CUDA_VISIBLE_DEVICES=0 python -m src.analysis.prefix_denoising_surgery_probing.fn_visual_value_component_recompose_probe \
  --plan-rows /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_token_probe/v2_sorted_14439_far_background_l16_h8h12h13_contrast0_1_249_236_gpu0/fn_visual_value_token_probe_plan_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/prefix_denoising_surgery_probing/fn_visual_value_component_recompose_probe/v3_val_14439_layer_tomography_h8h12_gpu0 \
  --image-root /data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60 \
  --selection-ranks 2 \
  --layers 8,12,16,20,24 \
  --heads 8,12 \
  --regions "$REGIONS" \
  --custom-token-regions "$CUSTOM" \
  --contrast-bins 0,236 \
  --component-modes "$MODES" \
  --device cuda:0 \
  --attn-implementation eager \
  --torch-dtype bfloat16
```

Probe health:

```text
plan rows: 1
condition rows: 161
membership rows: 12
readout rows: 161
errors: 0
non-baseline reduced rows: 160
baseline: target coord_729, top1 coord_0, target rank 441
```

Full-basin summary:

```text
effect labels:
  neutral_or_weak: 40
  partial_rescue_no_top1_flip: 32
  target_rank_regression: 72
  wrong_basin_amplifier: 14
  target_logit_up_rank_down: 2

rank movement:
  improved: 32
  regressed: 88
  unchanged: 40

top1 flips: 0
margin/basin conflicts: 14
```

Layer-localization read:

```text
L16/h12/h12_contrast_core is the dominant rescue/damage site:
  target_component_only, contrast 236:
    rank 441 -> 321
  remove_orthogonal_remainder, contrast 236:
    rank 441 -> 321
  remove_full:
    rank 441 -> 348
  add_full:
    rank 441 -> 545

L16/h8/h8_target_core is the clean full-route support site:
  add_full:
    rank 441 -> 391
  remove_full:
    rank 441 -> 544
  target_component_only/remove_orthogonal_remainder:
    can regress, especially contrast 236 -> rank 537

Other sampled layers are much weaker:
  L8 mostly neutral or damaging; h8 target add_full only rank 441 -> 440.
  L12 is weak/damaging except a small L12/h12/h8_target_core component
    improvement to rank 429.
  L20 has only small effects, e.g. L20/h8/h8_target_core component
    improvement to rank 433.
  L24 is mostly neutral or mildly damaging.
```

Mechanism update from tomography:

```text
For 14439, the useful and harmful visual-value control is not spread evenly
through the decoder. It is sharply localized around layer 16.

The h8 route behaves like a relatively clean full-route target support at L16:
adding the full h8_target_core contribution helps, and removing it hurts. But
isolating only the pulled-back target component can hurt, which means the useful
h8 signal is not aligned to the simple target-minus-contrast axis we used for
decomposition.

The h12 route behaves like a suppressive mixed route at L16: adding the full
h12_contrast_core hurts badly, removing it helps, and isolating only its
target-direction component helps most. This is a stronger form of the "small
target component inside an otherwise suppressive route" hypothesis.

The layer-localization matters for the final picture: this is not just a final
LM-head readout artifact and not just raw visual perception. A mid-decoder
attention value route around L16 appears to be a bottleneck where visual
evidence, contrast/border attraction, and coordinate-token basin geometry are
mixed before the later residual stream commits to coord_0.
```

Next deterministic step:

```text
Run the same L16-centered h8/h12 layer-tomography recipe on one matched-control
sample-base from the representative panel. If the L16 suppressive-mixed h12
route appears only in false-negative/border-lock rows, it becomes a stronger
candidate mechanism. If it also appears in clean matched rows, then the failure
is more likely downstream basin selection or continuation-state competition.
```
