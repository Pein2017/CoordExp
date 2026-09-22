---
doc_id: progress.diagnostics.source36_coordinate_threshold_hidden_state_readout
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: tiny-one-state-desc-first-source36-object14-y2-hidden-readout
domain: autoregressive-binding-template-ablation
updated: 2026-06-20
branch: codex/autoregressive-binding-template-study
---

# Source-36 Coordinate Threshold Hidden-State Readout

## Scope

This is a tiny follow-up to the strict object-step projection/residual
factorial note:

```text
progress/diagnostics/2026-06-20_strict_object_step_projection_residual_factorial_findings.md
```

The target state is the sharp source-36/object-14 `y2` case:

```text
family: desc_first
source_line_idx: 36
image_id: 3255
object_idx: 14
selection_event: pre_y2
target_next_coord_slot: y2
target_next_coord_bin: 996
state_key: trajectory_object_step_v2||family=desc_first||source_line_idx=36||image_id=3255||object_idx=14||selection_event=pre_y2||role_index=141||prefix_sha1=5f237191dd93a45a
```

This is not population evidence. It is a one-state microscope pass meant to
localize where the coordinate basin appears and how it relates to the earlier
projection/residual threshold observation.

## Helper Added

A small artifact helper was added:

```text
src/analysis/autoregressive_binding_template_ablation/coordinate_threshold_hidden_state_probe.py
scripts/analysis/run_autoregressive_binding_coordinate_threshold_hidden_state_probe.py
tests/analysis/test_coordinate_threshold_hidden_state_probe.py
```

It provides two cheap post-hoc surfaces:

```text
--stage build-panel
--stage summarize-hidden-state-readout
```

The summary table is coordinate-oriented: `target_rank` and `target_prob` prefer
coord-only readout fields over full-vocab fields.

Review/verification:

```text
spec review: passed after coord-only rank/prob precedence fix
code-quality review: passed; minor CLI/order hardening suggestions accepted
python -m pytest tests/analysis/test_coordinate_threshold_hidden_state_probe.py
  Pytest: 7 passed
python -m py_compile src/analysis/autoregressive_binding_template_ablation/coordinate_threshold_hidden_state_probe.py scripts/analysis/run_autoregressive_binding_coordinate_threshold_hidden_state_probe.py tests/analysis/test_coordinate_threshold_hidden_state_probe.py
  exit 0
python scripts/analysis/run_autoregressive_binding_coordinate_threshold_hidden_state_probe.py --help
  exit 0
```

## Artifacts

Source panel:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source36_object14_y2_panel_v1
```

Files:

```text
selected_rows.jsonl
threshold_probe_manifest.json
```

Hidden-state readout:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_probe/source36_object14_y2_threshold_layers0_4_8_12_16_20_24_28_v1
```

Helper summary:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source36_object14_y2_hidden_state_summary_v1
```

Hidden-delta readouts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_delta_probe/source36_object14_y2_s20_t24_coord_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_delta_probe/source36_object14_y2_s24_t28_coord_v1
```

Causal activation patch probes:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/source36_object14_y2_s20_t24_coord_probe996_999_997_234_456_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/source36_object14_y2_s24_t28_coord_probe996_999_997_234_456_v1
```

Post-hoc causal probe-support reducers:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_causal_patch_probe_support/source36_object14_y2_s20_t24_coord_probe996_999_997_234_456_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_causal_patch_probe_support/source36_object14_y2_s24_t28_coord_probe996_999_997_234_456_v1
```

All model-backed runs used:

```text
pair config: configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
family: desc_first
intervention_arm: natural
target_next_kind: coord
stop_reason: object_step_role
```

## Hidden-State Readout

Run:

```text
stage: trajectory-hidden-state-readout
hidden_layers: 0,4,8,12,16,20,24,28
device: cuda:0
row_count: 8
state_row_count: 1
readout_status_counts: {"ok": 8}
model_perturbation_ran: false
training_ran: false
```

Layer table:

| layer | target | coord top1 | top1 delta | target rank | target prob | coord mass | radius4 mass | surface top1 | adapter delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 996 | 456 | 540 | 790 | 0.000000 | 1.000000 | 0.000000 | 456 | -0.036 |
| 4 | 996 | 811 | 185 | 398 | 0.000645 | 0.000200 | 0.008911 | 850 | 0.064 |
| 8 | 996 | 846 | 150 | 392 | 0.000778 | 0.011215 | 0.006321 | 876 | -0.022 |
| 12 | 996 | 987 | 9 | 265 | 0.001188 | 0.001874 | 0.007250 | 876 | 0.158 |
| 16 | 996 | 942 | 54 | 383 | 0.001025 | 0.005589 | 0.008458 | 367 | -0.238 |
| 20 | 996 | 959 | 37 | 290 | 0.000946 | 0.132235 | 0.012079 | 866 | -0.175 |
| 24 | 996 | 997 | 1 | 6 | 0.045102 | 0.056593 | 0.394107 | 999 | 12.905 |
| 28 | 996 | 999 | 3 | 5 | 0.049927 | 0.999998 | 0.885170 | 999 | 38.565 |

Derived conservative hints:

```text
first_layer_abs_top1_minus_target_le_4: 24
first_layer_target_rank_le_5: 28
final_layer_top1: 999
final_layer_target_rank: 5
```

Interpretation:

```text
Layer 24 is the first observed layer where the coordinate readout is locally
near the y2 target. Layer 28 is where coord mass becomes almost fully saturated,
but the basin peak is <|coord_999|>, not the target <|coord_996|>.
```

This supports a late coordinate-basin picture:

```text
early/mid layers: weak or unrelated coordinate readout
layer 24: near-target basin appears
layer 28: coordinate emission surface saturates and sharpens toward 999
```

## Important Replay Caveat

The selected-row artifact carries `trajectory_greedy_token_text=<|coord_996|>`
from its original object-step trace. The current bridge replay/readout baseline
for the same state is already in the `<|coord_999|>` basin:

```text
layer 28 coord top1: 999
coord-only p(999): 0.473696
coord-only p(997): 0.119769
coord-only p(996): 0.049927
```

This is consistent with the earlier strict-factorial note, where zero-patch and
single-component settings also surfaced `<|coord_999|>` despite the selected
trace target being `996`. Treat this as a decode/replay-surface mismatch signal,
not as proof that the original rollout emitted 999.

## Hidden-Delta Readout

Two paired-layer delta readouts were run:

```text
20 -> 24 on cuda:0
24 -> 28 on cuda:1
```

The existing hidden-delta analyzer is mostly a schema-boundary lens, but it
still provides useful structure:

```text
20 -> 24:
  hidden_delta_norm: 989.872
  hidden_delta_over_source_norm: 2.099
  box_end_minus_object_ref_start: 1.703 -> 4.424
  delta margin: +2.721

24 -> 28:
  hidden_delta_norm: 2235.496
  hidden_delta_over_source_norm: 1.953
  box_end_minus_object_ref_start: 4.424 -> 0.935
  delta margin: -3.489
```

This does not directly explain coordinate target rank, but it suggests the
coordinate-basin transition and schema-boundary pressure are not a single
monotone late-layer variable. The `20 -> 24` step increases box-end structural
margin while local coordinate readout moves near the target; `24 -> 28` reduces
that structural margin while the final coord-token surface becomes much sharper.

## Causal Patch Probe

Two raw decoder-layer activation patch probes were run with:

```text
patch alphas: 0,0.5,1.0
direction strengths: 0,64,128
direction bases:
  box_end_minus_object_ref_start
  box_end_minus_object_ref_boundaries
  box_end_minus_im_end
probe coord bins: 996,999,997,234,456
```

Counters:

```text
20 -> 24:
  row_count: 14
  realized_direction_patch_count: 9
  patched_coord_probe_readout_status_counts: {"ok": 14}

24 -> 28:
  row_count: 14
  realized_direction_patch_count: 9
  patched_coord_probe_readout_status_counts: {"ok": 14}
```

Main result:

```text
All tested patches keep <|coord_999|> as the top coordinate bin.
```

Examples:

```text
baseline:
  p999 0.473696 rank 1
  p997 0.119769 rank 2
  p996 0.049927 rank 5
  p234 ~5.98e-09 rank 494

24 -> 28 interpolate_source_to_target_alpha_0:
  p999 0.685010 rank 1
  p997 0.016110 rank 4
  p996 0.012546 rank 9
  p234 ~1.06e-06 rank 397

24 -> 28 interpolate_source_to_target_alpha_0p5:
  p999 0.722674 rank 1
  p997 0.024729 rank 3
  p996 0.021823 rank 5
  p234 ~1.04e-07 rank 394
```

Adding standard box-end directions changes the structural box-end margin but
does not move the coordinate basin away from 999:

```text
24 -> 28 add_box_end_direction_128:
  box_end margin delta: +4.362
  p999 0.491529 rank 1
  p997 0.096788 rank 2
  p996 0.040347 rank 5

24 -> 28 add_box_end_minus_object_ref_boundaries_direction_128:
  box_end margin delta: +3.620
  p999 0.517412 rank 1
  p997 0.101884 rank 2
  p996 0.037481 rank 5
```

The far catastrophic `<|coord_234|>` basin from the earlier value-region
factorial is not reproduced by these standard structural direction patches:

```text
coord_234 rank: about 394 to 581 across tested rows
coord_234 coord-only probability: about 1e-09 to 1e-06
```

## Current Mechanistic Reading

This tiny readout supports the following narrower picture:

```text
1. A near-target y2 basin forms late, around layer 24.
2. The final emission surface sharpens the basin into a ceiling-side attractor:
   996 remains rank 5, but 999 is rank 1 with much larger probability.
3. The token_embeddings_adapter surface is not passive here: adapter target
   delta jumps from approximately 0 near layers 0-20 to +12.9 at layer 24 and
   +38.6 at layer 28.
4. Ordinary box-end structural direction injection changes schema margin but
   does not explain the 999 coordinate attraction.
5. The earlier far jump to 234 likely requires the harmful attention value
   route's projection/residual interaction, not just generic structural
   box-end pressure.
```

This is a promising bridge between the coordinate-token basin question and the
value-region factorial result:

```text
natural/replay surface: near-ceiling 997/999 attraction
harmful value-region projection/residual grid: can cross into far 234 basin
```

The next useful probe should therefore be coordinate-specific, not only
schema-direction-specific:

```text
1. Build a paired coord-basin reducer for causal patch rows that explicitly
   compares target, near-ceiling, and far-attractor bins.
2. Run aligned source-36-like rows for source 123 and source 145 to see whether
   their residual-sufficient behavior shares the layer-24 basin onset.
3. Repeat the hidden-state readout on geometry_first aligned states by image
   and realized slot, not by copied source/object indices.
4. If implementing new causal patch modes, prefer coordinate-basis directions:
   target-vs-ceiling coord embedding/logit gradients or learned adapter
   residual directions, rather than only box-end structural directions.
```

## Comparator y2 Readouts

After the source-36 microscope pass, the same layer grid was run on the two
other y2 states called out by the strict-factorial note as residual-sufficient
or residual-dominated comparators.

Panels:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source123_object5_y2_panel_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source145_object20_y2_panel_v1
```

Hidden-state readouts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_probe/source123_object5_y2_threshold_layers0_4_8_12_16_20_24_28_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_probe/source145_object20_y2_threshold_layers0_4_8_12_16_20_24_28_v1
```

Helper summaries:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source123_object5_y2_hidden_state_summary_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source145_object20_y2_hidden_state_summary_v1
```

Compact final-layer comparison:

| source | image | object | target y2 | first radius4 layer | first rank<=5 layer | final coord top1 | final top1 delta | final target rank | final target prob | final coord mass | final radius4 mass | final adapter delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 36 | 3255 | 14 | 996 | 24 | 28 | 999 | 3 | 5 | 0.049927 | 0.999998 | 0.885170 | 38.565 |
| 123 | 12670 | 5 | 270 | none | none | 259 | 11 | 12 | 0.014741 | 0.999956 | 0.093611 | 47.911 |
| 145 | 15254 | 20 | 146 | none | 24 | 154 | 8 | 6 | 0.052712 | 0.999998 | 0.410379 | 61.760 |

Comparator layer-24 rows:

| source | target y2 | layer24 top1 | layer24 delta | layer24 target rank | layer24 target prob | layer24 coord mass | layer24 radius4 mass | layer24 adapter delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 36 | 996 | 997 | 1 | 6 | 0.045102 | 0.056593 | 0.394107 | 12.905 |
| 123 | 270 | 188 | 82 | 97 | 0.002970 | 0.049369 | 0.016432 | 7.265 |
| 145 | 146 | 158 | 12 | 5 | 0.029383 | 0.683321 | 0.178640 | 22.699 |

Comparator read:

```text
The layer-24/28 late-basin pattern is not unique to source 36, but its strength
is state-dependent. Source 36 is the cleanest near-target basin: radius4 support
appears at layer 24 and stays strong at layer 28. Source 145 has rank support by
layer 24 but remains spatially offset by 8-12 bins. Source 123 has a much weaker
target basin under the same replay surface: final target rank is 12 and final
top1 is still 11 bins away.
```

This makes the next mechanistic split sharper:

```text
source 36: late near-target basin plus final 999 ceiling attraction
source 145: late target-rank support but shifted basin center
source 123: weak/diffuse target support despite coord-mass saturation
```

The three states all show large final `token_embeddings_adapter` deltas:

```text
source36 final adapter delta: +38.565
source123 final adapter delta: +47.911
source145 final adapter delta: +61.760
```

So adapter magnitude alone is not sufficient to predict correct basin centering.
The more precise question is what determines the center of the late coordinate
basin after coord-mass saturation.
