---
doc_id: progress.diagnostics.train_destination_component_and_val_mirror_findings_2026_06_22
date: 2026-06-22
scope: ckpt928 bbox_len12000 token_embeddings_adapter staged-slot mechanism probes
status: current
---

# Train Destination Components And Val Mirror

## Question

The current train-only per256 failure chain showed many trained-sequence
pre-x1 coordinate-onset failures. The next question is whether these failures
are a narrow anecdote or a broader row-population effect, and whether the
destination-family mechanism differs between train failures and unseen val
analogs.

This note records the completed train-only component localization and the
in-progress val mirror setup.

## Tooling Change

The guided-delta patch runner now supports destination-family filtering and
summary grouping:

```text
src/analysis/autoregressive_binding_template_ablation/staged_slot_guided_delta_patch.py
tests/analysis/test_staged_slot_guided_delta_patch.py
```

New CLI/filter surface:

```text
--slot-destination-families baseline_inertia,donor_slot_capture,control_slot_pull
```

The plan/output rows preserve panel-side `slot_destination_*`, `slot_phase_*`,
and related launch context fields so destination-family labels remain attached
after the staged receiver row is resolved.

Verification:

```bash
python -m pytest tests/analysis/test_staged_slot_guided_delta_patch.py -q
python -m py_compile src/analysis/autoregressive_binding_template_ablation/staged_slot_guided_delta_patch.py scripts/analysis/run_autoregressive_binding_staged_slot_guided_delta_patch.py
```

Result:

```text
15 passed
py_compile passed
```

## Train-Only Destination-Family Component Localization

Input destination panel:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_destination_panel_selector/v2_v3_per256_train_only_destination_uncapped/staged_slot_destination_panel_rows.jsonl
```

Input staged rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_rows/v7_v8_per256_train_only_failure_panel_all_slots/staged_slot_readout_rows.jsonl
```

Panel slice:

```text
split: train
donor_position: staged_after_x1_y1
delta_mode: donor_minus_previous_slot
patch_scale: 1.0
layer_index: 24
slot_destination_families:
  baseline_inertia
  donor_slot_capture
  control_slot_pull
```

Dry plan artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v21_v2_train_destination_core_families_prevslot_afterx1y1_layerinput_plan
```

Plan rows:

```text
187 total
baseline_inertia: 108
donor_slot_capture: 47
control_slot_pull: 32
```

Model-forward artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v21_v2_train_destination_core_families_prevslot_afterx1y1_layerinput
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v22_v2_train_destination_core_families_prevslot_afterx1y1_selfattn
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v23_v2_train_destination_core_families_prevslot_afterx1y1_mlp
```

All three component sites completed:

```text
layer_input: 187 patch rows, 0 errors
self_attn:   187 patch rows, 0 errors
mlp:         187 patch rows, 0 errors
```

### Overall Component Readout

```text
site         improved  d16     intrusion  donor_nearer  mean_rank_delta
layer_input 0.3476    0.0000  1.0000     1.0000        +3582.47
self_attn   0.4118    0.0107  0.8930     0.8930        -1581.22
mlp         0.6096    0.0107  0.8824     0.8824        +261.03
```

`mean_rank_delta` is patched receiver target-token rank minus baseline
receiver target-token rank, so negative is better.

The whole residual-stream patch is maximally intrusive and never lands within
16 coordinate bins of the receiver target. Component output patches are less
intrusive, but they still almost never repair coordinate distance.

### Destination-Family Split

```text
site         family              n    improved  d16     intrusion  mean_rank_delta
layer_input baseline_inertia     108  0.3241    0.0000  1.0000     +3442.73
layer_input control_slot_pull    32   0.1875    0.0000  1.0000     +3271.88
layer_input donor_slot_capture   47   0.5106    0.0000  1.0000     +4115.02
self_attn   baseline_inertia     108  0.2870    0.0000  1.0000     -38.24
self_attn   control_slot_pull    32   0.4062    0.0000  1.0000     -8.91
self_attn   donor_slot_capture   47   0.7021    0.0426  0.5745     -6197.30
mlp         baseline_inertia     108  0.5833    0.0000  1.0000     +1323.99
mlp         control_slot_pull    32   0.5000    0.0000  1.0000     -484.12
mlp         donor_slot_capture   47   0.7447    0.0426  0.5319     -1674.19
```

## Interpretation

This is no longer a single object-pair anecdote. The patch was run over 187
trained-sequence failure rows selected from three destination families.

The train-only component result suggests a destination-family mechanism fork:

1. `baseline_inertia` and `control_slot_pull` are sticky across component
   sites. Even when component patches improve target rank, their patched top-1
   coordinate remains nearer to the intrusive destination 100 percent of the
   time in this slice. This points to a strong receiver-side coordinate-basin
   inertia or previous-slot anchoring effect, not a clean missing-coordinate
   repair.
2. `donor_slot_capture` is component-sensitive. At layer 24, self-attention and
   MLP output patches both reduce intrusion compared with whole residual-stream
   patching, and both improve target rank in about 70 to 75 percent of rows.
   This makes donor-slot capture the best target for localizing where an
   object/slot transition vector is written.
3. Successful exact repair is still rare. The best `coord_distance_lte_16`
   rate is only 0.0426 in `donor_slot_capture`, and zero for the other two
   families. The model can move rank without reliably entering the correct
   coordinate basin.

Working hypothesis update:

The next mechanistic picture should distinguish at least two failure modes:

```text
sticky-basin failures:
  baseline_inertia, control_slot_pull
  likely dominated by local coordinate basin inertia or previous-slot anchoring

reroutable slot-transition failures:
  donor_slot_capture
  partially localized to self-attention/MLP component outputs at layer 24
```

This supports deeper probing by destination family, not by aggregate
pre-x1 failure severity alone.

## Val Mirror Setup

The v8 candidate bank is already broad and balanced:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/train_val_candidate_bank/v8_bbox_len12000_regimes_per256_imgcap2_desccap12

3072 candidates
1536 train
1536 val
256 per split per regime
```

The formation rows are also balanced:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/panel_formation_row_builder/v8_v8_per256_candidate_bank_teacher_forced_gt_desccap12

24576 rows
12288 train
12288 val
8 positions per candidate
```

The current readout/failure/staged/destination chain after v25 is train-only,
so the existing per256 destination taxonomy is not train-vs-val evidence.

Val robust-position dry run:

```bash
python scripts/analysis/run_autoregressive_binding_formation_readout_probe.py \
  --formation-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/panel_formation_row_builder/v8_v8_per256_candidate_bank_teacher_forced_gt_desccap12/panel_formation_position_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v26_v8_per256_val_robust_positions_dry_run \
  --positions pre_x1,post_x1,box_close,next_object_onset \
  --splits val \
  --layers 24,-1 \
  --dry-run
```

Dry-run result:

```text
selected_row_count: 6144
splits: val
positions: pre_x1, post_x1, box_close, next_object_onset
regime rows:
  crowded: 1024
  duplicate_basin_nearby: 1024
  repeated_class: 1024
  simple_control: 1024
  small_object: 1024
  termination_tail: 1024
```

Val readout shards completed and were merged under:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v27_v8_per256_val_robust_positions_sharded3_image_root
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v27_v8_per256_val_robust_positions_sharded3_image_root/merged
```

Merged result:

```text
selected_row_count: 6144
output_row_count: 6144
hidden_row_count: 12288
error_count: 0
```

Train/val robust-position readout comparison:

```text
split  position  rank<=10  d<=16   mean_rank  mean_dist
train  pre_x1    0.3151    0.4473  132.58     116.13
val    pre_x1    0.2617    0.4486  131.54     101.82
train  post_x1   0.6400    0.8359  16.08      10.49
val    post_x1   0.5794    0.8216  18.91      13.04
```

Val is not simply worse. It is slightly lower on exact pre-x1 rank, similar on
pre-x1 coordinate distance, worse after x1 guidance, and in small-object cases
less catastrophic by mean pre-x1 rank than the selected train failures.

Combined train+val readout artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v28_v25_v27_per256_train_val_robust_positions_combined

row_count: 12288
case_count: 3072
train rows: 6144
val rows: 6144
```

Raw matched failure panel:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_failure_panel/v9_v28_per256_prex1_train64_val1_analog_panel

row_count: 692
train_seed_count: 346
val_analog_count: 346
matched_pair_count: 346
train_candidate_count: 986
val_candidate_count: 1031
same_regime_same_desc: 254
same_regime: 92
```

Raw matched panel split means:

```text
split  n    mean_score  mean_rank  mean_dist  post_rank  post_dist
train  346  753.16      390.25     362.91     20.77      14.78
val    346  363.47      195.38     168.09     19.97      14.37
```

The raw analogs are broad and mostly non-person:

```text
rows: 692
unique descs: 71
person rows: 42
non-person rows: 650
```

The raw analog panel is useful for broad symptom comparison, but loose for
causal claims. The mean absolute analog deltas are large for box area, center,
and pre-x1 failure severity.

Filtered launch panel:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_failure_launch_filter/v3_v9_per256_train64_val1_loose_balanced_pairs16

pair_count: 82
row_count: 164
train rows: 82
val rows: 82
same_regime_same_desc: 77
same_regime: 5
```

Filtered pair counts:

```text
crowded: 16
duplicate_basin_nearby: 16
repeated_class: 16
small_object: 16
termination_tail: 16
simple_control: 2
```

Filtered match quality:

```text
mean_abs_delta object_count: 6.79
mean_abs_delta tail: 3.96
mean_abs_delta same_desc_count: 1.07
mean_abs_delta bbox_area: 10310.02
mean_abs_delta center_l1_x2: 821.51
mean_abs_delta post_x1_rank: 15.24
mean_abs_delta post_x1_distance: 9.34
```

Staged-slot rows for the filtered launch panel:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_rows/v8_v3_per256_train_val_launch_filter_all_slots

output_row_count: 820
```

Matched train/val guided-delta dry plan:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v24_v3_per256_train_val_launch_filter_prevslot_afterx1y1_layerinput_plan

plan_row_count: 164
train rows: 82
val rows: 82
donor_position: staged_after_x1_y1
delta_mode: donor_minus_previous_slot
patch_scale: 1.0
layer_index: 24
```

Matched train/val component-site runs:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v24_v3_per256_train_val_launch_filter_prevslot_afterx1y1_layerinput
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v25_v3_per256_train_val_launch_filter_prevslot_afterx1y1_selfattn
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v26_v3_per256_train_val_launch_filter_prevslot_afterx1y1_mlp
```

All three completed:

```text
layer_input: 164 patch rows, 0 errors
self_attn:   164 patch rows, 0 errors
mlp:         164 patch rows, 0 errors
```

Matched-panel component summary:

```text
site         split  n   improved  d16     intrusion  mean_rank_delta
layer_input train  82  0.4756    0.0854  0.6829     +2218.40
layer_input val    82  0.4390    0.0976  0.6585     +994.93
self_attn   train  82  0.4878    0.0488  0.5976     +337.24
self_attn   val    82  0.5000    0.1098  0.5122     -1347.29
mlp         train  82  0.5610    0.0854  0.5976     -333.07
mlp         val    82  0.5732    0.0610  0.5366     -329.24
```

`mean_rank_delta` is patched receiver target-token rank minus baseline
receiver target-token rank, so negative is better.

The matched train/val result does not support a simple train-memorized versus
val-unseen split. The matched val analogs are at least as responsive as train
under self-attention and MLP output patches, and val has slightly lower
intrusion in all three component sites. The larger raw train failure scores are
therefore better interpreted as severity/selection differences over a shared
slot-basin mechanism than as evidence that val objects are uniquely visually
unperceived.

Post-hoc destination artifacts:

```text
contrast:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_delta_contrast_selector/v7_v24_train_val_layerinput_uncapped
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_delta_contrast_selector/v8_v25_train_val_selfattn_uncapped
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_delta_contrast_selector/v9_v26_train_val_mlp_uncapped

geometry:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_coord_value_geometry/v4_v7_train_val_layerinput_geometry
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_coord_value_geometry/v5_v8_train_val_selfattn_geometry
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_coord_value_geometry/v6_v9_train_val_mlp_geometry

destination:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_destination_panel_selector/v3_v4_train_val_layerinput_destination_uncapped
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_destination_panel_selector/v4_v5_train_val_selfattn_destination_uncapped
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_destination_panel_selector/v5_v6_train_val_mlp_destination_uncapped
```

Destination-family counts:

```text
site         family                    train  val   total
layer_input baseline_inertia           24     25    49
layer_input off_basin_escape           18     17    35
layer_input donor_slot_capture         11     7     18
layer_input control_slot_pull          5      5     10
layer_input coordinate_edge_ambiguity  4      3     7
layer_input receiver_repair            2      3     5

self_attn   baseline_inertia           35     28    63
self_attn   control_slot_pull          10     9     19
self_attn   off_basin_escape           6      5     11
self_attn   receiver_repair            0      5     5
self_attn   donor_slot_capture         1      1     2
self_attn   coordinate_edge_ambiguity  1      0     1

mlp         baseline_inertia           25     26    51
mlp         off_basin_escape           15     8     23
mlp         control_slot_pull          9      9     18
mlp         donor_slot_capture         3      3     6
mlp         receiver_repair            3      2     5
mlp         coordinate_edge_ambiguity  2      1     3
```

Destination interpretation:

1. `baseline_inertia` is the dominant selected destination family in the
   matched train/val panel for all three component sites.
2. `self_attn` almost eliminates `donor_slot_capture` as a landing family
   compared with whole residual-stream patching, but it increases
   `baseline_inertia` and `control_slot_pull`. This suggests self-attention
   participates in suppressing direct donor-slot capture while leaving sticky
   coordinate basins and previous-slot anchoring intact.
3. MLP behaves similarly to self-attention in split balance, but has more
   off-basin escape and slightly more donor-slot capture. This makes MLP a
   likely write surface for rerouting, but not a reliable repair surface.
4. Train and val have very similar destination-family profiles after launch
   filtering. The useful axis is therefore destination family and component
   site, not train versus val by itself.

## Revised Next Directions

1. Treat train and val as a shared mechanism population, with train providing
   more severe seeds and val providing matched analog controls.
2. Move the next deep probe from train-vs-val prevalence to destination-family
   mechanism:
   `baseline_inertia`, `control_slot_pull`, `off_basin_escape`, and
   `donor_slot_capture`.
3. For `baseline_inertia`, test whether the top coordinate basin is locked by
   local prior coordinate history, visual-region evidence, or late unembedding
   geometry. Good next probes are layer scans, pre/post component patching,
   and logit-lens basin tracking across layers.
4. For `control_slot_pull`, explicitly patch or ablate the previous-slot
   control coordinate state and test whether y1-like anchoring is written
   before x1 onset.
5. For `donor_slot_capture`, keep component localization alive, but treat it as
   a smaller reroutable subset rather than the main failure population.
6. For `small_object`, continue coordinate-edge and width-bucket analysis,
   because it is the best place to separate true object binding from coordinate
   quantization/smoothness loss.
7. For `termination_tail`, test object-boundary state and stop/continue
   competition, because it consistently appears in both severe train failures
   and destination-family counts.
