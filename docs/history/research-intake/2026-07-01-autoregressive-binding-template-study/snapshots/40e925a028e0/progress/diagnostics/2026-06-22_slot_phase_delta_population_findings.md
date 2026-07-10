---
doc_id: progress.diagnostics.slot_phase_delta_population_findings_2026_06_22
date: 2026-06-22
scope: causal-readout, teacher-forced, bbox_len12000, train-val, checkpoint-928, no-training
status: active-evidence
---

# Slot-Phase Delta Population Findings

This note records the first population-shaped slot-phase disentanglement run
over the train-failure/val-analog launch cohorts. It follows the decision to
move beyond the original person/backpack microscope and explicitly compare
trained rows against held-out validation analogs.

## Code Surface

```text
src/analysis/autoregressive_binding_template_ablation/staged_slot_guided_delta_patch.py
src/analysis/autoregressive_binding_template_ablation/staged_slot_readout_rows.py
src/analysis/autoregressive_binding_template_ablation/staged_slot_coord_value_geometry.py
scripts/analysis/run_autoregressive_binding_staged_slot_guided_delta_patch.py
scripts/analysis/run_autoregressive_binding_staged_slot_coord_value_geometry.py
tests/analysis/test_staged_slot_guided_delta_patch.py
tests/analysis/test_staged_slot_readout_rows.py
tests/analysis/test_staged_slot_coord_value_geometry.py
```

New behavior:

```text
launch-filter panel rows without state_key are resolved through panel_case_id,
receiver_case_id, or source_readout_case_id.

staged-slot row building now also accepts source_readout_case_id and
candidate_bank_case_id as panel/source identity fields, and preserves
source_readout_case_id, matched_train_case_id, and analog_match_tier from
train/val failure panels.

--delta-modes raw_donor_minus_receiver,donor_minus_previous_slot
```

The new `donor_minus_previous_slot` mode computes:

```text
donor_hidden - control_hidden
```

where the control is the immediately previous staged slot from the same
`panel_case_id`. This is intended to reduce direct donor-state import and ask
whether a slot-transition vector can repair pre-x1 coordinate onset with less
donor-basin capture.

For `staged_after_x1`, the previous slot is `staged_pre_x1`, which is also the
receiver. Therefore `donor_minus_previous_slot` is intentionally identical to
the raw delta for that donor position. The meaningful disentanglement contrast
starts at `staged_after_x1_y1`, where the control is `staged_after_x1`.

## Artifact Roots

Broad launch cohort plan:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v10_v1_launch_raw_vs_prevslot_plan
```

Same-desc launch cohort plan:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v11_v2_same_desc_raw_vs_prevslot_plan
```

Broad launch cohort smoke:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v12_v1_launch_raw_vs_prevslot_smoke4_g7
```

Broad launch cohort full merged artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v13_v1_launch_raw_vs_prevslot_sharded4/merged
```

Same-desc launch cohort full merged artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v14_v2_same_desc_raw_vs_prevslot_sharded4/merged
```

Expanded v6 train-first failure / val-analog staged rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_rows/v4_v6_full_train_first_failure_panel
```

Expanded v6 train-first failure / val-analog guided-delta plan:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v15_v6_full_prevslot_afterx1y1_plan
```

Expanded v6 train-first failure / val-analog guided-delta full merged artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v16_v6_full_prevslot_afterx1y1_sharded3/merged
```

Expanded v6 clean/intrusive coordinate-value geometry reducer:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_coord_value_geometry/v1_v4_uncapped_prevslot_geometry
```

## Broad Launch Cohort

Scope:

```text
panel: v1_v6_train16_val2_loose_balanced_pairs8
staged rows: v2_v1_launch_filter_broad_balanced_pairs8
receiver states: 80
patch rows: 320
train rows: 160
val rows: 160
donor positions: staged_after_x1, staged_after_x1_y1
delta modes: raw_donor_minus_receiver, donor_minus_previous_slot
patch scale: 1.0
error_count: 0
```

Aggregate:

```text
target_rank_improved_rate: 0.5625
target_top1_rate: 0.00625
coord_rank_lte_10_rate: 0.0375
coord_distance_lte_16_rate: 0.146875
donor_target_top1_rate: 0.090625
donor_nearer_than_receiver_rate: 0.89375
slot_intrusion_rate: 0.9
mean_receiver_target_rank_delta: +568.453125
mean_receiver_coord_distance_delta: -52.18125
```

By delta mode:

```text
raw_donor_minus_receiver:
  row_count: 160
  target_rank_improved_rate: 0.5625
  coord_distance_lte_16_rate: 0.19375
  coord_rank_lte_10_rate: 0.01875
  donor_nearer_than_receiver_rate: 0.96875
  slot_intrusion_rate: 0.975

donor_minus_previous_slot:
  row_count: 160
  target_rank_improved_rate: 0.5625
  coord_distance_lte_16_rate: 0.1
  coord_rank_lte_10_rate: 0.05625
  donor_nearer_than_receiver_rate: 0.81875
  slot_intrusion_rate: 0.825
```

By split:

```text
train:
  target_rank_improved_rate: 0.7125
  coord_distance_lte_16_rate: 0.1375
  donor_nearer_than_receiver_rate: 0.88125
  slot_intrusion_rate: 0.89375
  mean_receiver_target_rank_delta: +151.3375

val:
  target_rank_improved_rate: 0.4125
  coord_distance_lte_16_rate: 0.15625
  donor_nearer_than_receiver_rate: 0.90625
  slot_intrusion_rate: 0.90625
  mean_receiver_target_rank_delta: +985.56875
```

By regime:

```text
small_object:
  target_rank_improved_rate: 0.8125
  coord_distance_lte_16_rate: 0.28125
  slot_intrusion_rate: 0.890625
  mean_receiver_target_rank_delta: -196.59375

duplicate_basin_nearby:
  target_rank_improved_rate: 0.40625
  coord_distance_lte_16_rate: 0.171875
  slot_intrusion_rate: 0.828125
  mean_receiver_target_rank_delta: +1393.703125

crowded:
  target_rank_improved_rate: 0.535714
  coord_distance_lte_16_rate: 0.071429
  slot_intrusion_rate: 0.964286
  mean_receiver_target_rank_delta: +28.714286

termination_tail:
  target_rank_improved_rate: 0.421875
  coord_distance_lte_16_rate: 0.09375
  slot_intrusion_rate: 0.859375
  mean_receiver_target_rank_delta: +321.09375
```

Clean improvements, defined as receiver target rank improves and
`slot_intrusion` is false:

```text
25 / 320 = 0.0781
clean coord-close improvements: 9 / 320 = 0.0281
```

These clean improvements are dominated by:

```text
donor_position: staged_after_x1_y1
delta_mode: donor_minus_previous_slot
```

The strongest clean rows include train small-object, train duplicate-nearby,
and some val termination-tail/duplicate-nearby rows.

## Same-Desc Cohort

Scope:

```text
panel: v2_v6_train16_val2_loose_same_desc_pairs8
staged rows: v3_v2_launch_filter_same_desc_pairs8
receiver states: 58
patch rows: 232
train rows: 116
val rows: 116
donor positions: staged_after_x1, staged_after_x1_y1
delta modes: raw_donor_minus_receiver, donor_minus_previous_slot
patch scale: 1.0
error_count: 0
```

Aggregate:

```text
target_rank_improved_rate: 0.564655
target_top1_rate: 0.0
coord_rank_lte_10_rate: 0.034483
coord_distance_lte_16_rate: 0.159483
donor_target_top1_rate: 0.064655
donor_nearer_than_receiver_rate: 0.896552
slot_intrusion_rate: 0.896552
mean_receiver_target_rank_delta: +495.831897
mean_receiver_coord_distance_delta: -43.693966
```

By delta mode:

```text
raw_donor_minus_receiver:
  row_count: 116
  target_rank_improved_rate: 0.551724
  coord_distance_lte_16_rate: 0.206897
  coord_rank_lte_10_rate: 0.008621
  donor_nearer_than_receiver_rate: 0.991379
  slot_intrusion_rate: 0.991379

donor_minus_previous_slot:
  row_count: 116
  target_rank_improved_rate: 0.577586
  coord_distance_lte_16_rate: 0.112069
  coord_rank_lte_10_rate: 0.060345
  donor_nearer_than_receiver_rate: 0.801724
  slot_intrusion_rate: 0.801724
```

By split:

```text
train:
  target_rank_improved_rate: 0.689655
  coord_distance_lte_16_rate: 0.155172
  donor_nearer_than_receiver_rate: 0.896552
  slot_intrusion_rate: 0.896552
  mean_receiver_target_rank_delta: +259.163793

val:
  target_rank_improved_rate: 0.439655
  coord_distance_lte_16_rate: 0.163793
  donor_nearer_than_receiver_rate: 0.896552
  slot_intrusion_rate: 0.896552
  mean_receiver_target_rank_delta: +732.5
```

Clean improvements:

```text
20 / 232 = 0.0862
clean coord-close improvements: 7 / 232 = 0.0302
```

The same-desc filter does not qualitatively change the broad mechanism. It
slightly raises the clean-improvement rate but preserves the same high
slot-intrusion signature and the same concentration in
`donor_minus_previous_slot / staged_after_x1_y1`.

## Expanded V6 Train-First Failure / Val-Analog Cohort

Scope:

```text
panel: v6_v17_per96_prex1_train_failure_val_analog_panel_train16_val2
source pre-x1 rows: pre_x1_source_rows_from_v17.jsonl
staged rows: v4_v6_full_train_first_failure_panel
receiver states: 249
patch rows: 249
train failure seeds: 83
val analogs: 166
same-regime val analogs: 84
same-regime/same-desc val analogs: 82
donor_position: staged_after_x1_y1
delta_mode: donor_minus_previous_slot
patch_scale: 1.0
layer_index: 24
patch_site: layer_input
shards: 3
devices: cuda:2, cuda:4, cuda:0
error_count: 0
```

Aggregate:

```text
target_rank_improved_rate: 0.714859
target_top1_rate: 0.008032
coord_rank_lte_10_rate: 0.076305
coord_distance_lte_16_rate: 0.236948
donor_target_top1_rate: 0.028112
donor_nearer_than_receiver_rate: 0.638554
slot_intrusion_rate: 0.638554
mean_receiver_target_rank_delta: -166.394
```

By split:

```text
train:
  row_count: 83
  target_rank_improved_rate: 0.867470
  coord_distance_lte_16_rate: 0.192771
  donor_nearer_than_receiver_rate: 0.602410
  slot_intrusion_rate: 0.602410
  mean_receiver_target_rank_delta: -324.663

val:
  row_count: 166
  target_rank_improved_rate: 0.638554
  coord_distance_lte_16_rate: 0.259036
  donor_nearer_than_receiver_rate: 0.656627
  slot_intrusion_rate: 0.656627
  mean_receiver_target_rank_delta: -87.259
```

By regime:

```text
small_object:
  row_count: 48
  target_rank_improved_rate: 0.979167
  coord_distance_lte_16_rate: 0.395833
  slot_intrusion_rate: 0.645833

duplicate_basin_nearby:
  row_count: 48
  target_rank_improved_rate: 0.666667
  coord_distance_lte_16_rate: 0.375
  slot_intrusion_rate: 0.541667

crowded:
  row_count: 48
  target_rank_improved_rate: 0.6875
  coord_distance_lte_16_rate: 0.104167
  slot_intrusion_rate: 0.770833

termination_tail:
  row_count: 48
  target_rank_improved_rate: 0.6875
  coord_distance_lte_16_rate: 0.1875
  slot_intrusion_rate: 0.5
```

This broader run strengthens the same mechanism instead of overturning it:
trained rows fail under teacher-forced GT prefixes and respond to the same
slot-transition handle as val analogs, but the response remains mostly
rank/basin steering rather than exact x1 repair. The lower intrusion rate in
v16 versus v13/v14 is useful but not sufficient: exact target top1 remains
below 1%, so this is still a diagnostic handle, not a correction objective.

## Coordinate-Value Geometry Check

The v4 contrast rows were reduced post-hoc by the coordinate values already
present in each staged-slot row. This is readout-only: no model perturbation
or training ran.

Scope:

```text
input contrast rows: 227
output geometry rows: 227
invalid_geometry_row_count: 0
clean_transition rows: 68
intrusive_transition rows: 159
train rows: 79
val rows: 148
near_coord_distance: 16
```

The reducer treats `receiver_target_next_coord_bin` as the receiver x1 target,
`donor_target_next_coord_bin` as the donor later-slot coordinate value
(`x2` for `staged_after_x1_y1` donors), and
`patched_receiver_coord_top1_bin` as the patched receiver coordinate landing.
It then asks whether the patched coordinate lands near receiver x1, near donor
x2, near both, or elsewhere.

Aggregate landing regions:

```text
receiver_x1_near: 12
receiver_and_donor_near: 40
donor_slot_near: 58
baseline_near: 20
other: 97
```

By label:

```text
clean_transition:
  row_count: 68
  clean_coord_close_rate: 0.338235
  donor_slot_near_receiver_x1_rate: 0.382353
  patched_near_receiver_x1_rate: 0.338235
  patched_near_donor_slot_rate: 0.161765
  patched_near_both_receiver_and_donor_rate: 0.161765
  coord_edge_ambiguity_candidate_rate: 0.117647
  mean_target_width_bin: 49.926471

intrusive_transition:
  row_count: 159
  slot_intrusion_rate: 1.0
  donor_slot_near_receiver_x1_rate: 0.345912
  patched_near_receiver_x1_rate: 0.182390
  patched_near_donor_slot_rate: 0.547170
  patched_near_both_receiver_and_donor_rate: 0.182390
  coord_edge_ambiguity_candidate_rate: 0.157233
  mean_target_width_bin: 88.867925
```

By split:

```text
train / clean_transition:
  row_count: 29
  clean_coord_close_rate: 0.172414
  patched_near_both_receiver_and_donor_rate: 0.068966

val / clean_transition:
  row_count: 39
  clean_coord_close_rate: 0.461538
  patched_near_both_receiver_and_donor_rate: 0.230769

train / intrusive_transition:
  row_count: 50
  patched_near_donor_slot_rate: 0.600000
  patched_near_receiver_x1_rate: 0.200000

val / intrusive_transition:
  row_count: 109
  patched_near_donor_slot_rate: 0.522936
  patched_near_receiver_x1_rate: 0.174312
```

Width and coordinate-edge findings:

```text
clean_transition / width_001_016:
  row_count: 26
  donor_slot_near_receiver_x1_rate: 1.0
  patched_near_receiver_x1_rate: 0.346154
  patched_near_both_receiver_and_donor_rate: 0.307692

intrusive_transition / width_001_016:
  row_count: 55
  donor_slot_near_receiver_x1_rate: 1.0
  patched_near_donor_slot_rate: 0.600000
  patched_near_both_receiver_and_donor_rate: 0.454545

intrusive_transition / width_065_plus:
  row_count: 45
  patched_near_donor_slot_rate: 0.444444
  patched_near_receiver_x1_rate: 0.0
  mean_patched_top1_to_donor_slot_abs: 37.711111
  mean_patched_top1_to_receiver_x1_abs: 253.8
```

Interpretation:

```text
Clean and intrusive transitions are not mirror images of one coordinate axis.
Some strict clean coord-close rows are coordinate-edge ambiguous because the
donor later-slot value x2 is itself near the receiver x1 target. These rows are
not pure evidence for object-binding repair.

However, intrusive rows are not just small-object edge ambiguity. They often
land near donor x2 even when donor x2 is far from receiver x1, especially in
wide objects. That is a stronger slot-value capture signature.
```

This changes the next causal split. The clean strict coord-close set should be
subdivided into:

```text
receiver_and_donor_near:
  useful for coordinate-locality / smoothness / small-object ambiguity tests

receiver_x1_near with donor far:
  stronger evidence for true receiver-basin repair
```

The intrusive set should be subdivided into:

```text
donor_slot_near with donor far:
  strongest evidence for donor slot-value capture

receiver_and_donor_near:
  ambiguous small-object or local-edge cases
```

## Per128 Train-Val Expansion

The previous v6 cohort was already population-shaped, but the user's warning
about over-focusing on the person/backpack microscope motivated one more
dataset-row expansion before promoting the slot-phase picture. The expansion
starts from GT-structure sampling, not from a semantic pair.

Artifact roots:

```text
candidate bank:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/train_val_candidate_bank/v7_bbox_len12000_regimes_per128_imgcap2_desccap8

teacher-forced formation rows:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/panel_formation_row_builder/v7_v7_per128_candidate_bank_teacher_forced_gt_desccap8

dry-run robust-position readout selection:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v18_per128_train_val_robust_positions_dry_run

model-backed robust-position readout:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v20_per128_train_val_robust_positions_sharded3_image_root/merged

train-first failure / val-analog panel:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_failure_panel/v7_v20_per128_prex1_train_failure_val_analog_panel_train24_val2

staged-slot rows:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_rows/v5_v7_full_train_first_failure_panel

previous-slot guided-delta plan:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v17_v7_full_prevslot_afterx1y1_plan

model-backed previous-slot guided-delta:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v18_v7_full_prevslot_afterx1y1_sharded2/merged

clean/intrusive selector:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_delta_contrast_selector/v5_v18_v7_prevslot_afterx1y1_uncapped

coordinate-value geometry reducer:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_coord_value_geometry/v2_v5_v18_v7_prevslot_geometry
```

Candidate-bank scope:

```text
candidate rows: 1536
train: 768
val: 768
per split-regime cell: 128
regimes: crowded, duplicate_basin_nearby, repeated_class,
         simple_control, small_object, termination_tail
desc cap violations: 0
image cap violations: 0
```

Formation/readout scope:

```text
formation rows: 12288
robust readout rows: 6144
hidden rows: 12288
readout shards: 3
devices: cuda:5, cuda:3, cuda:4
error_count: 0
positions: pre_x1, post_x1, box_close, next_object_onset
```

Per128 readout repeats the central train-vs-val finding. Even trained rows
are weak at pre-x1 under teacher-forced GT prefix, while post-x1 guidance
largely restores coordinate locality:

```text
pre_x1:
  n: 1536
  target_top1_rate: 0.114583
  target_rank_lte10_rate: 0.330078
  coord_rank_lte10_rate: 0.300781
  coord_distance_lte16_rate: 0.343099
  mean_target_rank: 125.860677
  mean_coord_top1_distance: 109.224609

post_x1:
  n: 1536
  target_top1_rate: 0.165365
  target_rank_lte10_rate: 0.655599
  coord_rank_lte10_rate: 0.638021
  coord_distance_lte16_rate: 0.703776
  mean_target_rank: 15.869792
  mean_coord_top1_distance: 11.002604
```

By split:

```text
train / pre_x1:
  n: 768
  target_top1_rate: 0.138021
  target_rank_lte10_rate: 0.354167
  coord_distance_lte16_rate: 0.319010
  mean_target_rank: 131.040365

val / pre_x1:
  n: 768
  target_top1_rate: 0.091146
  target_rank_lte10_rate: 0.305990
  coord_distance_lte16_rate: 0.367188
  mean_target_rank: 120.680990

train / post_x1:
  n: 768
  target_rank_lte10_rate: 0.684896
  coord_distance_lte16_rate: 0.678385

val / post_x1:
  n: 768
  target_rank_lte10_rate: 0.626302
  coord_distance_lte16_rate: 0.729167
```

The small-object pre-x1 family remains the sharpest coordinate-onset failure:

```text
pre_x1 / small_object:
  n: 256
  target_top1_rate: 0.007812
  target_rank_lte10_rate: 0.117188
  coord_distance_lte16_rate: 0.222656
  mean_target_rank: 212.636719

train / pre_x1 / small_object:
  n: 128
  target_top1_rate: 0.007812
  target_rank_lte10_rate: 0.062500
  mean_target_rank: 239.242188

val / pre_x1 / small_object:
  n: 128
  target_top1_rate: 0.007812
  target_rank_lte10_rate: 0.171875
  mean_target_rank: 186.031250
```

The expanded failure panel selected:

```text
panel rows: 378
train_pre_x1_failure_seed: 126
val_failure_analog: 252
same_regime_same_desc analogs: 126
same_regime analogs: 126
```

This confirms that trained-sequence pre-x1 failures are common in the expanded
dataset row bank. The remaining caveat is analog looseness: object-count,
tail, area, and coordinate-center deltas are still large, so v7 is a mining
and destination-taxonomy panel rather than a strict paired causal contrast.

## Per128 Guided-Delta Replication

The same previous-slot handle was run on the v7 expanded failure panel:

```text
planned rows: 378
model-backed patch rows: 378
shards: 2
devices: cuda:5, cuda:3
error_count: 0
donor_position: staged_after_x1_y1
delta_mode: donor_minus_previous_slot
patch_scale: 1.0
layer_index: 24
patch_site: layer_input
```

Aggregate:

```text
target_rank_improved_rate: 0.452381
target_top1_rate: 0.002646
coord_rank_lte_10_rate: 0.034392
coord_distance_lte_16_rate: 0.100529
donor_target_top1_rate: 0.007937
donor_nearer_than_receiver_rate: 0.690476
slot_intrusion_rate: 0.690476
mean_receiver_target_rank_delta: +1031.624339
```

By split:

```text
train:
  row_count: 126
  target_rank_improved_rate: 0.436508
  target_top1_rate: 0.007937
  coord_distance_lte_16_rate: 0.103175
  slot_intrusion_rate: 0.611111
  mean_receiver_target_rank_delta: +1673.174603

val:
  row_count: 252
  target_rank_improved_rate: 0.460317
  target_top1_rate: 0.0
  coord_distance_lte_16_rate: 0.099206
  slot_intrusion_rate: 0.730159
  mean_receiver_target_rank_delta: +710.849206
```

By regime:

```text
small_object:
  row_count: 72
  target_rank_improved_rate: 0.472222
  coord_distance_lte_16_rate: 0.166667
  slot_intrusion_rate: 0.597222
  target_top1_rate: 0.013889

duplicate_basin_nearby:
  row_count: 72
  target_rank_improved_rate: 0.569444
  coord_distance_lte_16_rate: 0.111111
  slot_intrusion_rate: 0.722222

crowded:
  row_count: 72
  target_rank_improved_rate: 0.458333
  coord_distance_lte_16_rate: 0.069444
  slot_intrusion_rate: 0.708333

termination_tail:
  row_count: 72
  target_rank_improved_rate: 0.375000
  coord_distance_lte_16_rate: 0.069444
  slot_intrusion_rate: 0.763889
```

The v7 clean/intrusive selector produced:

```text
input rows: 378
candidate contrast rows: 313
clean_transition: 52
intrusive_transition: 261
strict clean coord-close rows: 16
train contrast rows: 98
val contrast rows: 215
```

The v7 coordinate-value reducer changes the interpretation from the v6
geometry pass. Donor-slot capture is present, but it is not the only or even
dominant destination in the larger panel. Many intrusive rows are baseline-
near or control-slot-near, which means pooled `slot_intrusion` is too coarse
for the next intervention.

Landing regions:

```text
baseline_near: 132
control_slot_near: 46
donor_slot_near: 44
receiver_and_donor_near: 28
receiver_x1_near: 7
other: 56
```

By label:

```text
clean_transition:
  row_count: 52
  clean_coord_close_rate: 0.307692
  patched_near_receiver_x1_rate: 0.307692
  patched_near_donor_slot_rate: 0.173077
  coord_edge_ambiguity_candidate_rate: 0.115385
  mean_rank_delta: -296.711538

intrusive_transition:
  row_count: 261
  slot_intrusion_rate: 1.0
  patched_near_receiver_x1_rate: 0.072797
  patched_near_donor_slot_rate: 0.241379
  coord_edge_ambiguity_candidate_rate: 0.072797
  mean_rank_delta: +1138.823755
```

The donor-far intrusive slice is still real:

```text
intrusive_transition / width_065_plus:
  row_count: 76
  patched_near_donor_slot_rate: 0.236842
  patched_near_receiver_x1_rate: 0.0
  mean_patched_top1_to_donor_slot_abs: 242.013158
  mean_patched_top1_to_receiver_x1_abs: 438.276316
```

But the broader panel also exposes another family:

```text
intrusive rows can be baseline_near or control_slot_near after the patch.
This suggests a destination taxonomy:
  receiver repair
  donor-slot value capture
  previous/control-slot pull
  baseline inertia
  off-basin escape
```

This is a better next target than simply "remove donor coordinate value." The
next localization pass should condition on landing region and compare hidden
state/component signatures for:

```text
clean receiver_x1_near with donor far
clean receiver_and_donor_near small-object ambiguity
intrusive donor_slot_near with donor far
intrusive control_slot_near
intrusive baseline_near
```

## Mechanistic Interpretation

The latest slot states contain useful coordinate and slot-transition
information, but not a clean transferable object-identity repair vector.

Raw donor-state import can make the receiver's top coordinate locally closer
to the receiver target more often than the previous-slot delta, but this
benefit is entangled with extreme donor-nearer behavior:

```text
broad raw slot_intrusion_rate: 0.975
same-desc raw slot_intrusion_rate: 0.991379
```

The previous-slot delta is a better causal handle because it reduces donor
capture while retaining comparable rank-improvement rates:

```text
broad previous-slot slot_intrusion_rate: 0.825
same-desc previous-slot slot_intrusion_rate: 0.801724
```

However, it still rarely solves the actual pre-x1 decision:

```text
broad target_top1_rate: 0.00625
same-desc target_top1_rate: 0.0
broad clean coord-close rate: 0.0281
same-desc clean coord-close rate: 0.0302
```

This supports a sharper version of the pre-x1 onset hypothesis:

```text
The model often has usable local geometry after x1 or x1/y1 is supplied, but
the pre-x1 state lacks a stable ownership/cursor attractor. Later slot states
carry coordinate-basin and slot-phase information that can be projected back
into pre-x1, but that information remains entangled with the donor slot and
donor coordinate basin.
```

Train rows respond more often than val rows to these deltas, but they exhibit
the same high intrusion profile. That keeps trained-sequence failure central:
the failure is not explained away as held-out visual generalization alone.

## Next Directions

1. Keep `donor_minus_previous_slot / staged_after_x1_y1` as the immediate
   slot-transition handle, but treat it as a weak handle requiring purification,
   not as a solved repair vector.
2. Add a true same-slot or same-coordinate control if the staged rows can supply
   a donor with the same target slot but a different object. The current
   previous-slot control only subtracts adjacent slot phase.
3. Add projection or regression to remove donor-coordinate value and preserve
   only the residual transition direction. The current clean-improvement rows
   are too rare to justify training. The v6 geometry reducer said this
   projection should be evaluated separately on receiver-only clean rows and
   donor-far intrusive rows; the v7 expansion adds that baseline/control
   inertia must be separated as a third destination family.
4. Cross-link clean versus intrusive patch rows to attention/value tomography:
   the question is which heads write transition-like local geometry without
   donor-basin takeover, and which component carries donor x2 slot-value
   capture when donor x2 is far from receiver x1.
5. Continue broad train/val row selection. The next reducer should explicitly
   stratify by object area, coordinate distance between x1 and x2, same-desc
   pressure, and remaining-object tail, instead of letting sorted launch rows
   bias early smokes toward small objects.
6. Treat train-side failures as teacher-forced/readout failures until a small
   train free-rollout attachment is built. The current train evidence is still
   highly valuable because trained rows fail under their exact GT prefix, but
   it is not yet the same evidence type as val free-rollout false negatives.
7. For the next GPU localization pass, use a destination-aware panel from v7:
   receiver repair, donor-slot capture, previous/control-slot pull, baseline
   inertia, and off-basin escape. This should expose whether different heads or
   MLP phases write the different attractors, instead of treating all
   intrusive rows as donor capture.

## Verification

```bash
python -m pytest tests/analysis/test_staged_slot_guided_delta_patch.py -q
python -m pytest tests/analysis/test_staged_slot_coord_value_geometry.py -q
```

Result:

```text
9 passed
4 passed
```

GPU runs:

```text
v12 smoke: 16/16 patch rows, 0 errors
v13 broad sharded4: 320/320 patch rows, 0 errors
v14 same-desc sharded4: 232/232 patch rows, 0 errors
v16 expanded v6 sharded3: 249/249 patch rows, 0 errors
v20 per128 readout sharded3: 6144/6144 readout rows, 0 errors
v18 expanded v7 guided-delta sharded2: 378/378 patch rows, 0 errors
```
