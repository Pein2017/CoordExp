---
doc_id: progress.diagnostics.per256_train_failure_destination_findings_2026_06_22
date: 2026-06-22
scope: teacher-forced, bbox_len12000, checkpoint-928, train-only, per256, no-training
status: active-evidence
---

# Per256 Train-Failure Destination Findings

## Purpose

This note records the first train-heavy expansion after the destination-family
staged-slot panel. The goal was to avoid centering a single semantic pair or
held-out validation failures, and instead ask whether trained COCO sequences
still show the same coordinate-onset failure modes under teacher-forced GT
prefixes.

## Code Surface

Extended:

```text
src/analysis/autoregressive_binding_template_ablation/formation_readout_probe.py
tests/analysis/test_formation_readout_probe.py
```

New readout filter behavior:

```text
--splits
--candidate-regimes
```

The filters apply before sharding and limit selection. This allows train-first
mining without writing ad hoc filtered JSONL files.

Verification:

```text
python -m pytest tests/analysis/test_formation_readout_probe.py -q
10 passed
```

## Artifact Roots

Per256 candidate bank:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/train_val_candidate_bank/v8_bbox_len12000_regimes_per256_imgcap2_desccap12
```

Teacher-forced formation rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/panel_formation_row_builder/v8_v8_per256_candidate_bank_teacher_forced_gt_desccap12
```

Train-only robust-position readout:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v25_v8_per256_train_robust_positions_sharded3_image_root/merged
```

Train-only failure seed panel:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_failure_panel/v8_v25_per256_train_only_prex1_failure_seeds64
```

Train-only staged rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_rows/v7_v8_per256_train_only_failure_panel_all_slots
```

Previous-slot guided-delta patch:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v20_v8_per256_train_only_prevslot_afterx1y1_sharded2/merged
```

Clean/intrusive contrast:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_delta_contrast_selector/v6_v20_per256_train_only_prevslot_afterx1y1_uncapped
```

Coordinate-value geometry:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_coord_value_geometry/v3_v6_v20_per256_train_only_prevslot_geometry
```

Destination selector:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_destination_panel_selector/v2_v3_per256_train_only_destination_uncapped
```

## Train-Only Readout

The per256 bank contains 3072 candidate objects: 1536 train and 1536 val, with
256 rows per split-regime cell. The train-only robust-position readout selected
6144 rows:

```text
positions: pre_x1, post_x1, box_close, next_object_onset
split: train
layers: 24, -1
shards: 3
CUDA_VISIBLE_DEVICES: 5, 3, 2
readout_rows: 6144
hidden_rows: 12288
error_count: 0
```

Direct final-logit summary:

```text
pre_x1:
  coord_rank<=10: 0.3151
  coord_distance<=16: 0.4473
  mean_coord_rank: 132.58

post_x1:
  coord_rank<=10: 0.6400
  coord_distance<=16: 0.8359
  mean_coord_rank: 16.08

box_close:
  box_end_top1: 0.8620

next_object_onset:
  top1 <|object_ref_start|>: 1066 / 1536
  top1 <|im_end|>: 470 / 1536
```

By regime at pre_x1:

```text
repeated_class:
  coord_rank<=10: 0.1523
  coord_distance<=16: 0.1992
  mean_coord_rank: 170.66

duplicate_basin_nearby:
  coord_rank<=10: 0.2773
  coord_distance<=16: 0.4727
  mean_coord_rank: 118.69

crowded:
  coord_rank<=10: 0.1523
  coord_distance<=16: 0.2891
  mean_coord_rank: 169.04

small_object:
  coord_rank<=10: 0.0586
  coord_distance<=16: 0.2070
  mean_coord_rank: 236.38

termination_tail:
  coord_rank<=10: 0.3633
  coord_distance<=16: 0.5898
  mean_coord_rank: 94.65

simple_control:
  coord_rank<=10: 0.8867
  coord_distance<=16: 0.9258
  mean_coord_rank: 6.04
```

Interpretation: trained rows still fail strongly at pre_x1, especially
small-object, crowded, and repeated-class rows. Simple-control rows mostly do
not have the coordinate-onset failure, so the failure is not a universal train
readout defect.

## Train-Failure Panel

The train-only failure selector chose 346 trained-sequence failures:

```text
train_candidate_count: 986
train_seed_count: 346
val_analog_count: 0
mean_pre_x1_failure_score: 753.16
```

Regime counts:

```text
crowded: 64
duplicate_basin_nearby: 64
repeated_class: 64
simple_control: 26
small_object: 64
termination_tail: 64
```

Failure labels:

```text
severe_pre_x1_basin_failure: 328
moderate_pre_x1_basin_failure: 10
near_target_rank_failure: 7
target_or_near_target_basin: 1
```

## Previous-Slot Guided Delta

Patch setup:

```text
receiver: staged_pre_x1
donor: staged_after_x1_y1
control: staged_after_x1
delta_mode: donor_minus_previous_slot
patch_scale: 1.0
layer_index: 24
patch_site: layer_input
patch_rows: 346
error_count: 0
```

Patch summary:

```text
target_rank_improved_rate: 0.4162
target_top1_rate: 0.0029
coord_distance_lte_16_rate: 0.0694
donor_nearer_than_receiver_rate: 0.6821
slot_intrusion_rate: 0.6821
mean_receiver_target_rank_delta: +2223.87
```

The previous-slot delta is causal, but not mostly corrective on this larger
train-only panel. It often worsens the receiver target rank or lands in another
attractor.

## Destination Families

Destination selector retained 269 rows:

```text
baseline_inertia: 108
control_slot_pull: 32
coordinate_edge_ambiguity: 12
donor_slot_capture: 47
off_basin_escape: 62
receiver_repair: 8
```

Regime by family:

```text
crowded:
  baseline_inertia 22
  control_slot_pull 8
  coordinate_edge_ambiguity 1
  donor_slot_capture 9
  off_basin_escape 9
  receiver_repair 1

duplicate_basin_nearby:
  baseline_inertia 21
  control_slot_pull 6
  coordinate_edge_ambiguity 2
  donor_slot_capture 4
  off_basin_escape 11
  receiver_repair 0

repeated_class:
  baseline_inertia 20
  control_slot_pull 6
  coordinate_edge_ambiguity 1
  donor_slot_capture 10
  off_basin_escape 11
  receiver_repair 0

simple_control:
  baseline_inertia 0
  control_slot_pull 0
  coordinate_edge_ambiguity 0
  donor_slot_capture 15
  off_basin_escape 0
  receiver_repair 3

small_object:
  baseline_inertia 23
  control_slot_pull 4
  coordinate_edge_ambiguity 7
  donor_slot_capture 3
  off_basin_escape 14
  receiver_repair 0

termination_tail:
  baseline_inertia 22
  control_slot_pull 8
  coordinate_edge_ambiguity 1
  donor_slot_capture 6
  off_basin_escape 17
  receiver_repair 4
```

Main read: true receiver repair is rare even among trained rows. The dominant
failure after applying the previous-slot handle is baseline inertia, followed
by off-basin escape and donor-slot capture. This makes the mechanism less like
a missing object representation and more like a coordinate-slot routing and
attractor-selection failure.

## Updated Mechanism Picture

The train-heavy result strengthens four claims:

```text
1. Trained-sequence pre_x1 failures are common and severe.
2. Correct x1 guidance still dramatically improves post_x1 coordinate readiness.
3. Later-slot hidden deltas are not generic repairs; they expose attractor
   competition.
4. Baseline inertia is the largest train-only destination family, larger than
   donor-slot capture.
```

The next localization pass should not pool all trained failures. Use
destination-pure cohorts:

```text
baseline_inertia:
  test why the coordinate top1 remains at the baseline coordinate basin.

donor_slot_capture:
  localize donor x2 value transport through attention/value and residual paths.

control_slot_pull:
  test previous-slot anchoring and slot-phase state.

coordinate_edge_ambiguity:
  study coordinate-token locality/smoothness, especially for small objects.

receiver_repair:
  use only as a positive microscope; there are too few rows for population
  claims.
```
