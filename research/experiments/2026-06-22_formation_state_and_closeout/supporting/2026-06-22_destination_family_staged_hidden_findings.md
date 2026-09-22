---
doc_id: progress.diagnostics.destination_family_staged_hidden_findings_2026_06_22
date: 2026-06-22
scope: teacher-forced, bbox_len12000, checkpoint-928, train-val, staged-slot, no-training
status: active-evidence
---

# Destination-Family Staged Hidden Findings

## Purpose

This note records the first destination-family staged-slot readout over the
per128 train/val panel. It follows the decision to stop centering the old
person/backpack-style microscope and instead compare broader trained-sequence
failures with held-out validation analogs.

The specific question is whether pre-x1 failures mean the model cannot perceive
the object visually, or whether the object/box state becomes usable only after
language-side coordinate guidance pushes the autoregressive state into the
right slot/basin.

## Code Surface

Added or extended:

```text
src/analysis/autoregressive_binding_template_ablation/staged_slot_destination_panel_selector.py
scripts/analysis/run_autoregressive_binding_staged_slot_destination_panel_selector.py
tests/analysis/test_staged_slot_destination_panel_selector.py

src/analysis/autoregressive_binding_template_ablation/staged_slot_readout_rows.py
tests/analysis/test_staged_slot_readout_rows.py

src/analysis/autoregressive_binding_template_ablation/formation_readout_probe.py
tests/analysis/test_formation_readout_probe.py

src/analysis/autoregressive_binding_template_ablation/staged_slot_hidden_reducer.py
tests/analysis/test_staged_slot_hidden_reducer.py
```

New behavior:

```text
staged_slot_destination_panel_selector:
  classifies guided-delta landing rows into receiver_repair,
  donor_slot_capture, baseline_inertia, control_slot_pull,
  coordinate_edge_ambiguity, and off_basin_escape.

staged_slot_readout_rows:
  preserves slot_destination_* metadata when building staged rows from a panel.

formation_readout_probe:
  preserves destination-family and panel context in hidden logit-lens rows.

staged_slot_hidden_reducer:
  groups by configured_layer, formation_position, slot_destination_family,
  and split.
```

Verification:

```text
python -m pytest tests/analysis/test_formation_readout_probe.py tests/analysis/test_staged_slot_hidden_reducer.py tests/analysis/test_staged_slot_readout_rows.py tests/analysis/test_staged_slot_destination_panel_selector.py -q
25 passed
```

## Artifact Roots

Destination selector:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_destination_panel_selector/v1_v2_v7_destination_panel_cap6
```

Destination-filtered staged rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_rows/v6_v1_destination_panel_cap6_all_slots
```

GPU readout with hidden context:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v23_v6_destination_staged_slots_hidden_context_sharded3_image_root/merged
```

Hidden reducer:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_hidden_reducer/v4_v23_destination_family_hidden_context
```

The v23 readout used:

```text
config: configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
family: desc_first
image_root: /data/CoordExp/public_data/coco/raw
positions: staged_pre_x1, staged_after_x1, staged_after_x1_y1,
           staged_after_x1_y1_x2, staged_after_full_box
layers: 16, 20, 24, -1
shards: 3
CUDA_VISIBLE_DEVICES: 5, 1, 3
selected rows: 1065
readout rows: 1065
hidden rows: 4260
error_count: 0
training_ran: false
```

## Destination Panel

The destination selector consumed the v7 coordinate-value geometry rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_coord_value_geometry/v2_v5_v18_v7_prevslot_geometry/staged_slot_coord_value_geometry_rows.jsonl
```

Input and selected counts:

```text
input rows: 313
candidate rows: 296
selected rows: 213
cap violations: 0
```

Selected family counts:

```text
baseline_inertia: 61
control_slot_pull: 35
coordinate_edge_ambiguity: 25
donor_slot_capture: 39
off_basin_escape: 46
receiver_repair: 7
```

Selected split counts:

```text
train: 85
val: 128
```

Staged rows expand each selected panel case into five slot positions:

```text
total staged rows: 1065
train: 425
val: 640
staged_pre_x1: 213
staged_after_x1: 213
staged_after_x1_y1: 213
staged_after_x1_y1_x2: 213
staged_after_full_box: 213
```

## Main Readout

Direct final logits over v23:

```text
staged_pre_x1:
  coord_rank<=10: 0.0094
  coord_distance<=16: 0.0188
  mean_coord_rank: 445.16

staged_after_x1:
  coord_rank<=10: 0.2723
  coord_distance<=16: 0.4038
  mean_coord_rank: 176.82

staged_after_x1_y1:
  coord_rank<=10: 0.2066
  coord_distance<=16: 0.4272
  mean_coord_rank: 204.77

staged_after_x1_y1_x2:
  coord_rank<=10: 0.3380
  coord_distance<=16: 0.5634
  mean_coord_rank: 106.69

staged_after_full_box:
  box_end_top1: 0.5023
```

Layer-24 hidden logit-lens has the same shape but weaker final sharpening:

```text
staged_pre_x1:
  coord_rank<=10: 0.0094
  coord_distance<=16: 0.0235
  mean_coord_rank: 466.66

staged_after_x1:
  coord_rank<=10: 0.2300
  coord_distance<=16: 0.4225
  mean_coord_rank: 177.86

staged_after_x1_y1:
  coord_rank<=10: 0.1596
  coord_distance<=16: 0.3427
  mean_coord_rank: 201.03

staged_after_x1_y1_x2:
  coord_rank<=10: 0.2254
  coord_distance<=16: 0.5023
  mean_coord_rank: 129.64

staged_after_full_box:
  box_end_top1: 0.3803
```

## Mechanism Read

The broad pattern is not pure visual non-perception. The model often has enough
information to make later coordinate slots locally readable once the correct
coordinate prefix is supplied. The primary fault is the onset/routing state at
pre-x1 and the subsequent choice of coordinate-slot basin.

Destination families differ strongly:

```text
receiver_repair:
  n: 7
  pre_x1 remains poor, but after_x1/after_y1 and box_end are often readable.
  Direct box_end_top1 after full box: 0.8571.

donor_slot_capture:
  n: 39
  pre_x1 is poor, but after_x1_y1 becomes very readable
  (direct coord_rank<=10 0.5128, distance<=16 0.8718).
  This is not necessarily correct receiver repair; it is compatible with
  strong donor-slot value capture.

baseline_inertia:
  n: 61
  pre_x1 and after_x1_y1 remain poor, while after_x1_y1_x2 becomes readable.
  Direct box_end_top1 after full box is only 0.1148.
  This family is a cursor/inertia failure, not donor capture.

control_slot_pull:
  n: 35
  poor throughout, including after forced x1/y1.
  This is the strongest previous/control-slot anchoring candidate.

coordinate_edge_ambiguity:
  n: 25
  high coordinate locality after x1/y1
  (direct after_x1_y1 distance<=16 1.0).
  Treat this as locality/smoothness or edge ambiguity, not a clean binding
  repair family.

off_basin_escape:
  n: 46
  many rows become guidable after x1 but remain mixed.
  This family should be subdivided before causal claims.
```

Train rows also fail under teacher-forced GT prefix. Examples:

```text
baseline_inertia direct pre_x1:
  train mean rank: 473.23
  val mean rank: 544.45

control_slot_pull direct pre_x1:
  train mean rank: 485.00
  val mean rank: 599.74

off_basin_escape direct pre_x1:
  train mean rank: 609.47
  val mean rank: 452.52
```

This keeps trained-sequence failures central. The next dataset pass should mine
more training rows directly instead of waiting for rollout labels.

## Interpretation

The current mechanism picture:

```text
1. Pre-x1 onset is the brittle point.
2. Correct coordinate guidance often unlocks later slot readability.
3. Later-slot donor deltas are causal but often import the wrong slot/value.
4. Destination-family labels reveal at least four distinct attractors:
   receiver repair, donor-slot capture, previous/control-slot pull, and
   baseline inertia.
5. Some small/local rows are coordinate-edge ambiguity, where receiver and
   donor coordinates are too close to separate with a simple near/far label.
```

Therefore, do not pool all pre-x1 failures into one donor-patching or attention
tomography panel. The next localization pass should be destination-pure.

## Next Directions

Recommended next implementation sequence:

```text
1. Train-row expansion:
   mine a larger training-dataset sample for teacher-forced pre_x1 failures,
   then label rows by destination family. Use readout evidence first; rollout
   labels are optional post-hoc attachments.

2. Destination-pure attention/value localization:
   compare donor_slot_capture, baseline_inertia, control_slot_pull, and
   receiver_repair rows separately. The current evidence says these are
   different mechanisms.

3. Small-object locality/smoothness:
   use coordinate_edge_ambiguity and strict receiver-near rows to study whether
   coordinate-token locality is preserved but smoothness/exact rank is broken.

4. False-negative guidance-versus-perception:
   for missing objects, run staged forcing/continuation panels:
   descriptor/box-start guidance, x1, x1+y1, x1+y1+x2, full box, and box_end
  routing. Only call a row visually imperceptible after these guidance modes
  fail or remain weak.
```
