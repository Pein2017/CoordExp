---
doc_id: progress.diagnostics.per96_population_scaffold_readout_findings_2026_06_22
date: 2026-06-22
scope: readout-only, teacher-forced, bbox_len12000, train-val, checkpoint-928
status: active-evidence
---

# Per96 Train/Val Population Scaffold Readout

This note records the first descriptor-capped per96 train/val population
scaffold for the autoregressive binding-template study. The point is to move
the mechanism search away from the original person/backpack microscope and
toward broader trained-sequence failures plus held-out validation analogs.

## Code Surface

```text
src/analysis/autoregressive_binding_template_ablation/train_val_candidate_bank.py
scripts/analysis/run_autoregressive_binding_train_val_candidate_bank.py
tests/analysis/test_train_val_candidate_bank.py
```

New selector option:

```text
--max-per-desc-per-regime-split
```

This caps each target descriptor within each `(split, candidate_regime)` cell,
so larger population banks do not silently collapse back into person, bird,
chair, or another high-frequency class.

## Candidate Bank

Command:

```bash
python scripts/analysis/run_autoregressive_binding_train_val_candidate_bank.py \
  --train-jsonl /data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl \
  --val-jsonl /data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl \
  --per-regime-per-split 96 \
  --max-per-image-per-regime 2 \
  --max-per-desc-per-regime-split 12 \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/train_val_candidate_bank/v6_bbox_len12000_gt_structure_regimes_per96_imgcap2_desccap12
```

Artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/train_val_candidate_bank/v6_bbox_len12000_gt_structure_regimes_per96_imgcap2_desccap12
```

Key counters:

```text
row_count: 1152
unique_object_count: 1135
train rows: 576
val rows: 576
per_regime_per_split: 96
max_per_image_per_regime: 2
max_per_desc_per_regime_split: 12
max_rows_per_image_regime_observed: 2
image_cap_violation_count: 0
max_rows_per_desc_regime_split_observed: 12
desc_cap_violation_count: 0
```

Regime counts are balanced:

```text
crowded: 192
duplicate_basin_nearby: 192
repeated_class: 192
simple_control: 192
small_object: 192
termination_tail: 192
```

The descriptor cap does not make the bank uniform, but it prevents any single
descriptor from dominating an individual split/regime cell. Total descriptor
counts still reflect COCO frequency and multi-regime reuse; for example,
`person` remains the largest total descriptor count at 106/1152.

## Formation Rows

Command:

```bash
python scripts/analysis/run_autoregressive_binding_panel_formation_rows.py \
  --panel-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/train_val_candidate_bank/v6_bbox_len12000_gt_structure_regimes_per96_imgcap2_desccap12/train_val_formation_candidate_bank.jsonl \
  --train-jsonl /data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl \
  --val-jsonl /data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl \
  --tokenizer-path /data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/panel_formation_row_builder/v6_v6_per96_candidate_bank_teacher_forced_gt_desccap12
```

Artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/panel_formation_row_builder/v6_v6_per96_candidate_bank_teacher_forced_gt_desccap12
```

Key counters:

```text
input_panel_row_count: 1152
selected_panel_row_count: 1152
output_row_count: 9216
skipped none: 0
```

Each formation position has 1152 rows:

```text
descriptor_onset
descriptor_end
object_ref_end
box_start
pre_x1
post_x1
box_close
next_object_onset
```

Descriptor-position aggregate claims remain blocked until multi-token target
descriptor scoring is fixed. The first GPU readout therefore used robust
coordinate/schema positions only.

## Robust-Position GPU Readout

Failed diagnostic run:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v15_per96_train_val_robust_positions_sharded4_image_root
```

The v15 run failed because the configured image root resolved to the
bbox_len12000 dataset tree, which does not contain copied image files. This was
an artifact wiring failure, not a model failure. The correct image root is:

```text
/data/CoordExp/public_data/coco/raw
```

Smoke artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v16_per96_robust_positions_image_root_smoke_g3
```

Full merged artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v17_per96_train_val_robust_positions_sharded4_image_root/merged
```

Readout counters:

```text
selected_row_count: 4608
output_row_count: 4608
hidden_row_count: 9216
error_count: 0
train rows: 2304
val rows: 2304
positions: pre_x1, post_x1, box_close, next_object_onset
```

## Surface Readout Findings

At `pre_x1`, the model is weak even under teacher-forced GT prefix:

```text
target_top1_rate: 0.1111
target_rank_lte10_rate: 0.3194
coord_rank_lte10_rate: 0.2908
coord_distance_lte16_rate: 0.4410
mean_target_rank: 131.15
mean_coord_top1_distance: 109.42
```

At `post_x1`, locality largely recovers after the correct x1 has been supplied:

```text
target_top1_rate: 0.1623
target_rank_lte10_rate: 0.6649
coord_rank_lte10_rate: 0.6441
coord_distance_lte16_rate: 0.8411
mean_target_rank: 16.14
mean_coord_top1_distance: 11.33
```

This is the central broad-row result: many failures are not raw object
non-perception. The model often lacks a strong enough pre-x1 coordinate basin
or object-cursor state, then becomes locally coherent after the first coordinate
is supplied.

Train and val split comparison:

```text
pre_x1 train:
  target_top1_rate: 0.1406
  target_rank_lte10_rate: 0.3438
  coord_rank_lte10_rate: 0.3229
  coord_distance_lte16_rate: 0.4375
  mean_target_rank: 129.61

pre_x1 val:
  target_top1_rate: 0.0816
  target_rank_lte10_rate: 0.2951
  coord_rank_lte10_rate: 0.2587
  coord_distance_lte16_rate: 0.4444
  mean_target_rank: 132.69

post_x1 train:
  target_rank_lte10_rate: 0.7031
  coord_rank_lte10_rate: 0.6927
  coord_distance_lte16_rate: 0.8611
  mean_target_rank: 14.74

post_x1 val:
  target_rank_lte10_rate: 0.6267
  coord_rank_lte10_rate: 0.5955
  coord_distance_lte16_rate: 0.8212
  mean_target_rank: 17.55
```

Train is somewhat better on exact/rank statistics, but trained rows still fail
severely at `pre_x1`. This weakens a pure unseen-val/generalization explanation
and keeps trained-sequence failures as a high-value causal population.

By regime, `pre_x1` remains hardest in repeated-class, small-object, and
crowded scenes:

```text
crowded pre_x1:
  target_rank_lte10_rate: 0.1563
  coord_distance_lte16_rate: 0.2865
  mean_target_rank: 157.23

repeated_class pre_x1:
  target_rank_lte10_rate: 0.1198
  coord_distance_lte16_rate: 0.1875
  mean_target_rank: 177.79

small_object pre_x1:
  target_rank_lte10_rate: 0.1042
  coord_distance_lte16_rate: 0.2500
  mean_target_rank: 223.33

simple_control pre_x1:
  target_rank_lte10_rate: 0.8438
  coord_distance_lte16_rate: 0.9167
  mean_target_rank: 14.51
```

At `box_close`, schema boundary prediction is mostly strong:

```text
target_top1_rate: 0.8542
target_rank_lte10_rate: 1.0000
mean_target_rank: 1.28
```

At `next_object_onset`, target rank is also near-saturated:

```text
target_top1_rate: 0.7144
target_rank_lte10_rate: 1.0000
mean_target_rank: 1.29
```

Caveat: `next_object_onset` should not be over-interpreted for
`simple_control` and `termination_tail` rows without checking natural terminal
semantics. The formation builder uses a stress readout at that position, but a
simple-control or tail row may not naturally need another object in the same
way as crowded/repeated rows. This position is useful for router pressure, not
for a blanket continuation-success claim.

Additional caveats from read-only audit:

```text
next_object_onset target: <|object_ref_start|> for all 1152 rows
```

That means it does not score the next descriptor text. In `simple_control` and
`termination_tail`, rows can have `objects_remaining_after=0` while still being
scored against `<|object_ref_start|>`. Treat these as boundary-pressure or
counterfactual continuation probes, not as evidence that another GT object
should be emitted.

The `simple_control` split comparison is also geometry-sensitive. These rows
are structurally simple, but train controls tend to be larger boxes than val
controls in the audited sample. Do not interpret the weaker val simple-control
coordinate ranks as a pure generalization effect without geometry matching.

Termination-tail rows combine several pressures at once: terminality, long
prefixes, high object count, and repeated descriptors. Their saturated
`box_close` scores are therefore less informative than `pre_x1`, `post_x1`,
and terminal boundary competition.

## Layer-24 Logit Lens

Hidden rows were recorded at final/layer `-1` and layer `24`.

Layer 24 already contains some coordinate locality, especially after x1 has
been supplied, but the final surface still sharpens exact token rank:

```text
layer24 pre_x1:
  target_rank_lte10_rate: 0.1311
  coord_rank_lte10_rate: 0.1944
  coord_distance_lte16_rate: 0.3377
  mean_target_rank: 3180.12
  mean_coord_top1_distance: 118.47

final pre_x1:
  target_rank_lte10_rate: 0.3151
  coord_rank_lte10_rate: 0.2925
  coord_distance_lte16_rate: 0.4410
  mean_target_rank: 131.28
  mean_coord_top1_distance: 109.68

layer24 post_x1:
  target_rank_lte10_rate: 0.3559
  coord_rank_lte10_rate: 0.4757
  coord_distance_lte16_rate: 0.7535
  mean_coord_top1_distance: 15.75

final post_x1:
  target_rank_lte10_rate: 0.6641
  coord_rank_lte10_rate: 0.6432
  coord_distance_lte16_rate: 0.8420
  mean_coord_top1_distance: 11.34
```

This supports the coordinate-token basin framing: locality and exact rank are
not the same phenomenon. The model can be near the correct coordinate basin
while still not assigning enough exact mass to the target `<|coord_*|>` token.

## Failure Panel

Command:

```bash
python scripts/analysis/run_autoregressive_binding_formation_failure_panel.py \
  --readout-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v17_per96_train_val_robust_positions_sharded4_image_root/merged/formation_readout_rows.jsonl \
  --train-seeds-per-regime 8 \
  --val-analogs-per-train 1 \
  --min-train-failure-score 25 \
  --min-val-failure-score 25 \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_failure_panel/v5_v17_per96_prex1_train_failure_val_analog_panel
```

Artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_failure_panel/v5_v17_per96_prex1_train_failure_val_analog_panel
```

Key counters:

```text
readout_row_count: 4608
readout_case_count: 1152
row_count: 86
train_seed_count: 43
val_analog_count: 43
matched_pair_count: 43
train_candidate_count: 367
val_candidate_count: 387
mean_pre_x1_failure_score_train: 1070.56
mean_pre_x1_failure_score_val: 329.88
```

Regime counts:

```text
crowded: 16
duplicate_basin_nearby: 16
repeated_class: 16
simple_control: 6
small_object: 16
termination_tail: 16
```

Failure labels:

```text
severe_pre_x1_basin_failure: 71
near_target_rank_failure: 10
moderate_pre_x1_basin_failure: 5
```

Interpretation caveat: this panel is deliberately biased toward hard train
seeds and matched val analogs. It is a causal launch panel, not a population
prevalence estimate. In the v5 panel, all train seeds were selected as severe
pre-x1 failures, while val analogs were mixed across severe, near-target-rank,
and moderate labels. Split means from this panel are therefore selection
diagnostics, not population-level train-vs-val prevalence.

Expanded train-first pass:

```bash
python scripts/analysis/run_autoregressive_binding_formation_failure_panel.py \
  --readout-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v17_per96_train_val_robust_positions_sharded4_image_root/merged/formation_readout_rows.jsonl \
  --train-seeds-per-regime 16 \
  --val-analogs-per-train 2 \
  --min-train-failure-score 25 \
  --min-val-failure-score 25 \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_failure_panel/v6_v17_per96_prex1_train_failure_val_analog_panel_train16_val2
```

Artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_failure_panel/v6_v17_per96_prex1_train_failure_val_analog_panel_train16_val2
```

Key counters:

```text
row_count: 249
train_seed_count: 83
val_analog_count: 166
matched_pair_count: 166
train_candidate_count: 367
val_candidate_count: 387
```

Regime counts:

```text
crowded: 48
duplicate_basin_nearby: 48
repeated_class: 48
simple_control: 9
small_object: 48
termination_tail: 48
```

Analog match tiers:

```text
same_regime_same_desc: 82
same_regime: 84
```

Analog pair deltas show that v6 is a broader mining panel, not yet a strict
matched causal panel:

```text
mean_abs_delta object_count: 11.77
mean_abs_delta objects_remaining_after: 8.40
mean_abs_delta same_desc_count: 1.42
mean_abs_delta bbox_area: 12525.10
mean_abs_delta bbox_center_x2: 612.41
mean_abs_delta bbox_center_y2: 369.14
mean_abs_delta pre_x1_failure_score: 583.94
mean_abs_delta post_x1_coord_rank_gt: 24.30
mean_abs_delta post_x1_coord_top1_distance: 19.18
```

The good news is that every val analog is same-regime and the panel is much
broader than the old 86-row v5 panel. The caution is that coordinate location,
tail length, and failure severity can still differ substantially. Use v6 to
split launch cohorts and identify failure families; use a stricter analog
selector before interpreting train-vs-val pairs as causal matches.

## Launch Filters

The failure-panel launch filter consumes v6 rows and writes both selected panel
rows and pair-row sidecars. It keeps downstream tools compatible with the panel
row format while making match quality auditable.

Code surface:

```text
src/analysis/autoregressive_binding_template_ablation/formation_failure_launch_filter.py
scripts/analysis/run_autoregressive_binding_formation_failure_launch_filter.py
tests/analysis/test_formation_failure_launch_filter.py
```

Broad balanced launch cohort:

```bash
python scripts/analysis/run_autoregressive_binding_formation_failure_launch_filter.py \
  --panel-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_failure_panel/v6_v17_per96_prex1_train_failure_val_analog_panel_train16_val2/formation_failure_panel_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_failure_launch_filter/v1_v6_train16_val2_loose_balanced_pairs8 \
  --max-pairs-per-regime 8 \
  --val-analogs-per-train 1 \
  --max-object-count-delta 20 \
  --max-tail-delta 24 \
  --max-same-desc-delta 5 \
  --max-center-l1-x2-delta 1800 \
  --max-area-delta 100000 \
  --max-post-x1-rank-delta 120 \
  --max-post-x1-distance-delta 80
```

Artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_failure_launch_filter/v1_v6_train16_val2_loose_balanced_pairs8
```

Key counters:

```text
candidate_pair_count: 102
pair_count: 40
row_count: 80
pair_counts_by_regime:
  crowded: 7
  duplicate_basin_nearby: 8
  repeated_class: 8
  simple_control: 1
  small_object: 8
  termination_tail: 8
analog_match_tiers:
  same_regime_same_desc: 29
  same_regime: 11
```

This is the recommended regime-coverage launch cohort. It preserves crowded
and termination-tail rows, but a small number of pairs are same-regime rather
than same-desc.

Selected-pair match quality:

```text
mean_abs_delta object_count: 8.08
mean_abs_delta objects_remaining_after: 5.18
mean_abs_delta same_desc_count: 1.33
mean_abs_delta bbox_area: 9233.05
mean_abs_delta center_l1_x2: 884.03
mean_abs_delta post_x1_coord_rank_gt: 10.60
mean_abs_delta post_x1_coord_top1_distance: 8.65
```

Same-desc launch cohort:

```bash
python scripts/analysis/run_autoregressive_binding_formation_failure_launch_filter.py \
  --panel-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_failure_panel/v6_v17_per96_prex1_train_failure_val_analog_panel_train16_val2/formation_failure_panel_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_failure_launch_filter/v2_v6_train16_val2_loose_same_desc_pairs8 \
  --max-pairs-per-regime 8 \
  --val-analogs-per-train 1 \
  --require-same-desc \
  --max-object-count-delta 20 \
  --max-tail-delta 24 \
  --max-same-desc-delta 5 \
  --max-center-l1-x2-delta 1800 \
  --max-area-delta 100000 \
  --max-post-x1-rank-delta 120 \
  --max-post-x1-distance-delta 80
```

Artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_failure_launch_filter/v2_v6_train16_val2_loose_same_desc_pairs8
```

Key counters:

```text
candidate_pair_count: 50
pair_count: 29
row_count: 58
pair_counts_by_regime:
  crowded: 4
  duplicate_basin_nearby: 5
  repeated_class: 7
  small_object: 8
  termination_tail: 5
analog_match_tiers:
  same_regime_same_desc: 29
```

This is the cleaner identity-binding cohort, but it sacrifices some crowded
coverage and all simple-control pairs.

Launch-filter interpretation:

1. Use v1 for regime-balanced causal smoke tests and router/closure coverage.
2. Use v2 when descriptor identity must be held fixed across train/val pairs.
3. Keep pair-level deltas in joins so a causal result can be stratified by
   center/tail/area/post_x1 mismatch rather than averaged away.

## Launch-Cohort Staged Readout

Staged-slot rows for the broad v1 launch filter:

```bash
python scripts/analysis/run_autoregressive_binding_staged_slot_readout_rows.py \
  --source-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_failure_launch_filter/v1_v6_train16_val2_loose_balanced_pairs8/pre_x1_source_rows_from_v17.jsonl \
  --panel-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_failure_launch_filter/v1_v6_train16_val2_loose_balanced_pairs8/formation_failure_launch_panel_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_rows/v2_v1_launch_filter_broad_balanced_pairs8 \
  --slots staged_pre_x1,staged_after_x1,staged_after_x1_y1,staged_after_x1_y1_x2,staged_after_full_box
```

Artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_rows/v2_v1_launch_filter_broad_balanced_pairs8
```

Counters:

```text
source_row_count: 1152
valid_source_case_count: 1152
panel_row_count: 80
output_row_count: 400
skipped none: 0
train rows: 200
val rows: 200
```

Staged-slot rows for the same-desc v2 launch filter:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_rows/v3_v2_launch_filter_same_desc_pairs8
```

Counters:

```text
panel_row_count: 58
output_row_count: 290
skipped none: 0
train rows: 145
val rows: 145
```

Model-backed v1 broad launch readout:

```bash
python scripts/analysis/run_autoregressive_binding_formation_readout_probe.py \
  --config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --formation-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_rows/v2_v1_launch_filter_broad_balanced_pairs8/staged_slot_readout_rows.jsonl \
  --image-root /data/CoordExp/public_data/coco/raw \
  --positions staged_pre_x1,staged_after_x1,staged_after_x1_y1,staged_after_x1_y1_x2,staged_after_full_box \
  --layers 24,-1 \
  --shard-index {0..3} \
  --num-shards 4
```

GPU assignment:

```text
shard_000-of-004: GPU 0
shard_001-of-004: GPU 3
shard_002-of-004: GPU 5
shard_003-of-004: GPU 6
```

Merged artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_readout_probe/v19_v1_launch_staged_slots_sharded4/merged
```

Counters:

```text
selected_row_count: 400
output_row_count: 400
hidden_row_count: 800
error_count: 0
```

Reducers:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_readout_reducer/v2_v19_v1_launch_staged_slots
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_hidden_reducer/v2_v19_v1_launch_staged_slots
```

Final surface position summary:

```text
staged_pre_x1:
  n: 80
  coord_rank_lte10_rate: 0.0000
  coord_distance_lte16_rate: 0.1125
  mean_coord_rank_gt: 365.5
  mean_coord_top1_distance: 278.7

staged_after_x1:
  n: 80
  coord_rank_lte10_rate: 0.6750
  coord_distance_lte16_rate: 0.8125
  mean_coord_rank_gt: 12.85
  mean_coord_top1_distance: 9.10

staged_after_x1_y1:
  n: 80
  coord_rank_lte10_rate: 0.4375
  coord_distance_lte16_rate: 0.8250
  mean_coord_rank_gt: 20.01
  mean_coord_top1_distance: 11.43

staged_after_x1_y1_x2:
  n: 80
  coord_rank_lte10_rate: 0.4625
  coord_distance_lte16_rate: 0.7750
  mean_coord_rank_gt: 22.26
  mean_coord_top1_distance: 11.68

staged_after_full_box:
  n: 80
  box_end_top1_rate: 0.9500
  top1 tokens: <|box_end|> 76, <|object_ref_end|> 3, <|object_ref_start|> 1
```

Split view:

```text
staged_pre_x1 train:
  coord_rank_lte10_rate: 0.0000
  coord_distance_lte16_rate: 0.0000
  mean_coord_rank_gt: 525.88
  mean_coord_top1_distance: 418.80

staged_pre_x1 val:
  coord_rank_lte10_rate: 0.0000
  coord_distance_lte16_rate: 0.2250
  mean_coord_rank_gt: 205.15
  mean_coord_top1_distance: 138.55

staged_after_x1 train:
  coord_rank_lte10_rate: 0.6500
  coord_distance_lte16_rate: 0.8000
  mean_coord_rank_gt: 13.63

staged_after_x1 val:
  coord_rank_lte10_rate: 0.7000
  coord_distance_lte16_rate: 0.8250
  mean_coord_rank_gt: 12.08
```

The train/val split here should not be read as a population split: train rows
were selected as harder failure seeds and val rows as analogs. The important
finding is that both split roles recover strongly after x1 is supplied.

Regime view:

```text
staged_after_x1 coord_distance_lte16_rate:
  crowded: 0.9286
  duplicate_basin_nearby: 0.9375
  repeated_class: 0.8125
  simple_control: 1.0000
  small_object: 0.8125
  termination_tail: 0.5625

staged_after_full_box box_end_top1_rate:
  crowded: 0.7857
  duplicate_basin_nearby: 1.0000
  repeated_class: 1.0000
  simple_control: 0.5000
  small_object: 1.0000
  termination_tail: 1.0000
```

The four full-box boundary misses are:

```text
train crowded person, image 469859 object 0/69: top1 <|object_ref_end|>, rank 3
train simple_control teddy bear, image 400872 object 0/1: top1 <|object_ref_start|>, rank 4
val crowded person, image 303566 object 0/53: top1 <|object_ref_end|>, rank 3
val crowded person, image 31296 object 1/52: top1 <|object_ref_end|>, rank 2
```

Layer-24 versus final:

```text
layer24 staged_pre_x1:
  coord_rank_lte10_rate: 0.0375
  coord_distance_lte16_rate: 0.1000
  mean_coord_rank_gt: 356.1

final staged_pre_x1:
  coord_rank_lte10_rate: 0.0000
  coord_distance_lte16_rate: 0.1125
  mean_coord_rank_gt: 366.4

layer24 staged_after_x1:
  coord_rank_lte10_rate: 0.5375
  coord_distance_lte16_rate: 0.7875
  mean_coord_rank_gt: 20.09

final staged_after_x1:
  coord_rank_lte10_rate: 0.6500
  coord_distance_lte16_rate: 0.8125
  mean_coord_rank_gt: 12.91

layer24 staged_after_full_box:
  box_end_top1_rate: 0.8750

final staged_after_full_box:
  box_end_top1_rate: 0.9500
```

Current interpretation:

1. The broader launch cohort strongly reproduces x1-onset failure: exact
   coordinate rank is never top-10 at `staged_pre_x1`, and locality is rare.
2. Supplying x1 restores local coordinate readiness in both trained failure
   seeds and held-out analogs. This argues against a visual non-perception
   explanation for most selected rows.
3. Later coordinate slots still show an exact-rank versus locality gap:
   `staged_after_x1_y1` and `staged_after_x1_y1_x2` have high distance<=16
   rates but lower rank<=10 rates. This keeps coordinate smoothness/exact-token
   basin sharpness as a separate mechanism from object perception.
4. Full-box closure is mostly solved after the box is supplied, but residual
   misses concentrate in crowded boundary states. The simple-control miss is a
   terminal/counterfactual edge case and should not be generalized.
5. Layer 24 already carries much of the post-x1 locality and box-end evidence;
   the final surface sharpens rank and boundary probability. At pre_x1, neither
   layer 24 nor the final surface has a usable coordinate-onset state.

## Current Diagnosis

The broad scaffold strengthens the current mechanism picture:

1. Trained-sequence failures exist under teacher-forced GT prefix, so failure
   cannot be explained away as unseen validation exposure.
2. The largest broad failure is the `pre_x1` coordinate-basin onset state.
   `post_x1` recovery is common, which points to autoregressive cursor/basin
   fragility rather than raw visual non-perception in most rows.
3. Repeated-class, small-object, and crowded regimes are the most valuable
   next-row populations because they stress different aspects of the same
   object-state problem: identity disambiguation, coordinate smoothness/local
   evidence, and router/boundary pressure.
4. Layer-24 hidden readouts show partial locality but weak exact rank,
   especially at `pre_x1`. This is consistent with coordinate-token basin
   attraction being present but not smooth or decisive enough.
5. `next_object_onset` and `box_close` should be treated as schema/router
   probes, not merged into coordinate-basin evidence.

## Next Experiments

The next round should be population-first:

1. Mine additional training rows that fail at `pre_x1` under teacher-forced GT
   prefix. Match held-out val analogs by candidate regime, object order,
   repeated-desc status, area, remaining tail length, and target-coordinate
   neighborhood. Rollout labels are useful but not required.
2. Tighten analog matching before the next causal train-vs-val comparison.
   The v6 expanded panel is same-regime but still loose on coordinate center,
   object-count/tail, and failure-score deltas. Add coordinate-neighborhood,
   bbox-shape, and post_x1 recovery/nonrecovery terms to the analog sort or
   emit a stricter filtered launch cohort.
3. For `pre_x1` failures, split later-slot guided states into object identity,
   coordinate value, and slot phase. Direct donor-state deltas mostly caused
   donor-nearer slot intrusion in the previous run, so the next probe should
   subtract same-slot controls or project out slot-position directions before
   claiming object-state transfer.
4. For crowded closure/router rows, mine beyond the current staged mechanism
   panel and replicate the layer-24 box_end/router boundary-direction threshold
   on additional train rows plus held-out analogs.
5. For small-object and final-extent rows, treat coordinate locality and exact
   coordinate-token rank as separate hypotheses. Inspect whether the model has
   local mass near the target but lacks smoothness/exact-rank concentration, or
   whether visual evidence is genuinely weak.
6. Keep person/backpack as an interpretability microscope only. Promotion
   requires the same slot-locus signature across trained rows and val analogs.

## Verification

Commands:

```bash
python -m pytest tests/analysis/test_train_val_candidate_bank.py -q
python -m py_compile src/analysis/autoregressive_binding_template_ablation/train_val_candidate_bank.py scripts/analysis/run_autoregressive_binding_train_val_candidate_bank.py tests/analysis/test_train_val_candidate_bank.py
python -m pytest tests/analysis/test_formation_failure_panel.py -q
python -m pytest tests/analysis/test_formation_failure_launch_filter.py -q
git diff --check
```

Result:

```text
tests/analysis/test_train_val_candidate_bank.py: 14 passed
tests/analysis/test_formation_failure_panel.py: 2 passed
tests/analysis/test_formation_failure_launch_filter.py: 2 passed
py_compile: passed
git diff --check: passed
```
