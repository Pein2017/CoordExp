---
doc_id: progress.diagnostics.slot_phase_clean_intrusive_contrast_findings_2026_06_22
date: 2026-06-22
scope: artifact-reducer, no-model-load, bbox_len12000, train-val, checkpoint-928-derived
status: active-evidence
---

# Slot-Phase Clean-vs-Intrusive Contrast Selector

This note records a follow-up bridge from the raw-vs-previous-slot guided-delta
artifacts to future attention/value tomography. The goal is to avoid mixing
rare non-intrusive transition rows with the much larger donor-capture
population.

## Code Surface

```text
src/analysis/autoregressive_binding_template_ablation/staged_slot_delta_contrast_selector.py
scripts/analysis/run_autoregressive_binding_staged_slot_delta_contrast_selector.py
tests/analysis/test_staged_slot_delta_contrast_selector.py
```

The selector consumes `staged_slot_guided_delta_rows.jsonl` and focuses on the
only currently meaningful slot-transition contrast:

```text
delta_mode: donor_minus_previous_slot
donor_position: staged_after_x1_y1
```

It labels:

```text
clean_transition:
  patched_receiver_target_token_rank_delta < 0
  slot_intrusion is false
  donor_nearer_than_receiver is false

intrusive_transition:
  slot_intrusion is true
  or donor_nearer_than_receiver is true
```

It also flags a stricter subtype:

```text
slot_phase_clean_coord_close:
  clean_transition
  patched_receiver_coord_top1_distance <= 16
```

Rows are capped per `(label, split, candidate_regime)` so future tomography
does not silently collapse to one regime.

## Artifact Roots

Broad launch cohort:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_delta_contrast_selector/v1_v13_broad_prevslot_clean_vs_intrusive_cap8
```

Same-desc cohort:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_delta_contrast_selector/v2_v14_same_desc_prevslot_clean_vs_intrusive_cap8
```

Expanded v6 train-first failure / val-analog cohort:

```text
capped:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_delta_contrast_selector/v3_v16_full_v6_prevslot_afterx1y1_cap16

uncapped:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_delta_contrast_selector/v4_v16_full_v6_prevslot_afterx1y1_uncapped
```

## Broad Cohort Result

Source:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v13_v1_launch_raw_vs_prevslot_sharded4/merged/staged_slot_guided_delta_rows.jsonl
```

Counters:

```text
input_row_count: 320
focus_row_count: 80
candidate_row_count: 74
output_row_count: 74
filtered_row_count: 240
unlabeled_focus_row_count: 6
candidate_label_counts:
  clean_transition: 22
  intrusive_transition: 52
selected_split_counts:
  train: 38
  val: 36
clean_coord_close_count: 6
cap_violation_count: 0
```

Label summaries:

```text
clean_transition:
  row_count: 22
  rank_improved_rate: 1.0
  clean_coord_close_rate: 0.272727
  mean_rank_delta: -279.91
  mean_coord_top1_distance: 284.73

intrusive_transition:
  row_count: 52
  rank_improved_rate: 0.788462
  clean_coord_close_rate: 0.0
  mean_rank_delta: -233.87
  mean_coord_top1_distance: 93.23
```

The broad panel is balanced enough for a first tomography pass across train and
val, but the strict non-intrusive local-repair subset is small:

```text
slot_phase_clean_coord_close: 6 rows
```

## Same-Desc Cohort Result

Source:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v14_v2_same_desc_raw_vs_prevslot_sharded4/merged/staged_slot_guided_delta_rows.jsonl
```

Counters:

```text
input_row_count: 232
focus_row_count: 58
candidate_row_count: 54
output_row_count: 54
filtered_row_count: 174
unlabeled_focus_row_count: 4
candidate_label_counts:
  clean_transition: 19
  intrusive_transition: 35
selected_split_counts:
  train: 29
  val: 25
clean_coord_close_count: 6
cap_violation_count: 0
```

Label summaries:

```text
clean_transition:
  row_count: 19
  rank_improved_rate: 1.0
  clean_coord_close_rate: 0.315789
  mean_rank_delta: -272.32
  mean_coord_top1_distance: 266.63

intrusive_transition:
  row_count: 35
  rank_improved_rate: 0.885714
  clean_coord_close_rate: 0.0
  mean_rank_delta: -316.66
  mean_coord_top1_distance: 51.69
```

## Expanded V6 Train-First Result

Source:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v16_v6_full_prevslot_afterx1y1_sharded3/merged/staged_slot_guided_delta_rows.jsonl
```

Scope:

```text
panel: v6_v17_per96_prex1_train_failure_val_analog_panel_train16_val2
staged rows: v4_v6_full_train_first_failure_panel
patch rows: 249
train failure seeds: 83
val analogs: 166
donor_position: staged_after_x1_y1
delta_mode: donor_minus_previous_slot
patch_scale: 1.0
layer_index: 24
patch_site: layer_input
error_count: 0
```

The capped selector keeps 202 rows for balanced follow-up:

```text
input_row_count: 249
focus_row_count: 249
candidate_row_count: 227
output_row_count: 202
unlabeled_focus_row_count: 22
candidate_label_counts:
  clean_transition: 68
  intrusive_transition: 159
selected_split_counts:
  train: 79
  val: 123
clean_coord_close_count: 23
```

The uncapped selector preserves all 227 labeled rows:

```text
clean_transition:
  train: 29
  val: 39

intrusive_transition:
  train: 50
  val: 109

strict clean coord-close:
  train: 5
  val: 18
```

By regime, strict clean coord-close rows are distributed rather than a single
semantic anecdote:

```text
duplicate_basin_nearby: 9
termination_tail: 5
small_object: 4
repeated_class: 3
crowded: 2
```

The expanded result changes the practical follow-up: strict clean coord-close
is still a minority route, but it is no longer only a handful of rows. There
are now enough train/val examples to localize what distinguishes clean local
geometry from donor-basin capture without using the old person/backpack case as
the sampling prior.

## Mechanistic Interpretation

This selector sharpens the previous finding. The previous-slot transition
handle has two separable effects:

1. It can move receiver target rank upward without donor capture.
2. It more often moves the top coordinate closer by importing donor basin.

The surprising part is that the intrusive rows are much closer in coordinate
distance on average than the clean rows:

```text
broad mean_coord_top1_distance:
  clean_transition: 284.73
  intrusive_transition: 93.23

same-desc mean_coord_top1_distance:
  clean_transition: 266.63
  intrusive_transition: 51.69

expanded v6 raw patch behavior:
  target_rank_improved_rate: 0.7149
  coord_distance_lte_16_rate: 0.2369
  target_top1_rate: 0.0080
  donor_nearer_than_receiver_rate: 0.6386
  slot_intrusion_rate: 0.6386
```

So rank improvement alone is not enough evidence of object-cursor repair.
Future localization should focus on the strict intersection:

```text
clean_transition AND slot_phase_clean_coord_close
```

and compare it against:

```text
intrusive_transition AND donor_nearer_than_receiver
```

This should isolate the subspace/head behavior that writes local geometry
without importing donor ownership.

## Next Use

Recommended next panel for population-first pre-x1 localization:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_delta_contrast_selector/v4_v16_full_v6_prevslot_afterx1y1_uncapped/staged_slot_delta_contrast_rows.jsonl
```

Use capped v3 for balanced aggregate summaries, and uncapped v4 for row mining.
The older v1/v2 contrast panels remain useful controls because they include the
raw-vs-previous-slot and same-desc launch-filter comparisons, but v4 is the
better next source for strict clean/intrusive localization.

The next GPU-backed step should build attention/value rows for:

```text
label=clean_transition, slot_phase_clean_coord_close=true
label=intrusive_transition, donor_nearer_than_receiver=true
```

over the receiver pre-x1 states and, if budget allows, over the donor/control
states as paired contexts.

## Verification

```bash
python -m pytest tests/analysis/test_staged_slot_delta_contrast_selector.py -q
python -m py_compile \
  src/analysis/autoregressive_binding_template_ablation/staged_slot_delta_contrast_selector.py \
  scripts/analysis/run_autoregressive_binding_staged_slot_delta_contrast_selector.py \
  tests/analysis/test_staged_slot_delta_contrast_selector.py
```

Results:

```text
3 passed
py_compile passed
```
