---
doc_id: progress.diagnostics.train_val_candidate_bank_findings_2026_06_22
layer: progress
doc_type: diagnostic-implementation-note
status: active-branch-evidence
evidence_scope: cpu-gt-structure-candidate-bank-over-bbox-len12000-train-val
domain: autoregressive-binding-template-ablation
updated: 2026-06-22
branch: codex/autoregressive-binding-template-study
---

# Train-Vs-Val Candidate Bank Findings

## Purpose

This note records the first Task 7 implementation slice from the binding-state
formation roadmap: a deterministic train-vs-val candidate bank over
`bbox_len12000` COCO coord-token rows.

The purpose is not to claim that a selected row is a model failure. It prevents
the next hidden-state and suffix-replay probes from staying trapped in the
current person/backpack-style pair by selecting object rows from broader GT
structure regimes before model evidence is attached.

## Implementation

Added:

```text
src/analysis/autoregressive_binding_template_ablation/train_val_candidate_bank.py
scripts/analysis/run_autoregressive_binding_train_val_candidate_bank.py
tests/analysis/test_train_val_candidate_bank.py
```

The builder scans COCO image rows and emits object-level candidate rows for:

```text
repeated_class
duplicate_basin_nearby
crowded
small_object
termination_tail
simple_control
```

Rows include:

```text
split
source_line_idx
image_id
file_name
object_idx
object_count
objects_remaining_after
target_desc
target_category_name
target_category_id
target_coco_ann_id
target_bbox
bbox_area
same_desc_count
nearest_same_desc_object_idx
nearest_same_desc_center_l1
nearest_same_desc_iou
candidate_regime
selection_rank
candidate_bank_case_id
selection_scope=gt_structure_only
rollout_evidence_status=not_attached
```

Selection is deterministic and capped by both:

```text
per_regime_per_split
max_per_image_per_regime
```

The image cap was added after the first full scan showed that top-k selection
could over-concentrate on one extreme image per regime. This is important for
the research question: a diverse bank is more useful than a maximum-extreme bank
when probing whether train and val failures share the same binding/coordinate
basin mechanism.

## Verification

TDD red:

```text
python -m pytest tests/analysis/test_train_val_candidate_bank.py -q
```

failed before implementation with:

```text
ModuleNotFoundError: No module named 'src.analysis.autoregressive_binding_template_ablation.train_val_candidate_bank'
```

The diversity-cap test then failed before the cap implementation with:

```text
TypeError: build_train_val_candidate_bank() got an unexpected keyword argument 'max_per_image_per_regime'
```

Green:

```text
python -m pytest tests/analysis/test_train_val_candidate_bank.py -q
```

result:

```text
6 passed
```

Compile check:

```text
python -m py_compile src/analysis/autoregressive_binding_template_ablation/train_val_candidate_bank.py scripts/analysis/run_autoregressive_binding_train_val_candidate_bank.py tests/analysis/test_train_val_candidate_bank.py
```

result:

```text
passed
```

## Full-Split Bank

Input:

```text
train: /data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl
val:   /data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl
```

Output:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/train_val_candidate_bank/v2_bbox_len12000_gt_structure_regimes_per8_imgcap2
```

Counters:

```text
row_count=96
unique_object_count=96
source_row_counts={train: 117266, val: 4952}
processed_source_row_counts={train: 117266, val: 4952}
skipped_counts_by_reason={}
row_counts_by_split={train: 48, val: 48}
row_counts_by_regime={
  crowded: 16,
  duplicate_basin_nearby: 16,
  repeated_class: 16,
  simple_control: 16,
  small_object: 16,
  termination_tail: 16
}
max_per_image_per_regime=2
max_rows_per_image_regime_observed=2
image_cap_violation_count=0
```

The selected full-split val rows mostly do not overlap the existing val200
rollout:

```text
val selected source_line_idx range: 181..4744
val selected rows with source_line_idx < 200: 3 / 48
```

This means the full-split bank is the right broad target, but it needs a new
rollout or probe capture before post-hoc model-failure labels can be attached to
most selected val rows.

## First-200 Bank

Output:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/train_val_candidate_bank/v3_bbox_len12000_first200_gt_structure_regimes_per8_imgcap2
```

Counters:

```text
row_count=96
unique_object_count=93
processed_source_row_counts={train: 200, val: 200}
skipped_counts_by_reason={}
row_counts_by_split={train: 48, val: 48}
row_counts_by_regime={
  crowded: 16,
  duplicate_basin_nearby: 16,
  repeated_class: 16,
  simple_control: 16,
  small_object: 16,
  termination_tail: 16
}
val selected rows with source_line_idx < 200: 48 / 48
max_rows_per_image_regime_observed=2
image_cap_violation_count=0
```

This first-200 bank is the bridge artifact for immediate post-hoc labeling
against the existing val200 rollout:

```text
/data/CoordExp/outputs/infer/natadj_len12000_free_val200/natadj_sorted_ckpt928_val200_free_temp0_rp1p10_max3084_bsz1_8gpu_symlinkbase
```

The train side still needs either a matched train rollout or a teacher-forced /
prefix-state probe capture before it can answer whether trained-sequence
failures share the same mechanism as unseen-val failures.

## Interpretation

Current conclusion:

```text
The candidate-selection bottleneck is now fixed enough to leave the
person/backpack pair behind. It is not yet a mechanism result.
```

Most useful next step:

```text
Attach val200 rollout labels to the first-200 val candidate rows, then select a
small matched train cohort for either free rollout or teacher-forced
formation-state capture. Only after that should GPU suffix-replay probes run on
the broader formation-position rows.
```

The next probe should preserve both banks:

```text
v2 full-split bank:
  broad target for new train/val rollout or probe capture

v3 first-200 bank:
  immediate bridge to existing val200 rollout evidence
```
