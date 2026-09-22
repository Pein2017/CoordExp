---
doc_id: progress.diagnostics.candidate_rollout_label_findings_2026_06_22
layer: progress
doc_type: diagnostic-implementation-note
status: active-branch-evidence
evidence_scope: cpu-posthoc-rollout-labels-for-first200-train-val-candidate-bank
domain: autoregressive-binding-template-ablation
updated: 2026-06-22
branch: codex/autoregressive-binding-template-study
---

# Candidate Rollout Label Findings

## Purpose

This note records a CPU-only bridge from GT-structure candidate rows to observed
rollout behavior for the existing checkpoint-928 val200 run.

The previous candidate bank intentionally avoided claiming that selected rows
were model failures. This step attaches existing evaluation matches to the
first-200 val candidates so we can separate real false-negative/control rows
from merely interesting GT structure before running hidden-state or suffix-replay
probes.

Train rows remain unlabeled by this artifact because no matched train rollout is
attached yet.

## Implementation

Added:

```text
src/analysis/autoregressive_binding_template_ablation/candidate_rollout_labels.py
scripts/analysis/run_autoregressive_binding_candidate_rollout_labels.py
tests/analysis/test_candidate_rollout_labels.py
```

The labeler indexes eval match rows by true COCO image id parsed from
`file_name`, because `eval/matches.jsonl` stores an ordinal eval `image_id`.

Candidate rows are joined by:

```text
split == val
candidate image_id == parsed match-row image id
candidate object_idx == match gt_idx
```

Label statuses:

```text
matched_iou50_sem_ok
matched_iou50_sem_bad
relaxed_match_iou30_only_sem_ok
relaxed_match_iou30_only_sem_bad
unmatched_iou30
unmatched_iou50
missing_match_image
not_val_split
```

Rows also carry:

```text
rollout_evidence_status
rollout_label_is_false_negative_iou50
rollout_label_is_false_negative_iou30
rollout_iou50
rollout_iou30
rollout_pred_idx_iou50
rollout_pred_desc_iou50
rollout_pred_idx_iou30
rollout_pred_desc_iou30
rollout_match_image_id
rollout_eval_image_id
```

## Verification

TDD red:

```text
python -m pytest tests/analysis/test_candidate_rollout_labels.py -q
```

failed before implementation with:

```text
ModuleNotFoundError: No module named 'src.analysis.autoregressive_binding_template_ablation.candidate_rollout_labels'
```

Green:

```text
python -m pytest tests/analysis/test_candidate_rollout_labels.py -q
```

result:

```text
5 passed
```

Adjacent regression slice:

```text
python -m pytest tests/analysis/test_candidate_rollout_labels.py tests/analysis/test_train_val_candidate_bank.py tests/analysis/test_formation_position_rows.py -q
```

result:

```text
24 passed
```

Compile check:

```text
python -m py_compile src/analysis/autoregressive_binding_template_ablation/candidate_rollout_labels.py scripts/analysis/run_autoregressive_binding_candidate_rollout_labels.py tests/analysis/test_candidate_rollout_labels.py
```

result:

```text
passed
```

## Artifact

Candidate input:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/train_val_candidate_bank/v3_bbox_len12000_first200_gt_structure_regimes_per8_imgcap2/train_val_formation_candidate_bank.jsonl
```

Rollout input:

```text
/data/CoordExp/outputs/infer/natadj_len12000_free_val200/natadj_sorted_ckpt928_val200_free_temp0_rp1p10_max3084_bsz1_8gpu_symlinkbase
```

Output:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/candidate_rollout_labels/v1_first200_candidate_bank_ckpt928_val200_iou50_iou30
```

Counters:

```text
row_count=96
val_candidate_rows=48
val_attached_rows=48
val_missing_image_rows=0
row_counts_by_rollout_match_status={
  matched_iou50_sem_ok: 28,
  not_val_split: 48,
  relaxed_match_iou30_only_sem_ok: 2,
  unmatched_iou30: 18
}
row_counts_by_false_negative_iou50={
  False: 28,
  None: 48,
  True: 20
}
row_counts_by_false_negative_iou30={
  False: 30,
  None: 48,
  True: 18
}
```

Val status by GT-structure regime:

```text
crowded:
  matched_iou50_sem_ok: 3
  unmatched_iou30: 5

duplicate_basin_nearby:
  matched_iou50_sem_ok: 4
  relaxed_match_iou30_only_sem_ok: 1
  unmatched_iou30: 3

repeated_class:
  matched_iou50_sem_ok: 5
  relaxed_match_iou30_only_sem_ok: 1
  unmatched_iou30: 2

simple_control:
  matched_iou50_sem_ok: 8

small_object:
  matched_iou50_sem_ok: 1
  unmatched_iou30: 7

termination_tail:
  matched_iou50_sem_ok: 7
  unmatched_iou30: 1
```

## Immediate Read

This is not yet a hidden-state mechanism result, but it gives a better next
probe panel:

```text
strong FN pool:
  small_object: 7/8 unmatched at IoU 0.30
  crowded: 5/8 unmatched at IoU 0.30
  duplicate_basin_nearby: 3/8 unmatched at IoU 0.30

clean control pool:
  simple_control: 8/8 matched at IoU 0.50

mixed structure pool:
  repeated_class and termination_tail include both matched and unmatched rows
```

Promising next probe rows include:

```text
source_line_idx=123 image_id=12670 object_idx=3  regime=crowded       desc=person  unmatched_iou30
source_line_idx=72  image_id=7281  object_idx=9  regime=small_object  desc=person  unmatched_iou30
source_line_idx=137 image_id=14038 object_idx=15 regime=duplicate_basin_nearby desc=book unmatched_iou30
source_line_idx=123 image_id=12670 object_idx=1  regime=crowded       desc=person  matched_iou50_sem_ok
source_line_idx=1   image_id=285   object_idx=0  regime=simple_control desc=bear   matched_iou50_sem_ok
```

The useful immediate comparison is no longer "person/backpack pair only." It is:

```text
same image / same crowded sequence matched vs unmatched
small-object false negatives vs simple-control matches
nearby same-desc duplicate-basin false negatives vs nearby same-desc matches
```

Train-vs-val remains only half solved:

```text
val:
  rollout labels attached for first-200 selected candidates

train:
  GT-structure candidate rows exist, but observed failure/control labels require
  either a matched train rollout or teacher-forced formation-state probes
```
