---
doc_id: progress.diagnostics.candidate_probe_panel_findings_2026_06_22
layer: progress
doc_type: diagnostic-implementation-note
status: active-branch-evidence
evidence_scope: cpu-probe-panel-selection-from-rollout-labeled-first200-candidates
domain: autoregressive-binding-template-ablation
updated: 2026-06-22
branch: codex/autoregressive-binding-template-study
---

# Candidate Probe Panel Findings

## Purpose

This note records the compact CPU probe panel selected from the rollout-labeled
first-200 candidate bank.

The purpose is to choose a small, evidence-backed set of matched/unmatched rows
for the next hidden-state and suffix-replay probes. This is still not a
mechanism result. It is the bridge between broad row discovery and expensive
model-state analysis.

## Implementation

Added:

```text
src/analysis/autoregressive_binding_template_ablation/candidate_probe_panel.py
scripts/analysis/run_autoregressive_binding_candidate_probe_panel.py
tests/analysis/test_candidate_probe_panel.py
```

The selector preserves all labeled candidate-row fields and adds:

```text
panel_role
panel_group_id
panel_selection_rank
panel_priority
panel_selection_reason
panel_case_id
recommended_next_probe
```

The selector prioritizes:

```text
same-image matched/unmatched contrasts
small-object false negatives and matched controls
nearby same-desc duplicate-basin false negatives and matched controls
crowded false negatives and matched controls
termination-tail false negatives and matched controls
IoU30-only relaxed boundary cases
simple matched controls
```

## Verification

TDD red:

```text
python -m pytest tests/analysis/test_candidate_probe_panel.py -q
```

failed before implementation with:

```text
ModuleNotFoundError: No module named 'src.analysis.autoregressive_binding_template_ablation.candidate_probe_panel'
```

Green:

```text
python -m pytest tests/analysis/test_candidate_probe_panel.py -q
```

result:

```text
5 passed
```

Adjacent regression slice:

```text
python -m pytest tests/analysis/test_candidate_probe_panel.py tests/analysis/test_candidate_rollout_labels.py tests/analysis/test_train_val_candidate_bank.py -q
```

result:

```text
24 passed
```

Compile check:

```text
python -m py_compile src/analysis/autoregressive_binding_template_ablation/candidate_probe_panel.py scripts/analysis/run_autoregressive_binding_candidate_probe_panel.py tests/analysis/test_candidate_probe_panel.py
```

result:

```text
passed
```

## Artifact

Input:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/candidate_rollout_labels/v1_first200_candidate_bank_ckpt928_val200_iou50_iou30/train_val_formation_candidate_bank_rollout_labeled.jsonl
```

Output:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/candidate_probe_panel/v1_first200_ckpt928_val200_matched_unmatched_panel
```

Counters:

```text
input_row_count=96
output_row_count=31
unique_candidate_count=31
duplicate_candidate_skipped_count=4
row_counts_by_rollout_match_status={
  matched_iou50_sem_ok: 16,
  relaxed_match_iou30_only_sem_ok: 2,
  unmatched_iou30: 13
}
```

Panel roles:

```text
same_image_matched_control: 4
same_image_unmatched_fn: 4
small_object_fn: 3
small_object_matched_control: 1
duplicate_nearby_fn: 2
duplicate_nearby_matched_control: 3
crowded_fn: 3
crowded_matched_control: 2
termination_tail_fn: 1
termination_tail_matched_control: 3
relaxed_match_boundary: 2
simple_control_matched: 3
```

## High-Value Rows

Same-image matched/unmatched contrasts:

```text
image_id=18380 source_line_idx=181 desc=person regime=repeated_class
  matched object_idx=6  vs unmatched object_idx=12

image_id=12670 source_line_idx=123 desc=person regime=crowded
  matched object_idx=1  vs unmatched object_idx=3

image_id=4134 source_line_idx=42 desc=person regime=repeated_class
  matched object_idx=7  vs unmatched object_idx=13

image_id=16228 source_line_idx=159 desc=person regime=duplicate_basin_nearby
  matched object_idx=12 vs unmatched object_idx=15
```

Other high-value roles:

```text
small_object_fn:
  image_id=7281  object_idx=9  desc=person
  image_id=13348 object_idx=4  desc=person
  image_id=18380 object_idx=36 desc=carrot

duplicate_nearby_fn:
  image_id=14038 object_idx=15 desc=book
  image_id=14038 object_idx=16 desc=book

relaxed_match_boundary:
  image_id=12670 object_idx=21 desc=person
  image_id=17182 object_idx=3  desc=book

simple_control_matched:
  image_id=285   object_idx=0 desc=bear
  image_id=5503  object_idx=0 desc=toilet
  image_id=10995 object_idx=0 desc=bed
```

## Immediate Read

The panel now gives three mechanistic comparison types:

```text
same-image contrast:
  visually similar context, same image, matched versus unmatched target object

regime contrast:
  small-object / nearby-duplicate / crowded false negatives versus matched
  controls from the same structural regime

boundary contrast:
  IoU30-only relaxed matches for coordinate-basin margin and weak-localization
  analysis
```

Recommended next CPU step:

```text
materialize teacher-forced formation-position rows from the GT target sequence
for this panel, preserving panel_role and rollout_match_status.
```

Recommended next GPU step after that:

```text
compare teacher-forced target-desc / target-coordinate logits against rollout
termination or matched-emission prefixes to separate visual availability from
autoregressive routing failure.
```
