---
doc_id: progress.diagnostics.panel_formation_rows_findings_2026_06_22
layer: progress
doc_type: diagnostic-implementation-note
status: active-branch-evidence
evidence_scope: cpu-teacher-forced-formation-rows-from-rollout-labeled-val-panel
domain: autoregressive-binding-template-ablation
updated: 2026-06-22
branch: codex/autoregressive-binding-template-study
---

# Panel Formation Rows Findings

## Purpose

This note records the CPU bridge from the compact rollout-labeled probe panel to
teacher-forced formation-position rows.

The goal is not to conclude a mechanism yet. The goal is to produce exact
hidden-state probe rows for the next GPU pass, while preserving rollout false
negative/control labels and the GT object context that precedes each target
object.

This step also records a boundary for the next round: the current artifact is a
val-only panel and should be expanded across more dataset rows, including train
rows, before claiming a train-vs-val mechanism.

## Implementation

Added:

```text
src/analysis/autoregressive_binding_template_ablation/panel_formation_rows.py
scripts/analysis/run_autoregressive_binding_panel_formation_rows.py
tests/analysis/test_panel_formation_rows.py
```

The builder:

```text
joins panel rows back to the split-specific bbox_len12000 JSONL row by source_line_idx
validates image_id, object_idx, target desc, and target bbox
renders the teacher-forced GT prefix through the target object's object_ref_start
tokenizes the target descriptor with the local tokenizer when provided
emits the eight formation positions used by formation_position_rows.py
preserves panel_role, rollout_match_status, candidate regime, and rollout evidence
falls back from panel_role to candidate_regime for broader candidate-bank rows
```

Tokenizer surface:

```text
/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent
```

This matters because multi-token descriptors keep space-bearing token pieces
such as:

```text
sports ball -> ["sports", " ball"]
potted plant -> ["p", "otted", " plant"]
```

## Verification

TDD red:

```text
python -m pytest tests/analysis/test_panel_formation_rows.py -q
```

failed before implementation or wrapper completion.

Green:

```text
python -m pytest tests/analysis/test_panel_formation_rows.py -q
```

result:

```text
5 passed
```

Compile check:

```text
python -m py_compile src/analysis/autoregressive_binding_template_ablation/panel_formation_rows.py scripts/analysis/run_autoregressive_binding_panel_formation_rows.py tests/analysis/test_panel_formation_rows.py
```

result:

```text
passed
```

## Artifact

Compact rollout-labeled val panel input:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/candidate_probe_panel/v1_first200_ckpt928_val200_matched_unmatched_panel/candidate_probe_panel_rows.jsonl
```

Dataset inputs:

```text
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl
/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl
```

Compact rollout-labeled val panel output:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/panel_formation_row_builder/v1_first200_ckpt928_val200_probe_panel_teacher_forced_gt
```

Counters:

```text
input_panel_row_count=31
selected_panel_row_count=31
output_row_count=248
skipped_counts_by_reason={}
```

Formation-position counts:

```text
descriptor_onset: 31
descriptor_end: 31
object_ref_end: 31
box_start: 31
pre_x1: 31
post_x1: 31
box_close: 31
next_object_onset: 31
```

Rollout-status counts after expansion to positions:

```text
matched_iou50_sem_ok: 128
relaxed_match_iou30_only_sem_ok: 16
unmatched_iou30: 104
```

Panel-role counts after expansion to positions:

```text
same_image_matched_control: 32
same_image_unmatched_fn: 32
small_object_fn: 24
small_object_matched_control: 8
duplicate_nearby_fn: 16
duplicate_nearby_matched_control: 24
crowded_fn: 24
crowded_matched_control: 16
termination_tail_fn: 8
termination_tail_matched_control: 24
relaxed_match_boundary: 16
simple_control_matched: 24
```

Candidate-regime counts after expansion to positions:

```text
crowded: 56
duplicate_basin_nearby: 64
repeated_class: 40
simple_control: 24
small_object: 32
termination_tail: 32
```

Broader train/val candidate-bank input:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/train_val_candidate_bank/v2_bbox_len12000_gt_structure_regimes_per8_imgcap2/train_val_formation_candidate_bank.jsonl
```

Broader train/val candidate-bank output:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/panel_formation_row_builder/v2_full_train_val_candidate_bank_teacher_forced_gt
```

Broader counters:

```text
input_panel_row_count=96
selected_panel_row_count=96
output_row_count=768
skipped_counts_by_reason={}
row_counts_by_split={train: 384, val: 384}
```

Broader candidate-regime counts after expansion to positions:

```text
crowded: 128
duplicate_basin_nearby: 128
repeated_class: 128
simple_control: 128
small_object: 128
termination_tail: 128
```

The broader artifact covers 28 target descriptors after expansion:

```text
apple, backpack, banana, bicycle, bird, book, bowl, bus, cake, car,
cell phone, chair, dog, fork, giraffe, knife, orange, person, pizza, sheep,
spoon, sports ball, suitcase, tennis racket, tie, traffic light, train,
wine glass
```

## Immediate Read

This gives a clean teacher-forced readout panel for checkpoint-928 hidden-state
and suffix-replay probes:

```text
same-image matched versus unmatched controls
small-object false negatives
nearby same-desc duplicate-basin rows
crowded rows
termination-tail rows
IoU30-only coordinate-boundary rows
simple matched controls
```

The artifact is intentionally conservative:

```text
scope: val200-derived panel, teacher-forced GT prefix
not yet: train rollout labels
not yet: rollout-prefix hidden states
not yet: direct evidence that the model visually cannot perceive a missing object
```

## Next Direction

The next round should not stay constrained to the original person/backpack pair
or even to this compact val panel.

Promising near-term expansion after this artifact:

```text
compare train versus val teacher-forced target mass, coordinate-basin margins,
and termination/continue margins at descriptor_onset, box_start, pre_x1, and
next_object_onset
join model-backed readouts back to candidate_regime, target_desc, object_idx,
objects_remaining_after, and same_desc_count
```

Mechanistic question:

```text
If a trained-sequence object still fails under teacher-forced readout, the issue
is likely not only held-out visual generalization.

If train teacher-forced states are strong but free rollout omits the object, the
main failure may be language-side cursor/context fragility or autoregressive
competition.

If val false negatives are weak only when the GT prefix reaches ambiguous local
neighborhoods, the candidate mechanism becomes distributed visual evidence plus
late coordinate-basin competition rather than simple absence of object
perception.
```
