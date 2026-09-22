---
doc_id: progress.diagnostics.formation_position_row_builder_findings_2026_06_22
layer: progress
doc_type: diagnostic-implementation-note
status: active-branch-evidence
evidence_scope: cpu-row-builder-smoke-over-current-selected-boundary-rows
domain: autoregressive-binding-template-ablation
updated: 2026-06-22
branch: codex/autoregressive-binding-template-study
---

# Formation Position Row Builder Findings

## Purpose

This note records Task 1 from the binding-state formation roadmap: a deterministic
CPU-only row builder for meaningful formation cutpoints around selected
post-box/pre-x1 object states.

The purpose is not to claim a new mechanism result. It creates the row surface
needed for suffix-replayed formation probes.

## Implementation

Added:

```text
src/analysis/autoregressive_binding_template_ablation/formation_position_rows.py
scripts/analysis/run_autoregressive_binding_formation_position_rows.py
tests/analysis/test_formation_position_rows.py
```

The builder consumes selected post-box descriptor-boundary rows and emits one row
per formation position:

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

Rows preserve:

```text
post_box_boundary_case_id
post_box_boundary_variant_role
source_line_idx
image_id
object_idx
target_desc
target_bbox
completed_box_bbox
current_generated_bbox
state_key provenance
```

Rows also include replay-oriented fields:

```text
formation_position
formation_source_state_key
formation_suffix_text
trajectory_prefix_text
assistant_prefix_text
target_next_kind
target_next_token_text
target_next_coord_slot
target_next_coord_bin
coord_probe_bins_requested
```

This is intentionally model-free. The next model-backed stage must patch earlier
positions and replay the suffix so later layer-input states are recomputed.

## Verification

TDD red:

```text
python -m pytest tests/analysis/test_formation_position_rows.py -q
```

failed before implementation with:

```text
ModuleNotFoundError: No module named 'src.analysis.autoregressive_binding_template_ablation.formation_position_rows'
```

Green:

```text
python -m pytest tests/analysis/test_formation_position_rows.py -q
```

result:

```text
6 passed
```

Compile check:

```text
python -m py_compile src/analysis/autoregressive_binding_template_ablation/formation_position_rows.py scripts/analysis/run_autoregressive_binding_formation_position_rows.py tests/analysis/test_formation_position_rows.py
```

result:

```text
passed
```

Adjacent regression slice:

```text
python -m pytest tests/analysis/test_formation_position_rows.py tests/analysis/test_post_box_pre_x1_rows.py tests/analysis/test_forced_coordinate_rows.py -q
```

result:

```text
19 passed
```

## Smoke Artifact

Input selected rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_box_counterfactual_boundary_selection/v1_desc_first_strict_selected_descriptor_flips_v1/post_box_counterfactual_boundary_selected_rows.jsonl
```

Output:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/formation_position_row_builder/v1_desc_first_strict_selected_descriptor_flips_formation_positions_v1
```

Counters:

```text
input_selected_row_count=7
output_row_count=56
skipped_counts_by_reason={}
formation_position_counts={
  box_close: 7,
  box_start: 7,
  descriptor_end: 7,
  descriptor_onset: 7,
  next_object_onset: 7,
  object_ref_end: 7,
  post_x1: 7,
  pre_x1: 7
}
post_box_boundary_variant_role_counts={
  baseline: 16,
  flip: 24,
  stable_counterfactual: 16
}
```

This smoke includes the current selected rows beyond the synthetic unit-test
fixture. It is still limited to the existing selected descriptor-flip cohort, so
it is not a train-vs-val or broad-regime panel.

## Train-Vs-Val Expansion Hook

The next selector should not stay constrained to the person/backpack pair or
the seven current selected rows. The active bbox_len12000 object JSONL roots are:

```text
train: /data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl
val:   /data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl
```

Observed row counts:

```text
train rows: 117266
val rows: 4952
```

The next case-selection layer should sample both:

```text
training-sequence failures or weak states:
  tests how the trained sequence itself can still fall into wrong binding basins

validation/unseen failures:
  tests whether unseen examples fail through the same row-state mechanism or a
  different visual/language generalization path
```

Recommended next move:

```text
build a train-vs-val formation candidate bank, then materialize formation rows
with this builder before running GPU suffix-replay probes.
```

This should be a separate selector task. The current row builder stays generic
and deterministic: it consumes selected rows from any cohort once those rows
exist.
