---
doc_id: progress.diagnostics.row_specific_bbox_divergence_contrast_panel_findings
layer: progress
doc_type: findings
status: active
domain: mechanistic-diagnosis
summary: Region-preserving contrast panel that joins corrected row-specific bbox divergence rows back to rich token-surface rows for the next hidden-state and intervention probes.
tags:
  - autoregressive-binding-template-ablation
  - row-specific-bbox
  - divergence-panel
  - contrast-panel
  - token-surface
updated: 2026-06-20
---

# Row-Specific Bbox Divergence Contrast Panel

## Scope

This note records the selector artifact built after the corrected
row-specific bbox surface divergence pass. The purpose is operational: select
model-ready rows for the next GPU-backed hidden-state, continuation, or
intervention probes while preserving the `value_source_region` split.

This is a readout-only selector. It does not run model perturbations or training.

## Inputs

- Corrected divergence selector rows:
  `/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/row_specific_bbox_surface_divergence/joint_boundary_head26_10_row_specific_bbox_surfaces_all36_v2/row_specific_bbox_surface_divergence_state_rows.jsonl`
- Rich row-specific bbox token-surface rows:
  `/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_residual_token_surface_alignment/joint_boundary_head26_10_row_specific_bbox_surfaces_all36_union_v1/row_specific_bbox_token_surface_alignment_union_rows.jsonl`

## Outputs

Artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/row_specific_bbox_divergence_contrast_panel/joint_boundary_head26_10_v1
```

Primary files:

- `row_specific_bbox_divergence_contrast_panel_rows.jsonl`
- `row_specific_bbox_divergence_contrast_panel_summary.json`
- `row_specific_bbox_divergence_contrast_panel.md`
- `selected_rows_by_value_source_region/current_object_ref_boundaries.jsonl`
- `selected_rows_by_value_source_region/current_prefix_all.jsonl`
- `selected_rows_by_value_source_region/pre_prefix_non_image_context.jsonl`

CLI:

```bash
python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage select-row-specific-bbox-divergence-contrast-panel \
  --row-specific-bbox-divergence-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/row_specific_bbox_surface_divergence/joint_boundary_head26_10_row_specific_bbox_surfaces_all36_v2/row_specific_bbox_surface_divergence_state_rows.jsonl \
  --row-specific-bbox-surface-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_residual_token_surface_alignment/joint_boundary_head26_10_row_specific_bbox_surfaces_all36_union_v1/row_specific_bbox_token_surface_alignment_union_rows.jsonl \
  --contrast-panel-max-rows-per-role 3 \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/row_specific_bbox_divergence_contrast_panel/joint_boundary_head26_10_v1
```

## Summary

```text
input_divergence_row_count: 36
input_surface_row_count: 1080
row_count: 15
state_count: 7
case_count: 6
contrast_panel_role_counts:
  duplicate_all_negative_region_split: 3
  duplicate_current_open_box_owner: 3
  failure_target_owner_control: 3
  previous_basin_control: 3
  unmatched_current_owner_control: 3
eligible_role_counts:
  duplicate_all_negative_region_split: 3
  duplicate_current_open_box_owner: 3
  failure_target_owner_control: 6
  previous_basin_control: 4
  unmatched_current_owner_control: 8
missing_surface_match_counts: {}
duplicate_suppressed_counts: {}
role_selection_shortfall_counts:
  duplicate_all_negative_region_split: 0
  duplicate_current_open_box_owner: 0
  failure_target_owner_control: 0
  previous_basin_control: 0
  unmatched_current_owner_control: 0
value_source_region_counts:
  current_object_ref_boundaries: 6
  current_prefix_all: 4
  pre_prefix_non_image_context: 5
readout_only: true
```

## Main Rows

The panel keeps both high-value duplicate-onset contrasts from the corrected
divergence selector:

- `desc_first-885-8-0-desc_end`: `duplicate_current_open_box_owner`, strong
  target/current divergence, positive current-open-box surface evidence across
  all three selected regions.
- `desc_first-2685-33-11-desc_end`: `duplicate_all_negative_region_split`,
  strong target/current divergence, all projections negative with the least
  negative owner changing by region (`current`, `target`, then `previous`).

The panel also keeps three control roles:

- `failure_target_owner_control`: target-owner rows from
  `failure_default128_rescued`.
- `previous_basin_control`: previous-owner rows on a clean unmatched onset.
- `unmatched_current_owner_control`: weak current-owner rows from
  `failure_joint128_only`.

## Interpretation

This selector strengthens the current evidence hierarchy in one narrow way: the
duplicate-onset surface picture is not one averaged phenomenon. The two strongest
duplicate cases require separate downstream probes:

1. A positive current-open-box owner regime, where the candidate state already
   aligns with the open current box rather than the target box.
2. An all-negative, region-split regime, where relative owner preference changes
   across `current_prefix_all`, `pre_prefix_non_image_context`, and
   `current_object_ref_boundaries`.

The next hidden-state or value-region intervention run should consume one split
file at a time and pass the matching `--value-source-regions` value. Combining
the split files without preserving region would reintroduce the collapse fixed in
the divergence selector.

## Verification

```text
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q -k 'bbox_divergence_contrast_panel or row_specific_bbox_surface_divergence'
Pytest: 6 passed
```

The real artifact run above completed with `row_count=15`,
`input_divergence_row_count=36`, `input_surface_row_count=1080`,
`missing_surface_match_counts={}`, and `role_selection_shortfall_counts=0` for
all selected roles.
