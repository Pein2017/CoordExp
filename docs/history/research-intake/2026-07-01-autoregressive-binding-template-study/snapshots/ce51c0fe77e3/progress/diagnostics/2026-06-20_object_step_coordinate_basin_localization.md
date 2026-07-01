---
doc_id: progress.diagnostics.object_step_coordinate_basin_localization
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: val200-desc-first-20state-object-step-value-region-panel
domain: autoregressive-binding-template-ablation
updated: 2026-06-20
branch: codex/autoregressive-binding-template-study
---

# Object-Step Coordinate Basin Localization

## Scope

This note records the first selector-backed object-step coordinate panel for the
`token_embeddings_adapter` / `bbox_len12000` checkpoint pair. The panel is a
small mechanistic slice, not a full validation run.

The goal was to move from rough rollout symptoms to exact object-span
trajectory states:

- where the next token is a coordinate token from the emitted object box;
- where the prefix is the realized assistant prefix immediately before that
  coordinate slot;
- where source rows are joined back to `object_step_rows.jsonl` and
  `pred_token_trace.jsonl` with family and role-token guards.

Primary model/data surface:

```text
pair config:
configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml

probe root:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200_deep_probe_v2_token_embeddings_surface

rollout root:
/data/CoordExp/outputs/infer/natadj_len12000_free_val200/natadj_sorted_ckpt928_val200_free_temp0_rp1p10_max3084_bsz1_8gpu_symlinkbase
```

## Implementation

New selector:

```text
src/analysis/autoregressive_binding_template_ablation/trajectory_object_step_selector.py
tests/analysis/test_autoregressive_binding_template_trajectory_object_step_selector.py
```

Core contract:

- emits value-region continuation selected rows at realized object coordinate
  slots;
- uses `intervention_arm=natural`, `guidance_source=object_step_bbox_tokens`,
  `trajectory_stop_reason=object_step_role`, and `target_next_kind=coord`;
- reconstructs the assistant prefix from `pred_token_trace.jsonl`;
- parses target boxes from `<|coord_*|>` tokens rather than pixel `points`;
- validates that the token trace role index actually points to the same
  coordinate token as the object row's `bbox_tokens` slot;
- rejects mixed-family raw object-step inputs unless an explicit single-family
  `families` filter is supplied.

The mixed-family guard matters because the current object-step artifact is
mixed-family:

```text
187 desc_first
274 geometry_first
```

A no-family selector call over the mixed artifact now fails even if another
filter such as `source_line_idxs` would happen to select only one family. The
explicit-family path was smoke-tested on the real artifact:

```text
selected_row_count: 4
input_object_row_count: 461
candidate_object_row_count: 35
row_counts_by_family: {"desc_first": 4}
skipped_role_bbox_token_mismatch_count: 0
```

Targeted verification:

```text
python -m pytest tests/analysis/test_autoregressive_binding_template_trajectory_object_step_selector.py
# Pytest: 11 passed

python -m py_compile src/analysis/autoregressive_binding_template_ablation/trajectory_object_step_selector.py
git diff --check
```

## Selector Panel

Strict selector panel:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_panel_selection/v2_desc_first_duplicate_unmatched_slots_selector_strict_v1
```

Key files:

```text
selected_rows.jsonl
selection_summary.json
```

Summary:

```text
row_count: 20
missing_panel_objects: []
stage: trajectory-object-step-selector-panel-strict
```

Panel objects:

```text
(33,10) wine glass unmatched, x1
(33,11) wine glass repeated_gt duplicate, x1/y1/x2/y2
(36,6)  person repeated_gt duplicate, x1
(36,9)  snowboard new_gt control, x1
(36,14) snowboard unmatched duplicate, x1/y1/x2/y2
(123,5) backpack unmatched diffuse basin, x1/y1/x2/y2
(145,18) broccoli unmatched precursor, x1
(145,20) broccoli duplicate/high-entropy case, x1/y1/x2/y2
```

All per-object selector summaries reported:

```text
skipped_missing_trace_count: 0
skipped_malformed_bbox_tokens_count: 0
skipped_missing_token_window_count: 0
skipped_invalid_role_index_count: 0
skipped_role_bbox_token_mismatch_count: 0
```

## Main Artifacts

Selector-backed combined-head value-region continuation:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v2_desc_first_duplicate_unmatched_slots_selector_combined_heads_regions2_scales012_steps4_v2
```

Summary:

```text
row_count: 404
continuation_status_counts: {"ok": 380, "intervention_skipped": 24}
candidate_head_group_mode: combined_all
value_source_regions: current_partial_coords,current_prefix_all
value_region_scales: 0,1,2
```

Top-tie analysis:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_top_ties/v2_desc_first_duplicate_unmatched_slots_selector_combined_heads_regions2_scales012_steps4_v2
```

Summary:

```text
row_count: 404
observed_tie_rate: 0.142105263158
```

Fractional threshold run:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v2_desc_first_slots_combined_heads_regions2_scales0_025_05_075_1_steps4_v1
```

Summary:

```text
row_count: 670
continuation_status_counts: {"ok": 630, "intervention_skipped": 40}
value_region_scales: 0,0.25,0.5,0.75,1
```

Independent head sweep A:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v2_desc_first_slots_independent_heads_a_regions2_scales012_steps4_v1
```

Heads:

```text
27:15, 27:14, 20:14, 20:3
```

Summary:

```text
row_count: 1608
continuation_status_counts: {"ok": 1512, "intervention_skipped": 96}
```

Independent head sweep B:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v2_desc_first_slots_independent_heads_b_regions2_scales012_steps4_v1
```

Heads:

```text
27:2, 24:9, 24:2, 27:3
```

Summary:

```text
row_count: 1608
continuation_status_counts: {"ok": 1512, "intervention_skipped": 96}
```

## Local Basin Thresholds

The combined-head intervention is not a generic duplicate repair. It behaves
like a slot- and prefix-dependent coordinate-basin shaper.

Helpful repeated-person x1 case:

```text
source/object/slot: 36/6/x1
target: <|coord_368|>
region: current_prefix_all

scale 0.00 -> <|coord_370|>
scale 0.25 -> <|coord_370|>
scale 0.50 -> <|coord_368|>
scale 0.75 -> <|coord_368|>
scale 1.00 -> <|coord_368|>
```

The repair threshold is around scale `0.5`.

Harmful snowboard x2 case:

```text
source/object/slot: 36/14/x2
target: <|coord_456|>
region: current_prefix_all

scale 0.00 -> <|coord_456|>
scale 0.25 -> <|coord_447|>
scale 0.50 -> <|coord_444|>
scale 0.75 -> <|coord_444|>
scale 1.00 -> <|coord_444|>
```

This slot is already correct at scale `0`; the same intervention damages it by
scale `0.25`.

Harmful backpack y1 case:

```text
source/object/slot: 123/5/y1
target: <|coord_163|>
region: current_prefix_all

scale 0.00 -> <|coord_163|>
scale 0.25 -> <|coord_163|>
scale 0.50 -> <|coord_222|>
scale 0.75 -> <|coord_219|>
scale 1.00 -> <|coord_217|>
```

This looks like a sharper basin transition rather than gradual coordinate
smoothing.

Broccoli y2 case:

```text
source/object/slot: 145/20/y2
target: <|coord_146|>

current_partial_coords scale 0 -> <|coord_154|>
current_partial_coords scale 1 -> <|coord_152|>
current_partial_coords scale 2 -> <|coord_143|>

current_prefix_all scale 0 -> <|coord_154|>
current_prefix_all scale 1 -> <|coord_154|>
current_prefix_all scale 2 -> <|coord_143|>
```

This improves only under some source-region/scale settings and overshoots at
stronger scale.

## Head Localization

The harmful snowboard x2 push is localized to independent sweep A:

```text
20:14 scale 1 -> <|coord_447|>, scale 2 -> <|coord_444|>
20:3  scale 1 -> <|coord_445|>, scale 2 -> <|coord_444|>
27:14 scale 1 -> <|coord_447|>, scale 2 -> <|coord_444|>
27:15 scale 1 -> <|coord_447|>, scale 2 -> <|coord_439|>
```

Sweep B mostly does not reproduce that destructive x2 effect:

```text
27:2, 24:9, 24:2 remain at or near <|coord_456|> through scale 1
27:3 produces only the milder <|coord_447|> shift
```

The backpack y1 high-threshold flip is mainly a `20:14` effect:

```text
20:14 scale 1 -> <|coord_217|>, scale 2 -> <|coord_222|>
20:3 stays at <|coord_163|>
27:14 drifts mildly to <|coord_158|>/<|coord_157|>
27:15 is exact at scale 1 and mildly drifts at scale 2
```

The repeated-person x1 repair is distributed over a different set of heads:

```text
27:2 repairs at scale 1/2
27:3 repairs at scale 1/2
27:15 repairs at scale 1 but reverts at scale 2
20:3 and 27:14 repair only at stronger scale
20:14 does not repair this slot
```

The broccoli y2 improvement is mostly `20:3` and partly `27:14`:

```text
20:3  scale 2 -> <|coord_145|>
27:14 scale 2 -> <|coord_145|>
20:14 and 27:15 stay around <|coord_154|>
```

## Interpretation

Current evidence favors this mechanism:

```text
the model has several nearby coordinate basins available at object coordinate
slots;
late current-prefix/current-partial value vectors select among those basins;
the same subtraction direction can repair one local duplicate basin, damage a
different already-correct coordinate slot, or push a weak slot into a far basin;
therefore duplication is not controlled by one scalar anti-duplication head or
one universal coordinate smoothness axis.
```

This is compatible with the user's earlier coordinate-token observation:
coordinate locality can survive CE supervision while coordinate smoothness is
not a reliable guarantee. In this panel, the important object-level failure is
not that the model cannot emit coordinate tokens. It is that local prefix/value
routing selects the wrong basin among legal coordinate-token basins.

The result also sharpens the false-negative question. If an object is missing,
the first question should not only be whether the vision tower perceives it.
For object spans that can be guided into existence, the likely failure mode may
be earlier binding/routing: the language-side object pointer never stabilizes
into the coordinate basin that would let the visual evidence become locally
decodable.

## Boundaries

This is a tiny 20-state diagnostic panel from val200, focused on `desc_first`.
It should not be read as an aggregate benchmark.

The post-hoc coordinate-copy reduction currently loses some source/object
metadata in its reduced rows. Use the full continuation rows for object-level
attribution until that analyzer is upgraded.

The intervention is a value-region subtraction along a selected head direction.
It is causal for that intervention surface, but it is not yet a full causal
model of visual perception, object identity formation, or termination.

## Next Directions

High-value next work:

- preserve source/object metadata in post-hoc top-tie and coordinate-copy
  reduced rows;
- run route/content attention tomography for the localized head groups:
  harmful A heads (`20:14`, `20:3`, `27:14`, `27:15`) versus repair B heads
  (`27:2`, `27:3`, `24:2`, `24:9`);
- add a bidirectional intervention mode or paired direction test to separate
  "subtracting a harmful basin selector" from "adding a target basin selector";
- build hidden-state probes around object start, desc end, box start, and each
  coordinate slot for the same selected panel;
- run a false-negative guidance bridge: for missing objects, compare visual
  evidence availability against whether language-side object/prefix guidance
  can make the correct object span and coordinate basin decodable.
