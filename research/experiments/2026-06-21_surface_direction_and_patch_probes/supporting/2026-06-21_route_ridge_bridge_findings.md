---
doc_id: progress.diagnostics.route_ridge_bridge_findings
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: tiny-four-state-y2-posthoc-route-ridge-bridge
domain: autoregressive-binding-template-ablation
updated: 2026-06-21
branch: codex/autoregressive-binding-template-study
---

# Route-Ridge Bridge Findings

## Scope

This note follows:

```text
progress/diagnostics/2026-06-21_ridge_depth_panel_reducer_findings.md
```

The previous note made ridge-depth machine-readable. This note joins that
ridge panel to existing source-region attention contrast and head value
contribution artifacts.

This is a CPU-only post-hoc artifact join. It does not run the model, patch
activations, decode, or train. Attention contrast is observational routing
evidence; value contribution rows are diagnostic head-slice readouts. The
bridge should guide the next causal intervention, not replace it.

## Helper Added

New reducer:

```text
scripts/analysis/run_autoregressive_binding_route_ridge_bridge.py \
  --ridge-summary <ridge_depth_panel_summary.json> \
  --attention-contrast-rows <attention_head_group_tomography_contrast_rows.jsonl> \
  --value-contribution-rows <trajectory_boundary_head_value_contribution_rows.jsonl> \
  --source-line-idxs 33,36,123,145 \
  --target-next-coord-slot y2 \
  --output-root <summary_root>
```

It writes:

```text
route_ridge_bridge_summary.json
route_ridge_bridge_summary.md
route_ridge_bridge_manifest.json
```

The reducer ignores non-object JSONL footer records and joins states by
`source_line_idx`, `object_idx`, and coordinate slot.

## Artifact

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/route_ridge_bridge/four_state_y2_harmful_repair_routes_v1
```

Inputs:

```text
ridge_summary:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/ridge_depth_panel/four_state_y2_onset_late_v1/ridge_depth_panel_summary.json

attention_contrast_rows:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_source_region_attention_head_group_tomography/v2_desc_first_layers20_24_27_harmful_vs_repair_regions13_v2/attention_head_group_tomography_contrast_rows.jsonl

value_contribution_rows:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_head_value_contribution/v2_desc_first_layers20_24_27_harmful_repair_regions_intrinsic_v1/trajectory_boundary_head_value_contribution_rows.jsonl
```

Manifest:

```text
state_count: 4
ridge_rows_used_count: 29
attention_rows_used_count: 52
value_rows_used_count: 1120
```

Global counts:

```text
ridge_panel_row_count: 29
attention_contrast_row_count: 260
attention_rows_ignored_count: 208
value_contribution_row_count: 5152
value_rows_ignored_count: 4032
```

The ignored attention/value rows are outside the selected source-line filter or
target slot.

## Ridge Summary

| state | ridge directions | successes | single-bin successes | local-mean successes | min first success | best target prob | best direction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| source33/object11/y2 | 10 | 5 | 2 | 3 | 256 | 0.378010 | `coord_target_minus_mean:413+414+415+416+418+419` |
| source36/object14/y2 | 9 | 4 | 0 | 4 | 224 | 0.661797 | `coord_target_minus_mean:999+995` |
| source123/object5/y2 | 5 | 5 | 4 | 1 | 64 | 0.350768 | `coord_target_minus_bin:188` |
| source145/object20/y2 | 5 | 3 | 2 | 1 | 128 | 0.550767 | `coord_target_minus_bin:158` |

This preserves the earlier ridge taxonomy:

```text
source123: broad/easy target controllability
source145: specific-antagonist steerability
source33: low-side history-anchor steerability
source36: multi-anchor local-ridge steerability
```

## Attention Route Signature

Top harmful-vs-repair attention contrast:

| state | strongest harmful routes | strongest repair route |
| --- | --- | --- |
| source33/object11/y2 | current_partial_box_span `+0.268`, current_partial_coords `+0.266`, current_partial_row_span `+0.261` | pre_prefix_non_image_context `-0.299` |
| source36/object14/y2 | current_partial_box_span `+0.604`, current_partial_coords `+0.601`, current_partial_row_span `+0.601` | pre_prefix_non_image_context `-0.565` |
| source123/object5/y2 | current_partial_coords `+0.515`, current_partial_box_span `+0.515`, current_partial_row_span `+0.509` | pre_prefix_non_image_context `-0.491` |
| source145/object20/y2 | current_partial_box_span `+0.471`, current_partial_coords `+0.470`, current_partial_row_span `+0.469` | pre_prefix_non_image_context `-0.454` |

Focused region deltas:

| state | current_partial_coords | current_partial_box_span | pre_prefix_non_image_context | pre_prefix_image_tokens | same_desc_history_coords | same_desc_history_box_spans |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| source33/object11/y2 | +0.266 | +0.268 | -0.299 | +0.0026 | +0.045 | +0.047 |
| source36/object14/y2 | +0.601 | +0.604 | -0.565 | -0.0002 | -0.008 | -0.011 |
| source123/object5/y2 | +0.515 | +0.515 | -0.491 | +0.0020 | -0.0016 | -0.0021 |
| source145/object20/y2 | +0.470 | +0.471 | -0.454 | +0.0002 | -0.013 | -0.015 |

The shared route signature is strong:

```text
harmful head group: current partial coordinates / current partial box span
repair head group: pre-prefix non-image context
direct image-token mass: near zero in this source-region readout
```

This does not mean the image is irrelevant. It says that, at these y2 emission
states and for these heads, the immediate route distinction is mediated by
textual/object/coordinate prefix regions rather than direct image tokens.

## Value Projection Signature

Max absolute projection in selected focus regions:

| state | current_partial_coords | current_partial_box_span | pre_prefix_non_image_context | same_desc_history_coords | same_desc_history_box_spans |
| --- | ---: | ---: | ---: | ---: | ---: |
| source33/object11/y2 | `-13.13` at L27H3 | `-13.14` at L27H3 | 3.11 | 0.77 | 0.97 |
| source36/object14/y2 | `-50.17` at L27H14 | `-50.21` at L27H14 | 2.08 | 2.23 | 2.28 |
| source123/object5/y2 | `-54.51` at L27H14 | `-54.53` at L27H14 | 2.19 | 0.07 | 0.10 |
| source145/object20/y2 | `+54.53` at L27H15 | `+54.52` at L27H15 | 1.53 | 1.86 | 2.29 |

The value projection table strengthens the route picture:

```text
source36 and source123 share a large negative L27H14 current-prefix/box-span
projection. source145 instead has a large positive L27H15 current-coordinate
projection. source33 is smaller but still dominated by current-prefix/box-span
regions over image-token regions.
```

The sign should be interpreted relative to the specific head direction basis
used by the value-contribution artifact, not as a generic good/bad sign.

## Mechanistic Update

The route bridge makes the ridge-depth taxonomy less isolated:

```text
Coordinate-basin difficulty is not only a local token-surface issue. The tested
heads route strongly through the current partial coordinate/box prefix. Repair
heads shift away from that current-prefix basin toward broader non-image
context. Different ridge categories then appear to differ in which current
prefix value route dominates, not in whether direct image-token attention is
large at the final y2 step.
```

For the main-course hidden-state and attention-mechanism analysis, the next
causal question becomes:

```text
Can we subtract or rotate the current_partial_coords/current_partial_box_span
value contribution for the relevant head group and change the coordinate ridge
winner, without collapsing the schema slot?
```

This is especially promising for:

```text
source36: multi-anchor ridge, large negative L27H14 current-prefix projection
source145: specific-antagonist ridge, large positive L27H15 current-coordinate projection
```

## Verification

```text
python -m pytest tests/analysis/test_route_ridge_bridge.py -q
  6 passed

python -m py_compile src/analysis/autoregressive_binding_template_ablation/route_ridge_bridge.py scripts/analysis/run_autoregressive_binding_route_ridge_bridge.py tests/analysis/test_route_ridge_bridge.py
  exit 0

git diff --check -- src/analysis/autoregressive_binding_template_ablation/route_ridge_bridge.py scripts/analysis/run_autoregressive_binding_route_ridge_bridge.py tests/analysis/test_route_ridge_bridge.py
  exit 0

python scripts/analysis/run_autoregressive_binding_route_ridge_bridge.py \
  --ridge-summary /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/ridge_depth_panel/four_state_y2_onset_late_v1/ridge_depth_panel_summary.json \
  --attention-contrast-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_source_region_attention_head_group_tomography/v2_desc_first_layers20_24_27_harmful_vs_repair_regions13_v2/attention_head_group_tomography_contrast_rows.jsonl \
  --value-contribution-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_head_value_contribution/v2_desc_first_layers20_24_27_harmful_repair_regions_intrinsic_v1/trajectory_boundary_head_value_contribution_rows.jsonl \
  --source-line-idxs 33,36,123,145 \
  --target-next-coord-slot y2 \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/route_ridge_bridge/four_state_y2_harmful_repair_routes_v1
  state_count=4 ridge_rows_used_count=29 attention_rows_used_count=52 value_rows_used_count=1120
```

## Next Step

Run a focused causal value-route intervention:

```text
1. source36/object14/y2: intervene on L27H14 current_partial_coords and
   current_partial_box_span value contribution;
2. source145/object20/y2: intervene on L27H15 current_partial_coords and
   current_partial_box_span value contribution;
3. measure whether target coord rank/top1 changes while coord-token schema
   mass remains stable.
```

This should connect the ridge-depth intervention and the attention/value route
mechanism more directly than another post-hoc surface table.
