---
doc_id: progress.diagnostics.strict_object_step_projection_residual_factorial_findings
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: val200-desc-first-20state-object-step-projection-residual-factorial
domain: autoregressive-binding-template-ablation
updated: 2026-06-20
branch: codex/autoregressive-binding-template-study
---

# Strict Object-Step Projection/Residual Factorial Findings

## Scope

This note extends the single-row projection/residual smoke in:

```text
progress/diagnostics/2026-06-20_value_region_projection_residual_factorial_probe.md
```

It runs the same `direction_span_projection_residual_grid_subtract` continuation
probe over the full strict 20-state `desc_first` object-step panel:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_panel_selection/v2_desc_first_duplicate_unmatched_slots_selector_strict_v1/selected_rows.jsonl
```

The panel is still a deliberately sharp local diagnostic, not full validation.
It contains four COCO val200 source lines/images, repeated or unmatched object
steps, and all coordinate phases:

```text
x1: 8
y1: 4
x2: 4
y2: 4
```

The main question is whether the previously observed catastrophic full-vector
case is a general projection/residual superposition effect, a residual-only
effect, a route-specific current-state effect, or a one-row anomaly.

## Implementation Artifact

A tracked post-hoc reducer was added for this class of artifacts:

```text
src/analysis/autoregressive_binding_template_ablation/object_step_value_region_factorial_summary.py
scripts/analysis/run_autoregressive_binding_object_step_value_region_factorial_summary.py
tests/analysis/test_object_step_value_region_factorial_summary.py
```

The reducer reads one or more
`trajectory_boundary_head_value_region_continuation_rows.jsonl` files and writes:

```text
factorial_first_step_records.jsonl
factorial_first_step_summary.json
factorial_first_step_summary.md
```

Its primary metric is the realized first emitted coordinate token:

```text
trajectory_greedy_token_text
```

That is intentionally distinct from `intervention_top_token_text`, which is the
patched local top-logit token before the continuation decode surface. The
distinction matters because decode-time surface behavior can differ from the
local top-logit readout, even when the patch scale is zero.

Guardrails:

- `target_coord` is metric-bearing only when derived from
  `target_next_coord_bin` or a four-coordinate `target_bbox`.
- `value_region_scale_pair` is canonicalized from numeric
  `value_region_projection_scale` and `value_region_residual_scale` when both
  are present.
- `value_region_intervention_scale` is not used for grid grouping.
- `current_partial_coords` on `x1` is expected to be non-metric because no
  prior coordinate exists inside the current object span.

## Run Matrix

Both runs used:

```text
stage: trajectory-boundary-head-value-region-continuation
pair config: configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
selected rows: strict 20-state object-step panel above
trajectory_steps: 6
candidate_head_group_mode: combined_all
head_direction_basis: structural_logit_gradient_span
value_region_patch_mode: direction_span_projection_residual_grid_subtract
value_source_regions: current_partial_coords,current_partial_box_span,pre_prefix_non_image_context
projection scales: 0.0,0.5,1.0,1.5,2.0
residual scales: 0.0,0.5,1.0,1.5,2.0
families: desc_first
intervention_arms: natural
target_next_kinds: coord
stop_reasons: object_step_role
```

Head groups:

```text
harmful = 20:14,20:3,27:14,27:15
repair  = 27:2,27:3,24:2,24:9
```

Run roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v4_desc_first_strict_harmful_combined_span_factorial_p012_r012_regions3_steps6_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_continuation/v4_desc_first_strict_repair_combined_span_factorial_p012_r012_regions3_steps6_v1
```

Run counters:

```text
harmful row_count: 7637
repair  row_count: 7625
model_perturbation_ran: true
source_row_count: 20
selected_state_row_count: 20
```

Combined reducer output:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_value_region_factorial_summary/v2_desc_first_strict_harmful_repair_combined_span_factorial_p012_r012_regions3_steps6_v1
```

Reducer counters:

```text
input_row_count: 15262
patched_first_step_row_count: 3000
record_count: 3000
metric-bearing records: 2600
non_ok_patched_first_step_count: 400
```

The 400 non-ok first-step rows are the expected `current_partial_coords` x1
cases across two head groups and 25 projection/residual pairs.

## Main Finding

The panel-level result supports a stronger but narrower version of the smoke
hypothesis:

```text
The harmful group's dangerous route is later-slot current partial coord/box
state. Repair heads and broader pre-prefix non-image context do not reproduce
the catastrophic far-coordinate basin in this panel.
```

Overall first-step metrics:

```text
harmful:
  metric_count 1300
  mean_abs_error 10.748
  mean_delta_error +4.095
  changed_first_token_rate 0.294
  target_match_rate 0.233
  catastrophic_abs_error_ge_100_count 14
  max_abs_error 762

repair:
  metric_count 1300
  mean_abs_error 7.135
  mean_delta_error +0.482
  changed_first_token_rate 0.157
  target_match_rate 0.265
  catastrophic_abs_error_ge_100_count 0
  max_abs_error 61
```

Phase split:

```text
harmful later_x2_y2:
  mean_abs_error 13.328
  mean_delta_error +8.578
  changed_first_token_rate 0.380
  catastrophic_count 14

repair later_x2_y2:
  mean_abs_error 4.815
  mean_delta_error +0.065
  changed_first_token_rate 0.177
  catastrophic_count 0
```

Region split:

```text
harmful current_partial_coords:
  mean_abs_error 13.030
  mean_delta_error +9.030
  changed_first_token_rate 0.453
  catastrophic_count 7

harmful current_partial_box_span:
  mean_abs_error 12.678
  mean_delta_error +5.228
  changed_first_token_rate 0.368
  catastrophic_count 7

harmful pre_prefix_non_image_context:
  mean_abs_error 7.450
  mean_delta_error +0.000
  changed_first_token_rate 0.124
  catastrophic_count 0
```

Repair shows no catastrophic cases in any region.

## Later-Slot Grid Read

For later `x2/y2` states, the harmful current-state routes separate cleanly
from repair and from broad pre-prefix context.

Harmful `current_partial_coords`:

```text
projection=0,residual=0:
  mean_abs_error 3.875
  max_abs_error 11
  catastrophic_count 0

projection=0,residual=2:
  mean_abs_error 26.500
  max_abs_error 107
  catastrophic_count 1

projection=2,residual=0:
  mean_abs_error 5.375
  max_abs_error 11
  catastrophic_count 0

projection=2,residual=1.5:
  mean_abs_error 8.000
  max_abs_error 17
  catastrophic_count 0

projection=2,residual=2:
  mean_abs_error 122.375
  max_abs_error 762
  catastrophic_count 2
```

Harmful `current_partial_box_span`:

```text
projection=0,residual=0:
  mean_abs_error 3.875
  max_abs_error 11
  catastrophic_count 0

projection=0,residual=2:
  mean_abs_error 25.625
  max_abs_error 106
  catastrophic_count 1

projection=2,residual=0:
  mean_abs_error 4.625
  max_abs_error 11
  catastrophic_count 0

projection=2,residual=1.5:
  mean_abs_error 8.750
  max_abs_error 18
  catastrophic_count 0

projection=2,residual=2:
  mean_abs_error 125.375
  max_abs_error 762
  catastrophic_count 2
```

Harmful `pre_prefix_non_image_context` remains near the zero-patch basin:

```text
projection=2,residual=2:
  mean_abs_error 4.125
  max_abs_error 17
  catastrophic_count 0
```

Repair current-state routes stay local:

```text
repair current_partial_coords, projection=2,residual=2:
  mean_abs_error 4.500
  max_abs_error 17
  catastrophic_count 0

repair current_partial_box_span, projection=2,residual=2:
  mean_abs_error 5.500
  max_abs_error 16
  catastrophic_count 0
```

## Threshold Signatures

The source-36/object-14 `y2` case remains the cleanest true component
interaction:

```text
source_line_idx=36, image_id=3255, object_idx=14, y2, target=996

current_partial_coords:
  projection=0,residual=0 -> 999, error 3
  projection=0,residual=2 -> 999, error 3
  projection=2,residual=0 -> 999, error 3
  projection=2,residual=1.5 -> 988, error 8
  projection=1.5,residual=2 -> 234, error 762
  projection=2,residual=2 -> 234, error 762

current_partial_box_span:
  projection=0,residual=0 -> 999, error 3
  projection=0,residual=2 -> 999, error 3
  projection=2,residual=0 -> 999, error 3
  projection=2,residual=1.5 -> 988, error 8
  projection=1.5,residual=2 -> 234, error 762
  projection=2,residual=2 -> 234, error 762
```

This is not residual-only. It requires high residual and enough structural
projection at the same time.

Other harmful `y2` states are different. For source 123 and source 145, high
residual alone already causes a large shift, and projection mainly modulates or
slightly intensifies it:

```text
source_line_idx=123, image_id=12670, object_idx=5, y2, target=270:
  current_partial_coords projection=0,residual=2 -> 377, error 107
  current_partial_coords projection=2,residual=2 -> 377, error 107
  current_partial_box_span projection=0,residual=2 -> 376, error 106
  current_partial_box_span projection=2,residual=2 -> 377, error 107

source_line_idx=145, image_id=15254, object_idx=20, y2, target=146:
  current_partial_coords projection=0,residual=2 -> 210, error 64
  current_partial_coords projection=2,residual=2 -> 210, error 64
  current_partial_box_span projection=0,residual=2 -> 204, error 58
  current_partial_box_span projection=2,residual=2 -> 210, error 64
```

So the mechanism is not a single universal formula. The sharper read is:

```text
High residual damage is the main later-slot vulnerability. In some states, the
residual component is sufficient. In the source-36 y2 basin, residual damage
needs structural projection pressure to cross the far-coordinate threshold.
```

## Decode-Surface Nuance

The reducer uses realized first emitted token for primary metrics:

```text
trajectory_greedy_token_text
```

This can differ from `intervention_top_token_text`. In the harmful run, mismatch
rates by slot/region include:

```text
x1 current_partial_box_span: 18 / 200
x2 current_partial_coords: 14 / 100
y1 current_partial_coords: 19 / 100
y2 current_partial_coords: 15 / 100
y2 pre_prefix_non_image_context: 16 / 100
```

This says the local top-logit surface and realized decode surface are related
but not identical. The distinction is useful, not noise: autoregressive
mechanism claims should name whether they refer to local logits or emitted
next-token behavior.

## Mechanistic Interpretation

Safe claim:

```text
On this strict 20-state desc_first object-step panel, selected harmful heads
can causally move later-coordinate basins through current partial coord/box
value routes. The damaging signal is mostly not broad pre-prefix context, and
it is not reproduced by the selected repair head group.
```

More speculative but now better supported:

```text
The harmful route seems to carry a high-dimensional residual value-state vector
that can destabilize coordinate basins. The structural projection component is
not usually sufficient by itself, but it can act as a threshold gate when the
residual perturbation is already near a basin boundary.
```

In the language of the larger mechanistic picture:

```text
The model is not simply copying a coordinate, nor simply losing visual
perception. At late object steps, it appears to be maintaining an object-local
coordinate state in the text prefix. Certain heads read that state through the
current partial coord/box span. When the harmful group's value contribution is
suppressed in the wrong projection/residual mixture, the local coordinate basin
can jump to a distant attractor. The visual evidence may still be present, but
the autoregressive binding state no longer routes it into the correct coord
slot.
```

## Interpretation Boundary

This is not a full population result. It is a strong local causal diagnostic
over a sharp 20-state val200 panel. The results support deeper mechanism
selection for the next round:

1. Compare local logit-top versus realized greedy-token surfaces directly.
2. Run hidden-state probes around the source-36 `y2` boundary to locate the
   residual amplitude and structural projection threshold before token emission.
3. Repeat the factorial grid on aligned bbox_len12000 checkpoint variants to
   test whether token_embeddings_adapter training changes the threshold,
   residual sensitivity, or decode-surface mismatch.
4. For false negatives, adapt the same current-state versus guidance route:
   test whether adding language-side object guidance repairs missing objects
   without changing visual input, and check whether the hidden state already
   carries object evidence before the missed emission.
