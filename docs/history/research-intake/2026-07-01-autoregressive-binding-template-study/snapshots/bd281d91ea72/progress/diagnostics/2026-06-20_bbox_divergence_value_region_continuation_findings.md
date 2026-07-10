---
doc_id: progress.diagnostics.bbox_divergence_value_region_continuation_findings
layer: progress
doc_type: findings
status: active
domain: mechanistic-diagnosis
summary: GPU-backed value-region continuation probes over the row-specific bbox divergence contrast panel, showing a current-object/current-prefix off-axis value component that repairs local box closure while exposing a duplicate-coordinate loop.
tags:
  - autoregressive-binding-template-ablation
  - value-region-continuation
  - duplication
  - bbox-divergence
  - attention-head-26-10
updated: 2026-06-20
---

# Bbox Divergence Value-Region Continuation Findings

## Scope

This note records GPU-backed continuation probes launched from the
row-specific bbox divergence contrast panel. The input rows are targeted
contrast-panel rows, not a statistically balanced validation slice.

The question was whether the selected source regions at head `26:10` merely
correlate with duplicate onset, or whether changing their value contribution
changes the next object-span trajectory.

## Inputs

Contrast-panel root:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/row_specific_bbox_divergence_contrast_panel/joint_boundary_head26_10_v1
```

Split JSONLs:

- `selected_rows_by_value_source_region/current_object_ref_boundaries.jsonl`
- `selected_rows_by_value_source_region/current_prefix_all.jsonl`
- `selected_rows_by_value_source_region/pre_prefix_non_image_context.jsonl`

Pair config:

```text
configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
```

## Artifacts

Compact first pass, full-vector subtraction, scales `0.0,0.5,1.0`, steps `4`:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/row_specific_bbox_divergence_contrast_panel_value_region_continuation/joint_boundary_head26_10_v1
```

Expanded component pass, scales `0.0,1.0,2.0`, steps `8`:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/row_specific_bbox_divergence_contrast_panel_value_region_continuation/joint_boundary_head26_10_v2_steps8_component_modes
```

One-case long duplicate-completion pass for `desc_first-885-8-0-desc_end`,
steps `12`:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/row_specific_bbox_divergence_contrast_panel_value_region_continuation/joint_boundary_head26_10_case885_long_duplicate_completion_v1
```

## Main Finding

The strongest duplicate-current-open-box case,
`desc_first-885-8-0-desc_end`, has a locally malformed boundary state:

```text
seed prefix tail:
<|object_ref_start|>person<|object_ref_end|><|box_start|>
<|coord_0|><|coord_0|><|coord_47|><|coord_22|>

scale 0.0 next token:
<|object_ref_end|>

target next token:
<|box_end|>
```

Subtracting the head `26:10` value contribution from current-object/current-
prefix regions repairs the immediate boundary decision:

```text
current_object_ref_boundaries, full_vector_subtract, scale 1.0:
  first token: <|box_end|>
  target prob: 0.666193
  target rank: 1

current_prefix_all, full_vector_subtract, scale 1.0:
  first token: <|box_end|>
  target prob: 0.692636
  target rank: 1
```

However, the repair does not eliminate duplication. With a longer continuation,
the model closes the current box, then emits a duplicate object row with the
same description and coordinates, then starts the same row again:

```text
scale 1.0, current_prefix_all, full_vector_subtract, steps 12:
<|box_end|> <|object_ref_start|> person <|object_ref_end|> <|box_start|>
<|coord_0|> <|coord_0|> <|coord_47|> <|coord_22|> <|box_end|>
<|object_ref_start|> person
```

This gives a concrete bridge between local schema repair and autoregressive
duplicate-loop dynamics: the bad head contribution can hide the loop behind an
invalid local boundary token, and removing it reveals the loop rather than
solving it.

## Region Split

Full-vector subtraction at steps `8` showed a region-specific dose response:

```text
current_object_ref_boundaries:
  scale 0.0 first_box_rate: 0.333333
  scale 1.0 first_box_rate: 0.5
  scale 2.0 first_box_rate: 0.5

current_prefix_all:
  scale 0.0 first_box_rate: 0.5
  scale 1.0 first_box_rate: 0.75
  scale 2.0 first_box_rate: 0.75

pre_prefix_non_image_context:
  scale 0.0 first_box_rate: 0.4
  scale 1.0 first_box_rate: 0.4
  scale 2.0 first_box_rate: 0.6
```

The current-object/current-prefix regions are the efficient levers. The
pre-prefix non-image context can move logits but needs stronger scale and is not
the clean source of the local boundary repair.

## Component Split

Projection-only interventions do not repair the `desc_first-885` local boundary:

```text
current_prefix_all, direction_projection_subtract, logit_box_end_minus_object_ref_boundaries_minus_im_end:
  scale 0.0: <|object_ref_end|>, target prob 0.105051, rank 3
  scale 1.0: <|object_ref_end|>, target prob 0.145799, rank 3
  scale 2.0: <|object_ref_end|>, target prob 0.198782, rank 3

current_prefix_all, direction_span_projection_subtract, structural_logit_gradient_span:
  scale 0.0: <|object_ref_end|>, target prob 0.105051, rank 3
  scale 1.0: <|object_ref_start|>, target prob 0.180066, rank 3
  scale 2.0: <|object_ref_start|>, target prob 0.248084, rank 2
```

Residual-only subtraction does repair it:

```text
current_prefix_all, direction_span_residual_subtract, structural_logit_gradient_span:
  scale 1.0: <|box_end|>, target prob 0.423936, rank 1
  scale 2.0: <|box_end|>, target prob 0.865244, rank 1

current_object_ref_boundaries, direction_span_residual_subtract, structural_logit_gradient_span:
  scale 1.0: <|box_end|>, target prob 0.454763, rank 1
  scale 2.0: <|box_end|>, target prob 0.862593, rank 1
```

Interpretation: the problematic current-row contribution is not just a scalar
component along the explicit boundary-logit direction. The causal repair lives
largely in the structural-span residual, which suggests an off-axis value-content
component carried by head `26:10`.

## Source Tokens

For `desc_first-885-8-0-desc_end` under `current_prefix_all`, the dominant
source tokens are the current row itself:

```text
top value-source tokens at scale 1.0 full-vector subtraction:
  <|object_ref_end|> attention 0.197266 contribution_norm 43.7361 projection -7.1701
  <|object_ref_start|> attention 0.209961 contribution_norm 39.8487 projection -2.2276
  <|coord_22|> attention 0.019531 contribution_norm 4.0165 projection -0.3489
  <|box_start|> attention 0.025146 contribution_norm 6.3064 projection 0.1791
  <|coord_0|> attention 0.007874 contribution_norm 1.6624 projection 0.1320
  <|coord_47|> attention 0.000973 contribution_norm 0.2444 projection 0.0443
  <|coord_0|> attention 0.002914 contribution_norm 0.7250 projection 0.0308
  person attention 0.005433 contribution_norm 0.9779 projection -0.0298
```

The harmful value contribution is therefore current-row local, boundary-token
heavy, and coordinate-bearing. It is not explained by a broad pre-prefix context
or a single explicit logit direction.

## Working Mechanistic Picture

The current evidence supports a sharper two-stage picture for this subset:

1. **Current-row self-binding pressure:** At the open-box boundary, head `26:10`
   reads heavily from the current row's own object-ref boundary and coordinate
   tokens. Its off-axis value component pushes the state away from valid
   `<|box_end|>` closure and toward a malformed object-ref boundary token.
2. **Latent autoregressive duplicate basin:** If that local boundary pressure is
   removed, the model can close the box, but the next natural tokens repeat the
   same `person` object and same coordinates. Local repair exposes a duplicate
   loop rather than eliminating the deeper basin.

This separates local schema failure from deeper object-transition failure. The
former is causally addressable at head `26:10`; the latter lives in the
post-closure transition dynamics and likely needs a different probe over
object-ref-start/desc/coordinate recurrence after the repaired closure.

## Verification

GPU scope: targeted panel rows, not full validation.

Successful runs:

- v1 full-vector, 3 regions, steps `4`, scales `0.0,0.5,1.0`.
- v2 full-vector, 3 regions, steps `8`, scales `0.0,1.0,2.0`.
- v2 direction-projection and structural-span projection/residual modes for
  `current_object_ref_boundaries` and `current_prefix_all`.
- one-case `desc_first-885-8-0-desc_end` long continuation, steps `12`.

Two exploratory `direction_span_residual_subtract` launches with the single
`logit_box_end_minus_object_ref_boundaries_minus_im_end` basis failed fast
because that mode requires `structural_logit_gradient_span`; they were replaced
by the successful structural-span runs above.

No training was run.
