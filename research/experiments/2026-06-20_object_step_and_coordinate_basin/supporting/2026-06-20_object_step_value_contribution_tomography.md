---
doc_id: progress.diagnostics.object_step_value_contribution_tomography
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: val200-desc-first-20state-object-step-head-value-contribution
domain: autoregressive-binding-template-ablation
updated: 2026-06-20
branch: codex/autoregressive-binding-template-study
---

# Object-Step Value-Contribution Tomography

## Scope

This note records a value-vector companion pass for the strict object-step
coordinate panel used in `progress.diagnostics.object_step_head_route_tomography`.
The source panel is:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_panel_selection/v2_desc_first_duplicate_unmatched_slots_selector_strict_v1
```

The goal is to move from "where do harmful/repair heads attend?" to "what do
those attended value slices contribute along the box-end versus object-boundary
readout direction?"

This is a 20-state `desc_first` val200 diagnostic panel. It is readout-only,
not a perturbation result and not a full validation metric.

## Implementation

New reusable reducer:

```text
src/analysis/autoregressive_binding_template_ablation/value_contribution_head_group_tomography.py
tests/analysis/test_value_contribution_head_group_tomography.py
```

The reducer consumes `trajectory_boundary_head_value_contribution_rows.jsonl`,
assigns candidate heads to named groups, aggregates by condition-preserving
object/slot/source-region keys, and writes harmful-vs-repair contrast rows.

Two review-driven guardrails are important:

- high-cardinality identity fields such as `source_line_idx`, `image_id`,
  target token/description/index, bbox, and bbox tokens are grouping keys, not
  loose metadata;
- `value_contribution_status` is summarized inside the bucket rather than used
  as a primary grouping key, so ok/skipped asymmetries remain visible instead
  of becoming unmatchable contrast rows.

Targeted verification:

```text
python -m pytest tests/analysis/test_value_contribution_head_group_tomography.py
# Pytest: 8 passed

python -m py_compile \
  src/analysis/autoregressive_binding_template_ablation/value_contribution_head_group_tomography.py \
  tests/analysis/test_value_contribution_head_group_tomography.py
```

## Raw Value-Contribution Artifact

Model-backed readout:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_head_value_contribution/v2_desc_first_layers20_24_27_harmful_repair_regions_intrinsic_v1
```

Command shape:

```text
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage trajectory-boundary-head-value-contribution \
  --trajectory-boundary-routing-selected-rows .../object_step_panel_selection/v2_desc_first_duplicate_unmatched_slots_selector_strict_v1/selected_rows.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --output-root .../object_step_head_value_contribution/v2_desc_first_layers20_24_27_harmful_repair_regions_intrinsic_v1 \
  --allow-model-load \
  --device cuda:0 \
  --candidate-heads 20:14,20:3,27:14,27:15,27:2,27:3,24:2,24:9 \
  --head-direction-bases logit_box_end_minus_object_ref_boundaries_minus_im_end \
  --value-top-k 8 \
  --families desc_first \
  --intervention-arms natural \
  --target-next-kinds coord \
  --stop-reasons object_step_role
```

Summary:

```text
input/source states: 20
value-contribution rows: 5152
candidate heads: 20:14, 20:3, 27:14, 27:15, 27:2, 27:3, 24:2, 24:9
direction basis: logit_box_end_minus_object_ref_boundaries_minus_im_end
status_counts: ok 5152
model_perturbation_ran: false
training_ran: false
readout_only: true
```

This stage does not accept `--value-source-regions`; it walks the intrinsic
trajectory boundary value regions and region filtering must be post-hoc.

## Head-Group Tomography Artifact

Reducer output:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_head_value_contribution_head_group_tomography/v2_desc_first_layers20_24_27_harmful_vs_repair_intrinsic_regions_v2
```

Head groups:

```text
harmful = 20:14, 20:3, 27:14, 27:15
repair  = 27:2, 27:3, 24:2, 24:9
contrast = harmful - repair
```

Summary:

```text
input_row_count: 5152
aggregate_row_count: 1288
contrast_row_count: 644
unmatched_row_count: 0
raw_head_group_row_counts: harmful 2576, repair 2576
raw_head_group_ok_row_counts: harmful 2576, repair 2576
value_contribution_status_counts: ok 5152
target_next_coord_slot_counts: x1 472, y1 264, x2 272, y2 280
```

`v1` of this reducer artifact is superseded by `v2`, which keeps stricter
identity keys and status handling.

## Group-Level Split

The value pass reproduces the routing split from source-region attention, but
adds magnitude and direction:

```text
harmful current_partial_coords attention_mass: 0.610539
repair  current_partial_coords attention_mass: 0.233999
harmful current_partial_coords contribution_norm: 145.241
repair  current_partial_coords contribution_norm: 51.361

harmful current_partial_box_span attention_mass: 0.545467
repair  current_partial_box_span attention_mass: 0.355514
harmful current_partial_box_span contribution_norm: 136.613
repair  current_partial_box_span contribution_norm: 76.366

harmful pre_prefix_non_image_context attention_mass: 0.313833
repair  pre_prefix_non_image_context attention_mass: 0.557111
harmful pre_prefix_non_image_context contribution_norm: 3.348
repair  pre_prefix_non_image_context contribution_norm: 6.004

harmful pre_prefix_image_tokens attention_mass: 0.002929
repair  pre_prefix_image_tokens attention_mass: 0.001195
harmful pre_prefix_image_tokens contribution_norm: 0.161
repair  pre_prefix_image_tokens contribution_norm: 0.112
```

The late object-step split is therefore not direct image-token access. In this
readout, the candidate heads' direct image-region value contribution is tiny.
The main split is current partial object/box/coord state versus broader
pre-prefix non-image context.

## Slot-Phase Pattern

The largest harmful-minus-repair norm and attention deltas concentrate on
later coordinate slots where partial-coordinate history already exists:

```text
123/5/y2 current_partial_coords norm delta: +152.997
123/5/y2 current_partial_box_span norm delta: +152.945
123/5/y2 current_partial_row_span norm delta: +152.596

36/14/x2 current_partial_coords norm delta: +141.778
36/14/x2 current_partial_box_span norm delta: +139.970

36/14/x2 current_partial_coords attention delta: +0.668137
36/14/y2 current_partial_coords attention delta: +0.601085
```

Directional projection is not a single monotone "harmful is larger" scalar.
Under `logit_box_end_minus_object_ref_boundaries_minus_im_end`, it is
slot-phase dependent:

```text
123/5/x1 current_partial_box_span projection delta: -17.075
123/5/x1 current_prefix_all projection delta:        -16.615

36/14/x2 all_context projection delta:               +12.596
36/14/x2 current_partial_coords projection delta:     +11.487
36/14/y2 all_context projection delta:               +11.564
```

This supports a more precise picture than "duplication heads attend to local
history." Harmful heads do over-route high-norm value through current partial
object/box/coord state, but the direction carried by those values changes with
slot phase. x1 behaves like a box-onset/object-boundary problem; x2/y2 behave
like a later-slot coordinate-state problem.

## Interpretation Boundary

This is readout-only evidence. It does not show that the named heads are
causally sufficient by themselves, and it does not prove visual information is
unused. It says that at these late object-step query positions, for this
strict duplicated/unmatched panel, the harmful-vs-repair split is expressed
primarily through text-prefix object/box/coord state and not through direct
image-token value contribution.

The next strongest bridge is a controlled value-region continuation or patch
that contrasts:

- current partial coordinate/box regions versus pre-prefix non-image context;
- x1 onset states versus x2/y2 later-slot states;
- same object-step rows under the two template/checkpoint variants, if the
  selected rows can be aligned without changing the prompt contract.
