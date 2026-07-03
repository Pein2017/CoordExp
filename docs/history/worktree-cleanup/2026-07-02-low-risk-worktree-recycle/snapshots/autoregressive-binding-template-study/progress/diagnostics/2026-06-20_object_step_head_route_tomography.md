---
doc_id: progress.diagnostics.object_step_head_route_tomography
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: val200-desc-first-20state-object-step-source-region-attention
domain: autoregressive-binding-template-ablation
updated: 2026-06-20
branch: codex/autoregressive-binding-template-study
---

# Object-Step Head Route Tomography

## Scope

This note records the first source-region attention tomography pass over the
strict object-step coordinate panel from:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_panel_selection/v2_desc_first_duplicate_unmatched_slots_selector_strict_v1
```

The goal was to connect the prior value-region continuation result to routing:
do the harmful heads and repair heads read different prefix regions before
they push coordinate basins?

This is a 20-state diagnostic panel from `desc_first` val200. It is not a full
validation metric.

## Implementation

New reusable reducer:

```text
src/analysis/autoregressive_binding_template_ablation/attention_head_group_tomography.py
tests/analysis/test_attention_head_group_tomography.py
```

The reducer consumes `trajectory_source_region_attention_rows.jsonl`, assigns
heads to named groups, aggregates by object/slot/source-region keys, and writes
head-group contrast rows.

Targeted verification:

```text
python -m pytest tests/analysis/test_attention_head_group_tomography.py
# Pytest: 3 passed

python -m py_compile src/analysis/autoregressive_binding_template_ablation/attention_head_group_tomography.py
git diff --check
```

## Attention Artifact

Source-region attention run:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_source_region_attention/v2_desc_first_duplicate_unmatched_slots_layers20_24_27_regions13_v1
```

Command shape:

```text
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage trajectory-source-region-attention \
  --trajectory-boundary-routing-selected-rows .../selected_rows.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --output-root .../object_step_source_region_attention/v2_desc_first_duplicate_unmatched_slots_layers20_24_27_regions13_v1 \
  --allow-model-load --device cuda:0 \
  --hidden-layers 20,24,27 \
  --trajectory-prefix-source state \
  --families desc_first \
  --intervention-arms natural \
  --target-next-kinds coord \
  --stop-reasons object_step_role \
  --value-source-regions current_partial_object_ref_boundaries,current_partial_descriptor,current_partial_box_start,current_partial_coords,current_partial_box_span,current_partial_row_span,same_desc_history_object_ref_boundaries,same_desc_history_coords,same_desc_history_box_ends,same_desc_history_box_spans,previous_box_end,pre_prefix_image_tokens,pre_prefix_non_image_context
```

Summary:

```text
input/source states: 20
attention rows: 12480
ok rows: 11904
skipped rows: 576
layers: 20,24,27
heads per layer: 16
regions: 13
model_perturbation_ran: false
training_ran: false
readout_only: true
```

## Head-Group Tomography Artifact

Reducer output:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_source_region_attention_head_group_tomography/v2_desc_first_layers20_24_27_harmful_vs_repair_regions13_v2
```

Head groups:

```text
harmful = 20:14, 20:3, 27:14, 27:15
repair  = 27:2, 27:3, 24:2, 24:9
contrast = harmful - repair
```

The harmful group is the group that localized destructive x2/y1 coordinate
basin pushes in the value-region continuation sweeps. The repair group is the
group that more often repaired the repeated-person x1 basin.

Summary:

```text
input_row_count: 12480
aggregate_row_count: 520
contrast_row_count: 260
unmatched_row_count: 10400
aggregate_head_group_counts: harmful 260, repair 260
raw_head_group_row_counts: harmful 1040, repair 1040
raw_head_group_ok_row_counts: harmful 992, repair 992
```

`unmatched_row_count` is expected: the reducer keeps only the 8 named heads out
of 48 layer/head combinations.

`v1` was an initial run over the same object-step panel and is superseded by
the `v2` artifact above. `v2` keeps condition/state
discriminators such as `state_key`, `intervention_arm`, `guidance_source`,
`trajectory_step_idx`, and `trajectory_attention_prefix_source` in the
aggregation key and rejects duplicate or overlapping contrasted head specs:

## Group-Level Route Split

Mean normalized attention by group and region:

```text
harmful current_partial_row_span:        0.551241
repair  current_partial_row_span:        0.382871

harmful current_partial_box_span:        0.545533
repair  current_partial_box_span:        0.355394

harmful current_partial_coords:          0.366359
repair  current_partial_coords:          0.140365

harmful same_desc_history_coords:        0.086845
repair  same_desc_history_coords:        0.012564

harmful pre_prefix_non_image_context:    0.313879
repair  pre_prefix_non_image_context:    0.556928

harmful pre_prefix_image_tokens:         0.002930
repair  pre_prefix_image_tokens:         0.001194
```

The strongest split is not visual attention. Both groups put little direct
mass on image tokens in this self-attention readout. The split is local text
state versus broader pre-prefix context:

```text
harmful heads: high current partial row/box/coord attention
repair heads:  higher pre-prefix non-image context attention
```

## Slot-Level Route Split

For slots where prior coordinates exist (`x2`, `y1`, `y2`), harmful heads
route much more strongly through current partial coordinates:

```text
x2 harmful current_partial_coords: 0.801251
x2 repair  current_partial_coords: 0.336113
x2 harmful pre_prefix_non_image:   0.169933
x2 repair  pre_prefix_non_image:   0.584751

y1 harmful current_partial_coords: 0.314309
y1 repair  current_partial_coords: 0.112699
y1 harmful pre_prefix_non_image:   0.293457
y1 repair  pre_prefix_non_image:   0.819565

y2 harmful current_partial_coords: 0.716233
y2 repair  current_partial_coords: 0.253011
y2 harmful pre_prefix_non_image:   0.243015
y2 repair  pre_prefix_non_image:   0.695301
```

For `x1`, no current partial coordinate token exists yet, so the split moves
to current partial row/box span versus broader context:

```text
x1 harmful current_partial_coords: 0
x1 repair  current_partial_coords: 0
x1 harmful pre_prefix_non_image:   0.431496
x1 repair  pre_prefix_non_image:   0.342513
```

This explains why x1 repair should be interpreted separately from later-slot
coordinate basin damage. The x1 state has object/box-start routing but not
coordinate-history routing.

## Object-Slot Contrasts

The largest harmful-minus-repair positive contrasts are exactly the damaged
or duplicate-prone later coordinate slots:

```text
36/14/x2 current_partial_coords: +0.668391
36/14/x2 current_partial_box_span: +0.650723
36/14/x2 current_partial_row_span: +0.610672

36/14/y2 current_partial_coords: +0.601386
36/14/y2 current_partial_box_span: +0.603610
36/14/y2 current_partial_row_span: +0.601035

33/11/x2 current_partial_coords: +0.558940
33/11/x2 current_partial_box_span: +0.560622
33/11/x2 current_partial_row_span: +0.554404

123/5/y2 current_partial_coords: +0.515277
145/20/y2 current_partial_coords: +0.469915
145/20/x2 current_partial_coords: +0.421230
```

The matching negative contrasts are mostly pre-prefix non-image context, where
repair heads attend more:

```text
36/14/x2 pre_prefix_non_image_context: -0.567511
36/14/y2 pre_prefix_non_image_context: -0.564803
33/11/x2 pre_prefix_non_image_context: -0.550052
145/20/y1 pre_prefix_non_image_context: -0.541198
36/14/y1 pre_prefix_non_image_context: -0.534565
123/5/y1 pre_prefix_non_image_context: -0.530039
```

Selected behavior-linked cases:

```text
36/14/x2 destructive snowboard x2:
  current_partial_coords harmful 0.870648 vs repair 0.202257
  pre_prefix_non_image   harmful 0.102563 vs repair 0.670074

123/5/y1 harmful backpack y1 flip:
  current_partial_coords harmful 0.344578 vs repair 0.143182
  pre_prefix_non_image   harmful 0.258787 vs repair 0.788826

145/20/y2 broccoli y2 partial improvement:
  current_partial_coords harmful 0.800997 vs repair 0.331082
  pre_prefix_non_image   harmful 0.164075 vs repair 0.618345

36/6/x1 repeated-person x1 repair case:
  current_partial_coords harmful 0 vs repair 0
  current_partial_row_span harmful 0.287105 vs repair 0.487856
  same_desc_history_coords harmful 0.150764 vs repair 0.024449
  pre_prefix_non_image harmful 0.525588 vs repair 0.385039
```

## Interpretation

The route evidence supports a sharper mechanism than "these heads cause
duplication":

```text
later coordinate slots expose a local coordinate-history basin;
harmful heads read heavily from the current partial coordinate/box span;
their value vectors can then push the next coordinate toward the local basin;
repair heads keep more mass on broader pre-prefix context and are less locked
to the current partial coordinate stream.
```

This aligns with the previous value-region continuation result:

- destructive x2/y2 shifts are A-head/current-partial-coordinate dominated;
- repair of the repeated-person x1 slot is a different regime because no
  prior coordinate exists at the x1 query;
- direct image-token attention is not the dominant visible route in this
  readout, so visual perception failure should not be inferred from low
  image-token mass alone.

The result is consistent with a coordinate-basin selector story:

```text
visual and global context may prepare a set of possible object/coordinate
basins earlier;
at emission time, late heads choose among those basins using local textual
history;
when local coordinate history dominates, the model collapses toward a repeated
or nearby coordinate basin even while staying inside the legal coord-token set.
```

## Boundaries

This is attention mass, not an attention intervention. It is routing evidence
that should be paired with causal value-region interventions.

The source regions overlap. For example, `current_partial_row_span` contains
object-ref, box-start, and coordinate tokens. Do not sum overlapping region
masses as if they partition the context.

The image-token result is last-token self-attention mass at these layers and
heads. It does not prove that visual evidence is unused by the model; it says
the localized late head split is not primarily direct image-token attention at
the measured query.

## Next Directions

High-value next steps:

- run value-contribution tomography for the same A/B heads and regions, so
  attention mass is paired with value-vector direction rather than interpreted
  alone;
- run a counterfactual prefix-guidance bridge for false negatives, separating
  "model cannot perceive object" from "language-side object pointer never
  stabilizes into a decodable coordinate basin";
- add a bidirectional value-region intervention mode for selected heads, to
  test whether adding repair-context value vectors can rescue later-slot basins
  more cleanly than subtracting harmful local-coordinate vectors.
