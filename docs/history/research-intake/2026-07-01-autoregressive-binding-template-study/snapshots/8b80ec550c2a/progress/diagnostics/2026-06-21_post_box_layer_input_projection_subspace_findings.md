# Post-Box Layer-Input Projection Subspace Findings

Date: 2026-06-21

Worktree: `/data/CoordExp/.worktrees/autoregressive-binding-template-study`

Relevant commits:

- `dae3aebfa2d1cb9649ebcd7e7a6739ad63c51d22` - `Add component projection patch bases`
- `7bac0f199c5ebf15b90cc2fb14dfb39965c464c6` - `Preserve default projection output schema`
- `3a1b2c9dee620d56eb868150a3890a0723c43aeb` - `Track component projection attempts honestly`
- `4e4e8c7f2fefd82c84772284d91f7fcbacdb19f7` - `Include layer pair in projection summary aggregation`
- `4714efbd` - `Record post-box layer-input residual carrier findings`

## Question

The previous layer-input probe showed that the natural post-box rescue is carried by the incoming residual stream at late decoder blocks:

```text
baseline_peer_layer_input - active_layer_input
```

This note asks what subspace of that natural `layer_input` delta is causal for the next descriptor token:

1. descriptor axis: `target descriptor token - current wrong top descriptor token`
2. structural axis: `<|box_end|> - <|object_ref_start|>`
3. bbox-token axis: mean target bbox coord embeddings minus mean completed-box coord embeddings
4. each axis's orthogonal remainder

The key distinction is whether the rescue is mostly a direct descriptor-logit component or a broader residual-state repair whose descriptor effect appears only after later blocks.

## Implementation And Review

The `trajectory-hidden-causal-activation-patch` stage now accepts:

```text
--patch-component-projection-bases <csv>
```

for realized component patches such as:

```text
--patch-component-sites layer_input
--patch-direction-bases paired_post_box_baseline_minus_current
```

Projection rows use the same component additive patch path as full component rows. The original full component row is still emitted. Projection metadata includes:

```text
component_patch_projection_basis
component_patch_projection_kind
component_patch_projection_along_basis_norm
component_patch_projection_orthogonal_residual_norm
component_patch_projection_patch_delta_norm
component_patch_projection_cosine
component_patch_projection_unit_projection
component_patch_projection_basis_statuses
component_patch_projection_basis_attempt_counts
component_patch_projection_skipped_basis_reason_counts
```

Artifact-trust fixes were applied before running the full probe:

- default runs do not add projection fields to rows, summary, manifest, or internal runtime unless requested;
- mixed realized/skipped outcomes are summarized as `partial`;
- skipped reasons are preserved under `partial`;
- summary aggregation distinguishes layer-pair invocations;
- structural and bbox projection bases have focused realization tests.

Verification:

```text
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -k 'component_projection or projection_basis or projection_bases or omits_default_projection_bases_runtime or layer_pair'
python -m py_compile src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py
```

Result:

```text
10 passed
py_compile passed
```

## Inputs

Selected descriptor-boundary rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_box_counterfactual_boundary_selection/v1_desc_first_strict_selected_descriptor_flips_v1/post_box_counterfactual_boundary_selected_rows.jsonl
```

Rows: 7

Pair config:

```text
configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
```

## Artifacts

Smoke:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_layer_input_projection_smoke_m1_flip_v1
```

Four-site projection sweep:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_layer_input_projection_site_m4_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_layer_input_projection_site_m3_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_layer_input_projection_site_m2_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_layer_input_projection_site_m1_v1
```

Each full sweep produced:

```text
row_count=213
state_row_count=7
realized_direction_patch_count=192
requested_direction_strength_count=6
patch_component_sites=["layer_input"]
```

Projection bases:

```text
desc_target_minus_boundary_variant_top
orthogonal_to_desc_target_minus_boundary_variant_top
box_end_minus_object_ref_start
orthogonal_to_box_end_minus_object_ref_start
target_bbox_tokens_minus_completed_box_tokens
orthogonal_to_target_bbox_tokens_minus_completed_box_tokens
```

Strength grid:

```text
0.25,0.5,0.75,1.0,1.25,1.5
```

## Backpack Thresholds

Backpack case:

```text
source_line_idx=123
image_id=12670
object_idx=5
target=back
competitor=person
```

The table reports the smallest strength that makes `back` top-1. Empty means the patch never made `back` top-1 in the tested strength range.

| site | variant | full hidden | layer_input | desc proj | desc orth | box-end proj | box-end orth | bbox proj | bbox orth |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m4 | nearest_previous_same_desc_box | 0.75 | 0.50 |  | 0.75 |  | 0.50 |  | 0.50 |
| m4 | next_generated_object_box | 0.75 | 0.75 |  | 0.75 |  | 0.75 |  |  |
| m3 | nearest_previous_same_desc_box | 0.50 | 0.50 | 1.25 | 0.75 |  | 0.75 |  | 0.50 |
| m3 | next_generated_object_box | 0.75 | 0.75 |  | 1.00 |  | 0.75 |  |  |
| m2 | nearest_previous_same_desc_box | 0.50 | 0.50 | 1.00 | 1.00 |  | 0.50 |  | 0.50 |
| m2 | next_generated_object_box | 0.75 | 0.75 | 1.25 | 1.50 |  | 0.75 |  |  |
| m1 | nearest_previous_same_desc_box | 0.50 | 0.50 | 0.75 |  |  | 0.50 |  | 0.75 |
| m1 | next_generated_object_box | 0.75 | 0.75 | 1.00 |  |  | 0.75 |  |  |

## Strength 1.0 Split

At strength `1.0`, the descriptor-axis and descriptor-orthogonal pieces trade causal roles across the late stack:

| site | variant | full layer_input p(back) | desc proj p(back) | desc orth p(back) | box-end orth p(back) | bbox orth p(back) | desc along norm | desc orth norm | desc cos |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m4 | nearest_previous_same_desc_box | 0.640 | 0.370 | 0.509 | 0.661 | 0.640 | 29.76 | 164.14 | 0.178 |
| m4 | next_generated_object_box | 0.640 | 0.297 | 0.509 | 0.640 |  | 29.96 | 160.79 | 0.183 |
| m3 | nearest_previous_same_desc_box | 0.640 | 0.429 | 0.447 | 0.640 | 0.640 | 30.19 | 174.56 | 0.170 |
| m3 | next_generated_object_box | 0.640 | 0.376 | 0.441 | 0.640 |  | 32.91 | 172.83 | 0.187 |
| m2 | nearest_previous_same_desc_box | 0.640 | 0.486 | 0.441 | 0.661 | 0.640 | 37.51 | 196.48 | 0.187 |
| m2 | next_generated_object_box | 0.640 | 0.433 | 0.334 | 0.640 |  | 46.30 | 192.71 | 0.234 |
| m1 | nearest_previous_same_desc_box | 0.640 | 0.543 | 0.311 | 0.661 | 0.640 | 39.32 | 206.55 | 0.187 |
| m1 | next_generated_object_box | 0.640 | 0.557 | 0.260 | 0.661 |  | 45.77 | 207.89 | 0.215 |

The blank bbox-orth cells correspond to `next_generated_object_box`, where the completed box is the target/next box. The bbox-difference direction is therefore zero or incompatible, and the projection bases correctly report `partial` at the run level rather than pretending all requested bases realized everywhere.

## Main Finding

The rescue subspace rotates across the late decoder stack.

At m4 and m3, the descriptor-aligned projection is not sufficient for the backpack flips. The descriptor-orthogonal remainder is sufficient:

```text
m4 desc projection: no rescue
m4 desc-orthogonal: rescues both backpack flips at strength 0.75
m3 desc projection: weak or partial
m3 desc-orthogonal: rescues at 0.75 / 1.00
```

At m2, both the descriptor projection and descriptor-orthogonal remainder carry usable rescue signal:

```text
m2 desc projection: rescues at 1.00 / 1.25
m2 desc-orthogonal: rescues at 1.00 / 1.50
```

At m1, the descriptor projection becomes sufficient and the descriptor-orthogonal remainder no longer rescues the backpack flips:

```text
m1 desc projection: rescues at 0.75 / 1.00
m1 desc-orthogonal: no rescue up to 1.50
```

This reconciles the previous two observations:

1. the natural hidden delta has low cosine with the simple descriptor axis, so most of its norm is not descriptor-readout direction;
2. by the final late site, the minority descriptor-axis projection is still causally sufficient for the immediate next descriptor token.

The missing picture is temporal/layerwise conversion, not a contradiction. Earlier late-block residual state contains an object-binding repair that is mostly descriptor-orthogonal. Later layers rotate or compress that repair into a descriptor-readable axis.

## Non-Causal Axes

The structural and bbox axes are not the source of the semantic rescue:

- `box_end_minus_object_ref_start` projection never rescues the backpack flips;
- removing that structural axis leaves the full rescue intact;
- `target_bbox_tokens_minus_completed_box_tokens` projection never rescues;
- removing the bbox axis leaves the full rescue intact where the bbox axis is defined.

This argues against a simple explanation where the successful `layer_input` delta is only correcting box/row boundary syntax or only correcting a coordinate-token basin. Those directions are measurable but not the causal semantic lever for `back` versus `person` in this selected case.

## Mechanistic Read

The current best model is:

```text
post-box wrong basin
  -> early late residual correction is object/binding-state-like and mostly descriptor-orthogonal
  -> subsequent blocks transform that state toward descriptor-readout coordinates
  -> final late input has a small but sufficient descriptor-axis component
  -> next token flips from `person` to `back`
```

In other words, the model's immediate descriptor output is not born as a naked lexical shove. It appears to be prepared as a broader residual state and then rotated into a token-readable direction near the final block. The late descriptor axis is a bottleneck/readout of a prepared object state, not necessarily the origin of the object state.

## Boundaries

- Evidence scope is selected-row causal evidence: two real backpack semantic flips, one snow/sk tokenization/control flip, and stable controls.
- The result is a next-token readout result. It does not yet prove that a patched descriptor-axis continuation produces a coherent full object span.
- The bbox projection is defined only when target and completed boxes differ and the direction is nonzero.
- These projection axes are output-embedding operational axes. They are useful causal bases, not a complete basis for the residual stream.

## Next Probe

The next highest-value test is patched continuation:

1. At m1, keep the descriptor-axis projection and full `layer_input` patch alive for generation of the next object span.
2. Compare whether the first-token rescue becomes a coherent `backpack` object or only a one-token lexical nudge.
3. At m4/m3, compare descriptor-orthogonal patch continuation against descriptor projection continuation.

Predictions:

- If m1 descriptor projection produces `back...` but then fails to form a coherent object, descriptor readout and object-state continuity are separable.
- If m4/m3 descriptor-orthogonal patch yields a coherent object while the descriptor projection does not, the earlier orthogonal state is closer to the true object-binding state.
- If both continuation and termination repair together, the post-box state is a broader row-control manifold rather than a descriptor-only basin.
