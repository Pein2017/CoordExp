# Post-Box Descriptor Causal Patch Findings

Date: 2026-06-21

Worktree: `/data/CoordExp/.worktrees/autoregressive-binding-template-study`

Primary code commit:

- `9e25e24bb8eacef3aa7c5234af238b9002890ecb` - `Add descriptor competitor causal patch basis`

Implementation scope:

- Added `desc_target_minus_boundary_variant_top` as a row-specific `trajectory-hidden-causal-activation-patch` direction basis.
- Direction semantics: `embedding(target_next_token) - embedding(post_box_boundary_variant_top_token_text)`.
- The basis is intentionally not a generic arbitrary-token patch surface. It is tied to the post-box boundary competitor evidence row field.
- Independent spec review passed.
- Independent code-quality review passed.

## Inputs

Selected descriptor-boundary rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_box_counterfactual_boundary_selection/v1_desc_first_strict_selected_descriptor_flips_v1/post_box_counterfactual_boundary_selected_rows.jsonl
```

Rows: 7

- Backpack case, source line 123, object 5: 2 real flips (`back` losing to `person`) plus controls.
- Snowboard case, source line 36, object 9: `snow` versus `sk` tokenization/control case.

Pair config:

```text
configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
```

## Artifacts

Control spot check, first selected baseline row:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_desc_target_minus_boundary_variant_top_spotcheck_v1
```

Flip spot check, first `back` versus `person` row:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_desc_target_minus_boundary_variant_top_flip_spotcheck_v1
```

Final-site descriptor direction sweep, default high strengths:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_desc_target_minus_boundary_variant_top_v1
```

Layer-25-to-final interpolation sanity probe:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_desc_layer25_to_final_smoke_v1
```

Layer-site low-strength maps:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_desc_target_minus_boundary_variant_top_site_m4_low_strength_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_desc_target_minus_boundary_variant_top_site_m3_low_strength_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_desc_target_minus_boundary_variant_top_site_m2_low_strength_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_desc_target_minus_boundary_variant_top_site_m1_low_strength_v1
```

Layer-site high-strength extensions:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_desc_target_minus_boundary_variant_top_site_m4_high_strength_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_desc_target_minus_boundary_variant_top_site_m3_high_strength_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_desc_target_minus_boundary_variant_top_site_m2_high_strength_v1
```

## Live Contract Checks

The first one-row spot check selected a baseline/control row where the boundary-variant top token equaled the target token (`back`). The new basis skipped cleanly with:

```text
desc_target_minus_boundary_variant_top_equals_target
```

The flip-only spot check selected:

```text
source_line_idx=123
object_idx=5
completed_box_variant=nearest_previous_same_desc_box
target_next_token_text=back
post_box_boundary_variant_top_token_text=person
```

It realized the basis and resolved real tokenizer ids:

```text
target back -> token id 1419
competitor person -> token id 8987
```

At strength 4, the patch raised the target probability but did not flip top-1:

```text
baseline top: person
patched top: person
patched target token: back
patched target prob: 0.34471565485
patched target rank: 2
```

This was useful as a scale check: small direction gain moves the basin but is not sufficient at the final patch site.

## Descriptor Vector Patch

Final-site high-strength sweep over all 7 selected rows:

```text
row_count=36
state_row_count=7
realized_direction_patch_count=15
requested_direction_strength_count=5
direction_patch_basis_counts={"desc_target_minus_boundary_variant_top_output_embedding": 15}
```

For the two real backpack flips, strength 32 was already sufficient to flip `person` back to `back`:

| case | variant | baseline top | target | competitor | strength | patched top | target prob | target rank |
|---|---|---|---|---|---:|---|---:|---:|
| 1 | nearest_previous_same_desc_box | person | back | person | 32 | back | 0.802501559258 | 1 |
| 1 | next_generated_object_box | person | back | person | 32 | back | 0.776337265968 | 1 |

The snowboard tokenization/control flip also moved to `snow` with strength 32:

| case | variant | baseline top | target | competitor | strength | patched top | target prob | target rank |
|---|---|---|---|---|---:|---|---:|---:|
| 2 | previous_object_box | sk | snow | sk | 32 | snow | 0.800092399120 | 1 |

Interpretation: the output-embedding descriptor direction is sufficient at the final decoder-layer output. This is a proximal causal intervention, not yet a claim about natural source of the direction.

## Layer-25-To-Final Interpolation

The interpolation probe patched the final decoder-layer output site using the same row's earlier layer-25 state (`alpha=0`) through final state (`alpha=1`).

For the backpack flips:

| variant | alpha 0 top | alpha 0 target prob | alpha 0.25 top | alpha 0.5 top | alpha 1 top |
|---|---|---:|---|---|---|
| nearest_previous_same_desc_box | back | 0.350713372231 | back | person | person |
| next_generated_object_box | back | 0.301256805658 | person | person | person |

For backpack controls, `back` stayed top-1 across the interpolation path.

Interpretation: in the two backpack flips, a recoverable `back` state exists at layer 25 when decoded directly at the final readout site, but the path from layer 25 to the final state shifts the row into the `person` descriptor basin.

For the snowboard case, layer-25 states strongly favored `sk`, while the final row was a `snow` / `sk` tie-like tokenization basin. Treat this as a tokenization/control case rather than the same semantic overwrite as backpack.

## Layer-Site Dose Map

The same `target - competitor` descriptor direction was injected at four source sites. The table reports the smallest strength that made the target token top-1 at the final logits.

Resolved source/site mapping:

| site label | source hidden resolved index | decoder patch index |
|---|---:|---:|
| m4 | 25 | 24 |
| m3 | 26 | 25 |
| m2 | 27 | 26 |
| m1 | 28 | 27 |

Thresholds:

| site | variant | target | competitor | first top-1 strength |
|---|---|---|---|---:|
| m4 | nearest_previous_same_desc_box | back | person | 64 |
| m4 | next_generated_object_box | back | person | 64 |
| m3 | nearest_previous_same_desc_box | back | person | 64 |
| m3 | next_generated_object_box | back | person | 64 |
| m2 | nearest_previous_same_desc_box | back | person | 28 |
| m2 | next_generated_object_box | back | person | 64 |
| m1 | nearest_previous_same_desc_box | back | person | 12 |
| m1 | next_generated_object_box | back | person | 16 |

Snowboard control:

| site | variant | target | competitor | first top-1 strength |
|---|---|---|---|---:|
| m4 | previous_object_box | snow | sk | 4 |
| m3 | previous_object_box | snow | sk | 4 |
| m2 | previous_object_box | snow | sk | 4 |
| m1 | previous_object_box | snow | sk | 4 |

High-strength endpoints for backpack still recover the target at early sites:

| site | variant | target prob at strength 160 | top at strength 160 |
|---|---|---:|---|
| m4 | nearest_previous_same_desc_box | 0.891563713551 | back |
| m4 | next_generated_object_box | 0.881502449512 | back |
| m3 | nearest_previous_same_desc_box | 0.897381961346 | back |
| m3 | next_generated_object_box | 0.875999152660 | back |
| m2 | nearest_previous_same_desc_box | 0.965969860554 | back |
| m2 | next_generated_object_box | 0.962458133698 | back |

Interpretation: this is a graded attenuation or overwrite effect across the final decoder layers, not an absolute impossibility. Earlier-layer injections can survive, but require much larger gain. The final site is much more sensitive.

## Mechanistic Read

Current best explanation from this probe family:

1. The model can still carry a decodable target-descriptor trace before the final state. This is visible because layer-25 direct readout restores `back` in the backpack flip rows.
2. The last few decoder layers transform or overwrite that trace into a competing descriptor basin (`person`) when the completed box context is in the problematic post-box counterfactual configuration.
3. The output-embedding direction `back - person` is causally sufficient to rescue the final descriptor decision, but the needed gain depends strongly on injection layer:
   - final site: threshold 12-16 for the backpack rows in the low-strength map;
   - layer 27 site: threshold 28 or 64;
   - layer 25-26 sites: threshold 64.
4. This points toward late-layer descriptor-basin dynamics, not a simple visual-perception failure. The object trace is not absent; it is fragile and can be overwritten by local post-box/contextual basin attraction.
5. The snow/sk case behaves differently and should remain a tokenization/control warning, not central evidence for semantic duplication.

## Boundaries

Scope:

- Selected rows only: 7 states, with 2 real backpack flips and 1 snow/sk control flip.
- This is mechanism evidence, not validation-set statistics.
- Direction vectors are output-embedding-space interventions; their strength is not yet calibrated to natural residual update magnitudes.
- We have not localized the responsible attention heads or MLP blocks inside the late decoder layers.
- We have not yet tested paired hidden deltas between baseline and flip rows as a naturalistic direction.

## Next Promising Step

The next high-value bridge is to localize the late overwrite within the final decoder layers:

- Build post-box paired baseline-vs-flip hidden-delta directions keyed by `post_box_boundary_case_id` and `post_box_boundary_variant_role`.
- Patch natural baseline/flip residual deltas at sites m4..m1 and compare against the artificial `target - competitor` output-embedding direction.
- Split late-layer intervention by residual stream component if possible: attention output versus MLP output at decoder patch indices 24-27.
- For the backpack case, prioritize the transition from layer 25 target-readable state to layer 28 `person` basin. That is currently the most promising path toward the core mechanism.
