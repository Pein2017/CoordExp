# Post-Box Component Localization Findings

Date: 2026-06-21

Worktree: `/data/CoordExp/.worktrees/autoregressive-binding-template-study`

Relevant commits:

- `7aab4316c754f9a8ad254f1b2769e919fb47ba9d` - `Localize post-box hidden delta components`
- `f7c73f9fcba27e849013a95d9d7aa7db9413190c` - `Expose component patch skip metadata`
- `10d542af1dfca7e5139226573328fe7bcde70c8b` - `Record post-box hidden delta descriptor alignment`

## Question

The full natural post-box hidden delta rescues backpack descriptor flips:

```text
baseline_peer_hidden - active_source_hidden
```

This note asks whether that rescue is carried by the selected decoder block's own component outputs:

```text
baseline_peer_self_attn_output - active_self_attn_output
baseline_peer_mlp_output - active_mlp_output
```

Each component delta was added back to the active row at the same decoder layer and final prefix token.

## Implementation And Review

The existing `trajectory-hidden-causal-activation-patch` stage now accepts:

```text
--patch-component-sites self_attn,mlp
```

For realized `paired_post_box_baseline_minus_current` rows, the bridge emits extra component-patch rows with:

```text
patch_space=decoder_component_output:self_attn
patch_space=decoder_component_output:mlp
component_patch_site
component_patch_status
component_patch_delta_norm
component_patch_source_component_norm
component_patch_peer_component_norm
component_patch_requested_sites
component_patch_site_statuses
component_patch_skipped_site_reasons
```

No raw vectors are serialized. Absent `--patch-component-sites` preserves legacy row shape.

Review status:

- spec compliance review passed for `10d542af..7aab4316`
- code-quality review found two artifact-trust issues
- fix review passed for `7aab4316..f7c73f9`
- focused tests passed after the fix, including component skip metadata and realized-count semantics

## Artifacts

Initial live smoke:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_component_smoke_m1_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_component_smoke_m1_flip_v1
```

Four-site component map:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_component_site_m4_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_component_site_m3_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_component_site_m2_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_component_site_m1_v1
```

High-gain m1 component check:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_component_high_strength_m1_flip_v1
```

All four component site-map runs produced:

```text
row_count=111
state_row_count=7
realized_direction_patch_count=90
requested_direction_strength_count=6
patch_component_sites=["self_attn","mlp"]
```

## Backpack Semantic Flip Results

Backpack case:

```text
source_line_idx=123
image_id=12670
object_idx=5
target=back
competitor=person
```

The table reports the smallest strength making `back` top-1. Empty means it never became top-1 in the tested low-strength range `0.25,0.5,0.75,1.0,1.25,1.5`.

| site | layer | variant | full hidden | self_attn output | mlp output |
|---|---:|---|---:|---:|---:|
| m4 | 25 | nearest_previous_same_desc_box | 0.75 |  |  |
| m4 | 25 | next_generated_object_box | 0.75 |  |  |
| m3 | 26 | nearest_previous_same_desc_box | 0.50 |  |  |
| m3 | 26 | next_generated_object_box | 0.75 |  |  |
| m2 | 27 | nearest_previous_same_desc_box | 0.50 |  |  |
| m2 | 27 | next_generated_object_box | 0.75 |  |  |
| m1 | 28 | nearest_previous_same_desc_box | 0.50 |  |  |
| m1 | 28 | next_generated_object_box | 0.75 |  |  |

At strength `1.0`, the full hidden delta consistently rescued `back`, while individual component deltas did not:

| site | variant | full hidden p(back) | self_attn p(back) | mlp p(back) |
|---|---|---:|---:|---:|
| m4 | nearest_previous_same_desc_box | 0.660945 | 0.249621 | 0.244911 |
| m4 | next_generated_object_box | 0.660857 | 0.190512 | 0.168090 |
| m3 | nearest_previous_same_desc_box | 0.639979 | 0.247544 | 0.271125 |
| m3 | next_generated_object_box | 0.640094 | 0.207699 | 0.207581 |
| m2 | nearest_previous_same_desc_box | 0.640068 | 0.268858 | 0.296904 |
| m2 | next_generated_object_box | 0.660822 | 0.205948 | 0.209471 |
| m1 | nearest_previous_same_desc_box | 0.640066 | 0.271242 | 0.057880 |
| m1 | next_generated_object_box | 0.640066 | 0.189252 | 0.036426 |

Mean over the eight backpack flip/site rows:

| patch | mean p(back) at strength 1.0 | mean p(back) at strength 1.5 |
|---|---:|---:|
| full hidden delta | 0.647862 | 0.783225 |
| self_attn output delta | 0.228834 | 0.233599 |
| mlp output delta | 0.186549 | 0.182965 |

Component delta norms were real and nonzero, so this is not a missing-hook artifact. Example m1:

| variant | full hidden norm | self_attn delta norm | mlp delta norm |
|---|---:|---:|---:|
| nearest_previous_same_desc_box | 227.635 | 39.574 | 128.483 |
| next_generated_object_box | 221.825 | 39.602 | 121.904 |

## High-Gain m1 Check

High strengths tested at m1 on the backpack rows:

```text
2,4,8,16
```

Results:

| variant | patch | p(back) at 2 | p(back) at 4 | p(back) at 8 | p(back) at 16 | top at 16 |
|---|---|---:|---:|---:|---:|---|
| nearest_previous_same_desc_box | full hidden | 0.855426 | 0.973199 | 0.995834 | 0.998446 | back |
| nearest_previous_same_desc_box | self_attn | 0.271045 | 0.296232 | 0.322139 | 0.430978 | person |
| nearest_previous_same_desc_box | mlp | 0.010852 | 0.000203 | 0.000000 | 0.000000 | person |
| next_generated_object_box | full hidden | 0.895272 | 0.982482 | 0.997244 | 0.999103 | back |
| next_generated_object_box | self_attn | 0.247328 | 0.273738 | 0.350876 | 0.497294 | back |
| next_generated_object_box | mlp | 0.005178 | 0.000066 | 0.000000 | 0.000000 | person |

This separates the two components:

- `self_attn` carries a weak helpful direction, but it is far too weak to explain the natural rescue at normal scale.
- `mlp` is anti-rescue for the clean backpack semantic flips at m1; increasing its baseline-minus-current output delta suppresses `back`.

## Mechanistic Read

The natural rescue is not localized to the selected block's own `self_attn` or `mlp` output deltas.

The more likely missing term is the incoming/cumulative residual stream:

```text
full layer output delta
  = incoming residual delta
  + self_attn output delta
  + mlp/within-block transformation effects
```

The component probes show that the current block's output components are insufficient on their own. The full hidden-state patch works because it replaces or adds the whole residual state at the layer boundary. That boundary state likely carries a distributed object-binding or row-manifold correction that the later decoder layers can propagate.

This changes the highest-value hypothesis:

1. The model is not failing at the final descriptor token alone.
2. It is also not simply that one final-block attention or MLP component emits the correct descriptor.
3. The key state seems already present in the residual stream entering or crossing the late block boundary.
4. The MLP output delta can actively reinforce the wrong `person` basin when isolated, especially at m1.
5. The full residual replacement likely restores a coherent object-binding manifold; descriptor logits follow from that state.

## Boundaries

- Evidence scope is selected-row mechanism evidence, not val200 statistics.
- Main semantic evidence remains the two backpack flips; snow/sk remains a tokenization/control case.
- Component rows patch one component site at a time, not combined self-attn+MLP.
- The current component sites do not test incoming layer residual directly.
- Generic hidden-state norm fields on component rows are a proxy; component-specific norm fields should be used for component interpretation.

## Next Implementation Target

Add an incoming residual component site, probably named `layer_input` or `block_input`, for the same stage:

```text
baseline_peer_layer_input - active_layer_input
```

Patch it with a decoder-layer forward-pre-hook at the same final token. This should answer whether the successful full hidden delta is carried mainly by the state entering the block rather than by that block's own `self_attn` or `mlp` outputs.

Predictions:

- If `layer_input` alone rescues near the full hidden threshold, then the basin correction is already upstream of the selected block.
- If `layer_input` plus self-attn helps but MLP hurts, the late block may partly preserve and partly overwrite the binding state.
- If `layer_input` also fails, then the full boundary state may require a combined residual/output replacement that individual additive component hooks cannot decompose linearly.
