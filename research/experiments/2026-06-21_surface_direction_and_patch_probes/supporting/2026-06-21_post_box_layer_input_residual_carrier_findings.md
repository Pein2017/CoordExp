# Post-Box Layer-Input Residual Carrier Findings

Date: 2026-06-21

Worktree: `/data/CoordExp/.worktrees/autoregressive-binding-template-study`

Relevant commits:

- `ca647590a1ddfe8041a9990545b329027deda13d` - `Add layer input component patch probe`
- `319fb3f8d933d2ef60c704da07f34857189996e6` - `Record post-box component localization findings`
- `10d542af1dfca7e5139226573328fe7bcde70c8b` - `Record post-box hidden delta descriptor alignment`

## Question

Previous component localization showed that the full natural post-box hidden delta rescues descriptor flips, while the selected block's isolated `self_attn` and `mlp` output deltas do not.

This note tests the missing term:

```text
baseline_peer_layer_input - active_layer_input
```

at the same decoder layer and final prefix token. The goal is to distinguish these hypotheses:

1. The rescue is produced by the selected block's own component outputs.
2. The rescue is already present in the incoming cumulative residual stream.
3. The full hidden-state replacement succeeds only through an inseparable nonlinear combination of input, self-attn, and MLP effects.

## Implementation

The `trajectory-hidden-causal-activation-patch` stage now accepts:

```text
--patch-component-sites layer_input,self_attn,mlp
```

For realized `paired_post_box_baseline_minus_current` rows, `layer_input` adds the baseline-minus-current delta at the decoder-layer forward input using a pre-hook. Output rows are labeled:

```text
patch_space=decoder_component_input:layer_input
component_patch_site=layer_input
component_patch_status=realized
```

The same run can emit `layer_input`, `self_attn`, and `mlp` rows, which keeps artifact contracts aligned across component sites.

Focused verification from the implementation commit:

```text
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -k 'layer_input or component'
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
python -m py_compile src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py
```

Independent code-quality review passed for `319fb3f8..ca647590`.

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
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_layer_input_smoke_m1_flip_v1
```

Four-site layer-input/component sweep:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_layer_input_site_m4_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_layer_input_site_m3_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_layer_input_site_m2_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_layer_input_site_m1_v1
```

Each full sweep produced:

```text
row_count=141
state_row_count=7
realized_direction_patch_count=120
requested_direction_strength_count=6
patch_component_sites=["layer_input","self_attn","mlp"]
```

Strength grid:

```text
0.25,0.5,0.75,1.0,1.25,1.5
```

## Primary Result

`layer_input` reproduces the full hidden-state rescue, while `self_attn` and `mlp` output deltas remain insufficient.

Elementwise comparison against the full hidden patch over matched source/object/variant/strength rows:

| site | component | matched rows | same top token | same top on flip rows | mean abs p(target) diff | max abs p(target) diff |
|---|---|---:|---:|---:|---:|---:|
| m4 | layer_input | 30 | 28/30 | 16/18 | 0.016618 | 0.061851 |
| m4 | self_attn | 30 | 17/30 | 5/18 | 0.176578 | 0.608706 |
| m4 | mlp | 30 | 18/30 | 6/18 | 0.177748 | 0.630410 |
| m3 | layer_input | 30 | 29/30 | 17/18 | 0.013900 | 0.060261 |
| m3 | self_attn | 30 | 20/30 | 8/18 | 0.171198 | 0.582840 |
| m3 | mlp | 30 | 15/30 | 3/18 | 0.176958 | 0.583058 |
| m2 | layer_input | 30 | 30/30 | 18/18 | 0.018799 | 0.061935 |
| m2 | self_attn | 30 | 16/30 | 4/18 | 0.159827 | 0.597990 |
| m2 | mlp | 30 | 15/30 | 3/18 | 0.153946 | 0.550048 |
| m1 | layer_input | 30 | 30/30 | 18/18 | 0.011271 | 0.052199 |
| m1 | self_attn | 30 | 16/30 | 4/18 | 0.172396 | 0.563320 |
| m1 | mlp | 30 | 15/30 | 3/18 | 0.248744 | 0.778082 |

The small nonzero probability differences are expected: the `layer_input` patch is propagated through the selected block, while the full hidden patch is inserted at the layer boundary. The important result is that top-token behavior and rescue thresholds nearly coincide.

## Backpack Flip Thresholds

Backpack case:

```text
source_line_idx=123
image_id=12670
object_idx=5
target=back
competitor=person
```

The table reports the smallest strength making `back` top-1.

| site | variant | full hidden | layer_input | self_attn output | mlp output |
|---|---|---:|---:|---:|---:|
| m4 | nearest_previous_same_desc_box | 0.75 | 0.50 |  |  |
| m4 | next_generated_object_box | 0.75 | 0.75 |  |  |
| m3 | nearest_previous_same_desc_box | 0.50 | 0.50 |  |  |
| m3 | next_generated_object_box | 0.75 | 0.75 |  |  |
| m2 | nearest_previous_same_desc_box | 0.50 | 0.50 |  |  |
| m2 | next_generated_object_box | 0.75 | 0.75 |  |  |
| m1 | nearest_previous_same_desc_box | 0.50 | 0.50 |  |  |
| m1 | next_generated_object_box | 0.75 | 0.75 |  |  |

At strength `1.0`, `layer_input` matches the full hidden rescue probabilities closely:

| site | variant | full hidden p(back) | layer_input p(back) | self_attn p(back) | mlp p(back) |
|---|---|---:|---:|---:|---:|
| m4 | nearest_previous_same_desc_box | 0.660945 | 0.640141 | 0.249621 | 0.244911 |
| m4 | next_generated_object_box | 0.660857 | 0.640138 | 0.190512 | 0.168090 |
| m3 | nearest_previous_same_desc_box | 0.639979 | 0.640165 | 0.247544 | 0.271125 |
| m3 | next_generated_object_box | 0.640094 | 0.640010 | 0.207699 | 0.207581 |
| m2 | nearest_previous_same_desc_box | 0.640068 | 0.640034 | 0.268858 | 0.296904 |
| m2 | next_generated_object_box | 0.660822 | 0.640000 | 0.205948 | 0.209471 |
| m1 | nearest_previous_same_desc_box | 0.640066 | 0.640078 | 0.271242 | 0.057880 |
| m1 | next_generated_object_box | 0.640066 | 0.640118 | 0.189252 | 0.036426 |

The isolated MLP output delta is especially anti-rescue at m1, decreasing `back` as strength increases:

| variant | mlp p(back) at 0.25 | mlp p(back) at 0.50 | mlp p(back) at 1.00 | mlp p(back) at 1.50 |
|---|---:|---:|---:|---:|
| nearest_previous_same_desc_box | 0.207922 | 0.125709 | 0.057880 | 0.022953 |
| next_generated_object_box | 0.139205 | 0.081551 | 0.036426 | 0.011580 |

## Snow/sk Control

Snow case:

```text
source_line_idx=36
image_id=3255
object_idx=9
target=snow
competitor=sk
variant=previous_object_box
```

`layer_input` also tracks the full hidden rescue here, but this remains a tokenization/control case rather than the primary semantic evidence.

| site | full hidden first top-1 | layer_input first top-1 | self_attn first top-1 | mlp first top-1 |
|---|---:|---:|---:|---:|
| m4 | 0.50 | 0.25 |  | 1.00 |
| m3 | 0.25 | 0.50 | 0.50 |  |
| m2 | 0.25 | 0.25 | 0.25 |  |
| m1 | 0.25 | 0.25 | 1.00 |  |

## Mechanistic Read

The post-box descriptor rescue is carried primarily by the cumulative residual stream entering the late decoder block, not by that block's isolated current-step attention output or MLP output.

This is a stronger statement than the previous component note:

1. The model has not lost the object trace after the wrong box-conditioned context.
2. The active row after the completed box is displaced into a wrong local descriptor basin.
3. The same-case baseline row contains a coherent residual-stream correction before the selected block runs.
4. Later computation can read and propagate that correction almost as effectively as direct full-hidden replacement.
5. Isolated within-block output components are too weak, too local, or even directionally opposed.

In plainer terms: the object-binding repair is upstream and state-like. It is not a single attention head or MLP output at the final block saying "emit `back`." The correction looks like a row-state manifold shift that the later decoder stack already knows how to turn into the descriptor token.

## Updated Hypothesis

The causal chain for these selected post-box flips is likely:

```text
box/context perturbation
  -> active post-box residual state moves toward wrong object/descriptor basin
  -> incoming late-block residual stream differs from baseline by a high-norm binding-state vector
  -> adding that vector restores the correct row state
  -> descriptor logits follow, with only a minority of the vector aligned to the simple descriptor-token axis
```

This pushes the next investigation away from "which late block component emits the descriptor" and toward "where, earlier in the trajectory, is the residual binding state formed or displaced?"

## Boundaries

- Scope is selected-row mechanism evidence, not val200 statistics.
- The strongest semantic evidence is still the two backpack flips; snow/sk is a useful control.
- `layer_input` is an additive delta patch, not a full replacement of all cached state.
- The probe localizes the carrier to the incoming residual stream at late blocks; it does not yet identify the upstream source of that residual-stream difference.
- The result does not rule out attention/MLP involvement earlier in the stack or earlier in the object span.

## Next Probe Direction

The highest-value follow-up is no longer another late-component decomposition. It is a formation-time trace:

1. Patch the same baseline-minus-current residual vector at earlier token positions in the object span: descriptor onset, x/y coordinate slots, box close, post-box boundary.
2. Track when the `back`/`person` basin separates under identical decode prefix.
3. Decompose attention at the formation step into source-token routes: image tokens, previous object spans, current box coord tokens, and descriptor tokens.
4. If possible, ablate or patch only the incoming residual segment associated with the current object row to test whether the binding state is row-local or inherited from prior autoregressive history.

This would turn the current result from a carrier localization into an origin map.
