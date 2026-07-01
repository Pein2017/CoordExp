# Post-Box Baseline Hidden-Delta Causal Patch Findings

Date: 2026-06-21

Worktree: `/data/CoordExp/.worktrees/autoregressive-binding-template-study`

Relevant commits:

- `a12e9c3933a5a0a95ea855e3a23aff87fd8c125a` - `Add post-box baseline hidden delta patch basis`
- `e4b48c16a58a804b9ce169f455799da2d591d3dc` - `Cover post-box hidden delta skip failures`
- `52509aa38b012602f388d576cc10971ddd225ca2` - `Harden post-box hidden delta patch basis`

## Probe Purpose

The previous descriptor-vector probe showed that an artificial output-embedding direction such as `back - person` can rescue post-box descriptor flips. This probe asks a more natural question:

```text
Does the same-case baseline trajectory itself contain a hidden-state correction direction that rescues the flip?
```

The new basis is:

```text
paired_post_box_baseline_minus_current
```

For each active post-box row, the patch direction is:

```text
same-case baseline hidden state - active current hidden state
```

At strength `1.0`, the patched source hidden state equals the same-case baseline hidden state at the chosen source layer.

## Inputs

Selected descriptor-boundary rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_box_counterfactual_boundary_selection/v1_desc_first_strict_selected_descriptor_flips_v1/post_box_counterfactual_boundary_selected_rows.jsonl
```

Rows: 7

- Case 1: source line 123, object 5, backpack. Baseline target token `back`, flip top token `person`.
- Case 2: source line 36, object 9, snowboard. Baseline target token `snow`, flip/control token `sk`.

Pair config:

```text
configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
```

## Artifacts

Natural hidden-delta site map:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_baseline_minus_current_site_m4_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_baseline_minus_current_site_m3_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_baseline_minus_current_site_m2_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_baseline_minus_current_site_m1_v1
```

Layer/site mapping:

| site label | source hidden resolved index | decoder patch index |
|---|---:|---:|
| m4 | 25 | 24 |
| m3 | 26 | 25 |
| m2 | 27 | 26 |
| m1 | 28 | 27 |

All four runs used:

```text
stage=trajectory-hidden-causal-activation-patch
patch_direction_bases=paired_post_box_baseline_minus_current
patch_strengths=0.25,0.5,0.75,1.0,1.25,1.5
max_rows=7
```

The post-box-only writer default selected descriptor rows correctly:

```text
target_next_kinds=["desc"]
```

Each run produced:

```text
state_row_count=7
realized_direction_patch_count=30
requested_direction_strength_count=6
skipped_direction_basis_reason_counts={
  "paired_post_box_baseline_minus_current": {
    "paired_post_box_hidden_delta_active_role_baseline": 6
  }
}
```

Baseline rows skip by design; active baseline rows would otherwise produce a zero delta.

## Backpack Flip Rescue

The two real backpack flips are:

| case | variant | target | competitor |
|---|---|---|---|
| 1 | nearest_previous_same_desc_box | back | person |
| 1 | next_generated_object_box | back | person |

Thresholds for the target token becoming top-1:

| site | nearest_previous_same_desc_box | next_generated_object_box |
|---|---:|---:|
| m4 / layer 25 | 0.75 | 0.75 |
| m3 / layer 26 | 0.50 | 0.75 |
| m2 / layer 27 | 0.50 | 0.75 |
| m1 / layer 28 | 0.50 | 0.75 |

At strength `1.0`, both flip rows recover the same-case baseline descriptor probability across all sites:

| site | variant | target prob at strength 1.0 | top token at strength 1.0 |
|---|---|---:|---|
| m4 | nearest_previous_same_desc_box | 0.660945117474 | back |
| m4 | next_generated_object_box | 0.660856962204 | back |
| m3 | nearest_previous_same_desc_box | 0.639979124069 | back |
| m3 | next_generated_object_box | 0.640093743801 | back |
| m2 | nearest_previous_same_desc_box | 0.640067875385 | back |
| m2 | next_generated_object_box | 0.660821795464 | back |
| m1 | nearest_previous_same_desc_box | 0.640066325665 | back |
| m1 | next_generated_object_box | 0.640066325665 | back |

Baseline-minus-current hidden-delta norms for the backpack flips:

| site | nearest_previous_same_desc_box | next_generated_object_box |
|---|---:|---:|
| m4 | 177.153976440430 | 175.934829711914 |
| m3 | 200.029678344727 | 198.196578979492 |
| m2 | 210.262756347656 | 212.868896484375 |
| m1 | 227.635330200195 | 221.824874877930 |

## Controls

Backpack stable counterfactual:

| site | variant | target | competitor | first top-1 strength |
|---|---|---|---|---:|
| m4 | previous_object_box | back | back | 0.25 |
| m3 | previous_object_box | back | back | 0.25 |
| m2 | previous_object_box | back | back | 0.25 |
| m1 | previous_object_box | back | back | 0.25 |

Snowboard tokenization/control rows:

| site | variant | target | competitor | first top-1 strength |
|---|---|---|---|---:|
| m4 | previous_object_box | snow | sk | 0.50 |
| m3 | previous_object_box | snow | sk | 0.25 |
| m2 | previous_object_box | snow | sk | 0.25 |
| m1 | previous_object_box | snow | sk | 0.25 |

The snowboard rows remain useful as tokenization/control evidence, but the primary semantic overwrite evidence is the backpack pair.

## Comparison To Descriptor-Vector Patch

The artificial descriptor-vector patch used:

```text
desc_target_minus_boundary_variant_top
```

For the backpack flips, the artificial output-embedding direction required much larger layer-dependent strengths:

| site | nearest_previous_same_desc_box | next_generated_object_box |
|---|---:|---:|
| m4 | 64 | 64 |
| m3 | 64 | 64 |
| m2 | 28 | 64 |
| m1 | 12 | 16 |

By contrast, the natural baseline-minus-current hidden delta rescued the same rows with:

| site | nearest_previous_same_desc_box | next_generated_object_box |
|---|---:|---:|
| m4 | 0.75 | 0.75 |
| m3 | 0.50 | 0.75 |
| m2 | 0.50 | 0.75 |
| m1 | 0.50 | 0.75 |

The absolute strength scales are not directly comparable because the descriptor-vector direction is unit-normalized while the natural hidden delta is not. The important comparison is qualitative:

- the output-embedding direction proves a final descriptor basin can be rescued;
- the natural hidden delta proves the baseline trajectory contains a much more model-native correction direction;
- the natural correction works even when injected before the final decoder layers, not only at the final readout site.

## Mechanistic Read

Current best interpretation:

1. The model does not simply fail to visually perceive the missing or duplicated object candidate in these rows.
2. The flip row has entered a late descriptor/context basin (`person`) after a completed-box counterfactual, but the same-case baseline row contains a nearby hidden-state direction that restores the correct descriptor (`back`).
3. Strength `1.0` is especially interpretable: it replaces the active row's source hidden state with the same-case baseline hidden state at that layer. Across layer sites m4..m1, this is enough to recover the baseline descriptor probability and top token for both backpack flips.
4. The source-layer thresholds stay low across m4..m1. That suggests the baseline correction direction is compatible with the model's own remaining computation, unlike the artificial output-embedding direction, which needed much larger gain when injected earlier.
5. This supports a more precise hypothesis: duplication/near-duplication descriptor collapse is caused by a post-box hidden-state displacement toward a wrong contextual descriptor basin. The origin is not absence of the object trace, but a fragile binding state that can be redirected by a same-case baseline residual trajectory.

## Boundaries

- Scope is selected-row evidence: 7 states, with 2 real backpack flips and 1 snow/sk control flip.
- Hidden-delta strengths are raw residual-space deltas, not unit directions.
- These probes do not yet localize whether the correction direction is carried by attention output, MLP output, or both.
- These probes do not yet identify which tokens or heads produce the baseline-minus-current displacement.

## Next Promising Step

The highest-value next bridge is component localization inside decoder patch indices 24-27:

- Split the baseline-minus-current correction into attention-output and MLP-output contributions if hook surfaces allow it.
- If not immediately available, approximate by per-layer replacement/patching around the final decoder blocks and compare against attention-head/value-region probes.
- Add a projection comparison between the natural baseline-minus-current hidden delta and the artificial `target - competitor` output-embedding direction to quantify whether the natural correction is mostly descriptor-logit aligned or carries broader state repair.
- Prioritize the backpack case, because it is the cleaner semantic basin flip; keep snowboard as tokenization/control evidence.
