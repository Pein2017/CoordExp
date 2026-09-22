# Post-Box Hidden-Delta Descriptor Alignment

Date: 2026-06-21

Worktree: `/data/CoordExp/.worktrees/autoregressive-binding-template-study`

Relevant commits:

- `6c1c60520d9fcada188e832460f4b80e24c2ee54` - `Add post-box hidden delta descriptor alignment metrics`
- `57de67e28d087e0a17d68dc0e59196ba20c72892` - `Record post-box baseline delta causal patch findings`

## Question

The previous natural hidden-delta patch showed that the same-case baseline trajectory can rescue post-box descriptor flips:

```text
paired_post_box_baseline_minus_current = baseline_peer_hidden - active_source_hidden
```

This note asks whether that successful natural correction is mostly the simple descriptor-token output-embedding direction:

```text
embedding(target_next_token) - embedding(post_box_boundary_variant_top_token)
```

If the natural delta were just a naked descriptor-logit shove, the cosine/projection onto this direction should explain most of the rescue. If the natural delta is a richer state repair, the rescue may succeed while only weakly aligning with the descriptor-token vector.

## Implementation And Review

The bridge now emits scalar alignment metrics for realized `paired_post_box_baseline_minus_current` rows:

```text
paired_post_box_desc_direction_status
paired_post_box_desc_direction_target_token_id
paired_post_box_desc_direction_target_token_text
paired_post_box_desc_direction_competitor_token_id
paired_post_box_desc_direction_competitor_token_text
paired_post_box_hidden_delta_desc_target_minus_boundary_variant_top_cosine
paired_post_box_hidden_delta_desc_target_minus_boundary_variant_top_dot
paired_post_box_hidden_delta_desc_target_minus_boundary_variant_top_unit_projection
paired_post_box_hidden_delta_desc_target_minus_boundary_variant_top_direction_norm
```

Non-computable rows emit stable status plus null scalar metrics. No raw vectors are serialized.

Independent checks passed:

- spec compliance review over `57de67e2..6c1c6052`
- code-quality review over `57de67e2..6c1c6052`
- focused test slice: `9 passed`
- full touched test module: `400 passed`

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

Enriched natural hidden-delta alignment site map:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_baseline_minus_current_alignment_site_m4_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_baseline_minus_current_alignment_site_m3_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_baseline_minus_current_alignment_site_m2_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/post_box_baseline_minus_current_alignment_site_m1_v1
```

Each run used:

```text
stage=trajectory-hidden-causal-activation-patch
patch_direction_bases=paired_post_box_baseline_minus_current
patch_strengths=0.25,0.5,0.75,1.0,1.25,1.5
max_rows=7
```

Each run produced:

```text
row_count=51
state_row_count=7
realized_direction_patch_count=30
requested_direction_strength_count=6
```

## Row-Level Evidence

The table reports only realized natural hidden-delta rows and stable controls. `unit_proj` is the scalar projection of the natural hidden delta onto the unit descriptor-token direction. `cos` is therefore also `unit_proj / hidden_norm`.

| site | layer | source line | image | object | variant | role | target | competitor | status | cos | unit_proj | hidden_norm | first top-1 strength | source target prob | p@0.5 | p@0.75 | p@1.0 |
|---|---:|---:|---:|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m4 | 25 | 36 | 3255 | 9 | next_generated_object_box | stable_counterfactual | snow | snow | equals_target |  |  | 18.7063 | 0.25 | 0.514987 | 0.483768 | 0.517271 | 0.486181 |
| m4 | 25 | 36 | 3255 | 9 | previous_object_box | flip | snow | sk | realized | 0.070411 | 4.26383 | 60.5563 | 0.50 | 0.386811 | 0.468286 | 0.477711 | 0.486253 |
| m4 | 25 | 123 | 12670 | 5 | nearest_previous_same_desc_box | flip | back | person | realized | 0.170413 | 30.1893 | 177.1540 | 0.75 | 0.247348 | 0.395542 | 0.563966 | 0.660945 |
| m4 | 25 | 123 | 12670 | 5 | next_generated_object_box | flip | back | person | realized | 0.187034 | 32.9058 | 175.9348 | 0.75 | 0.187867 | 0.395550 | 0.502026 | 0.660857 |
| m4 | 25 | 123 | 12670 | 5 | previous_object_box | stable_counterfactual | back | back | equals_target |  |  | 53.8358 | 0.25 | 0.564050 | 0.617979 | 0.640050 | 0.640081 |
| m3 | 26 | 36 | 3255 | 9 | next_generated_object_box | stable_counterfactual | snow | snow | equals_target |  |  | 22.6836 | 0.25 | 0.514987 | 0.486051 | 0.486080 | 0.517281 |
| m3 | 26 | 36 | 3255 | 9 | previous_object_box | flip | snow | sk | realized | 0.088902 | 6.07350 | 68.3167 | 0.25 | 0.386811 | 0.461013 | 0.474978 | 0.486169 |
| m3 | 26 | 123 | 12670 | 5 | nearest_previous_same_desc_box | flip | back | person | realized | 0.187499 | 37.5054 | 200.0297 | 0.50 | 0.247348 | 0.445742 | 0.564170 | 0.639979 |
| m3 | 26 | 123 | 12670 | 5 | next_generated_object_box | flip | back | person | realized | 0.233590 | 46.2967 | 198.1966 | 0.75 | 0.187867 | 0.395539 | 0.563875 | 0.640094 |
| m3 | 26 | 123 | 12670 | 5 | previous_object_box | stable_counterfactual | back | back | equals_target |  |  | 63.6369 | 0.25 | 0.564050 | 0.587976 | 0.610716 | 0.660824 |
| m2 | 27 | 36 | 3255 | 9 | next_generated_object_box | stable_counterfactual | snow | snow | equals_target |  |  | 25.5554 | 0.25 | 0.514987 | 0.515051 | 0.486195 | 0.517319 |
| m2 | 27 | 36 | 3255 | 9 | previous_object_box | flip | snow | sk | realized | 0.068277 | 5.28157 | 77.3547 | 0.25 | 0.386811 | 0.468253 | 0.477709 | 0.486161 |
| m2 | 27 | 123 | 12670 | 5 | nearest_previous_same_desc_box | flip | back | person | realized | 0.187010 | 39.3212 | 210.2628 | 0.50 | 0.247348 | 0.451238 | 0.502243 | 0.640068 |
| m2 | 27 | 123 | 12670 | 5 | next_generated_object_box | flip | back | person | realized | 0.215017 | 45.7705 | 212.8689 | 0.75 | 0.187867 | 0.399799 | 0.508070 | 0.660822 |
| m2 | 27 | 123 | 12670 | 5 | previous_object_box | stable_counterfactual | back | back | equals_target |  |  | 67.1624 | 0.25 | 0.564050 | 0.588007 | 0.640118 | 0.640099 |
| m1 | 28 | 36 | 3255 | 9 | next_generated_object_box | stable_counterfactual | snow | snow | equals_target |  |  | 30.6003 | 0.25 | 0.514987 | 0.515004 | 0.486084 | 0.486176 |
| m1 | 28 | 36 | 3255 | 9 | previous_object_box | flip | snow | sk | realized | 0.056265 | 5.32715 | 94.6794 | 0.25 | 0.386811 | 0.468312 | 0.477620 | 0.486176 |
| m1 | 28 | 123 | 12670 | 5 | nearest_previous_same_desc_box | flip | back | person | realized | 0.089476 | 20.3678 | 227.6353 | 0.50 | 0.247348 | 0.451117 | 0.532979 | 0.640066 |
| m1 | 28 | 123 | 12670 | 5 | next_generated_object_box | flip | back | person | realized | 0.112243 | 24.8982 | 221.8249 | 0.75 | 0.187867 | 0.399804 | 0.502135 | 0.640066 |
| m1 | 28 | 123 | 12670 | 5 | previous_object_box | stable_counterfactual | back | back | equals_target |  |  | 75.8856 | 0.25 | 0.564050 | 0.640206 | 0.640106 | 0.640066 |

## Main Read

The natural correction is not mostly the simple descriptor-token direction.

For the two clean backpack semantic flips, the natural hidden delta strongly rescues `back` but has only low-to-moderate cosine with the `back - person` output-embedding direction:

| variant | m4 cos | m3 cos | m2 cos | m1 cos |
|---|---:|---:|---:|---:|
| nearest_previous_same_desc_box | 0.1704 | 0.1875 | 0.1870 | 0.0895 |
| next_generated_object_box | 0.1870 | 0.2336 | 0.2150 | 0.1122 |

Projection magnitudes are still nontrivial:

| variant | m4 unit_proj | m3 unit_proj | m2 unit_proj | m1 unit_proj |
|---|---:|---:|---:|---:|
| nearest_previous_same_desc_box | 30.19 | 37.51 | 39.32 | 20.37 |
| next_generated_object_box | 32.91 | 46.30 | 45.77 | 24.90 |

So the natural hidden delta contains a descriptor-aligned component, but most of its norm is elsewhere. That is the key distinction:

- the descriptor-aligned component plausibly supplies the last-mile logit bias;
- the orthogonal majority likely repairs or relocates a broader latent state: object identity, row role, local context, box-conditioned binding, or continuation basin.

This explains why the natural baseline-minus-current delta works at strengths `0.50-0.75`, while the artificial descriptor-vector direction needed much larger gains at early sites. The artificial direction can push the final readout, but the natural direction is shaped like something the remaining decoder layers already know how to propagate.

## Layer Pattern

For backpack, descriptor projection is largest around layers 26-27 and drops at the final site:

```text
nearest_previous_same_desc_box: 30.19 -> 37.51 -> 39.32 -> 20.37
next_generated_object_box:      32.91 -> 46.30 -> 45.77 -> 24.90
```

This suggests the descriptor-readable component is not simply accumulated monotonically into the final state. The final-layer basin may compress or rotate the correction direction while preserving enough decision mass for `back` when the full natural delta is inserted.

The snow/sk control remains a tokenization warning. Its projection magnitudes are small (`4.26-6.07`) and match the much lower artificial descriptor-vector threshold (`4`) from the earlier descriptor patch note. Keep it as a useful control, not as the primary semantic-duplication evidence.

## Mechanistic Hypothesis Update

The current best hypothesis becomes sharper:

1. The target object trace is not absent in the backpack flips.
2. After the completed box, the active row is pulled into a wrong descriptor basin (`person`) by a local post-box binding/context displacement.
3. The same-case baseline row contains a nearby high-norm hidden-state correction vector that restores `back`.
4. That correction vector has a real descriptor-token component, but most of its energy is not on the simple `back - person` axis.
5. Therefore the core mechanism is likely a latent state transition or binding-state repair, not merely a final descriptor logit correction.

In plainer terms: the model seems to know the object, but the autoregressive row state after the box points to the wrong object-binding basin. The baseline hidden delta nudges the whole row state back onto the right manifold; the descriptor token then follows.

## Boundaries

- Scope is selected-row mechanism evidence: 7 states, with 2 real backpack semantic flips, 1 snow/sk tokenization flip, and stable controls.
- This is not a val200 statistical claim.
- Cosines are measured in raw hidden/output-embedding operational space, by design.
- The probe does not yet decompose whether the high-norm non-descriptor component is attention output, MLP output, or a distributed residual effect.
- The probe does not yet identify source tokens or heads responsible for creating the baseline-minus-current displacement.

## Next Implementation Target

Highest-value next bridge: decompose the baseline-minus-current correction inside the final decoder blocks.

Priority order:

1. Split the natural correction into attention-output versus MLP-output contributions at decoder patch indices 24-27, if current hook surfaces allow it.
2. If direct MLP/attention component hooks are unavailable, approximate with per-block residual replacement around layers 25-28 plus existing head/value-region probes.
3. For the backpack rows, compare component patches against the existing descriptor-vector patch:
   - does attention carry object/box binding state while MLP sharpens descriptor logits?
   - does one component reproduce the low-cosine high-norm natural correction?
   - does the `person` basin appear as an attention-routed previous-object anchor or as an MLP lexical attractor?
4. Keep stable rows as null controls and snow/sk as tokenization controls.
