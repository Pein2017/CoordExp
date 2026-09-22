---
doc_id: progress.diagnostics.adaptive_basin_attention_value_bridge
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: val200-3case-57state-adaptive-basin-readout-and-causal-join
domain: autoregressive-binding-template-ablation
updated: 2026-06-20
branch: codex/autoregressive-binding-template-study
---

# Adaptive Basin Attention/Value Bridge

## Scope

This note records the first post-hoc bridge between adaptive paired-basis
causal-patch outcomes, trajectory-prefix next-token readout, source-region
attention, a small head value-contribution probe, and the adaptive
value-region causal intervention sweep. It does not claim a full validation
result, a training recommendation, or a complete visual grounding mechanism.

## Inputs

Adaptive selected states:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_basis_adaptive_basin_topstates_input/selected_rows.jsonl
```

This input contains 57 state rows from 3 cases and 3 counterfactual variants:
`natural`, `same_desc_history_coords_to_sentinel`, and
`same_desc_history_coords_to_target_bbox`.

Existing paired-basis contrast roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_basis_adaptive_basin_topstates_s20_t28_v1/paired_basis_contrast_analysis
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_basis_adaptive_basin_topstates_s24_t28_v1/paired_basis_contrast_analysis
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/paired_basis_adaptive_basin_topstates_s28_t24_v1/paired_basis_contrast_analysis
```

## New Artifacts

Trajectory-prefix next-token readout:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_next_token_readout/paired_basis_adaptive_basin_topstates_v1
```

Source-region attention, using trajectory prefixes and attention layers
`20,24,27`:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_source_region_attention/paired_basis_adaptive_basin_topstates_layers20_24_27_trajectory_prefix_v1
```

Score-attention join:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_score_attention_join/paired_basis_adaptive_basin_topstates_layers20_24_27_trajectory_prefix_v1
```

Paired-basis outcome to score-attention joins:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_paired_basis_attention_join/paired_basis_adaptive_basin_topstates_s20_t28_layers20_24_27_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_paired_basis_attention_join/paired_basis_adaptive_basin_topstates_s24_t28_layers20_24_27_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_paired_basis_attention_join/paired_basis_adaptive_basin_topstates_s28_t24_layers20_24_27_v1
```

Small head value-contribution probe:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_contribution/paired_basis_adaptive_basin_topstates_heads21_2_26_10_v1
```

Paired-basis outcome to value-contribution joins:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_paired_basis_value_contribution_join/paired_basis_adaptive_basin_topstates_s20_t28_heads21_2_26_10_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_paired_basis_value_contribution_join/paired_basis_adaptive_basin_topstates_s24_t28_heads21_2_26_10_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_paired_basis_value_contribution_join/paired_basis_adaptive_basin_topstates_s28_t24_heads21_2_26_10_v1
```

Adaptive value-region causal interventions:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_intervention/adaptive_basin_head26_10_regions_scales00512_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_intervention/adaptive_basin_head21_2_regions_scales00512_v1
```

Paired-basis outcome to value-region causal intervention joins:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_paired_basis_value_region_intervention_join/adaptive_basin_head26_10_s20_t28_regions_scales00512_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_paired_basis_value_region_intervention_join/adaptive_basin_head26_10_s24_t28_regions_scales00512_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_paired_basis_value_region_intervention_join/adaptive_basin_head26_10_s28_t24_regions_scales00512_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_paired_basis_value_region_intervention_join/adaptive_basin_head21_2_s20_t28_regions_scales00512_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_paired_basis_value_region_intervention_join/adaptive_basin_head21_2_s24_t28_regions_scales00512_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_paired_basis_value_region_intervention_join/adaptive_basin_head21_2_s28_t24_regions_scales00512_v1
```

## Integrity Checks

- Source-region attention initially failed with `attention layer index 28 out of
  range for 28 layers`; the successful attention run uses `20,24,27`, with
  `27` as the final attention-block counterpart to hidden residual layer `28`.
- The score-attention join matched all 57 score rows:
  `matched_score_row_count=57`, `missing_attention_score_row_count=0`.
- Each paired-basis outcome join matched all 304 contrast rows and all matched
  rows also had upstream attention present:
  `matched_row_count=304`, `upstream_attention_matched_row_count=304`,
  `missing_upstream_attention_row_count=0`.
- Each paired-basis outcome to value-contribution join matched all 304 contrast
  rows against the shared 2,280-row value table, with expected fanout:
  `row_count=12160`, `matched_contrast_row_count=304`,
  `matched_join_row_count=12160`,
  `missing_value_contribution_contrast_row_count=0`.
- Each adaptive value-region causal intervention root used all 57 selected
  states and produced `row_count=1824`, with `1776` realized rows and `48`
  skipped `current_coords` rows where that region had no tokens. Both roots
  are model perturbation runs (`model_perturbation_ran=true`) on the
  `token_embeddings_adapter` / `bbox_len12000` pair config.
- Each value-region intervention to paired-basis outcome join preserves the
  full intervention-left surface rather than dropping unmatched states:
  `value_region_intervention_row_count=1824`, `contrast_row_count=304`,
  `row_count=10944`, `matched_join_row_count=9728`,
  `matched_intervention_row_count=608`,
  `missing_contrast_intervention_row_count=1216`, and
  `matched_contrast_row_count=304`. The missing rows are expected because each
  paired-basis contrast root covers the corresponding selected hidden-layer
  subset, while the intervention roots cover all 57 adaptive selected states.

## Readout/Attention Findings

Across the 57 trajectory-prefix readout states:

- `natural`: mean target rank `32.37`, mean coord top1 distance `20.11`, mean
  radius-4 target mass `0.384`.
- `same_desc_history_coords_to_sentinel`: mean target rank `84.53`, mean coord
  top1 distance `65.68`, mean radius-4 target mass `0.072`.
- `same_desc_history_coords_to_target_bbox`: mean target rank `101.11`, mean
  coord top1 distance `67.58`, mean radius-4 target mass `0.265`.

Attention summary over the same joined states:

- Current object-reference attention correlates positively with worse local
  coordinate readout: within-case correlation with target rank `0.635`, and
  within-case correlation with coord top1 distance `0.677`.
- Current box/span and current-coordinate attention are also worse-leaning in
  this adaptive subset.
- Same-desc-history coordinate attention is negatively correlated with worse
  target rank and coordinate distance: within-case correlation with target rank
  `-0.408`, and with coord top1 distance `-0.453`.
- Pre-prefix image-token attention is also negatively correlated with worse
  target rank and coordinate distance in this subset.

Interpretation boundary: this is attention mass over text/prefix/image regions,
not a causal attention intervention. It should be treated as routing evidence.

## Paired-Basis Outcome Bridge

The new post-hoc join shows two regimes.

For source `20 -> target 28`, target-basis harm is widespread but attention
aggregates are less separable:

- target basis worse rate `0.924`
- target catastrophic anchor-drift rate `0.181`
- sentinel catastrophic anchor-drift rate `0.132`
- destination-basin-like rate `0.549`
- target-basis-worse rows have mean current-coordinate attention `0.1015`
  versus `0.1015` for non-worse rows.

For source `24 -> target 28`, harmful rows separate sharply:

- target basis worse rate `0.641`
- destination-basin-like rate `0.401`
- non-worse rows: mean target readout rank `9.22`, coord distance `7.08`,
  current-coordinate attention `0.0507`, same-desc-history coordinate attention
  `0.0498`, image attention `0.0672`.
- worse rows: mean target readout rank `45.31`, coord distance `27.38`,
  current-coordinate attention `0.1410`, same-desc-history coordinate attention
  `0.0101`, image attention `0.0162`.

For source `28 -> target 24`, the same shape holds:

- target basis worse rate `0.668`
- destination-basin-like rate `0.286`
- non-worse rows: mean target readout rank `13.16`, coord distance `9.54`,
  current-coordinate attention `0.0568`, same-desc-history coordinate attention
  `0.0478`, image attention `0.0647`.
- worse rows: mean target readout rank `41.93`, coord distance `25.36`,
  current-coordinate attention `0.1318`, same-desc-history coordinate attention
  `0.0127`, image attention `0.0194`.

This supports a refined basin-selector hypothesis:

```text
pre-existing coordinate basins are already rank-visible ->
middle/late hidden delta changes selector state ->
harmful states over-focus on current local coord/box text while losing
same-desc-history/image support ->
the target-history basis then shifts selection toward a wrong coordinate basin
rather than creating a new coordinate from scratch
```

## Value-Contribution Probe

The small value-contribution probe used heads `21:2` and `26:10` with direction
`logit_box_end_minus_object_ref_boundaries_minus_im_end`.

Head `21:2` is locally constructive for box/coordinate continuation:

- `current_box_span`: attention `0.0390`, projection `+0.498`
- `current_object_ref_boundaries`: attention `0.0108`, projection `+0.077`
- `same_desc_history_box_spans`: attention `0.0086`, projection `+0.079`
- `pre_prefix_non_image_context`: attention `0.917`, projection `+0.643`

Head `26:10` is broadly suppressive along this direction:

- `current_object_ref_boundaries`: attention `0.0086`, projection `-0.639`
- `same_desc_history_object_ref_boundaries`: attention `0.0274`, projection
  `-2.472`
- `same_desc_history_box_spans`: attention `0.0257`, projection `-1.020`
- `pre_prefix_image_tokens`: attention `0.0290`, projection `-1.043`
- `pre_prefix_non_image_context`: attention `0.873`, projection `-11.114`

The value evidence suggests attention mass alone is insufficient. At least in
these two heads, different heads assign opposite-sign value contributions to
neighboring prefix regions. The next probe should therefore join outcome rows to
head-region value contributions, not just attention mass.

## Outcome/Value Join Findings

The outcome/value join sharpens the mechanism picture. All three source/target
layer-pair roots matched fully, so these findings are not driven by join loss.

For source `20 -> target 28`, target-basis harm is nearly ubiquitous
(`0.924` target-worse rate), so harmful/non-harmful separation is less
diagnostic. Still, target-worse rows show a more negative `26:10` value
signature than non-worse rows:

- `26:10 all_context`: projection `-15.961 -> -17.203`
- `26:10 pre_prefix_non_image_context`: projection `-10.727 -> -11.630`
- `26:10 same_desc_history_object_ref_boundaries`: projection
  `-2.284 -> -2.731`

For source `24 -> target 28`, harmful rows separate strongly:

- target-basis worse rate `0.641`
- target-specific destination-basin-like rate `0.401`
- `26:10 all_context`: projection `-5.113 -> -23.814`
- `26:10 pre_prefix_non_image_context`: projection `-3.542 -> -16.044`
- `26:10 current_prefix_all`: projection `-1.259 -> -6.745`
- `26:10 same_desc_history_object_ref_boundaries`: projection
  `-0.535 -> -3.906`

For source `28 -> target 24`, the same signature appears:

- target-basis worse rate `0.668`
- target-specific destination-basin-like rate `0.286`
- `26:10 all_context`: projection `-6.856 -> -22.210`
- `26:10 pre_prefix_non_image_context`: projection `-4.732 -> -14.959`
- `26:10 current_prefix_all`: projection `-1.762 -> -6.279`
- `26:10 same_desc_history_object_ref_boundaries`: projection
  `-0.898 -> -3.592`

The destination-basin-like slice is cleaner than the broad target-worse slice.
For `24 -> 28`, basin-like rows shift `26:10 all_context` from `-13.992` to
`-21.759`; for `28 -> 24`, from `-14.143` to `-24.507`. The same rows also
increase the negative contribution from current prefix aggregate and same-desc
object-reference regions.

Catastrophic anchor drift has a different signature. In all three layer-pair
roots, rows with target-basis catastrophic anchor drift have less negative
`26:10` all-context and pre-prefix projections than non-drift rows, while
`current_prefix_other` becomes more negative. This suggests anchor drift is not
the same failure mode as the destination-basin-like duplication basin. It may be
a context-shift or boundary-displacement failure that bypasses the same
pre-prefix/same-desc suppressive channel.

## Adaptive Value-Region Causal Intervention

I promoted the `26:10` value signature from readout to a causal next-token
intervention over the same 57 adaptive states. The intervention subtracts
`scale * region_contribution` at the final prefix token before
`self_attn.o_proj`. It is not a full rollout repair test; every row is still a
single next-token perturbation at a coordinate query.

The causal sign matches the value-decomposition sign for the boundary axis.
Suppressing negative `26:10` contributions increases the
`box_end - object_ref_boundaries` margin:

```text
26:10, scale 1.0 / 2.0
pre_prefix_non_image_context             +0.624 / +1.295
current_prefix_all                       +0.325 / +0.677
same_desc_history_object_ref_boundaries  +0.155 / +0.337
current_object_ref_boundaries            +0.043 / +0.082
pre_prefix_image_tokens                  +0.046 / +0.093
```

The sign-control head `21:2` is much weaker and often opposite:

```text
21:2, scale 1.0 / 2.0
pre_prefix_non_image_context             -0.047 / -0.114
current_prefix_all                       -0.031 / -0.055
same_desc_history_object_ref_boundaries  +0.012 / +0.012
current_object_ref_boundaries            +0.004 / +0.002
pre_prefix_image_tokens                  +0.022 / +0.003
```

This establishes a causal boundary-axis role for `26:10`, but with a sharp
caveat: these selected states are coordinate-next states, not boundary-next
states. Coordinate target probability shifts are tiny on average
(`~1e-4` to `~1e-3`), and top-token changes are coordinate-to-coordinate
swaps rather than object-ref/box/termination transitions. Therefore the
intervention supports "26:10 controls the boundary/continuation margin under
these contexts"; it does not by itself prove that `26:10` directly selects the
coordinate basin.

The durable intervention-to-outcome join confirms the aggregate causal
separation between `26:10` and the sign-control head. For the intervention
table aggregate, repeated across the three paired-basis contrast joins per
head, `26:10` has mean `box_end - object_ref_boundaries` logit-margin delta
`+0.142`, while `21:2` is slightly negative at `-0.008`. The coordinate target
probability deltas remain tiny (`26:10`: `+6.2e-5`; `21:2`: `+2.1e-4`), so this
still points at a boundary/continuation margin rather than a direct coordinate
selector.

The most reliable basin-aligned amplification appears in the `24 -> 28`
paired-basis contrast, where destination-basin-like rows have stronger `26:10`
boundary-margin shifts than non-basin rows:

```text
26:10, 24 -> 28, false -> true destination-basin-like,
box-vs-boundary margin delta at scale 2.0

pre_prefix_non_image_context             +1.280 -> +1.794
current_prefix_all                       +0.671 -> +0.946
same_desc_history_object_ref_boundaries  +0.365 -> +0.532
current_object_ref_boundaries            +0.075 -> +0.115
```

The `21:2` control does not reproduce this pattern: the same `24 -> 28`
pre-prefix non-image scale-2 slice stays weakly negative (`-0.122 -> -0.112`).
The durable join also corrects the earlier ad hoc read: `20 -> 28` and
`28 -> 24` do not show a clean destination-basin true/false separation in these
intervention means, even though the aggregate `26:10` sign remains positive.
The cautious mechanism claim is therefore: `26:10` is a causal
boundary-gating/continuation-axis head, and that gate is most clearly
basin-aligned in the middle-to-late `24 -> 28` contrast. It may be an internal
state marker or gate coupled to basin selection, rather than the coordinate
selector itself.

## Current Mechanism Update

The adaptive-basin evidence weakens a pure "more same-desc history causes
duplication" story. In the sharper middle/late layer pairs, harmful target-basis
rows are marked by high current-coordinate/box attention together with low
same-desc-history coordinate and image attention, plus a strong suppressive
`26:10` value signature from pre-prefix context, current-prefix aggregate, and
same-desc object/box history regions.

The causal intervention refines this further: `26:10` is not merely a readout
artifact. Removing its negative region contributions causally shifts the
boundary/continuation logit margin, with the clearest destination-basin
amplification in the `24 -> 28` paired-basis contrast. However, because the
same perturbation barely moves the coordinate target distribution at
coordinate-next positions, the current best mechanism is:

```text
fragile local coordinate basin selection co-occurs with a late boundary-gating
state ->
26:10 carries a suppressive boundary/continuation signal from pre-prefix and
same-desc/current prefix regions ->
hidden-state paired-basis deltas push the model into states where this gate is
more strongly engaged ->
visible duplication/termination failures emerge when coordinate selection,
object-span continuation, and boundary closure become mis-synchronized
```

This is a deeper, less one-dimensional picture than "same-desc history causes
duplication": coordinate-basin selection and boundary-gating appear coupled but
not identical.

## Next Experiments

- Run a short continuation probe on the adaptive selected states for `26:10`
  regions where the boundary margin moves most strongly, especially
  `pre_prefix_non_image_context`, `current_prefix_all`, and
  `same_desc_history_object_ref_boundaries`. The key question is whether the
  boundary-axis shift later changes object-span closure, duplication, or
  premature termination across 2-6 generated tokens.
- Add a coordinate-direction causal probe, because the current value-region
  intervention uses a boundary/termination direction and barely moves
  coordinate target probabilities at coordinate-next states.
- Add more heads/directions around the `26:10` suppressive channel and `21:2`
  constructive channel before overgeneralizing from two heads.
- Split attention/value summaries by target coordinate slot (`x1`, `y1`, `x2`,
  `y2`) and by top destination basin, not just target-worse flags.
- Run one narrow visual-region alignment pass over the same 57 states to test
  whether low image-token attention corresponds to weak target object evidence
  or merely a text-side selector bottleneck.
- Keep dynamically deepening any path that changes the final mechanism picture:
  if a region/head pair cleanly separates basin-like harm from non-harm, promote
  it to causal intervention rather than exhausting the full planned grid first.

## Adaptive Natural Continuation Probe

This follow-up tested whether the single-token `26:10` boundary-axis effect
survives into short autoregressive continuations on the sharper `24 -> 28`
adaptive natural states. Before interpreting results, the continuation artifact
contract was fixed so that selected states without `state_key` no longer collide
under the coarse fallback identity. The coordinate-copy analyzer now also
copies `source_state_key`, `state_key`, `adaptive_subset_label`, and
`counterfactual_variant`, and both continuation and coordinate-copy summaries
include adaptive-label groupings.

Input:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/adaptive_basin_s24_t28_natural_input_v1/selected_rows.jsonl
```

Scope:

```text
19 natural selected states
13 s24_t28_natural_destination_basin
6  s24_t28_natural_non_basin_control
pair config: configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
target_next_kinds: coord
trajectory_steps: 8
decode: deterministic greedy continuation after patched first step
```

Continuation roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/adaptive_basin_s24_t28_natural_scale0_baseline_steps8_v2
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/adaptive_basin_s24_t28_natural_head26_10_regions_objctx_scale12_steps8_v2
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/adaptive_basin_s24_t28_natural_head21_2_regions_objctx_scale12_steps8_v2
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/adaptive_basin_s24_t28_natural_head26_10_image_region_scale12_steps8_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/adaptive_basin_s24_t28_natural_head21_2_image_region_scale12_steps8_v1
```

Post-hoc coordinate-copy roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_value_region_continuation_coordinate_copy/adaptive_basin_s24_t28_natural_scale0_baseline_steps8_v2
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_value_region_continuation_coordinate_copy/adaptive_basin_s24_t28_natural_head26_10_regions_objctx_scale12_steps8_v2
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_value_region_continuation_coordinate_copy/adaptive_basin_s24_t28_natural_head21_2_regions_objctx_scale12_steps8_v2
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_value_region_continuation_coordinate_copy/adaptive_basin_s24_t28_natural_head26_10_image_region_scale12_steps8_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_value_region_continuation_coordinate_copy/adaptive_basin_s24_t28_natural_head21_2_image_region_scale12_steps8_v1
```

Baseline phenotype, scale `0`, head `26:10`, `current_prefix_all`:

```text
coordinate_copy_class_counts: incomplete 15, target_like 4
destination-basin rows:       incomplete 9,  target_like 4
non-basin controls:           incomplete 6,  target_like 0
first_step_box_end_rate:      0.0
final_stop_reason_counts:     im_end 1, max_steps 18
```

The intervention arms preserved this coarse phenotype. For both `26:10` and
the sign-control `21:2`, every object/context region/scale arm has the same
per-arm class distribution as baseline: `15` incomplete and `4` target-like
continuations. The image-token arms also reproduce the same per-arm pattern.
No arm caused first-step `<|box_end|>` emission; all arms retained the same
`1/19` final `im_end` and `18/19` max-step profile.

The single-token boundary-axis signal is still present, especially for `26:10`:

```text
mean first-step box_end - object_ref_boundaries logit delta

26:10 object/context aggregate: +0.442
26:10 destination-basin rows:   +0.626
26:10 non-basin controls:       +0.043

26:10 strongest arm:
pre_prefix_non_image_context scale 2.0: +1.280

21:2 object/context aggregate: -0.034
21:2 image-token aggregate:    +0.010
```

But this logit-margin movement does not become a multi-token repair at these
coordinate-next states. The observable sequence changes are local coordinate
swaps, not changed object-span control:

```text
non-basin control, target_next <|coord_524|>:
baseline: <|coord_519|> <|coord_691|> <|coord_717|> <|box_end|> ...
patched:  <|coord_520|> <|coord_691|> <|coord_716|> <|box_end|> ...

destination-basin row, target_next <|coord_94|>:
baseline: <|coord_47|> <|coord_22|> <|box_end|> <|object_ref_start|> ...
patched:  <|coord_33|> <|coord_22|> <|box_end|> <|object_ref_start|> ...
```

Thus the bridge result is a useful negative for the strong hypothesis "`26:10`
boundary-gate intervention alone repairs short rollout behavior from
coordinate-next states." It is a positive result for a split mechanism:

```text
coordinate basin selection can be nudged locally by value-route perturbations,
often by one coordinate bin, but object-span closure/transition remains a
separate autoregressive attractor that is not moved by the same boundary-axis
head intervention.
```

This also sharpens the basin/control interpretation. The selected
destination-basin states already contain the target-like continuation branch
under baseline (`4/13`), while non-basin controls do not (`0/6`). The tested
value-region interventions mostly select among nearby already-available
coordinate bins; they do not create a missing object continuation and do not
resynchronize coordinate selection with termination. The next higher-value
direction should therefore test coordinate-selector directions and
boundary-next positions separately, rather than treating the boundary-margin
head as the whole duplication mechanism.

Next branch:

- Build a coordinate-selector intervention orthogonal to the boundary gate:
  target coordinate logit minus wrong-anchor logit, slot-wise target radius
  mass, or paired-basis target-vs-sentinel coordinate direction.
- Run boundary-next continuation states, not only coordinate-next states, to
  ask whether `26:10` controls closure when the model is actually at an object
  boundary decision.
- Keep the visual-region thread, but interpret the current image-token result
  as weak: image-token value perturbation mostly mirrors the same tiny
  coordinate-bin nudges and does not by itself reveal a target-object
  perception-to-emission bridge.

## Boundary-Next Continuation Probe

The coordinate-next continuation probe showed that `26:10` can move the
single-token boundary margin without repairing object-span continuation. The
next test moved to actual boundary-next states: selected prefixes whose target
next token is `<|box_end|>`. This uses the existing joint boundary seed set,
not the adaptive coordinate-basin selected rows.

Input:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_routing_selection/joint_boundary_probe_seed_v1/trajectory_boundary_routing_selected_rows.jsonl
```

Scope:

```text
12 selected boundary-next states
target_next_kind: box_end
causal buckets: clean 4, failure_default128_rescued 4, failure_joint128_only 4
onset labels: neutral 5, next_step_duplicate_onset 2, next_step_unmatched_onset 5
pair config: configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
trajectory_steps: 8
decode: deterministic greedy continuation after patched first step
```

Continuation roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/joint_boundary_boxend_scale0_baseline_steps8_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/joint_boundary_boxend_head26_10_regions_all_scale12_steps8_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/joint_boundary_boxend_head21_2_regions_all_scale12_steps8_v1
```

The baseline exposes the actual boundary failure:

```text
baseline, current_prefix_all scale 0

clean:                         4/4 first-token <|box_end|>
failure_default128_rescued:    0/4 first-token <|box_end|>
failure_joint128_only:         0/4 first-token <|box_end|>

failure_default128_rescued first tokens:
  <|im_end|> 1, <|object_ref_start|> 2, <|object_ref_end|> 1

failure_joint128_only first tokens:
  <|object_ref_start|> 4
```

`26:10` reproduces the boundary-margin asymmetry at the correct decision
surface and has a selective behavioral effect:

```text
26:10, failure_default128_rescued, next_step_duplicate_onset

current_object_ref_boundaries scale 1.0: 1/1 flips to <|box_end|>
current_object_ref_boundaries scale 2.0: 1/1 flips to <|box_end|>
current_prefix_all scale 1.0:              1/1 flips to <|box_end|>
current_prefix_all scale 2.0:              1/1 flips to <|box_end|>
pre_prefix_non_image_context scale 2.0:    1/1 flips to <|box_end|>

example:
baseline: <|object_ref_end|>
patched:  <|box_end|> <|object_ref_start|> person <|object_ref_end|>
          <|box_start|> <|coord_0|> <|coord_0|> <|coord_47|>
```

The sign-control head `21:2` does not reproduce this. On the same rescued
duplicate-onset row, it keeps `<|object_ref_end|>` as the first token for
current-object/current-prefix/pre-prefix non-image regions, with negative
box-end-vs-boundary margin deltas.

The joint-only unmatched rows behave differently. Strong `26:10` current
regions make the boundary margin very positive, but the model goes to global
termination rather than box closure:

```text
26:10, failure_joint128_only, next_step_unmatched_onset

current_object_ref_boundaries scale 1.0: <|object_ref_start|> 4/4
current_object_ref_boundaries scale 2.0: <|im_end|> 4/4
current_prefix_all scale 1.0:             <|object_ref_start|> 4/4
current_prefix_all scale 2.0:             <|im_end|> 4/4

mean box_end - object_ref_boundaries logit delta:
current_object_ref_boundaries scale 2.0: +7.188
current_prefix_all scale 2.0:             +7.719
```

This is the clearest split so far:

```text
duplicate-onset rescued failures:
  boundary gate can be pushed back to local box closure by 26:10.

unmatched joint-only failures:
  suppressing object-ref continuation exposes im_end/global termination,
  not box closure.
```

Mechanism update:

`26:10` is a real boundary/termination competition head, but the competition is
not binary `box_end` vs `object_ref_start`. It is at least three-way:

```text
local box closure <|box_end|>
object-span continuation/restart <|object_ref_start|> or <|object_ref_end|>
global termination <|im_end|>
```

The failure family determines which alternative basin is exposed when the
object-ref boundary is suppressed. Rescued duplicate-onset failures can be
pushed into local closure, while joint-only unmatched failures fall into
premature image termination. This helps explain why the coordinate-next probe
did not repair rollout behavior: a boundary-margin intervention can select the
wrong boundary alternative if the underlying object/basin evidence is not
ready for closure.

Next branch:

- Separate boundary directions into explicit pairwise axes:
  `box_end - object_ref_start`, `box_end - object_ref_end`,
  `box_end - im_end`, and `object_ref_boundaries - im_end`.
- For joint-only unmatched rows, test whether adding a coordinate-selector or
  visual-target evidence direction before the boundary intervention changes the
  exposed basin from `<|im_end|>` to `<|box_end|>`.
- For rescued duplicate-onset rows, run a short post-closure duplicate check:
  the first-token repair closes the current box, but the following object span
  still starts another `person` with coordinates near the old duplicate basin.

## Pairwise Direction Component Probe

The next branch separated two meanings that had been conflated in the first
value-region probe:

```text
full_vector_subtract:
  patch = -scale * source_region_contribution

direction_projection_subtract:
  patch = -scale * dot(source_region_contribution, unit_direction) * unit_direction
```

The original value-region intervention used `head_direction_basis` only as a
readout/projection axis. It always subtracted the full source-region
contribution vector. That is useful, but it cannot answer whether the causal
effect is carried by one local pairwise boundary direction.

Implementation update:

```text
--value-region-patch-mode full_vector_subtract                 # default
--value-region-patch-mode direction_projection_subtract
```

Verification:

```text
python -m py_compile src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
passed

python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q -k "value_region_continuation or value_region_intervention"
21 passed initially; after paired-basis grouping/report coverage updates:
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q -k "value_region_continuation or value_region_intervention or paired_basis_value_region_intervention_join"
22 passed

python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q
298 passed

git diff --check -- src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
passed

subagent spec review for direction_projection_subtract:
passed

subagent code-quality review after fixes:
important paired-basis grouping risk addressed by grouping value-region
intervention joins by value_source_region, value_region_intervention_scale,
head_direction_basis, and value_region_patch_mode.
```

Full-vector pairwise-labeled controls:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/joint_boundary_pairwise_head26_10_box_end_minus_object_ref_start_regions_curr_obj_prectx_scales12_steps8_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/joint_boundary_pairwise_head26_10_box_end_minus_object_ref_end_regions_curr_obj_prectx_scales12_steps8_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/joint_boundary_pairwise_head26_10_box_end_minus_im_end_regions_curr_obj_prectx_scales12_steps8_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/joint_boundary_pairwise_head26_10_object_ref_boundaries_minus_im_end_regions_curr_obj_prectx_scales12_steps8_v1
```

These four artifacts are intentionally diagnostic controls. They all produced
the same first-token behavior because the causal patch was the same full region
vector. Only the projection/readout basis changed.

Directional-component artifacts:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/joint_boundary_directionproj_head26_10_box_end_minus_object_ref_start_regions_curr_obj_prectx_scales12_steps8_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/joint_boundary_directionproj_head26_10_box_end_minus_object_ref_end_regions_curr_obj_prectx_scales12_steps8_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/joint_boundary_directionproj_head26_10_box_end_minus_im_end_regions_curr_obj_prectx_scales12_steps8_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/joint_boundary_directionproj_head26_10_object_ref_boundaries_minus_im_end_regions_curr_obj_prectx_scales12_steps8_v1
```

Scope:

```text
input rows: same 12 boundary-next states
head: 26:10
regions: current_prefix_all, current_object_ref_boundaries, pre_prefix_non_image_context
scales: 1.0, 2.0
steps: 8 greedy continuation steps
patch mode: direction_projection_subtract
```

Aggregate result:

```text
baseline, current_prefix_all scale 0:
  first <|box_end|>: 4/12
  first <|im_end|>:  1/12

direction_projection_subtract, all three regions and scales:
  box_end - object_ref_start:
    first <|box_end|>: 24/72
    first <|im_end|>:  6/72
    mean first-step box_end - object_ref_start delta: +0.544
    mean first-step box_end - im_end delta:            +0.210

  box_end - object_ref_end:
    first <|box_end|>: 24/72
    first <|im_end|>:  6/72
    mean first-step box_end - object_ref_end delta: +1.487
    mean first-step box_end - im_end delta:         +0.041

  box_end - im_end:
    first <|box_end|>: 24/72
    first <|im_end|>:  1/72
    mean first-step box_end - im_end delta: +0.095

  object_ref_boundaries - im_end:
    first <|box_end|>: 24/72
    first <|im_end|>:  6/72
    mean first-step object_ref_boundaries - im_end delta: -0.921
```

The directional patches were not null: first-step patch norms were nonzero
across all four bases.

```text
box_end - object_ref_start:      mean patch norm 10.72, max 54.57
box_end - object_ref_end:        mean patch norm 21.31, max 55.78
box_end - im_end:                mean patch norm  4.47, max 16.96
object_ref_boundaries - im_end:  mean patch norm 14.37, max 43.77
```

Per-row basin comparison at `current_prefix_all / scale 2.0`:

```text
duplicate-onset rescued row:
  baseline:    <|object_ref_end|>
  full-vector: <|box_end|>
  directional:
    box_end - object_ref_start:     <|object_ref_end|>
    box_end - object_ref_end:       <|object_ref_start|>
    box_end - im_end:               <|object_ref_end|>
    object_ref_boundaries - im_end: <|object_ref_end|>

joint-only unmatched rows:
  baseline:    <|object_ref_start|>
  full-vector: <|im_end|>
  directional:
    all four tested axes: <|object_ref_start|>
```

Interpretation:

The 26:10 value-region causal effect is not carried by a single local
logit-gradient axis such as `box_end - object_ref_start`, `box_end -
object_ref_end`, or `box_end - im_end`. Those one-dimensional components can
move the intended margins slightly, and `box_end - im_end` suppresses premature
`<|im_end|>` in the aggregate, but they do not reproduce either of the two
important full-vector basin transitions:

```text
rescued duplicate-onset:
  object_ref_end -> box_end

joint-only unmatched:
  object_ref_start -> im_end
```

This makes the full 26:10 source-region value vector look more like a
multi-dimensional boundary-state edit than a scalar "turn up box_end" knob. The
directional axes are useful diagnostic coordinates, but the behavioral basin
shift appears to require a richer component of the attention value contribution:
possibly a coupled template-state feature that jointly alters object-boundary,
termination, and post-closure continuation readiness.

Next branch:

- Decompose the residual full-vector effect after removing tested pairwise
  logit-gradient components. If the residual still reproduces the basin shifts,
  the causal vector is outside the local structural-token gradient subspace.
- Add a multi-axis projected patch using the span of
  `{box_end - object_ref_start, box_end - object_ref_end, box_end - im_end,
  object_ref_boundaries - im_end}`. Compare span-only vs orthogonal-residual
  patches against the full-vector patch.
- For duplicate-onset rescued rows, trace which top value tokens in 26:10 carry
  the residual component. Candidate sources are repeated descriptor/context
  tokens rather than image tokens, because image-token suppression remains
  near-null.

## Structural-Span vs Residual Causal Probe

I implemented and ran an explicit structural-gradient span decomposition for
the boundary-next states. The aim was to test whether the earlier full-vector
`26:10` source-region effect is explained by local structural-token readout
directions, or by the orthogonal residual of the value contribution.

New implementation surface:

```text
head_direction_basis=structural_logit_gradient_span
value_region_patch_mode=direction_span_projection_subtract
value_region_patch_mode=direction_span_residual_subtract
```

The span is built from four local `26:10` logit-gradient directions:

```text
box_end - object_ref_start
box_end - object_ref_end
box_end - im_end
object_ref_boundaries - im_end
```

Those component gradients are orthonormalized. The projection mode subtracts
only the source-region contribution inside that span. The residual mode
subtracts the orthogonal complement. The implementation also records span-rank
health fields and makes continuation IDs include `value_region_patch_mode`, so
projection and residual artifacts can be safely unioned post-hoc.

Inputs:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_routing_selection/joint_boundary_probe_seed_v1/trajectory_boundary_routing_selected_rows.jsonl
```

Shared scope:

```text
12 boundary-next selected states
head: 26:10
head_direction_basis: structural_logit_gradient_span
regions: current_object_ref_boundaries,current_prefix_all,pre_prefix_non_image_context
scales: 1.0,2.0
steps: 8
pair config: configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
checkpoint surface: token_embeddings_adapter / bbox_len12000
```

Corrected continuation roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/joint_boundary_structural_spanproj_head26_10_regions_curr_obj_prectx_scales12_steps8_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/joint_boundary_structural_residual_head26_10_regions_curr_obj_prectx_scales12_steps8_v1
```

Artifact integrity:

```text
span projection:
  continuation_count: 72
  value_region_patch_mode: direction_span_projection_subtract
  head_direction_span_rank_counts: {'4': 240}
  head_direction_span_component_count_counts: {'4': 240}
  head_direction_span_skipped_component_basis_counts: {}

orthogonal residual:
  continuation_count: 72
  value_region_patch_mode: direction_span_residual_subtract
  head_direction_span_rank_counts: {'4': 296}
  head_direction_span_component_count_counts: {'4': 296}
  head_direction_span_skipped_component_basis_counts: {}

continuation_id mode check:
  all first-step IDs include the patch mode
  span/residual continuation_id intersection: 0
```

Aggregate comparison over the shared three regions and scales:

```text
baseline:
  first_step_box_end_rate: 4/12 = 0.333
  first_step_im_end_rate:  1/12 = 0.083

full vector, same regions/scales:
  first_step_box_end: 29/72
  first_step_im_end:  18/72
  mean box-vs-object-boundary delta: +2.897

structural span projection:
  first_step_box_end: 24/72
  first_step_im_end:  6/72
  mean box-vs-object-boundary delta: +1.271
  mean box-vs-im_end delta:          +0.013
  mean span projection norm: 19.08
  mean residual norm:        46.68
  mean projection fraction:  0.389

orthogonal residual:
  first_step_box_end: 32/72
  first_step_im_end:  6/72
  mean box-vs-object-boundary delta: +1.022
  mean box-vs-im_end delta:          -0.069
  mean span projection norm: 19.08
  mean residual norm:        46.68
  mean projection fraction:  0.389
```

The bucket split is the key result:

```text
failure_default128_rescued:
  baseline:              0/4 box_end, 1/4 im_end
  full vector:           5/24 box_end, 10/24 im_end
  structural span:       0/24 box_end, 6/24 im_end
  orthogonal residual:   8/24 box_end, 6/24 im_end

failure_joint128_only:
  baseline:              0/4 box_end, 0/4 im_end
  full vector:           0/24 box_end, 8/24 im_end
  structural span:       0/24 box_end, 0/24 im_end
  orthogonal residual:   0/24 box_end, 0/24 im_end

next_step_duplicate_onset:
  baseline repeated over arms: 6/12 box_end
  full vector:                11/12 box_end
  structural span:             6/12 box_end
  orthogonal residual:        10/12 box_end

next_step_unmatched_onset:
  baseline repeated over arms: 6/30 box_end
  full vector:                 6/30 box_end, 8/30 im_end
  structural span:             6/30 box_end, 0/30 im_end
  orthogonal residual:         6/30 box_end, 0/30 im_end
```

This is the strongest bridge result so far. The structural-token gradient span
explains a visible part of the logit-margin movement, but not the behaviorally
selective duplicate-onset rescue. The rescue follows the orthogonal residual:
the failed duplicate-onset row `desc_first-885-8-0-desc_end` changes from
baseline `<|object_ref_end|>` to `<|box_end|>` under residual subtraction from
`current_object_ref_boundaries` and `current_prefix_all`, at both scales. The
same residual perturbation does not create box closure for the joint-only
unmatched rows, and it avoids the full-vector run's extra `im_end` failures on
those rows.

Interpretation:

```text
the causal 26:10 boundary rescue is not mainly a scalar local boundary-token
readout direction ->
the structural-gradient span is a visible readout/control surface but is not
the behavior-carrying state ->
the orthogonal residual carries a higher-dimensional object-span transition
state that can restore box closure for duplicate-onset rows without turning
unmatched rows into termination
```

This materially changes the mechanism picture. `26:10` is still a boundary
gate, but the meaningful gate vector is not reducible to "increase box_end over
object_ref/im_end" in local logit space. It looks more like a latent
serialization/object-span state vector whose downstream readout includes
boundary-token logits as only one shadow.

Next branch:

- Factor the residual itself: SVD/PCA over residual contributions by
  row/region, then patch top residual PCs against the same 12 boundary states.
- Run random residual-subspace controls with matched norm to check whether the
  rescue is residual-specific or just large-norm non-structural perturbation.
- Compare residual PC directions to special-token embeddings and unembedding
  directions for `<|box_end|>`, `<|object_ref_*|>`, `<|im_end|>`, and
  coordinate tokens to localize whether the residual is format-state,
  termination-state, coordinate-basin, or object-identity state.
- Extend only the most promising residual PCs to the adaptive coordinate-basin
  states and a small train/val split before claiming generality.

## Residual PCA/SVD Readout

Implemented a readout-only residual factorization stage:

```text
stage: trajectory-boundary-head-residual-pca
rows: 12 selected boundary-next states
head: 26:10
regions: current_object_ref_boundaries,current_prefix_all,pre_prefix_non_image_context
components requested: 8
model perturbation: false
training: false
```

Artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_residual_pca/joint_boundary_head26_10_regions_curr_obj_prectx_v1
```

Outputs:

```text
trajectory_boundary_head_residual_pca_rows.jsonl
trajectory_boundary_head_residual_pca_components.jsonl
trajectory_boundary_head_residual_pca_summary.json
trajectory_boundary_head_residual_pca.md
```

Integrity:

```text
row_count: 36
realized_vector_count: 36
status_counts: {'ok': 36}
head_direction_span_rank_counts: {'4': 36}
```

Row-weighted uncentered SVD over residual vectors:

```text
singular_values:
  [302.966, 76.074, 49.231, 45.739, 31.504, 27.277, 22.998, 19.485]

explained_energy_fraction:
  [0.87065, 0.05489, 0.02299, 0.01984, 0.00941, 0.00706, 0.00502, 0.00360]

cumulative:
  [0.87065, 0.92554, 0.94853, 0.96838, 0.97779, 0.98485, 0.98987, 0.99347]
```

PC1 dominates all three source regions:

```text
current_object_ref_boundaries:
  mean residual norm: 48.60
  top_abs_pc_counts: {'0': 12}

current_prefix_all:
  mean residual norm: 58.80
  top_abs_pc_counts: {'0': 12}

pre_prefix_non_image_context:
  mean residual norm: 32.63
  top_abs_pc_counts: {'0': 10, '1': 1, '7': 1}
```

The largest PC1 scores come from the `failure_joint128_only` state
`desc_first-1268-10-0-desc_end`, especially `current_prefix_all` and
`current_object_ref_boundaries`. The `failure_default128_rescued` rows also
live strongly on PC1, while clean rows have much smaller object-boundary
residuals. This means PC1 is not simply the previous causal-rescue direction;
it looks like a broad high-norm residual attractor shared by severe local
boundary failures. Smaller PCs separate source-region and case-specific
structure, for example PC1/PC2 mixture in `pre_prefix_non_image_context` and
PC2/PC3 mass on `desc_first-285-1-0-desc_end`.

Sampling caveat and deweighted check:

```text
row-weighted vectors: 36
unique model-input/region vectors: 27
duplicate multiplicity counts: {1: 18, 2: 9}
```

Some selected rows differ by intervention-arm metadata but present identical
model input prefixes and targets. Deweighting each identical model-input/region
vector to one mean vector gives:

```text
unique-input singular_values:
  [236.202, 65.617, 46.934, 35.480, 28.859, 24.371, 20.193, 16.140]

unique-input explained_energy_fraction:
  [0.84259, 0.06503, 0.03327, 0.01901, 0.01258, 0.00897, 0.00616, 0.00393]
```

More aggressive case/region deweighting gives a similar spectrum:

```text
case-region explained_energy_fraction:
  [0.84608, 0.06625, 0.03465, 0.01538, 0.01328, 0.00936, 0.00666, 0.00267]
```

So PC1 concentration is not just duplicate-row weighting. It is a stable
dominant residual direction, but the artifact should still be interpreted as a
boundary-probe cohort factorization, not a population PCA.

Current interpretation:

```text
structural span:
  local boundary-token readout/control shadow

orthogonal residual:
  behavior-associated candidate object-span transition correlate

residual PC1:
  broad high-norm residual attractor shared by severe boundary failures;
  promising but too broad to call the duplicate-rescue direction directly

residual PCs 2-4:
  likely where source-region and case-specific rescue/termination distinctions
  begin; these should be patched separately rather than only patching PC1
```

Next causal step:

- Patch row-specific residual PC projections, not the whole residual, using the
  same continuation protocol.
- Use at least PC1, PCs 1-2, PCs 1-4, and PC2-only/PC3-only ablations.
- Add matched-norm random residual-subspace controls.
- Evaluate separately for duplicate-onset rescue, unmatched-onset termination,
  and clean-state disruption.

## Residual PC Continuation Causal Probe

Implemented stage:

```text
trajectory-boundary-head-residual-pc-continuation
```

Artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_residual_pc_continuation/joint_boundary_head26_10_pc012_01_0123_regions_curr_obj_prectx_scales12_steps8_v1
```

Inputs:

```text
selected rows:
  /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_routing_selection/joint_boundary_probe_seed_v1/trajectory_boundary_routing_selected_rows.jsonl

residual PCA components:
  /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_residual_pca/joint_boundary_head26_10_regions_curr_obj_prectx_v1/trajectory_boundary_head_residual_pca_components.jsonl

pair config:
  configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
```

Scope:

```text
candidate head: 26:10
states: 12
value source regions:
  current_object_ref_boundaries
  current_prefix_all
  pre_prefix_non_image_context
scales: 1.0, 2.0
residual PC groups: 0, 1, 2, 0,1, 0,1,2,3
trajectory steps: 8
row_count: 1298
continuation_count: 360
```

Here `residual_pc_group=0` is the first SVD component, corresponding to the
previous section's conventional "PC1" wording.

Overall first-step token counts:

```text
<|box_end|>: 134 / 360
<|im_end|>: 45 / 360
<|object_ref_start|>: 157 / 360
<|object_ref_end|>: 24 / 360
```

First-step behavior by residual PC group:

```text
group 0:
  n=72, box=30, im=6, object_start=32, object_end=4
  mean delta box_end_minus_object_ref_boundaries: +1.2483
  mean delta box_end_minus_im_end: -0.0512

group 1:
  n=72, box=24, im=3, object_start=39, object_end=6
  mean delta box_end_minus_object_ref_boundaries: -0.0230
  mean delta box_end_minus_im_end: +0.0035

group 2:
  n=72, box=24, im=4, object_start=38, object_end=6
  mean delta box_end_minus_object_ref_boundaries: +0.0547
  mean delta box_end_minus_im_end: -0.0174

group 0,1:
  n=72, box=30, im=14, object_start=24, object_end=4
  mean delta box_end_minus_object_ref_boundaries: +1.2721
  mean delta box_end_minus_im_end: -0.0408

group 0,1,2,3:
  n=72, box=26, im=18, object_start=24, object_end=4
  mean delta box_end_minus_object_ref_boundaries: +1.6033
  mean delta box_end_minus_im_end: -0.1606
```

Targeted failure splits:

```text
failure_default128_rescued / next_step_duplicate_onset:
  group 0:       n=6, box=2, im=0, object_end=4
  group 1:       n=6, box=0, im=0, object_end=6
  group 2:       n=6, box=0, im=0, object_end=6
  group 0,1:     n=6, box=2, im=0, object_end=4
  group 0,1,2,3: n=6, box=2, im=0, object_end=4

failure_default128_rescued / neutral:
  group 0:       n=18, box=4, im=6, object_start=8
  group 1:       n=18, box=0, im=3, object_start=15
  group 2:       n=18, box=0, im=4, object_start=14
  group 0,1:     n=18, box=4, im=6, object_start=8
  group 0,1,2,3: n=18, box=0, im=10, object_start=8

failure_joint128_only / next_step_unmatched_onset:
  group 0:       n=24, box=0, im=0, object_start=24
  group 1:       n=24, box=0, im=0, object_start=24
  group 2:       n=24, box=0, im=0, object_start=24
  group 0,1:     n=24, box=0, im=8, object_start=16
  group 0,1,2,3: n=24, box=0, im=8, object_start=16
```

Clean rows remain stable in this probe: both clean duplicate-onset and clean
neutral slices emit `<|box_end|>` for every first-step row across all residual
PC groups. That makes the failure slices the interpretable part of this run.

Interpretation:

- The first residual component, group `0`, carries nearly all effective
  boundary-margin movement in this global PCA basis.
- Groups `1` and `2` alone are mostly inert for first-step behavior and have
  near-zero mean boundary-margin deltas.
- Adding later components to group `0` does not rescue the hard
  `failure_joint128_only / next_step_unmatched_onset` states into
  `<|box_end|>`. Instead, broader groups increasingly open a termination
  failure mode via `<|im_end|>`, especially at scale 2.0 on
  `current_object_ref_boundaries` and `current_prefix_all`.
- For `failure_default128_rescued / next_step_duplicate_onset`, group `0`
  gives a small selective rescue: 2/6 first-step rows switch to `<|box_end|>`,
  specifically at scale 2.0 for the current-object and current-prefix source
  regions. The non-image pre-prefix source region stays on
  `<|object_ref_end|>`.
- Therefore the global residual PCA is not a one-axis box-closure mechanism.
  It captures a broad high-norm boundary attractor that can move the
  `<|box_end|>` margin, but the stronger whole-residual rescue observed in the
  previous continuation probe is likely row-conditioned and local to the
  current object-span state.

Mechanistic consequence:

```text
structural span:
  local boundary-token readout/control shadow

global residual PC0:
  broad boundary-margin attractor; partly causal, not sufficient

later global PCs:
  weak alone; when bundled with PC0, can push fragile rows toward premature
  image termination rather than object closure

row-specific residual:
  still the stronger candidate mechanism for object-span transition and
  duplicate-basin escape
```

Next steps:

- Add matched-norm random-subspace controls for the residual-PC continuation
  stage before calling the PC0 effect specific.
- Add row-conditioned residual-basis probes: leave-one-case-out PCA,
  per-bucket PCA, and direct row residual with norm-matched random controls.
- Split the hard unmatched-onset states into language-guidance versus visual
  non-perception cases by prefix guidance/intervention, because global PC
  movement increases boundary margin without producing closure there.

Verification:

```text
python -m py_compile src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q -k "residual_pc"
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q
```

## Matched Random Off-Span Residual Controls

Implemented patch mode:

```text
direction_span_random_residual_subtract
```

Purpose:

```text
Test whether the structural-residual rescue is specific to the actual
row/region residual direction, or whether any matched-norm vector in the
orthogonal complement of the structural logit-gradient span can cause the same
object-closure behavior.
```

Control construction:

- Use the same `structural_logit_gradient_span` basis as the earlier
  span-vs-residual probe.
- For each selected row, source region, and scale, compute the actual
  off-span residual norm.
- Sample deterministic random vectors in the orthogonal complement of that
  structural span.
- Normalize each random vector to the actual residual norm.
- Subtract `scale * random_residual_control`.

Code surface:

```text
value_region_patch_mode=direction_span_random_residual_subtract
--value-region-random-controls
--value-region-random-seed
```

Run scope:

```text
12 boundary-next selected states
head: 26:10
head_direction_basis: structural_logit_gradient_span
regions:
  current_object_ref_boundaries
  current_prefix_all
  pre_prefix_non_image_context
scales: 1.0, 2.0
steps: 8
random controls: 4 seeds x 4 controls = 16 controls per row/region/scale
GPUs: 0,1,2,3 in parallel
pair config: configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
```

Random-control roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/joint_boundary_structural_random_residual_head26_10_regions_curr_obj_prectx_scales12_steps8_seed101_controls4_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/joint_boundary_structural_random_residual_head26_10_regions_curr_obj_prectx_scales12_steps8_seed202_controls4_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/joint_boundary_structural_random_residual_head26_10_regions_curr_obj_prectx_scales12_steps8_seed303_controls4_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/joint_boundary_structural_random_residual_head26_10_regions_curr_obj_prectx_scales12_steps8_seed404_controls4_v1
```

Union artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_continuation/joint_boundary_structural_random_residual_union_head26_10_regions_curr_obj_prectx_scales12_steps8_seeds101_202_303_404_v1
```

Artifact integrity:

```text
per seed:
  row_count: 960
  first_step_count: 288

union:
  row_count: 3840
  first_step_count: 1152
  continuation_count: 1152
```

Overall comparison:

```text
random off-span controls:
  first_step_box_end: 384/1152 = 0.3333
  first_step_im_end:   87/1152 = 0.0755
  mean box-vs-object-boundary delta: +0.0196
  mean box-vs-im_end delta:          +0.0174

true structural residual:
  first_step_box_end: 32/72 = 0.4444
  first_step_im_end:   6/72 = 0.0833
  mean box-vs-object-boundary delta: +1.0221
  mean box-vs-im_end delta:          -0.0686

structural span projection:
  first_step_box_end: 24/72 = 0.3333
  first_step_im_end:   6/72 = 0.0833
  mean box-vs-object-boundary delta: +1.2708
  mean box-vs-im_end delta:          +0.0130
```

Key failure split:

```text
failure_default128_rescued / next_step_duplicate_onset

random off-span controls:
  first_step_box_end: 0/96
  first_step_im_end:  0/96
  first_step tokens: all <|object_ref_end|>
  mean box-vs-object-boundary delta: +0.1250

true structural residual:
  first_step_box_end: 4/6
  first_step_im_end:  0/6
  first_step tokens: 4 <|box_end|>, 2 <|object_ref_end|>
  mean box-vs-object-boundary delta: +2.6354
```

Other important splits:

```text
failure_default128_rescued / neutral

random off-span controls:
  first_step_box_end: 0/288
  first_step_im_end:  87/288

true structural residual:
  first_step_box_end: 4/18
  first_step_im_end:  6/18

failure_joint128_only / next_step_unmatched_onset

random off-span controls:
  first_step_box_end: 0/384
  first_step_im_end:  0/384

true structural residual:
  first_step_box_end: 0/24
  first_step_im_end:  0/24
```

Seed stability:

```text
seed 101: first_step_box_end_rate 0.3333, im_end_rate 0.0590
seed 202: first_step_box_end_rate 0.3333, im_end_rate 0.0799
seed 303: first_step_box_end_rate 0.3333, im_end_rate 0.0799
seed 404: first_step_box_end_rate 0.3333, im_end_rate 0.0833
```

Interpretation:

- The row-specific structural residual is behaviorally specific.
- The duplicate-onset rescue is not caused by generic matched-norm energy in
  the orthogonal complement of the structural logit-gradient span.
- Random off-span controls preserve the clean rows and hard unmatched rows, but
  they do not turn the rescued duplicate-onset failure row into `<|box_end|>`.
- This strengthens the view that head `26:10` carries an object-span transition
  vector in the residual direction, not merely a large non-structural
  perturbation and not merely a local boundary-token readout axis.

Updated mechanism picture:

```text
structural logit-gradient span:
  visible boundary-token readout/control shadow

true row-specific residual:
  specific latent object-span transition direction; can rescue duplicate-onset
  closure

matched random off-span residual controls:
  mostly null for duplicate rescue despite matched norm; can induce some
  termination on neutral fragile rows but not object closure
```

Next branch:

- Factor row-specific residuals with case-conditioned or leave-one-case-out
  bases rather than global PCA.
- Compare the true residual direction against token embedding/unembedding
  directions for boundary, termination, object-ref, and coordinate tokens.
- For hard unmatched false negatives, run prefix guidance/visual-ablation probes
  because neither true residual nor random residual control creates box closure
  there.

Verification:

```text
python -m py_compile src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q -k "value_region_continuation or value_region_intervention or random_residual or residual_pc_projection_patch"
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q
```
