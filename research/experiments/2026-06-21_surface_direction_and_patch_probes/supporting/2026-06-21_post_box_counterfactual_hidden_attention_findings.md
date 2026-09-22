---
doc_id: progress.diagnostics.post_box_counterfactual_hidden_attention_findings
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: tiny-desc-first-strict-selected-counterfactual-hidden-attention-readout
domain: autoregressive-binding-template-ablation
updated: 2026-06-21
branch: codex/autoregressive-binding-template-study
---

# Post-Box Counterfactual Hidden/Attention Findings

## Scope

This note records the first hidden-state and attention follow-up on the
post-box counterfactual transition probe. It uses a tiny selected panel from the
desc-first `ckpt928` token-embeddings-adapter surface.

The question is narrow:

```text
When a completed-box counterfactual flips the next descriptor basin, does the
flip look like early visual/semantic unavailability, or like a late overwrite
of an already-available target descriptor?
```

This is not a full validation result and not causal proof. It is a selected
diagnostic panel that should be used to choose the next causal perturbation.

## Implemented Helpers

Boundary selector:

```text
scripts/analysis/run_autoregressive_binding_post_box_counterfactual_boundary_selection.py
```

The selector reads post-box counterfactual next-token readout rows, filters to
`post_object_ref_desc` descriptor states, deduplicates repeated coordinate-slot
copies, and emits a compact hidden/attention-ready panel.

Hidden competitor lens:

```text
src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py
```

The trajectory hidden-state readout now adds layerwise metrics for each row's
`post_box_boundary_variant_top_token_text`:

```text
layer_boundary_variant_top_token_text
layer_boundary_variant_top_token_id
layer_boundary_variant_top_token_logit
layer_boundary_variant_top_token_prob
layer_boundary_variant_top_token_rank
layer_boundary_variant_top_is_target_token
layer_target_minus_boundary_variant_top_logit
layer_boundary_variant_top_minus_target_logit
```

These fields let the readout track the target descriptor token against the
alternative descriptor token that actually won in the next-token readout.

## Commits

```text
8666a4fc Add post-box counterfactual transition probe
3a5b1507 Add post-box counterfactual boundary selection
083f61ae Add boundary competitor token hidden lens
```

The earlier self-next baseline commits are:

```text
46017eb6 Add post-box self-next row builder
ac170af4 Record post-box self-next transition probe
```

## Inputs

Counterfactual next-token readout:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_next_token_readout/post_box_counterfactual_v1_desc_first_strict_selected_box_variants/trajectory_prefix_next_token_readout_rows.jsonl
```

Pair config:

```text
configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
```

## Boundary Selection Artifact

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_box_counterfactual_boundary_selection/v1_desc_first_strict_selected_descriptor_flips_v1
```

Command:

```text
python scripts/analysis/run_autoregressive_binding_post_box_counterfactual_boundary_selection.py \
  --readout-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_next_token_readout/post_box_counterfactual_v1_desc_first_strict_selected_box_variants/trajectory_prefix_next_token_readout_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_box_counterfactual_boundary_selection/v1_desc_first_strict_selected_descriptor_flips_v1
```

Summary:

```text
input_row_count: 126
filtered_probe_row_count: 63
logical_case_count: 7
selected_case_count: 2
output_row_count: 7
duplicate_row_count: 36
conflicting_variant_prefix_count: 0
selected_cases_by_target_desc:
  backpack: 1
  snowboard: 1
selected_rows_by_variant_role:
  baseline: 2
  flip: 3
  stable_counterfactual: 2
flipped_rows_by_completed_box_variant:
  nearest_previous_same_desc_box: 1
  next_generated_object_box: 1
  previous_object_box: 1
```

Selected cases:

```text
case 1: source_line_idx=123, object_idx=5, target_desc=backpack
  self_current_box: target 'back' rank 1, prob 0.640066
  nearest_previous_same_desc_box: target 'back' rank 2, prob 0.247348, top 'person'
  previous_object_box: target 'back' rank 1, prob 0.564050
  next_generated_object_box: target 'back' rank 2, prob 0.187867, top 'person'

case 2: source_line_idx=36, object_idx=9, target_desc=snowboard
  self_current_box: target 'snow' rank 1, prob 0.486176
  previous_object_box: target 'snow' rank 1, prob 0.386811, top 'sk'
  next_generated_object_box: target 'snow' rank 1, prob 0.514987
```

## Hidden-State Readout Artifact

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_readout/post_box_counterfactual_boundary_descriptor_flips_competitor_lens_v1
```

Command:

```text
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage trajectory-hidden-state-readout \
  --prefix-greedy-trajectory-readout-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_box_counterfactual_boundary_selection/v1_desc_first_strict_selected_descriptor_flips_v1/post_box_counterfactual_boundary_selected_rows.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_readout/post_box_counterfactual_boundary_descriptor_flips_competitor_lens_v1 \
  --device cuda:0 \
  --allow-model-load \
  --hidden-layers=-1,-4,-8,-12,-16,-20,-24,-28 \
  --target-next-kinds desc
```

Summary:

```text
state_row_count: 7
row_count: 56
readout_status_counts:
  ok: 56
hidden layers requested:
  -1,-4,-8,-12,-16,-20,-24,-28
resolved layer indices:
  28,25,21,17,13,9,5,1
```

### Backpack Case

For the two flip variants, target `back` is weak early, competing `person`
becomes strong by layer 21, `back` briefly wins at layer 25, then `person`
retakes the final layer.

```text
nearest_previous_same_desc_box:
  layer 21: back prob 0.0000242 rank 3329; person prob 0.155129 rank 2; back-person logit -8.765625
  layer 25: back prob 0.350713 rank 1; person prob 0.176349 rank 2; back-person logit +0.6875
  layer 28: back prob 0.247348 rank 2; person prob 0.672361 rank 1; back-person logit -1.0

next_generated_object_box:
  layer 21: back prob 0.0000241 rank 3342; person prob 0.092234 rank 2; back-person logit -8.25
  layer 25: back prob 0.301257 rank 1; person prob 0.207051 rank 2; back-person logit +0.375
  layer 28: back prob 0.187867 rank 2; person prob 0.743030 rank 1; back-person logit -1.375
```

The stable rows do not show that final target loss:

```text
self_current_box final: back rank 1, prob 0.640066
previous_object_box final: back rank 1, prob 0.564050
```

### Snowboard Case

The snowboard case looks different. It is a softer descriptor-token basin wobble
between `snow` and `sk`, not the clean semantic overwrite seen in the backpack
case.

```text
previous_object_box:
  layer 21: snow prob 0.002501 rank 44; sk prob 0.006799 rank 14; snow-sk logit -1.0
  layer 25: snow prob 0.000402 rank 14; sk prob 0.822964 rank 1; snow-sk logit -7.625
  layer 28: snow prob 0.386811 rank 1; sk prob 0.386811 rank 1; snow-sk logit 0.0
```

The final-layer tie-like behavior means this row should not be overinterpreted
as a strong semantic collapse. It is still useful as a tokenization/local
descriptor-basin wobble control.

## Attention Readout Artifact

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_anchor_attention/post_box_counterfactual_boundary_descriptor_flips_layers_v1
```

Command:

```text
CUDA_VISIBLE_DEVICES=1 python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage trajectory-boundary-anchor-attention \
  --trajectory-boundary-routing-selected-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_box_counterfactual_boundary_selection/v1_desc_first_strict_selected_descriptor_flips_v1/post_box_counterfactual_boundary_selected_rows.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_anchor_attention/post_box_counterfactual_boundary_descriptor_flips_layers_v1 \
  --device cuda:0 \
  --allow-model-load \
  --hidden-layers=-1,-4,-8 \
  --target-next-kinds desc
```

Summary:

```text
selected_state_row_count: 7
row_count: 8160
attention_implementation: eager
layers requested: -1,-4,-8
```

The coarse attention summary does not show a large average same-desc-history
reroute. For the backpack final layer, flip variants have small mean increases
to same-desc-history regions but not enough to treat attention mass as the
primary finding:

```text
nearest_previous_same_desc_box vs self_current_box, final layer:
  same_desc_history_object_ref_boundaries mean delta: +0.000453
  same_desc_history_box_spans mean delta: +0.000042

next_generated_object_box vs self_current_box, final layer:
  same_desc_history_object_ref_boundaries mean delta: +0.000294
  same_desc_history_box_spans mean delta: +0.000060
```

The largest head-region differences are instead concentrated around the current
partial object-ref boundary:

```text
next_generated_object_box vs self_current_box, final layer:
  head 07 current_partial_row_span delta: +0.058390
  head 15 current_partial_row_span delta: +0.052371

nearest_previous_same_desc_box vs self_current_box, final layer:
  head 15 current_partial_row_span delta: +0.029591
```

## Interpretation

The backpack flip is not well explained by early target invisibility or by a
generic inability to continue after a completed object span. The hidden-state
readout shows that the target descriptor token can become linearly dominant at
layer 25, then lose to `person` by the final layer only for the completed-box
variants that anchor the current object to a previous/next backpack-like box.

This points to a late-layer descriptor-basin overwrite:

```text
The model has a recoverable `back` continuation for the next object, but the
final transformation can overwrite it with a scene/identity prior (`person`)
conditioned on the completed coordinate anchor.
```

Attention evidence is weaker and should be treated as routing context rather
than a cause. The average same-desc-history attention change is small, while a
few heads shift attention around the current partial object-ref boundary. The
next causal probe should therefore intervene in late hidden/residual space or
descriptor-token directions before spending heavily on broad attention claims.

## Next Probe

Highest-value next causal test:

```text
For backpack flip rows, intervene near layers 25-28 on the target-vs-competitor
descriptor direction (`back` minus `person`) or on the baseline-vs-flip residual
difference, then measure whether the final descriptor readout/continuation
returns from `person` to `back`.
```

Recommended minimal run:

```text
case: source_line_idx=123, object_idx=5, target_desc=backpack
rows: self_current_box, nearest_previous_same_desc_box, next_generated_object_box
layers: final late layers around resolved 25 and 28
metrics: final target-minus-person logit, top token, greedy continuation
controls: previous_object_box stable counterfactual and snowboard tokenization wobble
```

This is the first place where the evidence is pointing at a concrete causal
mechanism rather than just a surface phenotype.
