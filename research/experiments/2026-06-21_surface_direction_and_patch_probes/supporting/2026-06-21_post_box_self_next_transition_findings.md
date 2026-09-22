---
doc_id: progress.diagnostics.post_box_self_next_transition_findings
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: tiny-desc-first-strict-selected-self-next-and-counterfactual-readout
domain: autoregressive-binding-template-ablation
updated: 2026-06-21
branch: codex/autoregressive-binding-template-study
---

# Post-Box Self-Next Transition Findings

## Scope

This note records a tiny bridge probe for the desc-first `ckpt928`
token-embeddings-adapter surface. It asks whether, after replaying a model-owned
completed object span through `<|box_end|>`, the model is locally prepared to
continue into the next object that it actually generated in the same rollout.

This is a self-trajectory baseline. It does not prove that the model would
choose the right object under a counterfactual prefix, and it does not address
GT recall directly. It isolates whether the boundary after a valid completed
generated object is already hostile to continuation.

## Helper Added

New CPU-only row builder:

```text
scripts/analysis/run_autoregressive_binding_post_box_self_next_rows.py \
  --selected-rows <selected_rows.jsonl> \
  --pred-token-trace <pred_token_trace.jsonl> \
  --output-root <output_root>
```

It parses `pred_token_trace.generated_token_text` into generated object spans
and emits two trajectory-prefix rows per nonterminal selected object:

```text
post_box_object_ref_start
post_object_ref_desc
```

The first prefix ends at the current object's `<|box_end|>`. The second appends
`<|object_ref_start|>` so existing next-token readout can score the next
generated descriptor token.

## Inputs

Selected duplicate/unmatched object-step rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_panel_selection/v2_desc_first_duplicate_unmatched_slots_selector_strict_v1/selected_rows.jsonl
```

Rollout token trace:

```text
/data/CoordExp/outputs/infer/natadj_len12000_free_val200/natadj_sorted_ckpt928_val200_free_temp0_rp1p10_max3084_bsz1_8gpu_symlinkbase/pred_token_trace.jsonl
```

Pair config:

```text
configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
```

Important contract detail:

```text
selected_rows.object_idx is a generated-object ordinal, not a GT object index.
```

## Row Artifact

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_box_self_next_rows/v1_desc_first_strict_selected_self_next_v1
```

Summary:

```text
input_selected_row_count: 20
output_row_count: 32
post_box_probe_kind_counts:
  post_box_object_ref_start: 16
  post_object_ref_desc: 16
skipped_counts_by_reason:
  terminal_no_next_generated_object: 4
next_generated_desc_row_counts:
  backpack: 8
  broccoli: 10
  person: 2
  snowboard: 2
  wine glass: 10
next_generated_desc_transition_counts:
  backpack: 4
  broccoli: 5
  person: 1
  snowboard: 1
  wine glass: 5
```

The four skipped selected rows are terminal self-trajectory cases. For those,
the generated object list has no `object_idx + 1`.

## Readout Artifact

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_next_token_readout/post_box_self_next_v1_desc_first_strict_selected
```

Command:

```text
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage trajectory-prefix-next-token-readout \
  --trajectory-boundary-routing-selected-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_box_self_next_rows/v1_desc_first_strict_selected_self_next_v1/post_box_self_next_rows.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_next_token_readout/post_box_self_next_v1_desc_first_strict_selected \
  --device cuda:0 \
  --allow-model-load \
  --trajectory-prefix-source trajectory
```

Summary:

```text
row_count: 32
readout_status_counts: {'ok': 32}
target_next_kind_counts: {'desc': 16, 'object_ref_start': 16}
mean_target_next_token_prob: 0.782292533666
mean_target_next_token_rank: 1.0
```

By probe kind:

```text
post_box_object_ref_start:
  n: 16
  mean_target_next_token_prob: 0.962833583355
  target_rank: 1 for 16/16

post_object_ref_desc:
  n: 16
  mean_target_next_token_prob: 0.601751483977
  target_rank: 1 for 16/16
```

Lower-probability but still top-rank descriptor examples:

```text
snowboard first token: prob 0.486176431179, rank 1
person first token: prob 0.537941873074, rank 1
```

## Interpretation

On exact self-trajectory prefixes, the model is not locally confused at the
post-box boundary. After the current generated box is replayed through
`<|box_end|>`, it strongly ranks `<|object_ref_start|>` first, and after
appending `<|object_ref_start|>` it ranks the next generated descriptor token
first for every nonterminal selected row.

This pushes the next mechanism question upstream:

```text
Duplication is less likely to be caused by a generic inability to hand off after
a valid completed object span. The sharper failure likely occurs in the hidden
state that chooses the next identity/spatial basin, or in how a partially formed
object is attracted toward a nearby repeated-object trajectory before the
completed self-trajectory boundary is reached.
```

The next high-value probe should compare this self-trajectory boundary against
counterfactual completed boxes: GT target box, nearest previous same-class box,
and local basin-shifted boxes. If the continuation remains schema-strong but
descriptor/spatial target switches under those boxes, that would directly test
whether completed coordinate anchors steer the next object basin.

## Counterfactual Completed-Box Extension

Follow-up helper:

```text
scripts/analysis/run_autoregressive_binding_post_box_counterfactual_rows.py \
  --selected-rows <selected_rows.jsonl> \
  --pred-token-trace <pred_token_trace.jsonl> \
  --output-root <output_root>
```

It keeps the current object descriptor and pre-box prefix, but completes the
current object with alternate generated boxes:

```text
self_current_box
nearest_previous_same_desc_box
previous_object_box
next_generated_object_box
```

The next-token targets remain the model-owned next generated object after the
current selected object.

Row artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_box_counterfactual_rows/v1_desc_first_strict_selected_box_variants_v1
```

Summary:

```text
input_selected_row_count: 20
output_row_count: 126
completed_box_variant_row_counts:
  self_current_box: 32
  nearest_previous_same_desc_box: 30
  previous_object_box: 32
  next_generated_object_box: 32
completed_box_variant_transition_counts:
  self_current_box: 16
  nearest_previous_same_desc_box: 15
  previous_object_box: 16
  next_generated_object_box: 16
skipped_counts_by_reason:
  terminal_no_next_generated_object: 4
  box_variant_nearest_previous_same_desc_missing: 1
next_generated_desc_transition_counts:
  backpack: 16
  broccoli: 20
  person: 4
  snowboard: 3
  wine glass: 20
```

Readout artifact:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_next_token_readout/post_box_counterfactual_v1_desc_first_strict_selected_box_variants
```

Command:

```text
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage trajectory-prefix-next-token-readout \
  --trajectory-boundary-routing-selected-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/post_box_counterfactual_rows/v1_desc_first_strict_selected_box_variants_v1/post_box_counterfactual_rows.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_prefix_next_token_readout/post_box_counterfactual_v1_desc_first_strict_selected_box_variants \
  --device cuda:0 \
  --allow-model-load \
  --trajectory-prefix-source trajectory
```

Readout summary:

```text
row_count: 126
readout_status_counts: {'ok': 126}
target_next_kind_counts: {'desc': 63, 'object_ref_start': 63}
mean_target_next_token_prob: 0.756985279776
mean_target_next_token_rank: 1.063492063492
```

By completed-box variant and probe kind:

```text
post_box_object_ref_start:
  self_current_box: n=16, mean_prob=0.962833583355, top1=16/16
  nearest_previous_same_desc_box: n=15, mean_prob=0.975608003139, top1=15/15
  previous_object_box: n=16, mean_prob=0.967972464860, top1=16/16
  next_generated_object_box: n=16, mean_prob=0.965265076607, top1=16/16

post_object_ref_desc:
  self_current_box: n=16, mean_prob=0.601751483977, top1=16/16
  nearest_previous_same_desc_box: n=15, mean_prob=0.517633700371, top1=11/15
  previous_object_box: n=16, mean_prob=0.576474104077, top1=16/16
  next_generated_object_box: n=16, mean_prob=0.487048268318, top1=12/16
```

The `self_current_box` rows exactly reproduce the previous self-next baseline:

```text
self_current_vs_self_next max_abs_prob_delta: 0
diff_count: 0
```

### Counterfactual Interpretation

The schema boundary is stable under these completed-box counterfactuals:
`<|object_ref_start|>` remains rank 1 for every variant. The semantic handoff is
not stable. The descriptor target remains rank 1 for all self-current and
previous-object completions, but it drops to rank 2 in selected nearest-same and
next-object box variants.

The sharpest case is source123/object5 (`backpack`):

```text
self_current_box:
  target 'back' prob about 0.640066, rank 1
previous_object_box (sports ball box):
  target 'back' prob about 0.564050, rank 1
nearest_previous_same_desc_box (backpack box):
  target 'back' prob about 0.247348, rank 2; top token 'person'
next_generated_object_box (backpack box):
  target 'back' prob about 0.187867, rank 2; top token 'person'
```

This suggests a more specific mechanism than generic autoregressive
continuation failure:

```text
The completed coordinate anchor appears to help choose the next semantic basin.
Object-start formatting is robust, but the first descriptor token is sensitive
to which completed box anchors the just-finished object, and can flip toward a
scene/location prior such as 'person' even when the descriptor context remains
the current object's category.
```

This is still a tiny, selected-row diagnostic. It is strong enough to justify a
larger hidden-state probe at the completed-box boundary: compare residual and
attention features for self-current versus nearest-same/next-object box
completions, especially in source123 where the descriptor top token changes.
