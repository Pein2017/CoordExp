# Per256 Staged Failure Mode Findings

Date: 2026-06-23

Scope: post-hoc reduction over the per256 train/val launch-filter guided-delta
artifacts, followed by a bounded model-backed layer/site scan over the resulting
recommended panel. The reducer separates broad trained-sequence and held-out
pre-x1 failures by baseline token mode, failure locus, patch-site outcome,
configured layer/site, and next recommended panel role.

## Tooling

Added:

```text
src/analysis/autoregressive_binding_template_ablation/staged_slot_failure_mode_reducer.py
scripts/analysis/run_autoregressive_binding_staged_slot_failure_mode_reduce.py
tests/analysis/test_staged_slot_failure_mode_reducer.py
```

The reducer consumes one or more `staged_slot_guided_delta_rows.jsonl` files and
writes:

```text
staged_slot_failure_mode_rows.jsonl
staged_slot_failure_mode_recommended_panel_rows.jsonl
staged_slot_failure_mode_summary.json
staged_slot_failure_mode_summary.md
```

## Artifacts

Input guided-delta rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v24_v3_per256_train_val_launch_filter_prevslot_afterx1y1_layerinput/staged_slot_guided_delta_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v25_v3_per256_train_val_launch_filter_prevslot_afterx1y1_selfattn/staged_slot_guided_delta_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v26_v3_per256_train_val_launch_filter_prevslot_afterx1y1_mlp/staged_slot_guided_delta_rows.jsonl
```

Reducer output:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_failure_mode_reducer/v1_v24_v25_v26_per256_train_val_failure_modes
```

Scope:

```text
input rows: 492
patch sites: layer_input=164, self_attn=164, mlp=164
split rows: train=246, val=246
recommended panel rows: 118
model_perturbation_ran: false
training_ran: false
```

## Main Result

The broad per256 train/val launch cohort is not one mechanism family. It splits
into coordinate-mode wrong-basin rows and a large premature boundary/router-mode
subset:

```text
baseline_token_mode_counts:
  coord_mode: 348
  wrapper_mode: 144

baseline_top1_family_counts:
  coord: 348
  box_end: 138
  object_ref_end: 3
  object_ref_start: 3
```

Failure locus counts:

```text
x1_onset_anchor_failure: 213
premature_boundary_mode: 138
small_object_extent_or_weak_evidence: 66
tail_no_rescue_or_delayed_evidence: 69
wrapper_mode_router_competition: 6
```

This matters because many rows selected as pre-x1 failures are not merely
choosing the wrong coordinate bin. They have already fallen into a boundary
mode where `<|box_end|>` or another wrapper token beats coordinate emission.
That boundary-mode subset exists in trained rows too:

```text
train premature_boundary_mode rows: 78
val premature_boundary_mode rows: 60
```

So trained-sequence failure is not a simple visual-exposure or unseen-val
problem. Some trained rows fail by wrong coordinate-basin selection, and some
fail by switching out of coordinate mode entirely before x1.

## Patch-Site Outcome

Previous-slot guided deltas remain mostly diagnostic rather than corrective:

```text
transition_outcome_counts:
  intrusive_transition: 294
  no_repair_or_worse: 86
  rank_only_clean_transition: 80
  strict_clean_local_repair: 32
```

By patch site:

```text
layer_input:
  strict_clean_local_repair_rate=0.0610
  intrusive_transition_rate=0.6707
  mean_rank_delta=+1606.66

self_attn:
  strict_clean_local_repair_rate=0.0732
  intrusive_transition_rate=0.5549
  mean_rank_delta=-505.02

mlp:
  strict_clean_local_repair_rate=0.0610
  intrusive_transition_rate=0.5671
  mean_rank_delta=-331.16
```

Self-attention and MLP improve rank more often than full layer-input on this
filtered per256 cohort, but exact/local repair remains rare. This is consistent
with the ownership-transition result: component paths expose or reshape basin
information, while generic previous-slot import usually still carries slot/value
intrusion.

## Layer 17-18 Failure-Mode Scan

A bounded follow-up GPU run used the 118-row recommended panel above and scanned
the layer-17/layer-18 self-attention and MLP component sites with the same
`donor_minus_previous_slot` handle at `staged_after_x1_y1`.

Artifacts:

```text
plan:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v29_failure_mode_panel_layers17_18_selfattn_mlp_plan

model-backed shards:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v30_failure_mode_panel_layers17_18_selfattn_mlp_shard0of3_gpu0
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v30_failure_mode_panel_layers17_18_selfattn_mlp_shard1of3_gpu1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_guided_delta_patch/v30_failure_mode_panel_layers17_18_selfattn_mlp_shard2of3_gpu7

layer-aware reducer:
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/staged_slot_failure_mode_reducer/v3_v30_layers17_18_selfattn_mlp_layer_aware
```

Scope:

```text
plan rows: 472
model-backed rows: 472
error_count: 0
layers: 17=236, 18=236
patch sites: self_attn=236, mlp=236
split rows: train=216, val=256
```

The layer/site aggregate is not a single monotonic repair surface:

```text
17 / mlp:
  strict_clean_local_repair_rate=0.1525
  intrusive_transition_rate=0.5169
  mean_rank_delta=+1074.30

17 / self_attn:
  strict_clean_local_repair_rate=0.1441
  intrusive_transition_rate=0.5169
  mean_rank_delta=-363.50

18 / mlp:
  strict_clean_local_repair_rate=0.1102
  intrusive_transition_rate=0.5508
  mean_rank_delta=+2127.16

18 / self_attn:
  strict_clean_local_repair_rate=0.1102
  intrusive_transition_rate=0.4915
  mean_rank_delta=-2084.22
```

The critical split is by failure locus:

```text
premature_boundary_mode:
  strict_clean_local_repair_rate=0.0 at every layer/site
  intrusive_transition_rate=1.0 at every layer/site
  layer18 self_attn mean_rank_delta=-5545.64
  layer18 mlp mean_rank_delta=+5515.82

x1_onset_anchor_failure:
  layer17 mlp strict_clean_local_repair_rate=0.2941
  layer17 self_attn strict_clean_local_repair_rate=0.2549
  layer18 mlp strict_clean_local_repair_rate=0.1765
  layer18 self_attn strict_clean_local_repair_rate=0.0980

small_object_extent_or_weak_evidence:
  strict_clean_local_repair_rate=0.1333-0.2000
  layer17 self_attn has the best mean patched distance among these sites

tail_no_rescue_or_delayed_evidence:
  layer18 self_attn strict_clean_local_repair_rate=0.3125
  layer18 self_attn intrusive_transition_rate=0.2500
```

Interpretation: for boundary/wrapper-mode rows, component patches can cause very
large coordinate-rank swings, especially at layer 18, but they still do not
become clean coordinate-local repairs. These rows are router/boundary failures,
not coordinate-basin repair targets. For coordinate-mode x1 onset rows, the best
clean/local repairs are earlier, concentrated at layer 17 MLP and layer 17
self-attention. This makes the previous component-tomography result more
specific: layer-18 self-attention is a powerful resolver/redistribution site,
but rank movement there is not equivalent to clean coordinate-basin ownership.

## Next Panel

The reducer materialized a 118-row recommended panel:

```text
intrusive_slot_transition: 31
premature_boundary_mode: 33
rank_only_clean_transition: 19
strict_clean_local_repair: 20
small_object_extent_or_weak_evidence: 6
tail_no_rescue_or_delayed_evidence: 6
wrapper_mode_router_competition: 3
```

Recommended next use:

1. Use `strict_clean_local_repair` versus `intrusive_slot_transition` rows for
   layer-17 self-attention and MLP source/value tomography, especially
   coordinate-mode `x1_onset_anchor_failure` rows.
2. Use `premature_boundary_mode` and `wrapper_mode_router_competition` rows for
   boundary/router-margin probes, not coordinate-basin repair probes.
3. Use `small_object_extent_or_weak_evidence` and
   `tail_no_rescue_or_delayed_evidence` rows for locality/smoothness and
   delayed-evidence checks. The tail subset is now a plausible layer-18
   resolver-locality target.
4. Keep train and val rows paired but not merged. The same surface symptom now
   hides at least two loci in both splits.

## Temporary Stop Note

This is a coherent pause point. The current finding does not complete the
overall mechanistic diagnosis goal, but it prevents the next GPU pass from
mixing coordinate-onset failures with premature boundary-mode failures. Per the
current execution instruction, stop after committing this round and wait for
approval before launching the next GPU probe.
