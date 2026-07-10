---
doc_id: progress.diagnostics.source36_combined_value_region_coord_probe_findings
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: tiny-one-state-source36-y2-combined-head-value-region-intervention
domain: autoregressive-binding-template-ablation
updated: 2026-06-21
branch: codex/autoregressive-binding-template-study
---

# Source36 Combined Value-Region Coord-Probe Findings

## Scope

This note follows:

```text
progress/diagnostics/2026-06-21_source36_value_region_coord_probe_causal_findings.md
```

The previous single-head result found that `26:10` from
`pre_prefix_non_image_context` perturbs source36/object14/y2 in the predicted
direction, but far too weakly to recenter the coordinate slot:

```text
target coord bin: 996
baseline local coord top1: 999
baseline target-minus-999 margin: -2.375
best single-head value-region margin improvement: +0.125
target rank after single-head interventions: 5
```

This follow-up tests whether the failure is just a distributed sum of several
attention-head value-region contributors. It applies multiple head patches in
one forward pass with:

```text
candidate_head_group_mode: combined_all
```

Evidence is still tiny: one source row, one source region, two combined-head
sets, five intervention scales each.

## Implementation Surface

`trajectory-boundary-head-value-region-intervention` now supports the same
candidate-head grouping mode that continuation already used:

```text
--candidate-head-group-mode combined_all
```

For immediate intervention, `combined_all` applies all requested attention-head
value-region patches in one first-step forward pass, then writes the same
baseline/intervention coord-probe fields as the independent single-head path.

Verification:

```text
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -k "value_region_intervention and (combined_mode or cli)" -q
  4 passed
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -k "trajectory_boundary_head_value_region_intervention or value_region_intervention_cli or coord_probe" -q
  20 passed
python -m py_compile src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
  exit 0
```

The first attempted model-backed combined runs exposed a real plumbing bug:
`candidate_head_group_mode` reached the runtime but was not copied into the
prepared intervention context, so the branch still ran independently. After the
handoff fix, both model-backed combined runs wrote exactly five rows, one per
scale, proving true combined execution rather than one panel per head.

## Artifacts

Input selected row:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source36_object14_y2_panel_v1/selected_rows.jsonl
```

Pair config:

```text
configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
```

Both model-backed runs used:

```text
family: desc_first
target_next_kind: coord
stop_reason: object_step_role
value_source_region: pre_prefix_non_image_context
value_region_patch_mode: full_vector_subtract
value_region_scales: 0,0.25,0.5,0.75,1.0
probe_coord_bins: 996,999,998,997
```

Artifact roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_intervention/source36_object14_y2_heads26_10_18_0_prectx_combined_local_family_fullsub_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_intervention/source36_object14_y2_heads26_10_18_0_21_3_prectx_combined_local_family_fullsub_v1
```

Both summaries report:

```text
candidate_head_group_mode: combined_all
candidate_head_group_mode_counts: {"combined_all": 5}
intervention_kind_counts: {"attention_head_value_region_combined_subtract": 5}
row_count: 5
probe_coord_bins: [996, 999, 998, 997]
```

## Main Contrast

| panel | row count | mean target-minus-999 delta | mean probe999 logit delta | mean target prob delta | mean box-vs-object-ref delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 26:10 pre-prefix single | 5 | +0.075 | -0.100 | +0.001179 | +0.141309 |
| 18:0 pre-prefix single | 5 | -0.050 | -0.150 | -0.002324 | -0.067822 |
| 26:10 + 18:0 combined | 5 | +0.000 | -0.175 | -0.000744 | +0.066113 |
| 26:10 + 18:0 + 21:3 combined | 5 | +0.075 | -0.275 | +0.001718 | +0.025684 |

Per-scale for the stronger three-head combined panel:

| scale | target-minus-999 delta | probe999 logit delta | target prob delta | local coord top1 after intervention | target rank |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | +0.000 | +0.000 | +0.000000 | 999 | 5 |
| 0.25 | +0.000 | -0.125 | -0.000588 | 999 | 5 |
| 0.50 | +0.125 | -0.375 | +0.002868 | 999 | 5 |
| 0.75 | +0.125 | -0.375 | +0.003910 | 999 | 5 |
| 1.00 | +0.125 | -0.500 | +0.002403 | 999 | 5 |

## Interpretation

The combined interventions are stronger at suppressing the visible 999
attractor logit, but they still do not change the local coordinate winner or
the target rank:

```text
baseline target rank: 5
post-combined target rank: 5
baseline local top1: 999
post-combined local top1: 999
largest observed 999-logit suppression: -0.500
largest observed target-minus-999 margin improvement: +0.125
```

This extends the previous negative-causal bridge:

```text
The source36 coordinate error is not explained by one removable head route, and
not rescued by jointly subtracting the most suspicious small set of pre-prefix
head value contributors.
```

The important nuance is that the three-head panel does push on the expected
coordinate-family surface. It reduces 999 more than the single-head panel and
raises p996 at medium/high scales. But the target remains rank 5, which means
the basin is deeper or more structured than this local route removal.

## Mechanistic Update

The source36 case now looks less like a simple additive-route error and more
like a local coordinate-simplex or adapter-surface basin:

```text
1. Head/value routes contribute measurable pressure toward or away from the
   local attractor, but removing several of them does not recenter the slot.

2. The local coordinate family is highly responsive in small logit increments
   of 0.125, but the winner remains locked to 999.

3. Stronger 999 suppression does not imply target recovery; the target bin 996
   remains rank 5. This points to a rugged local simplex where several adjacent
   bins, not only 999, absorb probability mass.

4. The next core test should operate on the local coordinate family itself or
   on the token_embeddings_adapter surface, rather than adding more individual
   head removals.
```

## Local-6 Attractor Follow-Up

After the first combined panel, the same intervention family was rerun with the
full local attractor set suggested by the coordinate-mean causal patch:

```text
probe_coord_bins: 996,999,998,997,995,994
```

Additional artifact roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_intervention/source36_object14_y2_head26_10_prectx_local6_fullsub_v2
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_intervention/source36_object14_y2_heads26_10_18_0_prectx_combined_local6_fullsub_v2
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_intervention/source36_object14_y2_heads26_10_18_0_21_3_prectx_combined_local6_fullsub_v2
```

All three wrote five rows and valid coord-probe readouts.

The local-family scoring used:

```text
target: 996
competitor family: 999,998,997,995,994
```

Best nonzero-scale results:

| panel | best target-minus-max delta | best target-minus-mean delta | local top1 after intervention | target family rank |
| --- | ---: | ---: | ---: | ---: |
| 26:10 single | +0.125 | +0.025 | 999 | 5 |
| 26:10 + 18:0 combined | +0.125 | +0.000 | 999 | 5 |
| 26:10 + 18:0 + 21:3 combined | +0.125 | +0.050 | 999 | 5 |

The strongest three-head setting at scale `0.75` had:

```text
probe999 logit delta: -0.375
probe997 logit delta: -0.375
probe995 logit delta: -0.250
target-minus-local-family-mean delta: +0.050
target-minus-local-family-max delta: +0.125
target family rank: 5
local top1: 999
```

This is the most useful negative result in the panel. The intervention can
jointly suppress several attractor-family members, but only slightly improves
the target relative to the family as a whole. The causal coordinate-mean patch
needed a much stronger direct simplex direction to make 996 win. That means the
missing operation is not simply "remove these attention-head values"; it is
closer to an adapter/residual surface rotation that changes the local coordinate
simplex itself.

## Next Probe

Highest-value continuation:

```text
Implement a coordinate-simplex direction such as:

  coord_target_minus_mean_bins:999,998,997,995,994

and compare source36 against steerable source123/source145. If this rescues
source36, the basin is local-simplex distributed. If it still does not rescue,
the remaining likely mechanism is adapter/residual geometry rather than local
competitor suppression.
```
