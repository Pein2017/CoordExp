---
doc_id: progress.diagnostics.source36_value_region_coord_probe_causal_findings
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: tiny-one-state-source36-y2-single-head-value-region-intervention
domain: autoregressive-binding-template-ablation
updated: 2026-06-21
branch: codex/autoregressive-binding-template-study
---

# Source36 Value-Region Coord-Probe Causal Findings

## Scope

This note follows the local-ridge readout alignment result:

```text
progress/diagnostics/2026-06-21_source36_local_ridge_head_surface_alignment_findings.md
```

That note identified source36/object14/y2 as a stubborn local coordinate-basin
case:

```text
target coord bin: 996
baseline coord top1: 999
target rank in coord-only distribution: 5
target-minus-999 baseline margin: -2.375
```

It also found the strongest readout-aligned candidate route:

```text
head: 26:10
source region: pre_prefix_non_image_context
surface basis: coord_996_minus_coord_999
residual projection: -5.0583
contribution projection: -4.7085
```

The new probe asks whether suppressing specific single-head value-region
components can causally move the coordinate slot away from the 999 attractor
and toward target bin 996. Evidence here is still tiny: one source row and five
single-head/source-region panels.

## Implementation Surface

The trajectory-boundary head value-region intervention stage now supports local
coordinate-family probe bins:

```text
--probe-coord-bins 996,999,998,997
```

For each baseline/intervention row, the stage records local logits,
full-vocab probabilities, coord-only probabilities, coord-only ranks, and
target-minus-probe margins for requested coord bins. Summary and report outputs
also include mean per-bin deltas and coord-probe readout status counts.

Verification:

```text
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -k "coord_probe or probe_bins" -q
  8 passed
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -k "value_region_intervention and coord_probe" -q
  1 passed
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -q
  376 passed
python -m py_compile src/analysis/autoregressive_binding_template_ablation/realized_behavior_bridge.py tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py
  exit 0
git diff --check
  exit 0
```

The CLI dispatch path is covered:

```text
python -m pytest tests/analysis/test_autoregressive_binding_template_realized_behavior_bridge.py -k "value_region_intervention_cli" -q
  2 passed
```

## Artifacts

Input selected row:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source36_object14_y2_panel_v1/selected_rows.jsonl
```

All model-backed panels used:

```text
pair_config: configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
candidate row: source36/object14/y2
family: desc_first
target_next_kind: coord
stop_reason: object_step_role
value_region_patch_mode: full_vector_subtract
value_region_scales: 0,0.25,0.5,0.75,1.0
probe_coord_bins: 996,999,998,997
```

Artifact roots:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_intervention/source36_object14_y2_head26_10_prectx_local_family_fullsub_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_intervention/source36_object14_y2_head26_10_currentobj_local_family_fullsub_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_intervention/source36_object14_y2_head21_2_prectx_local_family_fullsub_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_intervention/source36_object14_y2_head21_2_currentobj_local_family_fullsub_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_boundary_head_value_region_intervention/source36_object14_y2_head18_0_prectx_local_family_fullsub_v1
```

All five regenerated summaries report:

```text
row_count: 5
probe_coord_bins: [996, 999, 998, 997]
baseline_coord_probe_readout_status_counts: {"ok": 5}
intervention_coord_probe_readout_status_counts: {"ok": 5}
```

## Main Contrast

Mean deltas are over the five tested scales, including the scale-0 baseline
row. Positive `mean target-minus-999 delta` means the intervention moved the
target 996 logit upward relative to attractor 999.

| head | source region | mean target-minus-999 delta | mean probe999 logit delta | mean target prob delta | mean box-vs-object-ref delta |
| --- | --- | ---: | ---: | ---: | ---: |
| 26:10 | pre_prefix_non_image_context | +0.075 | -0.100 | +0.001179 | +0.141309 |
| 26:10 | current_object_ref_boundaries | -0.075 | +0.000 | -0.002680 | +0.012402 |
| 21:2 | pre_prefix_non_image_context | -0.025 | +0.000 | -0.001165 | -0.103223 |
| 21:2 | current_object_ref_boundaries | -0.100 | +0.050 | -0.002660 | -0.048096 |
| 18:0 | pre_prefix_non_image_context | -0.050 | -0.150 | -0.002324 | -0.067822 |

The strongest prior readout candidate, `26:10` from
`pre_prefix_non_image_context`, is the only panel that moves the
target-minus-999 margin in the predicted direction on average.

Per-scale readout for that panel:

| scale | target-minus-999 delta | probe999 logit delta | target prob delta | local coord top1 after intervention | target rank |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | +0.000 | +0.000 | +0.000000 | 999 | 5 |
| 0.25 | +0.125 | -0.125 | +0.002501 | 999 | 5 |
| 0.50 | +0.125 | -0.125 | +0.002724 | 999 | 5 |
| 0.75 | +0.000 | -0.125 | -0.001854 | 999 | 5 |
| 1.00 | +0.125 | -0.125 | +0.002525 | 999 | 5 |

## Interpretation

This is a useful negative-causal bridge:

```text
The strongest local-ridge readout route is real enough to perturb the local
coordinate simplex in the predicted direction, but it is far too small to
recenter the coordinate slot.
```

For source36/object14/y2:

```text
baseline target-minus-999 margin: -2.375
best single-head value-region margin improvement: +0.125
coord top1 after all tested single-head interventions: 999
target rank after all tested single-head interventions: 5
```

That rules out a simple story in which the bad 999 emission is caused by one
isolated head-value route that can be subtracted cleanly. The 26:10 pre-prefix
component looks like one contributor to the 999 ridge, not the whole basin.

The current-object boundary panels are especially informative. They do not
rescue the coordinate bin, and in several cases they make the target-vs-999
margin worse. The observed local-ridge pressure is therefore more history and
prefix routed than direct-current-object routed for this state.

## Mechanistic Update

The source36 case now looks like a distributed local coordinate-attractor basin:

```text
1. Visual/locality information is not absent. The target neighborhood is
   present, and coordinate-family mass is available.

2. Single coordinate contrast directions can raise p996, but source36 falls
   into adjacent or nearby basins instead of centering on 996.

3. Single-head value-region subtraction can slightly weaken 999, but not enough
   to change the winner or even the target rank.

4. The token_embeddings_adapter likely amplifies an already-displaced local
   coordinate surface rather than creating the whole displacement alone.
```

This preserves the more important hypothesis:

```text
CE has preserved geometry locality while producing a nonsmooth local coordinate
slot basin. The autoregressive state is near the right region, but the emission
surface is locally rugged and distributed across residual, head, and adapter
routes.
```

## Next Useful Probes

Prefer probes that can distinguish a distributed basin from a missed visual
perception failure:

```text
1. Multi-head combined intervention:
   jointly subtract 26:10 pre-prefix with other aligned contributors such as
   18:0 pre-prefix and selected late residual/MLP components. Success would
   imply distributed but removable route pressure.

2. Local coordinate simplex intervention:
   intervene against a family mean such as mean(999,998,997,995,994) instead of
   one competitor at a time. Success would imply the basin is a local simplex
   attractor rather than a single wrong token.

3. Adapter-surface causal bridge:
   patch or project the token_embeddings_adapter delta for source36 against
   steerable source123/source145 cases. Success would tie the stubborn basin to
   the new coordinate-token embedding surface.

4. Visual-guidance false-negative bridge:
   for missing-object cases, compare unguided prefix states to guided
   object-name or region-cue states. If guidance recovers the object without
   changing image features, the failure is language/context routing rather than
   visual perception absence.

5. Tiny training intervention:
   run a limited local-coordinate smoothness or simplex contrast update on
   selected coord slots, then check whether source36-like stubborn displaced
   basins recenter without hurting broad geometry locality.
```
