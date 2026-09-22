---
doc_id: progress.diagnostics.source33_y2_local_anchor_basin_probe
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: tiny-one-state-desc-first-source33-object11-y2-readout-only
domain: autoregressive-binding-template-ablation
updated: 2026-06-21
branch: codex/autoregressive-binding-template-study
---

# Source33 Y2 Local Anchor Basin Probe

## Scope

This note records a one-state GPU readout that completes the strict-selector
`y2` comparator set with `source_line_idx=33`, `object_idx=11`, and target
`y2=417`.

The case is useful because the prefix already contains a tight run of
same-description `wine glass` y2 anchors:

```text
408, 415, 413, 414, 416
```

The selected target row then asks for `417`, making this a clean local-anchor
basin test. This is tiny readout evidence, not full decode or population
evidence.

## Commands

```text
python scripts/analysis/run_autoregressive_binding_coordinate_threshold_hidden_state_probe.py \
  --stage build-panel \
  --source-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/object_step_panel_selection/v2_desc_first_duplicate_unmatched_slots_selector_strict_v1/selected_rows.jsonl \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source33_object11_y2_panel_v1 \
  --family desc_first \
  --source-line-idx 33 \
  --image-id 2685 \
  --object-idx 11 \
  --coord-slots y2 \
  --target-next-kind coord \
  --trajectory-stop-reason object_step_role

CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_binding_realized_behavior_bridge.py \
  --stage trajectory-hidden-state-readout \
  --prefix-greedy-trajectory-readout-rows /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source33_object11_y2_panel_v1/selected_rows.jsonl \
  --pair-config configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml \
  --output-root /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_probe/source33_object11_y2_layers20_24_28_surface_probe_bins_v1 \
  --device cuda:0 \
  --allow-model-load \
  --max-rows 1 \
  --hidden-layers 20,24,28 \
  --probe-coord-bins 413,414,415,416,417,418,419 \
  --families desc_first \
  --intervention-arms natural \
  --target-next-kinds coord \
  --stop-reasons object_step_role
```

Post-hoc reducers were then run for hidden-state summary,
adapter-surface summary, local-simplex summary, and adapter-scale sweep.

## Artifacts

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source33_object11_y2_panel_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_state_probe/source33_object11_y2_layers20_24_28_surface_probe_bins_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source33_object11_y2_hidden_state_summary_surface_bins_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source33_object11_y2_adapter_surface_probe_summary_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source33_object11_y2_local_simplex_surface_summary_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source33_object11_y2_adapter_scale_sweep_v1
```

Artifact checks:

```text
trajectory-hidden-state-readout: row_count=3, state_row_count=1, readout_status_counts={"ok": 3}, model_perturbation_ran=false, training_ran=false
coordinate-threshold-hidden-state-summary: row_count=3, state_count=1, readout_status_counts={"ok": 3}, readout_only=true
local-simplex-surface-summary: row_count=3, state_count=1, readout_status_counts={"ok": 3}, readout_only=true
adapter-scale-sweep-summary: row_count=3, state_count=1, readout_status_counts={"ok": 3}, readout_only=true
```

## Source33 Findings

Final-layer hidden-state readout:

```text
target: 417
layer: 28
local top1: 415
top1_delta: 2
target_rank: 3
```

Final-layer local-simplex surface:

```text
layer surface: top=415, target_minus_max=-0.250
base surface: top=415, target_minus_max=-24.739
effective surface: top=415, target_minus_max=-26.856
adapter delta: top=418, target_minus_max=-2.895
adapter-scale exact interval: empty
best sampled adapter scale: 0, target_minus_max=-24.739
```

Interpretation:

The hidden state is already locally near the intended coordinate, but the
local family is not target-centered. The readout prefers an earlier same-desc
anchor around 415. The adapter does not simply rescue the target; at the final
layer it makes the target worse relative to 415, while the adapter delta itself
peaks at 418. No scalar adapter-scale along `effective - base` makes target
417 strictly beat all requested local competitors.

## Four-State Y2 Comparison

| source/object | target | final layer top | rank | local layer top/margin | base top/margin | effective top/margin | adapter delta top/margin | exact scale interval | best sampled scale/margin |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| 33/11 | 417 | 415 | 3 | 415 / -0.250 | 415 / -24.739 | 415 / -26.856 | 418 / -2.895 | empty | 0 / -24.739 |
| 36/14 | 996 | 999 | 5 | 999 / -2.250 | 998 / -4.408 | 999 / -56.986 | 999 / -55.256 | scale < -1.03957160748 | -1 / -0.168 |
| 123/5 | 270 | 259 | 12 | 259 / -0.625 | 188 / -135.481 | 188 / -101.887 | 270 / 18.992 | scale > 4.83895920484 | 1.25 / -93.488 |
| 145/20 | 146 | 154 | 6 | 154 / -0.250 | 158 / -38.610 | 158 / -46.866 | 158 / -8.256 | empty | 0 / -38.610 |

This splits the y2 cases into at least three candidate readout-surface
regimes:

1. `33/11` and `36/14`: local hidden-state basin is close to the target, but
   the post-adapter surface is attracted to a nearby repeated anchor.
2. `123/5`: adapter delta is target-helpful, but the target would win this
   local one-dimensional adapter-scale surface only for
   `scale > 4.83895920484`, outside the sampled sweep.
3. `145/20`: neither base/effective surface nor adapter-delta direction points
   cleanly at the target; a one-dimensional adapter-scale explanation fails.

The next promising direction is a model-backed causal test that separates
local hidden-state coordinate evidence from post-adapter surface attraction:
patch or project only the coordinate-token surface direction for the local
family while preserving the object prefix, then test whether repeated-anchor
basins move without changing the semantic object state.
