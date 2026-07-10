---
doc_id: progress.diagnostics.source33_coordinate_causal_recenter_probe
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: tiny-one-state-source33-y2-model-backed-activation-patch
domain: autoregressive-binding-template-ablation
updated: 2026-06-21
branch: codex/autoregressive-binding-template-study
---

# Source33 Coordinate Causal Recenter Probe

## Scope

This note follows the readout-only source33 local-anchor basin note:

```text
progress/diagnostics/2026-06-21_source33_y2_local_anchor_basin_probe.md
```

The target state is:

```text
family: desc_first
source_line_idx: 33
image_id: 2685
object_idx: 11
slot: y2
target coord: 417
prefix same-desc y2 anchors: 408, 415, 413, 414, 416
```

The readout-only evidence showed a local but off-target final basin:

```text
layer 28 top: 415
target 417 rank: 3
adapter-scale interval: empty
```

This probe asks whether model-backed raw decoder-layer activation patches can
recenter the local coordinate family toward target `417` while leaving the
object prefix fixed. This is tiny one-state causal-patch evidence, not
population validation or natural decode proof.

## Runs

All runs used:

```text
pair_config: configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
source rows: /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source33_object11_y2_panel_v1/selected_rows.jsonl
family: desc_first
intervention_arm: natural
target_next_kind: coord
stop_reason: object_step_role
patch_alphas: 0
probe_coord_bins: 413,414,415,416,417,418,419
training_ran: false
model_weight_update_ran: false
```

### Late Ridge Resolution: 24 -> 28

Single-bin sweep:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/source33_object11_y2_s24_t28_coord_target_minus_local_bins_v1
direction bases: coord_target_minus_bin:415, coord_target_minus_bin:418, coord_target_minus_bin:413, coord_target_minus_bin:416
strengths: 0,8,16,32,64,128,256,512
row_count: 35
realized_direction_patch_count: 32
readout_status_counts: {"ok": 35}
```

Local-mean sweep:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/source33_object11_y2_s24_t28_coord_target_minus_mean_local_bins_v1
direction bases: coord_target_minus_mean:415+418, coord_target_minus_mean:413+414+415+416+418+419, coord_target_minus_mean:413+414+415+416, coord_target_minus_mean:415+416
strengths: 0,16,32,64,128,256,512,1024
row_count: 35
realized_direction_patch_count: 32
readout_status_counts: {"ok": 35}
```

Post-hoc probe-support reducers:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_causal_patch_probe_support/source33_object11_y2_s24_t28_coord_target_minus_local_bins_v1
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_causal_patch_probe_support/source33_object11_y2_s24_t28_coord_target_minus_mean_local_bins_v1
probe_support_row_count: 245 each
strongest_row_count: 35 each
```

### Onset Formation: 20 -> 24

The 24 -> 28 run showed source33 is late-stage steerable, so a tighter onset
repeat was run at the transition where the local family first appears.

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/source33_object11_y2_s20_t24_coord_target_minus_onset_v1
direction bases: coord_target_minus_bin:413, coord_target_minus_mean:413+414+415+416+418+419
strengths: 0,16,32,64,128,256,512,1024
row_count: 19
realized_direction_patch_count: 16
readout_status_counts: {"ok": 19}
```

Post-hoc reducer:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_causal_patch_probe_support/source33_object11_y2_s20_t24_coord_target_minus_onset_v1
probe_support_row_count: 133
strongest_row_count: 19
```

## Late Ridge Results

Baseline for 24 -> 28:

```text
top: 415/416 tie
p417: 0.078470
rank417: 3
radius4_mass: 0.66248
```

Single-bin directions:

| direction | first target top1 | best sampled p417 | best sampled top | interpretation |
| --- | ---: | ---: | ---: | --- |
| target - 413 | strength 256 | 0.316600 at 512 | 417 | single low-side anchor suppression recenters target |
| target - 415 | never | 0.104050 at 256 | 418 | suppressing current top exposes high-side 418 |
| target - 418 | never | 0.082016 at 16 | 415 | suppressing adapter-delta top mostly leaves 415 |
| target - 416 | never | 0.081114 at 64 | 418 | suppressing near-target prefix anchor drifts high-side |

Local-mean directions:

| direction | first target top1 | best sampled p417 | best sampled top | interpretation |
| --- | ---: | ---: | ---: | --- |
| target - mean(415,418) | strength 256 | 0.293351 at 512 | 417 | suppressing the current-top/high-side pair recenters |
| target - mean(413,414,415,416,418,419) | strength 256 | 0.378010 at 1024 | 417 | full local-attractor suppression is strongest |
| target - mean(413,414,415,416) | never | 0.190963 at 512 | 418 | suppressing low/current side exposes 418 |
| target - mean(415,416) | never | 0.103917 at 256 | 418 | suppressing only current tie is insufficient |

The key asymmetry is that `413` is not the final top bin, but `target - 413`
is the strongest single-bin direction. This suggests the low-side history
anchor contributes to the ridge geometry even when the visible final argmax is
415/416.

## Onset Results

The onset run repeats the most informative directions at 20 -> 24.

| direction | first target top1 | best sampled p417 | best sampled top | high-strength behavior |
| --- | ---: | ---: | ---: | --- |
| target - 413 | strength 256 | 0.176198 at 256 | 417 | at 1024, top leaves local family to 823 |
| target - mean(413,414,415,416,418,419) | strength 256 | 0.189793 at 256 | 417 | at 1024, top leaves local family to 105 |

At moderate strength, target 417 becomes top1 already during local-basin
formation. At very high strength, the patch overshoots out of the local family.
That makes the result more specific than "coordinate direction always helps":
the target is accessible in the onset transition, but the useful scale range is
bounded.

## Mechanistic Update

Source33 is not just late-emission steerable. It is onset-steerable under these
coordinate embedding directions.

The current candidate picture is:

```text
1. The natural source33 trajectory forms a local coordinate basin, but it is
   centered slightly behind the target around 415/416.
2. The target bin 417 is not absent; a small number of coordinate directions
   can make it top1 without changing the object prefix.
3. The most effective single direction is target-minus-413, a low-side prefix
   anchor, not target-minus-current-top 415.
4. Suppressing the wrong subset of anchors exposes 418 instead of 417, so the
   basin behaves like a local ridge rather than an independent one-bin error.
5. Scalar adapter scaling cannot solve this case, but hidden activation
   coordinate directions can, so the failure is not reducible to a global
   token_embeddings_adapter scale problem.
```

Compared with source36, source33 is more single-axis recoverable. Source36
needed multi-bin local-attractor suppression to recenter target 996; source33
can be recentered by `target - 413` alone at both 24 -> 28 and 20 -> 24.

This sharpens the broader hypothesis:

```text
Duplication/near-duplication coordinate errors may arise from local coordinate
ridge resolution under autoregressive history, rather than necessarily from
missing visual evidence or schema instability. Different objects can sit at
different ridge depths: some are single-anchor steerable, while others require
multi-anchor suppression.
```

## Next Questions

The next useful experiments are:

1. Apply the same onset/late split to source36 with local-mean directions at
   20 -> 24, not only single-bin directions, to compare ridge depth directly.
2. Run source123/source145 onset checks to see whether target-helpful adapter
   deltas correspond to earlier or later controllability.
3. For source33, test a history-anchor leakage direction that includes y2=408
   only if we specifically want to distinguish local ridge recentering from
   broader same-description history anchoring.
