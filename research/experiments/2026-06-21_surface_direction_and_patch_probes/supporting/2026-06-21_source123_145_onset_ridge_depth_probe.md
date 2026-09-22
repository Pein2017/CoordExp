---
doc_id: progress.diagnostics.source123_145_onset_ridge_depth_probe
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: tiny-two-state-source123-source145-y2-model-backed-activation-patch
domain: autoregressive-binding-template-ablation
updated: 2026-06-21
branch: codex/autoregressive-binding-template-study
---

# Source123/145 Onset Ridge-Depth Probe

## Scope

This note adds `20 -> 24` onset activation-patch checks for the two y2
comparator states that already had `24 -> 28` single-bin causal evidence:

```text
source123/object5/y2 target=270
source145/object20/y2 target=146
```

Together with the source33 and source36 onset notes, this gives a first tiny
four-state ridge-depth panel. This is model-backed activation-patch evidence on
selected states, not population validation or natural-decode evidence.

## Runs

All runs used:

```text
pair_config: configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
family: desc_first
intervention_arm: natural
target_next_kind: coord
stop_reason: object_step_role
patch_alphas: 0
training_ran: false
model_weight_update_ran: false
```

Source123 onset:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/source123_object5_y2_s20_t24_coord_target_minus_onset_v1
source rows: /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source123_object5_y2_panel_v1/selected_rows.jsonl
direction bases: coord_target_minus_bin:188, coord_target_minus_bin:259, coord_target_minus_mean:188+259
strengths: 0,16,32,64,128,256,512
probe_coord_bins: 188,254,259,266,270,274
row_count: 24
realized_direction_patch_count: 21
readout_status_counts: {"ok": 24}
```

Source145 onset:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/source145_object20_y2_s20_t24_coord_target_minus_onset_v1
source rows: /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source145_object20_y2_panel_v1/selected_rows.jsonl
direction bases: coord_target_minus_bin:158, coord_target_minus_bin:154, coord_target_minus_mean:154+158
strengths: 0,16,32,64,128,256,512
probe_coord_bins: 142,144,145,146,150,154,158
row_count: 24
realized_direction_patch_count: 21
readout_status_counts: {"ok": 24}
```

Post-hoc probe-support reducers:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_causal_patch_probe_support/source123_object5_y2_s20_t24_coord_target_minus_onset_v1
probe_support_row_count: 144
strongest_row_count: 24

/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_causal_patch_probe_support/source145_object20_y2_s20_t24_coord_target_minus_onset_v1
probe_support_row_count: 168
strongest_row_count: 24
```

## Source123 Result

Target: `270`

| direction | first target top1 | best sampled target prob | best sampled top | interpretation |
| --- | ---: | ---: | ---: | --- |
| target - 188 | strength 64 | 0.303104 at 512 | 270 | strong early steerability away from base/effective top |
| target - 259 | strength 64 | 0.200626 at 512 | 270 | visible layer-top antagonist also works |
| target - mean(188,259) | strength 64 | 0.310896 at 512 | 270 | combined direction is slightly strongest |

Source123 is onset-easy in the tested coordinate basis. Even the first sampled
nonzero regime that succeeds is strength 64, and all tested directions can make
the target top1. This matches the adapter-scale finding that the adapter delta
is target-helpful, but the onset patch shows the target is already controllable
before the final layer.

## Source145 Result

Target: `146`

| direction | first target top1 | best sampled target prob | best sampled top | interpretation |
| --- | ---: | ---: | ---: | --- |
| target - 154 | never | 0.229949 at 512 | 145 | suppressing visible layer-top side does not select target |
| target - 158 | strength 256 | 0.458186 at 512 | 146 | suppressing base/effective surface top recovers target |
| target - mean(154,158) | strength 256 | 0.412720 at 512 | 146 | combined direction works but is weaker than 158 alone |

Source145 is onset-steerable, but only through the right antagonist. The visible
layer top `154` did not select the target under these tested directions;
`158`, the base/effective surface top, did. This mirrors the late source145
result and supports the narrower claim that visible top1 and the best tested
steering antagonist can differ.

## Four-State Ridge-Depth Panel

| state | target | onset single-bin success | onset local-mean success | ridge-depth read |
| --- | ---: | --- | --- | --- |
| source123/object5/y2 | 270 | yes, 188 and 259 at strength 64 | yes, mean(188,259) at strength 64 | broad/easy target controllability |
| source145/object20/y2 | 146 | yes, 158 at strength 256; 154 fails | yes, mean(154,158) at strength 256 | specific antagonist; visible top is not enough |
| source33/object11/y2 | 417 | yes, 413 at strength 256 | yes, full local mean at strength 256 | low-side history-anchor steerable |
| source36/object14/y2 | 996 | no tested single-bin onset success | yes, mean(999,995) at strength 224 | multi-anchor/local-simplex steerable |

This panel updates the working taxonomy:

```text
1. Easy/broad steerability: source123.
2. Specific antagonist steerability: source145.
3. Single history-anchor steerability: source33.
4. Multi-anchor ridge steerability: source36.
```

The shared theme is that all four target coordinates are patch-selectable under
some constructed coordinate-token activation direction at or before the
local-basin formation transition. The difference in this selected-state panel
is how many and which coordinate-history anchors must be counteracted before
the target becomes the selected coordinate token.

## Mechanistic Update

The current evidence favors a coordinate-ridge resolution model:

```text
target-coordinate selection is often activation-patch accessible,
but emission depends on a local ridge shaped by autoregressive coordinate
history, adapter surface geometry, and the particular antagonist anchors.
```

This does not rule out visual perception failures elsewhere, especially for
false negatives. For these selected y2 duplication/near-duplication states, it
says only that a pure "target coordinate is unreachable" account is too shallow:
targeted hidden-space coordinate directions can select the target without
changing the object prefix.

## Next Questions

1. Build a compact machine-readable ridge-depth table from the existing causal
   patch artifacts so future cases can be appended without hand tables.
2. Extend the panel to false-negative states: test whether missing objects are
   visually absent or merely need language-side/object-prefix guidance.
3. Link the ridge-depth categories back to attention/value routes: determine
   which heads carry the antagonist-anchor directions before y2 emission.
