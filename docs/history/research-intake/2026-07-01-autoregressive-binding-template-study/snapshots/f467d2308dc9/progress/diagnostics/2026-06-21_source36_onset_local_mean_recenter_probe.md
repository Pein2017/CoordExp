---
doc_id: progress.diagnostics.source36_onset_local_mean_recenter_probe
layer: progress
doc_type: diagnostic-findings
status: active-evidence
evidence_scope: tiny-one-state-source36-y2-model-backed-activation-patch
domain: autoregressive-binding-template-ablation
updated: 2026-06-21
branch: codex/autoregressive-binding-template-study
---

# Source36 Onset Local-Mean Recenter Probe

## Scope

This note fills the missing source36 comparator after the source33 onset result:

```text
progress/diagnostics/2026-06-21_source33_coordinate_causal_recenter_probe.md
```

Prior source36 evidence showed:

```text
20 -> 24 single-bin directions: p996 can rise, but top remains 995.
24 -> 28 local-mean directions: multi-anchor suppression can recenter target 996.
```

The open question was whether source36 is also recoverable during onset if the
patch uses multi-anchor local-mean directions instead of single-bin directions.
This is one-state model-backed activation-patch evidence, not population or
natural-decode evidence.

## Runs

All runs used:

```text
pair_config: configs/analysis/autoregressive_binding_template_ablation/ckpt928_pair_val200.yaml
source rows: /data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/coordinate_threshold_hidden_state_probe/source36_object14_y2_panel_v1/selected_rows.jsonl
family: desc_first
source_line_idx: 36
object_idx: 14
slot: y2
target coord: 996
intervention_arm: natural
target_next_kind: coord
stop_reason: object_step_role
patch_alphas: 0
probe_coord_bins: 994,995,996,997,998,999
training_ran: false
model_weight_update_ran: false
```

Coarse local-mean onset sweep:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/source36_object14_y2_s20_t24_coord_target_minus_mean_local_bins_v1
direction bases: coord_target_minus_mean:999+998+997+995+994, coord_target_minus_mean:999+998+997, coord_target_minus_mean:995+994, coord_target_minus_mean:999+995
strengths: 0,16,32,64,128,256,512,1024
row_count: 35
realized_direction_patch_count: 32
readout_status_counts: {"ok": 35}
```

Threshold local-mean onset sweep:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_hidden_causal_activation_patch_probe/source36_object14_y2_s20_t24_coord_target_minus_mean_local_threshold_v1
direction bases: coord_target_minus_mean:999+995, coord_target_minus_mean:999+998+997+995+994
strengths: 128,160,192,224,256,320,384,448,512
row_count: 21
realized_direction_patch_count: 18
readout_status_counts: {"ok": 21}
```

Post-hoc probe-support reducers:

```text
/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_causal_patch_probe_support/source36_object14_y2_s20_t24_coord_target_minus_mean_local_bins_v1
probe_support_row_count: 210
strongest_row_count: 35

/data/CoordExp/outputs/analysis/autoregressive_binding_template_ablation/phase4_realized_behavior_bridge/trajectory_causal_patch_probe_support/source36_object14_y2_s20_t24_coord_target_minus_mean_local_threshold_v1
probe_support_row_count: 126
strongest_row_count: 21
```

## Coarse Result

| direction | first target top1 | best p996 row | p996 at max strength | top at max strength |
| --- | --- | --- | ---: | ---: |
| mean(995,994) | never | strength 128, top 999, p996 0.058482 | 0.001461 | 498 |
| mean(999,998,997) | never | strength 256, top 995, p996 0.151376 | 0.002051 | 964 |
| mean(999,995) | strength 256 | strength 512, top 996, p996 0.367505 | 0.147822 | 965 |
| mean(999,998,997,995,994) | strength 512 | strength 512, top 996, p996 0.369526 | 0.057150 | 964 |

The same qualitative split from late source36 recurs at onset:

```text
low-side-only or ceiling-side-only suppression is insufficient;
joint suppression of ceiling and 995-family anchors can recenter target 996.
```

Very high strength overshoots out of the local family, so the useful range is
bounded.

## Threshold Result

| direction | strength | top1 | target rank | p996 | p995 | p997 | p999 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean(999,995) | 192 | 997 | 2 | 0.176335 | 0.155615 | 0.199814 | 0.106952 |
| mean(999,995) | 224 | 996 | 1 | 0.211424 | 0.186581 | 0.211424 | 0.068639 |
| mean(999,995) | 256 | 996 | 1 | 0.219704 | 0.193888 | 0.219704 | 0.049023 |
| mean(999,995) | 320 | 996 | 1 | 0.300213 | 0.182088 | 0.206333 | 0.016937 |
| mean(999,995) | 384 | 996 | 1 | 0.314393 | 0.190689 | 0.216079 | 0.005082 |
| mean(999,995) | 448 | 996 | 1 | 0.352237 | 0.188539 | 0.166385 | 0.001631 |
| mean(999,995) | 512 | 996 | 1 | 0.367505 | 0.173597 | 0.135198 | 0.000430 |
| mean(999,998,997,995,994) | 192 | 997 | 2 | 0.192623 | 0.192623 | 0.218271 | 0.103104 |
| mean(999,998,997,995,994) | 224 | 995 | 1 | 0.223459 | 0.223459 | 0.223459 | 0.064022 |
| mean(999,998,997,995,994) | 256 | 995 | 1 | 0.237170 | 0.237170 | 0.209302 | 0.041214 |
| mean(999,998,997,995,994) | 320 | 996 | 1 | 0.320317 | 0.249463 | 0.194282 | 0.014074 |
| mean(999,998,997,995,994) | 384 | 996 | 1 | 0.355598 | 0.244399 | 0.167973 | 0.004476 |
| mean(999,998,997,995,994) | 448 | 996 | 1 | 0.366101 | 0.251617 | 0.134681 | 0.001028 |
| mean(999,998,997,995,994) | 512 | 996 | 1 | 0.369526 | 0.224129 | 0.093431 | 0.000232 |

Threshold summary:

```text
mean(999,995): first target top1 at strength 224
mean(999,998,997,995,994): first target top1 at strength 320
```

The apparent rank-1 rows where the selected top1 is still `995` are tie/ridge
rows: target 996 has rank 1, but is not necessarily the selected argmax under
the tie-breaking/top-id ordering. At strength 224 for the full local-mean
direction, 995/996/997 are all tied in the displayed local probabilities.

## Mechanistic Update

Source36 is not onset-unsteerable. It is onset local-mean steerable.

The stronger source33/source36 contrast is:

```text
source33:
  target - 413 alone can recenter target 417 at both 20 -> 24 and 24 -> 28.

source36:
  single-bin directions fail at both 20 -> 24 and 24 -> 28.
  multi-anchor mean directions recenter target 996 at both 20 -> 24 and 24 -> 28.
```

So the current ridge-depth axis is not simply "early vs late." It is:

```text
single-anchor steerable  <->  multi-anchor/local-simplex steerable
```

This is a more precise candidate mechanism for coordinate duplication basins:
the model may have enough hidden geometric evidence for the target coordinate,
but the autoregressive coordinate-history ridge can require suppressing a
specific set of neighboring anchors before the target becomes the selected
coord token.

## Next Questions

1. Run source123/source145 onset checks to test whether adapter-helpful and
   adapter-hostile cases differ by ridge depth or by earlier target evidence.
2. Build a small cross-case ridge-depth table with:
   single-bin success, local-mean success, onset success, late success,
   first-success strength, best target probability, and overshoot behavior.
3. If source36/source33 remain representative, connect ridge depth back to
   textual history anchors and attention/value routes rather than only
   coordinate-surface directions.
