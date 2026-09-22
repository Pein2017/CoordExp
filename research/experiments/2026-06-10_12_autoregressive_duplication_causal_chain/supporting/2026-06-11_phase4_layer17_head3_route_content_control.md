# Phase 4 Layer17/Head3 Route/Content Control

Date: 2026-06-11

Scope: all-shard route/content patch pass for layer17/head3 after the
candidate-head pilot suggested head 3 was the only alternate worth a second
look. This note compares head 3 against the existing layer17/head1 evidence and
decides whether to branch into paired manifests and cross-row swaps for head 3.

## Artifacts

All-shard head3 route/content patch root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_route_content_patch_layer17_head3_duplicate_allshards_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_route_content_patch_layer17_head3_duplicate_allshards_summary.json
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_route_content_patch_layer17_head3_duplicate_allshards_report.md
```

Run shape:

- `attention_layer=17`
- `attention_head=3`
- `patch_site=self_attn_output`
- `patch_component=duplicate_basin`
- `patch_directions=control_to_masked,masked_to_control`
- `effect_kinds=total_delta,route_delta_masked_values,value_delta_control_route`
- `row_count=7536`
- shard row counts: `1128, 1008, 1008, 1104, 1032, 888, 672, 696`

Comparison baseline:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_route_content_patch_layer17_head1_route_value_visual_triplet_top4_allshards_rows.jsonl
```

## Main Comparison

The table uses `control_to_masked`, `duplicate_basin`, and all-shard means.

### none_latest_ckpt32 / post_y1/pre_x2

| head | effect | n | prob delta | r4 recovery | projected L2 |
|---:|---|---:|---:|---:|---:|
| 1 | route_delta_masked_values | 62 | -0.002909 | +0.010580 | 22.688698 |
| 1 | total_delta | 62 | -0.003199 | +0.008359 | 25.453282 |
| 1 | value_delta_control_route | 62 | -0.000266 | +0.028513 | 12.333199 |
| 3 | route_delta_masked_values | 62 | +0.000124 | +0.031078 | 1.754803 |
| 3 | total_delta | 62 | +0.000014 | +0.030386 | 1.968613 |
| 3 | value_delta_control_route | 62 | +0.000338 | +0.031434 | 1.897961 |

### aux_latest_ckpt32 / post_x1/pre_y1

| head | effect | n | prob delta | r4 recovery | projected L2 |
|---:|---|---:|---:|---:|---:|
| 1 | route_delta_masked_values | 81 | -0.000067 | +0.021611 | 1.084693 |
| 1 | total_delta | 81 | +0.000106 | +0.021929 | 1.024961 |
| 1 | value_delta_control_route | 81 | +0.000068 | +0.021714 | 0.455300 |
| 3 | route_delta_masked_values | 81 | +0.000168 | +0.021683 | 1.670478 |
| 3 | total_delta | 81 | +0.000205 | +0.022813 | 1.901903 |
| 3 | value_delta_control_route | 81 | +0.000108 | +0.022293 | 1.646655 |

### no_aligner_parent_ckpt3668 / post_y1/pre_x2

| head | effect | n | prob delta | r4 recovery | projected L2 |
|---:|---|---:|---:|---:|---:|
| 1 | route_delta_masked_values | 113 | -0.000651 | +0.012245 | 9.151916 |
| 1 | total_delta | 113 | -0.000797 | +0.010998 | 10.805283 |
| 1 | value_delta_control_route | 113 | -0.000570 | +0.014253 | 5.019570 |
| 3 | route_delta_masked_values | 113 | -0.000359 | +0.017242 | 0.822355 |
| 3 | total_delta | 113 | -0.000216 | +0.017656 | 1.052832 |
| 3 | value_delta_control_route | 113 | -0.000207 | +0.016754 | 0.694758 |

## Interpretation

Head 3 is a useful alternate-head control, but it does not currently justify a
new paired-manifest/cross-row branch.

- For the key `none_latest_ckpt32/post_y1/pre_x2` slice, head 3 has much smaller
  projected deltas than head 1 and lacks the large route-vector behavior that
  made head 1 mechanistically sharp.
- Head 3 gives smoother radius-basin recovery with near-zero exact target
  probability movement. That makes it look like a low-amplitude basin smoothing
  or residual/value-like effect, not a specific route vector that should carry
  cross-row coordinate identity.
- In the aux slice, head 3 is close to head 1 and mildly larger in L2, but both
  are low-amplitude there. This is not enough to displace head 1 as the main
  causal route/content path.

Decision: keep layer17/head1 as the main route/content mechanism candidate.
Treat layer17/head3 as an all-shard negative/control reference. Do not spend
the next GPU wave on head3 paired manifests unless later artifacts specifically
need an alternate-head control.

## Verification

GPU execution completed on shards `00` through `07` using available GPUs
`0-3` in two waves. GPUs `4-7` had resident jobs and were intentionally not
used.

Combined artifact verification:

```text
7536 /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260610-073920/phase4_route_content_patch_layer17_head3_duplicate_allshards_rows.jsonl
```

Shard row counts:

```text
00 1128
01 1008
02 1008
03 1104
04 1032
05 888
06 672
07 696
```
