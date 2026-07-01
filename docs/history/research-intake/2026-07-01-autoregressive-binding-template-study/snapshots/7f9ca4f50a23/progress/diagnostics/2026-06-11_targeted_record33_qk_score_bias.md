# Targeted Record-33 Q/K and Score-Bias Probe

Date: 2026-06-11

## Scope

This slice follows the targeted route/content patch for the cross-phase top
case. The previous targeted patch showed that `none_latest_ckpt32` record 33 at
`post_y1/pre_x2` is route dominated:

- `total_delta` mean probability recovery: `0.018371`
- `route_delta_masked_values` mean probability recovery: `0.018168`
- `value_delta_control_route` mean probability recovery: `0.000121`

This run asks where that route delta originates and whether an attention-score
bias proxy can causally recover part of the next-coordinate basin.

## Artifact Handles

Target panel root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel
```

Q/K route-origin output:

```text
qk_route_origin_layer17_head1/qk_route_origin_rows.jsonl
qk_route_origin_layer17_head1/phase4_qk_route_origin_summary.json
```

Attention-score-bias causal proxy output:

```text
attention_score_bias_patch_layer17_head1/attention_score_bias_patch_rows.jsonl
attention_score_bias_patch_layer17_head1/phase4_attention_score_bias_patch_summary.json
```

Joint reduced synthesis:

```text
targeted_qk_score_bias_synthesis_layer17_head1/targeted_qk_score_bias_synthesis_summary.json
targeted_qk_score_bias_synthesis_layer17_head1/targeted_qk_score_bias_synthesis_report.md
```

Run scale:

- target cases: `5`
- Q/K rows: `82`
- score-bias rows: `164`
- Q/K groups: `7`
- score-bias groups: `14`
- layer/head: `17/1`
- component: `duplicate_basin`
- GPUs used: `CUDA_VISIBLE_DEVICES=0` for Q/K,
  `CUDA_VISIBLE_DEVICES=1` for score bias

## Commands

Q/K route-origin:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_qk_route_origin_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/target_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/target_region_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/qk_route_origin_layer17_head1 \
  --device auto \
  --attention-layer 17 \
  --attention-head 1 \
  --region-kind duplicate_basin \
  --patch-components duplicate_basin
```

Attention-score-bias patch:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_attention_score_bias_patch_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/target_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/target_region_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/attention_score_bias_patch_layer17_head1 \
  --device auto \
  --attention-layer 17 \
  --attention-head 1 \
  --region-kind duplicate_basin \
  --top-k 8 \
  --patch-directions masked_to_control \
  --patch-components duplicate_basin \
  --score-bias-modes scalar_logsumexp,per_source_delta
```

## Q/K Origin Result

Grouped Q/K score-origin readout:

| rank | checkpoint | record | phase | rows | attention mass delta | score mean delta | query delta masked keys | key delta control query |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 1 | `none_latest_ckpt32` | 33 | `post_y1/pre_x2` | 12 | 0.415218 | 4.15748 | -1.45842 | 5.61591 |
| 2 | `aux_latest_ckpt32` | 33 | `post_y1/pre_x2` | 12 | 0.300586 | 1.83017 | -2.10516 | 3.93533 |
| 3 | `aux_latest_ckpt32` | 33 | `box_start/pre_x1` | 12 | 0.219320 | 5.22539 | 0.814704 | 4.41069 |
| 4 | `none_latest_ckpt32` | 33 | `box_start/pre_x1` | 12 | 0.218008 | 5.81510 | 0.423987 | 5.39111 |
| 5 | `aligner_parent_ckpt1824` | 54 | `post_y1/pre_x2` | 11 | 0.094638 | 0.622950 | 0.102239 | 0.520711 |
| 6 | `no_aligner_parent_ckpt3668` | 48 | `post_y1/pre_x2` | 12 | 0.085237 | 0.249214 | -0.036629 | 0.285843 |
| 7 | `no_aligner_parent_ckpt3668` | 36 | `post_y1/pre_x2` | 11 | 0.040407 | 0.649447 | -1.18007 | 1.82952 |

For the main record-33 case, the score-space origin is again key-side:
`key_delta_control_query_score_mean=5.61591` while
`query_delta_masked_keys_score_mean=-1.45842`. This is stronger than the broad
panel average and keeps pointing toward duplicate-basin visual key state as the
proximal routing source.

## Score-Bias Proxy Result

Grouped score-bias causal proxy:

| rank | checkpoint | record | phase | mode | rows | attention mass recovery | probability recovery | rank recovery | abs-error recovery | mass16 recovery |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `none_latest_ckpt32` | 33 | `post_y1/pre_x2` | `per_source_delta` | 12 | 0.292988 | 0.010221 | 15.3333 | -6.29252 | 0.186307 |
| 2 | `none_latest_ckpt32` | 33 | `post_y1/pre_x2` | `scalar_logsumexp` | 12 | 0.292988 | 0.010221 | 15.3333 | -6.29252 | 0.186307 |
| 3 | `aux_latest_ckpt32` | 33 | `box_start/pre_x1` | `per_source_delta` | 12 | 0.589144 | 0.004537 | 265.5000 | -44.8661 | 0.108065 |
| 5 | `none_latest_ckpt32` | 33 | `box_start/pre_x1` | `per_source_delta` | 12 | 0.477944 | 0.002827 | 215.4167 | -30.3502 | 0.071268 |
| 13 | `aux_latest_ckpt32` | 33 | `post_y1/pre_x2` | `per_source_delta` | 12 | 0.288827 | 0.000514 | -2.7500 | -1.02412 | 0.016460 |

For the main case, score-bias patching recovers about `0.010221` probability,
compared with the route/content `route_delta_masked_values` recovery
`0.018168`. That is a substantial but incomplete causal proxy: manipulating
attention scores toward the control route restores a large part of the
coordinate-basin repair, but not all of it.

The identical scalar and per-source numbers are expected for this targeted
component where the duplicate-basin bucket often has one effective source token
per anchor after the region materialization.

## Interpretation

The target-panel mechanism is now narrowed further:

- The record-33 duplicate-basin route delta is strongest exactly at
  `post_y1/pre_x2`, the cross-phase top hit.
- Q/K decomposition says the route delta is dominated by visual-key-side state,
  not by a query becoming more compatible with masked keys.
- Attention-score-bias patching is causally sufficient for a large fraction of
  the probability/radius recovery, so the score-space route is not merely a
  correlational readout.
- The incomplete recovery leaves room for lower-level K-state patching,
  residual normalization effects, or coordinate-slot basin amplification after
  attention output.

## Next Deterministic Step

The next highest-value probe is a direct key-state intervention for the
duplicate-basin source bucket at layer 17 head 1 on this same target panel. If
the current model hooks make direct K-state patching too invasive, keep the
score-bias proxy as the validated causal score target and first inspect
post-RoPE key-state deltas for coordinate-token locality and smoothness around
the implicated `<|coord_*|>` bins.

## Verification

- Q/K route-origin run completed with summary:
  `qk_route_origin_row_count=82`, `checkpoint_count=3`,
  `replay_case_count=5`.
- Score-bias patch run completed with summary:
  `attention_score_bias_patch_row_count=164`, `checkpoint_count=3`,
  `replay_case_count=5`.
- Joint synthesis output was materialized from those two JSONL files:
  `qk_row_count=82`, `score_bias_row_count=164`, `qk_group_count=7`,
  `score_bias_group_count=14`.
