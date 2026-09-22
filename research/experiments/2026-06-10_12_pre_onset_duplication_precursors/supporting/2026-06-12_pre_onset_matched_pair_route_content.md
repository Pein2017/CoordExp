# Pre-Onset Matched Pair Route/Content Probe

## Scope

This slice follows the stratified selector result by comparing two same-checkpoint, same-phase pre-onset rows:

- harmful-attractor row: `no_aligner_parent_ckpt3668`, record `33`, row `2`, `post_y1/pre_x2`, desc `wine glass`, target `<|coord_145|>`;
- useful-support row: `no_aligner_parent_ckpt3668`, record `48`, row `2`, `post_y1/pre_x2`, desc `bus`, target `<|coord_899|>`.

The question was whether the layer-17/head-1 duplicate-basin route/content circuit explains the opposite mask signs observed in the residual-patch selector.

## Target Panel

Output root:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48`

Generated target files:

- `target_token_windows.jsonl`
- `target_region_rows.jsonl`
- `target_summary.json`

Counts:

- target token windows: `2`
- target region rows: `14`
- checkpoint count: `1`
- phase: `post_y1/pre_x2`

## Runs

First, duplicate-basin-only route/content patch:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_route_content_patch_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48/target_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48/target_region_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48/route_content_patch_layer17_head1_selected_rows \
  --device auto \
  --attention-layer 17 \
  --attention-head 1 \
  --region-kind duplicate_basin \
  --top-k 8 \
  --patch-directions masked_to_control \
  --patch-sites self_attn_output \
  --patch-components duplicate_basin \
  --effect-kinds total_delta,route_delta_masked_values,value_delta_control_route
```

Then, component-expanded route/content patch:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_route_content_patch_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48/target_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48/target_region_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48/route_content_patch_layer17_head1_components_selected_rows \
  --device auto \
  --attention-layer 17 \
  --attention-head 1 \
  --region-kind duplicate_basin \
  --top-k 8 \
  --patch-directions masked_to_control \
  --patch-sites self_attn_output \
  --patch-components duplicate_basin,visual_non_basin,text_prefix,special_control \
  --effect-kinds total_delta,route_delta_masked_values,value_delta_control_route
```

Both successful route/content runs used the CLI default attention backend, `eager`. An attempted `--attn-implementation auto` run failed because flash attention does not return attention tensors under `output_attentions=True`; this route/content surface therefore requires eager attention.

## Duplicate-Basin Direct Patch

Duplicate-basin-only patching produced zero recovery for both rows:

| record | desc | component | best prob recovery | best rank recovery |
| ---: | --- | --- | ---: | ---: |
| 33 | wine glass | `duplicate_basin` | 0.000000 | 0 |
| 48 | bus | `duplicate_basin` | 0.000000 | 0 |

The companion attention-routing readout explains why:

| record | desc | component | ctrl mass | masked mass | source tokens |
| ---: | --- | --- | ---: | ---: | ---: |
| 33 | wine glass | `duplicate_basin` | 0.00000000 | 0.00000000 | 0 |
| 48 | bus | `duplicate_basin` | 0.00000000 | 0.00000000 | 1 |
| 33 | wine glass | `visual_non_basin` | 0.94934630 | 0.93712217 | 340 |
| 48 | bus | `visual_non_basin` | 0.99862629 | 0.99764150 | 259 |
| 33 | wine glass | `text_prefix` | 0.04945764 | 0.06278287 | 300 |
| 48 | bus | `text_prefix` | 0.00049936 | 0.00042442 | 298 |

At the exact selected row-2 slots, layer 17/head 1 is not directly routing through duplicate-basin source tokens. It routes almost entirely through `visual_non_basin`, with a meaningful `text_prefix` share only for the wine-glass row.

## Component-Expanded Patch

Component-expanded route/content patching found the recoverable layer-17/head-1 effect in `visual_non_basin`, not direct duplicate-basin tokens:

| record | desc | component | effect | masked rank/prob | patched rank/prob | prob recovery | rank recovery |
| ---: | --- | --- | --- | --- | --- | ---: | ---: |
| 48 | bus | `visual_non_basin` | `total_delta` | 3/0.055724 | 1/0.063309 | +0.007585 | +2 |
| 48 | bus | `visual_non_basin` | `value_delta_control_route` | 3/0.055724 | 2/0.062196 | +0.006473 | +1 |
| 48 | bus | `visual_non_basin` | `route_delta_masked_values` | 3/0.055724 | 2/0.061275 | +0.005552 | +1 |
| 33 | wine glass | `visual_non_basin` | `value_delta_control_route` | 6/0.033065 | 2/0.037297 | +0.004232 | +4 |
| 33 | wine glass | `visual_non_basin` | `total_delta` | 6/0.033065 | 2/0.037089 | +0.004024 | +4 |
| 33 | wine glass | `text_prefix` | `value_delta_control_route` | 6/0.033065 | 3/0.036873 | +0.003808 | +3 |
| 33 | wine glass | `text_prefix` | `route_delta_masked_values` | 6/0.033065 | 4/0.035463 | +0.002398 | +2 |

The bus support row is cleanly visual-context dominated. The wine-glass harmful row has both visual-context and text-prefix contributions, with direct duplicate-basin source tokens still inert.

## Interpretation

This refines the current mechanism picture:

1. The selected row-level pre-onset sign split is not explained by direct layer-17/head-1 attention to duplicate-basin visual tokens.
2. Pixel-masking the duplicate-basin bbox changes the model state, but at this head/slot the recoverable causal effect is carried by broader visual context (`visual_non_basin`) and, for the wine-glass row, text-prefix state.
3. The earlier broad record-33 route/content result remains valid for its larger selected panel, but it should not be projected onto the exact row-2 harmful/support matched pair without checking source-token membership.
4. The next mechanistic branch should move later in the decoder residual stack or compare source buckets at layers closer to the residual-patch effect (`20,24,26,27`), rather than assuming layer 17/head 1 duplicate-basin routing is the universal origin.

One reproducibility guardrail: route/content patching must use eager attention because it needs attention tensors. Residual-state probes can use `attn_implementation=auto`, so direct numeric equality between the two surfaces should not be assumed without a backend-aligned replay.

## Verification

Artifacts checked:

- `target_summary.json`
- `target_token_windows.jsonl`
- `target_region_rows.jsonl`
- `route_content_patch_layer17_head1_selected_rows/route_content_patch_rows.jsonl`
- `route_content_patch_layer17_head1_selected_rows/route_content_patch_reduced_report.md`
- `attention_routing_layer17_head1_selected_rows/attention_routing_rows.jsonl`
- `route_content_patch_layer17_head1_components_selected_rows/route_content_patch_rows.jsonl`
- `route_content_patch_layer17_head1_components_selected_rows/route_content_patch_reduced_report.md`

Successful run counts:

- duplicate-basin-only route/content rows: `6`
- attention-routing rows: `8`
- component-expanded route/content rows: `24`

The attempted auto-attention route/content rerun failed as expected with `RuntimeError: model forward did not return attentions`; no interpretation uses that failed artifact.
