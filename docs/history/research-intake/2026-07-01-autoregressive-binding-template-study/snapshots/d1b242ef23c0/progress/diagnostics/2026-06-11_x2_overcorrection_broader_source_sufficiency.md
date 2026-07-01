# X2 Overcorrection Broader Source Sufficiency

Date: 2026-06-11

## Scope

Ran a broader-source key+value state patch on the validated x2 overcorrection
candidate panel to test whether the duplicate-basin counterexample row 20 needs
larger route context at head `17/1`.

GPU use was restricted to `CUDA_VISIBLE_DEVICES=0`, within the available device
set `0,1,2,3`.

## Inputs

- Candidate token windows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_token_windows.jsonl`
- Candidate region rows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_region_rows.jsonl`
- Candidate rows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_probe_rows.jsonl`

## Outputs

Broader-source patch root:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_kv_state_patch_layer17_head1_broader_sources`

Summary root:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_kv_state_patch_layer17_head1_broader_sources/sufficiency_summary`

Counts:

- Replay cases: `2`
- Region rows: `14`
- Patch rows: `21`
- Patch direction: `masked_to_control`
- Mode: `key_value_state_control`
- Components: `visual_near_ring`, `visual_non_basin`, `whole_head`

## Main Readout

Auxiliary component means:

| component | n | attention recovery | recovery fraction | prob recovery | rank recovery | top1-distance recovery | top1 outcomes |
|---|---:|---:|---:|---:|---:|---:|---|
| `visual_near_ring` | 6 | -0.113603 | 0.464081 | -0.001451 | 0.500000 | 1.833333 | `away=3, partial=1, unchanged=2` |
| `visual_non_basin` | 6 | -0.055951 | 1.546510 | -0.002119 | 0.500000 | 0.166667 | `away=1, partial=1, unchanged=4` |
| `whole_head` | 6 | 0.000467 | -0.577290 | -0.006324 | -12.333333 | 4.000000 | `match=2, partial=1, unchanged=2, away=1` |

Duplicate-basin key+value baseline from the prior run:

| component | n | attention recovery | recovery fraction | prob recovery | rank recovery | top1-distance recovery | top1 outcomes |
|---|---:|---:|---:|---:|---:|---:|---|
| `duplicate_basin` | 6 | 0.440243 | 1.214354 | -0.003773 | -8.500000 | 4.166667 | `match=3, partial=1, unchanged=1, away=1` |

Key row comparison:

| row | component | masked-target | patched-target | control-target | outcome |
|---:|---|---:|---:|---:|---|
| 20 | duplicate_basin | 0 | -7 | -14 | partial toward control |
| 20 | visual_near_ring | 0 | 5 | -14 | away from control |
| 20 | visual_non_basin | 0 | 0 | -14 | unchanged |
| 20 | whole_head | 0 | -7 | -14 | partial toward control |
| 21 | duplicate_basin | 0 | -12 | -12 | patched matches control |
| 21 | whole_head | 0 | -12 | -12 | patched matches control |
| 23 | duplicate_basin | 7 | -5 | -5 | patched matches control |
| 23 | whole_head | 7 | -5 | -5 | patched matches control |
| 28 | duplicate_basin | -5 | -6 | -6 | patched matches control |
| 28 | whole_head | -5 | -5 | -6 | unchanged |

## Interpretation

Broader source patching does not explain the remaining row 20 gap. It sharpens
the locality of the causal source:

- Duplicate-basin key+value remains the strongest route-content patch.
- Whole-head key+value does not outperform duplicate basin and loses the row 28
  full recovery.
- Visual near-ring and visual non-basin patches are mostly non-sufficient and
  can push top1 away from the control coordinate.
- Row 20 remains only partially recovered even under whole-head key+value patch,
  so its missing piece is unlikely to be a simple broader-source content
  inclusion at the same head.

This points the next step toward downstream residual or multi-layer dynamics:
duplicate-basin key/content at head `17/1` is causal and localized, but full row
20 overcorrection likely requires either additional layers/heads or residual
state accumulation beyond this single-head source set.

## Reproduction

Broader-source key+value patch:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python scripts/analysis/run_autoregressive_duplication_phase4_attention_score_bias_patch_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_region_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_kv_state_patch_layer17_head1_broader_sources \
  --device auto \
  --torch-dtype bfloat16 \
  --attn-implementation eager \
  --prompt-ordering random \
  --attention-layer 17 \
  --attention-head 1 \
  --region-kind duplicate_basin \
  --top-k 8 \
  --patch-directions masked_to_control \
  --patch-components visual_near_ring,visual_non_basin,whole_head \
  --score-bias-modes key_value_state_control
```

Sufficiency summary:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/analysis/run_autoregressive_duplication_phase4_score_bias_sufficiency_summary.py \
  --score-bias-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_kv_state_patch_layer17_head1_broader_sources/attention_score_bias_patch_rows.jsonl \
  --candidate-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_probe_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_kv_state_patch_layer17_head1_broader_sources/sufficiency_summary
```

Verification:

```bash
python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_score_bias_sufficiency_summary.py \
  scripts/analysis/run_autoregressive_duplication_phase4_score_bias_sufficiency_summary.py \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_score_bias_sufficiency_summary.py
```

The direct harness for
`tests/analysis/autoregressive_duplication_mechanism/test_phase4_score_bias_sufficiency_summary.py`
passed.
