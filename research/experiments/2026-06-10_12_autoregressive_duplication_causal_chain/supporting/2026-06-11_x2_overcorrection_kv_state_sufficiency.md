# X2 Overcorrection Key/Value State Sufficiency

Date: 2026-06-11

## Scope

Ran duplicate-basin key/value content patches on the validated x2 overcorrection
candidate panel. This follows the score-bias sufficiency boundary, where
restoring duplicate-basin attention scores recovered route mass but did not
usually recover the damaging/control coordinate top1.

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

Key/value patch root:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_kv_state_patch_layer17_head1_duplicate_basin`

Summary root:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_kv_state_patch_layer17_head1_duplicate_basin/sufficiency_summary`

Files:

- `attention_score_bias_patch_rows.jsonl`
- `phase4_attention_score_bias_patch_summary.json`
- `sufficiency_summary/score_bias_sufficiency_rows.jsonl`
- `sufficiency_summary/phase4_score_bias_sufficiency_summary.json`
- `sufficiency_summary/phase4_score_bias_sufficiency_summary.md`

Counts:

- Replay cases: `2`
- Region rows: `14`
- Patch rows: `21`
- Patch component: `duplicate_basin`
- Patch direction: `masked_to_control`
- Modes: `key_state_control`, `value_state_control`, `key_value_state_control`

## Main Readout

Auxiliary checkpoint means:

| mode | n | attention recovery | recovery fraction | prob recovery | rank recovery | top1-distance recovery | top1 outcomes |
|---|---:|---:|---:|---:|---:|---:|---|
| `key_state_control` | 6 | 0.440243 | 1.214354 | -0.000755 | -4.333333 | 2.833333 | `match=2, partial=1, unchanged=3` |
| `key_value_state_control` | 6 | 0.440243 | 1.214354 | -0.003773 | -8.500000 | 4.166667 | `match=3, partial=1, unchanged=1, away=1` |
| `value_state_control` | 6 | 0.000000 | 0.000000 | -0.000297 | 0.500000 | 0.333333 | `unchanged=4, away=2` |

Score-only baseline on the same candidates:

| mode | n | attention recovery | recovery fraction | top1 outcomes |
|---|---:|---:|---:|---|
| `scalar_logsumexp` | 6 | 0.373023 | 0.887315 | `match=1, partial=1, unchanged=4` |
| `per_source_delta` | 6 | 0.373023 | 0.887315 | `match=1, partial=1, unchanged=4` |

Key row split:

| row | mode | masked-target | patched-target | control-target | outcome |
|---:|---|---:|---:|---:|---|
| 20 | key | 0 | -7 | -14 | partial toward control |
| 20 | key+value | 0 | -7 | -14 | partial toward control |
| 21 | key | 0 | -12 | -12 | patched matches control |
| 21 | key+value | 0 | -12 | -12 | patched matches control |
| 23 | key | 7 | -5 | -5 | patched matches control |
| 23 | key+value | 7 | -5 | -5 | patched matches control |
| 28 | key | -5 | -5 | -6 | unchanged |
| 28 | key+value | -5 | -6 | -6 | patched matches control |

## Interpretation

This strengthens the mechanism picture beyond the score-only result:

- Key-state restoration is a stronger causal handle than score-only restoration.
- Key+value restoration is strongest on top1 outcomes, moving three of six
  auxiliary x2 rows to the control coordinate and one more partially toward it.
- Value-only restoration does not recover attention mass and mostly fails,
  suggesting the value content alone is not the primary route driver.
- Row 20 remains a key counterexample: key/key+value state patching moves it
  halfway toward the control coordinate but does not fully reproduce the control
  top1. That suggests either multi-source route interaction or downstream
  residual dynamics beyond the duplicate-basin token set.

The most plausible current mechanism is therefore:

1. The x2 overcorrection surface is strongly tied to duplicate-basin route
   dependence at head `17/1`.
2. Score restoration is enough to restore attention mass but usually not enough
   to restore the coordinate decision.
3. Key-state restoration, especially key+value together, is substantially more
   sufficient for the damaging/control coordinate basin.
4. Full sufficiency probably needs route content plus broader residual or
   multi-component source context for the remaining counterexamples.

## Reproduction

Key/value state patch:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python scripts/analysis/run_autoregressive_duplication_phase4_attention_score_bias_patch_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_region_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_kv_state_patch_layer17_head1_duplicate_basin \
  --device auto \
  --torch-dtype bfloat16 \
  --attn-implementation eager \
  --prompt-ordering random \
  --attention-layer 17 \
  --attention-head 1 \
  --region-kind duplicate_basin \
  --top-k 8 \
  --patch-directions masked_to_control \
  --patch-components duplicate_basin \
  --score-bias-modes key_state_control,value_state_control,key_value_state_control
```

Sufficiency summary:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/analysis/run_autoregressive_duplication_phase4_score_bias_sufficiency_summary.py \
  --score-bias-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_kv_state_patch_layer17_head1_duplicate_basin/attention_score_bias_patch_rows.jsonl \
  --candidate-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_probe_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_kv_state_patch_layer17_head1_duplicate_basin/sufficiency_summary
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
