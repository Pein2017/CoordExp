# X2 Overcorrection Score-Bias Sufficiency

Date: 2026-06-11

## Scope

Ran a targeted causal intervention on the validated x2 overcorrection candidate
panel: restore duplicate-basin attention score bias from control into the
duplicate-basin-masked run at route head `17/1`, then measure coordinate-logit
movement.

GPU use was restricted to `CUDA_VISIBLE_DEVICES=0`, within the available device
set `0,1,2,3`.

This is a sufficiency probe for one mechanism component: duplicate-basin route
score mass. It is not a full key/value or residual-state patch.

## Inputs

- Candidate token windows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_token_windows.jsonl`
- Candidate region rows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_region_rows.jsonl`
- Candidate rows:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_probe_rows.jsonl`

## Outputs

Score-bias patch root:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_score_bias_patch_layer17_head1_duplicate_basin`

Summary root:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_score_bias_patch_layer17_head1_duplicate_basin/sufficiency_summary`

Files:

- `attention_score_bias_patch_rows.jsonl`
- `phase4_attention_score_bias_patch_summary.json`
- `sufficiency_summary/score_bias_sufficiency_rows.jsonl`
- `sufficiency_summary/phase4_score_bias_sufficiency_summary.json`
- `sufficiency_summary/phase4_score_bias_sufficiency_summary.md`

Counts:

- Replay cases: `2`
- Region rows: `14`
- Score-bias patch rows: `14`
- Patch component: `duplicate_basin`
- Patch direction: `masked_to_control`
- Score-bias modes: `scalar_logsumexp`, `per_source_delta`

## Main Readout

For the six auxiliary x2 candidates, duplicate-basin score-bias patching
recovers attention mass strongly but coordinate top1 movement only weakly:

| group | n | attention recovery | recovery fraction | prob recovery | rank recovery | top1-distance recovery | outcomes |
|---|---:|---:|---:|---:|---:|---:|---|
| `aux_latest_ckpt32|scalar_logsumexp` | 6 | 0.373023 | 0.887315 | -0.001237 | -5.000000 | 1.166667 | `unchanged=4, match_control=1, partial=1` |
| `aux_latest_ckpt32|per_source_delta` | 6 | 0.373023 | 0.887315 | -0.001237 | -5.000000 | 1.166667 | `unchanged=4, match_control=1, partial=1` |
| `no_aligner_parent_ckpt3668|scalar_logsumexp` | 1 | 0.003311 | 2.237113 | 0.000936 | 0.000000 | 6.000000 | `match_control=1` |
| `no_aligner_parent_ckpt3668|per_source_delta` | 1 | 0.003311 | 2.237113 | 0.000936 | 0.000000 | 6.000000 | `match_control=1` |

Key candidate rows:

| row | masked-target | patched-target | control-target | outcome | attention recovery | prob recovery | rank recovery |
|---:|---:|---:|---:|---|---:|---:|---:|
| aux row 20 | 0 | 0 | -14 | unchanged | 0.325684 | 0.002438 | 0 |
| aux row 21 | 0 | -12 | -12 | patched matches control | 0.649170 | -0.014873 | -18 |
| aux row 23 | 7 | 2 | -5 | partial toward control | 0.181641 | 0.002071 | -2 |
| aux row 28 | -5 | -5 | -6 | unchanged | 0.752441 | 0.001392 | -11 |

Scalar and per-source score-bias modes were identical on this selected panel
because each duplicate-basin bucket effectively had a single aligned source
token in these rows.

## Interpretation

This is the first direct sufficiency boundary:

- Restoring duplicate-basin attention score mass is causally effective for the
  route distribution itself.
- It is not sufficient for the full coordinate-basin movement in most auxiliary
  rows.
- The exact row 21 case is a positive sufficiency example: the patch moves top1
  from target-exact to the control/damaging below-target x2 value.
- Row 20 is the important counterexample: attention mass recovery is large, but
  the coordinate top1 remains target-exact.

This refines the mechanism picture: route probability is necessary-looking and
diagnostic, but the full x2 overcorrection likely depends on route content
and/or downstream residual-state effects, not score mass alone. The next high
value probe is therefore a duplicate-basin key/value or key+value state patch on
the same selected rows, compared against this score-only baseline.

## Reproduction

Score-bias patch:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python scripts/analysis/run_autoregressive_duplication_phase4_attention_score_bias_patch_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_region_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_score_bias_patch_layer17_head1_duplicate_basin \
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
  --score-bias-modes scalar_logsumexp,per_source_delta
```

Sufficiency summary:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/analysis/run_autoregressive_duplication_phase4_score_bias_sufficiency_summary.py \
  --score-bias-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_score_bias_patch_layer17_head1_duplicate_basin/attention_score_bias_patch_rows.jsonl \
  --candidate-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head/candidate_probe_rows.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_score_bias_patch_layer17_head1_duplicate_basin/sufficiency_summary
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
