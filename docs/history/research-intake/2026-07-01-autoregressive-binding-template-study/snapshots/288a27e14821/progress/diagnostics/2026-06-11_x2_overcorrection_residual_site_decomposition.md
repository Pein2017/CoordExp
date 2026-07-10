# X2 Overcorrection Residual Site Decomposition

Date: 2026-06-11

## Scope

Follow-up to `2026-06-11_x2_overcorrection_residual_state_sufficiency.md`.

Question:

- Once decoder-layer residual output patching restores the x2 coordinate basin for row 20, which internal site carries that correction?

Probe uses the same row-filtered 7-candidate x2 panel.

## Inputs

- Candidate panel:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_probe_panel_whole_head`
- Token windows:
  `candidate_token_windows.jsonl`
- Region rows:
  `candidate_region_rows.jsonl`
- Candidate targets:
  `phase4_candidate_probe_target_manifest.json`
- Candidate rows:
  `candidate_probe_rows.jsonl`

Probe settings:

- Patch direction: `masked_to_control`
- Patch layers: `20,24`
- Target top-k: `7`
- GPUs used in parallel: `0,1,2,3`

## Artifact Roots

Decoder layer boundary:

- `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_residual_patch_layers20_24_decoder_io_row_filtered`
- Sites: `decoder_layer_input`, `decoder_layer`

Self-attention:

- `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_residual_patch_layers20_24_self_attn_io_row_filtered`
- Sites: `self_attn_input`, `self_attn`

Post-attention boundary:

- `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_residual_patch_layers20_24_post_attention_sites_row_filtered`
- Sites: `post_attention_residual`, `post_attention_norm`

MLP:

- `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_residual_patch_layers20_24_mlp_io_row_filtered`
- Sites: `mlp_input`, `mlp`

Each root contains:

- `residual_patch_rows.jsonl`
- `phase4_residual_patch_summary.json`
- `residual_sufficiency_rows.jsonl`
- `phase4_residual_sufficiency_summary.json`
- `phase4_residual_sufficiency_summary.md`

## Aux Summary

Aux checkpoint: `aux_latest_ckpt32`, record 33, six candidate rows.

| site | layer | outcomes | mean patched-control top1 gap | mean expected-bin recovery |
| --- | ---: | --- | ---: | ---: |
| `decoder_layer_input` | 20 | `match=2, partial=2, unchanged=2` | 2.333 | -6.604 |
| `decoder_layer` | 20 | `match=2, partial=1, unchanged=3` | 2.333 | -6.107 |
| `decoder_layer_input` | 24 | `match=3, partial=1, unchanged=2` | 1.167 | -6.057 |
| `decoder_layer` | 24 | `match=3, partial=1, unchanged=2` | 1.167 | -6.018 |
| `self_attn_input` | 20 | `away=4, unchanged=2` | 13.000 | 1.989 |
| `self_attn` | 20 | `away=5, unchanged=1` | 13.833 | 2.979 |
| `self_attn_input` | 24 | `match=1, away=2, unchanged=3` | 8.333 | -0.273 |
| `self_attn` | 24 | `match=1, away=2, unchanged=3` | 8.333 | -0.137 |
| `post_attention_residual` | 20 | `away=2, partial=1, unchanged=3` | 10.000 | 0.400 |
| `post_attention_norm` | 20 | `away=2, partial=1, unchanged=3` | 10.000 | 0.400 |
| `post_attention_residual` | 24 | `away=1, partial=2, unchanged=3` | 7.000 | -0.978 |
| `post_attention_norm` | 24 | `away=1, partial=2, unchanged=3` | 7.000 | -0.978 |
| `mlp_input` | 20 | `away=2, partial=1, unchanged=3` | 10.000 | 0.400 |
| `mlp` | 20 | `away=2, partial=1, unchanged=3` | 10.000 | 0.400 |
| `mlp_input` | 24 | `away=1, partial=2, unchanged=3` | 7.000 | -0.978 |
| `mlp` | 24 | `away=1, partial=2, unchanged=3` | 7.000 | -0.978 |

## Row 20

Target bin: `165`; control top1: `158`; masked top1: `170`.

Layer 20:

- `decoder_layer`: patched top1 `158`, full control match.
- `decoder_layer_input`: patched top1 `151`, moves past the control basin but still to the low side.
- `self_attn_input`, `self_attn`, `post_attention_residual`, `post_attention_norm`, `mlp_input`, `mlp`: patched top1 remains `170`.

Layer 24:

- `decoder_layer_input` and `decoder_layer`: patched top1 `158`, full control match.
- `post_attention_residual`, `post_attention_norm`, `mlp_input`, `mlp`: patched top1 `165`, partial movement.
- `self_attn_input` and `self_attn`: patched top1 remains `170`.

## Row 21

Target bin: `170`; control top1: `151`; masked top1: `170`.

Layer 20:

- `decoder_layer_input` and `decoder_layer`: patched top1 `158`, partial movement toward control.
- `post_attention_residual`, `post_attention_norm`, `mlp_input`, `mlp`: patched top1 `165`, weaker partial movement.
- `self_attn_input` and `self_attn`: patched top1 `178`, away from control.

Layer 24:

- `decoder_layer_input` and `decoder_layer`: patched top1 `158`, partial movement toward control.
- `post_attention_residual`, `post_attention_norm`, `mlp_input`, `mlp`: patched top1 `165`, weaker partial movement.
- `self_attn_input` and `self_attn`: patched top1 remains `170`.

## Interpretation

The corrective basin is strongest at the decoder-layer boundary, especially layer 24. Isolated self-attention patching does not repair the x2 basin and often pushes away from control. Post-attention and MLP sites have weak/partial movement but do not reproduce the full decoder-layer effect.

This suggests the row-20/row-21 downstream correction is not a simple isolated self-attention output or isolated MLP output at layers 20/24. It appears as a residual-stream state at the full decoder block boundary, likely reflecting integrated state accumulated across block internals and earlier layers.

Mechanism chain update:

1. Layer-17 duplicate-basin route/key state is a real upstream source.
2. Same-head source expansion does not close row 20.
3. Layer-20/24 decoder-boundary residual state can close row 20 at the coordinate-top1 level.
4. Internal site decomposition localizes the strongest correction to full decoder-layer boundary state rather than self-attn or MLP submodule outputs alone.

## Guardrails

- This is still a seven-candidate panel, not full validation.
- The module-site names follow the current patch runner's resolved Qwen/PEFT module paths. `post_attention_residual`, `post_attention_norm`, `mlp_input`, and `mlp` give identical summaries here, so future deeper probes should verify whether those hooks resolve to equivalent boundary tensors for this model wrapper.
- Probability/rank movement is not the same as coordinate-basin/top1 movement. Site interpretation should prioritize the top1/expected-bin summaries for this x2 basin question.
