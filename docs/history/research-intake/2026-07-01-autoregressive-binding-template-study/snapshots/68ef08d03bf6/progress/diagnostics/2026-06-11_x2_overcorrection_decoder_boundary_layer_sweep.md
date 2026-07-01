# X2 Overcorrection Decoder-Boundary Layer Sweep

Date: 2026-06-11

## Scope

Follow-up to the residual-state sufficiency and residual site-decomposition probes.

Question:

- Where along the decoder stack does the downstream x2 coordinate-basin correction become available at the full decoder-layer boundary?

This probe patches masked runs with control residual state at `decoder_layer_input` and `decoder_layer` sites over selected layers before, around, and after the layer-17 route/key event.

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
- Patch sites: `decoder_layer_input`, `decoder_layer`
- Target top-k: `7`
- GPUs used: `0,1,2`

## Artifact Roots

Early/route-adjacent:

- `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_residual_decoder_boundary_sweep_layers12_16_17`
- Layers: `12,16,17`

Middle:

- `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_residual_decoder_boundary_sweep_layers18_20_22`
- Layers: `18,20,22`

Late:

- `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_residual_decoder_boundary_sweep_layers24_26_27`
- Layers: `24,26,27`

Each valid root contains:

- `residual_patch_rows.jsonl`
- `phase4_residual_patch_summary.json`
- `residual_sufficiency_rows.jsonl`
- `phase4_residual_sufficiency_summary.json`
- `phase4_residual_sufficiency_summary.md`

Invalid attempted root:

- `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_residual_decoder_boundary_sweep_layers24_26_28`

Reason: the current model wrapper cannot resolve decoder layer `28`; valid upper boundary in this probe is layer `27`.

## Aux Layer Table

Aux checkpoint: `aux_latest_ckpt32`, record 33, six candidate rows.

| layer | site | outcomes | mean patched-control top1 gap | mean expected-bin recovery |
| ---: | --- | --- | ---: | ---: |
| 12 | `decoder_layer_input` | `unchanged=4, away=2` | 9.167 | 0.169 |
| 12 | `decoder_layer` | `unchanged=4, away=2` | 9.500 | 0.195 |
| 16 | `decoder_layer_input` | `match=3, partial=1, unchanged=2` | 2.333 | -3.096 |
| 16 | `decoder_layer` | `match=2, partial=1, unchanged=3` | 2.333 | -5.137 |
| 17 | `decoder_layer_input` | `match=2, partial=1, unchanged=3` | 2.333 | -5.137 |
| 17 | `decoder_layer` | `match=1, partial=1, unchanged=4` | 3.500 | -6.011 |
| 18 | `decoder_layer_input` | `match=1, partial=1, unchanged=4` | 3.500 | -6.011 |
| 18 | `decoder_layer` | `match=2, partial=1, unchanged=3` | 2.333 | -7.344 |
| 20 | `decoder_layer_input` | `match=2, partial=2, unchanged=2` | 2.333 | -6.604 |
| 20 | `decoder_layer` | `match=2, partial=1, unchanged=3` | 2.333 | -6.107 |
| 22 | `decoder_layer_input` | `match=2, partial=1, unchanged=2, away=1` | 2.500 | -6.129 |
| 22 | `decoder_layer` | `match=2, partial=1, unchanged=3` | 2.333 | -5.841 |
| 24 | `decoder_layer_input` | `match=3, partial=1, unchanged=2` | 1.167 | -6.057 |
| 24 | `decoder_layer` | `match=3, partial=1, unchanged=2` | 1.167 | -6.018 |
| 26 | `decoder_layer_input` | `match=2, partial=1, unchanged=3` | 2.333 | -5.977 |
| 26 | `decoder_layer` | `match=3, partial=1, unchanged=1, away=1` | 2.333 | -6.117 |
| 27 | `decoder_layer_input` | `match=3, partial=1, unchanged=1, away=1` | 2.333 | -6.117 |
| 27 | `decoder_layer` | `match=4, unchanged=2` | 0.000 | -6.171 |

## Row 20

Target bin: `165`; control top1: `158`; masked top1: `170`.

- Layer 12 input/output: unchanged at masked top1 `170`.
- Layer 16 input/output: patched top1 `158`, full control match.
- Layer 17 input/output: patched top1 `158`, full control match.
- Layer 18 input: patched top1 `158`; layer 18 output overcorrects to `151` while still moving to the low side.
- Layers 20,22,24,26,27 input/output: mostly full control match at `158`; layer 20 input gives `151`.

Row 20 therefore has a usable low-side/control-basin correction by layer 16, shortly before the previously identified layer-17 route/key event.

## Row 21

Target bin: `170`; control top1: `151`; masked top1: `170`.

- Layer 12 input/output: unchanged at masked top1 `170`.
- Layers 16 and 17 input/output: patched top1 `158`, partial movement toward control.
- Layer 18 output: patched top1 `151`, full control match; layer 18 input remains `158`.
- Layers 20,22,24,26 input/output: patched top1 `158`, stable partial low-side basin.
- Layer 27 output: patched top1 `151`, full control match; layer 27 input remains `158`.

Row 21 shows a stable low-side basin from layer 16 onward, but exact control-basin selection appears at decoder output layer 18 and again at final decoder output layer 27.

## Interpretation

The downstream residual correction is not a monotonic single-layer onset. Layer 12 does not carry the x2 correction. Starting around layer 16, the residual stream carries a low-side correction that is sufficient for row 20 and partially sufficient for row 21. The exact choice among nearby low-side coordinate basins sharpens later, with the strongest aggregate decoder-boundary result at layer 27 output (`match=4/6`, mean patched-control top1 gap `0.000`).

This updates the mechanism picture:

1. The layer-17 duplicate-basin route/key event remains a strong upstream causal handle.
2. Downstream residual state before/around layer 16 already carries enough low-side coordinate-basin information to repair row 20.
3. Exact basin selection for harder cases like row 21 is refined across later decoder blocks, most cleanly visible at layer 27 output.

The likely mechanism is therefore a two-stage process: an upstream route/key/visual-basin event establishes or amplifies low-side coordinate attraction, while later decoder residual dynamics sharpen the final basin choice among nearby `<|coord_*|>` tokens.

## Guardrails

- Candidate-panel scale only: seven selected rows, six aux rows.
- Do not interpret the invalid `layer=28` attempt.
- Coordinate-top1 and expected-bin movement are the primary readouts here; target probability/rank often disagree with basin restoration.
- The layer-16 signal does not contradict the layer-17 route/key result. It means the residual stream already contains useful basin state near that region, while the route/key intervention remains the traced upstream visual/source handle.
