# X2 Overcorrection Residual-State Sufficiency Probe

Date: 2026-06-11

## Scope

Follow-up probe for the x2 overcorrection panel after route/key/value evidence showed that duplicate-basin key/value state repairs some rows but leaves row 20 partially unresolved under same-head source expansion.

Primary question:

- Can downstream decoder residual state carry the remaining coordinate-basin correction for x2 overcorrection candidates?

This is a coordinate-top1/basin sufficiency probe, not a full target-probability restoration claim.

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
- Patch site: `decoder_layer`
- Patch layers: `16,20,24`
- Target top-k: `7`
- Device: `CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`

## Contract Fix

The first residual run exposed a target-selection contract bug in `phase4_residual_patch.py`: target specs were filtered by checkpoint, record, and phase, but not by `row_idx`, so all same-phase target specs crossed with all same-phase anchors. For the aux case this produced `36` rows per layer instead of the intended `6`.

Fix:

- Added row-aware target-spec filtering for optional `row_idx`, `generated_token_index`, and `next_token_index`.
- Manifests without those fields retain previous broad phase-level behavior.

Invalid exploratory artifact, kept only as a cautionary trace:

- `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_residual_patch_layers16_20_24_decoder_layer`

Use this row-filtered rerun instead:

- `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/targeted_route_content_patch_record33_panel/x2_overcorrection_residual_patch_layers16_20_24_decoder_layer_row_filtered`

## Artifacts

Row-filtered residual patch:

- `residual_patch_rows.jsonl`
- `phase4_residual_patch_summary.json`

Residual sufficiency summary:

- `residual_sufficiency_rows.jsonl`
- `phase4_residual_sufficiency_summary.json`
- `phase4_residual_sufficiency_summary.md`

## Findings

The row-filtered residual probe produced `21` rows: 7 candidate anchors times 3 patch layers.

Coordinate-top1 outcomes across all rows:

- `patched_matches_control`: 9
- `unchanged_from_masked`: 9
- `partial_toward_control`: 3

Aux checkpoint (`aux_latest_ckpt32`, record 33):

- Layer 16: `patched_matches_control=2`, `partial_toward_control=1`, `unchanged_from_masked=3`
- Layer 20: `patched_matches_control=2`, `partial_toward_control=1`, `unchanged_from_masked=3`
- Layer 24: `patched_matches_control=3`, `partial_toward_control=1`, `unchanged_from_masked=2`

Key rows:

- Row 20: all tested residual layers move masked top1 `170` to control top1 `158`.
- Row 21: all tested residual layers move masked top1 `170` toward the control side, but land at `158` while control top1 is `151`.
- Row 23: layer 16 is unchanged from masked; layers 20 and 24 match control top1.
- Row 26: layers 16 and 24 match control top1; layer 20 remains unchanged from masked.

No-aligner candidate (`no_aligner_parent_ckpt3668`, record 36 row 2):

- Layer 16 remains unchanged from masked.
- Layers 20 and 24 match control top1.

## Interpretation

Residual output patching is often sufficient to restore the coordinate top1 basin, including the previously stubborn row 20. This is not accompanied by positive target-probability/rank recovery; mean `prob_recovery_from_masked` is near-zero or negative in the row-filtered summary. The likely reading is that downstream residual state carries a basin-level coordinate distribution correction that can restore the low-side x2 attractor without simply making the literal target coordinate token more probable.

This extends the mechanism chain:

1. Duplicate-basin key/value restoration can recover the route and some x2 top1 choices.
2. Same-head broader source expansion does not solve row 20.
3. Later decoder residual state can solve row 20 at the coordinate-basin level.

So the remaining gap after key/value patching is likely downstream residual integration or multi-layer transformation after the layer-17 route/key event, not missing visual source breadth within the same head.

## Guardrails

- Treat the first 111-row residual artifact as invalid for interpretation because it crossed same-phase targets with anchors.
- Treat this as a small candidate-panel causal probe, not full validation.
- The top1/basin result and the target-probability result disagree; do not collapse them into one scalar "repair" claim.
- Next high-value probe is site decomposition at row 20 and row 21: `self_attn`, `post_attention_residual`, `mlp`, and their input sites at layers 20 and 24.
