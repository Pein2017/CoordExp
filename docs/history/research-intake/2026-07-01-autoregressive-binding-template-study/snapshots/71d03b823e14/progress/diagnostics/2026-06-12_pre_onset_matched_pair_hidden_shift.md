# Pre-Onset Matched Pair Hidden-Shift Bridge

## Scope

This slice follows the matched-pair route/content result. The previous probe showed that at the exact selected row-2 slots, layer-17/head-1 direct duplicate-basin source tokens are inert, while broader `visual_non_basin` and sometimes `text_prefix` buckets can recover the masked state.

The next question was whether duplicate-basin pixel masking nevertheless creates a hidden-state perturbation that grows into the late residual layers where residual patching was causal.

Matched pair:

- harmful-attractor row: `no_aligner_parent_ckpt3668`, record `33`, row `2`, `post_y1/pre_x2`, desc `wine glass`, target `<|coord_145|>`;
- useful-support row: `no_aligner_parent_ckpt3668`, record `48`, row `2`, `post_y1/pre_x2`, desc `bus`, target `<|coord_899|>`.

Artifact root:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48`

## Run

Target manifest:

`matched_pair_hidden_shift_target_manifest.json`

Command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_hidden_shift_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48/target_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48/target_region_rows.jsonl \
  --target-manifest-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48/matched_pair_hidden_shift_target_manifest.json \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48/hidden_shift_selected_rows \
  --device auto \
  --torch-dtype bfloat16 \
  --attn-implementation auto \
  --top-k 2
```

Output files:

- `hidden_shift_selected_rows/hidden_shift_rows.jsonl`
- `hidden_shift_selected_rows/phase4_hidden_shift_summary.json`

Counts:

- source replay cases: `2`
- targeted replay cases: `2`
- intervention plan rows: `4`
- hidden-shift rows: `18`
- checkpoint count: `1`

## Layerwise Shift

The table reports `masked_hidden - control_hidden` at the generated-token state for the selected coordinate slot.

| rec | desc | layer | delta L2 | rel to ctrl norm | growth from previous |
| ---: | --- | ---: | ---: | ---: | ---: |
| 33 | wine glass | 0 | 0.000000 | 0.000000 |  |
| 33 | wine glass | 4 | 0.463400 | 0.009173 |  |
| 33 | wine glass | 8 | 0.610040 | 0.011192 | 1.32 |
| 33 | wine glass | 12 | 1.133510 | 0.012409 | 1.86 |
| 33 | wine glass | 16 | 3.427734 | 0.021606 | 3.02 |
| 33 | wine glass | 20 | 11.459646 | 0.022704 | 3.34 |
| 33 | wine glass | 24 | 30.438532 | 0.026953 | 2.66 |
| 33 | wine glass | 28 | 123.875168 | 0.036067 | 4.07 |
| 48 | bus | 0 | 0.000000 | 0.000000 |  |
| 48 | bus | 4 | 0.394437 | 0.006534 |  |
| 48 | bus | 8 | 0.628707 | 0.009425 | 1.59 |
| 48 | bus | 12 | 1.134067 | 0.011934 | 1.80 |
| 48 | bus | 16 | 2.653101 | 0.015590 | 2.34 |
| 48 | bus | 20 | 7.847399 | 0.014118 | 2.96 |
| 48 | bus | 24 | 24.035570 | 0.022539 | 3.06 |
| 48 | bus | 28 | 86.898491 | 0.027498 | 3.62 |

## Interpretation

This run bridges the route/content negative result and the residual-patch positive result.

At layer 17/head 1, direct duplicate-basin source-token routing is not the selected-row origin. But duplicate-basin pixel masking still produces a hidden-state perturbation that grows through the later decoder stack:

- record 33 `wine glass`: L2 shift rises from `3.43` at layer `16` to `11.46` at layer `20`, `30.44` at layer `24`, and `123.88` by the final state;
- record 48 `bus`: L2 shift rises from `2.65` at layer `16` to `7.85` at layer `20`, `24.04` at layer `24`, and `86.90` by the final state.

This growth aligns with the residual-patch layer range already tested (`20,24,26,27`). The harmful wine-glass row has the larger late perturbation at every layer from `16` onward, which is consistent with its stronger rank-level residual-patch sign. The useful bus row also has a large late perturbation, but its sign in the coordinate readout is support rather than harm.

So the current mechanism picture is:

1. Duplicate-basin masking first perturbs broader visual/text context rather than direct duplicate-basin token routing at layer 17/head 1.
2. That perturbation is amplified through later decoder layers into a substantial residual-state difference.
3. The late residual state can be causally helpful or harmful depending on how the induced coordinate-basin direction aligns with the current row's target coordinate.
4. The next deterministic probe should project the late hidden-state delta onto target-coordinate directions and/or patch narrower late-layer sites, rather than continuing to treat layer-17/head-1 duplicate-basin attention as the universal origin.

## Verification

Checked non-empty artifacts:

- `matched_pair_hidden_shift_target_manifest.json`
- `hidden_shift_selected_rows/hidden_shift_rows.jsonl`
- `hidden_shift_selected_rows/phase4_hidden_shift_summary.json`

The hidden-shift run completed with `hidden_shift_row_count=18` and `targeted_replay_case_count=2`.
