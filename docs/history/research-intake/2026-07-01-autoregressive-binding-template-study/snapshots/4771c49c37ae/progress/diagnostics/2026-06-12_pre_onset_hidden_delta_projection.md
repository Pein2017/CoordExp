# Pre-Onset Hidden-Delta Projection

## Scope

This slice follows the matched-pair hidden-shift bridge. The previous run showed that duplicate-basin pixel masking creates a perturbation that grows through later decoder layers, especially around layers `20`, `24`, and final state. This probe asks whether that hidden-state delta points toward or away from each row's target coordinate direction.

Matched pair:

- record `33`, row `2`, `wine glass`, target `<|coord_145|>`;
- record `48`, row `2`, `bus`, target `<|coord_899|>`;
- checkpoint: `no_aligner_parent_ckpt3668`;
- phase: `post_y1/pre_x2`.

Artifact root:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48`

## Implementation

New helper:

`src/analysis/autoregressive_duplication_mechanism/phase4_hidden_delta_projection.py`

New CLI:

`scripts/analysis/run_autoregressive_duplication_phase4_hidden_delta_projection_shard.py`

Test:

`tests/test_autoreg_hidden_delta_projection.py`

Projection definition:

- hidden delta: `masked_hidden - control_hidden`;
- target direction: `output_weight[target_coord_token] - mean(output_weight[coord_tokens])`;
- positive dot means the masked hidden state moves toward the target coordinate token relative to the coordinate-token mean;
- negative dot means the masked hidden state moves away from that target direction.

The rows also include direct final coordinate-logit readouts from the same replay, because raw hidden-state projection is a directional approximation and can differ from final coordinate probability after downstream normalization/competition.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_hidden_delta_projection_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48/target_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48/target_region_rows.jsonl \
  --target-manifest-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48/matched_pair_hidden_shift_target_manifest.json \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48/hidden_delta_projection_selected_rows \
  --device auto \
  --torch-dtype bfloat16 \
  --attn-implementation auto \
  --top-k 2
```

Output files:

- `hidden_delta_projection_selected_rows/hidden_delta_projection_rows.jsonl`
- `hidden_delta_projection_selected_rows/phase4_hidden_delta_projection_summary.json`

Counts:

- source replay cases: `2`
- targeted replay cases: `2`
- intervention plan rows: `4`
- hidden-delta projection rows: `18`
- checkpoint count: `1`

## Late-Layer Direction Table

| rec | desc | prior mask sign | layer | target | dot | direction | final ctrl rank/prob | final mask rank/prob | final d_rank/d_prob |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 33 | wine glass | `mask_helps_rank` | 16 | 145 | +0.006979 | toward_target | 4/0.035321 | 4/0.032718 | +0/-0.002603 |
| 33 | wine glass | `mask_helps_rank` | 20 | 145 | -0.026223 | away_from_target | 4/0.035321 | 4/0.032718 | +0/-0.002603 |
| 33 | wine glass | `mask_helps_rank` | 24 | 145 | -0.138290 | away_from_target | 4/0.035321 | 4/0.032718 | +0/-0.002603 |
| 33 | wine glass | `mask_helps_rank` | 28 | 145 | -0.637906 | away_from_target | 4/0.035321 | 4/0.032718 | +0/-0.002603 |
| 48 | bus | `mask_harms_rank` | 16 | 899 | +0.002923 | toward_target | 1/0.062125 | 1/0.062967 | +0/+0.000842 |
| 48 | bus | `mask_harms_rank` | 20 | 899 | +0.001894 | toward_target | 1/0.062125 | 1/0.062967 | +0/+0.000842 |
| 48 | bus | `mask_harms_rank` | 24 | 899 | +0.023624 | toward_target | 1/0.062125 | 1/0.062967 | +0/+0.000842 |
| 48 | bus | `mask_harms_rank` | 28 | 899 | +0.065393 | toward_target | 1/0.062125 | 1/0.062967 | +0/+0.000842 |

## Read

The two exact-row replays diverge cleanly in the late projection:

- Record `33` wine-glass flips from weak target-positive at layer `16` to increasingly target-negative from layer `20` onward.
- Record `48` bus remains target-positive from layer `16` onward, and the target-positive dot grows by layer `24` and final state.

This supports a late residual-direction split: the same duplicate-basin image perturbation can grow through the decoder stack, but the late vector can point away from one row's target coordinate and toward another row's target coordinate.

The final-logit readout included in this artifact shows that, under this exact replay, masking lowers target probability for record `33` and raises it slightly for record `48`. This does not match the earlier residual-patch selector labels exactly. The older residual-patch runner selected all anchors in the phase for a targeted record, while this probe uses exact row targets; both use the same generated-token logit index. Treat the old labels as target-selection priors, and use this artifact's final-logit fields for interpretation of this exact-row replay.

## Mechanism Update

The strongest current picture is now:

1. Direct layer-17/head-1 duplicate-basin source-token routing is not the selected-row origin.
2. Duplicate-basin pixel masking perturbs broader visual/text context.
3. That perturbation is amplified through later decoder layers.
4. By layer `20+`, the hidden delta has row-specific coordinate direction: away from the wine-glass target in this exact replay, toward the bus target.
5. The late residual coordinate direction, not merely the presence of duplicate-basin attention, is the right object for the next causal slice.

Next deterministic probe: patch/project narrower late sites around layers `20` and `24`, ideally separating post-attention residual, post-attention norm, MLP input/output, and final norm effects for the same exact-row pair.

## Verification

Ran:

```bash
python -m pytest tests/test_autoreg_hidden_delta_projection.py -q
python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_hidden_delta_projection.py \
  scripts/analysis/run_autoregressive_duplication_phase4_hidden_delta_projection_shard.py
```

The local pytest wrapper printed its usual compact `No tests collected` label but exited `0`; direct invocation of both test functions also passed.

Checked non-empty artifacts:

- `hidden_delta_projection_selected_rows/hidden_delta_projection_rows.jsonl`
- `hidden_delta_projection_selected_rows/phase4_hidden_delta_projection_summary.json`
