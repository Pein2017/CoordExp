# Pre-Onset Matched-Pair Residual Site Patch

Date: 2026-06-12

## Scope

This slice follows `2026-06-12_pre_onset_hidden_delta_projection.md`.
The projection probe showed that duplicate-basin pixel masking creates different
late coordinate directions for two no-aligner pre-onset rows:

- record `33`, row `2`, `wine glass`, target `<|coord_145|>`;
- record `48`, row `2`, `bus`, target `<|coord_899|>`;
- checkpoint: `no_aligner_parent_ckpt3668`;
- phase: `post_y1/pre_x2`.

This probe asks whether those late hidden-delta directions have a causal
residual-patch expression at selected decoder sites.

Artifact root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48/residual_patch_layers16_20_24_27_late_sites_matched_pair
```

Inputs:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48/target_token_windows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48/target_region_rows.jsonl
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48/matched_pair_hidden_shift_target_manifest.json
```

## Run

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
python scripts/analysis/run_autoregressive_duplication_phase4_residual_patch_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48/target_token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48/target_region_rows.jsonl \
  --target-manifest-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48/matched_pair_hidden_shift_target_manifest.json \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48/residual_patch_layers16_20_24_27_late_sites_matched_pair \
  --device auto \
  --torch-dtype bfloat16 \
  --attn-implementation auto \
  --patch-layers 16,20,24,27 \
  --patch-sites decoder_layer,post_attention_residual,self_attn,mlp \
  --patch-directions masked_to_control,control_to_masked \
  --top-k 8 \
  --target-top-k 2
```

Important run detail: `--target-top-k 2` is required here. The first attempt
used the CLI default `--target-top-k 1`, which silently selected only the first
manifest target, record `33`. The matched-pair artifact above has:

- source replay cases: `2`;
- targeted replay cases: `2`;
- residual patch rows: `64`;
- checkpoint count: `1`.

Layer `28` is not a decoder hook in this wrapped model. Projection artifacts can
include final hidden state, but residual hooks stop at decoder layer `27`.

## Key Rows

Baseline exact-row readouts:

| record | desc | target | control rank/prob | masked rank/prob | mask effect |
| ---: | --- | ---: | --- | --- | --- |
| `33` | `wine glass` | `145` | `4 / 0.035321` | `4 / 0.032718` | probability down, rank unchanged |
| `48` | `bus` | `899` | `1 / 0.062125` | `1 / 0.062967` | probability up, rank unchanged |

Best causal patch rows by case:

| record | direction | site | layer | patched rank/prob | recovery from masked | patched top1 |
| ---: | --- | --- | ---: | --- | --- | ---: |
| `33` | `masked_to_control` | `mlp` | `27` | `2 / 0.037059` | `+2 / +0.004341` | `154` |
| `33` | `masked_to_control` | `post_attention_residual` | `27` | `2 / 0.037059` | `+2 / +0.004341` | `154` |
| `33` | `control_to_masked` | `mlp` | `16` | `2 / 0.037887` | `+2 / +0.005169` | `154` |
| `33` | `control_to_masked` | `post_attention_residual` | `16` | `2 / 0.037887` | `+2 / +0.005169` | `154` |
| `33` | `control_to_masked` | `mlp` | `20` | `2 / 0.035935` | `+2 / +0.003217` | `154` |
| `48` | `masked_to_control` | `decoder_layer` | `16` | `1 / 0.063191` | `+0 / +0.000224` | `899` |
| `48` | `masked_to_control` | `self_attn` | `24` | `1 / 0.063174` | `+0 / +0.000207` | `899` |
| `48` | `control_to_masked` | `mlp` | `16` | `2 / 0.059660` | `-1 / -0.003308` | `901` |

Selected late-site table:

| record | direction | site | layer | patched rank/prob | recovery from masked | top1 |
| ---: | --- | --- | ---: | --- | --- | ---: |
| `33` | `control_to_masked` | `mlp` | `20` | `2 / 0.035935` | `+2 / +0.003217` | `154` |
| `33` | `control_to_masked` | `mlp` | `24` | `4 / 0.035299` | `+0 / +0.002581` | `154` |
| `33` | `control_to_masked` | `mlp` | `27` | `5 / 0.032228` | `-1 / -0.000490` | `154` |
| `33` | `masked_to_control` | `mlp` | `20` | `5 / 0.033463` | `-1 / +0.000745` | `154` |
| `33` | `masked_to_control` | `mlp` | `24` | `5 / 0.033411` | `-1 / +0.000693` | `154` |
| `33` | `masked_to_control` | `mlp` | `27` | `2 / 0.037059` | `+2 / +0.004341` | `154` |
| `48` | `control_to_masked` | `mlp` | `20` | `1 / 0.061683` | `+0 / -0.001284` | `899` |
| `48` | `control_to_masked` | `mlp` | `24` | `1 / 0.061355` | `+0 / -0.001612` | `899` |
| `48` | `control_to_masked` | `mlp` | `27` | `1 / 0.062290` | `+0 / -0.000677` | `899` |
| `48` | `masked_to_control` | `mlp` | `20` | `1 / 0.061581` | `+0 / -0.001387` | `899` |
| `48` | `masked_to_control` | `mlp` | `24` | `1 / 0.062026` | `+0 / -0.000941` | `899` |
| `48` | `masked_to_control` | `mlp` | `27` | `1 / 0.061969` | `+0 / -0.000998` | `899` |

## Read

Record `33` has a causal rank-moving pathway. The strongest rank repairs are
MLP-side or post-attention-residual patches, not attention-output patches:

- `masked_to_control` at layer `27` MLP/post-attention moves target rank
  `4 -> 2` and raises probability by `+0.004341` versus the masked baseline.
- `control_to_masked` at layer `16` MLP/post-attention also moves rank
  `4 -> 2`, and layer `20` MLP/post-attention gives the same rank repair with
  smaller probability lift.
- Self-attention patches are weaker and mostly probability-only in this row.

This is not a simple monotonic late-layer story. The layer-`27` MLP direction is
beneficial when patching masked with control, while layer-`16` and layer-`20`
MLP directions can be beneficial in the opposite control-to-masked direction.
That suggests the exact-row coordinate basin is shaped by multiple residual
components with different signs across depth, rather than one single
duplicate-basin vector.

Record `48` is a contrast case rather than a rank-repair case. The target is
already top-1 in both control and masked replay, and patches mostly change
probability or occasionally damage the local basin:

- most late MLP patches keep target rank `1` and top1 `899`, with small negative
  probability deltas;
- `control_to_masked` at layer `16` MLP/post-attention damages the basin:
  target rank `1 -> 2`, probability `0.062967 -> 0.059660`, top1 `901`.

So the bus row is not evidence of hidden target recovery. It is evidence that an
already-correct coordinate basin is locally fragile to early MLP/post-attention
state insertion, while late site patches mostly leave the top-1 basin intact.

## Mechanism Update

The current best picture is:

1. Duplicate-basin masking perturbs hidden state broadly enough to show
   measurable coordinate-direction effects by later decoder layers.
2. Those effects are row-specific. In this matched pair, wine-glass has a
   rank-moving causal pathway; bus is already rank-1 and mostly calibration-only.
3. The strongest causal site for the wine-glass row is MLP-side or
   post-attention residual state, not raw attention output.
4. The sign is depth-dependent, which argues against a single clean
   "duplicate-basin attention vector" account.
5. The next useful object is not broader attention concentration; it is
   coordinate-basin residual composition across depth.

## Next Step

Promising deterministic continuation:

1. Run a small multi-case selector over pre-onset rows where exact replay has
   target rank `2-10` or target probability drift under duplicate-basin masking.
2. For those rows, collect only MLP/post-attention sites at layers `16,20,24,27`
   and keep both patch directions.
3. Split cases into:
   - rank-moving repair;
   - top-1 stable calibration;
   - patch-damage fragile basin;
   - no-effect hard case.

This would connect the current matched-pair read to the broader hypothesis:
duplication onset is preceded by local coordinate-basin instability, but only
some rows have a causally recoverable residual direction.

## Verification

Artifact existence check:

```bash
python - <<'PY'
from pathlib import Path
root=Path('/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_matched_pair_no_aligner_r33_r48/residual_patch_layers16_20_24_27_late_sites_matched_pair')
for name in ['residual_patch_rows.jsonl','phase4_residual_patch_summary.json']:
    p=root/name
    assert p.is_file() and p.stat().st_size > 0, p
    print(name, p.stat().st_size)
print('matched-pair residual patch artifacts exist')
PY
```

Observed files:

- `residual_patch_rows.jsonl`: `324999` bytes;
- `phase4_residual_patch_summary.json`: `18077` bytes.
