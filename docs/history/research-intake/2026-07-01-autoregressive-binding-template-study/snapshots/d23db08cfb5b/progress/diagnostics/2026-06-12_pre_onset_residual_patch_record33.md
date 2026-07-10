# Pre-Onset Residual Patch: Record 33 No-Aligner

## Scope

This is the first causal follow-up to the duplication onset precursor panel. It targets one high-value pre-onset coordinate-basin row:

- Manifest root: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133`
- Checkpoint: `no_aligner_parent_ckpt3668`
- Rollout root: `/data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_no_aligner_parent_val128_freegreedy_ckpt3668_val128_bsz8_temp0_rp1p10_max3084_chatfix_8gpu`
- Record: `33`
- Image: `2685`
- Target row: row `2`, relative offset `-2`, desc `wine glass`
- Phase: `post_y1/pre_x2`
- Target next coord: `<|coord_145|>`
- Onset row: row `4`
- Duplicate-basin mask region from `phase2_region_rows.jsonl`: `[197,478,218,564]` in norm-1000 coordinates

Question: does the duplicate-basin visual region causally support or harm the pre-onset coordinate basin, and where is that effect carried in residual state?

## Run

Target manifest:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_causal_patch_record33_no_aligner/pre_onset_patch_target_manifest.json`

Command:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_duplication_phase4_residual_patch_shard.py \
  --token-windows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/shards/shard-00-of-04/token_windows.jsonl \
  --region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/phase2_region_rows.jsonl \
  --target-manifest-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_causal_patch_record33_no_aligner/pre_onset_patch_target_manifest.json \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_causal_patch_record33_no_aligner/residual_patch_layers20_24_26_27_mlp_decoder \
  --patch-layers 20,24,26,27 \
  --patch-sites mlp,decoder_layer \
  --patch-directions masked_to_control,control_to_masked \
  --target-top-k 1 \
  --top-k 8 \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --attn-implementation auto
```

Output root:

`/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/phase1_hidden_logit_manifest_20260611-092133/pre_onset_causal_patch_record33_no_aligner/residual_patch_layers20_24_26_27_mlp_decoder`

Files:

- `residual_patch_rows.jsonl`
- `phase4_residual_patch_summary.json`

Counts:

- source replay cases in shard: `8`
- targeted replay cases: `1`
- checkpoint count: `1`
- residual patch rows: `16`

## Baseline Mask Effect

Masking the duplicate-basin region **improves** the pre-onset target coordinate:

| condition | target prob | target rank | top1 |
| --- | ---: | ---: | ---: |
| control | `0.032961` | `5` | `154` |
| duplicate-basin masked | `0.036959` | `2` | `154` |

So for this row, the duplicate-basin visual region is not supporting the correct pre-onset x2 token. It is a harmful visual attractor: removing it moves the coordinate distribution closer to the target.

## Residual Patch Result

`masked_to_control` means: run the image with duplicate basin masked, but patch in the control residual vector at the same token/site/layer.

The decoder-layer residual fully reintroduces the harmful control state at late layers:

| direction | site | layer | control rank | masked rank | patched rank | control prob | masked prob | patched prob |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `masked_to_control` | `decoder_layer` | 20 | 5 | 2 | 5 | 0.032961 | 0.036959 | 0.033187 |
| `masked_to_control` | `decoder_layer` | 24 | 5 | 2 | 5 | 0.032961 | 0.036959 | 0.033273 |
| `masked_to_control` | `decoder_layer` | 26 | 5 | 2 | 5 | 0.032961 | 0.036959 | 0.032961 |
| `masked_to_control` | `decoder_layer` | 27 | 5 | 2 | 5 | 0.032961 | 0.036959 | 0.032961 |

MLP-only patch is weaker and layer-dependent:

| direction | site | layer | control rank | masked rank | patched rank | patched prob |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `masked_to_control` | `mlp` | 20 | 5 | 2 | 4 | 0.034359 |
| `masked_to_control` | `mlp` | 24 | 5 | 2 | 2 | 0.036112 |
| `masked_to_control` | `mlp` | 26 | 5 | 2 | 2 | 0.036173 |
| `masked_to_control` | `mlp` | 27 | 5 | 2 | 2 | 0.036515 |

`control_to_masked` means: run the normal image, but patch in the masked residual vector. At late MLP layers it slightly improves the target probability:

| direction | site | layer | patched rank | patched prob | prob delta from control |
| --- | --- | ---: | ---: | ---: | ---: |
| `control_to_masked` | `mlp` | 26 | 2 | 0.037209 | +0.004248 |
| `control_to_masked` | `mlp` | 27 | 2 | 0.037936 | +0.004975 |
| `control_to_masked` | `decoder_layer` | 26 | 2 | 0.036927 | +0.003966 |
| `control_to_masked` | `decoder_layer` | 27 | 2 | 0.036959 | +0.003998 |

## Mechanism Read

This is a useful inversion of the previous attention-facing story. The duplicate-basin visual region is not always useful evidence that pulls the model toward the correct target. In this pre-onset row, it acts as a harmful coordinate attractor:

1. Masking the duplicate basin improves the target coordinate rank from `5` to `2`.
2. Patching control residual state into the masked run removes that improvement, especially at the whole decoder layer in layers `20,24,26,27`.
3. Patching masked state into the control run improves or preserves the masked-like rank, especially at late MLP/decoder layers `26,27`.

So the pre-onset coordinate-basin signal is causally tied to visual-state differences, but the sign is harmful: the duplicate-basin visual region installs a residual state that worsens the target coordinate slot before the visible duplicate burst.

## Relation To Current Picture

The FN work found late-layer coordinate-basin repair under prefix guidance. This duplication probe suggests a complementary mechanism: local visual/prefix history can install a harmful coordinate basin before onset, and late residual state carries that basin. The common thread is coordinate-basin attraction in autoregressive coordinate slots, not a simple "attend to duplicate region and copy it" mechanism.

## Guardrail

This is one targeted row from one checkpoint. It is causal for that row, not a population claim. The next useful check is to repeat the same residual patch on:

- `no_aligner_parent_ckpt3668`, record `36` or `48`, `post_y1/pre_x2`
- `none_latest_ckpt32`, record `33`, `post_y1/pre_x2`
- a high pre-onset aligner case where `post_y1/pre_x2` offset `-1` spikes sharply

## Verification

The run completed successfully and produced:

- `residual_patch_rows.jsonl`, size `81016`
- `phase4_residual_patch_summary.json`, size `9370`
- target manifest size `561`

