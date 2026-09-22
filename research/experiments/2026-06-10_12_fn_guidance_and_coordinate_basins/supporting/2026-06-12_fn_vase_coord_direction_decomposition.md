# FN Vase Coordinate Direction Decomposition

## Scope

- Question: in the recovered vase FN, does the late source-to-target residual patch mainly lift the true `y1` coordinate basin, suppress the wrong full-prefix basin, or both?
- Checkpoint: `/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668`
- Case: COCO image `139`, GT index `7`, desc `vase`, target box `[526, 468, 542, 508]`.
- Contrast: source `prefix=0, tier=desc_x1` into target `prefix=all, tier=desc_x1`.
- Patch/readout layers: decoder layers `24,25,26,27`.
- Patch/readout sites: `self_attn,self_attn_input,mlp,mlp_input`.
- Artifact root: `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/no_aligner_parent_ckpt3668_vase_prefix_coordslot_direction_patch_sites_l24_27`

## New Artifact Contract

The FN coord-slot probe now emits:

- `fn_coordslot_direction_rows.jsonl`
- `direction_rows_path` and `direction_row_count` in `phase4_fn_coordslot_logit_probe_summary.json`
- a `Source-Target Coord Direction Rows` section in `phase4_fn_coordslot_logit_probe_report.md`

Each direction row captures the source and target vectors at the selected patch site, projects both through the model norm/head readout, and stores the coordinate-logit delta `source - target`. The row includes the target-bin delta, the baseline top1/wrong-bin delta, the margin change between them, positive/negative delta top bins, and selected-bin deltas.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_duplication_phase4_fn_coordslot_logit_probe.py \
  --guidance-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/no_aligner_parent_ckpt3668/fn_visibility_guidance_probe_rows.jsonl \
  --gt-vs-pred-scored-path /data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_no_aligner_parent_val128_freegreedy_ckpt3668_val128_bsz8_temp0_rp1p10_max3084_chatfix_8gpu/gt_vs_pred_scored.jsonl \
  --checkpoint-path /data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668 \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/no_aligner_parent_ckpt3668_vase_prefix_coordslot_direction_patch_sites_l24_27 \
  --image-id 139 \
  --gt-idx 7 \
  --guidance-tiers desc_x1,desc_x1_y1,desc_x1_y1_x2 \
  --prefix-object-limits 0,all \
  --layers all \
  --patch-layers 24,25,26,27 \
  --patch-sites self_attn,self_attn_input,mlp,mlp_input \
  --patch-source 0,desc_x1 \
  --patch-target all,desc_x1 \
  --top-k 8 \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --attn-implementation auto
```

Output counts:

- condition rows: `6`
- layer rows: `174`
- hidden delta rows: `58`
- direction rows: `16`
- residual patch rows: `16`

## Direction Readout

Baseline target condition remains the full-prefix `desc_x1` failure:

- target `y1` bin: `468`
- baseline top1/wrong bin: `501`
- baseline target rank: `52`

Most relevant direction rows:

| site | layer | target delta at 468 | wrong-bin delta at 501 | target-vs-wrong margin change | positive bins | negative bins |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `mlp` | 26 | `+1.2500` | `+0.0156` | `+1.2344` | `[441,440,444,445,446]` | `[522,520,517,514,521]` |
| `mlp` | 27 | `+0.2500` | `-1.6562` | `+1.9062` | `[452,461,456,458,454]` | `[502,501,499,503,500]` |
| `self_attn` | 27 | `+1.1953` | `+0.0312` | `+1.1641` | `[546,550,547,553,543]` | `[165,169,167,155,163]` |
| `self_attn_input` | 27 | `+0.9688` | `-0.0312` | `+1.0000` | `[441,439,438,435,446]` | `[519,517,526,527,522]` |

The corresponding residual patch rows still show the strongest causal recovery on the MLP side:

| site | layer | patched rank | rank recovery | patched top1 | prob delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mlp` | 26 | `19` | `33` | `483` | `+0.011056` |
| `mlp` | 27 | `12` | `40` | `479` | `+0.009618` |
| `self_attn` | 27 | `38` | `14` | `486` | `+0.005325` |
| `self_attn_input` | 27 | `38` | `14` | `483` | `+0.005244` |

## Mechanism Read

This readout supports a two-part late-layer mechanism for this FN case:

1. Layer-26 MLP-side state lifts target-adjacent vertical bins. It gives the exact target bin `468` a large positive direction delta (`+1.25`) while leaving the wrong `501` bin almost unchanged (`+0.0156`).
2. Layer-27 MLP-side state suppresses the wrong full-prefix vertical basin. It only modestly lifts the exact target bin (`+0.25`) but strongly decreases bins around the wrong basin (`499-503`, including `501` at `-1.6562`).

This is stronger than the earlier "MLP carries the correction" statement. The correction is not merely an exact target-token boost; it appears to reshape the coordinate basin by both lifting target-local alternatives and knocking down the full-prefix wrong-y basin.

## Interpretation Boundary

This is a one-case causal target selector for the recovered vase FN. It should guide the next probes, not be presented as the global FN mechanism. The next attractive path is to repeat the direction decomposition on additional recoverable FNs and compare against the hard residual `book` FN to see whether this lift-then-suppress pattern separates recoverable prefix-basin failures from genuine visibility/extent failures.

## Verification

```bash
python - <<'PY'
import importlib.util
path='tests/analysis/autoregressive_duplication_mechanism/test_phase4_fn_coordslot_logit_probe.py'
spec=importlib.util.spec_from_file_location('coordslot_logit_tests', path)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for name in sorted(n for n in dir(mod) if n.startswith('test_')):
    getattr(mod, name)()
print('direct fn coordslot logit probe harness passed')
PY

python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_fn_coordslot_logit_probe.py \
  scripts/analysis/run_autoregressive_duplication_phase4_fn_coordslot_logit_probe.py \
  tests/analysis/autoregressive_duplication_mechanism/test_phase4_fn_coordslot_logit_probe.py
```

Both checks passed before the GPU run.
