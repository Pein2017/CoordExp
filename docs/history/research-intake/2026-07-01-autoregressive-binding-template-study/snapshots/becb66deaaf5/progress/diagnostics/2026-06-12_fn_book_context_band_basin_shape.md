# FN Book Context-Band Basin Shape

Date: 2026-06-12

## Scope

This closes the latest book false-negative context-band experiment by rerunning
the exact interaction panel with full coordinate-basin readout fields persisted
in patch rows.

Case:

```text
image_id = 139
gt_idx = 17
desc = book
target y2 = 826
prompt tier = desc_x1_y1_x2
prefixes = 0, all
```

The previous interaction note showed that aux `upper+target-adjacent` gives the
closest partial rescue by top1, while no-aligner `upper+lower` and whole-ring
patches enter a boundary `999` basin. This run asks whether that top1 movement
also corresponds to stronger mass around the gold coordinate.

## Implementation Update

The visual-token patch probe now persists all `coord_*` fields from
`summarize_coord_logits` into patch rows with `baseline_` and `patched_`
prefixes. This keeps the existing rank/prob/top1 columns while adding:

```text
coord_entropy
coord_expected_bin
coord_std
coord_expected_abs_error
coord_mass_radius_0
coord_mass_radius_4
coord_mass_radius_8
coord_mass_radius_16
coord_mass_radius_32
coord_mass_radius_64
```

This is a schema extension for research artifacts, not a benchmark contract
change.

## Artifacts

Rerun root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent_context_ring_band_interactions_basin_shape
```

No-aligner parent root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent_context_ring_band_interactions_basin_shape/no_aligner_parent_ckpt3668
```

Aux checkpoint root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent_context_ring_band_interactions_basin_shape/aux_latest_ckpt32
```

Reduction:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent_context_ring_band_interactions_basin_shape/book17_context_ring_band_interactions_basin_shape_reduction.json
```

Counts per checkpoint:

```text
baseline_row_count = 2
membership_row_count = 26
patch_row_count = 112
```

## Prefix-All Basin Shape

Headline rows for prefix `all`:

| checkpoint | intervention | rank recovery | top1 | top1 distance | mass r16 | delta r16 | mass r64 | delta r64 | expected bin | error delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-aligner | `upper` | 68 | 743 | 83 | 0.168925 | +0.150163 | 0.457782 | +0.355270 | 787.9 | -22.5 |
| no-aligner | `upper+target-adjacent` | 85 | 743 | 83 | 0.166109 | +0.147348 | 0.481629 | +0.379118 | 776.6 | -21.4 |
| no-aligner | `upper+lower` | 114 | 999 | 173 | 0.244280 | +0.225519 | 0.611923 | +0.509411 | 818.9 | -30.4 |
| no-aligner | `whole ring` | 121 | 999 | 173 | 0.219430 | +0.200668 | 0.512108 | +0.409597 | 831.8 | -20.0 |
| aux | `upper` | 77 | 778 | 48 | 0.163307 | +0.153714 | 0.789767 | +0.661240 | 788.4 | -44.8 |
| aux | `upper+target-adjacent` | 115 | 796 | 30 | 0.281250 | +0.271657 | 0.846251 | +0.717724 | 810.5 | -51.3 |
| aux | `upper+lower` | 79 | 778 | 48 | 0.165726 | +0.156133 | 0.782549 | +0.654022 | 791.5 | -43.8 |
| aux | `whole ring` | 107 | 999 | 173 | 0.228512 | +0.218918 | 0.783097 | +0.654571 | 814.5 | -42.9 |

## Mechanism Read

Aux `upper+target-adjacent` is a real partial basin improvement, not only a
top1 curiosity:

```text
top1 = 796
distance to target = 30
mass radius 16 = 0.281250
mass radius 64 = 0.846251
expected bin = 810.5
```

It has the strongest target-neighborhood mass among the aux headline rows and
the best expected absolute error shift. The mass is still not centered tightly
enough to rescue the exact `826` coordinate.

No-aligner `upper+lower` is the cautionary counterpart:

```text
top1 = 999
distance to target = 173
mass radius 16 = 0.244280
mass radius 64 = 0.611923
expected bin = 818.9
```

It increases gold-neighborhood mass and improves expected bin, but the modal
coordinate is still the boundary token. So rank recovery and local target mass
can improve while visible decoding remains wrong.

This closes the current experiment with a sharper interpretation:

1. The hard FN is not visual blindness; visual-context perturbations move both
   target mass and expected coordinate.
2. Aux has a more usable `upper+target-adjacent` synchronization path.
3. No-aligner more readily couples upper/lower context to the boundary basin.
4. Exact-coordinate failure remains a basin-selection problem: the distribution
   can shift toward the gold neighborhood without making `826` the local winner.

## Verification

Commands run:

```bash
python - <<'PY'
import pytest
raise SystemExit(pytest.main(['tests/analysis/autoregressive_duplication_mechanism/test_phase4_fn_visual_token_patch.py','-q']))
PY
```

```text
1 passed in 1.08s
```

```bash
python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_fn_visual_token_patch.py \
  scripts/analysis/run_autoregressive_duplication_phase4_fn_visual_token_patch.py
```

GPU reruns:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_duplication_phase4_fn_visual_token_patch.py \
  --guidance-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/no_aligner_parent_ckpt3668/fn_visibility_guidance_probe_rows.jsonl \
  --gt-vs-pred-scored-path /data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_no_aligner_parent_val128_freegreedy_ckpt3668_val128_bsz8_temp0_rp1p10_max3084_chatfix_8gpu/gt_vs_pred_scored.jsonl \
  --candidate-region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent_context_ring_band_interactions/candidate_regions/no_aligner_parent_ckpt3668/candidate_region_rows.jsonl \
  --checkpoint-path /data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668 \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent_context_ring_band_interactions_basin_shape/no_aligner_parent_ckpt3668 \
  --image-id 139 \
  --gt-idx 17 \
  --guidance-tiers desc_x1_y1_x2 \
  --prefix-object-limits 0,all \
  --patch-layers 13,14,16,17,24,25,26,27 \
  --interventions zero:context_ring_upper_band,zero:context_ring_target_adjacent_band,zero:context_ring_lower_band,zero:context_ring_upper_plus_target_adjacent,zero:context_ring_upper_plus_lower,zero:context_ring_target_adjacent_plus_lower,zero:context_ring \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --attn-implementation auto
```

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/analysis/run_autoregressive_duplication_phase4_fn_visual_token_patch.py \
  --guidance-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/aux_latest_ckpt32_guarded/fn_visibility_guidance_probe_rows.jsonl \
  --gt-vs-pred-scored-path /data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_loss_only_aux_latest_aligner_dora_val128_freegreedy_checkpoint-32-inference-clean_val128_bsz8_temp0_rp1p10_max3084_chatfix_4gpu/gt_vs_pred_scored.jsonl \
  --candidate-region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent_context_ring_band_interactions/candidate_regions/aux_latest_ckpt32/candidate_region_rows.jsonl \
  --checkpoint-path /data/CoordExp/outputs/stage1_2b/loss_only_instance_enumeration_ablation_active_vs_none/random_active_hp02_latest_aligner_dora_4096_128_no_newline_1epoch32_4gpu/loss-only-instance-enum-random-active-hp02-latest-aligner-dora-4096-128-no-newline-1epoch32-4gpu/v0-20260609-115128/checkpoint-32-inference-clean \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent_context_ring_band_interactions_basin_shape/aux_latest_ckpt32 \
  --image-id 139 \
  --gt-idx 17 \
  --guidance-tiers desc_x1_y1_x2 \
  --prefix-object-limits 0,all \
  --patch-layers 13,14,16,17,24,25,26,27 \
  --interventions zero:context_ring_upper_band,zero:context_ring_target_adjacent_band,zero:context_ring_lower_band,zero:context_ring_upper_plus_target_adjacent,zero:context_ring_upper_plus_lower,zero:context_ring_target_adjacent_plus_lower,zero:context_ring \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --attn-implementation auto
```

Reduction verification checked:

```text
both checkpoint roots: baseline=2, membership=26, patch=112
patched_coord_entropy present
patched_coord_mass_radius_16 present
patched_coord_mass_radius_64 present
aux upper+target-adjacent all: top1 796, mass r16 0.281250, mass r64 0.846251
no-aligner upper+lower all: top1 999, mass r16 0.244280, mass r64 0.611923
```
