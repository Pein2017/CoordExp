# FN Book Context-Band Interaction Patch

Date: 2026-06-12

## Scope

This probe follows the exact context-ring band decomposition for the hard book
false negative:

```text
image_id = 139
gt_idx = 17
desc = book
target y2 = 826
prompt tier = desc_x1_y1_x2
prefixes = 0, all
```

The previous run showed that `context_ring_upper_band` dominates the single-band
effect. This run tests whether pairwise band interactions explain the whole-ring
effect and the occasional over-extended `999` basin.

## Artifacts

Composite candidate regions:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent_context_ring_band_interactions/candidate_regions
```

No-aligner parent root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent_context_ring_band_interactions/no_aligner_parent_ckpt3668
```

Aux checkpoint root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent_context_ring_band_interactions/aux_latest_ckpt32
```

Reduction:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent_context_ring_band_interactions/book17_context_ring_band_interactions_reduction.json
```

The composite regions are represented as repeated `region_kind` rows over
disjoint exact bands. `build_patch_region_membership` unions rows with the same
kind, so this preserves target exclusion without using a single loose bbox that
would accidentally include the middle band.

Counts per checkpoint:

```text
baseline_row_count = 2
membership_row_count = 26
patch_row_count = 112
```

Patch layers:

```text
13,14,16,17,24,25,26,27
```

## Interventions

```text
zero:context_ring_upper_band
zero:context_ring_target_adjacent_band
zero:context_ring_lower_band
zero:context_ring_upper_plus_target_adjacent
zero:context_ring_upper_plus_lower
zero:context_ring_target_adjacent_plus_lower
zero:context_ring
```

Composite token counts:

| region | tokens |
| --- | ---: |
| `context_ring_upper_band` | 10 |
| `context_ring_target_adjacent_band` | 8 |
| `context_ring_lower_band` | 10 |
| `context_ring_upper_plus_target_adjacent` | 18 |
| `context_ring_upper_plus_lower` | 20 |
| `context_ring_target_adjacent_plus_lower` | 18 |
| `context_ring` | 28 |

## Results

Best rank movements for prefix `all`:

| checkpoint | intervention | best layer | rank recovery | patched rank | patched top1 | top1 distance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| no-aligner parent | `upper` | 13 | 68 | 62 | 743 | 83 |
| no-aligner parent | `upper+target-adjacent` | 16 | 85 | 45 | 743 | 83 |
| no-aligner parent | `upper+lower` | 13 | 114 | 16 | 999 | 173 |
| no-aligner parent | `target-adjacent+lower` | 13 | 34 | 96 | 743 | 83 |
| no-aligner parent | `whole ring` | 13 | 121 | 9 | 999 | 173 |
| aux checkpoint | `upper` | 13 | 77 | 78 | 778 | 48 |
| aux checkpoint | `upper+target-adjacent` | 13 | 115 | 40 | 796 | 30 |
| aux checkpoint | `upper+lower` | 13 | 79 | 76 | 778 | 48 |
| aux checkpoint | `target-adjacent+lower` | 13 | 20 | 135 | 734 | 92 |
| aux checkpoint | `whole ring` | 13 | 107 | 48 | 999 | 173 |

Best rank movements for prefix `0`:

| checkpoint | intervention | best layer | rank recovery | patched rank | patched top1 | top1 distance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| no-aligner parent | `upper` | 16 | 35 | 96 | 751 | 75 |
| no-aligner parent | `upper+target-adjacent` | 16 | 12 | 119 | 730 | 96 |
| no-aligner parent | `upper+lower` | 16 | 33 | 98 | 751 | 75 |
| no-aligner parent | `target-adjacent+lower` | 13 | 11 | 120 | 739 | 87 |
| no-aligner parent | `whole ring` | 13 | 43 | 88 | 734 | 92 |
| aux checkpoint | `upper` | 16 | 44 | 117 | 769 | 57 |
| aux checkpoint | `upper+target-adjacent` | 16 | 69 | 92 | 778 | 48 |
| aux checkpoint | `upper+lower` | 13 | 43 | 118 | 769 | 57 |
| aux checkpoint | `target-adjacent+lower` | 13 | 16 | 145 | 734 | 92 |
| aux checkpoint | `whole ring` | 13 | 53 | 108 | 769 | 57 |

## Mechanism Read

The interaction is checkpoint-specific and non-monotonic.

For the no-aligner parent, adding the lower band to the upper band recreates the
over-extension mode:

```text
upper all:       top1 743, rank recovery 68
upper+lower all: top1 999, rank recovery 114
whole ring all:  top1 999, rank recovery 121
```

So for no-aligner, lower-context interaction with the upper band is a strong
candidate source of the `999` boundary basin. Adding target-adjacent to upper
improves rank but does not move top1 out of the short/mid basin.

For the aux checkpoint, the cleanest partial rescue is not the whole ring. It is
`upper+target-adjacent`:

```text
upper all:                 top1 778, distance 48, rank recovery 77
upper+target-adjacent all: top1 796, distance 30, rank recovery 115
whole ring all:            top1 999, distance 173, rank recovery 107
```

This is the closest guided y2 readout so far, but still not a clean rescue to
`826`. It suggests aux has more usable target-adjacent/upper coordination than
the no-aligner parent, while the full context ring still includes a component
that can push the coordinate basin to the image boundary.

The result sharpens the false-negative hypothesis:

1. The hard FN is not caused by absent visual perception; broad and pairwise
   visual context perturbations strongly move the coordinate slot.
2. The coordinate basin is interaction-sensitive, not region-size monotonic.
   More visual context can improve rank while making top1 worse.
3. The aux checkpoint appears to partially synchronize upper and target-adjacent
   evidence, but not enough to stabilize the exact lower edge.
4. The no-aligner parent routes upper+lower context into a boundary/extent basin
   much more readily.

## Next Hook

This makes two follow-ups especially promising:

1. Search for other hard false negatives where `upper+target-adjacent` improves
   aux but not no-aligner, to test whether this is a general auxiliary-loss
   mechanism or a book-specific accident.
2. Probe the y2 coordinate-token basin directly around `778..826..999` under
   upper, upper+target-adjacent, and whole-ring interventions, including entropy
   and rank mass across local coordinate windows.

## Verification

Commands run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_duplication_phase4_fn_visual_token_patch.py \
  --guidance-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/no_aligner_parent_ckpt3668/fn_visibility_guidance_probe_rows.jsonl \
  --gt-vs-pred-scored-path /data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_no_aligner_parent_val128_freegreedy_ckpt3668_val128_bsz8_temp0_rp1p10_max3084_chatfix_8gpu/gt_vs_pred_scored.jsonl \
  --candidate-region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent_context_ring_band_interactions/candidate_regions/no_aligner_parent_ckpt3668/candidate_region_rows.jsonl \
  --checkpoint-path /data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668 \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent_context_ring_band_interactions/no_aligner_parent_ckpt3668 \
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
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent_context_ring_band_interactions/aux_latest_ckpt32 \
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

Artifact verification checked:

```text
aux
('all', 'zero:context_ring_upper_band') rec 77 rank 78 top1 778 dist 48
('all', 'zero:context_ring_upper_plus_target_adjacent') rec 115 rank 40 top1 796 dist 30
('all', 'zero:context_ring_upper_plus_lower') rec 79 rank 76 top1 778 dist 48
('all', 'zero:context_ring') rec 107 rank 48 top1 999 dist 173

no_aligner
('all', 'zero:context_ring_upper_band') rec 68 rank 62 top1 743 dist 83
('all', 'zero:context_ring_upper_plus_target_adjacent') rec 85 rank 45 top1 743 dist 83
('all', 'zero:context_ring_upper_plus_lower') rec 114 rank 16 top1 999 dist 173
('all', 'zero:context_ring') rec 121 rank 9 top1 999 dist 173

VERIFIED band interaction reducer invariants and headline effects
```
