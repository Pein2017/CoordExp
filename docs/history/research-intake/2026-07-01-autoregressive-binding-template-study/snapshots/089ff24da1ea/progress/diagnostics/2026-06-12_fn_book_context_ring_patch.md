# FN Book Context-Ring Patch

Date: 2026-06-12

## Scope

This probe widens the prior visual-token patch for the hard book false negative:

```text
image_id = 139
gt_idx = 17
desc = book
target y2 = 826
prompt tier = desc_x1_y1_x2
prefixes = 0, all
```

The previous visual-token patch showed that the exact local tokens
`[922]` and `[961]` have only weak causal leverage over the short y2 basin.
This run asks whether the broader 28-token `context_ring`, which dominated the
attention atlas, has stronger causal leverage.

## Artifacts

No-aligner parent root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent_context_ring/no_aligner_parent_ckpt3668
```

Aux checkpoint root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent_context_ring/aux_latest_ckpt32
```

Shared reducer:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent_context_ring/book17_context_ring_patch_reduction.json
```

The intervention is:

```text
zero:context_ring
```

Patch layers:

```text
13,14,16,17,24,25,26,27
```

Counts per checkpoint:

```text
baseline_row_count = 2
membership_row_count = 14
patch_row_count = 16
```

## Context Tokens

The context ring is the same in both checkpoints:

```text
[802,803,804,805,806,841,842,843,844,845,880,881,882,884,
 919,920,921,923,958,959,960,961,962,997,998,999,1000,1001]
```

It includes the neighbor lower token `961`, nearby tokens around the target,
and lower/background tokens below the books.

## Baselines

| checkpoint | prefix | target rank | target prob | top1 | top1 distance | top bins |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| no-aligner parent | `0` | 131 | 0.000696 | 737 | 89 | `[737,736,739,743,741,740,730,734]` |
| no-aligner parent | `all` | 130 | 0.000650 | 739 | 87 | `[739,734,737,743,736,730,740,741]` |
| aux checkpoint | `0` | 161 | 0.000219 | 730 | 96 | `[730,731,722,734,723,718,724,725]` |
| aux checkpoint | `all` | 155 | 0.000278 | 734 | 92 | `[734,731,730,736,723,722,725,726]` |

## Patch Results

Best rank movements:

| checkpoint | prefix | layer | baseline rank | patched rank | recovery | prob delta | patched top1 | top1 distance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-aligner parent | `all` | 13 | 130 | 9 | 121 | +0.007869 | 999 | 173 |
| no-aligner parent | `0` | 13 | 131 | 88 | 43 | +0.002298 | 734 | 92 |
| no-aligner parent | `all` | 16 | 130 | 104 | 26 | +0.002546 | 730 | 96 |
| aux checkpoint | `all` | 13 | 155 | 48 | 107 | +0.007075 | 999 | 173 |
| aux checkpoint | `all` | 14 | 155 | 95 | 60 | +0.003230 | 778 | 48 |
| aux checkpoint | `0` | 13 | 161 | 108 | 53 | +0.003049 | 769 | 57 |
| aux checkpoint | `all` | 16 | 155 | 107 | 48 | +0.002194 | 769 | 57 |
| aux checkpoint | `0` | 16 | 161 | 118 | 43 | +0.002063 | 766 | 60 |

Compared with the prior token-local run, the context-ring effect is much larger:

| checkpoint | best local-token recovery | best context-ring recovery |
| --- | ---: | ---: |
| no-aligner parent | 5 | 121 |
| aux checkpoint | 6 | 107 |

But the best rank recovery is not the same thing as rescue. The top1 behavior is
often distorted:

- no-aligner `all`, layer 13 moves target rank to `9`, but top1 becomes `999`;
- aux `all`, layer 13 moves target rank to `48`, but top1 becomes `999`;
- aux layer 14/16 and aux prefix `0` layer 13/16 move top1 to `766..778`,
  closer to the true `826` but still far away.

Late layers `24..27` are mostly weak or flat, matching the prior local-token
patch result.

## Mechanism Read

This is a meaningful causal update:

1. The hard book y2 basin is sensitive to broad visual-neighborhood state in
   mid layers, especially `13`, `14`, and `16`.
2. The local edge/neighbor tokens `[922]` and `[961]` are too small a handle;
   the broader context ring carries much more causal mass.
3. The intervention still does not cleanly restore y2. It either leaves the
   short-box basin mostly intact, shifts toward mid-700 bins, or creates an
   over-extended `999` basin.
4. Aux is more responsive to context-ring ablation than to local-token ablation,
   consistent with the attention result that aux uses the visual neighborhood
   more strongly but does not convert it into a stable target y2 coordinate.

The emerging picture is not visual blindness and not a single missing edge
token. It is broader visual-context entanglement plus coordinate-basin
instability: the model has access to neighborhood evidence, but the y2 slot is
pulled between a short-box basin and an over-extended/context boundary basin.

## Next Hook

The most useful next direction is a targeted decomposition of the context ring:

```text
upper/context row tokens: 802..845
target-adjacent row tokens: 880..923
neighbor/lower row tokens: 958..1001
```

Two concrete tests:

1. Split the ring into row bands and patch each band separately to localize the
   short-box versus over-extension directions.
2. Join context-ring overlap and same-desc overlap features across all
   `hard_no_rescue` selector rows to test whether this is a repeated hard-FN
   pattern rather than a one-off book geometry.

## Verification

Commands run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_duplication_phase4_fn_visual_token_patch.py \
  --guidance-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/no_aligner_parent_ckpt3668/fn_visibility_guidance_probe_rows.jsonl \
  --gt-vs-pred-scored-path /data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_no_aligner_parent_val128_freegreedy_ckpt3668_val128_bsz8_temp0_rp1p10_max3084_chatfix_8gpu/gt_vs_pred_scored.jsonl \
  --candidate-region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_attention_book17_y2_extent/no_aligner_parent_ckpt3668/shards/shard_000-of-001/candidate_region_rows.jsonl \
  --checkpoint-path /data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668 \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent_context_ring/no_aligner_parent_ckpt3668 \
  --image-id 139 \
  --gt-idx 17 \
  --guidance-tiers desc_x1_y1_x2 \
  --prefix-object-limits 0,all \
  --patch-layers 13,14,16,17,24,25,26,27 \
  --interventions zero:context_ring \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --attn-implementation auto
```

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/analysis/run_autoregressive_duplication_phase4_fn_visual_token_patch.py \
  --guidance-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/aux_latest_ckpt32_guarded/fn_visibility_guidance_probe_rows.jsonl \
  --gt-vs-pred-scored-path /data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_loss_only_aux_latest_aligner_dora_val128_freegreedy_checkpoint-32-inference-clean_val128_bsz8_temp0_rp1p10_max3084_chatfix_4gpu/gt_vs_pred_scored.jsonl \
  --candidate-region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_attention_book17_y2_extent/aux_latest_ckpt32/shards/shard_000-of-001/candidate_region_rows.jsonl \
  --checkpoint-path /data/CoordExp/outputs/stage1_2b/loss_only_instance_enumeration_ablation_active_vs_none/random_active_hp02_latest_aligner_dora_4096_128_no_newline_1epoch32_4gpu/loss-only-instance-enum-random-active-hp02-latest-aligner-dora-4096-128-no-newline-1epoch32-4gpu/v0-20260609-115128/checkpoint-32-inference-clean \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent_context_ring/aux_latest_ckpt32 \
  --image-id 139 \
  --gt-idx 17 \
  --guidance-tiers desc_x1_y1_x2 \
  --prefix-object-limits 0,all \
  --patch-layers 13,14,16,17,24,25,26,27 \
  --interventions zero:context_ring \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --attn-implementation auto
```

Artifact verification checked:

- both checkpoint roots have baseline `2`, membership `14`, patch `16`;
- context ring has 28 visual tokens;
- best context-ring rank recovery exceeds prior token-local recovery;
- no context-ring patch is a clean y2 rescue because patched top1 remains
  either short/mid-range or over-extended.
