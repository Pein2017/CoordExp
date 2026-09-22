# FN Book Visual-Token Patch Negative

Date: 2026-06-12

## Scope

This probe tests whether the visual-token overlap found in the book attention
atlas is causally sufficient to move the hard `y2` extent basin.

Case:

```text
image_id = 139
gt_idx = 17
desc = book
target y2 = 826
prompt tier = desc_x1_y1_x2
prefixes = 0, all
```

Prior evidence:

- the coordslot probe showed hard short-box predictions around `730..739`;
- the attention atlas showed `target_gt == shared_book_overlap == [883,922]`;
- the neighboring same-desc book adds one lower token `[961]`;
- aux attends more strongly to target/shared visual tokens than no-aligner, but
  still has worse y2 rank.

This run patches those visual tokens at decoder-layer residual outputs and
measures the y2 coordinate distribution at the prompt boundary.

## Implementation

Added research-local probe code:

```text
src/analysis/autoregressive_duplication_mechanism/phase4_fn_visual_token_patch.py
scripts/analysis/run_autoregressive_duplication_phase4_fn_visual_token_patch.py
```

The script reuses the same prompt construction as the FN coordslot probe, finds
the actual visual-token span from processor inputs, maps candidate regions to
visual token indices, hooks selected decoder layers, and reads the next coord
slot logits.

Default interventions:

```text
zero:target_y2_edge_band
zero:same_desc_beyond_target_y2
zero:shared_book_overlap
copy:same_desc_beyond_target_y2->target_y2_edge_band
copy:target_y2_edge_band->same_desc_beyond_target_y2
```

Patch layers:

```text
13,14,16,17,24,25,26,27
```

## Artifacts

No-aligner parent root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent/no_aligner_parent_ckpt3668
```

Aux checkpoint root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent/aux_latest_ckpt32
```

Shared reducer:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent/book17_visual_token_patch_reduction.json
```

Each root contains:

```text
fn_visual_token_patch_baseline_rows.jsonl
fn_visual_token_patch_membership_rows.jsonl
fn_visual_token_patch_rows.jsonl
phase4_fn_visual_token_patch_summary.json
```

Counts per checkpoint:

```text
baseline_row_count = 2
membership_row_count = 14
patch_row_count = 80
```

## Baselines

| checkpoint | prefix | target rank | target prob | top1 | top1 distance | top bins |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| no-aligner parent | `0` | 131 | 0.000696 | 737 | 89 | `[737,736,739,743,741,740,730,734]` |
| no-aligner parent | `all` | 130 | 0.000650 | 739 | 87 | `[739,734,737,743,736,730,740,741]` |
| aux checkpoint | `0` | 161 | 0.000219 | 730 | 96 | `[730,731,722,734,723,718,724,725]` |
| aux checkpoint | `all` | 155 | 0.000278 | 734 | 92 | `[734,731,730,736,723,722,725,726]` |

These reproduce the earlier hard y2 extent basin.

## Visual Membership

Both checkpoints resolve the same visual-token memberships:

| region | tokens |
| --- | --- |
| `target_y2_edge_band` | `[922]` |
| `same_desc_beyond_target_y2` | `[961]` |
| `shared_book_overlap` | `[883,922]` |
| `target_gt` | `[883,922]` |
| `same_desc_gt` | `[883,922,961]` |
| `context_ring` | 28 neighboring tokens |

## Patch Results

Best rank movements:

| checkpoint | prefix | layer | intervention | baseline rank | patched rank | recovery | patched top1 | top1 distance |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| no-aligner parent | `all` | 13 | `zero:shared_book_overlap` | 130 | 125 | 5 | 743 | 83 |
| no-aligner parent | `all` | 13 | `zero:same_desc_beyond_target_y2` | 130 | 125 | 5 | 739 | 87 |
| no-aligner parent | `all` | 13 | `zero:target_y2_edge_band` | 130 | 129 | 1 | 739 | 87 |
| aux checkpoint | `all` | 13 | `zero:shared_book_overlap` | 155 | 149 | 6 | 734 | 92 |
| aux checkpoint | `all` | 13 | `zero:same_desc_beyond_target_y2` | 155 | 150 | 5 | 730 | 96 |
| aux checkpoint | `all` | 13 | `zero:target_y2_edge_band` | 155 | 152 | 3 | 730 | 96 |
| aux checkpoint | `0` | 13/14 | `zero:shared_book_overlap` | 161 | 158 | 3 | 734 | 92 |
| aux checkpoint | `0` | 13 | `zero:same_desc_beyond_target_y2` | 161 | 158 | 3 | 730 | 96 |

Worst movements are also small and mostly from copy-style perturbations:

- no-aligner `0`, layer 14, `copy:same_desc_beyond_target_y2->target_y2_edge_band`:
  rank `131 -> 138`;
- no-aligner `0`, layer 16, `zero:shared_book_overlap`: rank `131 -> 138`;
- aux `all`, layer 13, `copy:same_desc_beyond_target_y2->target_y2_edge_band`:
  rank `155 -> 158`;
- aux `0`, layer 13, `copy:same_desc_beyond_target_y2->target_y2_edge_band`:
  rank `161 -> 163`.

No intervention moves top1 anywhere close to `826`. The patched top1 remains in
the short-box basin:

```text
no-aligner patched top1 range in best rows: 739..743
aux patched top1 range in best rows: 730..734
```

## Mechanism Read

This is a negative causal result, but it is not empty.

1. The exact visual tokens implicated by attention do have measurable causal
   leverage: zeroing the shared/neighbor tokens at layer 13 can improve target
   rank by `5..6`.
2. The effect is too small to explain the missing object: ranks remain
   `125..152`, and top1 stays `83..96` bins away from the true y2.
3. Zeroing often helps more than copying, which suggests the local visual
   neighborhood may contribute noise or short-box anchoring rather than a clean
   missing lower-edge signal.
4. The aux checkpoint again shows stronger attention but not a stronger usable
   coordinate basin. Its best visual-token patch still leaves y2 at rank `149`.

So the hard book FN is not explained by a single target-edge visual token being
available but ignored. The stronger picture is distributed downstream
coordinate-basin attraction: coarse/overlapping visual evidence reaches the
model, but the y2 slot is attracted to a short-box basin that these local
visual-token interventions cannot dislodge.

## Next Hook

The next promising direction is broader than single-token patching:

1. Patch or ablate the whole `context_ring` and compare with the specific
   `[922]/[961]` interventions.
2. Test a token-separable hard-FN object to see whether weak causal leverage is
   specific to overlapping same-desc books.
3. Join visual-token overlap statistics with the `hard_no_rescue` selector rows
   to see whether hard cases are enriched for target/same-desc visual-token
   overlap.

## Verification

Commands run:

```bash
python -m py_compile \
  src/analysis/autoregressive_duplication_mechanism/phase4_fn_visual_token_patch.py \
  scripts/analysis/run_autoregressive_duplication_phase4_fn_visual_token_patch.py
```

No-aligner run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_duplication_phase4_fn_visual_token_patch.py \
  --guidance-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/no_aligner_parent_ckpt3668/fn_visibility_guidance_probe_rows.jsonl \
  --gt-vs-pred-scored-path /data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_no_aligner_parent_val128_freegreedy_ckpt3668_val128_bsz8_temp0_rp1p10_max3084_chatfix_8gpu/gt_vs_pred_scored.jsonl \
  --candidate-region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_attention_book17_y2_extent/no_aligner_parent_ckpt3668/shards/shard_000-of-001/candidate_region_rows.jsonl \
  --checkpoint-path /data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668 \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent/no_aligner_parent_ckpt3668 \
  --image-id 139 \
  --gt-idx 17 \
  --guidance-tiers desc_x1_y1_x2 \
  --prefix-object-limits 0,all \
  --patch-layers 13,14,16,17,24,25,26,27 \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --attn-implementation auto
```

Aux run:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/analysis/run_autoregressive_duplication_phase4_fn_visual_token_patch.py \
  --guidance-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/aux_latest_ckpt32_guarded/fn_visibility_guidance_probe_rows.jsonl \
  --gt-vs-pred-scored-path /data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_loss_only_aux_latest_aligner_dora_val128_freegreedy_checkpoint-32-inference-clean_val128_bsz8_temp0_rp1p10_max3084_chatfix_4gpu/gt_vs_pred_scored.jsonl \
  --candidate-region-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_attention_book17_y2_extent/aux_latest_ckpt32/shards/shard_000-of-001/candidate_region_rows.jsonl \
  --checkpoint-path /data/CoordExp/outputs/stage1_2b/loss_only_instance_enumeration_ablation_active_vs_none/random_active_hp02_latest_aligner_dora_4096_128_no_newline_1epoch32_4gpu/loss-only-instance-enum-random-active-hp02-latest-aligner-dora-4096-128-no-newline-1epoch32-4gpu/v0-20260609-115128/checkpoint-32-inference-clean \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visual_token_patch_book17_y2_extent/aux_latest_ckpt32 \
  --image-id 139 \
  --gt-idx 17 \
  --guidance-tiers desc_x1_y1_x2 \
  --prefix-object-limits 0,all \
  --patch-layers 13,14,16,17,24,25,26,27 \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --attn-implementation auto
```

Reducer verification checked:

- each checkpoint has `baseline_row_count=2`, `membership_row_count=14`,
  `patch_row_count=80`;
- `target_y2_edge_band=[922]`;
- `same_desc_beyond_target_y2=[961]`;
- `shared_book_overlap=[883,922]`;
- no patched top1 moves near target y2 `826`.
