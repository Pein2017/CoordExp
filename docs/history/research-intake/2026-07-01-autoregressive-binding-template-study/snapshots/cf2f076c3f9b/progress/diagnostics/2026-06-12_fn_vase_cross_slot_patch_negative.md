# FN Vase Cross-Slot Patch Negative

Date: 2026-06-12

## Scope

This is the first hidden-state follow-up to the FN guidance transition selector.

Question:

- For the no-aligner parent vase case, does the successful full-prefix
  `desc_x1_y1` coord-context state contain a portable residual direction that
  repairs the failed full-prefix `desc_x1` `y1` basin?

The answer in this panel is no. Patching across these two tiers makes the
`desc_x1` target-`y1` readout worse, not better.

## Case

Checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668
```

Case:

```text
image_id = 139
gt_idx = 7
desc = vase
target = [526, 468, 542, 508]
```

Guidance transition from the decode panel:

- `all,desc_x1`: generated `[526, 497, 553, 540]`, IoU `0.1083`, failed.
- `all,desc_x1_y1`: generated `[526, 468, 546, 520]`, IoU `0.6154`, rescued.
- `all,desc_x1_y1_x2`: generated `[526, 468, 542, 516]`, IoU `0.8333`, rescued.

## Artifact

Root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/no_aligner_parent_ckpt3668_vase_fullprefix_coord_context_patch_l24_27
```

Files:

```text
fn_coordslot_logit_condition_rows.jsonl
fn_coordslot_logit_layer_rows.jsonl
fn_coordslot_hidden_delta_rows.jsonl
fn_coordslot_direction_rows.jsonl
fn_coordslot_residual_patch_rows.jsonl
phase4_fn_coordslot_logit_probe_summary.json
phase4_fn_coordslot_logit_probe_report.md
```

## Probe Design

Conditions:

- `all,desc_x1`: known coords `[526]`, next slot `y1`, target bin `468`.
- `all,desc_x1_y1`: known coords `[526,468]`, next slot `x2`, target bin `542`.
- `all,desc_x1_y1_x2`: known coords `[526,468,542]`, next slot `y2`, target bin `508`.

Patch:

```text
source = all,desc_x1_y1
target = all,desc_x1
layers = 24,25,26,27
sites = decoder_layer,mlp,self_attn
```

This is intentionally a cross-slot patch: the source prompt-end state is
conditioned on a known `y1` and is preparing the next `x2`; the target state is
preparing the missing `y1`.

## Condition Readout

| condition | next slot | target bin | target rank | target prob | top1 | top1 distance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `all,desc_x1` | `y1` | 468 | 52 | 0.005097 | 501 | 33 |
| `all,desc_x1_y1` | `x2` | 542 | 14 | 0.032332 | 548 | 6 |
| `all,desc_x1_y1_x2` | `y2` | 508 | 15 | 0.026050 | 516 | 8 |

The decode success after adding `y1` is therefore not the same readout task:
it moves the model from a failed `y1` prediction into an `x2/y2` completion
problem where the remaining coordinate slots are locally accessible.

## Residual Patch Result

All tested cross-slot patches hurt the target `y1=468` rank in the failed
`all,desc_x1` condition.

| site | layer | patched rank | rank recovery | patched top1 | prob delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `decoder_layer` | 24 | 157 | -105 | 548 | -0.005083 |
| `decoder_layer` | 25 | 155 | -103 | 548 | -0.005081 |
| `decoder_layer` | 26 | 155 | -103 | 548 | -0.005082 |
| `decoder_layer` | 27 | 156 | -104 | 548 | -0.005081 |
| `mlp` | 24 | 102 | -50 | 527 | -0.003694 |
| `mlp` | 25 | 60 | -8 | 503 | -0.001632 |
| `mlp` | 26 | 100 | -48 | 539 | -0.004546 |
| `mlp` | 27 | 132 | -80 | 548 | -0.004909 |
| `self_attn` | 24 | 59 | -7 | 503 | -0.001516 |
| `self_attn` | 25 | 55 | -3 | 498 | -0.000004 |
| `self_attn` | 26 | 55 | -3 | 498 | +0.000069 |
| `self_attn` | 27 | 76 | -24 | 527 | -0.001276 |

The whole-layer patches are especially diagnostic: they push top1 to `548`,
which is near the source condition's `x2` target basin, not the failed target's
`y1=468` basin.

## Direction Readout

Selected source-minus-target direction rows:

| site | layer | target-delta at 468 | wrong/top1-delta at 501 | margin change | source-site top1 | target-site top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `decoder_layer` | 27 | -2.7500 | -3.3125 | +0.5625 | 548 | 498 |
| `mlp` | 25 | +0.5908 | +0.0703 | +0.5205 | 218 | 530 |
| `mlp` | 27 | -1.5312 | -0.7812 | -0.7500 | 546 | 472 |
| `self_attn` | 27 | -3.2578 | -7.4062 | +4.1484 | 650 | 504 |

Some directions suppress the wrong `501` basin more than the target bin, but
the intervention is still not causally sufficient. The actual patched forward
passes reduce target probability and do not restore the `468` basin.

## Mechanism Read

This is a useful negative result for the FN taxonomy:

1. A `coord_slot_unlock` decode transition does not imply the successful later
   prompt-end residual is a portable repair vector for the earlier missing
   coordinate slot.
2. The successful `desc_x1_y1` continuation is better interpreted as changing
   the autoregressive task: once `y1` is supplied, the model can complete
   `x2/y2` from a locally accessible box basin.
3. The failed `desc_x1` condition still requires a same-slot mechanism probe:
   repair should compare states that both predict `y1`, such as
   `prefix=0,desc_x1` into `prefix=all,desc_x1`, not `desc_x1_y1` into
   `desc_x1`.
4. This cleanly separates two mechanisms:
   - **prefix-state lock-in:** wrong `y1` basin under the full prefix;
   - **downstream coordinate accessibility:** after target `y1` is supplied,
     `x2/y2` become easy to complete.

This also prevents an overclaim: the `desc_x1_y1` rescue proves the object is
reachable in the conditional generative space, but it does not prove the
model's hidden state already contains an unexpressed correct `y1` plan.

## Next Hook

For hidden-state probing, use same-next-slot contrasts:

- parent vase prefix-state lock:
  `source=0,desc_x1`, `target=all,desc_x1`, next slot `y1`.
- aux coord-slot unlock rows should first be split into:
  - same-slot prefix contrasts when a successful alternative prefix exists;
  - downstream-completion contrasts when only target coordinate forcing
    succeeds.
- hard book rows remain the contrast where even target `x1,y1,x2` forcing
  cannot recover `y2`/extent.

The existing parent vase same-slot artifact already supports the first bullet:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/no_aligner_parent_ckpt3668_vase_prefix_coordslot_direction_patch_sites_l24_27
```

## Verification

Run command:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_duplication_phase4_fn_coordslot_logit_probe.py \
  --guidance-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/no_aligner_parent_ckpt3668/fn_visibility_guidance_probe_rows.jsonl \
  --gt-vs-pred-scored-path /data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_no_aligner_parent_val128_freegreedy_ckpt3668_val128_bsz8_temp0_rp1p10_max3084_chatfix_8gpu/gt_vs_pred_scored.jsonl \
  --checkpoint-path /data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668 \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/no_aligner_parent_ckpt3668_vase_fullprefix_coord_context_patch_l24_27 \
  --image-id 139 \
  --gt-idx 7 \
  --guidance-tiers desc_x1,desc_x1_y1,desc_x1_y1_x2 \
  --prefix-object-limits all \
  --layers all \
  --patch-layers 24,25,26,27 \
  --patch-sites decoder_layer,mlp,self_attn \
  --patch-source all,desc_x1_y1 \
  --patch-target all,desc_x1 \
  --top-k 8 \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --attn-implementation auto
```

The run completed with:

```text
condition_row_count = 3
layer_row_count = 87
hidden_delta_row_count = 29
direction_row_count = 12
patch_row_count = 12
```
