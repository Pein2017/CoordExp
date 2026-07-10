# FN Vase Same-Slot Aux vs Parent

Date: 2026-06-12

## Scope

This note compares same-next-slot hidden-state patches for the vase FN across
the no-aligner parent and the auxiliary-loss checkpoint.

The previous cross-slot negative showed that patching `desc_x1_y1` state into
`desc_x1` is not a valid repair mechanism for the missing `y1` slot. The right
causal object is a same-slot contrast where source and target both predict the
same next coordinate.

## Cases

Shared case:

```text
image_id = 139
gt_idx = 7
desc = vase
target = [526, 468, 542, 508]
next slot under desc_x1 = y1
target y1 bin = 468
```

Parent checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668
```

Aux checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/loss_only_instance_enumeration_ablation_active_vs_none/random_active_hp02_latest_aligner_dora_4096_128_no_newline_1epoch32_4gpu/loss-only-instance-enum-random-active-hp02-latest-aligner-dora-4096-128-no-newline-1epoch32-4gpu/v0-20260609-115128/checkpoint-32
```

## Artifact Roots

Parent same-slot prefix repair:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/no_aligner_parent_ckpt3668_vase_prefix_coordslot_direction_patch_sites_l24_27
```

Aux same-slot full-to-empty probe:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/aux_latest_ckpt32_vase_same_slot_full_to_empty_l24_27
```

Cross-slot negative reference:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/no_aligner_parent_ckpt3668_vase_fullprefix_coord_context_patch_l24_27
```

## Parent Same-Slot Repair

Contrast:

```text
source = 0,desc_x1
target = all,desc_x1
```

The parent full-prefix condition installs a wrong vertical basin:

| condition | target rank | target prob | top1 | top1 distance |
| --- | ---: | ---: | ---: | ---: |
| `all,desc_x1` | 52 | 0.005097 | 501 | 33 |

The same-slot source-to-target patch strongly repairs the failed `y1=468`
readout:

| site | layer | patched rank | rank recovery | patched top1 | prob delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mlp` | 26 | 19 | 33 | 483 | +0.011056 |
| `mlp` | 27 | 12 | 40 | 479 | +0.009618 |
| `self_attn` | 27 | 38 | 14 | 486 | +0.005325 |
| `self_attn_input` | 27 | 38 | 14 | 483 | +0.005244 |

The direction decomposition gives a cleaner basin story:

| site | layer | target delta at 468 | wrong-bin delta at 501 | margin change |
| --- | ---: | ---: | ---: | ---: |
| `mlp` | 26 | +1.2500 | +0.0156 | +1.2344 |
| `mlp` | 27 | +0.2500 | -1.6562 | +1.9062 |

Read: parent same-slot repair is a strong MLP-side late-layer prefix-state
repair. Layer 26 lifts target-local vertical bins; layer 27 suppresses the
wrong full-prefix basin around `499-503`.

## Aux Same-Slot Full-to-Empty Probe

Decode behavior in the aux panel was flipped relative to the parent:

- `prefix=0,desc_x1` failed with IoU `0.2563`.
- `prefix=all,desc_x1` succeeded with IoU `0.5901`.

So the same-slot hidden-state probe used:

```text
source = all,desc_x1
target = 0,desc_x1
layers = 24,25,26,27
sites = decoder_layer,mlp,self_attn
```

Prompt-end readout:

| condition | target rank | target prob | top1 | top1 distance |
| --- | ---: | ---: | ---: | ---: |
| `0,desc_x1` | 25 | 0.014093 | 483 | 15 |
| `all,desc_x1` | 23 | 0.016698 | 480 | 12 |

The aux source is better than the target, but only shallowly. Patching it into
the empty-prefix target gives modest rank recovery:

| site | layer | patched rank | rank recovery | patched top1 | prob delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `decoder_layer` | 24 | 18 | 7 | 483 | +0.003821 |
| `decoder_layer` | 25 | 18 | 7 | 476 | +0.004277 |
| `decoder_layer` | 26 | 17 | 8 | 476 | +0.004286 |
| `decoder_layer` | 27 | 23 | 2 | 480 | +0.002605 |
| `mlp` | 25 | 17 | 8 | 476 | +0.003761 |
| `mlp` | 26 | 18 | 7 | 476 | +0.003624 |
| `mlp` | 27 | 23 | 2 | 476 | +0.002083 |
| `self_attn` | 25 | 20 | 5 | 483 | +0.000929 |

Direction rows are similarly mild:

| site | layer | target delta at 468 | wrong/top1 delta at 483 | margin change |
| --- | ---: | ---: | ---: | ---: |
| `decoder_layer` | 26 | +0.2188 | +0.0938 | +0.1250 |
| `decoder_layer` | 27 | +0.5625 | +0.4375 | +0.1250 |
| `mlp` | 26 | +0.4062 | +0.3125 | +0.0938 |
| `mlp` | 27 | +0.6562 | +0.5312 | +0.1250 |

Read: the aux full-prefix state provides a small same-slot improvement, mostly
through decoder/MLP layers 25-26, but it is not the same strong basin-reset
mechanism seen in the parent.

## Cross-Slot Guardrail

The cross-slot parent probe:

```text
source = all,desc_x1_y1
target = all,desc_x1
```

hurt every tested patch row. Whole-layer patches drove top1 toward `548`, near
the source's `x2` basin, and pushed the target `y1=468` rank from `52` to
around `155-157`.

This makes the interpretation sharper:

- same-slot patches can test whether hidden state repairs a missing coordinate
  basin;
- cross-slot coord forcing tests downstream coordinate accessibility, not
  whether the earlier missing slot was already latent in the residual stream.

## Mechanism Split

The vase case now separates three related but distinct phenomena:

1. **Parent prefix-state lock-in:** full prefix creates a wrong `y1` basin;
   empty-prefix `desc_x1` state contains an MLP-side late-layer repair signal.
2. **Aux shallow full-prefix refinement:** full prefix is slightly better than
   empty prefix and gives modest same-slot repair, but the effect is much
   weaker than the parent prefix-lock repair.
3. **Downstream coordinate accessibility:** once `y1` is supplied, both parent
   and aux can often complete `x2/y2`, but that does not imply an earlier
   portable `y1` repair vector.

This is evidence against a single FN explanation. The same surface behavior
`desc_x1_y1 rescues` can arise from downstream coordinate accessibility, while
same-slot patching is needed to diagnose actual prefix-state repair.

## Next Hook

For aux, the more attractive follow-up is not more vase patching. The better
targets from the transition selector are:

- aux person image `1353`, GT `3`: `desc_x1` fails badly under both prefixes,
  but `desc_x1_y1` succeeds. This looks like a downstream coordinate
  accessibility case rather than same-slot prefix repair.
- aux book image `139`, GT `17`: hard no-rescue under both prefixes, useful as
  the extent/visibility contrast.

A useful next reducer would split transition-selector rows into:

- same-slot alternatives available;
- cross-slot only;
- hard no-rescue.

That would keep future hidden-state probes from mixing incompatible causal
questions.

## Verification

Aux run command:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_duplication_phase4_fn_coordslot_logit_probe.py \
  --guidance-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/aux_latest_ckpt32_guarded/fn_visibility_guidance_probe_rows.jsonl \
  --gt-vs-pred-scored-path /data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_loss_only_aux_latest_aligner_dora_val128_freegreedy_checkpoint-32-inference-clean_val128_bsz8_temp0_rp1p10_max3084_chatfix_4gpu/gt_vs_pred_scored.jsonl \
  --checkpoint-path /data/CoordExp/outputs/stage1_2b/loss_only_instance_enumeration_ablation_active_vs_none/random_active_hp02_latest_aligner_dora_4096_128_no_newline_1epoch32_4gpu/loss-only-instance-enum-random-active-hp02-latest-aligner-dora-4096-128-no-newline-1epoch32-4gpu/v0-20260609-115128/checkpoint-32 \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/aux_latest_ckpt32_vase_same_slot_full_to_empty_l24_27 \
  --image-id 139 \
  --gt-idx 7 \
  --guidance-tiers desc_x1,desc_x1_y1,desc_x1_y1_x2 \
  --prefix-object-limits 0,all \
  --layers all \
  --patch-layers 24,25,26,27 \
  --patch-sites decoder_layer,mlp,self_attn \
  --patch-source all,desc_x1 \
  --patch-target 0,desc_x1 \
  --top-k 8 \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --attn-implementation auto
```

The run completed with:

```text
condition_row_count = 6
layer_row_count = 174
hidden_delta_row_count = 58
direction_row_count = 12
patch_row_count = 12
```
