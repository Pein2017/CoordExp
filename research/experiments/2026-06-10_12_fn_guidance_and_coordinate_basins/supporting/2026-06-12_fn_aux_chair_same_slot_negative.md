# FN Aux Chair Same-Slot Negative

Date: 2026-06-12

## Scope

This probe tests the second aux `same_slot_alternative_available` row from the
probe-family selector.

Question:

- For aux checkpoint chair `image=139, gt=13`, does the successful
  `all,desc_x1` state contain a same-next-slot hidden-state repair for the
  failed `0,desc_x1` continuation?

Answer: no. Although the decode panel says full-prefix `desc_x1` succeeds and
empty-prefix `desc_x1` fails, the prompt-end y1 readout is already slightly
better for the empty prefix. Patching full-prefix state into empty-prefix state
does not repair the y1 basin; it mildly hurts it.

## Case

Checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/loss_only_instance_enumeration_ablation_active_vs_none/random_active_hp02_latest_aligner_dora_4096_128_no_newline_1epoch32_4gpu/loss-only-instance-enum-random-active-hp02-latest-aligner-dora-4096-128-no-newline-1epoch32-4gpu/v0-20260609-115128/checkpoint-32
```

Case:

```text
image_id = 139
gt_idx = 13
desc = chair
target = [495, 514, 530, 543]
```

Selector row:

```text
transition_class = coord_slot_unlock
causal_probe_family = same_slot_alternative_available
same_slot_probe_source_prefix = all
same_slot_probe_target_prefix = 0
best_unlock_tier = desc_x1_y1
```

Decode panel:

- `prefix=0,desc_x1`: IoU `0.4708`, below rescue threshold.
- `prefix=all,desc_x1`: IoU `0.6785`, successful.
- `prefix=0,desc_x1_y1`: IoU `0.6408`, successful.
- `prefix=all,desc_x1_y1`: IoU `0.7955`, successful.

## Artifact

Root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/aux_latest_ckpt32_chair13_same_slot_full_to_empty_l24_27
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

Same-next-slot patch:

```text
source = all,desc_x1
target = 0,desc_x1
next slot = y1
target bin = 514
layers = 24,25,26,27
sites = decoder_layer,mlp,self_attn
```

This mirrors the aux vase same-slot full-to-empty probe.

## Condition Readout

| condition | target rank | target prob | top1 | top1 distance |
| --- | ---: | ---: | ---: | ---: |
| `0,desc_x1` | 18 | 0.022752 | 509 | 5 |
| `all,desc_x1` | 20 | 0.020376 | 509 | 5 |

This is already a warning sign: the source condition that succeeds in decode
has a worse prompt-end target-y1 rank than the target condition that fails in
decode.

## Residual Patch Result

Every tested patch row is flat or negative:

| site | layer | patched rank | rank recovery | patched top1 | prob delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `decoder_layer` | 24 | 19 | -1 | 509 | -0.001750 |
| `decoder_layer` | 25 | 20 | -2 | 509 | -0.001215 |
| `decoder_layer` | 26 | 20 | -2 | 509 | -0.002212 |
| `decoder_layer` | 27 | 20 | -2 | 509 | -0.002376 |
| `mlp` | 24 | 19 | -1 | 515 | -0.001295 |
| `mlp` | 25 | 19 | -1 | 509 | -0.000586 |
| `mlp` | 26 | 19 | -1 | 509 | -0.000301 |
| `mlp` | 27 | 19 | -1 | 509 | -0.000021 |
| `self_attn` | 24 | 18 | 0 | 509 | -0.000906 |
| `self_attn` | 25 | 18 | 0 | 509 | -0.000370 |
| `self_attn` | 26 | 19 | -1 | 515 | -0.000738 |
| `self_attn` | 27 | 19 | -1 | 509 | -0.000399 |

Direction rows are also weak and mostly not target-specific:

| site | layer | target delta at 514 | wrong/top1 delta at 509 | margin change |
| --- | ---: | ---: | ---: | ---: |
| `decoder_layer` | 27 | +0.1250 | +0.2500 | -0.1250 |
| `mlp` | 27 | +0.1250 | +0.2500 | -0.1250 |
| `self_attn` | 26 | +0.4531 | +0.4375 | +0.0156 |
| `self_attn` | 27 | -0.0312 | -0.0625 | +0.0312 |

## Mechanism Read

This is a useful negative companion to aux vase:

1. `same_slot_alternative_available` is a valid routing label for patch design,
   but it is not a claim that the same-slot patch will be positive.
2. For aux chair, the decode success difference between `prefix=0` and
   `prefix=all` is not visible as a prompt-end y1 basin repair. The source
   prompt-end y1 readout is actually slightly worse.
3. The full-prefix decode success likely depends on later autoregressive
   continuation dynamics, duplicate rejection, y2/extent behavior, or local
   box-shape completion rather than an earlier portable y1 hidden-state repair.
4. Aux same-slot effects now split:
   - aux vase: shallow positive same-slot refinement;
   - aux chair: no prompt-end same-slot repair despite decode success.

Compared with the parent vase prefix lock, both aux cases look weaker and more
decode-dynamics dependent. The parent still has the clearest hidden-state
basin-repair mechanism.

## Next Hook

The next useful FN contrast is the hard book row rather than more shallow aux
same-slot patching:

```text
checkpoint = aux_latest_ckpt32_guarded or no_aligner_parent_ckpt3668
image = 139
gt = 17
desc = book
family = hard_no_rescue
```

For book, the question should be y2/extent: even after target `x1,y1,x2`, the
model predicts a short box. That is better aligned with the user's original
false-negative question: whether the missing object is visually available but
not synchronized, or genuinely weak/hidden in the visual evidence.

## Verification

Run command:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_duplication_phase4_fn_coordslot_logit_probe.py \
  --guidance-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/aux_latest_ckpt32_guarded/fn_visibility_guidance_probe_rows.jsonl \
  --gt-vs-pred-scored-path /data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_loss_only_aux_latest_aligner_dora_val128_freegreedy_checkpoint-32-inference-clean_val128_bsz8_temp0_rp1p10_max3084_chatfix_4gpu/gt_vs_pred_scored.jsonl \
  --checkpoint-path /data/CoordExp/outputs/stage1_2b/loss_only_instance_enumeration_ablation_active_vs_none/random_active_hp02_latest_aligner_dora_4096_128_no_newline_1epoch32_4gpu/loss-only-instance-enum-random-active-hp02-latest-aligner-dora-4096-128-no-newline-1epoch32-4gpu/v0-20260609-115128/checkpoint-32 \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/aux_latest_ckpt32_chair13_same_slot_full_to_empty_l24_27 \
  --image-id 139 \
  --gt-idx 13 \
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

Completed counts:

```text
condition_row_count = 6
layer_row_count = 174
hidden_delta_row_count = 58
direction_row_count = 12
patch_row_count = 12
```

Artifact verification checked the required files, counts, and that all patch
rank recoveries are `<= 0`.
