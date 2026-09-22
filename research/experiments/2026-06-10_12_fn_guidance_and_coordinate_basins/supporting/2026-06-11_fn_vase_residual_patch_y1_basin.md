# FN Vase Residual Patch for the Y1 Basin

Date: 2026-06-11

## Scope

Causal follow-up to the prompt-end FN vase coord-slot logit probe.

The previous probe showed that image `139`, GT `7`, desc `vase`, full-prefix
`desc_x1` already has a bad next-`y1` basin before generation:

- target `y1=468`;
- full prefix + `desc_x1`: target rank `52`, top1 `501`;
- empty prefix + `desc_x1`: target rank `12`, top1 `472`.

This patch smoke asks whether the empty-prefix prompt-end residual state is
sufficient to move the full-prefix prompt-end `y1` distribution back toward the
target basin.

## Inputs

Guidance rows:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/no_aligner_parent_ckpt3668/fn_visibility_guidance_probe_rows.jsonl
```

Scored rollout:

```text
/data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_no_aligner_parent_val128_freegreedy_ckpt3668_val128_bsz8_temp0_rp1p10_max3084_chatfix_8gpu/gt_vs_pred_scored.jsonl
```

Checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668
```

Run command:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_duplication_phase4_fn_coordslot_logit_probe.py \
  --guidance-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/no_aligner_parent_ckpt3668/fn_visibility_guidance_probe_rows.jsonl \
  --gt-vs-pred-scored-path /data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_no_aligner_parent_val128_freegreedy_ckpt3668_val128_bsz8_temp0_rp1p10_max3084_chatfix_8gpu/gt_vs_pred_scored.jsonl \
  --checkpoint-path /data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668 \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/no_aligner_parent_ckpt3668_vase_prefix_coordslot_patch_l24_27 \
  --image-id 139 \
  --gt-idx 7 \
  --guidance-tiers desc_x1,desc_x1_y1,desc_x1_y1_x2 \
  --prefix-object-limits 0,all \
  --layers all \
  --patch-layers 24,25,26,27 \
  --patch-sites decoder_layer \
  --patch-source 0,desc_x1 \
  --patch-target all,desc_x1 \
  --top-k 8 \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --attn-implementation auto
```

Note: an initial attempted run included patch layer `28`, but the model exposes
decoder blocks `0..27`. Hidden-state tuple index `28` is the final/post-stack
state, not a hookable decoder block. The successful patch run therefore uses
decoder layers `24..27`.

## Artifacts

Root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/no_aligner_parent_ckpt3668_vase_prefix_coordslot_patch_l24_27
```

Files:

```text
fn_coordslot_logit_condition_rows.jsonl
fn_coordslot_logit_layer_rows.jsonl
fn_coordslot_hidden_delta_rows.jsonl
fn_coordslot_residual_patch_rows.jsonl
phase4_fn_coordslot_logit_probe_summary.json
phase4_fn_coordslot_logit_probe_report.md
```

## Patch Result

Patch:

- source condition: prefix `0`, tier `desc_x1`;
- target condition: prefix `all`, tier `desc_x1`;
- site: `decoder_layer` output;
- token position: prompt-end partial-row token;
- metric: target `y1=468` coordinate-token distribution on the full-prefix
  target prompt.

Baseline target condition:

- target rank: `52`;
- target probability: `0.005097`;
- top1 bin: `501`;
- top1 distance: `33`.

Patch rows:

| layer | patched rank | rank recovery | patched prob | prob delta | patched top1 | patched top1 distance |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 10 | 42 | 0.018278 | 0.013181 | 472 | 4 |
| 25 | 13 | 39 | 0.018575 | 0.013478 | 479 | 11 |
| 26 | 12 | 40 | 0.018723 | 0.013625 | 473 | 5 |
| 27 | 12 | 40 | 0.018688 | 0.013590 | 472 | 4 |

## Interpretation

This is causal sufficiency evidence for the prefix-state coordinate-basin
mechanism in the vase FN:

- The full-prefix prompt has a bad `y1` basin around `501`.
- Patching only the prompt-end late residual from the empty-prefix `desc_x1`
  source into the full-prefix `desc_x1` target moves top1 back to the local
  target basin around `472`.
- Target `y1=468` improves from rank `52` to rank `10-13`, nearly matching the
  empty-prefix unpatched prompt-end rank `12`.

This makes the current mechanism sharper:

1. The failure is not merely a decoding tail artifact.
2. It is not simple visual blindness, because the target basin is accessible
   under the same image and desc.
3. The full autoregressive prefix installs a late residual state that steers the
   next coordinate into the wrong vertical basin.
4. Replacing that late residual state is enough to restore the correct basin at
   the prompt-end next-token distribution.

## Next Hook

The next useful split is to localize which subcomponent carries this residual
effect:

- `self_attn` / `self_attn_input`;
- `mlp` / `mlp_input`;
- possibly attention-source routing into the prompt-end token.

Layer `24` is the strongest first target by rank recovery (`42`) and top1
distance (`4`), but layers `26-27` are similarly sufficient. A good next smoke
is the same source/target pair over sites:

```text
self_attn,self_attn_input,mlp,mlp_input
```

## Guardrails

- One object and one checkpoint only.
- This patches prompt-end residual state, not visual features directly.
- Patch source and target prompts have different prefix lengths; token indices
  are prompt-local and explicitly recorded in the artifact.
- The result proves this late residual state is sufficient to restore the
  prompt-end coordinate basin for this case, not that it is the only causal
  pathway or that the same layer/site explains all FNs.
