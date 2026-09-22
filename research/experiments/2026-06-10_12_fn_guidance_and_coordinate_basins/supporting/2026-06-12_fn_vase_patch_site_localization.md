# FN Vase Patch-Site Localization

Date: 2026-06-12

## Scope

Subcomponent localization for the image `139`, GT `7`, desc `vase` false
negative. The previous residual patch showed that replacing the full-prefix
`desc_x1` prompt-end late residual with the empty-prefix `desc_x1` residual is
sufficient to move the next-`y1` distribution out of the wrong `501` basin and
back toward the target-local `472/473` basin.

This smoke keeps the same source/target pair and layers, but splits the patch
site across attention and MLP subcomponents:

- source: prefix `0`, tier `desc_x1`;
- target: prefix `all`, tier `desc_x1`;
- target coordinate: `y1=468`;
- layers: `24,25,26,27`;
- sites: `self_attn`, `self_attn_input`, `mlp`, `mlp_input`.

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
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/no_aligner_parent_ckpt3668_vase_prefix_coordslot_patch_sites_l24_27 \
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

## Artifacts

Root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_coordslot_logit_probe_parent_val128/no_aligner_parent_ckpt3668_vase_prefix_coordslot_patch_sites_l24_27
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

## Results

Baseline full-prefix `desc_x1` target condition:

- target `y1`: `468`;
- target rank: `52`;
- target probability: `0.005097`;
- top1 bin: `501`;
- top1 distance: `33`.

Best rows by site:

| site | avg rank recovery | best layer | best patched rank | best rank recovery | best patched top1 | best top1 distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlp` | 26.75 | 27 | 12 | 40 | 479 | 11 |
| `mlp_input` | 26.75 | 27 | 12 | 40 | 479 | 11 |
| `self_attn` | 5.75 | 27 | 38 | 14 | 486 | 18 |
| `self_attn_input` | 5.75 | 27 | 38 | 14 | 483 | 15 |

Layer/site rows:

| site | layer | patched rank | rank recovery | patched prob | prob delta | patched top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlp` | 24 | 37 | 15 | 0.010130 | 0.005033 | 491 |
| `mlp` | 25 | 33 | 19 | 0.011212 | 0.006115 | 491 |
| `mlp` | 26 | 19 | 33 | 0.016153 | 0.011056 | 483 |
| `mlp` | 27 | 12 | 40 | 0.014715 | 0.009618 | 479 |
| `mlp_input` | 24 | 37 | 15 | 0.010130 | 0.005033 | 491 |
| `mlp_input` | 25 | 33 | 19 | 0.011212 | 0.006115 | 491 |
| `mlp_input` | 26 | 19 | 33 | 0.016153 | 0.011056 | 483 |
| `mlp_input` | 27 | 12 | 40 | 0.014715 | 0.009618 | 479 |
| `self_attn` | 24 | 43 | 9 | 0.008398 | 0.003301 | 491 |
| `self_attn` | 25 | 52 | 0 | 0.005241 | 0.000144 | 497 |
| `self_attn` | 26 | 52 | 0 | 0.005318 | 0.000221 | 501 |
| `self_attn` | 27 | 38 | 14 | 0.010422 | 0.005325 | 486 |
| `self_attn_input` | 24 | 42 | 10 | 0.009368 | 0.004271 | 497 |
| `self_attn_input` | 25 | 52 | 0 | 0.005353 | 0.000256 | 497 |
| `self_attn_input` | 26 | 53 | -1 | 0.005232 | 0.000135 | 501 |
| `self_attn_input` | 27 | 38 | 14 | 0.010341 | 0.005244 | 483 |

For comparison, whole `decoder_layer` output patches from the prior artifact:

| layer | whole-layer patched rank | whole-layer rank recovery | whole-layer top1 |
| ---: | ---: | ---: | ---: |
| 24 | 10 | 42 | 472 |
| 25 | 13 | 39 | 479 |
| 26 | 12 | 40 | 473 |
| 27 | 12 | 40 | 472 |

## Interpretation

The late residual effect is mostly carried by the MLP-side state, not by the
attention output alone.

Evidence:

- `mlp` and `mlp_input` both recover rank strongly, especially layers `26-27`.
- Layer `27` MLP patch nearly matches whole-layer rank recovery (`40`) but does
  not fully match the whole-layer top1 bin: MLP gives top1 `479`, while
  whole-layer layer `27` gives top1 `472`.
- Attention-only patches are weak. Layers `25-26` are essentially flat, and
  layer `27` only improves rank from `52` to `38`.
- `mlp` and `mlp_input` are identical in this smoke, suggesting that for this
  hook placement the relevant state is already present at MLP input or is
  passed through the MLP patch site equivalently.

Mechanistically, this sharpens the FN vase story:

1. The full prefix installs a wrong vertical coordinate basin at the prompt-end
   next-`y1` site.
2. Replacing the late whole-layer residual is sufficient to restore the target
   basin.
3. The largest recoverable subcomponent is the MLP-side state, especially layer
   `27`, with layer `26` also strong.
4. Attention may contribute a smaller steering component at layer `27`, but it
   is not sufficient on its own to restore the correct basin.

## Next Hook

The next useful split is value/source attribution into the MLP-side state rather
than another broad hidden-state pass.

Two concrete options:

- Patch only MLP output vs MLP input with a more exact hook if the architecture
  exposes separable pre/post-MLP tensors for this wrapped Qwen3-VL module.
- Use route/content or projected-direction decomposition around layer `27` to
  identify whether the MLP-side vector is amplifying a coordinate-basin
  direction, suppressing the wrong `501` basin, or both.

## Guardrails

- One image/object/checkpoint only.
- Patch source and target prompts have different prefix lengths; token indices
  are prompt-local and recorded in artifact rows.
- Sites are local hook names exposed by the existing residual-patch utility.
  The identical `mlp` and `mlp_input` rows should be treated as a hook-placement
  finding, not as proof that mathematical pre/post-MLP states are identical.
- This localizes a sufficient pathway for the vase `y1` basin, not a universal
  FN mechanism.
