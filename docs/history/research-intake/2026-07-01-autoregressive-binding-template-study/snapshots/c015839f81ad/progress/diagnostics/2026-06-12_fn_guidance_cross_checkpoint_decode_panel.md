# FN Guidance Cross-Checkpoint Decode Panel

Date: 2026-06-12

## Scope

This panel extends the false-negative visibility-vs-guidance branch from the
no-aligner parent checkpoint to the aligner-tuned parent and the tiny
visual-region-loss auxiliary checkpoint.

Question:

- For selected valid FNs with nearby visual/proxy evidence, do deterministic
  continuation prompts recover the missing object through language/coordinate
  guidance, or do they behave like hard visual misses?

This is a small greedy continuation smoke, not a rescue-rate benchmark.

## Checkpoints

No-aligner parent:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668
```

Aligner-tuned parent:

```text
/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_llm_aligner_lora_packed_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-llm-aligner-lora-packed-bsz16-4epoch-tokenrows-v2/v5-20260608-081447/checkpoint-1824
```

Auxiliary-loss checkpoint:

```text
/data/CoordExp/outputs/stage1_2b/loss_only_instance_enumeration_ablation_active_vs_none/random_active_hp02_latest_aligner_dora_4096_128_no_newline_1epoch32_4gpu/loss-only-instance-enum-random-active-hp02-latest-aligner-dora-4096-128-no-newline-1epoch32-4gpu/v0-20260609-115128/checkpoint-32
```

## Artifact Roots

Visibility-guidance manifests:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/no_aligner_parent_ckpt3668
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/aligner_parent_ckpt1824
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/aux_latest_ckpt32_guarded
```

Decode panels:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_guidance_decode_smoke_parent_val128/no_aligner_parent_ckpt3668_prefix_depth_coordslot_cap1
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_guidance_decode_smoke_parent_val128/aligner_parent_ckpt1824_prefix_depth_coordslot_cap1
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_guidance_decode_smoke_parent_val128/aux_latest_ckpt32_guarded_prefix_depth_coordslot_cap1
```

Each decode root contains:

```text
fn_guidance_decode_smoke_rows.jsonl
fn_guidance_decode_smoke_case_rows.jsonl
phase4_fn_guidance_decode_smoke_summary.json
phase4_fn_guidance_decode_smoke_report.md
```

## Aux Manifest Regeneration

The first aux decode attempt failed before generation because the older aux
manifest did not expose `gt_bbox_norm1000_xyxy`, which is required for
`desc_x1_y1` and `desc_x1_y1_x2` prompt-side coordinate hints.

The aux manifest was regenerated from the guarded matching contract:

```text
/data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_loss_only_aux_latest_aligner_dora_val128_freegreedy_checkpoint-32-inference-clean_val128_bsz8_temp0_rp1p10_max3084_chatfix_4gpu/eval/matches@0.30_guarded.jsonl
```

Regenerated aux manifest counts:

| bucket | FN cases |
| --- | ---: |
| `artifact_invalid_or_parse_blocked` | 250 |
| `likely_language_or_prefix_guidance_fragile` | 139 |
| `likely_visual_available_but_binding_or_enumeration_failed` | 52 |
| `likely_visual_blind_or_no_object_proposal` | 32 |
| `likely_visual_low_salience_small_object` | 24 |

This gives `191` valid visual/proxy cases and `56` valid visual-miss or
low-salience cases under the regenerated guarded contract. It should replace
the earlier sampled aux manifest for coord-slot decode probes.

## Decode Settings

For aligner and aux runs:

```bash
python scripts/analysis/run_autoregressive_duplication_phase4_fn_guidance_decode_smoke.py \
  --per-bucket-cap 1 \
  --stratify-wrong-control-distance \
  --guidance-tiers desc_only,desc_x1,desc_x1_y1,desc_x1_y1_x2,desc_x1_wrong_control \
  --prefix-object-limits 0,all \
  --torch-dtype bfloat16 \
  --attn-implementation auto
```

GPU placement:

- aligner parent: `CUDA_VISIBLE_DEVICES=0`
- aux checkpoint: `CUDA_VISIBLE_DEVICES=1`

The no-aligner parent row is the previously completed matching panel.

## Cross-Checkpoint Results

All rows are small deterministic smokes: six selected FN cases, two prefix
depths, five guidance tiers.

| checkpoint | rows | desc-only | desc-x1 | desc-x1-y1 | desc-x1-y1-x2 | wrong-control x1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_aligner_parent_ckpt3668` | 60 | 0/12 | 9/12 | 10/12 | 10/12 | 4/11 |
| `aligner_parent_ckpt1824` | 60 | 3/12 | 10/12 | 10/12 | 12/12 | 5/12 |
| `aux_latest_ckpt32_guarded` | 60 | 1/11 | 6/12 | 9/12 | 10/12 | 4/9 |

Prefix split:

| checkpoint | prefix | desc-only | desc-x1 | desc-x1-y1 | desc-x1-y1-x2 | wrong-control x1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_aligner_parent_ckpt3668` | `0` | 0/6 | 5/6 | 5/6 | 5/6 | 2/6 |
| `no_aligner_parent_ckpt3668` | `all` | 0/6 | 4/6 | 5/6 | 5/6 | 2/5 |
| `aligner_parent_ckpt1824` | `0` | 1/6 | 5/6 | 5/6 | 6/6 | 3/6 |
| `aligner_parent_ckpt1824` | `all` | 2/6 | 5/6 | 5/6 | 6/6 | 2/6 |
| `aux_latest_ckpt32_guarded` | `0` | 1/6 | 2/6 | 4/6 | 5/6 | 2/4 |
| `aux_latest_ckpt32_guarded` | `all` | 0/5 | 4/6 | 5/6 | 5/6 | 2/5 |

Case-outcome summary:

| checkpoint | clean x1-specific | both x1 and wrong-control | desc-only success | no rescue |
| --- | ---: | ---: | ---: | ---: |
| `no_aligner_parent_ckpt3668` | 6 | 3 | 0 | 3 |
| `aligner_parent_ckpt1824` | 5 | 1 | 3 | 3 |
| `aux_latest_ckpt32_guarded` | 2 | 2 | 1 | 7 |

## Mechanism Read

The cross-checkpoint pattern argues against a single false-negative mechanism.

1. The parent checkpoints both retain many guidance-accessible FNs. The
   no-aligner parent has no `desc_only` successes in this selected panel, but
   `desc_x1` and later coordinate slots rescue most rows. The aligner parent
   is even more accessible under full coordinate-slot guidance, with
   `desc_x1_y1_x2` rescuing all 12 rows.
2. The aligner-tuned parent has more `desc_only` successes than the no-aligner
   parent. On this tiny panel, that suggests the tuned aligner may make some
   selected missing objects easier to recover from semantic/image context alone,
   rather than purely through an x-coordinate hint.
3. The aux checkpoint still has strong coordinate-slot accessibility
   (`desc_x1_y1_x2`: 10/12), but `desc_x1` alone is weaker (`6/12`) and the
   no-rescue count is higher (`7`). This is consistent with different mechanics
   after the visual/region auxiliary objective, but it is not evidence of a
   simple visual-perception failure.
4. Wrong-control successes remain non-negligible, especially when the control
   x1 is near the target. Any claim about exact coordinate specificity must
   keep wrong-control distance as a first-class variable.

Operationally, the result sharpens the FN taxonomy:

- **Guidance-fragile / coordinate-accessible FN:** target appears recoverable
  once the continuation is given enough coordinate context.
- **Prefix-state lock-in:** full prefix can reduce `desc_x1` rescue even when
  `prefix=0` or `desc_x1_y1` works, as in the prior vase panel.
- **Hard residual FN:** some cases remain unrescued despite x1/y1/x2 hints and
  should be treated as candidate visibility/extent failures.
- **Semantic-context accessible FN:** aligner parent has selected rows rescued
  even by `desc_only`, which is a distinct subfamily from coordinate-only
  basin unlocks.

## Interpretation Boundary

- This panel uses selected cases, not random FN sampling.
- Each checkpoint contributes only six selected FN cases and 60 continuation
  generations.
- `desc_x1_y1` and `desc_x1_y1_x2` use target coordinate slots by construction;
  they test whether the remaining coordinate basin is accessible, not whether
  the evaluation-time policy can discover those slots unaided.
- The aux checkpoint panel now uses the same guarded matching style as the
  parent manifests, but the model itself has different mechanics and rollout
  health. Compare mechanism patterns, not aggregate rescue rates.

## Next Hook

The highest-leverage hidden-state target is still the small set of
case-level transitions where `desc_x1` fails but `desc_x1_y1` or
`desc_x1_y1_x2` succeeds. These isolate coordinate-slot basin accessibility:
the object is not absent from the model's conditional generative space, but the
autoregressive state cannot reliably enter the right basin without an extra
coordinate token.

For hidden-state probing, prioritize:

- parent vase full-prefix transition: `desc_x1` fails, `desc_x1_y1` rescues;
- aux rows where `desc_x1` fails but `desc_x1_y1_x2` succeeds;
- hard rows where even `desc_x1_y1_x2` fails, as a contrast set for visibility
  or extent limitations.

## Verification

Commands completed:

```bash
python scripts/analysis/run_autoregressive_duplication_phase4_fn_visibility_guidance_probe.py \
  --gt-vs-pred-scored-path /data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_loss_only_aux_latest_aligner_dora_val128_freegreedy_checkpoint-32-inference-clean_val128_bsz8_temp0_rp1p10_max3084_chatfix_4gpu/gt_vs_pred_scored.jsonl \
  --matches-path /data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_loss_only_aux_latest_aligner_dora_val128_freegreedy_checkpoint-32-inference-clean_val128_bsz8_temp0_rp1p10_max3084_chatfix_4gpu/eval/matches@0.30_guarded.jsonl \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/aux_latest_ckpt32_guarded \
  --checkpoint-label aux_latest_ckpt32_guarded \
  --checkpoint-path /data/CoordExp/outputs/stage1_2b/loss_only_instance_enumeration_ablation_active_vs_none/random_active_hp02_latest_aligner_dora_4096_128_no_newline_1epoch32_4gpu/loss-only-instance-enum-random-active-hp02-latest-aligner-dora-4096-128-no-newline-1epoch32-4gpu/v0-20260609-115128/checkpoint-32 \
  --rollout-root /data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_loss_only_aux_latest_aligner_dora_val128_freegreedy_checkpoint-32-inference-clean_val128_bsz8_temp0_rp1p10_max3084_chatfix_4gpu
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_duplication_phase4_fn_guidance_decode_smoke.py \
  --guidance-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/aligner_parent_ckpt1824/fn_visibility_guidance_probe_rows.jsonl \
  --gt-vs-pred-scored-path /data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_original_latest_aligner_parent_val128_freegreedy_ckpt1824_val128_bsz8_temp0_rp1p10_max3084_chatfix_8gpu/gt_vs_pred_scored.jsonl \
  --checkpoint-path /data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_llm_aligner_lora_packed_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-llm-aligner-lora-packed-bsz16-4epoch-tokenrows-v2/v5-20260608-081447/checkpoint-1824 \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_guidance_decode_smoke_parent_val128/aligner_parent_ckpt1824_prefix_depth_coordslot_cap1 \
  --per-bucket-cap 1 \
  --stratify-wrong-control-distance \
  --guidance-tiers desc_only,desc_x1,desc_x1_y1,desc_x1_y1_x2,desc_x1_wrong_control \
  --prefix-object-limits 0,all \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --attn-implementation auto
```

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/analysis/run_autoregressive_duplication_phase4_fn_guidance_decode_smoke.py \
  --guidance-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/aux_latest_ckpt32_guarded/fn_visibility_guidance_probe_rows.jsonl \
  --gt-vs-pred-scored-path /data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_loss_only_aux_latest_aligner_dora_val128_freegreedy_checkpoint-32-inference-clean_val128_bsz8_temp0_rp1p10_max3084_chatfix_4gpu/gt_vs_pred_scored.jsonl \
  --checkpoint-path /data/CoordExp/outputs/stage1_2b/loss_only_instance_enumeration_ablation_active_vs_none/random_active_hp02_latest_aligner_dora_4096_128_no_newline_1epoch32_4gpu/loss-only-instance-enum-random-active-hp02-latest-aligner-dora-4096-128-no-newline-1epoch32-4gpu/v0-20260609-115128/checkpoint-32 \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_guidance_decode_smoke_parent_val128/aux_latest_ckpt32_guarded_prefix_depth_coordslot_cap1 \
  --per-bucket-cap 1 \
  --stratify-wrong-control-distance \
  --guidance-tiers desc_only,desc_x1,desc_x1_y1,desc_x1_y1_x2,desc_x1_wrong_control \
  --prefix-object-limits 0,all \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --attn-implementation auto
```

Artifact row counts were checked from each summary: all three decode roots have
`row_count=60` and `case_row_count=12`.
