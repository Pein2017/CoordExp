# FN Guidance Wrong-X1 Stratified Panel

Date: 2026-06-11

## Scope

Small follow-up to the first FN guidance decode smoke. The earlier smoke showed
that a nearby wrong-control `x1` can sometimes rescue the missing object, so this
panel makes wrong-control distance explicit instead of treating
`desc_x1_wrong_control` as a single undifferentiated control tier.

This remains a tiny deterministic greedy continuation panel, not a broad rescue
metric.

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

Decode command:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis/run_autoregressive_duplication_phase4_fn_guidance_decode_smoke.py \
  --guidance-rows-path /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/no_aligner_parent_ckpt3668/fn_visibility_guidance_probe_rows.jsonl \
  --gt-vs-pred-scored-path /data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_no_aligner_parent_val128_freegreedy_ckpt3668_val128_bsz8_temp0_rp1p10_max3084_chatfix_8gpu/gt_vs_pred_scored.jsonl \
  --checkpoint-path /data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668 \
  --output-dir /data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_guidance_decode_smoke_parent_val128/no_aligner_parent_ckpt3668_wrong_x1_stratified_cap1 \
  --per-bucket-cap 1 \
  --stratify-wrong-control-distance \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --attn-implementation auto
```

## Artifacts

Root:

```text
/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_guidance_decode_smoke_parent_val128/no_aligner_parent_ckpt3668_wrong_x1_stratified_cap1
```

Files:

```text
fn_guidance_decode_smoke_rows.jsonl
fn_guidance_decode_smoke_case_rows.jsonl
phase4_fn_guidance_decode_smoke_summary.json
phase4_fn_guidance_decode_smoke_report.md
```

## Results

Scope:

- Selected cases: `6`
- Decode rows: `18`
- Case rows: `6`
- Wrong-control distance bins:
  - `near_0_10`
  - `medium_11_64`
  - `far_65_plus`

Row-level outcome counts:

- `desc_only|success_iou50=False|valid=True`: 6
- `desc_x1|success_iou50=True|valid=True`: 4
- `desc_x1|success_iou50=False|valid=True`: 2
- `desc_x1_wrong_control|success_iou50=True|valid=True`: 2
- `desc_x1_wrong_control|success_iou50=False|valid=True`: 3
- `desc_x1_wrong_control|success_iou50=False|valid=False`: 1

Case outcome counts:

- `clean_desc_x1_specific`: 2
- `both_x1_and_wrong_control_success`: 2
- `no_rescue`: 2
- `desc_only_success`: 0
- `wrong_control_only_success`: 0

Case outcomes by wrong-control distance:

| bin | clean desc-x1 specific | both x1 and wrong-control success | no rescue |
| --- | ---: | ---: | ---: |
| `near_0_10` | 0 | 1 | 1 |
| `medium_11_64` | 1 | 1 | 0 |
| `far_65_plus` | 1 | 0 | 1 |

## Case Read

| image | gt | bucket | wrong x1 bin | wrong x1 dist | desc | outcome | desc-only IoU | desc-x1 IoU | wrong-control IoU |
| ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| 139 | 7 | likely_visual_available_but_binding_or_enumeration_failed | `near_0_10` | 1 | vase | `no_rescue` | 0.0000 | 0.1083 | 0.0561 |
| 139 | 8 | likely_visual_available_but_binding_or_enumeration_failed | `medium_11_64` | 38 | microwave | `clean_desc_x1_specific` | 0.0000 | 0.6266 | 0.0000 |
| 139 | 11 | likely_language_or_prefix_guidance_fragile | `near_0_10` | 6 | chair | `both_x1_and_wrong_control_success` | 0.0000 | 0.9156 | 0.7070 |
| 139 | 15 | likely_language_or_prefix_guidance_fragile | `medium_11_64` | 28 | chair | `both_x1_and_wrong_control_success` | 0.4722 | 0.8764 | 0.5073 |
| 139 | 17 | likely_visual_available_but_binding_or_enumeration_failed | `far_65_plus` | 70 | book | `no_rescue` | 0.0000 | 0.1030 | 0.0000 |
| 724 | 3 | likely_language_or_prefix_guidance_fragile | `far_65_plus` | 287 | truck | `clean_desc_x1_specific` | 0.4855 | 0.7859 | 0.0000 |

## Interpretation

The strongest read is not "the model can always see the FN object." It is more
specific: some FNs are fragile to language/coordinate continuation context even
after a full rollout omitted them.

Evidence:

- `desc_only` rescues `0/6` cases, so category text alone does not overcome the
  current prefix state in this panel.
- `desc_x1` rescues `4/6` cases, including clean medium/far wrong-control
  failures for the microwave and truck cases.
- The two wrong-control successes occur in near/medium bins, showing that
  coordinate-basin attraction is broad enough that an inexact or locally
  neighboring `x1` can still land inside the target basin.
- The vase and book no-rescue cases remain important. Their target-side `x1`
  hints move generation toward the local region but do not synchronize the full
  box well enough to cross IoU50.

Mechanistically, this supports keeping false negatives split into at least three
families:

1. Clean guidance-fragile misses where a target-side coordinate hint rescues and
   a distant wrong-control does not.
2. Basin-broad misses where target and nearby wrong-control hints both rescue,
   implying a local coordinate attractor rather than exact slot binding.
3. Unrescued misses where visual/proxy evidence may be present but a minimal
   `x1` hint is insufficient to bind the full object.

The next useful step is not a bigger headline rate yet. It is a controlled probe
that varies prefix depth and coordinate hint slots (`x1`, `y1`, `x2`, `y2`) on
these same cases, then links successes/failures to hidden-state and attention
precursors at the coordinate slot.

## Guardrails

- This is a six-case deterministic panel on the no-aligner random-SFT parent
  checkpoint.
- The continuation appends after the existing raw prediction prefix, so it tests
  late rescue from an already formed autoregressive state.
- Distance bins are based only on absolute `x1` separation between target and
  wrong-control hints.
- Treat `both_x1_and_wrong_control_success` as evidence for basin breadth or
  local attractor behavior, not evidence that the wrong-control hint is
  semantically correct.
