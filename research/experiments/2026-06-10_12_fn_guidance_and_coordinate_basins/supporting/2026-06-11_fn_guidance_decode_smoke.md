# FN Guidance Decode Smoke

Date: 2026-06-11

## Scope

First actual continuation-decode smoke for the false-negative guidance branch.

Question:

- For valid FN cases that have nearby visual/proxy evidence, can a lightweight continuation prompt recover the missing object when given `desc_only`, `desc_x1`, or `desc_x1_wrong_control` guidance?

This is a tiny smoke over two selected no-aligner-parent FN cases, not a broad rescue result.

## Inputs

Guidance rows:

- `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_visibility_guidance_parent_val128/no_aligner_parent_ckpt3668/fn_visibility_guidance_probe_rows.jsonl`

Scored rows:

- `/data/CoordExp/outputs/infer/loss_only_instance_enumeration_ablation_active_vs_none/compact_full_prefix_rollin_balance2_no_aligner_parent_val128_freegreedy_ckpt3668_val128_bsz8_temp0_rp1p10_max3084_chatfix_8gpu/gt_vs_pred_scored.jsonl`

Checkpoint:

- `/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_fullobj_random_sft_bsz16_4epoch_tokenrows_v2/compact-full-fullobj-random-sft-bsz16-4epoch-tokenrows-v2/v1-20260601-062428/checkpoint-3668`

Decode settings:

- `per_bucket_cap=1`
- Buckets:
  - `likely_language_or_prefix_guidance_fragile`
  - `likely_visual_available_but_binding_or_enumeration_failed`
- Tiers:
  - `desc_only`
  - `desc_x1`
  - `desc_x1_wrong_control`
- Greedy decode
- GPU: `CUDA_VISIBLE_DEVICES=0`

## Artifacts

- Root:
  `/data/CoordExp/outputs/analysis/autoregressive_duplication_mechanism/self_driven_fn_guidance_decode_smoke_parent_val128/no_aligner_parent_ckpt3668_per_bucket1`
- Rows:
  `fn_guidance_decode_smoke_rows.jsonl`
- Summary:
  `phase4_fn_guidance_decode_smoke_summary.json`
- Report:
  `phase4_fn_guidance_decode_smoke_report.md`

## Results

Rows: `6` continuation generations, two FN cases times three guidance tiers.

Outcome counts:

- `desc_only|success_iou50=False|valid=True`: 2
- `desc_x1|success_iou50=True|valid=True`: 1
- `desc_x1|success_iou50=False|valid=True`: 1
- `desc_x1_wrong_control|success_iou50=True|valid=True`: 1
- `desc_x1_wrong_control|success_iou50=False|valid=True`: 1

Case details:

1. Image `139`, GT `vase`, bucket `likely_visual_available_but_binding_or_enumeration_failed`
   - Target norm1000 box: `[526, 468, 542, 508]`
   - `desc_only`: generated `[860, 742, 907, 939]`, IoU `0.0000`
   - `desc_x1`: generated `[526, 497, 553, 540]`, IoU `0.1083`
   - `desc_x1_wrong_control`: generated `[525, 501, 560, 543]`, IoU `0.0561`
   - Read: coordinate hint moves the model toward the local vase/proxy basin but does not rescue the target.

2. Image `139`, GT `chair`, bucket `likely_language_or_prefix_guidance_fragile`
   - Target norm1000 box: `[454, 512, 551, 743]`
   - `desc_only`: generated `[611, 519, 685, 717]`, IoU `0.0000`
   - `desc_x1`: generated `[454, 516, 546, 739]`, IoU `0.9156`, primary rescue success
   - `desc_x1_wrong_control`: generated `[460, 512, 533, 729]`, IoU `0.7070`, primary rescue success
   - Read: language plus x1 guidance can rescue this FN, but the wrong-control x1 is only 6 bins from the true x1, so this is not clean evidence that exact x1 is uniquely causal.

## Interpretation

The smoke supports the existence of at least one guidance-fragile FN: the chair miss is not visually impossible, because a continuation with x1 guidance recovers a high-IoU box after the original rollout missed it.

However, the wrong-control success shows that nearby wrong-control hints can also rescue when they are spatially close. Future rescue probes must stratify wrong-control x1 distance or explicitly choose farther wrong-control regions to avoid overclaiming coordinate-hint causality.

The vase case is also informative: x1 guidance pulls the continuation from a far-away previous vase to the local vase/proxy region, but the generated box remains below rescue IoU. This looks like local evidence exists but is not synchronized tightly enough by the minimal x1 hint.

## Guardrails

- Tiny smoke only: two cases, six generations.
- The decode runner currently appends the rescue row after the full existing raw prediction prefix. This tests late continuation feasibility, not all possible prefix insertion depths.
- Success is scored in norm1000 coordinate space against `gt_bbox_norm1000_xyxy`.
- Wrong-control x1 distance must be treated as a first-class variable before making claims about `desc_x1` specificity.

## Next Probe

Scale to a small stratified panel with:

- 4 to 8 cases per bucket.
- Wrong-control x1 separation bins, for example `near`, `medium`, `far`.
- Separate reporting for `desc_x1_success && wrong_control_fail`, `both_success`, `both_fail`, and `desc_only_success`.
- Optional prefix-depth variants, because some FNs may be recoverable only before the model enters a local repetition/duplication basin.
