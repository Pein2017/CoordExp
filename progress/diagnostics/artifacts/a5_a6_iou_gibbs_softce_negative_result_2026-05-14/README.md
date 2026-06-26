# A5/A6 IoU-Gibbs SoftCE Negative-Result Artifact Archive

Status: Superseded / negative-result evidence

This directory preserves lightweight evidence from the deprecated `a5-iou-gibbs-softce` worktree.

## What Is Included

- `configs/prod/`: production YAML snapshots for A5 and A6.
- `configs/smoke/`: tiny and DDP4 preflight YAML snapshots.
- `analysis/`: data-driven IoU/CIoU-Gibbs tau/statistics notes.
- `training/`: training manifests and runtime metadata for final production runs.
- `eval/`: val200 `rp=1.10` metric summaries and per-image diagnostics for final `checkpoint-3664`.
- `visual/`: poor-case visualization manifests for the top 12 lowest-F1 samples.

## What Is Not Included

Large artifacts remain in their canonical output roots and are referenced from the copied manifests:

- checkpoints;
- raw/scored prediction JSONLs;
- prediction token traces;
- trainer states;
- rendered PNG visualization grids.

## Canonical Roots

A5 training:

`/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_iou_gibbs_softce_a5_support2_bsz16_4gpu_4epoch_tokenrows_v2/compact-full-et-rmp-iou-gibbs-softce-a5-support2-bsz16-4gpu-4epoch-tokenrows-v2/v0-20260512-163533`

A6 training:

`/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ciou_gibbs_softce_a6_support2_bsz16_4gpu_4epoch_tokenrows_v2/compact-full-et-rmp-ciou-gibbs-softce-a6-support2-bsz16-4gpu-4epoch-tokenrows-v2/v0-20260512-163531`

A5 eval:

`/data/CoordExp/outputs/infer/recursive_detection_ce_latest/a5_iou_gibbs_ckpt3664_val200_bsz8_temp0_rp1p10_max3084_chatfix_4gpu`

A6 eval:

`/data/CoordExp/outputs/infer/recursive_detection_ce_latest/a6_ciou_gibbs_ckpt3664_val200_bsz8_temp0_rp1p10_max3084_chatfix_4gpu`

## Interpretation

The IoU/CIoU-Gibbs coordinate-token target distributions are deprecated as active training candidates. The evidence supports replacing them with a coordinate-local, bbox-aware Gaussian target whose support loss is over the full coordinate vocabulary and whose balance target remains centered on the GT coordinate.
