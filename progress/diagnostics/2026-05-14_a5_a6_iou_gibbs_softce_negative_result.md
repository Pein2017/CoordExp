# A5/A6 IoU-Gibbs Coordinate SoftCE Negative Result

Status: Superseded

Date: 2026-05-14

## Scope

This note records the final evidence for the deprecated A5/A6 geometry-aware coordinate SoftCE attempt from branch/worktree `a5-iou-gibbs-softce`.

Evaluation scope:

- Dataset view: first 200 records from `public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`
- Template: `compact_full`
- Coordinate surface: coord tokens, `xyxy`, bins `0..999`
- Inference: HF, 4 GPUs per run, `batch_size=8`, `temperature=0`, `rp=1.10`, `max_new_tokens=3084`
- Checkpoints: final `checkpoint-3664`
- Metric scope: val200 diagnostic only, not full validation

## Final Training Roots

A5 IoU-Gibbs:

`/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_iou_gibbs_softce_a5_support2_bsz16_4gpu_4epoch_tokenrows_v2/compact-full-et-rmp-iou-gibbs-softce-a5-support2-bsz16-4gpu-4epoch-tokenrows-v2/v0-20260512-163533`

A6 CIoU-Gibbs:

`/data/CoordExp/outputs/stage1_2b/recursive_detection_ce_latest/compact_full_et_rmp_ciou_gibbs_softce_a6_support2_bsz16_4gpu_4epoch_tokenrows_v2/compact-full-et-rmp-ciou-gibbs-softce-a6-support2-bsz16-4gpu-4epoch-tokenrows-v2/v0-20260512-163531`

## Final Eval Roots

A5 IoU-Gibbs:

`/data/CoordExp/outputs/infer/recursive_detection_ce_latest/a5_iou_gibbs_ckpt3664_val200_bsz8_temp0_rp1p10_max3084_chatfix_4gpu`

A6 CIoU-Gibbs:

`/data/CoordExp/outputs/infer/recursive_detection_ce_latest/a6_ciou_gibbs_ckpt3664_val200_bsz8_temp0_rp1p10_max3084_chatfix_4gpu`

## Archived Evidence

Lightweight evidence was copied into:

`progress/diagnostics/artifacts/a5_a6_iou_gibbs_softce_negative_result_2026-05-14/`

The archive contains:

- production and smoke YAML snapshots for A5/A6;
- tau/statistics notes for `iou_gibbs_v0` and `ciou_gibbs_v0`;
- training `experiment_manifest.json`, `effective_runtime.json`, and `run_metadata.json`;
- eval `summary.json`, `resolved_config.json`, `metrics.json`, `metrics_guarded.json`, `per_image*.json`, `duplicate_guard_report.json`, and confidence summaries;
- poor-case visualization manifests for the top 12 lowest-F1 samples per run.

Large artifacts were intentionally not copied into git:

- checkpoints;
- `trainer_state.json`;
- raw/scored `gt_vs_pred*.jsonl`;
- prediction traces;
- rendered PNG grids.

Those remain in the canonical artifact roots above and are referenced from the copied manifests.

## Metrics

| Run | Rows | Eval Pred | Raw AP | Raw AP50 | Raw AP75 | Raw F1@0.50 | Guard AP | Guard AP50 | Guard AP75 | Guard F1@0.50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A5 `iou_gibbs_v0` | 200 | 1268 | 0.4018 | 0.5513 | 0.4246 | 0.5047 | 0.3950 | 0.5399 | 0.4184 | 0.5469 |
| A6 `ciou_gibbs_v0` | 200 | 1292 | 0.4053 | 0.5653 | 0.4169 | 0.5089 | 0.3965 | 0.5523 | 0.4072 | 0.5515 |

These results are diagnostic-val200 only. They should not be treated as full-val or benchmark conclusions.

## Failure Signal

The poor-case visualizations show a shared failure mode on crowded scenes. The top failure for both A5 and A6 is:

- `record_idx=27`
- image: `images/val2017/000000002299.jpg`
- A5: `GT=22`, `Pred=12`, `TP=0`, `FP=12`, `FN=22`, `F1@0.50=0.0000`
- A6: `GT=22`, `Pred=13`, `TP=0`, `FP=13`, `FN=22`, `F1@0.50=0.0000`

Visualization manifests:

- `progress/diagnostics/artifacts/a5_a6_iou_gibbs_softce_negative_result_2026-05-14/visual/a5_iou_gibbs/manifest.md`
- `progress/diagnostics/artifacts/a5_a6_iou_gibbs_softce_negative_result_2026-05-14/visual/a6_ciou_gibbs/manifest.md`

Interpretation:

- Direct IoU/CIoU-Gibbs coordinate-token target shaping made the target landscape depend on whole-box quality under one-coordinate replacement.
- For large objects and multi-positive states, this created broad/diffuse coordinate support rather than a strictly local target around the GT coordinate.
- The observed behavior is consistent with oversized or merged boxes in crowded scenes, where broad side movements can cover multiple objects instead of separating instances.

## Deprecation Decision

`iou_gibbs_v0` and `ciou_gibbs_v0` should be treated as negative-result target distributions.

Do not use these as active production settings for new coordinate SoftCE runs. Future coordinate-token SoftCE work should preserve Gaussian locality around the GT coordinate and make only the Gaussian variance geometry-aware, for example:

```text
q_x(k) proportional to exp(-(k - x_gt)^2 / (2 * width))
q_y(k) proportional to exp(-(k - y_gt)^2 / (2 * height))
```

The loss should keep coordinate support and target shape separate:

- support loss over all 1000 coordinate tokens;
- Gaussian balance target normalized over only geometrically legal bins;
- no IoU/CIoU score landscape as the target distribution.

## Cleanup Note

The physical worktree `.worktrees/a5-iou-gibbs-softce` can be removed after this archive is committed or otherwise preserved. The branch should remain available as historical provenance unless explicitly retired after a separate branch-archive decision.
