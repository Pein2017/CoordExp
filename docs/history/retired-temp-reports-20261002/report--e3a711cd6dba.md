# Unmatched Proposal Verifier Study

## Setup
- subset: `/data/CoordExp/output/analysis/unmatched-proposal-verifier-stage2parity-auth8x1-stage1-coco80-ckpt-1832-merged-t0p7/subset/sampled.coord.jsonl` from `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`
- sample_count: `200`
- seed: `42`
- root_image_dir: `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60`
- run stages: `prepare, collect, gate, score, audit, report`

## Collection
- collection backend mode: `stage2_parity_vllm`
- temperature: `0.7`
- repetition_penalty: `1.1`
- infer.generation.batch_size: `16`
- authoritative temperatures: `0.0, 0.3, 0.5, 0.7`
- infer.backend.server_options.vllm_gpu_memory_utilization: `0.8`

## Checkpoints
- `stage1-coco80-ckpt-1832-merged` -> `/data/CoordExp/output/stage1/coco_bbox_max60-coco80-desc_first/epoch_4-softce_w1-coco80-ckpt_1832-merged` (prompt_variant=`coco_80`, object_field_order=`desc_first`, source=`study_default`)

## Proxy Definitions
- commitment: desc-only average teacher-forced log-probability on the original image
- counterfactual: commitment(original) - commitment(masked bbox image)
- combined_linear: commitment + counterfactual

## Aggregate Tables
- clean GT summary: `/data/CoordExp/output/analysis/unmatched-proposal-verifier-stage2parity-auth8x1-stage1-coco80-ckpt-1832-merged-t0p7/gt_clean_proxy_metrics_by_temp.csv`
- collection health: `/data/CoordExp/output/analysis/unmatched-proposal-verifier-stage2parity-auth8x1-stage1-coco80-ckpt-1832-merged-t0p7/collection_health_by_temp.csv`
- rollout summary: `/data/CoordExp/output/analysis/unmatched-proposal-verifier-stage2parity-auth8x1-stage1-coco80-ckpt-1832-merged-t0p7/rollout_proxy_metrics_by_temp.csv`
- manual audit summary: `/data/CoordExp/output/analysis/unmatched-proposal-verifier-stage2parity-auth8x1-stage1-coco80-ckpt-1832-merged-t0p7/manual_audit_summary.json`

## Layer A: Clean Verifier Benchmark
### stage1-coco80-ckpt-1832-merged @ temperature=0.7
- `commitment` GT-vs-hard-neg: AUROC=`0.5015` AUPRC=`0.3004` counted=`5402`
- `counterfactual` GT-vs-hard-neg: AUROC=`0.6183` AUPRC=`0.4644` counted=`5402`
- `combined_linear` GT-vs-hard-neg: AUROC=`0.5460` AUPRC=`0.3935` counted=`5402`

## Layer B1: Rollout Collection Health
- `stage1-coco80-ckpt-1832-merged` temp=`0.7` valid=`yes` pred_total=`590` unmatched=`291` nonempty_rate=`0.6400` invalid_reason=`NA`

## Layer B2: Rollout Proposal Benchmark
### stage1-coco80-ckpt-1832-merged @ temperature=0.7
- `commitment` matched-vs-unmatched: AUROC=`0.5966` AUPRC=`0.5946` counted=`573`
- `commitment` unmatched top-k stats: `{"10": {"count": 10, "nearest_gt_iou_ge_0.3_rate": 0.6, "nearest_gt_iou_ge_0.5_rate": 0.1}, "25": {"count": 25, "nearest_gt_iou_ge_0.3_rate": 0.52, "nearest_gt_iou_ge_0.5_rate": 0.04}, "50": {"count": 50, "nearest_gt_iou_ge_0.3_rate": 0.34, "nearest_gt_iou_ge_0.5_rate": 0.02}}`
- `counterfactual` matched-vs-unmatched: AUROC=`0.6369` AUPRC=`0.6250` counted=`573`
- `counterfactual` unmatched top-k stats: `{"10": {"count": 10, "nearest_gt_iou_ge_0.3_rate": 0.4, "nearest_gt_iou_ge_0.5_rate": 0.2}, "25": {"count": 25, "nearest_gt_iou_ge_0.3_rate": 0.4, "nearest_gt_iou_ge_0.5_rate": 0.12}, "50": {"count": 50, "nearest_gt_iou_ge_0.3_rate": 0.4, "nearest_gt_iou_ge_0.5_rate": 0.14}}`
- `combined_linear` matched-vs-unmatched: AUROC=`0.6530` AUPRC=`0.6371` counted=`573`
- `combined_linear` unmatched top-k stats: `{"10": {"count": 10, "nearest_gt_iou_ge_0.3_rate": 0.4, "nearest_gt_iou_ge_0.5_rate": 0.2}, "25": {"count": 25, "nearest_gt_iou_ge_0.3_rate": 0.48, "nearest_gt_iou_ge_0.5_rate": 0.12}, "50": {"count": 50, "nearest_gt_iou_ge_0.3_rate": 0.5, "nearest_gt_iou_ge_0.5_rate": 0.12}}`
- commitment/counterfactual correlation: `0.0337`
- calibration: skipped (`v1 study reports raw log-probability proxies only`)
- audit pack: count=`24` index=`/data/CoordExp/output/analysis/unmatched-proposal-verifier-stage2parity-auth8x1-stage1-coco80-ckpt-1832-merged-t0p7/checkpoints/stage1-coco80-ckpt-1832-merged/audit_pack/index.jsonl`

## Layer C: Manual Audit
- labels loaded: `no`
- labeled count: `0`
- label counts: `{}`
- precision@k: `{"10": {"count": 0, "real_visible_object_rate": null}, "25": {"count": 0, "real_visible_object_rate": null}, "50": {"count": 0, "real_visible_object_rate": null}}`

## Recommendation
- strongest single proxy: `counterfactual`
- does commitment + counterfactual materially outperform either single proxy? `no`
- is the signal stable across checkpoints? `mixed`
- is rollout evidence valid enough for interpretation? `yes`
- is the proxy good enough for soft pseudo-label promotion? `promising but not yet promotion-ready`
- main observed failure modes:
  scoring drift / exclusions: `{"sequence_canonicalization_failed": 36}`
  collection gate exclusions: `{}`
- commitment can remain high on visually plausible wrong-location boxes; counterfactual is the intended corrective signal.
- manual audit labels are missing, so the final recommendation is intentionally downgraded.
- best overall proxy on the current summary tables: `counterfactual`.
