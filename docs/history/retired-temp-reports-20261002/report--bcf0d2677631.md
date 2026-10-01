# Unmatched Proposal Verifier Study

## Setup
- subset: `/data/CoordExp/output/analysis/unmatched-proposal-verifier-stage2parity-auth8x1-t0p7-ul-res-1024-v2-ckpt-300-merged/subset/sampled.coord.jsonl` from `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`
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
- `ul-res_1024-v2-ckpt_300_merged` -> `/data/CoordExp/output/stage2_ab/prod/ul-res_1024-v2-ckpt_300_merged` (prompt_variant=`coco_80`, object_field_order=`desc_first`, source=`study_default`)

## Proxy Definitions
- commitment: desc-only average teacher-forced log-probability on the original image
- counterfactual: commitment(original) - commitment(masked bbox image)
- combined_linear: commitment + counterfactual

## Aggregate Tables
- clean GT summary: `/data/CoordExp/output/analysis/unmatched-proposal-verifier-stage2parity-auth8x1-t0p7-ul-res-1024-v2-ckpt-300-merged/gt_clean_proxy_metrics_by_temp.csv`
- collection health: `/data/CoordExp/output/analysis/unmatched-proposal-verifier-stage2parity-auth8x1-t0p7-ul-res-1024-v2-ckpt-300-merged/collection_health_by_temp.csv`
- rollout summary: `/data/CoordExp/output/analysis/unmatched-proposal-verifier-stage2parity-auth8x1-t0p7-ul-res-1024-v2-ckpt-300-merged/rollout_proxy_metrics_by_temp.csv`
- manual audit summary: `/data/CoordExp/output/analysis/unmatched-proposal-verifier-stage2parity-auth8x1-t0p7-ul-res-1024-v2-ckpt-300-merged/manual_audit_summary.json`

## Layer A: Clean Verifier Benchmark
### ul-res_1024-v2-ckpt_300_merged @ temperature=0.7
- `commitment` GT-vs-hard-neg: AUROC=`0.5019` AUPRC=`0.3015` counted=`5402`
- `counterfactual` GT-vs-hard-neg: AUROC=`0.6244` AUPRC=`0.4695` counted=`5402`
- `combined_linear` GT-vs-hard-neg: AUROC=`0.5504` AUPRC=`0.4023` counted=`5402`

## Layer B1: Rollout Collection Health
- `ul-res_1024-v2-ckpt_300_merged` temp=`0.7` valid=`yes` pred_total=`489` unmatched=`283` nonempty_rate=`0.4800` invalid_reason=`NA`

## Layer B2: Rollout Proposal Benchmark
### ul-res_1024-v2-ckpt_300_merged @ temperature=0.7
- `commitment` matched-vs-unmatched: AUROC=`0.6541` AUPRC=`0.6292` counted=`420`
- `commitment` unmatched top-k stats: `{"10": {"count": 10, "nearest_gt_iou_ge_0.3_rate": 0.4, "nearest_gt_iou_ge_0.5_rate": 0.1}, "25": {"count": 25, "nearest_gt_iou_ge_0.3_rate": 0.36, "nearest_gt_iou_ge_0.5_rate": 0.04}, "50": {"count": 50, "nearest_gt_iou_ge_0.3_rate": 0.42, "nearest_gt_iou_ge_0.5_rate": 0.06}}`
- `counterfactual` matched-vs-unmatched: AUROC=`0.6271` AUPRC=`0.5842` counted=`420`
- `counterfactual` unmatched top-k stats: `{"10": {"count": 10, "nearest_gt_iou_ge_0.3_rate": 0.4, "nearest_gt_iou_ge_0.5_rate": 0.1}, "25": {"count": 25, "nearest_gt_iou_ge_0.3_rate": 0.48, "nearest_gt_iou_ge_0.5_rate": 0.08}, "50": {"count": 50, "nearest_gt_iou_ge_0.3_rate": 0.36, "nearest_gt_iou_ge_0.5_rate": 0.06}}`
- `combined_linear` matched-vs-unmatched: AUROC=`0.6881` AUPRC=`0.6405` counted=`420`
- `combined_linear` unmatched top-k stats: `{"10": {"count": 10, "nearest_gt_iou_ge_0.3_rate": 0.5, "nearest_gt_iou_ge_0.5_rate": 0.1}, "25": {"count": 25, "nearest_gt_iou_ge_0.3_rate": 0.44, "nearest_gt_iou_ge_0.5_rate": 0.04}, "50": {"count": 50, "nearest_gt_iou_ge_0.3_rate": 0.52, "nearest_gt_iou_ge_0.5_rate": 0.12}}`
- commitment/counterfactual correlation: `0.1278`
- calibration: skipped (`v1 study reports raw log-probability proxies only`)
- audit pack: count=`24` index=`/data/CoordExp/output/analysis/unmatched-proposal-verifier-stage2parity-auth8x1-t0p7-ul-res-1024-v2-ckpt-300-merged/checkpoints/ul-res-1024-v2-ckpt-300-merged/audit_pack/index.jsonl`

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
  scoring drift / exclusions: `{"sequence_canonicalization_failed": 195}`
  collection gate exclusions: `{}`
- commitment can remain high on visually plausible wrong-location boxes; counterfactual is the intended corrective signal.
- manual audit labels are missing, so the final recommendation is intentionally downgraded.
- best overall proxy on the current summary tables: `counterfactual`.
