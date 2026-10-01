# Unmatched Proposal Verifier Study

## Setup
- subset: `/data/CoordExp/output/analysis/coord-family-recall-pilot-center-val64/subset/sampled.coord.jsonl` from `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/val.coord.jsonl`
- sample_count: `64`
- seed: `42`
- root_image_dir: `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy`
- run stages: `prepare, collect, gate, score, audit, report`

## Collection
- collection backend mode: `hf`
- temperature: `0.3`
- repetition_penalty: `1.05`
- infer.generation.batch_size: `4`
- authoritative temperatures: `0.3`

## Checkpoints
- `center_parameterization` -> `/data/CoordExp/output/stage1_2b/coco_bbox_max60-1024-lvis_proxy-center_parameterization-ckpt_1564-merged` (prompt_variant=`coco_80`, object_field_order=`desc_first`, source=`study_default`)

## Proxy Definitions
- commitment: desc-only average teacher-forced log-probability on the original image
- counterfactual: commitment(original) - commitment(masked bbox image)
- combined_linear: commitment + counterfactual

## Aggregate Tables
- clean GT summary: `/data/CoordExp/output/analysis/coord-family-recall-pilot-center-val64/gt_clean_proxy_metrics_by_temp.csv`
- collection health: `/data/CoordExp/output/analysis/coord-family-recall-pilot-center-val64/collection_health_by_temp.csv`
- rollout summary: `/data/CoordExp/output/analysis/coord-family-recall-pilot-center-val64/rollout_proxy_metrics_by_temp.csv`
- manual audit summary: `/data/CoordExp/output/analysis/coord-family-recall-pilot-center-val64/manual_audit_summary.json`

## Layer A: Clean Verifier Benchmark
### center_parameterization @ temperature=0.3
- `commitment` GT-vs-hard-neg: AUROC=`0.5091` AUPRC=`0.3130` counted=`1942`
- `counterfactual` GT-vs-hard-neg: AUROC=`0.6067` AUPRC=`0.4637` counted=`1942`
- `combined_linear` GT-vs-hard-neg: AUROC=`0.5465` AUPRC=`0.3908` counted=`1942`

## Layer B1: Rollout Collection Health
- `center_parameterization` temp=`0.3` valid=`yes` pred_total=`516` unmatched=`165` nonempty_rate=`1.0000` invalid_reason=`NA`

## Layer B2: Rollout Proposal Benchmark
### center_parameterization @ temperature=0.3
- `commitment` matched-vs-unmatched: AUROC=`0.5928` AUPRC=`0.7882` counted=`516`
- `commitment` unmatched top-k stats: `{"10": {"count": 10, "nearest_gt_iou_ge_0.3_rate": 0.5, "nearest_gt_iou_ge_0.5_rate": 0.1}, "25": {"count": 25, "nearest_gt_iou_ge_0.3_rate": 0.44, "nearest_gt_iou_ge_0.5_rate": 0.04}, "50": {"count": 50, "nearest_gt_iou_ge_0.3_rate": 0.36, "nearest_gt_iou_ge_0.5_rate": 0.04}}`
- `counterfactual` matched-vs-unmatched: AUROC=`0.6711` AUPRC=`0.8144` counted=`516`
- `counterfactual` unmatched top-k stats: `{"10": {"count": 10, "nearest_gt_iou_ge_0.3_rate": 0.4, "nearest_gt_iou_ge_0.5_rate": 0.0}, "25": {"count": 25, "nearest_gt_iou_ge_0.3_rate": 0.4, "nearest_gt_iou_ge_0.5_rate": 0.04}, "50": {"count": 50, "nearest_gt_iou_ge_0.3_rate": 0.38, "nearest_gt_iou_ge_0.5_rate": 0.1}}`
- `combined_linear` matched-vs-unmatched: AUROC=`0.7184` AUPRC=`0.8393` counted=`516`
- `combined_linear` unmatched top-k stats: `{"10": {"count": 10, "nearest_gt_iou_ge_0.3_rate": 0.4, "nearest_gt_iou_ge_0.5_rate": 0.0}, "25": {"count": 25, "nearest_gt_iou_ge_0.3_rate": 0.44, "nearest_gt_iou_ge_0.5_rate": 0.08}, "50": {"count": 50, "nearest_gt_iou_ge_0.3_rate": 0.42, "nearest_gt_iou_ge_0.5_rate": 0.06}}`
- commitment/counterfactual correlation: `-0.1716`
- calibration: skipped (`v1 study reports raw log-probability proxies only`)
- audit pack: count=`24` index=`/data/CoordExp/output/analysis/coord-family-recall-pilot-center-val64/checkpoints/center-parameterization/audit_pack/index.jsonl`

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
  scoring drift / exclusions: `{}`
  collection gate exclusions: `{}`
- commitment can remain high on visually plausible wrong-location boxes; counterfactual is the intended corrective signal.
- manual audit labels are missing, so the final recommendation is intentionally downgraded.
- best overall proxy on the current summary tables: `counterfactual`.
