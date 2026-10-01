# Unmatched Proposal Verifier Study

## Setup
- subset: `/data/CoordExp/output/analysis/unmatched-proposal-verifier-stage2-parity-smoke/subset/sampled.coord.jsonl` from `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`
- sample_count: `8`
- seed: `42`
- root_image_dir: `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60`
- run stages: `prepare, collect, gate, score, audit, report`

## Collection
- collection backend mode: `stage2_parity_vllm`
- temperature: `0.3`
- repetition_penalty: `1.1`
- infer.generation.batch_size: `16`
- authoritative temperatures: `0.0, 0.3, 0.5, 0.7`
- infer.backend.server_options.vllm_gpu_memory_utilization: `0.8`

## Checkpoints
- `ul-res_1024-ckpt_300_merged` -> `/data/CoordExp/output/stage2_ab/prod/ul-res_1024-ckpt_300_merged` (prompt_variant=`coco_80`, object_field_order=`desc_first`, source=`study_default`)

## Proxy Definitions
- commitment: desc-only average teacher-forced log-probability on the original image
- counterfactual: commitment(original) - commitment(masked bbox image)
- combined_linear: commitment + counterfactual

## Aggregate Tables
- clean GT summary: `/data/CoordExp/output/analysis/unmatched-proposal-verifier-stage2-parity-smoke/gt_clean_proxy_metrics_by_temp.csv`
- collection health: `/data/CoordExp/output/analysis/unmatched-proposal-verifier-stage2-parity-smoke/collection_health_by_temp.csv`
- rollout summary: `/data/CoordExp/output/analysis/unmatched-proposal-verifier-stage2-parity-smoke/rollout_proxy_metrics_by_temp.csv`
- manual audit summary: `/data/CoordExp/output/analysis/unmatched-proposal-verifier-stage2-parity-smoke/manual_audit_summary.json`

## Layer A: Clean Verifier Benchmark
### ul-res_1024-ckpt_300_merged @ temperature=0.3
- `commitment` GT-vs-hard-neg: AUROC=`0.5606` AUPRC=`0.4664` counted=`94`
- `counterfactual` GT-vs-hard-neg: AUROC=`0.7221` AUPRC=`0.6358` counted=`94`
- `combined_linear` GT-vs-hard-neg: AUROC=`0.6465` AUPRC=`0.6127` counted=`94`

## Layer B1: Rollout Collection Health
- `ul-res_1024-ckpt_300_merged` temp=`0.3` valid=`no` pred_total=`30` unmatched=`6` nonempty_rate=`1.0000` invalid_reason=`low_pred_count_total,low_unmatched_count`

## Layer B2: Rollout Proposal Benchmark
### ul-res_1024-ckpt_300_merged @ temperature=0.3
- excluded from main rollout comparison: `low_pred_count_total,low_unmatched_count`

## Layer C: Manual Audit
- labels loaded: `no`
- labeled count: `0`
- label counts: `{}`
- precision@k: `{"10": {"count": 0, "real_visible_object_rate": null}, "25": {"count": 0, "real_visible_object_rate": null}, "50": {"count": 0, "real_visible_object_rate": null}}`

## Recommendation
- strongest single proxy: `counterfactual`
- does commitment + counterfactual materially outperform either single proxy? `no`
- is the signal stable across checkpoints? `mixed`
- is rollout evidence valid enough for interpretation? `no`
- is the proxy good enough for soft pseudo-label promotion? `promising but not yet promotion-ready`
- main observed failure modes:
  scoring drift / exclusions: `{"collection_invalid:low_pred_count_total,low_unmatched_count": 90}`
  collection gate exclusions: `{"ul-res_1024-ckpt_300_merged": "low_pred_count_total,low_unmatched_count"}`
- commitment can remain high on visually plausible wrong-location boxes; counterfactual is the intended corrective signal.
- manual audit labels are missing, so the final recommendation is intentionally downgraded.
- best overall proxy on the current summary tables: `counterfactual`.
