# Unmatched Proposal Verifier Study

## Setup
- subset: `/data/CoordExp/output/analysis/unmatched-proposal-verifier-temperature-sweep/unmatched-proposal-verifier-full-t0p0/subset/sampled.coord.jsonl` from `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl`
- sample_count: `200`
- seed: `42`
- root_image_dir: `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_max60`

## Collection
- collection backend mode: `local`
- temperature: `0.0`
- repetition_penalty: `1.1`
- infer.generation.batch_size: `16`
- infer.backend.server_options.vllm_gpu_memory_utilization: `0.9`

## Checkpoints
- `ul-res_1024-ckpt_300_merged` -> `/data/CoordExp/output/stage2_ab/prod/ul-res_1024-ckpt_300_merged` (prompt_variant=`coco_80`, object_field_order=`desc_first`, source=`study_default`)
- `ul-res_1024-v2-ckpt_300_merged` -> `/data/CoordExp/output/stage2_ab/prod/ul-res_1024-v2-ckpt_300_merged` (prompt_variant=`coco_80`, object_field_order=`desc_first`, source=`study_default`)

## Proxy Definitions
- commitment: desc-only average teacher-forced log-probability on the original image
- counterfactual: commitment(original) - commitment(masked bbox image)
- combined_linear: commitment + counterfactual

## Results
### ul-res_1024-ckpt_300_merged
- `commitment` GT-vs-hard-neg: AUROC=`0.5033` AUPRC=`0.3033` counted=`5402`
- `commitment` matched-vs-unmatched: AUROC=`NA` AUPRC=`NA` counted=`0`
- `counterfactual` GT-vs-hard-neg: AUROC=`0.6163` AUPRC=`0.4636` counted=`5402`
- `counterfactual` matched-vs-unmatched: AUROC=`NA` AUPRC=`NA` counted=`0`
- `combined_linear` GT-vs-hard-neg: AUROC=`0.5471` AUPRC=`0.3990` counted=`5402`
- `combined_linear` matched-vs-unmatched: AUROC=`NA` AUPRC=`NA` counted=`0`
- commitment/counterfactual correlation: `NA`
- calibration: skipped (`v1 study reports raw log-probability proxies only`)
- audit pack: count=`0` index=`/data/CoordExp/output/analysis/unmatched-proposal-verifier-temperature-sweep/unmatched-proposal-verifier-full-t0p0/checkpoints/ul-res-1024-ckpt-300-merged/audit_pack/index.jsonl`

### ul-res_1024-v2-ckpt_300_merged
- `commitment` GT-vs-hard-neg: AUROC=`0.5019` AUPRC=`0.3015` counted=`5402`
- `commitment` matched-vs-unmatched: AUROC=`NA` AUPRC=`NA` counted=`0`
- `counterfactual` GT-vs-hard-neg: AUROC=`0.6244` AUPRC=`0.4695` counted=`5402`
- `counterfactual` matched-vs-unmatched: AUROC=`NA` AUPRC=`NA` counted=`0`
- `combined_linear` GT-vs-hard-neg: AUROC=`0.5504` AUPRC=`0.4023` counted=`5402`
- `combined_linear` matched-vs-unmatched: AUROC=`NA` AUPRC=`NA` counted=`0`
- commitment/counterfactual correlation: `NA`
- calibration: skipped (`v1 study reports raw log-probability proxies only`)
- audit pack: count=`0` index=`/data/CoordExp/output/analysis/unmatched-proposal-verifier-temperature-sweep/unmatched-proposal-verifier-full-t0p0/checkpoints/ul-res-1024-v2-ckpt-300-merged/audit_pack/index.jsonl`

## Recommendation
- strongest single proxy: `counterfactual`
- does commitment + counterfactual materially outperform either single proxy? `no`
- is the signal stable across checkpoints? `yes`
- is the proxy good enough for soft pseudo-label promotion? `not yet`
- main observed failure modes:
  scoring drift / exclusions: `{}`
- commitment can remain high on visually plausible wrong-location boxes; counterfactual is the intended corrective signal.
- best overall proxy on the current summary tables: `counterfactual`.
