# CoordExp-Swift Wave 7 / Val200 Validation Packet

STATUS: VAL200_ACCEPTED_NO_FULL_DATASET_REQUIRED

This packet originally blocked on a full-dataset benchmark launch. The current
2026-07-03 decision supersedes that gate: full validation-dataset eval is not
required for CoordExp-Swift V1 readiness. The fixed val200 inference/eval run is
sufficient when scored artifacts and Swift evaluator mAP/mRecall metrics are
present. Tiny Wave 7 smokes remain implementation evidence only.

## Production Handles

- Accepted val200 config: `configs/coordexp_swift/infer/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_step917_val200.yaml`
- Accepted val200 dataset: `outputs/coordexp_swift/infer/val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl`
- Base model: `/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`
- Adapter checkpoint: `outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_llm_12000_accelerate8_ebs64_4epoch_warmup0p1-prod8-r16a32-ebs64-warmup0p1-20260702T170007Z/checkpoints/step-917/adapter`
- Official repaired embedding delta support payload for accepted val200 launch: `outputs/coordexp_swift/infer/val200_support/repaired_special_token_embeddings_step917`
- Repair receipt: `outputs/coordexp_swift/infer/val200_support/repaired_special_token_embeddings_step917/repair_receipt.json`
- Artifact root: `outputs/coordexp_swift/infer/val200/qwen3-vl-2b-desc-first-geo-sorted-pure-ce-dora-r16a32-step917-val200-20260703T035007Z`
- Evaluator command:

```bash
python - <<'PY'
from pathlib import Path
from src.eval.detection_consumer import evaluate_scored_detection_artifacts

	benchmark_run_dir = Path("outputs/coordexp_swift/infer/val200/qwen3-vl-2b-desc-first-geo-sorted-pure-ce-dora-r16a32-step917-val200-20260703T035007Z")
result = evaluate_scored_detection_artifacts(
    artifact_dir=benchmark_run_dir,
    output_dir=benchmark_run_dir / "eval",
)
print(result.metrics_path)
print(result.metrics)
PY
```
- Accepted evidence scope: fixed val200 local validation.
- Accepted metrics path: `outputs/coordexp_swift/infer/val200/qwen3-vl-2b-desc-first-geo-sorted-pure-ce-dora-r16a32-step917-val200-20260703T035007Z/eval_coco_fixed_gt_scale/metrics.json`.
- Accepted metrics: `mAP=0.4111788135144427`, `mAP_50=0.5616311086141887`, `mAP_75=0.43402742703563857`, `mRecall=0.4790356074108587`.
- Optional full-dataset benchmark config remains available at `configs/coordexp_swift/infer/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_benchmark.yaml`, but it is not a required V1 gate.
- Scheduler caveat: `step-917` remains valid model/eval evidence, but the
  historical production run should not be cited as clean warmup/cosine schedule
  evidence unless its actual LR trajectory is reconstructed.
- Rollback path: do not delete smoke, val200, or benchmark artifacts; preserve the run directory and revert only the relevant config/runtime/doc changes if a future broader launch is rejected.

## Current Blockers

- No blocker remains for the V1 val200 validation claim.
- Tiny real-smoke success alone still does not justify mAP claims; use the fixed
  val200 metrics above.
- Full validation-dataset evaluation remains optional and requires a new explicit
  user request.

## Adapter Smoke Evidence

- Adapter smoke command: `CUDA_VISIBLE_DEVICES=4 python -m src.infer --config configs/coordexp_swift/infer/wave7_real_adapter_smoke.yaml`
- Latest adapter smoke root: `outputs/coordexp_swift/infer/wave7_real_smokes/wave7-real-adapter-smoke-20260702T192643Z`
- Smoke-only repaired delta payload: `outputs/coordexp_swift/infer/wave7_real_smokes/repaired_special_token_embeddings_step2`
- Repair receipt: `outputs/coordexp_swift/infer/wave7_real_smokes/repaired_special_token_embeddings_step2/repair_receipt.json`
- `summary.json`: `terminal_status=completed`, `scored_artifact_materialized=true`, `benchmark_eligible=false`, `scoreable_prediction_count=0`.
- `run_manifest.json`: `model_identity.family=base-plus-adapter-plus-delta`, `adapter_identity.status=validated`, `adapter_identity.requires_grad={"default": false}`, `model_identity.embedding_delta.status=loaded`, `model_identity.embedding_delta.load.loaded=true`, `model_identity.embedding_delta.load.tensor_shape=[1004, 2048]`.
- Repair caveat: the original checkpoint delta had null SHA metadata; the smoke payload repaired only `base_config_sha256` and `tokenizer_sha256` after verifying resolved base path, token strings/ids, tensor shape, and byte-for-byte tensor copy.

## Production-Adapter Smoke Evidence

- Production-adapter smoke command: `CUDA_VISIBLE_DEVICES=4 python -m src.infer --config configs/coordexp_swift/infer/wave7_real_production_adapter_smoke.yaml`
- Smoke root: `outputs/coordexp_swift/infer/wave7_real_smokes/wave7-real-production-adapter-smoke`
- Production adapter: `outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate8_ebs128_4epoch-prod8-ebs128-4epoch-cocoprompt-20260702T052657Z/checkpoints/step-459/adapter`
- Locally repaired production delta payload: `outputs/coordexp_swift/infer/wave7_real_smokes/repaired_special_token_embeddings_step459`
- Repair receipt: `outputs/coordexp_swift/infer/wave7_real_smokes/repaired_special_token_embeddings_step459/repair_receipt.json`
- `summary.json`: `terminal_status=completed`, `row_count=2`, `decode_success_count=2`, `parser_failure_count=0`, `scoreable_prediction_count=4`, `scored_artifact_materialized=true`, `benchmark_eligible=false`.
- `pred_token_trace.jsonl`: 48 trace rows with `is_stop=2` and `is_pad=2`.
- Detection consumer wrote `eval/metrics.json` with `benchmark_metric=false`, `gt_object_count=4`, `pred_object_count=4`, and `scored_pred_count=4`.

## Wave 7 Smoke Commands

Base single-row smoke:

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.infer --config configs/coordexp_swift/infer/wave7_real_base_single_smoke.yaml
```

Base two-row batched smoke:

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.infer --config configs/coordexp_swift/infer/wave7_real_base_batched_smoke.yaml
```

Adapter-enabled smoke:

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.infer --config configs/coordexp_swift/infer/wave7_real_adapter_smoke.yaml
```

Production-adapter two-row smoke:

```bash
CUDA_VISIBLE_DEVICES=4 python -m src.infer --config configs/coordexp_swift/infer/wave7_real_production_adapter_smoke.yaml
```

## Approval Stop

Do not launch the optional full validation-dataset benchmark from this packet
unless the user explicitly asks for it. Do not claim final inference/eval
readiness from Wave 7 smoke artifacts alone; use the accepted fixed val200
artifact and metrics paths above.
