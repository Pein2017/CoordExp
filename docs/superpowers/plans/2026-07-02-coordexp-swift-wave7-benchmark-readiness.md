# CoordExp-Swift Wave 7 Benchmark Readiness Packet

STATUS: BLOCKED_PENDING_FULL_BENCHMARK_APPROVAL

This packet is a blocked launch gate only. Adapter smoke readiness is complete for Wave 7, but full benchmark launch is still blocked pending explicit user approval. Tiny and sample-limited smokes are smoke/partial evidence only, not benchmark evidence.

## Production Handles

- Production config: `configs/coordexp_swift/infer/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_benchmark.yaml`
- Dataset: `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl`
- Base model: `/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`
- Adapter checkpoint: `outputs/smoke/production_mimic/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_llm_12000_accelerate8_ebs64_2step_warmup0p1_eval_patchproof-smoke-a8-r16a32-ebs64-receipt-20260702T164342Z/checkpoints/step-2/adapter`
- Embedding delta for smoke: `outputs/coordexp_swift/infer/wave7_real_smokes/repaired_special_token_embeddings_step2`
- Original checkpoint delta source: `outputs/smoke/production_mimic/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_llm_12000_accelerate8_ebs64_2step_warmup0p1_eval_patchproof-smoke-a8-r16a32-ebs64-receipt-20260702T164342Z/checkpoints/step-2/special_token_embeddings`
- Artifact root: `outputs/coordexp_swift/infer/benchmark`
- Evaluator command:

```bash
python - <<'PY'
from pathlib import Path
from src.eval.detection_consumer import evaluate_scored_detection_artifacts

benchmark_run_dir = Path("<benchmark-run-dir>")
result = evaluate_scored_detection_artifacts(
    artifact_dir=benchmark_run_dir,
    output_dir=benchmark_run_dir / "eval",
)
print(result.metrics_path)
print(result.metrics)
PY
```
- Expected evidence scope: full validation inference only after approval; Wave 7 smokes remain tiny smoke gates.
- Rollback path: do not delete smoke or benchmark artifacts; stop the run, preserve the run directory, and revert only Wave 7 config/runtime/doc changes if the launch gate is rejected.

## Current Blockers

- Base smokes did not produce non-empty selected-token scoring evidence.
- Base smokes did not naturally observe `<|im_end|>` stop or post-stop padding.
- These gaps are covered by targeted tests only; they are not real-smoke evidence and do not justify final inference-correctness claims.
- Full benchmark launch still requires explicit user approval.

## Adapter Smoke Evidence

- Adapter smoke command: `CUDA_VISIBLE_DEVICES=4 python -m src.infer --config configs/coordexp_swift/infer/wave7_real_adapter_smoke.yaml`
- Latest adapter smoke root: `outputs/coordexp_swift/infer/wave7_real_smokes/wave7-real-adapter-smoke-20260702T190957Z`
- Smoke-only repaired delta payload: `outputs/coordexp_swift/infer/wave7_real_smokes/repaired_special_token_embeddings_step2`
- Repair receipt: `outputs/coordexp_swift/infer/wave7_real_smokes/repaired_special_token_embeddings_step2/repair_receipt.json`
- `summary.json`: `terminal_status=completed`, `scored_artifact_materialized=true`, `benchmark_eligible=false`, `scoreable_prediction_count=0`.
- `run_manifest.json`: `model_identity.family=base-plus-adapter-plus-delta`, `adapter_identity.status=validated`, `model_identity.embedding_delta.status=loaded`, `model_identity.embedding_delta.load.loaded=true`, `model_identity.embedding_delta.load.tensor_shape=[1004, 2048]`.
- Repair caveat: the original checkpoint delta had null SHA metadata; the smoke payload repaired only `base_config_sha256` and `tokenizer_sha256` after verifying resolved base path, token strings/ids, tensor shape, and byte-for-byte tensor copy.

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

## Approval Stop

Do not launch the production benchmark from this packet until the user explicitly approves the benchmark launch. Do not claim final inference correctness from Wave 7 smoke artifacts.
