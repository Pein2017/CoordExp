# CoordExp-Swift Wave 7 Benchmark Readiness Packet

STATUS: READY_FOR_APPROVAL

This packet is a launch gate only. Launch is blocked pending explicit user approval. Tiny and sample-limited smokes are smoke/partial evidence only, not benchmark evidence.

## Production Handles

- Production config: `configs/coordexp_swift/infer/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_benchmark.yaml`
- Dataset: `/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl`
- Base model: `/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`
- Adapter checkpoint: `outputs/smoke/production_mimic/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_llm_12000_accelerate8_ebs64_2step_warmup0p1_eval_patchproof-smoke-a8-r16a32-ebs64-receipt-20260702T164342Z/checkpoints/step-2/adapter`
- Embedding delta: `outputs/smoke/production_mimic/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_llm_12000_accelerate8_ebs64_2step_warmup0p1_eval_patchproof-smoke-a8-r16a32-ebs64-receipt-20260702T164342Z/checkpoints/step-2/special_token_embeddings`
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
