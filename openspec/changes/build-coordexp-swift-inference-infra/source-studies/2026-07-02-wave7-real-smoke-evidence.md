# Wave 7 Real-Smoke Evidence

This note records Wave 7 smoke handles and outcomes. Tiny runs are smoke/partial evidence only and are not final benchmark evidence.

## Pinned Fixtures And Configs

- Single-row fixture: `tests/fixtures/smoke/qwen3_vl_single_image_pack/examples.single.jsonl`
- Two-row fixture: `tests/fixtures/smoke/qwen3_vl_single_image_pack/examples.jsonl`
- Base single-row smoke config: `configs/coordexp_swift/infer/wave7_real_base_single_smoke.yaml`
- Base two-row smoke config: `configs/coordexp_swift/infer/wave7_real_base_batched_smoke.yaml`
- Adapter smoke config: `configs/coordexp_swift/infer/wave7_real_adapter_smoke.yaml`
- Benchmark packet: `docs/superpowers/plans/2026-07-02-coordexp-swift-wave7-benchmark-readiness.md`

## Commands Run

```bash
CUDA_VISIBLE_DEVICES=4 python -m src.infer --config configs/coordexp_swift/infer/wave7_real_base_single_smoke.yaml
CUDA_VISIBLE_DEVICES=4 python -m src.infer --config configs/coordexp_swift/infer/wave7_real_base_batched_smoke.yaml
CUDA_VISIBLE_DEVICES=4 python -m src.infer --config configs/coordexp_swift/infer/wave7_real_adapter_smoke.yaml
python - <<'PY'
from pathlib import Path
from src.eval.detection_consumer import evaluate_scored_detection_artifacts
artifact_dir = Path("outputs/coordexp_swift/infer/wave7_real_smokes/wave7-real-base-batched-smoke")
result = evaluate_scored_detection_artifacts(
    artifact_dir=artifact_dir,
    output_dir=artifact_dir / "eval",
)
print(result.metrics_path)
print(result.metrics)
PY
```

## Current Outcome

- PASS, base single-row smoke: `outputs/coordexp_swift/infer/wave7_real_smokes/wave7-real-base-single-smoke-20260702T182753Z`
  - Required artifacts present: `configs/resolved.json`, `configs/resolved.yaml`, `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, `gt_vs_pred_scored.jsonl.provenance.json`, `pred_token_trace.jsonl`, `parse_diagnostics.jsonl`, `image_plan.jsonl`, `summary.json`, `run_manifest.json`.
  - `summary.json`: `terminal_status=completed`, `row_count=1`, `decode_success_count=1`, `trace_row_count=32`, `scored_artifact_materialized=true`, `benchmark_eligible=false`.
  - The base output did not parse into valid compact predictions: `parser_failure_count=1`, `scoreable_prediction_count=0`.
- PASS, base two-row batched smoke: `outputs/coordexp_swift/infer/wave7_real_smokes/wave7-real-base-batched-smoke`
  - Required artifacts present: `configs/resolved.json`, `configs/resolved.yaml`, `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, `gt_vs_pred_scored.jsonl.provenance.json`, `pred_token_trace.jsonl`, `parse_diagnostics.jsonl`, `image_plan.jsonl`, `summary.json`, `run_manifest.json`.
  - `summary.json`: `terminal_status=completed`, `row_count=2`, `decode_success_count=2`, `trace_row_count=64`, `scored_artifact_materialized=true`, `benchmark_eligible=false`.
  - Trace flags: no observed `is_stop=true` and no observed `is_pad=true`; this real run covered length-stop handling, while targeted backend tests retain terminal stop and post-stop pad coverage.
  - Detection consumer wrote `outputs/coordexp_swift/infer/wave7_real_smokes/wave7-real-base-batched-smoke/eval/metrics.json` with `benchmark_metric=false`, `row_count=2`, `gt_object_count=4`, `pred_object_count=0`.
- BLOCKED, adapter-enabled smoke: `outputs/coordexp_swift/infer/wave7_real_smokes/wave7-real-adapter-smoke`
  - The run wrote only `configs/resolved.json` and `configs/resolved.yaml` before runtime setup failed.
  - Exact blocker: `RuntimeContractError[adapter.inference_load_result_missing]`, message `inference DoRA adapter load did not return load_result evidence`, context `adapter_name=default`, adapter path `outputs/smoke/production_mimic/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_llm_12000_accelerate8_ebs64_2step_warmup0p1_eval_patchproof-smoke-a8-r16a32-ebs64-receipt-20260702T164342Z/checkpoints/step-2/adapter`.
  - The embedding-delta identity check was not reached. Preflight risk remains: candidate metadata records `base_config_sha256: null` and a base-model path under `/data/Qwen3-VL/...`, while the assigned handle is `/data/CoordExp/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`.

## Launch Boundary

Full benchmark launch is blocked pending explicit user approval. Wave 7 smoke success, if achieved, only proves implementation readiness for the benchmark gate.
