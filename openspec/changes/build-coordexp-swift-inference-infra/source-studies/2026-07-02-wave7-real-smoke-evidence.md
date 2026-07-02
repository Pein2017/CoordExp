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
CUDA_VISIBLE_DEVICES=4 python -m src.infer --config configs/coordexp_swift/infer/wave7_real_adapter_smoke.yaml
CUDA_VISIBLE_DEVICES=4 python -m src.infer --config configs/coordexp_swift/infer/wave7_real_production_adapter_smoke.yaml
python - <<'PY'
from pathlib import Path
from src.eval.detection_consumer import evaluate_scored_detection_artifacts
artifact_dir = Path("outputs/coordexp_swift/infer/wave7_real_smokes/wave7-real-production-adapter-smoke")
result = evaluate_scored_detection_artifacts(
    artifact_dir=artifact_dir,
    output_dir=artifact_dir / "eval",
)
print(result.metrics_path)
print(result.metrics)
PY
```

## Current Outcome

- PARTIAL, base single-row smoke: `outputs/coordexp_swift/infer/wave7_real_smokes/wave7-real-base-single-smoke-20260702T182753Z`
  - Required artifacts present: `configs/resolved.json`, `configs/resolved.yaml`, `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, `gt_vs_pred_scored.jsonl.provenance.json`, `pred_token_trace.jsonl`, `parse_diagnostics.jsonl`, `image_plan.jsonl`, `summary.json`, `run_manifest.json`.
  - `summary.json`: `terminal_status=completed`, `row_count=1`, `decode_success_count=1`, `trace_row_count=32`, `scored_artifact_materialized=true`, `benchmark_eligible=false`.
  - Gap: the base output did not parse into valid compact predictions: `parser_failure_count=1`, `scoreable_prediction_count=0`; therefore it did not produce non-empty selected-token scoring evidence.
- PARTIAL, base two-row batched smoke: `outputs/coordexp_swift/infer/wave7_real_smokes/wave7-real-base-batched-smoke`
  - Required artifacts present: `configs/resolved.json`, `configs/resolved.yaml`, `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, `gt_vs_pred_scored.jsonl.provenance.json`, `pred_token_trace.jsonl`, `parse_diagnostics.jsonl`, `image_plan.jsonl`, `summary.json`, `run_manifest.json`.
  - `summary.json`: `terminal_status=completed`, `row_count=2`, `decode_success_count=2`, `trace_row_count=64`, `scored_artifact_materialized=true`, `benchmark_eligible=false`.
  - Gap: no observed `is_stop=true` and no observed `is_pad=true`; this real run covered length-stop handling, while targeted backend tests retain terminal stop and post-stop pad coverage. This is not real-smoke evidence for natural stop/pad handling.
  - Detection consumer wrote `outputs/coordexp_swift/infer/wave7_real_smokes/wave7-real-base-batched-smoke/eval/metrics.json` with `benchmark_metric=false`, `row_count=2`, `gt_object_count=4`, `pred_object_count=0`.
- PASS, adapter-enabled smoke after repaired smoke-only delta metadata and inference freeze/status fix: `outputs/coordexp_swift/infer/wave7_real_smokes/wave7-real-adapter-smoke-20260702T192643Z`
  - Repaired payload root: `outputs/coordexp_swift/infer/wave7_real_smokes/repaired_special_token_embeddings_step2`
  - Repair receipt: `outputs/coordexp_swift/infer/wave7_real_smokes/repaired_special_token_embeddings_step2/repair_receipt.json`
  - Repair scope: smoke-only metadata repair; tensor copied byte-for-byte from `outputs/smoke/production_mimic/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_r16a32_llm_12000_accelerate8_ebs64_2step_warmup0p1_eval_patchproof-smoke-a8-r16a32-ebs64-receipt-20260702T164342Z/checkpoints/step-2/special_token_embeddings`.
  - Repair checks: source `base_model_path` resolved to `/data/Qwen3-VL/model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp-natural-adjacent`; token strings/ids matched the runtime tokenizer default selection; tensor key was `shared_embed_delta`; tensor shape was `[1004, 2048]`; tensor SHA was `ed1fd32b65cc1a7df647e4d5d2307712f501a22b7b3c1ab0007b8565e3cfb726`.
  - Repaired identity fields: `base_config_sha256=c7d172360d0ff881db59a6f34865c379bbef40d976ad79cfe5fbbf50483655de`, `tokenizer_sha256=ca7e80dee65c629af3b314e76a7587490db3f4e6412df4af9f3b690a9e9916f8`.
  - `summary.json`: `terminal_status=completed`, `row_count=1`, `decode_success_count=1`, `trace_row_count=32`, `scored_artifact_materialized=true`, `benchmark_eligible=false`.
  - `run_manifest.json`: `model_identity.family=base-plus-adapter-plus-delta`, `adapter_identity.status=validated`, `adapter_identity.requires_grad={"default": false}`, `model_identity.embedding_delta.status=loaded`, `model_identity.embedding_delta.identity.status=validated`, `model_identity.embedding_delta.load.loaded=true`, `model_identity.embedding_delta.load.tensor_shape=[1004, 2048]`.
  - Gap: generated output did not parse into valid compact predictions, so this adapter smoke proves load/identity/artifact readiness only, not final inference correctness or mAP quality.
  - Historical blockers resolved before this pass: `adapter.inference_load_result_missing` was fixed by accepting Transformers `PeftAdapterMixin.load_adapter -> None` only with equivalent config, safetensors payload, status, and state evidence; the strict null-SHA blocker was avoided only through the smoke-only repaired metadata copy above; the PEFT `requires_grad={"default": true}` inference-status gap was fixed by freezing the loaded model before status validation and re-running this smoke.
- PASS, production-adapter two-row smoke with corrected COCO prompt: `outputs/coordexp_swift/infer/wave7_real_smokes/wave7-real-production-adapter-smoke`
  - Config: `configs/coordexp_swift/infer/wave7_real_production_adapter_smoke.yaml`
  - Production adapter: `outputs/prod/coordexp_swift/qwen3_vl_2b_desc_first_geo_sorted_pure_ce_dora_llm_12000_accelerate8_ebs128_4epoch-prod8-ebs128-4epoch-cocoprompt-20260702T052657Z/checkpoints/step-459/adapter`
  - Repaired production delta root: `outputs/coordexp_swift/infer/wave7_real_smokes/repaired_special_token_embeddings_step459`
  - Repair receipt: `outputs/coordexp_swift/infer/wave7_real_smokes/repaired_special_token_embeddings_step459/repair_receipt.json`
  - Repair scope: metadata-only repair for the legacy production checkpoint written before SHA propagation was fixed; tensor copied byte-for-byte from the production checkpoint delta. This repaired payload is a validated local benchmark input handle, not a benchmark-result artifact.
  - `summary.json`: `terminal_status=completed`, `row_count=2`, `decode_success_count=2`, `parser_failure_count=0`, `scoreable_prediction_count=4`, `scored_artifact_materialized=true`, `benchmark_eligible=false`.
  - `run_manifest.json`: `template_identity.object_ordering=geo_sorted`, `adapter_identity.status=validated`, `adapter_identity.requires_grad={"default": false}`, `model_identity.embedding_delta.status=loaded`.
  - `pred_token_trace.jsonl`: 48 trace rows, `is_stop=2`, `is_pad=2`, and finite selected-token logprobs for four parsed object spans.
  - `parse_diagnostics.jsonl`: both rows accepted with `valid_prediction_count=2`.
  - Detection consumer wrote `outputs/coordexp_swift/infer/wave7_real_smokes/wave7-real-production-adapter-smoke/eval/metrics.json` with `benchmark_metric=false`, `row_count=2`, `gt_object_count=4`, `pred_object_count=4`, and `scored_pred_count=4`.

## Launch Boundary

Full benchmark launch remains blocked on explicit user approval. Wave 7 smoke success proves implementation readiness for the benchmark gate, but it is still tiny smoke evidence and not final benchmark evidence.
