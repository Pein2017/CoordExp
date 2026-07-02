## Why

CoordExp-swift now has a rebuilt training path, but the repository still lacks
a self-contained inference and benchmark-evaluation surface that can load the
same base model, adapter checkpoint, template, Qwen processor policy, and
artifact contracts without falling back to legacy `src/infer/*` behavior.

This change creates the V1 offline inference infrastructure needed to compare
CoordExp-swift checkpoints through batched HF/Qwen generation, token-trace
scoring, parser diagnostics, scored artifacts, and mAP evaluation. The priority
order remains accuracy and precision first, throughput second, simplicity third,
and future extension fourth.

## What Changes

- Add a new inference entrypoint shape: thin `src/infer.py` public entry,
  invoked as `python -m src.infer --config CONFIG.yaml`, backed by implementation
  modules under `src/inference/`.
- Add a strict `InferConfig` schema rather than reusing `TrainConfig`; inference
  configs live under `configs/coordexp_swift/infer/`.
- Implement V1 HF Transformers `model.generate` as the only active backend,
  while reserving backend-neutral decode-result fields for future vLLM support.
- Require batched generation, deterministic greedy decoding by default,
  Qwen `<|im_end|>` stop handling, `max_new_tokens`, full generated-token
  traces, and no `skip_special_tokens=True` loss of raw stop evidence.
- Reuse CoordExp-swift Qwen/template/model-composition semantics: no-resize
  image processing, prompt-token parity, base model plus optional adapter, and
  optional special-token embedding delta reload.
- Parse compact object-box-closed generated text into canonical prediction rows,
  preserving raw decode text, invalid-span diagnostics, and generated
  coordinate-token evidence.
- Produce both raw and score-bearing artifacts in the same run when trace
  scoring is enabled:
  - `configs/resolved.json`
  - `configs/resolved.yaml`
  - `run_manifest.json`
  - `summary.json`
  - `gt_vs_pred.jsonl`
  - `gt_vs_pred_scored.jsonl`
  - `gt_vs_pred_scored.jsonl.provenance.json`
  - `pred_token_trace.jsonl`
  - `parse_diagnostics.jsonl`
  - `image_plan.jsonl`
- Define V1 object confidence as
  `exp(sum(selected_token_logprobs) / n_selected)` over selected schema wrapper
  and coordinate tokens, excluding free-text description/category text.
- Require scored artifacts to preserve one row per input/raw row, inline GT,
  image identity, image dimensions, row order, parser status, score provenance,
  and portable scored-artifact provenance.
- Add a minimal detection-eval consumer or explicit compatibility bridge so the
  final acceptance path can run mAP from `gt_vs_pred_scored.jsonl`.
- Keep vLLM implementation, Stage-2 rollout decoding, interactive demos, grammar
  constraints, multi-image/video inference, and old JSON assistant response
  parsing out of V1 unless later changes promote them.

## Capabilities

### New Capabilities

- `coordexp-swift-infer-config-runtime`: strict inference config, resolved
  config artifacts, model/adapter/delta identity, checkpoint alias resolution,
  no training-schema leakage, and runtime setup contracts.
- `coordexp-swift-infer-backend-trace`: backend-neutral decode request/result
  records, HF `generate` score tracing, prompt-width alignment, special-token
  stop handling, batch decode, and future vLLM reservation contracts.
- `coordexp-swift-infer-prompt-parsing`: training-aligned prompt construction,
  Qwen no-resize image-plan evidence, compact object-box-closed parsing,
  parser diagnostics, and geometry conversion contracts.
- `coordexp-swift-infer-scoring-artifacts`: selected-token scoring,
  `gt_vs_pred` / `gt_vs_pred_scored` row contracts, token trace artifacts,
  scored-artifact provenance, and metric-eligibility diagnostics.
- `coordexp-swift-infer-pipeline`: end-to-end inference orchestration, batching,
  artifact writing, counters, failure policy, and separation from metric
  reduction.
- `coordexp-swift-infer-benchmark-smoke`: real HF/Qwen smoke requirements,
  adapter-enabled smoke, evaluator-consumer compatibility, and final full
  benchmark acceptance gates.

### Modified Capabilities

- None. This change adds the first CoordExp-swift inference/eval contract set.
  Existing training-infrastructure capabilities remain unchanged.

## Impact

- Affected future source roots: `src/infer.py`, `src/inference/`, `src/config/`,
  `src/qwen/`, `src/templates/`, `src/data/geometry.py`, `src/artifacts/`, and
  `src/eval/`.
- Affected future tests: inference config validation, HF trace alignment,
  prompt/image parity, parser salvage, selected-token scoring, artifact
  provenance, evaluator compatibility, and real HF/Qwen smoke tests.
- Affected configs: new inference configs under `configs/coordexp_swift/infer/`;
  legacy `configs/infer/*` remains reference-only.
- External dependencies remain Transformers, PyTorch, PEFT, and Qwen3-VL
  processor/model behavior. vLLM is reserved for future support but not required
  by V1 implementation.
- Implementation remains blocked until this OpenSpec change, the corresponding
  superpower implementation plan, and review-convergence triage are approved by
  the user.
