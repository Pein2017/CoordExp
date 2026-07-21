## Why

CoordExp-Swift can already materialize the required Qwen3-VL base, DoRA
adapter, and selected-token embedding delta into a standard tied-weight model
snapshot that vLLM can load. Ordinary inference is nevertheless blocked by
historical qualification receipts whose exact application-source hashes,
engine arguments, concurrency values, and optional composition-comparison
evidence drift whenever the implementation or operational settings change.
Those gates make provenance evidence determine executability even when the
current model composition and current engine are structurally valid.

The runtime should fail when the requested model cannot be composed or
executed correctly. Historical source equality and small HF/vLLM numerical
differences should remain visible diagnostics, not launch authorization.

## What Changes

- Replace mandatory historical runtime-qualification binding with a compact
  operational preflight based on the current resolved inputs, validated
  execution-model receipt, supported engine argument ranges, actual engine
  construction, and actual decode-result contracts.
- Keep content-addressed base/DoRA/selected-token-delta materialization,
  source-payload hashing, atomic publication, tied-weight validation, tensor
  shape/dtype validation, residue checks, and cache-hit byte validation.
- Stop requiring a composition-fidelity sidecar before materialized vLLM
  inference. Existing HF comparison and composition probe receipts remain
  optional diagnostics and explicit audit tools.
- Stop requiring exact equality with historical application-source hashes,
  `gpu_memory_utilization`, `max_model_len`, or previously probed concurrency
  values. Record the current version and effective engine settings instead.
- Make raw-model likelihood replay depend on live token/prompt/stop/alignment
  validation rather than a historical forced-replay source receipt.
- Resolve every authored model, data, adapter, and embedding-delta path to an
  absolute path owned by its declaring config; fail early with the declared,
  declaring, and resolved paths when an input is absent. Do not search other
  worktrees or output roots implicitly.
- Treat the first successful real multimodal decode in a shard as the live
  operational smoke. Invalid native output, trace misalignment, or unusable
  stop/prompt/image evidence remains fatal before final artifacts publish.

## Capabilities

### Modified Capabilities

- `coordexp-swift-infer-config-runtime`: replace exact receipt qualification
  with live operational support and explicit absolute path ownership.
- `coordexp-swift-infer-execution-model`: retain strict A+B+C structural
  composition while making behavioral comparison receipts optional.
- `coordexp-swift-infer-backend-trace`: make actual engine/decode/replay
  evidence authoritative and historical qualification informational.
- `coordexp-swift-infer-pipeline`: keep execution-model resolution in the
  controller while removing composition-fidelity binding from its ownership.
- `coordexp-swift-infer-benchmark-smoke`: distinguish optional parity studies
  from the required real operational smoke.

## Impact

- Primary source impact: `src/config/inference.py`,
  `src/inference/execution_model.py`, `src/inference/runtime.py`,
  `src/inference/vllm_backend.py`, and `src/inference/vllm_qualification.py`.
- Existing qualification probe scripts and receipts remain readable historical
  evidence, but ordinary inference no longer depends on their exact hashes or
  parameter values.
- Evaluator confidence, `pred[*].score`, policy likelihood, raw likelihood,
  parser semantics, output ordering, and artifact filenames do not change.
- The public entrypoint remains `python -m src.infer --config CONFIG.yaml`.
