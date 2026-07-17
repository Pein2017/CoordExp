## Why

CoordExp-Swift inference currently exposes a reserved `vllm` backend name but
rejects it at validation time, leaving throughput and raw-model likelihood
research dependent on the slower HF execution path. The accepted checkpoints
also combine DoRA with selected-token embedding deltas, which vLLM cannot load
natively, so backend support must preserve the exact composed model rather than
silently evaluating different weights.

## What Changes

- Add an offline vLLM backend invoked from a fresh rank-local worker for every
  single-image Qwen3-VL run, including one-GPU runs. The qualified engine may
  own child processes, while the existing controller owns one worker per
  useful visible GPU and verifies process-exit cleanup before publication.
- Add a content-addressed execution-model materializer that merges DoRA and
  folds selected-token embedding deltas into a standard tied-weight HF
  checkpoint before vLLM loading.
- Replace the HF-shaped decode request and pipeline-owned backend lifecycle
  with semantic requests and backend-owned sessions.
- Preserve existing evaluator confidence semantics: `logprob`,
  `selected_logprobs`, `pred[*].score`, and `pred_score_version: 1` continue to
  use decode-policy likelihood.
- Add opt-in generated-token `raw_model_logprob` evidence without promoting it
  to evaluator score ownership.
- Make canonical scored inference deterministic and reject currently accepted
  but unsupported stochastic or evidence-disabling configurations instead of
  recording no-op settings.
- Qualify one exact vLLM version through real Qwen3-VL probes and a durable
  passed qualification receipt, and record the effective backend, model,
  processor, likelihood, CUDA, and engine identities in run artifacts.
- Preserve the public `python -m src.infer --config ...` command, canonical
  raw/scored artifact family, prompt and image semantics, deterministic row
  sharding, strict merge behavior, and the existing detection evaluator.
- Defer vLLM server mode, tensor or pipeline parallelism, async APIs,
  multi-image/video inputs, stochastic sampling, grammar constraints, and
  top-k or full-vocabulary traces.

## Capabilities

### New Capabilities

- `coordexp-swift-infer-execution-model`: content-addressed base/DoRA/selected-
  token-delta composition, identity, atomic publication, cache validation, and
  worker handoff.

### Modified Capabilities

- `coordexp-swift-infer-config-runtime`: accept a strict vLLM backend block,
  require deterministic/evidence-bearing inference, and record the qualified
  runtime version and actual execution-model identity.
- `coordexp-swift-infer-backend-trace`: replace the future-vLLM reservation
  with executable backend-session, policy-likelihood, raw-model-likelihood,
  stop, prompt-parity, and lifecycle contracts.
- `coordexp-swift-infer-prompt-parsing`: allow backend-owned Qwen multimodal
  projection while preserving one-image, no-resize, prompt-token, geometry,
  and image-plan evidence.
- `coordexp-swift-infer-pipeline`: move model loading, native input projection,
  batching, replay, and cleanup behind backend sessions.
- `coordexp-swift-infer-data-parallel-runtime`: execute one TP=1 vLLM engine
  per existing rank-local worker while preserving current outer sharding and
  deterministic merge behavior.
- `coordexp-swift-infer-scoring-artifacts`: add raw-model likelihood evidence
  and backend execution receipts while preserving policy-likelihood score
  ownership and evaluator compatibility.
- `coordexp-swift-infer-benchmark-smoke`: add real base, composed-checkpoint,
  likelihood, two-GPU, eight-GPU, and matched val200 acceptance gates.

## Impact

- Primary source impact: `src/config/inference.py`, `src/inference/`, existing
  adapter and special-token embedding owner APIs, and inference configs.
- Primary test impact: config projection, HF regression, execution-model
  materialization, Qwen multimodal parity, likelihood alignment, strict merge,
  worker lifecycle, and real GPU smoke coverage.
- Dependency impact: treat installed vLLM `0.14.1` as the initial candidate
  and qualify it only after the Wave 0 receipt passes with Transformers
  `4.57.1`, PEFT `0.17.1`, and Torch `2.9.1`; any version change requires a new
  qualification receipt before support is claimed.
- Artifact impact: `pred_token_trace.jsonl`, provenance, manifests, summaries,
  and rank-local merge identity gain additive backend and raw-likelihood
  evidence. Evaluator input rows and confidence meaning remain unchanged.
- Operational impact: composed vLLM models are derived caches under
  `model_cache/coordexp_swift/vllm_materialized/`, not canonical training
  checkpoints or exported replacement models.
