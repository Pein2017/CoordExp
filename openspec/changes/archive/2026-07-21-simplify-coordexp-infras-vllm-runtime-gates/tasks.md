## 1. Contract And Regression Baseline

- [x] 1.1 Add delta specs for live vLLM preflight, structural A+B+C execution
  identity, live raw-replay alignment, pipeline ownership, path ownership, and
  operational smoke.
- [x] 1.2 Add regression tests proving historical source hashes, exact engine
  values, concurrency receipts, and composition-fidelity receipts are not
  normal runtime gates.
- [x] 1.3 Add path-origin tests for missing cross-worktree adapter and delta
  payloads without fallback search.

## 2. Runtime Simplification

- [x] 2.1 Add a compact non-blocking vLLM runtime diagnostic/preflight receipt
  based on current version, current engine settings, and execution identity.
- [x] 2.2 Route vLLM session opening through live engine construction and
  backend-result validation instead of strict historical qualification.
- [x] 2.3 Route raw-model replay through live processor and token-alignment
  checks without requiring a historical forced-replay receipt.

## 3. Execution Model And Paths

- [x] 3.1 Stop automatically binding composition-fidelity sidecars during
  ordinary execution-model resolution while preserving explicit probe APIs.
- [x] 3.2 Remove composition-fidelity as a materialized vLLM launch
  prerequisite.
- [x] 3.3 Fail early on missing resolved model/data/adapter/delta inputs with
  declared path, declaring config, and absolute resolved path evidence.
- [x] 3.4 Validate adapter merge, embedding-delta fold, one-addition, and tied-
  weight outcomes against source identity before publishing a composed cache.
- [x] 3.5 Separate rank-local live/cleanup observations from semantic session
  identity and aggregate them during strict multi-rank merge.
- [x] 3.6 Reject raw tracing on an unverified vLLM version, cover the complete
  raw-engine lifecycle with cleanup, and require native request-id alignment.

## 4. Documentation And Verification

- [x] 4.1 Update current operator docs and delta specs to distinguish blocking
  live checks from optional historical diagnostics.
- [x] 4.2 Run targeted inference/config/execution-model tests and strict
  OpenSpec validation.
- [x] 4.3 Run one real BF16 composed-model vLLM multimodal smoke and retain its
  summary/provenance path. Accepted evidence:
  `outputs/coordexp_swift/infer/smoke/qwen3-vl-2b-step4887-vllm-parity-smoke-20260721T042408Z/`
  (two rank-local engines on GPUs 0 and 1).
- [x] 4.4 Run independent contract and implementation review and resolve every
  accepted P0/P1 finding.
