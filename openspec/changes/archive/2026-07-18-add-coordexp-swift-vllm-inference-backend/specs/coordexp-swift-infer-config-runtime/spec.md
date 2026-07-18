## ADDED Requirements

### Requirement: Strict backend-discriminated inference config
`backend.type` SHALL accept `hf` and `vllm`. Shared `model` config SHALL contain
base-model identity, dtype, and processor policy, but MUST NOT contain
backend-only execution controls. HF MUST require a strict `backend.hf` block
containing `attn_implementation` and `patch_embed_linearization`; vLLM MUST
require a strict `backend.vllm` block containing finite
`gpu_memory_utilization` in `(0, 1]`. The non-selected backend block MUST be
rejected, unknown fields MUST fail validation, and arbitrary engine kwargs MUST
NOT be exposed as stable config.

#### Scenario: Explicit vLLM backend
- **WHEN** a config selects `backend.type: vllm` with a valid vLLM block
- **THEN** config validation succeeds without loading either backend

#### Scenario: HF with vLLM settings
- **WHEN** a config selects HF and includes `backend.vllm`
- **THEN** validation fails before model loading

#### Scenario: vLLM carries an HF attention implementation
- **WHEN** a config selects vLLM but includes `backend.hf` or legacy
  `model.attn_implementation`
- **THEN** validation fails instead of recording an ignored execution setting

### Requirement: Explicit dual-backend precision roles
Dynamic HF SHALL remain a first-class backend for direct base, DoRA, and
selected-token embedding-delta composition. vLLM SHALL remain a first-class
offline backend over a validated execution model. FP32 SHALL be the strict
cross-backend parity mode. BF16 vLLM MAY be used for supported throughput runs,
but its artifacts and documentation MUST NOT claim strict HF parity.

#### Scenario: Dynamic HF composed checkpoint
- **WHEN** `backend.type: hf` selects a base plus DoRA plus embedding delta
- **THEN** runtime loads that composition directly without requiring a vLLM
  execution-model snapshot

#### Scenario: BF16 vLLM throughput run
- **WHEN** `backend.type: vllm` and `model.dtype: bf16` are selected
- **THEN** inference remains executable but the run cannot satisfy the strict
  FP32 cross-backend parity gate

### Requirement: Deterministic evidence-bearing scored inference
Canonical inference MUST require `temperature: 0.0`, `top_p: 1.0`, scoring
enabled, token trace writing enabled, and parse diagnostics writing enabled.
Unsupported stochastic or evidence-disabling values MUST fail validation
instead of being ignored. Explicit positive repetition penalties remain
supported and MUST be projected to the selected backend.

#### Scenario: Stochastic config rejected
- **WHEN** temperature is nonzero or top-p is below one
- **THEN** validation fails with a deterministic-inference contract error

#### Scenario: Required trace disabled
- **WHEN** scoring is enabled but token trace writing is false
- **THEN** validation fails before runtime setup

### Requirement: Optional raw-model likelihood trace
Inference config SHALL expose `artifacts.include_raw_model_logprob` with default
`false`. Enabling it MUST request aligned raw-model likelihoods for every
generated non-pad token. It MUST NOT change selected-token scoring policy,
generation behavior, parsing, or evaluator input rows.

#### Scenario: Raw likelihood omitted by default
- **WHEN** the field is absent
- **THEN** inference runs with policy likelihood only and records raw likelihood
  tracing as disabled

#### Scenario: Raw likelihood requested
- **WHEN** the field is true
- **THEN** backend setup requires raw-likelihood support and failures are fatal

### Requirement: Qualified vLLM version
The runtime MUST accept only versions whose Qwen3-VL prompt, image, stop,
policy-logprob, raw-replay, CUDA-binding, process-mode, and cleanup probes are
recorded in a matching passed runtime-qualification receipt. The initial
candidate is `0.14.1`; it becomes accepted only after its Wave 0 receipt passes.
The receipt MUST bind the probe implementation, fixture, exhaustive source-base
snapshot assets, installed dependencies, every loaded source module from the
qualified vLLM/Transformers/PEFT/Qwen utility boundary after execution, named
required execution owners, effective engine arguments, and observed results.
Runtime matching MUST distinguish invariant semantics
from explicitly qualified engine-argument values. An unqualified version,
source drift, invariant mismatch, or unprobed argument value MUST fail before
engine construction and MUST NOT be treated as benchmark evidence.
Runtime-owned qualification receipts MUST live outside an active OpenSpec
change and MUST bind repository sources by repo-relative path plus hash and
installed sources by package-relative path plus hash. Recorded absolute paths
MAY remain as diagnostics but MUST NOT determine whether an unchanged checkout
or environment is qualified.

#### Scenario: Unqualified installed version
- **WHEN** the installed vLLM version is outside the qualified set
- **THEN** runtime fails with the observed and supported versions

#### Scenario: Version string matches but receipt does not
- **WHEN** vLLM reports `0.14.1` but installed source, source-base assets,
  tokenizer/processor/template identity, or invariant engine semantics do not
  match the passed qualification receipt
- **THEN** runtime fails before engine construction

#### Scenario: Qualified composed derivative
- **WHEN** a materialized model derives from the qualified source base,
  preserves its architecture/tokenizer/processor/template identities, and has
  a passed composition-fidelity receipt with dynamic-HF behavioral diagnostics
- **THEN** its distinct execution-model fingerprint is accepted without being
  mistaken for runtime qualification drift

#### Scenario: Unprobed concurrency value
- **WHEN** `max_num_seqs` differs from every value covered by a passed
  concurrency receipt
- **THEN** runtime fails before engine construction and names the qualified
  values

#### Scenario: Equivalent checkout at a different root
- **WHEN** the same qualified repository and package source bytes are available
  under different absolute installation paths
- **THEN** qualification succeeds through relative identities without editing
  the receipts

#### Scenario: OpenSpec change is archived
- **WHEN** the change that introduced vLLM is archived
- **THEN** runtime still resolves its accepted receipts from the stable
  inference-owned qualification directory

## MODIFIED Requirements

### Requirement: Parallelism metadata in resolved runtime evidence
Every data-parallel inference run SHALL record its resolved parallelism policy.
The evidence MUST include parent visible CUDA tokens, active rank count,
per-device decode batch size, worker binding policy, shard-plan fingerprint,
and whether the run used direct single-process or controller/worker execution.

#### Scenario: Parallel run metadata
- **WHEN** a multi-rank inference run completes
- **THEN** manifest or provenance evidence records active rank count,
  per-device batch size, and rank-to-device mapping

#### Scenario: Direct single-rank path
- **WHEN** only one active rank is needed and HF is selected
- **THEN** manifest or summary evidence MAY record that the direct
  single-process path was used
- **AND** non-dry direct execution still records CUDA-required runtime evidence
- **AND** the manifest records direct-runtime device evidence after model load,
  including logical device and model first-parameter device when available

#### Scenario: Fresh single-rank vLLM worker
- **WHEN** only one active rank is needed and vLLM is selected
- **THEN** manifest or summary evidence records controller/worker execution
- **AND** the vLLM engine runs in a fresh rank-local worker process
