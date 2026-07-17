## ADDED Requirements

### Requirement: Backend-owned decode session
Each backend SHALL own native model or engine loading, native multimodal input
projection, per-device batching or scheduling, generation, optional raw replay,
output normalization, effective runtime receipts, and cleanup. The shared
pipeline MUST NOT receive HF tensors, vLLM request/output objects, or loaded
backend models.

#### Scenario: HF session
- **WHEN** HF inference executes
- **THEN** the HF session privately materializes Qwen tensors and returns only
  backend-neutral decode results and a session receipt

#### Scenario: vLLM session cleanup
- **WHEN** a vLLM shard completes or raises
- **THEN** its engine is closed and no live child process remains owned by the
  shard worker

### Requirement: Dual generated-token likelihood semantics
Every non-pad generated token MUST have a finite non-positive
`policy_logprob`, defined as FP32 normalized likelihood after every active
decode processor. When raw tracing is enabled, every non-pad generated token
MUST also have a finite non-positive `raw_model_logprob`, defined as FP32
normalized likelihood from unmodified LM-head logits before decode processors.
The two channels MUST align to the same request id, generated step, token id,
token text, stop status, and conditioning prefix.

#### Scenario: Repetition penalty distinguishes channels
- **WHEN** a repeated-token prefix is decoded with repetition penalty 1.10 and
  raw tracing enabled
- **THEN** policy and raw likelihoods are independently recorded and are not
  substituted for one another

#### Scenario: Replay alignment mismatch
- **WHEN** replay token ids or lengths differ from generated token evidence
- **THEN** inference fails before scored artifacts are published

### Requirement: HF raw and policy likelihood implementation
HF policy likelihood MUST be gathered from normalized generation scores. HF raw
likelihood MUST be gathered from raw generation logits with FP32 log-softmax.
Qualification MUST compare the raw generation result with an independent
teacher-forced forward reference.

#### Scenario: Raw logits unavailable
- **WHEN** raw likelihood is requested but HF generation returns no aligned raw
  logits
- **THEN** the HF session fails instead of reusing policy scores

### Requirement: vLLM policy and raw likelihood implementation
vLLM generation MUST request processed logprobs and MUST extract the chosen
token's likelihood at each generated step. Raw likelihood MUST be computed by
teacher-forced replay over the exact prompt plus generated token ids using
prompt-logprob evidence. Any replay-only generated token MUST be discarded.

#### Scenario: Chosen token missing from vLLM logprobs
- **WHEN** vLLM output does not include the chosen generated token likelihood
- **THEN** inference fails with request id and generated step diagnostics

#### Scenario: Raw replay disabled
- **WHEN** raw tracing is false
- **THEN** no replay request is issued and raw likelihood is recorded as absent

### Requirement: Executable offline vLLM backend
The schema and decode records SHALL support offline vLLM execution through the
same backend-neutral result contract as HF. The rank-local worker MUST invoke
the offline API with one visible GPU; the qualified engine process mode MAY use
engine-owned child processes, but those children remain owned by and bounded
to that worker. Raw vLLM objects MUST remain inside the backend module. Backend
name, mode, response family, version, effective engine settings,
execution-model identity, process mode, and likelihood semantics MUST be
recorded from actual backend receipts.

#### Scenario: Executable vLLM config
- **WHEN** a valid qualified vLLM config and supported execution model are used
- **THEN** the backend opens an offline vLLM engine and returns validated
  backend-neutral decode results

#### Scenario: Response family recorded
- **WHEN** HF or vLLM materializes a decode result
- **THEN** manifest and trace evidence identify the actual backend response
  family and mode

## REMOVED Requirements

### Requirement: Future vLLM reservation

**Reason**: This change replaces the reserved, fail-only vLLM surface with the
qualified executable offline backend requirement above.

**Migration**: Configs continue to select `backend.type`, but `vllm` now
requires its strict backend block and a matching passed qualification receipt.
