# coordexp-swift-infer-backend-trace Specification

## Purpose
TBD - created by archiving change build-coordexp-swift-inference-infra. Update Purpose after archive.
## Requirements
### Requirement: Backend-neutral decode records
The system SHALL define complete backend-neutral decode request, decode result, and token trace records.
These records MUST include backend name, backend mode, response family, prompt
token ids, generated token ids, generated token text, logprobs, stop reason,
model identity, tokenizer identity, and generation config fingerprint when
score-bearing output is requested.

#### Scenario: HF decode result
- **WHEN** the HF backend returns generated sequences and scores
- **THEN** the pipeline receives backend-neutral decode results without exposing
  raw HF output objects outside the backend module

#### Scenario: Missing trace field
- **WHEN** a scored decode result lacks token ids, token text, logprob, prompt
  token ids, stop reason, backend, model identity, tokenizer identity, or
  generation config fingerprint
- **THEN** scored inference fails before writing `gt_vs_pred_scored.jsonl`

### Requirement: HF scored generation
The HF backend SHALL request scored generation whenever trace scoring is enabled.
It MUST pass `return_dict_in_generate=True` and `output_scores=True`. The
backend MUST NOT use constant fallback scores for mAP-style evaluation.

#### Scenario: Scored generation enabled
- **WHEN** inference config enables trace scoring
- **THEN** the HF backend calls `model.generate` with scored output enabled

#### Scenario: Scores unavailable
- **WHEN** HF generation returns no per-step scores for a scored run
- **THEN** inference fails with a backend-trace contract error

### Requirement: HF token-score alignment
The backend SHALL pin the token-score alignment algorithm for batched decode.
It MUST define prompt padded width, generated-token indexing, stop-token
retention, post-stop padding exclusion, and normalized logprob gathering via
`compute_transition_scores(SEQUENCES, SCORES, normalize_logits=True)` or an equivalent
`log_softmax(scores[t])` gather.

#### Scenario: Variable prompt lengths in one batch
- **WHEN** two rows with different prompt lengths are decoded in the same batch
- **THEN** generated token ids, token text, and gathered logprobs align to each
  row's generated sequence after the configured prompt-width rule

#### Scenario: Shape mismatch
- **WHEN** sequence length, score-step count, prompt width, or generated token
  count disagree
- **THEN** the backend raises a backend-trace contract error before parsing

#### Scenario: Pad after stop
- **WHEN** a row stops before other rows in the batch
- **THEN** trace items after the row's stop token are marked pad or excluded
  according to the recorded policy and are not score candidates

### Requirement: Qwen stop-token policy
The default V1 decoding policy SHALL use deterministic greedy generation with
neutral decode knobs and the Qwen chat stop transition `<|im_end|>`. Code
defaults MUST keep `repetition_penalty=1.0`, `temperature=0.0`, `top_p=1.0`,
and `do_sample=False`. Non-neutral choices such as
`repetition_penalty=1.10` MUST come from explicit resolved config and MUST be
recorded in generation policy artifacts. The system MUST NOT add
`<|endoftext|>` as a default stop token.

#### Scenario: Default generation policy
- **WHEN** a production inference config omits sampling settings
- **THEN** generation resolves to deterministic greedy decoding with
  `<|im_end|>` stop handling and neutral repetition penalty

#### Scenario: Non-neutral repetition penalty configured
- **WHEN** a config explicitly sets `generation.repetition_penalty: 1.10`
- **THEN** the backend MUST pass that value to generation
- **AND** the summary, manifest, or provenance artifacts MUST record the
  resolved generation policy.

#### Scenario: End-of-text not default
- **WHEN** the generation config is resolved
- **THEN** `<|endoftext|>` is not included as a default stop token

### Requirement: Special-token-preserving raw trace
Raw generated-token trace text SHALL preserve special tokens.
The implementation MUST NOT use a post-processing path that silently applies
`skip_special_tokens=True` to the raw trace evidence.

#### Scenario: Generated stop token
- **WHEN** generated ids include `<|im_end|>`
- **THEN** raw trace output preserves the stop token and records stop reason
  `im_end`

#### Scenario: Parser stripped view
- **WHEN** parser-facing text strips terminal `<|im_end|>`
- **THEN** the strip policy is recorded and raw trace evidence remains
  unmodified

### Requirement: Rank-aware trace diagnostics
Data-parallel inference SHALL add rank and device identity to trace and diagnostic sidecars.
Rank/device fields MUST be diagnostic metadata only and MUST NOT change
selected-token scoring semantics, token alignment semantics, or evaluator row
schema. Rank-local and merged token trace and diagnostic sidecars MUST preserve
rank, world size, assigned parent-visible token, worker logical device, and
worker CUDA visibility evidence where available.

#### Scenario: Token trace rank evidence
- **WHEN** a token trace row is written by a data-parallel worker
- **THEN** it includes rank, world size, assigned device token, and logical
  device fields
- **AND** selected-token score recomputation remains based on token ids,
  generated-step indices, and logprobs

#### Scenario: Backend-neutral trace record
- **WHEN** a future backend writes trace sidecars through the same shard
  contract
- **THEN** rank/device diagnostic fields do not require exposing raw backend
  output objects outside the backend module

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
token's finite non-positive likelihood at every generated step. When raw
tracing is enabled, the policy engine MUST close before a fresh raw-logprob
engine replays the authoritative sequence. The replay MUST require exact
request id, prompt ids, generated ids, token count, stop semantics, and
per-token likelihood alignment before raw values attach to policy-owned
results.

Live replay validation SHALL be authoritative. Historical runtime,
concurrency, application-source, probe-source, or forced-processor receipts
MAY be reported as optional diagnostics but MUST NOT be required to start the
current policy engine or raw replay. Raw replay failure remains fatal when
`artifacts.include_raw_model_logprob` is true and MUST NOT silently substitute
policy likelihood.

Raw tracing on an unverified vLLM version MUST fail unless a version-specific
ordering probe establishes the raw-logprob capture point. Policy-only decode
MAY still proceed on that version through the ordinary live contracts.

#### Scenario: Historical forced-replay source drift

- **WHEN** the current replay processor differs from a historical source hash
  but live forced replay returns exactly aligned finite likelihood evidence
- **THEN** raw likelihood tracing succeeds and records the current processor
  identity

#### Scenario: Live raw replay misalignment

- **WHEN** raw replay differs in prompt id, generated token id, stop reason,
  length, or likelihood finiteness
- **THEN** inference fails before scored artifacts are published regardless of
  any historical passed receipt

#### Scenario: Raw tracing disabled

- **WHEN** raw tracing is false
- **THEN** no raw engine or historical forced-replay qualification is required

#### Scenario: Chosen token missing from vLLM logprobs

- **WHEN** vLLM output does not include the chosen generated token likelihood
- **THEN** inference fails with request id and generated step diagnostics

#### Scenario: Raw replay disabled

- **WHEN** raw tracing is false
- **THEN** no second engine or replay request is issued and raw likelihood is
  recorded as absent

#### Scenario: Forced raw replay differs from authoritative generation

- **WHEN** the raw engine returns a different request id, prompt id, generated
  token id, stop reason, continuation length, or likelihood alignment
- **THEN** inference fails before scored artifacts are published

### Requirement: Executable offline vLLM backend

The rank-local worker SHALL open one offline vLLM engine on its visible GPU and
return backend-neutral decode results. Raw vLLM objects MUST remain inside the
backend module. Actual backend name, mode, response family, installed version,
effective engine settings, execution-model identity, process mode, likelihood
semantics, and cleanup result MUST be recorded.

Engine creation, prompt projection, raw replay, and session close MUST remain
inside explicit cleanup scopes. Missing or failing owned-engine shutdown MUST
fail the shard. Rank-local process, CUDA, live-decode, and cleanup observations MUST be
aggregated separately from cross-rank semantic settings so strict merge accepts
different first request ids while still rejecting actual setting drift.

The first normal multimodal decode in each shard SHALL serve as the live
operational smoke. Empty generation, prompt/image mismatch, invalid stop
behavior, non-finite likelihood, trace misalignment, or unsupported native
output shape MUST fail the shard before completed top-level artifacts publish.
Object count and evaluator quality MUST NOT be part of this infrastructure
smoke.

#### Scenario: Current engine and first decode succeed

- **WHEN** a structurally valid execution model loads and the first real
  multimodal request returns contract-valid native evidence
- **THEN** the session records a passed live operational preflight and
  continues inference

#### Scenario: Current engine load fails

- **WHEN** vLLM cannot load the snapshot or allocate the requested engine
- **THEN** the shard fails with the current runtime error without consulting a
  historical receipt as a fallback authorization

#### Scenario: Executable vLLM config

- **WHEN** a structurally valid vLLM config and execution model are used with a
  compatible installed API
- **THEN** the backend opens an offline vLLM engine and returns validated
  backend-neutral decode results

#### Scenario: Response family recorded

- **WHEN** HF or vLLM materializes a decode result
- **THEN** manifest and trace evidence identify the actual backend response
  family and mode
