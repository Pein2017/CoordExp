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

### Requirement: Future vLLM reservation
The V1 schema and decode records SHALL reserve backend-neutral fields for future vLLM support.
Reserved fields include `backend`, `backend_mode`, and `response_family`. V1
MUST validate vLLM execution as not implemented.

#### Scenario: Reserved vLLM config
- **WHEN** an inference config selects `backend.type: vllm`
- **THEN** validation fails with a not-implemented contract error rather than
  attempting to run vLLM

#### Scenario: Response family recorded
- **WHEN** an HF decode result is materialized
- **THEN** the manifest and trace identify its response family as HF

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
