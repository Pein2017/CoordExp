## MODIFIED Requirements

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
