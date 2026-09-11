## MODIFIED Requirements

### Requirement: Explicit dual-backend precision roles

Dynamic HF SHALL remain a first-class backend for direct base, DoRA, and
selected-token embedding-delta composition. vLLM SHALL remain a first-class
offline backend over a structurally validated execution model. FP32 MAY be used
for narrow cross-backend numerical diagnosis, while BF16 vLLM MAY be used for
normal high-throughput inference. Backend-dependent numerical or token
differences MUST be reported when a parity study is claimed, but MUST NOT block
ordinary inference when current composition and backend-neutral output
contracts pass.

#### Scenario: Dynamic HF composed checkpoint

- **WHEN** `backend.type: hf` selects a base plus DoRA plus embedding delta
- **THEN** runtime loads that composition directly without requiring a vLLM
  execution-model snapshot

#### Scenario: BF16 vLLM operational run

- **WHEN** `backend.type: vllm` and `model.dtype: bf16` are selected
- **THEN** inference may produce operational or benchmark evidence after the
  current live runtime contracts pass
- **AND** it MUST NOT claim exact HF parity without a separate comparison

#### Scenario: BF16 vLLM throughput run

- **WHEN** `backend.type: vllm` and `model.dtype: bf16` are selected
- **THEN** inference remains executable after current runtime contracts pass
- **AND** the run cannot claim strict FP32 cross-backend parity without a
  separate matched study

### Requirement: Qualified vLLM version

The runtime MUST record the installed vLLM version, model dtype, effective
engine settings, execution-model identity, CUDA binding, process mode, and
cleanup result. Version `0.14.1` SHALL remain the documented known-working
version, but a version string, application-source hash, historical source
manifest, exact historical `gpu_memory_utilization`, exact historical
`max_model_len`, or previously probed concurrency value MUST NOT by itself
prevent engine construction.

The current run MUST fail when vLLM cannot construct the requested engine,
load the execution model, satisfy backend-neutral prompt/image/stop/likelihood
contracts, or clean up its owned runtime. Historical qualification receipts
MAY be inspected and recorded as matching, stale, missing, or unverified
diagnostics, but MUST NOT authorize or reject ordinary inference.

An unverified vLLM version MAY attempt policy-only inference. Raw-model
likelihood tracing MUST fail for an unverified version unless a version-specific
probe has established that raw logprobs are captured before the forcing
processor. Aligned finite replay tokens alone MUST NOT assert that ordering.

#### Scenario: Application source changed after a successful probe

- **WHEN** a CoordExp-owned source file differs from a historical vLLM
  qualification manifest
- **THEN** runtime records the source evidence as stale or unverified
- **AND** proceeds to current engine construction and live decode validation

#### Scenario: Operational engine setting differs from history

- **WHEN** finite `gpu_memory_utilization` in `(0, 1]` or positive per-device
  concurrency differs from a historical probe
- **THEN** runtime records the effective current value and attempts current
  engine construction without requiring a new receipt

#### Scenario: Installed vLLM API is incompatible

- **WHEN** the installed vLLM version cannot construct the requested engine or
  return contract-valid Qwen3-VL output
- **THEN** the run fails with the concrete current engine or decode error and
  MUST NOT publish completed top-level inference artifacts

#### Scenario: Unqualified installed version

- **WHEN** the installed vLLM version has no accepted raw-logprob ordering probe
- **THEN** policy-only inference may attempt the current engine contracts
- **AND** raw-model likelihood tracing fails before engine construction

#### Scenario: Version string matches but receipt does not

- **WHEN** vLLM reports a historically known version but historical source or
  engine receipt identity no longer matches
- **THEN** runtime records that receipt as stale or unverified and relies on
  current live contracts for policy-only inference

#### Scenario: Qualified composed derivative

- **WHEN** a materialized model derives from the configured source base and its
  current structural materialization receipt passes
- **THEN** its distinct execution-model fingerprint is accepted without being
  mistaken for vLLM runtime drift

#### Scenario: Unprobed concurrency value

- **WHEN** a positive `max_num_seqs` differs from every historical concurrency
  probe
- **THEN** runtime records the effective value and attempts current engine
  construction instead of requiring a new historical receipt

#### Scenario: Equivalent checkout at a different root

- **WHEN** equivalent repository and package bytes are available under a
  different absolute installation root
- **THEN** historical path differences do not authorize or reject the current
  engine attempt

#### Scenario: OpenSpec change is archived

- **WHEN** the change that introduced or simplified vLLM is archived
- **THEN** ordinary policy inference does not depend on active-change receipt
  paths
- **AND** raw tracing still requires its stable version-specific ordering probe

## ADDED Requirements

### Requirement: Resolved inference input path ownership

Every model, data, adapter, and embedding-delta path MUST resolve to an
absolute path before runtime owners receive it. A relative value MUST resolve
against the YAML file that authored that leaf. Missing inputs MUST fail before
model composition or generation with the field name, declared value,
declaring config path, and resolved absolute path. Runtime MUST NOT search
sibling worktrees, alternate output roots, checkpoint aliases, or similarly
named directories.

The boundary MUST validate `data.input_jsonl` as a file and model, adapter, and
embedding-delta roots as directories before CUDA discovery or JSONL loading.

#### Scenario: Inherited relative adapter path is absent

- **WHEN** an inherited adapter path resolves under the current worktree but
  the intended payload lives under another artifact root
- **THEN** setup fails at the input-path boundary and reports both the
  declaring config and resolved path
- **AND** does not search the other artifact root

#### Scenario: Explicit shared artifact path

- **WHEN** a config authors an absolute adapter and embedding-delta path under
  a shared output root
- **THEN** those exact paths are validated, recorded, and passed to the
  materializer unchanged
