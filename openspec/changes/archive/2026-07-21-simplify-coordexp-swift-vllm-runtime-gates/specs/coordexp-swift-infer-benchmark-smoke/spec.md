## MODIFIED Requirements

### Requirement: Real vLLM qualification smoke

Before a vLLM run is treated as completed evidence, at least one real
single-image Qwen3-VL request MUST execute through the same materialized model,
no-resize image path, prompt expansion, generation, likelihood normalization,
stop handling, and backend cleanup used by the run. The first normal shard
decode MAY satisfy this requirement. The smoke MUST validate infrastructure
evidence, not detection quality. Mock-only tests MUST NOT satisfy the real
operational smoke.

#### Scenario: Current real tracer request

- **WHEN** the first real request runs through vLLM
- **THEN** it returns contract-valid prompt, image, token, stop, likelihood,
  parser-input, and cleanup evidence before the run completes

#### Scenario: Base-only tracer bullet

- **WHEN** the fixed real base-only smoke runs through the current vLLM engine
- **THEN** it completes inference and evaluation with current live runtime and
  cleanup evidence

### Requirement: Fresh-worker repeatability

Fresh-worker repeatability SHALL be optional evidence for an explicit
repeatability claim. A repeated run MUST record prompt, generated-token,
likelihood, parser, ordering, and non-timing artifact differences. Missing
repeatability evidence or a small backend-dependent token/likelihood difference
MUST NOT block an ordinary operational run whose current contracts pass.

#### Scenario: Repeat run changes one generated token

- **WHEN** two fresh-worker diagnostic runs differ in one generated token
- **THEN** the repeatability claim records a mismatch
- **AND** the mismatch does not retroactively invalidate either operational run

### Requirement: Materialized checkpoint composition fidelity

Base-plus-DoRA-plus-selected-token-delta execution MUST pass current structural
composition validation before vLLM loading. The materializer MUST validate
source payload identities, merge/fold success, selected-token compatibility,
target dtype, tied weights, standard snapshot structure, adapter-residue
absence, and cache integrity. Dynamic-HF or materialized-HF bitwise/logit/token
comparisons MAY be retained as optional diagnostics but MUST NOT be runtime
prerequisites.

#### Scenario: Delta omitted during materialization

- **WHEN** the configured selected-token delta is absent, incompatible, or
  cannot be folded exactly once into the tied weight
- **THEN** materialization fails before vLLM loading

### Requirement: Dual likelihood numeric gate

Cross-backend policy/raw likelihood comparison SHALL be an optional numerical
study. It MUST use aligned prompt and generated-token evidence and MUST report
its thresholds and observed deltas. Historical tolerances or exact HF/vLLM
token equality MUST NOT gate ordinary vLLM inference. Each backend's own live
likelihood evidence still MUST be finite, non-positive, and token aligned.

#### Scenario: Small cross-backend likelihood difference

- **WHEN** live HF and vLLM traces differ slightly while each trace satisfies
  its own alignment and finiteness contracts
- **THEN** the comparison reports the difference without blocking vLLM output

#### Scenario: Numerically close but token shifted

- **WHEN** a parity study's likelihood values meet declared tolerances but any
  compared token id or conditioning position differs
- **THEN** the parity claim fails while independently valid backend artifacts
  retain their backend-labeled evidence scope

### Requirement: Real outer data-parallel smokes

Two-GPU, eight-GPU, failure-injection, and concurrency smokes SHALL gate only
the corresponding scale or cleanup claim. Ordinary vLLM inference MUST NOT
require a historical scale receipt for its current positive per-device batch
size. Every actual multi-rank run still MUST verify one logical GPU per engine,
exact row coverage/order, consistent execution-model identity, strict trace
merge, terminal cleanup, and evaluator consumption.

Strict merge MUST compare semantic runtime settings independently from
rank-local live-decode and cleanup observations, then retain one completed
observation per rank in merged provenance.

#### Scenario: Current worker failure

- **WHEN** any engine worker in the current run fails or leaves invalid shard
  evidence
- **THEN** the controller preserves diagnostics and does not publish completed
  top-level scored artifacts

#### Scenario: No historical eight-GPU receipt

- **WHEN** a current run has no historical eight-GPU receipt
- **THEN** current workers may launch and prove their own row, identity, merge,
  and cleanup contracts

#### Scenario: Worker failure

- **WHEN** any engine worker in the current run fails or leaves invalid shard
  evidence
- **THEN** the controller preserves diagnostics and does not publish completed
  top-level scored artifacts

#### Scenario: Required failure-injection matrix

- **WHEN** a distributed failure-handling acceptance claim is made
- **THEN** its named startup, OOM, timeout, termination, orphan, malformed
  shard, identity, and likelihood cases require executed terminal evidence
- **AND** absence of that matrix does not prevent a separate ordinary run from
  proving its current success contracts

#### Scenario: Eight visible GPUs but seven active ranks

- **WHEN** an alleged eight-GPU scale smoke activates fewer than eight nonempty
  ranks
- **THEN** the eight-GPU scale claim fails without becoming a global runtime
  authorization gate

### Requirement: Matched val200 acceptance

Matched HF/vLLM val200 studies SHALL preserve exact input, GT, checkpoint,
prompt, generation-policy, and evaluator identity and MUST report metric,
parse, drop, invalid, stop, truncation, throughput, and memory differences.
Metric or token parity thresholds qualify only the stated interchangeability
claim. Missing or failed parity evidence MUST NOT block ordinary operational
vLLM inference or forbid using its evaluator results as backend-labeled
research evidence.

#### Scenario: Metric drift

- **WHEN** HF and vLLM val200 metrics differ beyond a declared parity threshold
- **THEN** the run cannot claim that threshold's HF interchangeability
- **AND** its backend-labeled vLLM artifacts remain valid if their current
  execution and evaluator contracts pass

#### Scenario: BF16 throughput run meets the metric threshold

- **WHEN** a vLLM BF16 run happens to satisfy a declared val200 parity threshold
- **THEN** it remains BF16 backend-labeled evidence unless the study separately
  satisfies every declared HF-interchangeability condition
