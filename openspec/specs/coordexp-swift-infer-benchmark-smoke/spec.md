# coordexp-swift-infer-benchmark-smoke Specification

## Purpose
TBD - created by archiving change build-coordexp-swift-inference-infra. Update Purpose after archive.
## Requirements
### Requirement: Real HF Qwen tiny smoke
The first inference smoke SHALL use a real HF/Qwen path rather than mocked backend acceptance evidence.
It MUST exercise prompt construction, no-resize image processing, generation,
trace extraction, parsing, scoring, and artifact writing on a tiny input.

#### Scenario: Tiny real smoke passes
- **WHEN** the tiny smoke runs on a valid single-image fixture
- **THEN** required inference artifacts are written
- **AND** if the real model output contains valid compact predictions,
  trace-derived scoring is verified from artifacts
- **AND** if no valid compact predictions are produced, the run is partial
  smoke evidence only and non-empty selected-token scoring remains open

#### Scenario: Mock-only evidence
- **WHEN** only mocked backend tests pass
- **THEN** the implementation is not accepted as inference-smoke complete

### Requirement: Batched trace smoke
The smoke suite SHALL include a real HF/Qwen batch with at least two rows and different prompt lengths.
It MUST verify generated token ids, token text, normalized transition logprobs,
stop-token handling, and parser spans align.

#### Scenario: Two-row batch
- **WHEN** two rows with different prompt lengths are decoded in one HF batch
- **THEN** token trace alignment is verified for both rows

#### Scenario: Stop-token trace
- **WHEN** one row reaches `<|im_end|>` before the other row
- **THEN** post-stop handling is verified and no pad token is used for scoring
- **AND** when no row naturally reaches `<|im_end|>` in the tiny real smoke,
  the run is partial smoke evidence only and real stop/pad evidence remains open

### Requirement: Adapter-enabled smoke

After tiny base-path trace validation passes, the system SHALL run a real
adapter-enabled smoke. The smoke MUST configure the base model, explicit
adapter payload path, and any selected-token embedding-delta path, then verify
the exact identity checks promised by those loaders and scored artifact
production. It MUST NOT depend on checkpoint-final or handoff metadata path
resolution.

#### Scenario: Explicit adapter checkpoint payload

- **WHEN** smoke config points directly to an adapter directory and optional
  selected-token embedding-delta directory
- **THEN** runtime MUST load those payloads, validate their declared identity
  boundaries, and record the identities actually loaded.

#### Scenario: Adapter checkpoint final

- **WHEN** a new canonical smoke config depends on `checkpoint-final`,
  `checkpoint.json`, or `checkpoint_handoff.json` to resolve payload paths
- **THEN** config or smoke-contract validation MUST fail before generation.

#### Scenario: Wrong adapter identity

- **WHEN** smoke injects an adapter that violates the standard PEFT/DoRA
  identifier, model, target, tensor, or load-result contract
- **THEN** runtime MUST fail before generation.

### Requirement: Evaluator consumer compatibility
Before final benchmark claims, the system SHALL prove that `gt_vs_pred_scored.jsonl` can be consumed by the named mAP evaluator owner.
The V1 default owner is a minimal rebuilt `src.eval` detection consumer that
reads `gt_vs_pred_scored.jsonl` and its provenance sidecar directly. A bridge to
legacy/current evaluator tooling is allowed only after explicit user approval
and MUST NOT be treated as the default V1 path.

#### Scenario: Tiny scored fixture consumed
- **WHEN** a tiny scored fixture with valid provenance is passed to the named
  evaluator consumer
- **THEN** the evaluator writes `metrics.json` or an explicitly named metric
  artifact without row schema adaptation

#### Scenario: Missing score provenance
- **WHEN** the scored fixture lacks score provenance
- **THEN** evaluator consumption fails before metric computation

#### Scenario: Provenance binding mismatch
- **WHEN** the scored artifact row count, raw artifact SHA, scored artifact
  identity, prompt policy, decode policy, model identity, processor identity,
  template identity, parser policy, or score policy disagrees with provenance
- **THEN** evaluator consumption fails before metric computation

### Requirement: Val200 validation acceptance
Final CoordExp-Swift V1 inference/eval readiness SHALL be accepted from the fixed val200 inference/eval run when it uses real HF/Qwen decoding, scored artifacts, valid score provenance, and mAP/mRecall evaluation from the named Swift evaluator consumer.
Tiny debug smokes are implementation gates only and MUST NOT be presented as
val200 validation evidence.
Full validation-dataset or full benchmark inference is optional and MUST NOT be
required for the V1 readiness claim unless a future user request explicitly asks
for that broader scope.

#### Scenario: Accepted val200 run
- **WHEN** the approved val200 inference config is run over the fixed 200-row
  validation subset
- **THEN** batched decode, scored artifacts, and mAP/mRecall output artifacts are
  produced and linked in the manifest or acceptance report
- **AND** the result MAY be used as the V1 local validation gate

#### Scenario: Tiny smoke run
- **WHEN** a tiny one-row or two-row inference run succeeds
- **THEN** it is labeled as smoke or implementation evidence and not val200
  validation evidence

#### Scenario: Optional full-dataset run
- **WHEN** a full validation-dataset or benchmark run is requested
- **THEN** it is labeled with its broader evidence scope
- **AND** it is not a prerequisite for accepting the V1 val200 gate

### Requirement: Production benchmark leaf
The OpenSpec or implementation approval packet SHALL name the exact validation or production benchmark handles before launching a non-smoke evaluation run.
The required handles are config leaf, dataset path, model path, adapter
checkpoint path when used, artifact root, and mAP command.

#### Scenario: Validation launch gate complete
- **WHEN** the fixed val200 validation run is ready to launch
- **THEN** the approval packet contains exact config, dataset, model, adapter,
  artifact root, and evaluator command handles

#### Scenario: Missing launch handle
- **WHEN** any required validation or production benchmark handle is missing
- **THEN** the launch is blocked pending user approval or correction

### Requirement: Real multi-GPU HF smoke
Data-parallel inference SHALL be validated with a real multi-GPU HF smoke before any production data-parallel benchmark claim.
The smoke MUST use at least two visible CUDA devices, `backend.type: hf`, a real
Qwen model path, and a real input subset. It MUST verify worker binding,
rank-local artifacts, strict merge, merged scored artifacts, and evaluator
consumption. Evaluator smoke evidence MUST include an evaluation receipt that
binds the consumed raw, scored, provenance, and run-manifest artifacts by
SHA-256.

#### Scenario: Two-GPU smoke passes
- **WHEN** a two-GPU data-parallel HF smoke runs on a valid tiny or val subset
- **THEN** each active rank writes shard artifacts
- **AND** the merged top-level scored artifact has complete row coverage
- **AND** the evaluator consumes the merged scored artifact
- **AND** the evaluator writes a receipt binding the consumed artifact hashes

#### Scenario: Mock-only evidence
- **WHEN** only mocked worker-launch or merge tests pass
- **THEN** the implementation is not accepted as production data-parallel
  inference evidence

### Requirement: Evidence scope labeling
Data-parallel inference evidence SHALL label its scope.
Tiny and val-subset smokes are implementation evidence only. Production
benchmark claims require an explicitly launched production-scope run and must
name the config, dataset, checkpoint/adapter, visible GPU set, artifact root,
merged artifact path, and evaluator output.

#### Scenario: Tiny smoke evidence
- **WHEN** a tiny two-GPU smoke succeeds
- **THEN** the acceptance note labels it as smoke evidence, not full benchmark
  evidence
- **AND** evaluator metrics produced for that smoke are not marked as
  production benchmark metrics when the inference manifest is not benchmark
  eligible

#### Scenario: Production claim
- **WHEN** a production data-parallel benchmark result is reported
- **THEN** the report names the config, dataset, checkpoint or adapter, visible
  GPU set, merged inference artifact root, and evaluator metrics path

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

### Requirement: Benchmark scope eligibility
Canonical inference artifacts MUST set `benchmark_eligible: true` only when
`debug.smoke` is false and the run contains at least 200 input rows. Smaller
runs remain implementation evidence even when their authored config is not
marked as smoke. The evaluator MUST NOT promote an ineligible run to a
benchmark metric.

#### Scenario: Tiny non-smoke run
- **WHEN** a completed run has fewer than 200 rows and `debug.smoke: false`
- **THEN** inference and evaluation artifacts remain benchmark-ineligible

#### Scenario: Accepted val200 scope
- **WHEN** a completed non-smoke run contains 200 rows and all artifact gates pass
- **THEN** the evaluator MAY publish `benchmark_metric: true`
