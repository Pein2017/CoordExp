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
Backend support MUST include a real single-GPU Qwen3-VL smoke with the qualified
vLLM version. The smoke MUST verify one image, no resize, exact prompt token
ids, special-token-preserving generated ids/text, `<|im_end|>` retention,
policy logprobs, parser output, selected-token score replay, artifacts, and the
unchanged evaluator. Mock-only tests MUST NOT satisfy this gate.

#### Scenario: Base-only tracer bullet
- **WHEN** the fixed real base-only smoke runs through vLLM
- **THEN** it completes inference and evaluation with all required receipts

### Requirement: Fresh-worker repeatability
The fixed base-only tracer fixture MUST run at least twice in separate fresh
workers. Prompt ids, generated ids, semantic stop reason, parser output, scored
rows, row ordering, and non-timing artifact content MUST match exactly. Policy
likelihood values MUST satisfy the dual-likelihood numeric tolerances rather
than requiring bitwise floating-point identity. The initially supported maximum
`max_num_seqs` MUST also have a real concurrent-request receipt before use in a
production config.

#### Scenario: Repeat run changes one generated token
- **WHEN** two fresh-worker runs differ in any generated token id
- **THEN** repeatability qualification fails even if evaluator metrics match

### Requirement: Materialized checkpoint composition fidelity
The exact base-plus-DoRA-plus-selected-token-delta composition MUST pass state
composition fidelity before vLLM evaluation. Reloaded merged DoRA target
weights, tied weights, prompt ids, selected-token rows, and source fingerprints
MUST be verified exactly. Dynamic-HF versus materialized-HF logits and greedy
ids MUST be retained as non-blocking BF16 behavioral diagnostics. Materialized
HF MUST be the executable-model oracle for vLLM, while canonical dynamic HF
MUST remain supported and supply the matched-val200 behavioral baseline.

#### Scenario: Delta omitted during materialization
- **WHEN** selected-token logits reveal a missing or duplicated delta
- **THEN** adapter-enabled vLLM execution-model acceptance fails

### Requirement: Dual likelihood numeric gate
A fixed repeated-token fixture with repetition penalty 1.10 MUST compare HF and
vLLM policy and raw likelihood channels, using materialized HF as the oracle
for a composed checkpoint. Prompt and generated token ids MUST match exactly.
Across aligned tokens, median absolute likelihood difference
MUST be at most `0.002`, P99 at most `0.02`, maximum at most `0.05`, and
per-object absolute log-score difference at most `0.01`.

#### Scenario: Numerically close but token shifted
- **WHEN** likelihood values meet numeric tolerances but any token id or
  conditioning position differs
- **THEN** the gate fails

### Requirement: Real outer data-parallel smokes
The backend MUST pass two-GPU and eight-GPU production-like smokes using the
existing outer sharding controller. Smokes MUST verify one logical GPU per
engine, exact row coverage/order, identical execution-model fingerprints,
strict trace merge, terminal cleanup, and evaluator consumption.

The eight-GPU gate MUST have exactly eight visible CUDA tokens, exactly eight
active nonempty ranks, at least eight decode blocks, one validated engine
session receipt per rank, and a successful merged evaluator receipt. Merely
making eight GPUs visible while activating fewer ranks does not satisfy it.

#### Scenario: Worker failure
- **WHEN** any engine worker fails or leaves invalid shard evidence
- **THEN** the controller preserves diagnostics and does not publish canonical
  top-level scored artifacts

#### Scenario: Required failure-injection matrix
- **WHEN** distributed vLLM backend acceptance is evaluated
- **THEN** engine-startup failure, CUDA OOM, worker timeout, forced process-tree
  termination, orphan process, malformed shard, backend-identity mismatch, and
  likelihood mismatch each have an executed terminal receipt proving cleanup
  and no canonical top-level publication

#### Scenario: Eight visible GPUs but seven active ranks
- **WHEN** an alleged eight-GPU smoke activates fewer than eight nonempty ranks
- **THEN** the eight-GPU acceptance gate fails

### Requirement: Matched val200 acceptance
The same checkpoint, dataset, prompt, generation policy, and evaluator MUST be
run through dynamic HF and vLLM FP32 on the fixed val200 scope. Input and GT
identity MUST match exactly. Absolute bbox mAP and mRecall differences MUST
each be no greater than `0.005`. Parse, drop, invalid, stop, truncation,
throughput, and peak-memory counters MUST be reported even when the metric gate
passes. A BF16 vLLM run MAY be used as supported throughput evidence but MUST
NOT satisfy this strict parity gate or claim HF interchangeability.

#### Scenario: Metric drift
- **WHEN** either absolute mAP or mRecall difference exceeds `0.005`
- **THEN** vLLM remains experimental and MUST NOT replace HF benchmark evidence

#### Scenario: BF16 throughput run meets the metric threshold
- **WHEN** a vLLM BF16 run happens to satisfy the val200 metric threshold
- **THEN** it remains throughput-mode evidence and strict parity still requires
  the matched FP32 dynamic-HF/vLLM gate

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
