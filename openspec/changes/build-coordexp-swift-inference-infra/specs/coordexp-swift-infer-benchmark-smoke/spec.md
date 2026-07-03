## ADDED Requirements

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
After tiny base-path trace validation passes, the system SHALL run a real adapter-enabled smoke.
The smoke MUST use base model, adapter checkpoint, and any configured
special-token embedding delta, then verify identity checks and scored artifact
production.

#### Scenario: Adapter checkpoint final
- **WHEN** smoke config points at `checkpoint-final`
- **THEN** runtime resolves adapter and delta payloads, validates identity, and
  records resolved identities

#### Scenario: Wrong adapter identity
- **WHEN** smoke injects an incompatible adapter identity
- **THEN** runtime fails before generation

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
