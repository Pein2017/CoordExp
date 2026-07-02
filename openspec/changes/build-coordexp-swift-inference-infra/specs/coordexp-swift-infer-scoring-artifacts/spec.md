## ADDED Requirements

### Requirement: Selected-token score formula
For V1 scored inference, `pred[*].score` SHALL equal `exp(sum(selected_token_logprobs) / n_selected)`.
Selected token logprobs MUST be finite natural-log probabilities. Empty
selected-token sets or non-finite selected logprobs MUST invalidate the affected
prediction for scored output.

#### Scenario: Deterministic score
- **WHEN** selected token logprobs are `[log(0.2), log(0.2), log(0.2)]`
- **THEN** the prediction score equals `0.2` within numeric tolerance

#### Scenario: Empty selected set
- **WHEN** a parsed prediction has no selected token evidence
- **THEN** that prediction is excluded from the scored `pred` list and recorded
  diagnostically

#### Scenario: Non-finite logprob
- **WHEN** any selected token logprob is NaN or Inf
- **THEN** that prediction is excluded from the scored `pred` list and recorded
  diagnostically

### Requirement: Token selection policy
The default score token set SHALL include schema wrapper tokens and coordinate tokens for the parsed object span.
The exact selected wrapper tokens are `<|object_ref_start|>`,
`<|object_ref_end|>`, `<|box_start|>`, and `<|box_end|>`. The exact selected
coordinate tokens are the four coordinate tokens in the parsed box span. A
valid V1 compact object therefore MUST have `n_selected == 8`. The policy SHALL
exclude free-text description and category text. The scoring artifact MUST
persist per-object replay evidence: row id, object span id, generated-step
indices, token ids, token text, selected logprobs, selected count, and
score-policy fingerprint.

#### Scenario: Wrapper and coordinate tokens selected
- **WHEN** a parsed object span includes schema wrappers and four coordinate
  tokens
- **THEN** scoring selects exactly four wrapper token trace indices and four
  coordinate token trace indices for that object

#### Scenario: Description excluded
- **WHEN** the object span includes description/category words
- **THEN** those free-text tokens do not contribute to `pred[*].score`

#### Scenario: Selected-token count mismatch
- **WHEN** a compact object has fewer or more than four selected wrappers plus
  four selected coordinate tokens
- **THEN** the affected prediction is excluded from scored `pred` and recorded
  diagnostically

#### Scenario: Duplicate span ambiguity
- **WHEN** two objects have repeated schema and coordinate token subsequences
  such that alignment is ambiguous
- **THEN** the affected prediction is invalidated or recorded with explicit
  ambiguity policy rather than scored from an arbitrary span

#### Scenario: Object span trace mapping
- **WHEN** the parser emits a prediction object span
- **THEN** the span maps to one contiguous generated-token interval, and
  selected tokens are order-preserving within that interval even when
  description/category text creates gaps between selected tokens

### Requirement: Raw artifact row contract
`gt_vs_pred.jsonl` SHALL preserve one diagnostic-capable row per input row.
Each row includes raw decode text, GT objects, image identity, image dimensions,
salvaged prediction objects, parser status, and inline diagnostics. It MUST NOT
include extra diagnostic rows beyond the one-row-per-input contract; additional
diagnostics belong in sidecar artifacts.

#### Scenario: Malformed output row
- **WHEN** generated text is malformed
- **THEN** `gt_vs_pred.jsonl` still contains the row with raw decode text,
  parser diagnostics, GT, and image metadata

#### Scenario: Partial salvage
- **WHEN** one generated object is valid and another is malformed
- **THEN** the raw artifact preserves the valid prediction and records the
  malformed span diagnostics

### Requirement: Scored artifact row contract
`gt_vs_pred_scored.jsonl` SHALL preserve exactly one scored row per raw row.
The row keeps identical image identity, image dimensions, GT payload, row order,
and record index. Its `pred` list SHALL contain only predictions with finite
comparable scores. Rows with no scoreable predictions SHALL remain present with
`pred: []`.

#### Scenario: Trace-missing row
- **WHEN** a raw row has predictions but no valid trace alignment
- **THEN** the scored row remains present with the same GT and image metadata
  and an empty `pred` list

#### Scenario: Row-count parity
- **WHEN** raw and scored artifacts are compared after a run
- **THEN** they have the same row count and row identity order

### Requirement: Scored artifact provenance
Score-bearing artifacts SHALL use evaluator-readable provenance.
Each scored prediction MUST include a non-empty `pred_score_source`, integer
`pred_score_version`, and finite prediction score in `[0.0, 1.0]`. Rows with no
scoreable predictions MUST remain present as `pred: []` and MUST NOT invent
row-level score provenance. The run MUST write
`gt_vs_pred_scored.jsonl.provenance.json` with artifact schema version, source
raw artifact SHA256 identity, scored artifact SHA256 identity when available,
detection template id, prompt policy fingerprint, decode policy or generation
config fingerprint, model identity fingerprint, processor identity fingerprint,
template identity, parser policy, score policy fingerprint, and row-count or
row-identity binding evidence.

#### Scenario: Complete provenance
- **WHEN** scored output is written
- **THEN** every score-bearing prediction has score source/version metadata and
  scored sidecar provenance is present

#### Scenario: Invalid score value
- **WHEN** a scored prediction has a non-finite score or a score outside
  `[0.0, 1.0]`
- **THEN** evaluator consumption fails before metric computation

#### Scenario: Missing provenance
- **WHEN** scored output lacks row-local score source/version or sidecar
  provenance
- **THEN** official evaluator consumption fails before metric computation

#### Scenario: Moved artifact
- **WHEN** raw and scored artifact files are moved together
- **THEN** provenance remains portable because it binds to raw SHA identity, not
  only an absolute path

### Requirement: Token trace artifact
When scoring is enabled, `pred_token_trace.jsonl` SHALL record generated token trace evidence sufficient to recompute each scored prediction.
Score-bearing artifacts MUST be derivable from recorded trace evidence, not from
an unrecorded in-memory calculation.

#### Scenario: Score recomputation
- **WHEN** a scored prediction has selected-token indices
- **THEN** those indices resolve into `pred_token_trace.jsonl` rows whose
  token ids, token text, and logprobs recompute the stored score

#### Scenario: Trace artifact absent
- **WHEN** scoring is enabled but `pred_token_trace.jsonl` is absent
- **THEN** the artifact set is invalid for benchmark mAP claims

### Requirement: Artifact manifest
`run_manifest.json` SHALL record inference artifact paths and identity fields.
Required fields include resolved config fingerprints, model and adapter
identity, backend, backend mode, response family, dataset identity, generation
config, score policy fingerprint, trace/scoring status, prompt/template
identity, processor identity, and evaluator-consumer status.

#### Scenario: Manifest after successful run
- **WHEN** inference completes successfully
- **THEN** `run_manifest.json` links every required artifact and records
  trace/scoring status

#### Scenario: Partial artifact materialization
- **WHEN** required scored artifacts are missing after a scored run
- **THEN** manifest validation fails or records the run as not benchmark
  eligible
