# coordexp-swift-infer-scoring-artifacts Specification

## Purpose
TBD - created by archiving change build-coordexp-swift-inference-infra. Update Purpose after archive.
## Requirements
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
config fingerprint, full resolved generation policy, model identity
fingerprint, processor identity fingerprint, template identity, parser policy,
score policy fingerprint, and row-count or row-identity binding evidence.

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
config fingerprint, resolved generation policy, score policy fingerprint,
trace/scoring status, prompt/template identity, processor identity, and
evaluator-consumer status. `summary.json` SHALL record parse failure counts,
dropped prediction counts, truncation/length-stop counts, and decode stop
reason counts.

#### Scenario: Manifest after successful run
- **WHEN** inference completes successfully
- **THEN** `run_manifest.json` links every required artifact and records
  trace/scoring status and the resolved generation policy

#### Scenario: Decode truncation occurs
- **WHEN** any row stops because `max_new_tokens` was exhausted
- **THEN** summary artifacts MUST count the length-stop/truncated rows
- **AND** the scored artifact family MUST preserve parse-drop diagnostics.

#### Scenario: Partial artifact materialization
- **WHEN** required scored artifacts are missing after a scored run
- **THEN** manifest validation fails or records the run as not benchmark
  eligible

### Requirement: Strict shard merge validation
The data-parallel controller SHALL strictly validate every shard before creating benchmark-looking top-level scored artifacts.
Merge validation MUST reject failed workers, missing required shard artifacts,
missing input rows, duplicate row ids, row-order mismatches, shard-plan
fingerprint mismatches, and identity fingerprint mismatches across shards. The
merge identity vector MUST include base model identity, adapter identity/status
and active adapter when present, embedding-delta identity/load status when
present, tokenizer identity, processor identity, prompt/template/object-ordering
policy, dataset identity, generation/decode policy, parser policy, score
policy, raw/scored artifact SHA binding, row-count/row-identity binding, and
row-local `pred_score_source` / `pred_score_version` preservation.

#### Scenario: Missing row
- **WHEN** no shard artifact contains an input row id
- **THEN** merge fails before writing top-level scored artifacts

#### Scenario: Duplicate row
- **WHEN** two shard artifacts contain the same input row id
- **THEN** merge fails before writing top-level scored artifacts

#### Scenario: Identity mismatch
- **WHEN** two shards disagree on model identity, tokenizer identity, processor
  identity, adapter identity, embedding-delta identity, template identity,
  generation policy, parser policy, score policy, row identity binding, or
  row-local score provenance
- **THEN** merge fails before writing top-level scored artifacts

### Requirement: Canonical merged artifact contract
Successful data-parallel inference SHALL produce the same canonical top-level artifact family as single-process inference.
Merged `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, and `image_plan.jsonl`
MUST contain exactly one row per input row in original row order. Merged token
trace and diagnostics sidecars MUST preserve rank/world/device fields, but
evaluator input rows MUST remain schema-compatible with the existing
scored-artifact consumer.

#### Scenario: Row order restored
- **WHEN** rank-local shards complete in any order
- **THEN** merged raw and scored artifacts are ordered by original `row_index`

#### Scenario: Evaluator compatibility
- **WHEN** the merged scored artifact is passed to the CoordExp-swift detection
  evaluator
- **THEN** no shard-specific row schema adaptation is required

#### Scenario: Rank-aware sidecars
- **WHEN** merged token trace and parse diagnostic sidecars are written
- **THEN** every sidecar row that came from a worker preserves rank, world size,
  assigned parent-visible token, and logical device fields

### Requirement: Parallelism provenance
Merged scored provenance artifacts SHALL bind the top-level artifact set to the rank-local shard evidence.
The merged `gt_vs_pred_scored.jsonl.provenance.json` MUST be regenerated after
merge and MUST include the baseline evaluator-required fields plus a
parallelism/shards section. Required parallelism evidence includes shard-plan
fingerprint, active rank count, visible CUDA tokens, rank-to-device mapping,
per-device batch size, shard artifact hashes, merged artifact hashes, row
coverage summary, worker exit statuses, and merge status. `run_manifest.json`
MAY duplicate or summarize this evidence, but it MUST NOT substitute for the
scored provenance sidecar.

#### Scenario: Complete merge provenance
- **WHEN** data-parallel inference completes successfully
- **THEN** merged scored provenance can identify every shard that
  contributed to each merged artifact
- **AND** it records the resolved parallelism policy

#### Scenario: Worker failure
- **WHEN** any worker exits unsuccessfully
- **THEN** the controller preserves available shard evidence
- **AND** writes terminal status evidence without claiming scored benchmark
  eligibility

### Requirement: Merged trace replay completeness
Merged `pred_token_trace.jsonl` SHALL remain sufficient to recompute every scored prediction.
Strict merge MUST preserve every generated-token row and selected-token replay
row needed by merged `gt_vs_pred_scored.jsonl`. It MUST enforce uniqueness for
generated-token keys such as `(row_id, generated_step_index)` and replay keys
such as `(row_id, object_span_id)`. It MUST recompute every merged
`pred[*].score` from merged trace evidence within numeric tolerance before the
artifact set is accepted for evaluator consumption.

#### Scenario: Score recomputation after merge
- **WHEN** two shard artifacts each contain one scored prediction
- **THEN** the merged token trace evidence recomputes both stored prediction
  scores within tolerance

#### Scenario: Missing replay evidence
- **WHEN** a scored prediction lacks its selected-token replay row after merge
- **THEN** strict merge fails before publishing top-level scored artifacts

#### Scenario: Duplicate conflicting token trace
- **WHEN** merged token trace contains duplicate conflicting generated-token
  evidence for a row and generated step
- **THEN** strict merge fails before publishing top-level scored artifacts

### Requirement: Staged top-level publication
Data-parallel merge SHALL publish top-level artifacts only after complete staged validation.
The controller MUST stage merged raw, scored, provenance, trace, diagnostics,
image plan, summary, and manifest artifacts, validate the complete top-level
family including evaluator sidecar and trace replay contracts, and then publish
the complete set. On merge failure, root output MUST contain only terminal
status evidence with `benchmark_eligible: false`; shard directories remain
available for diagnosis. Merge failure cleanup MUST also remove known
metric-bearing evaluator outputs under the same run root, including
`eval_detection/metrics.json` and its COCO materialization files, so stale
metrics cannot survive beside a failed terminal manifest. Missing, malformed,
unreadable, or semantically malformed rank-local JSON/JSONL artifacts MUST be
represented as merge contract failures rather than raw filesystem, JSON parser,
or type-cast exceptions.

#### Scenario: Merge failure after staging starts
- **WHEN** merge validation fails after staged files have been created
- **THEN** top-level scored/provenance/token-trace/image-plan artifacts are not
  published as benchmark-looking outputs
- **AND** shard directories remain available
- **AND** stale evaluator metrics under the run root are absent or explicitly
  replaced by non-metric terminal status evidence

#### Scenario: Malformed rank-local JSON or JSONL semantics
- **WHEN** a worker exits successfully but a required rank-local JSON artifact
  is malformed or a rank-local JSONL row has invalid typed fields such as
  non-integer `row_index`, non-integer `generated_step_index`, missing
  `object_span_id`, invalid `score`, or malformed selected-token replay fields
- **THEN** strict merge fails with terminal root status evidence
- **AND** no benchmark-looking top-level artifacts are published
