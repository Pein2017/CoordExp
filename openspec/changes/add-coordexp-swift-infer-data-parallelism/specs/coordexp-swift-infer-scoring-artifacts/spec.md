## ADDED Requirements

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
