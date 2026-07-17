## ADDED Requirements

### Requirement: Backend launch preparation
The pipeline SHALL resolve shared request evidence before opening a backend and
SHALL obtain backend identity only from the opened session receipt. vLLM
controller mode MUST resolve and publish an execution-model receipt before
workers launch. The pipeline MUST NOT load an HF model on the vLLM path.

#### Scenario: vLLM launch
- **WHEN** backend type is vLLM
- **THEN** no HF inference model is loaded in the rank worker before the vLLM
  session opens

### Requirement: Exact backend result normalization
The pipeline MUST require exactly one decode result for every requested id,
reject missing, duplicate, or unknown results, and restore request order before
parsing. Backend scheduling order MUST NOT affect row order, parser input,
scoring alignment, or artifacts.

#### Scenario: vLLM returns out of order
- **WHEN** vLLM completes requests in a different order
- **THEN** results are restored by request id and final artifacts retain input
  row order

### Requirement: Backend trace integrity failure boundary
The pipeline SHALL distinguish parser-level salvage from backend evidence
integrity. Malformed or incomplete object text MAY exclude only the affected
prediction and MUST retain row diagnostics. Missing, duplicate, shifted,
positive, non-finite, or token-mismatched generated-token likelihood evidence
MUST terminate the scored run before canonical completed artifacts are
published.

#### Scenario: Malformed second object
- **WHEN** one generated object is valid and a second object is malformed while
  the backend trace is complete
- **THEN** parser salvage retains the valid object and records the malformed
  object drop

#### Scenario: Likelihood evidence shifts by one generated token
- **WHEN** backend likelihood evidence is shifted relative to generated ids
- **THEN** the scored run terminates rather than publishing an empty prediction
  for the affected row

## MODIFIED Requirements

### Requirement: Batched generation
The pipeline SHALL interpret `generation.batch_size` as the per-device decode
concurrency and shard-block size. Production configs MUST use a value greater
than one. HF sessions MUST submit native batches up to that size. vLLM sessions
MUST set their sequence-concurrency ceiling to that size and MAY submit all
rank-local requests to continuous scheduling. The pipeline itself MUST NOT
construct backend-native batches.

#### Scenario: HF production batch
- **WHEN** a production HF config sets `generation.batch_size: 8`
- **THEN** the HF session submits native decode batches up to size 8

#### Scenario: vLLM production concurrency
- **WHEN** a production vLLM config sets `generation.batch_size: 8`
- **THEN** the rank-local engine uses maximum sequence concurrency 8 without
  dividing it by rank count

#### Scenario: Debug batch size one
- **WHEN** a debug or smoke config sets `generation.batch_size: 1`
- **THEN** validation succeeds and the manifest records debug/smoke mode

#### Scenario: Production batch size one
- **WHEN** a production config sets `generation.batch_size: 1`
- **THEN** validation fails before runtime setup

### Requirement: Failure accounting
The pipeline SHALL record parser failures, parser-level score exclusions, image
validation failures, backend trace-integrity failures, engine failures, and
dropped predictions in terminal or successful diagnostics as appropriate. It
MUST NOT silently erase rows or predictions. Parser-level malformed-object
salvage MAY exclude only the affected prediction. Backend trace-integrity
failure MUST terminate the scored run and prevent a canonical completed
artifact set.

#### Scenario: Parser failure counted
- **WHEN** a row fails compact parser validation with otherwise valid backend
  evidence
- **THEN** summary counters include the parser failure and diagnostics identify
  the row

#### Scenario: Parser-level score exclusion counted
- **WHEN** a salvaged prediction lacks a complete eight-token score span
- **THEN** diagnostics count that prediction exclusion while preserving other
  valid predictions in the row

#### Scenario: Backend trace failure counted terminally
- **WHEN** backend-generated likelihood evidence is non-finite or misaligned
- **THEN** terminal diagnostics identify the trace failure and the run does not
  publish canonical completed scored artifacts
