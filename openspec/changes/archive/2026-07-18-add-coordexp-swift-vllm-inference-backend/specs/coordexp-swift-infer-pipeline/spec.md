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

### Requirement: Inference source ownership
The pipeline SHALL use the approved `src/inference/*` module ownership
boundaries. `runtime.py` SHALL own processor-only frontend setup and backend
session launch preparation. `backend.py` SHALL own semantic requests, results,
likelihood pairs, and the session protocol. `hf_backend.py` SHALL own dynamic
HF model composition and execution. `vllm_backend.py` SHALL own offline vLLM
engine execution. `execution_model.py` SHALL own immutable execution-model
materialization and composition-fidelity binding. `prompt.py`, `parsing.py`,
`scoring.py`, and `artifacts.py` SHALL own their corresponding semantic
contracts. `pipeline.py` MUST remain orchestration-only and MUST NOT import
`src.training.pipeline`, `TrainConfig`, `ResolvedTrainConfig`, or
`ResolvedStepSchedule`.

#### Scenario: No training pipeline import
- **WHEN** inference modules are imported
- **THEN** they do not import `src.training.pipeline`, `TrainConfig`,
  `ResolvedTrainConfig`, or `ResolvedStepSchedule`

#### Scenario: Runtime assembler
- **WHEN** runtime setup loads a base-plus-adapter checkpoint
- **THEN** it delegates Qwen, adapter, backend-session, and artifact identity
  mechanics to their owner modules rather than reimplementing them inline

#### Scenario: Dynamic HF runtime
- **WHEN** runtime setup loads a base-plus-adapter checkpoint for HF
- **THEN** `hf_backend.py` composes the base, DoRA adapter, and selected-token
  delta directly through their owner modules

#### Scenario: Materialized vLLM runtime
- **WHEN** runtime setup launches vLLM
- **THEN** `execution_model.py` resolves the validated immutable model before
  `vllm_backend.py` opens the rank-local engine

### Requirement: Batched generation
The pipeline SHALL interpret `generation.batch_size` as the per-device decode
concurrency and shard-block size. Production configs MUST use a value greater
than one. HF sessions MUST submit native batches up to that size. vLLM sessions
MUST set their sequence-concurrency ceiling to that size and MAY submit all
rank-local requests to continuous scheduling. The pipeline itself MUST NOT
construct backend-native batches.

#### Scenario: Production batch
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

#### Scenario: Score failure counted
- **WHEN** a salvaged prediction lacks a complete eight-token score span
- **THEN** diagnostics count that prediction exclusion while preserving other
  valid predictions in the row

#### Scenario: Backend trace failure counted terminally
- **WHEN** backend-generated likelihood evidence is non-finite or misaligned
- **THEN** terminal diagnostics identify the trace failure and the run does not
  publish canonical completed scored artifacts
