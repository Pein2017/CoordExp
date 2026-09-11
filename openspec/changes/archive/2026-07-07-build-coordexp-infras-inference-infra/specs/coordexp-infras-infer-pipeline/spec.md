## ADDED Requirements

### Requirement: End-to-end inference pipeline
The inference pipeline SHALL orchestrate the complete offline decode-to-artifact flow.
It MUST cover dataset iteration, batched prompt construction, Qwen image
materialization, backend generation, parsing, scoring, artifact writing, and
summary counters. It MUST NOT run metric reduction itself.

#### Scenario: Successful scored inference run
- **WHEN** a valid scored inference config is executed
- **THEN** the pipeline writes resolved config, manifest, summary, raw artifact,
  scored artifact, provenance, token trace, parse diagnostics, and image plan

#### Scenario: Metric reduction separation
- **WHEN** inference completes
- **THEN** mAP computation is delegated to the named evaluator consumer or
  explicit bridge rather than executed inside `src.infer`

### Requirement: Batched generation
The pipeline SHALL batch independent decode requests according to `generation.batch_size`.
Production configs MUST use batch size greater than one. `batch_size: 1` SHALL
be treated as an explicit debug or smoke override.

#### Scenario: Production batch
- **WHEN** a production config sets `generation.batch_size: 8`
- **THEN** the pipeline submits decode requests in batches up to size 8

#### Scenario: Debug batch size one
- **WHEN** a debug or smoke config sets `generation.batch_size: 1`
- **THEN** validation succeeds and the manifest records debug/smoke batch mode

#### Scenario: Production batch size one
- **WHEN** a production config sets `generation.batch_size: 1`
- **THEN** validation fails before runtime setup

### Requirement: Failure accounting
The pipeline SHALL record all inference failure classes in counters and diagnostics.
Failure classes include parse failures, trace failures, score failures, image
validation failures, and dropped predictions. It MUST NOT silently erase rows or
predictions without counters.

#### Scenario: Parser failure counted
- **WHEN** a row fails compact parser validation
- **THEN** summary counters include the parser failure and diagnostics identify
  the row

#### Scenario: Score failure counted
- **WHEN** a prediction cannot be trace-scored
- **THEN** summary counters include the score failure and the scored artifact
  excludes only that prediction

### Requirement: Artifact write order
The pipeline SHALL write artifacts in a deterministic and recoverable order.
Resolved configs come before heavy setup, row artifacts during execution, and
manifest plus summary after completion or terminal failure. It MUST avoid
partial scored artifact claims when required trace/provenance artifacts are
absent.

#### Scenario: Resolved configs before model generation
- **WHEN** inference starts
- **THEN** resolved config artifacts are written before backend generation

#### Scenario: Terminal failure manifest
- **WHEN** inference terminates due to a contract error after run directory
  creation
- **THEN** available manifest or summary evidence records terminal status
  without claiming benchmark eligibility

### Requirement: Inference source ownership
The pipeline SHALL use the approved `src/inference/*` module ownership boundaries.
It uses `runtime.py` for setup, `backend.py` for decode, `prompt.py` for prompt
construction, `parsing.py` for parsing, `scoring.py` for scoring, and
`artifacts.py` for inference artifact rows. It MUST NOT import
`src.training.pipeline`, `TrainConfig`, `ResolvedTrainConfig`, or
`ResolvedStepSchedule`.

#### Scenario: No training pipeline import
- **WHEN** inference modules are imported
- **THEN** they do not import `src.training.pipeline`, `TrainConfig`,
  `ResolvedTrainConfig`, or `ResolvedStepSchedule`

#### Scenario: Runtime assembler
- **WHEN** runtime setup loads a base-plus-adapter checkpoint
- **THEN** it delegates Qwen/adapter/artifact identity mechanics to their owner
  modules rather than reimplementing them inline

### Requirement: Implementation approval gate
The inference pipeline SHALL require explicit user approval before first source
implementation.
Required preconditions are this OpenSpec change, the superpower implementation
plan, and review-convergence triage. After approval is granted, source
implementation may proceed under the approved task ledger.

#### Scenario: Pre-approval state
- **WHEN** this OpenSpec is drafted and validated
- **THEN** implementation tasks remain unchecked and no source implementation
  claim is made

#### Scenario: Post-approval state
- **WHEN** the user grants implementation approval after source-study review
- **THEN** implementation tasks may be completed and verified without treating
  this approval gate as a remaining blocker
