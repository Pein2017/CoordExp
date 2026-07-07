# coordexp-swift-infer-pipeline Specification

## Purpose
TBD - created by archiving change build-coordexp-swift-inference-infra. Update Purpose after archive.
## Requirements
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

### Requirement: Controller-worker inference orchestration
The inference pipeline SHALL support a controller-worker execution path when more than one active rank is planned.
The controller owns resolved config writing, shard planning, worker launch,
worker exit validation, strict shard merge, and final top-level artifact
materialization. Workers own only their assigned shard rows and write
rank-local artifacts.

#### Scenario: Multi-rank execution
- **WHEN** more than one active rank is planned
- **THEN** the controller launches one worker subprocess per active rank
- **AND** top-level scored artifacts are materialized only after all workers
  succeed and strict merge validation passes

#### Scenario: Single-rank execution
- **WHEN** only one active rank is planned
- **THEN** the pipeline MAY use the existing direct single-process inference
  path
- **AND** the top-level artifact contract remains unchanged

### Requirement: Per-device batch decoding
The pipeline SHALL interpret `generation.batch_size` as the batch decoding size per device.
This value MUST be preserved for worker-local HF `generate` batching. Effective
maximum concurrent rows MAY equal `active_ranks * generation.batch_size`, but
that product is derived runtime capacity and MUST NOT replace the configured
per-device value.

#### Scenario: Four ranks with batch size four
- **WHEN** `generation.batch_size: 4` and four ranks are active
- **THEN** each worker batches up to four decode requests per HF generation call
- **AND** the merged manifest records per-device batch size `4`

### Requirement: Rank-local shard directories
Workers SHALL write rank-local artifacts under deterministic shard directories.
The canonical location is `run_dir/shards/rank-XXX/`, where `XXX` is the
zero-padded rank. Rank-local required artifacts are `gt_vs_pred.jsonl`,
`gt_vs_pred_scored.jsonl`, `gt_vs_pred_scored.jsonl.provenance.json`,
`pred_token_trace.jsonl`, `parse_diagnostics.jsonl`, `image_plan.jsonl`,
`summary.json`, and `run_manifest.json`. The shard manifest MUST include rank,
world size, parent-visible token, worker `CUDA_VISIBLE_DEVICES`, logical device,
assigned row indices and ids, shard-plan fingerprint, identity fingerprints,
artifact hashes, and terminal status.

#### Scenario: Rank-local output
- **WHEN** rank 3 completes
- **THEN** it writes required shard artifacts under `shards/rank-003/`
- **AND** its shard manifest records assigned rows and device binding evidence

#### Scenario: Missing required shard file
- **WHEN** a completed rank-local shard is missing any required artifact
- **THEN** strict merge fails before publishing top-level scored artifacts

### Requirement: Shard execution primitive
The pipeline SHALL expose an internal shard execution primitive for worker use.
The primitive MUST receive an already resolved config, assigned row ids or row
indices, a fixed shard output directory, and worker metadata. It MUST process
only assigned rows, MUST NOT re-resolve collision-policy run directories, and
MUST NOT publish canonical top-level artifacts from workers.

#### Scenario: Worker executes assigned rows only
- **WHEN** a worker receives a shard plan containing row ids A and C
- **THEN** it decodes and writes artifacts only for rows A and C
- **AND** rows not assigned to the worker are absent from the shard artifact

#### Scenario: Worker cannot publish root artifacts
- **WHEN** the shard primitive completes
- **THEN** artifacts are written only under the fixed shard output directory
- **AND** root `gt_vs_pred*.jsonl` files are not created by the worker

### Requirement: No metric reduction inside data-parallel inference
Data-parallel inference SHALL preserve the existing separation between inference and metric reduction.
The controller may merge inference artifacts, but it MUST NOT compute mAP or
mRecall inside `src.infer`. Evaluation remains delegated to the named evaluator
consumer over the merged top-level scored artifact.

#### Scenario: Merged artifact ready for eval
- **WHEN** data-parallel inference completes successfully
- **THEN** the evaluator consumes the merged top-level `gt_vs_pred_scored.jsonl`
- **AND** metric reduction is not performed by the inference controller
