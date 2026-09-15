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

The pipeline SHALL use the approved `src/inference/*` module ownership
boundaries. `runtime.py` SHALL own processor-only frontend setup and backend
session launch preparation. `backend.py` SHALL own semantic requests, results,
likelihood pairs, and the session protocol. `hf_backend.py` SHALL own dynamic
HF model composition and execution. `vllm_backend.py` SHALL own offline vLLM
engine execution and live operational diagnostics. `execution_model.py` SHALL
own immutable execution-model materialization and structural receipt
validation; explicit comparison probes MAY own optional composition-fidelity
diagnostics. `prompt.py`, `parsing.py`, `scoring.py`, and `artifacts.py` SHALL
own their corresponding semantic contracts. `pipeline.py` MUST remain
orchestration-only and MUST NOT import training runtime owners.

#### Scenario: Materialized vLLM runtime

- **WHEN** runtime setup launches vLLM
- **THEN** `execution_model.py` resolves a structurally validated immutable
  model before `vllm_backend.py` opens the rank-local engine
- **AND** the pipeline does not require or bind a historical behavioral proof

#### Scenario: No training pipeline import

- **WHEN** inference modules are imported
- **THEN** they do not import training pipeline or training schedule owners

#### Scenario: Runtime assembler

- **WHEN** runtime setup loads a base-plus-adapter checkpoint
- **THEN** it delegates Qwen, adapter, backend-session, and artifact identity
  mechanics to their owner modules rather than reimplementing them inline

#### Scenario: Dynamic HF runtime

- **WHEN** runtime setup loads a base-plus-adapter checkpoint for HF
- **THEN** `hf_backend.py` composes the base, DoRA adapter, and selected-token
  delta directly through their owner modules

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

### Requirement: Backend launch preparation

The pipeline SHALL resolve shared request evidence before opening a backend and
SHALL obtain backend identity only from the opened session receipt. vLLM
controller mode MUST resolve and publish a structurally validated
execution-model receipt before workers launch. The pipeline MUST NOT load an HF
model on the vLLM worker path or require a historical qualification receipt.

#### Scenario: vLLM launch

- **WHEN** backend type is vLLM
- **THEN** no HF inference model is loaded in the rank worker before the vLLM
  session opens
- **AND** actual engine construction and live decode evidence determine
  operational success

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
