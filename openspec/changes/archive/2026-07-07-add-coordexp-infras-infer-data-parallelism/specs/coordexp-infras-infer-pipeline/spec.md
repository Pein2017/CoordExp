## ADDED Requirements

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
