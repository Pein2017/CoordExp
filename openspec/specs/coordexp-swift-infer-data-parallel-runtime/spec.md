# coordexp-swift-infer-data-parallel-runtime Specification

## Purpose
TBD - created by archiving change add-coordexp-swift-infer-data-parallelism. Update Purpose after archive.
## Requirements
### Requirement: Default visible-CUDA data parallelism
Production inference SHALL use all useful CUDA devices visible to the process by default.
If `CUDA_VISIBLE_DEVICES` is set, its device tokens define the allowed resource
boundary. If it is unset, the controller MAY discover local CUDA device count
and synthesize tokens `0..N-1`. Inference MUST fail before model loading when no
CUDA device is visible.

#### Scenario: Visible devices are restricted externally
- **WHEN** inference starts with `CUDA_VISIBLE_DEVICES=2,3,5,7`
- **THEN** only device tokens `2`, `3`, `5`, and `7` are eligible for worker
  assignment
- **AND** no config or CLI value expands beyond that externally visible set

#### Scenario: No CUDA device visible
- **WHEN** inference starts and no CUDA device is visible
- **THEN** runtime validation fails before model loading
- **AND** no benchmark-looking scored artifact set is written

#### Scenario: Empty input
- **WHEN** inference input contains zero rows
- **THEN** inference fails before direct or worker execution
- **AND** no active rank count of zero is treated as a successful run

### Requirement: Active rank planning
The controller SHALL compute active ranks from visible CUDA devices and decode-batch count.
`generation.batch_size` is the immutable per-device decode batch size. The
controller MUST compute decode batches from the total ordered input rows and
MUST set `active_ranks = min(visible_cuda_device_count, decode_batch_count)`.

#### Scenario: More GPUs than decode batches
- **WHEN** four decode batches are planned and eight CUDA devices are visible
- **THEN** the run uses four active ranks
- **AND** no empty worker rank is launched only to load a model replica

#### Scenario: Per-device batch size
- **WHEN** `generation.batch_size: 4` and four ranks are active
- **THEN** each rank MAY submit HF decode batches of up to four rows
- **AND** the implementation MUST NOT divide `generation.batch_size` by rank
  count

### Requirement: Worker CUDA isolation
Each active worker SHALL be bound to exactly one CUDA device by environment isolation.
The controller MUST launch each worker with `CUDA_VISIBLE_DEVICES` narrowed to
one assigned parent-visible device token. Inside the worker, the only supported
runtime CUDA device is logical `cuda:0`. Workers MUST validate exactly one CUDA
device is visible before model loading. Worker metadata MUST normalize and
record `torch.cuda.device_count()==1`, `torch.cuda.current_device()==0`,
logical device `cuda:0`, and model first-parameter device after load.

#### Scenario: Worker bound to one token
- **WHEN** rank 2 is assigned parent-visible token `5`
- **THEN** the worker environment contains `CUDA_VISIBLE_DEVICES=5`
- **AND** the worker records logical runtime device `cuda:0`

#### Scenario: Worker sees multiple CUDA devices
- **WHEN** a worker starts and more than one CUDA device is visible inside that
  worker
- **THEN** the worker fails before model loading with a runtime contract error

#### Scenario: Device binding provenance
- **WHEN** a worker writes shard metadata
- **THEN** the metadata records rank, world size, assigned parent-visible token,
  worker `CUDA_VISIBLE_DEVICES`, CUDA device count, current CUDA device,
  worker logical device, and model first-parameter device

### Requirement: Fresh-interpreter worker launch
Multi-rank inference workers SHALL be launched with fresh-interpreter subprocess or explicit spawn semantics.
The controller MUST NOT use bare `multiprocessing.Process`, default
fork-context multiprocessing, or forkserver CUDA workers for multi-rank model
loading. If multiprocessing is used, the implementation MUST request an
explicit spawn context and test that behavior.

#### Scenario: Subprocess launch
- **WHEN** the controller launches rank-local workers
- **THEN** each worker is started through a fresh Python interpreter command or
  an explicitly spawned process
- **AND** the launch environment narrows `CUDA_VISIBLE_DEVICES` before worker
  imports can initialize CUDA

#### Scenario: Forked CUDA worker rejected
- **WHEN** the worker launch path uses default fork or forkserver semantics for
  CUDA model loading
- **THEN** the implementation is noncompliant even if environment variables are
  otherwise correct

### Requirement: Deterministic batch-block row sharding
The controller SHALL shard rows by deterministic decode-batch blocks.
Rows MUST first be grouped in original JSONL order into blocks of up to
`generation.batch_size`. Blocks MUST be assigned round-robin across active
ranks. The shard plan MUST record row indices, row ids, batch ids, rank
assignments, per-device batch size, and a shard-plan fingerprint.

#### Scenario: Round-robin block assignment
- **WHEN** five decode-batch blocks are planned across two active ranks
- **THEN** rank 0 receives blocks 0, 2, and 4
- **AND** rank 1 receives blocks 1 and 3

#### Scenario: Final row order
- **WHEN** workers complete out of order
- **THEN** merged top-level row artifacts are sorted by original `row_index`
- **AND** row ids match the full input row list exactly once

### Requirement: Backend-neutral shard contract with HF-only V1 execution
The data-parallel worker input and output contract SHALL be backend-neutral, but V1 execution SHALL remain HF-only.
The shard plan and merge layer MUST NOT assume HF-specific raw objects outside
the backend module. However, `backend.type: vllm` remains not implemented until
a later approved change.

#### Scenario: HF worker execution
- **WHEN** `backend.type: hf` is configured and multiple ranks are active
- **THEN** each worker executes the existing HF decode path over its assigned
  rows

#### Scenario: vLLM remains reserved
- **WHEN** `backend.type: vllm` is configured
- **THEN** config/runtime validation fails with a not-implemented contract error
- **AND** data-parallel shard planning does not treat vLLM as accepted evidence
