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

### Requirement: Controller-side execution-model handoff
For every vLLM inference run, including one active rank, the controller MUST resolve an
immutable execution-model receipt once before workers launch and pass that
receipt explicitly to every worker. Base-only receipts hash the direct model;
adapter or embedding-delta receipts identify the content-addressed derived
model. Each worker MUST validate the same fingerprint, required file hashes,
and path before engine startup.

#### Scenario: Eight-rank composed checkpoint
- **WHEN** eight workers execute one adapter-plus-delta run
- **THEN** materialization occurs once and every worker records the identical
  execution-model fingerprint

#### Scenario: One-GPU vLLM run
- **WHEN** shard planning selects one active rank for vLLM
- **THEN** the controller still launches one fresh worker with the immutable
  execution-model receipt instead of executing the engine in the controller

### Requirement: vLLM worker process lifecycle
Each vLLM worker MUST retain the existing one-parent-visible-GPU environment,
open one TP=1 and DP=1 engine, use an explicitly qualified engine process mode,
and terminate every engine-owned process before successful or failed worker
exit. Uniprocess engine execution MAY be qualified and is preferred when it
removes nested process ownership without changing engine semantics. The
controller MUST apply a bounded worker timeout and terminate the worker's
complete owned process tree before returning terminal failure. It MUST NOT
allocate another parent-visible GPU.

The fresh worker process MUST be the final vLLM resource boundary. The
controller MUST observe successful worker exit and no surviving owned process
group before publishing canonical top-level completion artifacts. Real smoke
receipts SHOULD record parent-observed pre/post GPU memory as diagnostic
evidence, but global device-memory equality MUST NOT gate publication because
unrelated processes may change occupancy on shared GPUs. HF MAY retain direct
one-rank execution.

#### Scenario: Worker engine escapes CUDA binding
- **WHEN** a vLLM engine reports or allocates a device other than logical
  `cuda:0` inside the worker
- **THEN** the shard fails and top-level scored artifacts are not published

#### Scenario: Worker never exits
- **WHEN** a worker or engine-owned child exceeds the configured internal
  completion timeout
- **THEN** the controller terminates the complete owned process tree, records
  terminal diagnostics, and publishes no canonical top-level artifacts

#### Scenario: Session closes but GPU memory remains in the worker
- **WHEN** in-process vLLM shutdown completes while compiled model tensors are
  still allocated
- **THEN** canonical publication waits for worker and owned process-group exit,
  after which the operating system releases that process-owned CUDA context

### Requirement: Backend-neutral HF and vLLM shard contract
The data-parallel worker input and output contract SHALL support HF and offline
vLLM. Shard plans and merge logic MUST remain free of raw backend objects. HF
executes native batches per worker. vLLM executes one qualified offline engine
per worker with tensor and data parallel size one. Both backends MUST preserve
the same assigned rows, trace semantics, strict shard validation, and merged
artifact contract.

#### Scenario: HF worker execution
- **WHEN** `backend.type: hf` is configured and multiple ranks are active
- **THEN** each worker executes the HF backend over only its assigned rows

#### Scenario: vLLM worker execution
- **WHEN** `backend.type: vllm` is configured and multiple ranks are active
- **THEN** each worker opens one rank-local vLLM engine and executes only its
  assigned rows
