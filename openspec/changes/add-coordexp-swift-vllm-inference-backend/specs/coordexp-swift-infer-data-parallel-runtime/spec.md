## ADDED Requirements

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
controller MUST observe successful worker exit, no surviving descendants, and
GPU memory returned to the recorded pre-worker baseline before publishing
canonical top-level completion artifacts. HF MAY retain direct one-rank
execution.

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
- **THEN** canonical publication waits for worker exit and parent-observed GPU
  memory return

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

## REMOVED Requirements

### Requirement: Backend-neutral shard contract with HF-only V1 execution

**Reason**: This change makes the previously reserved vLLM worker path
executable while retaining the backend-neutral shard boundary.

**Migration**: HF keeps its existing shard behavior; vLLM uses the new
backend-neutral HF and vLLM shard requirement above.
