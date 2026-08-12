## Purpose

Define exact, fail-closed continuation of CoordExp-Swift training state and
artifact lineage without conflating inference-only model payloads with a
reproducible training checkpoint.

## ADDED Requirements

### Requirement: Exact Resume State Is Distinct From Inference Payloads

An exact-resume checkpoint MUST be explicitly typed and MUST include every
trainable model surface, including adapter and selected-embedding state,
optimizer and scheduler state, mixed-precision scaler state when applicable,
optimizer-step and accumulation position, data and pack cursor, per-rank CPU
and CUDA RNG state, resolved configuration, cache and policy identities, world
size, and a versioned state manifest. Frozen base-model weights MUST be bound by
immutable content identity and need not be copied into each exact checkpoint. A
model-only or inference-minimal payload MUST remain readable for inference but
MUST NOT be accepted as exact training state.

#### Scenario: A complete training-state checkpoint is saved

- **WHEN** exact-resume checkpointing reaches an accepted save boundary
- **THEN** a versioned manifest atomically identifies every required state
  component, rank contribution, and parent run position

#### Scenario: A model-only checkpoint is selected for exact resume

- **WHEN** a user requests exact resume from an inference-minimal or incomplete
  checkpoint
- **THEN** admission fails before optimizer or model mutation and reports the
  missing training-state components

### Requirement: Exact Resume Storage Is Opt-In And Immutable

Exact training state MUST be disabled by default and, when enabled, MUST be
published as a versioned `training_state/` child alongside a scheduled
inference-loadable checkpoint at an optimizer-step boundary. The first
supported contract MUST NOT automatically prune committed exact-state
checkpoints. Storage growth SHALL be controlled by explicit enablement and
checkpoint cadence, while later retention or pruning requires a separate
policy that cannot silently remove an accepted resume ancestor.

The inference-loadable sibling MUST contain a self-authenticating, path-
independent payload manifest that enumerates the complete adapter and selected-
embedding payload. The authoritative checkpoint-publication event MUST bind
that manifest and MUST atomically persist the checkpoint-bound progress that
was durable when the event was committed.

#### Scenario: Exact resume is disabled

- **WHEN** a run uses the compatibility default with exact resume disabled
- **THEN** inference-loadable checkpoint publication remains unchanged and no
  `training_state/` payload is written

#### Scenario: Exact resume is enabled at a scheduled checkpoint

- **WHEN** an enabled run commits a checkpoint at an optimizer-step boundary
- **THEN** its exact state is atomically published under `training_state/`
- **AND** the sibling inference payload remains independently loadable
- **AND** no previously committed exact-state checkpoint is automatically
  overwritten or pruned

#### Scenario: An inference payload changes after publication

- **WHEN** any declared adapter or selected-embedding file is missing, added,
  or differs from the committed inference-payload manifest
- **THEN** checkpoint admission and exact-resume comparison fail before that
  payload is treated as independently loadable or equivalent

### Requirement: Exact Resume Uses A Declared Deterministic Replay Policy

The first exact-same-world-size contract MUST require the versioned runtime
policy `strict_cuda_replay_v1`. Before Accelerator, model, or CUDA
materialization on every rank, admission MUST verify launcher-provided
`FLASH_ATTENTION_DETERMINISTIC=1` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, enable PyTorch deterministic algorithms with
`warn_only=False`, set deterministic cuDNN behavior, disable cuDNN benchmarking,
and converge the resolved seed, policy, runtime baseline, and rank/world/device
mapping. An exact-resume request using the compatibility runtime policy MUST be
rejected. Runs with resume disabled MAY retain the compatibility default.

#### Scenario: Exact resume selects strict replay

- **WHEN** a same-world-size exact-resume run is resolved
- **THEN** every rank records and agrees on the complete strict replay policy
  before any CUDA owner is initialized
- **AND** the resolved policy is included in configuration, run, checkpoint,
  and resume-admission identities

#### Scenario: A strict replay prerequisite is absent or inconsistent

- **WHEN** a required environment value, deterministic algorithm state, cuDNN
  state, runtime baseline, or rank mapping is missing or differs on any rank
- **THEN** the run fails before model or CUDA materialization and does not
  silently set a weaker or warning-only policy

### Requirement: Resume Admission Is Fail Closed

Before restoring mutable training state, resume admission MUST validate the
checkpoint schema, integrity, configuration, cache fingerprint, packing and
input-provider policy, world size, model topology, trainable parameter set, and
dependency compatibility fields declared by the checkpoint contract. An
unsupported mismatch MUST fail rather than restart, skip state, or silently
degrade to model-only loading.

#### Scenario: Resume identities match

- **WHEN** every mandatory checkpoint and current-run identity matches
- **THEN** restore may proceed and the continuation receipt records the parent
  checkpoint and validated identities

#### Scenario: A mandatory identity differs

- **WHEN** cache identity, policy, world size, topology, trainable parameter
  set, or another mandatory compatibility field differs
- **THEN** resume fails before state mutation and names each incompatible field

#### Scenario: A checkpoint publication is incomplete

- **WHEN** one or more required rank-state files are absent, corrupt, or not
  covered by the committed manifest
- **THEN** the checkpoint is not admitted and no partial restore occurs

### Requirement: Resume Continues Presentation And Optimizer State

For the supported same-world-size contract, an interrupted-and-resumed run MUST
continue with the same next pack, example order, accumulation position,
optimizer and scheduler state, and restored RNG streams as the uninterrupted
reference. The acceptance test SHALL compare the next accepted optimizer steps
and declare dtype-appropriate numerical tolerances without claiming unsupported
bitwise reproducibility across independent CUDA launches.

Where the runtime owns aggregate accuracy sufficient statistics, every train
and evaluation row MUST persist exact integer `top1_correct`, `top5_correct`,
and `atom_count` values. Acceptance MUST compare those integers directly and
validate each published ratio against them; reconstructing a numerator by
rounding a floating ratio is not admissible evidence.

#### Scenario: Training is interrupted at an accumulation boundary

- **WHEN** a matched run is interrupted, saved, and resumed at a supported
  boundary
- **THEN** the next pack identities, optimizer-step sequence, learning rate,
  losses, and model updates match the uninterrupted reference within declared
  tolerances

#### Scenario: Training is interrupted inside accumulation

- **WHEN** the supported contract permits a checkpoint inside gradient
  accumulation
- **THEN** pending gradient, micro-step, cursor, and RNG state are restored; if
  the contract does not support that boundary, saving fails before publishing a
  checkpoint advertised as exact

### Requirement: Resume Lineage Is Append Only

Every resumed execution MUST create an additive run segment that references the
parent run, exact checkpoint identity, and continuation index. Resume MUST NOT
overwrite completed logs, receipts, inference payloads, or checkpoints from the
parent segment.

#### Scenario: A continuation run starts

- **WHEN** an exact checkpoint is admitted
- **THEN** a new run segment is created with a parent link and monotonically
  increasing continuation index while the parent artifacts remain unchanged

#### Scenario: A resumed segment publishes another checkpoint

- **WHEN** the continuation later saves exact training state
- **THEN** the new checkpoint records the continuation as its parent segment
  and preserves the full lineage chain
