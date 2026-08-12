## ADDED Requirements

### Requirement: Preflight Artifact Ownership Remains Rank-Zero Single-Writer

Cache admission that runs before Accelerator setup MUST still initialize and
finalize the canonical shared run artifact. The strictly validated launcher
rank zero SHALL be the only preflight writer through a bounded CPU control
plane. After admission, Accelerator rank/world identity MUST match the launcher
identity so the same logical rank zero continues ownership. Nonzero ranks MUST
NOT create or write alternative run trees.

#### Scenario: Cache admission fails before Accelerator setup

- **WHEN** one or more ranks reject a required cache during model-free
  preflight
- **THEN** all ranks converge on the same bounded failure
- **AND** launcher rank zero atomically finalizes the one shared run artifact
  with terminal phase `cache_admission`
- **AND** no nonzero rank writes a run artifact

#### Scenario: Cache admission succeeds before Accelerator setup

- **WHEN** all ranks admit their required cache state
- **THEN** the same shared run artifact records completed admission
- **AND** Accelerator setup may continue only after rank/world identity equality
  is attested

#### Scenario: One rank observes an Accelerator identity mismatch

- **WHEN** cache payload validation succeeds but one live rank observes an
  Accelerator rank/world identity different from the attested launcher world
- **THEN** all live ranks converge the mismatch before model work
- **AND** rank zero finalizes the one shared artifact as failed with terminal
  phase `cache_admission`

#### Scenario: Artifact-owner handshake fails after rank-zero initialization

- **WHEN** rank zero has initialized the shared run artifact but the bounded
  ownership broadcast or peer handshake fails before cache admission begins
- **THEN** rank zero best-effort finalizes that same artifact as failed before
  propagating the handshake error
- **AND** no initialized orphan or rank-local replacement run tree is accepted

### Requirement: Run Artifacts Record Executed Environment Provenance

Every training run MUST record enough provenance to identify the executed code
and critical runtime dependencies. The receipt SHALL include the repository
commit, whether tracked or untracked local changes were present, a stable digest
or explicit unavailable status for relevant local changes, and resolved
versions plus source or binary identities for Transformers, FlashAttention,
Torch, Accelerate, PEFT, and tokenizers. Provenance collection MUST NOT copy
credentials or secret environment values into artifacts.

#### Scenario: A clean checkout starts training

- **WHEN** training starts from a clean checkout
- **THEN** the run receipt records the exact commit, clean state, package
  versions, Transformers source identity, and FlashAttention binary identity

#### Scenario: A dirty checkout starts training

- **WHEN** tracked or untracked repository changes relevant to execution are
  present
- **THEN** the receipt marks the run dirty and records a stable non-secret
  identity for the executed state instead of presenting the commit as complete
  provenance

#### Scenario: A dependency identity cannot be resolved

- **WHEN** a critical source or binary identity cannot be obtained safely
- **THEN** the receipt records an explicit unavailable status and reason rather
  than substituting an assumed or package-name-only identity

### Requirement: Training Phase And Resource Receipts Are Comparable

Production-shaped training and benchmark artifacts MUST distinguish cache
preparation, publication, admission, model loading, first successful optimizer
step, and steady-state training. Each measured comparison SHALL identify the
input provider, packing policy, cache identity, world size, warm-up exclusion,
sample or step count, wall-clock scope, and relevant CPU RSS, I/O, and GPU-memory
high-water marks.

#### Scenario: A production-shaped benchmark succeeds

- **WHEN** a benchmark reaches its declared steady-state interval
- **THEN** its artifact contains phase timestamps or durations, resource
  high-water marks, the exact comparison arm, and the count of accepted measured
  steps after warm-up

#### Scenario: A launch fails before the first optimizer step

- **WHEN** preparation, publication, admission, or model loading fails
- **THEN** the failure artifact identifies the terminal phase and excludes the
  run from steady-state throughput claims

#### Scenario: Two optimization arms are compared

- **WHEN** a candidate and compatibility reference are evaluated for promotion
- **THEN** their receipts use the same wall-clock scope and workload identity
  and expose semantic-equivalence results alongside efficiency measurements

### Requirement: Artifact Policy Identity Is Additive

Run artifacts MUST record the resolved cache, packing, input-provider,
attention-proof, and resume policy identities without removing or changing the
meaning of existing fields. New run segments created by resume MUST reference
their parent run and checkpoint while preserving prior artifacts unchanged.

#### Scenario: A new policy field is introduced

- **WHEN** a run uses a newly supported packing or input-provider policy
- **THEN** the resolved policy and version are added to the run receipt while
  existing consumers can continue reading the prior stable fields

#### Scenario: Training resumes from an earlier run

- **WHEN** exact resume creates a continuation segment
- **THEN** the new artifact references the parent run, checkpoint identity, and
  continuation index without overwriting the parent's artifacts

### Requirement: Wave 2 Parity Artifacts Are Versioned And Evidence-Preserving

The conditionally reopened Wave 2 parity probe MUST use authenticated v3 plan
and receipt schemas and MUST preserve the immutable v1/v2 failed artifacts
without reinterpreting them. A v3 passed, failed, or `unmeasurable` receipt
SHALL bind its exact plan, parent-v2 workload identity, source/model/dependency
identity, expected and observed trainable inventories, executed arm/cadence
events, all-layer proof, completed comparisons, negative control, timing,
resources, phase state, and terminal reason. Publication SHALL remain strict
JSON, bounded, atomic, and absent-target-only.

#### Scenario: A v3 arm fails after producing evidence

- **WHEN** identity, coverage, cadence, proof, repeat measurability, parity,
  negative-control, finite-value, resource, or publication validation fails
- **THEN** the terminal receipt preserves every completed bounded evidence
  surface and exact completed-phase prefix
- **AND** it does not claim an unexecuted phase or erase the failure behind a
  successful process exit

#### Scenario: A historical v2 receipt is loaded

- **WHEN** an immutable v2 failed receipt is read after v3 is introduced
- **THEN** it remains readable as historical failure evidence
- **AND** it cannot satisfy the v3 schema, v3 gate, or a retroactive Wave 2 pass
- **AND** v3 does not recompute or publish a parallel legacy storage-dtype
  comparison

#### Scenario: The immutable executed v3 failure predates richer execution fields

- **WHEN** the current reader loads the historical Wave 2 v3
  `failed/qwen.parity.clean_failed` receipt at stage `comparisons`
- **THEN** omission of `execution.requested_device` and
  `execution.gpu_idle_preflight` is accepted only when plan hash
  `03834e309357dace9d7b51a6c51186b14b09bea8953b3c3d12e82609cff1a49d`,
  complete plan-payload SHA-256
  `6f27e22b3e8a2c8aff626e9062afc0d664cee04e77341d0d4fdf7723c08a3483`,
  and complete receipt-payload SHA-256
  `8bdea86554c64b656b6a60325c2b5d69a12d9e58b2a7cbfdac01e0a9f9aabe13`
  all match exactly
- **AND** any payload mutation rejects the exception
- **AND** every current rich receipt still requires both execution fields
- **AND** this read-only exception cannot alter the terminal result, satisfy a
  new gate, or authorize another execution

#### Scenario: The one v3 execution becomes consumed

- **WHEN** the audited v3 command is about to enter real-model GPU setup
- **THEN** it first atomically publishes one immutable absent-target
  attempt-start marker bound to the plan hash, command/source/dependency
  identities, receipt target, and exact installed 589-row trainable inventory
- **AND** a pre-existing marker rejects another invocation before model setup
- **AND** the terminal receipt references the marker
- **AND** marker publication consumes the one authorized attempt without
  automatic retry, sample substitution, tolerance change, or v4 continuation

#### Scenario: The process stops around attempt-marker publication

- **WHEN** a failure is injected before the attempt marker is published
- **THEN** no model execution or consumed-attempt claim is made and another
  launch requires a fresh audit
- **WHEN** a failure occurs after marker publication
- **THEN** the immutable marker remains the durable consumed-attempt record
