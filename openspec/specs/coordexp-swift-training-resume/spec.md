# coordexp-swift-training-resume Specification

## Purpose

Defines the bounded, opt-in contract for exact CoordExp-Swift training continuation while keeping inference checkpoint payloads independently loadable and minimal.

## Requirements

### Requirement: Exact Training State Is An Opt-In Typed Sibling

Exact training state SHALL be disabled by default. When enabled, a scheduled
checkpoint MUST publish a versioned and explicitly typed `training_state/`
sibling beside the independently loadable inference payload. The training-state
manifest MUST bind all trainable model state, optimizer and scheduler state,
mixed-precision scaler state when present, planned optimizer-step position,
data and pack cursor, per-rank random-number-generator state, resolved config,
cache and runtime-policy identities, topology, and every required rank
contribution. Frozen base weights MAY be bound by immutable identity rather
than copied. The inference payload MUST NOT require, parse, or infer semantics
from this sibling.

#### Scenario: Exact training state is disabled

- **WHEN** a run uses `resume.mode: disabled`
- **THEN** scheduled checkpoints MUST publish only their inference payloads
- **AND** no `training_state/` directory, training-state manifest, or exact-state
  publication identity may be written.

#### Scenario: Exact training state is enabled

- **WHEN** an exact-state-enabled run successfully publishes a scheduled
  checkpoint at a supported optimizer-step boundary
- **THEN** the checkpoint MUST contain a committed, typed `training_state/`
  sibling covering every required rank contribution
- **AND** its inference payload MUST remain independently loadable.

#### Scenario: Inference reads a checkpoint with exact state

- **WHEN** an inference loader consumes the adapter and optional selected-token
  embedding payload from a checkpoint that also contains `training_state/`
- **THEN** inference output and admission MUST be governed only by the explicit
  inference payload and its manifest
- **AND** inference MUST ignore the training-state sibling.

### Requirement: Exact Resume Is Same-World-Size And Optimizer-Boundary Only

The first supported exact continuation contract SHALL admit only checkpoints
committed at completed optimizer-step boundaries and only when the child launch
uses the same world size and compatible rank/device mapping as the parent.
Saving or resuming inside a gradient-accumulation window is unsupported and
MUST NOT be advertised as exact. Exact continuation MUST restore the next data
and pack cursor, trainable model state, optimizer, scheduler, scaler when
present, and per-rank random-number-generator streams before the next forward
or optimizer mutation.

Qualification of this contract MUST use two branches from the same committed
boundary: an uninterrupted control branch and a parent-to-resumed-child branch.
Before their corresponding next forwards, the branches MUST match on next
input and pack identity and on the trainable, optimizer, scheduler, scaler when
present, per-rank random-number-generator, and cursor state required by the
declared exact policy. Both branches MUST execute the corresponding next
forward and optimizer update, after which their objective/loss fields and
resulting trainable parameters MUST satisfy that policy's declared comparison
rule. Publication, admission, or restore inspection alone MUST NOT qualify the
checkpoint as exact continuation.

#### Scenario: Matching same-world-size continuation is admitted

- **WHEN** a committed optimizer-boundary checkpoint and child launch satisfy
  every declared compatibility identity at the same world size
- **THEN** restoration MAY proceed before the next forward
- **AND** the child MUST continue from the checkpoint's next planned step and
  next pack cursor.

#### Scenario: First post-resume update matches the uninterrupted branch

- **WHEN** an uninterrupted control and an admitted resumed child start from
  the same authenticated optimizer-step boundary
- **THEN** their next input/pack identity and declared pre-forward trainable,
  optimizer, scheduler, scaler, per-rank RNG, and cursor state MUST match
- **AND** each branch MUST execute the corresponding next forward and optimizer
  update
- **AND** the resulting objective/loss fields and trainable parameters MUST
  satisfy the declared exact-policy comparison rule.

#### Scenario: World size or save boundary differs

- **WHEN** the requested child world size differs from the parent or the state
  was not committed at a completed optimizer-step boundary
- **THEN** exact-resume admission MUST fail before model, optimizer, scheduler,
  scaler, cursor, or random-number-generator state is mutated
- **AND** it MUST NOT degrade to weights-only loading or restart silently.

### Requirement: Resume Admission Is Fail Closed

Before mutable state is restored, exact-resume admission MUST authenticate the
training-state schema and manifest, all required files and rank contributions,
the sibling inference payload identity, resolved-config compatibility, cache
fingerprint, packing and input-provider policy, deterministic-runtime policy,
world size and rank mapping, model topology, trainable-parameter inventory,
and declared dependency identities. Every mismatch MUST be reported as an
incompatibility; partial restore and best-effort fallback are forbidden.

#### Scenario: Required state is incomplete or corrupt

- **WHEN** a declared state file or rank contribution is absent, corrupt,
  duplicated, outside the manifest, or inconsistent with its digest
- **THEN** admission MUST reject the checkpoint before any mutable restore
- **AND** no subset of its state may be applied.

#### Scenario: Inference-only checkpoint is selected

- **WHEN** an operator requests exact resume from a valid inference payload
  without a committed typed training-state sibling
- **THEN** admission MUST identify the missing exact-state contract and fail
- **AND** the same payload MUST remain eligible for ordinary inference.

### Requirement: Exact-State Publication Is Atomic Across Ranks

A checkpoint MUST be advertised as exact only after all required ranks have
contributed successfully, the complete state has been authenticated, and the
authoritative publication event has committed its manifest identities. An
interruption, rank failure, malformed contribution, or publication failure
MUST leave no alias or completed event that advertises the incomplete state as
resumable. A separately committed inference payload MAY remain present, but it
MUST remain typed and treated as inference-only until exact-state publication
commits.

#### Scenario: One rank fails during exact-state publication

- **WHEN** any required rank fails, disappears, or contributes invalid state
  before the exact-state commit
- **THEN** all surviving ranks MUST converge a bounded checkpoint failure
- **AND** no final or best alias and no completed exact-state event may refer to
  that incomplete publication.

#### Scenario: Publication is interrupted before commit

- **WHEN** the process is interrupted after staging begins but before the
  exact-state manifest and publication event commit
- **THEN** historical readers and resume admission MUST treat the checkpoint as
  non-resumable
- **AND** a staging directory or unbound inference payload MUST NOT be promoted
  by inference from filenames or directory shape.

### Requirement: Resume Lineage Is Append Only

Every admitted continuation MUST create a new run segment that records the
parent run, parent checkpoint and exact-state identities, continuation index,
and resumed planned-step position. It MUST NOT overwrite the parent run's
config, log, checkpoint, selector, or publication records.

#### Scenario: A continuation run starts

- **WHEN** exact state is admitted successfully
- **THEN** the child run MUST record an authenticated parent link and a
  monotonically increasing continuation index
- **AND** all parent artifacts MUST remain unchanged.

#### Scenario: Historical artifacts contain extra metadata

- **WHEN** a historical reader encounters an older checkpoint or run with
  resume-like files but without the current committed schema and lineage
- **THEN** it MUST preserve the artifact as historical evidence
- **AND** it MUST NOT infer current exact-resume eligibility from extra files,
  names, or unchecked historical claims.
