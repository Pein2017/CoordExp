## Purpose

Schedule remote Pi compaction at user-task boundaries so a completed answer can
settle immediately while the next model request still receives compacted context.

## ADDED Requirements

### Requirement: Completed-answer threshold maintenance is deferred
When deferral is enabled, the plugin SHALL preserve a normally completed answer
without issuing a remote threshold compaction request or creating a checkpoint
solely because the answer crossed the threshold.

#### Scenario: Final answer crosses the threshold
- **WHEN** a normal final response crosses the compaction threshold with no queued work
- **THEN** the answer remains final, no compaction checkpoint is appended, and the prompt settles

### Requirement: The next request compacts before sampling
The next user prompt SHALL trigger required threshold compaction before its first
normal model request, including after reopening the persisted session.

#### Scenario: A subsequent prompt arrives
- **WHEN** the user submits another prompt after deferred threshold maintenance
- **THEN** one remote compaction completes before the normal model request and the new prompt remains available

#### Scenario: Session is reopened first
- **WHEN** the persisted session is reopened before submitting the next prompt
- **THEN** the same compaction-before-request ordering holds without a process-local pending flag

### Requirement: Other compaction semantics are retained
The policy SHALL preserve manual compaction, active-task continuation, overflow
recovery, cancellation, existing checkpoints, and Pi behavior when deferral is disabled.

#### Scenario: Manual compaction
- **WHEN** the user explicitly requests compaction
- **THEN** compaction executes immediately and waits without starting a task turn

#### Scenario: Active task or overflow recovery
- **WHEN** compaction occurs during an unfinished task or for context-overflow recovery
- **THEN** compression and the existing bounded continuation execute without repeating completed tools

#### Scenario: Opt-in setting is absent
- **WHEN** deferral is not enabled in the plugin settings
- **THEN** the existing post-answer threshold behavior is retained

### Requirement: Shared installation consumes the qualified local release
CLI and Web SHALL use the qualified immutable local plugin release and the shared
deferral setting. Installation SHALL preserve active work and existing histories.

#### Scenario: Runtime adoption
- **WHEN** the qualified release is selected and idle Web extensions are reloaded
- **THEN** new CLI loads and Web loads resolve the same local release with deferral enabled
