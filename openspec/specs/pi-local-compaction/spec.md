# pi-local-compaction Specification

## Purpose
Keep shared CLI and Web-Pi remote compaction locally maintained while preserving unfinished task continuation, user control, and historical checkpoint replay.
## Requirements
### Requirement: Task-sensitive continuation
Successful remote compaction SHALL preserve the existing execution loop's continuation of an unfinished automatic task without creating an additional task or repeating completed tools.

#### Scenario: Automatic compaction during tool execution
- **WHEN** an active task automatically compacts after a completed tool result
- **THEN** the next natural request carries explicit task continuation after the opaque checkpoint and maintenance output, and the completed tool is not executed again

#### Scenario: Manual or settled compaction
- **WHEN** the user manually compacts or automatic maintenance follows a finished answer
- **THEN** compaction waits for the next prompt and does not introduce task continuation

### Requirement: User control and compatible replay
The plugin SHALL retain cancellation, newer-user-message priority, and replay of existing supported checkpoints.

#### Scenario: New input during compaction
- **WHEN** the user provides a newer instruction during successful active compaction
- **THEN** it follows the continuation instruction in the request and remains authoritative

#### Scenario: Cancelled compaction
- **WHEN** compaction is aborted before completion
- **THEN** no replacement checkpoint or extra task turn is published

#### Scenario: Existing checkpoint
- **WHEN** an existing v3 checkpoint without the new continuation item is resumed
- **THEN** it remains readable and replayable

### Requirement: Local release ownership
CLI and Web-Pi SHALL select one verified, versioned local release of the maintained plugin without requiring npm publication or public repository writes.

#### Scenario: Activate local release
- **WHEN** a built local release replaces the npm plugin
- **THEN** the shared profile selects the local artifact exactly once, while unrelated settings and active sessions are preserved

