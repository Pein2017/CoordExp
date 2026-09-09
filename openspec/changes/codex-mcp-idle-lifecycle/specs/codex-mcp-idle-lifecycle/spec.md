## Purpose

Release idle, reconstructible MCP process resources without losing Desktop task
continuity or interrupting active work, and reconstruct those resources before
subsequent tool use through the owning Codex runtime.

## ADDED Requirements

### Requirement: Explicit and bounded idle policy

The system SHALL apply automatic idle suspension only when explicitly enabled
for eligible local stdio servers, SHALL accept positive main-task and
completed-subagent grace intervals, and SHALL preserve existing behavior when
the policy is disabled. Unsupported transports or unreconstructible session
state SHALL remain resident rather than being terminated speculatively.

#### Scenario: Policy disabled
- **WHEN** a task has no enabled idle policy
- **THEN** its MCP startup, retention, and call behavior remain unchanged

#### Scenario: Eligible idle task
- **WHEN** an eligible server belongs to an idle task, its configured grace has
  elapsed, and activity and reconstruction checks permit suspension
- **THEN** its stdio process is closed through its owner and its resources are
  released without deleting the task

#### Scenario: Noneligible server
- **WHEN** another server in that task is not opted in or uses an unsupported transport
- **THEN** that server is not suspended or restarted by the idle policy

### Requirement: Activity admission and suspension are mutually safe

The system SHALL serialize suspension against turn and direct MCP activity
admission. An active turn, in-flight MCP operation, pending elicitation, or
known unresolved timed-out/cancelled operation SHALL prevent automatic
suspension of the affected runtime resources.

#### Scenario: A new call races the idle deadline
- **WHEN** demand arrives while an idle suspension is being admitted
- **THEN** either the demand prevents suspension or it waits for suspension and
  reconstruction, and it does not execute against an intentionally closed client

#### Scenario: Long-running or uncertain call
- **WHEN** a call is still executing or its completion remains uncertain after timeout/cancellation
- **THEN** the idle deadline does not cause that server to be terminated

### Requirement: Demand reconstructs suspended resources once

The system SHALL reconstruct intentionally suspended resources before a new
turn or direct MCP tool/resource operation uses them. Concurrent demand SHALL
share one reconstruction, and a failed or ambiguous business operation SHALL
NOT be automatically replayed as a recovery strategy.

#### Scenario: Same main task continues
- **WHEN** a main task uses its server after idle suspension
- **THEN** the operation succeeds after reconstruction with the same task ID,
  history, effective workspace, and required server context

#### Scenario: Concurrent cold demand
- **WHEN** two operations arrive for the same suspended server
- **THEN** one replacement instance is created and both operations use a valid instance

#### Scenario: Replacement startup fails
- **WHEN** the replacement cannot initialize or restore required context
- **THEN** demand receives an actionable error and no original business operation
  is replayed or silently redirected to a different workspace

### Requirement: Project context is preserved or suspension is declined

The system SHALL preserve the effective Serena project/worktree context across
automatic suspension. It SHALL decline suspension when the relevant session
context cannot be safely reconstructed.

#### Scenario: Explicit project activation
- **WHEN** Serena has successfully activated a project different from its startup cwd
- **THEN** a subsequent query after suspension addresses that same project, or
  the runtime retained the original instance because restoration was not supported

### Requirement: Reclamation is scoped to MCP resources

The system SHALL preserve main-task history and identity, background execution,
noneligible connections, neighboring tasks, and independently supervised wake
monitors while suspending eligible MCP resources. It SHALL use the same MCP
lifecycle mechanism for main tasks and native subagents without imposing a new
subagent expiration or history-recovery contract.

#### Scenario: Background work and neighboring task
- **WHEN** one idle task's eligible MCP is suspended while a background command
  and another task remain active
- **THEN** those workloads continue without interruption

#### Scenario: Wake frontend is suspended
- **WHEN** an eligible wake MCP frontend is suspended
- **THEN** its independently running daemon and durable monitor records are not
  cancelled, deleted, marked successful, or otherwise changed by reclamation

### Requirement: Observation does not defeat suspension

The system SHALL retain bounded diagnostic state for suspension and recovery.
Passive status inspection SHALL NOT repeatedly recreate suspended processes,
and automatic background prewarming SHALL NOT undo intentional suspension.

#### Scenario: Desktop observes an idle task
- **WHEN** status is read repeatedly while a task's MCP is suspended
- **THEN** the status remains interpretable and no replacement starts until real demand
