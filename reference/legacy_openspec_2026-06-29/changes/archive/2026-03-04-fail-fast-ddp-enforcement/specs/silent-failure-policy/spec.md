# silent-failure-policy Specification (delta: DDP fail-fast; rank-symmetric collectives)

## Purpose
Extend the silent-failure policy with explicit DDP semantics: any region that may execute distributed collectives must be strict fail-fast and rank-symmetric to prevent deadlocks.

## Requirements

## ADDED Requirements

### Requirement: DDP-critical regions are strict fail-fast and rank-symmetric
When `torch.distributed` is initialized (`world_size > 1`), any code path that may execute distributed collectives is DDP-critical and MUST be strict.

Normative behavior under DDP:
- A DDP-critical region MUST NOT swallow exceptions and continue execution on only a subset of ranks.
- A DDP-critical region MUST NOT implement “local fallback” behavior (e.g., “proceed with local metrics”) when collectives are expected.
- Any decision that gates whether a collective executes (enter vs skip) MUST be **rank-symmetric**:
  - it MUST NOT depend on rank-local caches or rank-local error handling,
  - it MUST be either unconditional (all ranks execute), or computed via a collective so all ranks agree.
- Exception handling MAY catch exceptions **only** to coordinate a rank-symmetric termination for failures that occur **outside** distributed collectives (e.g., during local preprocessing before entering the next collective):
  - if an exception is caught on any rank, all ranks MUST still execute the same coordination collectives (e.g., reduce a failure-flag tensor and optionally exchange a short error summary), then all ranks MUST raise and terminate (non-zero).
  - Exceptions raised by `torch.distributed` collectives (or indicating process-group corruption) MUST NOT be caught for additional coordination collectives; doing so can deadlock. Such exceptions MUST be allowed to propagate.

DDP-critical collectives include (non-exhaustive):
- `dist.all_reduce`, `dist.all_gather_object`, `dist.broadcast`, `dist.broadcast_object_list`, `dist.barrier`, `dist.monitored_barrier`.

#### Scenario: Distributed metric aggregation failure terminates all ranks
- **GIVEN** DDP is initialized with `world_size=2`
- **AND** a metric aggregation path uses `all_reduce` or `all_gather_object`
- **WHEN** any unexpected exception occurs on any rank during aggregation
- **THEN** the run terminates (non-zero) on all ranks
- **AND** the error message is actionable and minimally includes `where`, `rank`, `world_size`
- **AND** the run does not hang waiting on a collective entered by only some ranks.

#### Scenario: Any-rank exception is propagated without hang
- **GIVEN** DDP is initialized with `world_size=2`
- **WHEN** rank 1 encounters an unexpected exception during local preprocessing within a DDP-critical metric aggregation step (before entering the next distributed collective)
- **THEN** both ranks execute the coordination step
- **AND** all ranks terminate (non-zero) with an error that includes `where` and the failing rank
- **AND** the run does not hang at a later collective.

#### Scenario: Rank0-only side effect uses coordinated failure propagation
- **GIVEN** DDP is initialized
- **AND** a rank0-only operation is performed (e.g., sync step, one-rank I/O decision)
- **WHEN** the rank0 operation fails unexpectedly
- **THEN** the failure is broadcast to all ranks
- **AND** all ranks raise and terminate together (no “rank0 fails, others hang”).

### Requirement: DDP coordination barriers MUST be bounded (no silent downgrade)
When DDP is initialized, any barrier used to enforce rank-symmetric control flow or to coordinate rank0-only side effects MUST be **bounded** with a **finite timeout** (e.g., via monitored barrier with a finite timeout, or an equivalent bounded mechanism that raises rather than waiting indefinitely).

If the bounded barrier mechanism is enabled/configured but cannot be initialized, the run MUST fail fast with actionable guidance and MUST NOT silently downgrade to an unbounded barrier.

#### Scenario: Coordination barrier does not hang indefinitely after early-rank failure
- **GIVEN** DDP is initialized
- **AND** a coordinated section uses a barrier for entry/exit alignment
- **WHEN** one rank exits early (unexpected exception) before reaching the barrier
- **THEN** other ranks observe a bounded timeout error and terminate (non-zero) rather than hanging indefinitely.

### Requirement: Best-effort exception handling MUST NOT wrap collectives
Best-effort exception handling MUST be limited to non-correctness sinks and MUST NOT affect rank symmetry.

Under DDP, best-effort wrappers MUST NOT be used for any code path that:
- calls distributed collectives, or
- gates whether distributed collectives execute.

#### Scenario: Best-effort diagnostics never cause collective divergence
- **GIVEN** DDP is initialized
- **WHEN** a diagnostics helper fails unexpectedly
- **THEN** the system does not skip a distributed collective on only some ranks
- **AND** either (a) the run fails fast, or (b) diagnostics are disabled via a rank-symmetric decision (e.g., broadcast a boolean with entry/exit barriers, or reduce a boolean flag) that all ranks follow.

### Requirement: Timeout-based waiting loops MUST bound each probe call
Any “wait until overall timeout” loop MUST ensure each probe call (e.g., HTTP readiness check) has a bounded duration so the overall timeout remains meaningful.

#### Scenario: Readiness wait respects WAIT_TIMEOUT (no stuck probe)
- **GIVEN** a readiness loop with overall timeout `WAIT_TIMEOUT`
- **WHEN** the target endpoint is unresponsive
- **THEN** the loop terminates within `WAIT_TIMEOUT` (plus a small constant overhead)
- **AND** no single probe blocks indefinitely.
