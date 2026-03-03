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

DDP-critical collectives include (non-exhaustive):
- `dist.all_reduce`, `dist.all_gather_object`, `dist.broadcast`, `dist.broadcast_object_list`, `dist.barrier`, `dist.monitored_barrier`.

#### Scenario: Distributed metric aggregation failure terminates all ranks
- **GIVEN** DDP is initialized with `world_size=2`
- **AND** a metric aggregation path uses `all_reduce` or `all_gather_object`
- **WHEN** any unexpected exception occurs on any rank during aggregation
- **THEN** the run terminates (non-zero) on all ranks
- **AND** the error message is actionable (includes the aggregation context / phase)
- **AND** the run does not hang waiting on a collective entered by only some ranks.

#### Scenario: Rank0-only side effect uses coordinated failure propagation
- **GIVEN** DDP is initialized
- **AND** a rank0-only operation is performed (e.g., sync step, one-rank I/O decision)
- **WHEN** the rank0 operation fails unexpectedly
- **THEN** the failure is broadcast to all ranks
- **AND** all ranks raise and terminate together (no “rank0 fails, others hang”).

### Requirement: Best-effort exception handling MUST NOT wrap collectives
Best-effort exception handling MUST be limited to non-correctness sinks and MUST NOT affect rank symmetry.

Under DDP, best-effort wrappers MUST NOT be used for any code path that:
- calls distributed collectives, or
- gates whether distributed collectives execute.

#### Scenario: Best-effort diagnostics never cause collective divergence
- **GIVEN** DDP is initialized
- **WHEN** a diagnostics helper fails unexpectedly
- **THEN** the system does not skip a distributed collective on only some ranks
- **AND** either (a) the run fails fast, or (b) diagnostics are disabled via a rank-symmetric decision that all ranks follow.
