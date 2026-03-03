# trainer-metrics-components Specification (delta: strict DDP-safe aggregation)

## Purpose
Extend the metrics/logging contract with explicit DDP safety requirements: aggregation uses distributed collectives and must be deterministic and strict to avoid deadlocks.

## Requirements

## ADDED Requirements

### Requirement: DDP metric aggregation MUST be deterministic and strict
When `torch.distributed` is initialized (`world_size > 1`), any cross-rank metric aggregation MUST satisfy:

- **Deterministic key set**:
  - Either the metric key set is statically known (preferred), OR
  - the key set is computed via a distributed union step (e.g., `all_gather_object`) and MUST succeed on all ranks.
- **Deterministic ordering**:
  - the ordered key list used to build any reduction tensor MUST be identical on every rank (e.g., `sorted(keys)` after union).
- **Strict failure semantics**:
  - aggregation MUST NOT fall back to rank-local metrics when DDP is initialized,
  - any unexpected exception in aggregation MUST abort all ranks with coordinated error propagation.
  - all ranks MUST participate in union/reduction collectives even if their local metric set is empty (missing keys MUST reduce as zeros so tensor shapes match).

#### Scenario: Metric key union failure aborts rather than “proceeding locally”
- **GIVEN** DDP is initialized
- **AND** aggregation requires a distributed key union step
- **WHEN** the key union step fails on any rank
- **THEN** all ranks terminate with a clear error message
- **AND** the system does not continue with rank-local key lists.

#### Scenario: All-reduce tensor shape is identical across ranks
- **GIVEN** DDP is initialized
- **WHEN** aggregation performs an `all_reduce` over a tensor built from metric keys
- **THEN** the tensor length and key ordering are identical on every rank
- **AND** ranks with empty local metrics still build the same ordered tensor length (zero-filled)
- **AND** the system does not hang due to mismatched tensor shapes or skipped collectives.

### Requirement: Best-effort diagnostics MUST NOT perform collective sync
Diagnostics-only best-effort paths MUST be local-only and MUST NOT perform distributed collective sync.

Under DDP, any diagnostic that requires collectives (e.g., key synchronization) MUST be:
- strict (fail-fast), or
- disabled globally via a rank-symmetric decision computed with collectives (so all ranks agree).

#### Scenario: A diagnostics failure never causes collective divergence
- **GIVEN** DDP is initialized
- **WHEN** a diagnostics helper raises unexpectedly
- **THEN** the system does not skip a required collective on only a subset of ranks
- **AND** the job either fails fast or disables the diagnostic via a rank-symmetric decision.
