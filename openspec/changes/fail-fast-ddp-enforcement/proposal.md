## Why

Stage-2 training runs in a multi-process (DDP) environment where **control-flow divergence across ranks is catastrophic**:
- If one rank enters a distributed collective (e.g., `dist.all_reduce`, `dist.barrier`, `dist.all_gather_object`) while another rank skips it (due to a swallowed exception, a “best-effort” early return, or a rank-local cache decision), the job can **deadlock/hang indefinitely** with little actionable signal.
- If failures are handled “locally” (warn-and-continue) under `torch.distributed`, downstream collectives frequently become the first observable symptom, turning a root-cause exception into a hang.

We recently fixed a Stage-2 AB DDP deadlock by aligning the Channel-A final backward semantics under DDP. That incident is a strong reminder that **DDP safety must be treated as a first-class contract**: failures must be coordinated across ranks, and exception-handling must be designed to preserve rank-symmetric collective behavior.

This change proposal enforces strict **fail-fast semantics** in DDP-critical code paths so unexpected failures become immediate, observable terminations rather than silent degradation or hanging behavior.

## What Changes

### Definitions (normative)

- **DDP initialized**: `torch.distributed.is_available() and torch.distributed.is_initialized()` and `world_size > 1`.
- **DDP-critical region**: any code path that can reach a distributed collective or barrier (including metric aggregation collectives in `Trainer.log()` and “diagnostics-only” sync calls).
- **Rank-symmetric control flow**: in a DDP-critical region, every rank MUST take the same “enter vs skip” decisions for collectives (no rank-local early return gates).
- **Fail-fast under DDP**: when DDP is initialized, any unexpected exception in a DDP-critical region MUST terminate the run, and termination MUST be coordinated so all ranks error out together (not “rank 0 continues” / “rank 3 stalls”).

### Enforced behavior (summary)

- **No silent recovery around collectives**:
  - DDP-critical code MUST NOT catch exceptions and “proceed with local fallback” when a collective was expected.
  - “Best-effort” wrappers MUST NOT be used for code that can call `torch.distributed` collectives.
- **Coordinated rank0-only operations** (canonical pattern):
  - For rank0-only side effects that are surrounded by barriers (e.g., model sync, server sync, metric key union decisions), failures MUST be broadcast to all ranks so all ranks raise together.
- **Bounded waits (launcher)**:
  - Keep the existing “wait until overall timeout” semantics for server readiness, but ensure no single readiness probe can block indefinitely (e.g., bound `curl` duration so `WAIT_TIMEOUT` is meaningful).

## Capabilities

### Modified Capabilities
- `silent-failure-policy`: explicitly treat DDP-critical regions as strict fail-fast zones; forbid “warn and continue” around collectives; define rank-symmetric control flow as a correctness requirement.
- `trainer-metrics-components`: require DDP-safe metric aggregation contracts (stable key sets and deterministic ordering) and forbid local fallback under DDP for aggregation failures.

## Impact

- **Breaking (intentional)**: failures that were previously “warning-only” (or silently downgraded to local metrics) will now terminate DDP runs to avoid deadlocks and silent corruption.
- Affected areas include:
  - DDP metrics aggregation paths (Stage-2 buffered metrics, rollout metrics, dataset metric key sync),
  - Stage-2 channel synchronization barriers / monitored barrier behavior,
  - Stage-2 launcher readiness checks (bounded waits).

