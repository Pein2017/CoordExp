## 0. Audit (DDP-critical suppression inventory)

- [ ] 0.1 Inventory all DDP collectives/barriers in core training paths and list the surrounding exception-handling behavior:
  - `dist.all_reduce`, `dist.all_gather_object`, `dist.broadcast`, `dist.broadcast_object_list`, `dist.barrier`, `dist.monitored_barrier`
  - rank-local early returns / caches that can gate collectives
  - “warn and continue” patterns
- [ ] 0.2 Record known high-risk sites (initial evidence):
  - Stage-2 AB pending-metric reduction (`src/trainers/stage2_two_channel.py`)
  - rollout-aligned metric reduction (`src/trainers/stage2_rollout_aligned.py`)
  - dataset metric key sync (`src/trainers/metrics/mixins.py`)
  - Stage-2 AB phase barriers / monitored-barrier fallback (`src/trainers/stage2_two_channel/executors.py`)
- [ ] 0.3 Document a short incident note referencing the recent DDP deadlock fix (“align final backward”) and the general rule it implies: no rank divergence around collectives.

## 1. Spec deltas (contract)

- [ ] 1.1 Update `silent-failure-policy` to explicitly define DDP-critical regions and forbid swallow-and-continue around distributed collectives.
- [ ] 1.2 Update `trainer-metrics-components` to require DDP-safe metric aggregation:
  - deterministic key sets + ordering across ranks,
  - no local fallback under DDP for aggregation failures,
  - no best-effort wrappers around collective-based sync.

## 2. Implementation (DDP fail-fast enforcement)

- [ ] 2.1 Add a small DDP helper (repo-owned) to implement coordinated failure propagation:
  - broadcast `{failed_flag, message}` from rank0,
  - enforce symmetric barriers where needed,
  - provide a simple `ddp_fail_fast(where, fn_rank0_only=...)` or equivalent primitive.
- [ ] 2.2 Fix Stage-2 AB pending metric reduction to be strict under DDP:
  - remove `try/except` “proceed without key union” fallback under DDP,
  - if key union or all-reduce fails, abort all ranks with coordinated error propagation.
- [ ] 2.3 Fix rollout-aligned metric reduction to be strict under DDP (same rule as above).
- [ ] 2.4 Fix dataset metric key sync to be rank-symmetric:
  - remove rank-local early return gating for `all_gather_object`, or
  - make the “do sync” decision a global (collective-reduced) decision so either all ranks call or none do.
  - ensure exceptions in this path are not swallowed under DDP.
- [ ] 2.5 Stage-2 AB phase barriers:
  - ensure monitored barriers remain bounded when enabled,
  - if monitor group init fails and DDP monitoring is enabled, fail fast with actionable guidance (do not silently downgrade to unbounded `dist.barrier()`).
- [ ] 2.6 Stage-2 launcher readiness probes:
  - keep “timeout-only” semantics, but bound each `curl` probe with connect+total time so `WAIT_TIMEOUT` cannot be bypassed.

## 3. Tests / verification

- [ ] 3.1 Add a minimal CPU DDP regression test (2 ranks) that validates:
  - metric aggregation paths do not hang when one rank triggers an error,
  - failures terminate all ranks with a clear exception message.
- [ ] 3.2 Run targeted tests:
  - `conda run -n ms python -m pytest tests/` (scoped to any new DDP regression tests)
  - a Stage-2 smoke run (short) to confirm no deadlocks at log/metric aggregation boundaries.

