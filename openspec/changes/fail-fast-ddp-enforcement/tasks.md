## 0. Audit (DDP-critical suppression inventory)

- [x] 0.1 Inventory all DDP collectives/barriers in core training paths and list the surrounding exception-handling behavior:
  - `dist.all_reduce`, `dist.all_gather_object`, `dist.broadcast`, `dist.broadcast_object_list`, `dist.barrier`, `dist.monitored_barrier`
  - rank-local early returns / caches that can gate collectives
  - “warn and continue” patterns
- [x] 0.2 Record known high-risk sites (initial evidence):
  - Stage-2 AB pending-metric reduction (`src/trainers/stage2_two_channel.py`)
  - rollout-aligned metric reduction (`src/trainers/stage2_rollout_aligned.py`)
  - dataset metric key sync (`src/trainers/metrics/mixins.py`)
  - Stage-2 AB phase barriers / monitored-barrier fallback (`src/trainers/stage2_two_channel/executors.py`)
- [x] 0.3 Document a short incident note referencing the recent DDP deadlock fix (“align final backward”) and the general rule it implies: no rank divergence around collectives.

## 1. Spec deltas (contract)

- [x] 1.1 Update `silent-failure-policy` to explicitly define DDP-critical regions and forbid swallow-and-continue around distributed collectives.
- [x] 1.2 Update `trainer-metrics-components` to require DDP-safe metric aggregation:
  - deterministic key sets + ordering across ranks,
  - no local fallback under DDP for aggregation failures,
  - no best-effort wrappers around collective-based sync.

## 2. Implementation (DDP fail-fast enforcement)

- [x] 2.1 Add a small DDP helper (repo-owned) to implement coordinated failure propagation:
  - rank0-only side effect wrapper:
    - bounded entry/exit alignment barriers,
    - rank0 captures failures without raising before broadcast,
    - broadcast `{failed_flag, error_summary}` and raise on all ranks after exit alignment.
  - any-rank wrapper (DDP-critical regions):
    - catch exceptions only to coordinate termination,
    - reduce a failure-flag tensor so all ranks agree on “any failure happened”,
    - raise on all ranks with a message that includes `where`, `rank`, `world_size` (rank0 logs full traceback).
- [x] 2.2 Fix Stage-2 AB pending metric reduction to be strict under DDP:
  - remove `try/except` “proceed without key union” fallback under DDP,
  - if key union or all-reduce fails, abort all ranks with coordinated error propagation.
- [x] 2.3 Fix rollout-aligned metric reduction to be strict under DDP (same rule as above).
- [x] 2.4 Fix dataset metric key sync to be rank-symmetric:
  - remove rank-local early return gating for `all_gather_object`, or
  - make the “do sync” decision a global (collective-reduced) decision so either all ranks call or none do.
  - ensure exceptions in this path are not swallowed under DDP.
- [x] 2.5 Stage-2 AB phase barriers:
  - ensure monitored barriers remain bounded when enabled,
  - if monitor group init fails and DDP monitoring is enabled, fail fast with actionable guidance (do not silently downgrade to unbounded `dist.barrier()`).
- [x] 2.6 Stage-2 launcher readiness probes:
  - keep “timeout-only” semantics, but bound each `curl` probe with connect+total time so `WAIT_TIMEOUT` cannot be bypassed.

## 3. Tests / verification

- [x] 3.1 Add a minimal CPU DDP regression test (2 ranks) that validates:
  - metric aggregation paths do not hang when one rank triggers an error,
  - failures terminate all ranks with a clear exception message.
- [ ] 3.2 Run targeted tests:
  - `conda run -n ms python -m pytest tests/` (scoped to any new DDP regression tests)
  - a Stage-2 smoke run (short) to confirm no deadlocks at log/metric aggregation boundaries.
- [x] 3.3 Add a regression check that readiness probing cannot exceed `WAIT_TIMEOUT` due to a stuck probe (e.g., verify `curl` uses connect+max time).
- [x] 3.4 Add a CPU DDP test (2 ranks) that simulates a rank0-only side effect failure and asserts both ranks exit (non-zero) without hanging at the exit barrier.
- [x] 3.5 Add a CPU DDP test (2 ranks) that simulates a non-rank0 exception inside a DDP-critical aggregation step and asserts both ranks terminate (non-zero) without hanging at a later collective.

## Evidence Notes (2026-03-04)

- Completed task evidence:
  - 0.1 inventory: `openspec/changes/fail-fast-ddp-enforcement/design.md` (section: `DDP-Critical Inventory (2026-03-04)`)
  - 0.2 high-risk sites: `openspec/changes/fail-fast-ddp-enforcement/design.md` (section: `DDP-Critical Inventory (2026-03-04)`)
  - 0.3 incident note: `openspec/changes/fail-fast-ddp-enforcement/design.md`, `openspec/changes/fail-fast-ddp-enforcement/proposal.md`
  - 1.1: `openspec/changes/fail-fast-ddp-enforcement/specs/silent-failure-policy/spec.md`
  - 1.2: `openspec/changes/fail-fast-ddp-enforcement/specs/trainer-metrics-components/spec.md`
  - 2.1: `src/utils/ddp_fail_fast.py`
  - 2.2: `src/trainers/stage2_two_channel.py`
  - 2.3: `src/trainers/stage2_rollout_aligned.py`
  - 2.4: `src/trainers/metrics/mixins.py`
  - 2.5: `src/trainers/stage2_two_channel/executors.py`
  - 2.6: `src/launchers/stage2_vllm_server.py`
- Targeted no-GPU checks run:
  - `conda run -n ms python -m pytest -q tests/test_ddp_fail_fast_stage2_metrics.py tests/test_ddp_vllm_sync_failure_propagation.py tests/test_stage2_vllm_server_launcher.py tests/test_stage2_ab_ddp_phase_monitor_disable.py`
  - Result: `18 passed`
- True CPU 2-rank DDP process tests added:
  - `tests/test_ddp_fail_fast_multiprocess.py`
  - Covers:
    - Stage-2 metric aggregation non-rank0 failure propagation (`3.1`, `3.5`)
    - rank0-only side-effect coordinated failure propagation (`3.4`)
  - Commands run:
    - `conda run -n ms python -m pytest -q tests/test_ddp_fail_fast_multiprocess.py`
    - Result: `3 passed`
    - `conda run -n ms python -m pytest -q tests/test_ddp_fail_fast_multiprocess.py tests/test_ddp_fail_fast_stage2_metrics.py tests/test_ddp_vllm_sync_failure_propagation.py tests/test_stage2_ab_ddp_phase_monitor_disable.py`
    - Result: `10 passed`
- Readiness-timeout budget regression added:
  - `tests/test_stage2_vllm_server_launcher.py::test_wait_for_server_health_bounds_probe_timeout_by_remaining_budget`
  - Launcher fix: `src/launchers/stage2_vllm_server.py::_wait_for_server_health` now caps each probe timeout by remaining `WAIT_TIMEOUT` budget.
  - Commands run:
    - `conda run -n ms python -m pytest -q tests/test_stage2_vllm_server_launcher.py tests/test_ddp_fail_fast_multiprocess.py`
    - Result: `15 passed`
    - `conda run -n ms python -m pytest -q tests/test_stage2_vllm_server_launcher.py tests/test_ddp_fail_fast_multiprocess.py tests/test_ddp_fail_fast_stage2_metrics.py tests/test_ddp_vllm_sync_failure_propagation.py tests/test_stage2_ab_ddp_phase_monitor_disable.py`
    - Result: `22 passed`
- Remaining open items require additional evidence:
  - Stage-2 smoke run evidence (`3.2`).
