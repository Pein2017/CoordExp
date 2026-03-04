# silent-failure-policy Specification

## Purpose
Define the exception-handling policy for CoordExp so that core training/inference/evaluation behavior is reproducible and failures are observable, while allowing a narrow set of best-effort I/O sinks.
## Requirements
### Requirement: Core execution paths do not swallow unexpected exceptions
The system SHALL NOT suppress unexpected exceptions in core execution paths (dataset encoding, trainer steps, inference pipeline stages, evaluation). Code MUST either raise the exception or catch only explicitly enumerated exception types and emit an actionable error message.

Blanket suppression patterns are forbidden in core execution paths, including:
- `except Exception: pass`
- `except: pass`
- `except BaseException: pass`
- blanket `except Exception` with `continue`, `break`, or semantics-changing default `return`.

#### Scenario: Dataset encoding error is surfaced
- **WHEN** a dataset raises an exception while encoding a sample for training
- **THEN** the run fails fast with a clear error message
- **AND** the exception is not discarded by a blanket catch-all.

#### Scenario: Deprecated/legacy knobs are not accepted in core paths
- **WHEN** a core execution path exposes an argument/config knob that is declared deprecated
- **THEN** the deprecated knob is removed (or causes fail-fast) rather than silently ignored for backward compatibility.
- **AND** warning-only behavior is not permitted for deprecated knobs (runs stop instead of continuing with a no-op).

### Requirement: Blanket suppression is forbidden by direct CI scanning
CoordExp SHALL NOT maintain exception-suppression registries. Compliance MUST be enforced directly by CI scanning source files.

At minimum, the CI check in `tests/test_silent_failure_policy.py` MUST treat the following as equivalent blanket suppression patterns and fail:
  - `except Exception: pass`
  - `except: pass` (bare except)
  - `except BaseException: pass`

The CI check MUST also detect and fail blanket suppression handlers that use:
  - `except Exception: continue`
  - `except Exception: break`
  - `except Exception: return <default>`
when these patterns suppress exception propagation in core paths.

#### Scenario: Log tee I/O failure does not abort training
- **WHEN** the file logging tee fails to write to its mirror file
- **THEN** training continues without corrupting model state
- **AND** exceptions in non-I/O code paths are not suppressed.

### Requirement: Operator-controlled input violations are strict fail-fast contracts
Operator-controlled input violations MUST fail fast and MUST NOT be handled as continue-and-skip behavior.

This includes deterministic training and inference/eval contracts such as malformed JSONL, missing required fields, unreadable images, and geometry/schema violations that can be validated ahead of compute.

#### Scenario: Invalid operator input terminates run
- **WHEN** an operator-controlled input record violates a required contract
- **THEN** the run terminates non-zero with actionable diagnostics
- **AND** processing does not continue by silently skipping that record.

### Requirement: Continue-but-observable behavior is restricted to explicit model-output consumers
Continue-and-salvage behavior MAY be used only for explicit model-output consumer paths (for example, prediction parse/validation over produced model text), and MUST remain observable.

Normative behavior:
- The failing sample MUST emit structured error metadata.
- Run-level counters MUST increment for the corresponding error code/class.
- This carve-out MUST NOT be used for operator-controlled input violations or unexpected internal exceptions.

#### Scenario: Invalid model output is recorded with counters
- **GIVEN** a model-output consumer path
- **WHEN** model-generated output is malformed/truncated for one sample
- **THEN** structured sample error metadata is emitted and counters increment
- **AND** subsequent samples may continue under the explicit consumer policy.

### Requirement: Best-effort handling is sink-scoped and non-correctness-only
Best-effort exception handling MUST be limited to explicit diagnostics/I-O sink scope that does not mutate correctness-affecting state.

Normative behavior:
- Best-effort handlers SHOULD use narrow, expected exception classes where feasible.
- Best-effort handlers MUST emit warnings/counters for observability.
- Correctness-affecting artifact/state paths MUST NOT rely on blanket best-effort suppression.

#### Scenario: Sink-scoped diagnostic failure does not mask core failures
- **WHEN** a diagnostics-only sink (for example, log mirroring or debug dump write) encounters an I/O/runtime error
- **THEN** the sink emits an observable warning/counter and may continue best-effort
- **AND** correctness-path exceptions outside that sink still propagate and terminate as required.

### Requirement: Temporary mutable state is restored deterministically
When code temporarily mutates shared mutable state to perform encoding (for example, overwriting `template.system` for one sample), the system MUST restore the original value in a `finally` block.

Failure to restore MUST terminate the run with an explicit error to prevent state leakage across samples.

#### Scenario: Template system prompt does not leak across samples
- **WHEN** encoding overrides `template.system` for one sample
- **THEN** the original value is restored before encoding the next sample
- **AND** restoration failure stops the run with an explicit error.

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

