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
