# silent-failure-policy Specification

## Purpose
Define the exception-handling policy for CoordExp so that core training/inference/evaluation behavior is reproducible and failures are observable, while allowing a narrow set of best-effort I/O sinks.

## ADDED Requirements

### Requirement: Core execution paths do not swallow unexpected exceptions
The system SHALL NOT suppress unexpected exceptions in core execution paths (dataset encoding, trainer steps, inference pipeline stages, evaluation). Code MUST either raise the exception or catch only explicitly enumerated exception types and emit an actionable error message.

Blanket `except Exception: pass` is forbidden in core execution paths.

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

#### Scenario: Log tee I/O failure does not abort training
- **WHEN** the file logging tee fails to write to its mirror file
- **THEN** training continues without corrupting model state
- **AND** exceptions in non-I/O code paths are not suppressed.

### Requirement: Temporary mutable state is restored deterministically
When code temporarily mutates shared mutable state to perform encoding (for example, overwriting `template.system` for one sample), the system MUST restore the original value in a `finally` block.

Failure to restore MUST terminate the run with an explicit error to prevent state leakage across samples.

#### Scenario: Template system prompt does not leak across samples
- **WHEN** encoding overrides `template.system` for one sample
- **THEN** the original value is restored before encoding the next sample
- **AND** restoration failure stops the run with an explicit error.
