# silent-failure-policy Specification (delta)

## Purpose
Define a strict-by-default exception-handling policy so that core training/inference/evaluation behavior is reproducible and failures are observable, while allowing a narrow set of best-effort I/O sinks that do not affect correctness.

## Requirements

### Requirement: Core execution paths do not silently swallow unexpected exceptions
The system SHALL NOT suppress unexpected exceptions in core execution paths (dataset encoding, trainer steps, inference pipeline stages, evaluation metric computation).

Code MUST either:
- allow the exception to propagate (fail fast), OR
- catch only explicitly-enumerated exception types and either re-raise with added context or handle them as an explicitly-defined expected per-sample error.

The following are forbidden in core execution paths:
- `except Exception: pass`
- `except: pass` (bare except)
- `except BaseException: pass`
- `except Exception: continue` (or equivalent suppression via `return <default>` / `break`)
- catching `Exception` / bare `except:` and returning semantics-changing “safe” defaults (e.g., `0.0`, empty dicts/lists, placeholder artifacts) without structured error recording and counters.

#### Scenario: Dataset encoding error is surfaced in training
- **WHEN** a dataset raises an exception while encoding a sample for training
- **THEN** the run fails fast with a clear error message
- **AND** the exception is not discarded by a blanket catch-all
- **AND** training does not continue with a silently-skipped sample.

### Requirement: Expected per-sample errors are explicit and observable
When the system intentionally continues past a sample-scoped validation/parse error (in inference/eval only), it MUST:
- record a structured per-sample error entry (e.g., `errors=[...]`) in the relevant artifact, AND
- increment a run-level counter/metric for that error class, AND
- avoid emitting “fake success” outputs (e.g., empty predictions) without an accompanying error record.

Training MUST fail fast on expected per-sample errors arising from deterministic inputs (dataset encoding, cooked targets, GT).

Explicitly salvage-mode training subpaths that consume model-generated outputs (e.g., rollout parsing/matching) MAY continue past invalid model outputs per-sample, but MUST be observable (structured errors + counters) and MUST NOT suppress unexpected internal exceptions.

#### Scenario: Inference skips an invalid-sample with an error record
- **GIVEN** inference/eval processing
- **WHEN** a sample fails validation (e.g., malformed geometry)
- **THEN** the sample is skipped
- **AND** the output artifact records an `errors` entry for that sample
- **AND** run-level counters reflect the failure classification.

### Requirement: Best-effort handling is limited to non-correctness sinks
Best-effort exception handling is allowed ONLY for explicitly sink-scoped code that cannot affect correctness-affecting state (e.g., log tee mirroring or diagnostics/telemetry reporting).

Such handlers MUST:
- catch narrow, expected exception types when possible (e.g., `OSError`, `PermissionError` for file I/O) rather than blanket `Exception`,
- emit explicit diagnostics (at least once),
- never suppress exceptions outside the sink itself.

#### Scenario: Log tee I/O failure does not abort training
- **WHEN** the file logging tee fails to write to its mirror file
- **THEN** training continues without corrupting model state
- **AND** exceptions in non-I/O code paths are not suppressed.

### Requirement: Blanket suppression is forbidden by direct CI scanning
CoordExp SHALL NOT maintain exception-suppression registries/allowlists. Compliance MUST be enforced directly by CI scanning source files.

At minimum, the CI check in `tests/test_silent_failure_policy.py` MUST fail on:
- bare `except:` under `src/`,
- `except Exception` or `except BaseException` handlers under `src/` unless they immediately re-raise,
- broad exception handlers that suppress/continue/return defaults in core execution paths.

### Requirement: Temporary mutable state is restored deterministically
When code temporarily mutates shared mutable state to perform encoding (for example, overwriting `template.system` for one sample), the system MUST restore the original value in a `finally` block.

Failure to restore MUST terminate the run with an explicit error to prevent state leakage across samples.

#### Scenario: Template system prompt does not leak across samples
- **WHEN** encoding overrides `template.system` for one sample
- **THEN** the original value is restored before encoding the next sample
- **AND** restoration failure stops the run with an explicit error.
