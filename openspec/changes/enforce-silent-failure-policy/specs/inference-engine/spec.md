# inference-engine Specification (delta)

## Purpose
Clarify inference error-handling behavior so that per-sample validation/parse failures are observable while unexpected internal exceptions fail fast (no silent partial successes).

This delta modifies only inference error-handling requirements; all other base `inference-engine` requirements remain unchanged.

## Requirements

### Requirement: Inference error reporting remains structured and sample-scoped
Inference-engine SHALL preserve structured, per-sample error reporting in output artifacts and summary counters.

Inference/eval inputs are operator-controlled and should be validated in advance. Therefore, any input-dependent validation/parse failure (invalid JSON line, missing/corrupt image, malformed geometry, wrong schema, etc.) MUST fail fast (terminate the run) rather than silently skipping.

If the implementation records a structured error entry for the failing sample, it MAY do so, but it MUST still terminate the run with a non-zero exit code.

Unexpected internal exceptions (anything not explicitly treated as an expected per-sample error) MUST terminate the run (fail fast) to prevent silent corruption of artifacts and metrics.

#### Scenario: Sample-level post-processing failure is reflected in structured errors
- **GIVEN** post-processing/validation fails for one sample (e.g., malformed geometry)
- **WHEN** the failure is encountered
- **THEN** the run terminates with a non-zero exit code
- **AND** the failure is surfaced with actionable diagnostics (sample identifier and reason).

#### Scenario: Unexpected internal exception terminates inference
- **GIVEN** an unexpected internal exception occurs during inference (not an enumerated expected per-sample error)
- **WHEN** the exception is raised
- **THEN** the run terminates with a non-zero exit code
- **AND** the failure is not silently converted into empty outputs or partial “success”.
