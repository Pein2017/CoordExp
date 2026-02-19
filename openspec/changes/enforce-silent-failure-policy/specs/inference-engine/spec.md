# inference-engine Specification (delta)

## Purpose
Clarify inference error-handling behavior so that per-sample validation/parse failures are observable while unexpected internal exceptions fail fast (no silent partial successes).

This delta modifies only inference error-handling requirements; all other base `inference-engine` requirements remain unchanged.

## Requirements

### Requirement: Inference error reporting remains structured and sample-scoped
Inference-engine SHALL preserve structured, per-sample error reporting in output artifacts and summary counters.

When inference/eval intentionally continues past an expected per-sample validation/parse failure, the failure MUST map to an explicit sample error entry rather than a silent skip.

Unexpected internal exceptions (anything not explicitly treated as an expected per-sample error) MUST terminate the run (fail fast) to prevent silent corruption of artifacts and metrics.

#### Scenario: Sample-level post-processing failure is reflected in structured errors
- **GIVEN** post-processing/validation fails for one sample (e.g., malformed geometry)
- **WHEN** inference continues for the remaining samples
- **THEN** the failed sample includes a structured error entry
- **AND** summary counters include the failure classification.

#### Scenario: Unexpected internal exception terminates inference
- **GIVEN** an unexpected internal exception occurs during inference (not an enumerated expected per-sample error)
- **WHEN** the exception is raised
- **THEN** the run terminates with a non-zero exit code
- **AND** the failure is not silently converted into empty outputs or partial “success”.
