# inference-engine Specification (delta)

## Purpose
Clarify inference/eval failure modes with a strict split between:
- operator-controlled input violations (fail fast),
- model-output/prediction parse+validation failures (continue-but-observable),
- unexpected internal exceptions (fail fast).

This delta modifies only inference error-handling requirements; all other base `inference-engine` requirements remain unchanged unless explicitly modified below.

## Requirements

### Requirement: Inference error reporting remains structured and sample-scoped
Inference-engine SHALL preserve structured, per-sample error reporting in output artifacts and run-level summary counters.

This delta modifies base behavior by making operator-controlled input violations fail-fast while preserving continue-but-observable behavior for model-output/prediction parse+validation failures.

### Requirement: Operator-controlled input violations MUST fail fast (no skip-and-continue)
Inference/eval inputs are operator-controlled and MUST be validated in advance (prefer a preflight pass). Any input-dependent validation/parse/contract failure MUST terminate the run with a non-zero exit code (fail fast); the system MUST NOT skip the sample and continue.

Examples (non-exhaustive):
- invalid JSON line,
- wrong schema / missing required keys,
- missing/corrupt image or unreadable image path,
- missing `width`/`height`,
- malformed GT geometry / wrong format,
- mode/coord_mode mismatch against GT.

If the implementation records a structured error entry for the failing sample, it MAY do so, but it MUST still terminate the run non-zero.

This requirement supersedes base scenarios that say “an error is recorded and processing continues” for operator-controlled input violations (e.g., missing height/width).

#### Scenario: Missing height terminates inference
- **GIVEN** inference/eval processing with operator-controlled GT inputs
- **WHEN** an input record is missing a required `height` (or `width`)
- **THEN** the run terminates with a non-zero exit code
- **AND** the failure is surfaced with actionable diagnostics (sample identifier and reason).

### Requirement: Model-output/prediction parse+validation failures MUST be continue-but-observable
Failures attributable to model-generated outputs (prediction parsing, schema validation, geometry validation) MUST be treated as per-sample errors:
- record a structured per-sample error entry,
- increment run-level counters by error code/class,
- drop only the invalid predicted objects (or set `pred=[]`) rather than emitting “fake success” outputs,
- continue processing subsequent samples.

Examples (non-exhaustive):
- invalid/truncated prediction JSON / CoordJSON,
- malformed predicted geometry (odd point count, non-numeric coord, etc.),
- out-of-range coord tokens in predictions.

#### Scenario: Out-of-range prediction token is skipped and processing continues
- **GIVEN** inference in `coord` mode
- **WHEN** a prediction contains a coord value outside 0–999
- **THEN** invalid predictions are dropped for that sample, a structured error is recorded, and processing continues for subsequent samples.

### Requirement: Unexpected internal exceptions MUST terminate (fail fast)
Unexpected internal exceptions (including CUDA out-of-memory) MUST terminate the run (fail fast) to prevent silent corruption of artifacts and metrics.

The system MUST NOT convert unexpected internal exceptions into per-sample errors and continue.

If feasible, implementations SHOULD annotate the current sample with a structured error entry and increment summary counters before re-raising.

#### Scenario: CUDA out-of-memory terminates inference
- **GIVEN** an unexpected internal exception occurs during inference (e.g., CUDA out-of-memory)
- **WHEN** the exception is raised
- **THEN** the run terminates with a non-zero exit code
- **AND** the failure is not silently converted into empty outputs or partial “success”.

### Requirement: Generation failures are treated as internal/runtime violations
If generation fails for a sample such that a valid `pred_text` cannot be produced (whether the backend raises or returns a failure), the run MUST terminate non-zero.

Generation failures are treated as internal/runtime/config violations and MUST NOT be handled as continue-but-observable per-sample errors in inference/eval.
