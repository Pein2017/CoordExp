# silent-failure-policy Specification (Delta)

## MODIFIED Requirements

### Requirement: Core execution paths fail fast on unexpected exceptions
Core execution paths under `src/` (dataset encoding, trainer steps, inference pipeline stages, evaluation metric computation) MUST NOT silently suppress unexpected exceptions.

Normative behavior:
- Unexpected internal exceptions MUST propagate and terminate the run.
- Catch handlers in core paths MUST be narrow and explicit; blanket suppression patterns are forbidden.
- In core paths, these patterns MUST NOT be used as silent suppression:
  - `except Exception: pass`
  - `except: pass`
  - `except BaseException: pass`
  - blanket `except Exception` with `continue`, `break`, or semantics-changing default `return`.

#### Scenario: Dataset encoding error is fail-fast
- **WHEN** dataset encoding raises an unexpected exception during training
- **THEN** the exception propagates and the run terminates non-zero
- **AND** training does not continue with silent sample skipping.

### Requirement: Operator-controlled input violations are strict contracts
Operator-controlled input violations MUST fail fast and MUST NOT be handled as continue-and-skip behavior.

Normative behavior:
- Input contract failures for deterministic inputs (training data contracts, inference/eval JSONL/image/schema/geometry contracts) MUST terminate the run non-zero.
- Implementations MAY emit actionable diagnostics before raising, but MUST still raise.

#### Scenario: Inference preflight rejects invalid deterministic input
- **GIVEN** inference/eval processing on operator-provided JSONL and images
- **WHEN** a record violates required schema or image resolvability/readability
- **THEN** inference terminates non-zero before generation/evaluation continues
- **AND** diagnostics include sample context and reason.

### Requirement: Continue-but-observable is restricted to explicit model-output consumers
Continue-and-salvage behavior MUST be limited to explicit model-output consumer paths and MUST remain observable.

Normative behavior:
- For invalid model-generated outputs (for example prediction parse/validation failures), implementations MAY continue per sample only in explicit model-output consumer paths.
- Such handling MUST record structured per-sample error information and update run-level counters.
- This carve-out MUST NOT be used for operator-controlled input violations or unexpected internal exceptions.

#### Scenario: Invalid model prediction is recorded and processing continues
- **GIVEN** prediction parsing in an explicit model-output consumer path
- **WHEN** model output is malformed/truncated for a sample
- **THEN** the sample emits structured error metadata and error counters are incremented
- **AND** subsequent samples continue processing.

### Requirement: Best-effort handling is sink-scoped and non-correctness-only
Best-effort exception handling MUST be limited to explicitly scoped diagnostics/I-O sinks that do not mutate correctness-affecting state.

Normative behavior:
- Best-effort sink handlers SHOULD catch narrow, expected exception classes whenever feasible.
- Best-effort sink handlers MUST emit explicit diagnostics (warnings or counters).
- Best-effort sink handlers MUST NOT suppress exceptions outside the sink scope.
- Correctness-affecting artifacts and state transitions (for example canonical prediction artifacts, evaluator metrics artifacts, model/trainer state updates) MUST NOT rely on blanket best-effort suppression.

#### Scenario: Log mirroring failure does not mask core failures
- **WHEN** a log tee write fails due to I/O error
- **THEN** log mirroring may be degraded with warning diagnostics
- **AND** unrelated correctness-path exceptions still propagate normally.

### Requirement: Policy compliance is enforced by direct source scanning
Policy compliance MUST be enforced by direct CI scanning of source code rather than suppression registries/allowlists.

Normative behavior:
- Tier 0 blocking checks MUST fail on blanket-pass suppression patterns under `src/`.
- Tier 1 checks MUST detect blanket `continue`/`break`/default-return suppression under `src/`.
- Violations MUST include file/line evidence to support deterministic remediation.

#### Scenario: CI blocks newly introduced blanket suppression
- **WHEN** a source change introduces `except Exception: pass` in `src/`
- **THEN** policy tests fail with file/line evidence
- **AND** the run is blocked until the suppression pattern is removed or narrowed.
