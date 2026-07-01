# trainer-metrics-components Specification (delta: explicit Stage-2 reduction contracts)

## Purpose
Extend the trainer metrics contract so Stage-2 optimizer-step aggregation uses explicit reduction ownership rather than duplicated string heuristics.

## Requirements

## ADDED Requirements

### Requirement: Stage-2 optimizer-step metrics MUST declare explicit reduction semantics
When a Stage-2 trainer emits metrics that participate in optimizer-step aggregation, the reduction semantics for those metrics MUST be explicit and shared across local aggregation and DDP aggregation.

Normative behavior:
- Each Stage-2 optimizer-step metric family MUST declare:
  - local aggregation behavior,
  - cross-rank aggregation behavior,
  - and whether the key is a current-step value or a carry-forward snapshot.
- Local pending-log aggregation and global DDP aggregation MUST consume the same declared semantics for the same key.
- The system MUST NOT rely on independent string-prefix or string-suffix heuristic tables in multiple reducers to infer the same metric behavior.

#### Scenario: New metric key does not need duplicate reducer heuristics
- **GIVEN** a new Stage-2 metric key is introduced for optimizer-step logging
- **WHEN** the key is aggregated locally and across ranks
- **THEN** both aggregation sites use the same declared reduction semantics
- **AND** the implementation does not require separate heuristic updates in multiple reducer functions.

### Requirement: Carry-forward Stage-2 snapshots MUST be explicit in operator-facing logs
When Stage-2 logs preserve a last-seen metric value from a previous channel or step, the emitted key MUST make that snapshot behavior explicit.

Normative behavior:
- A carry-forward metric MUST be distinguishable from a current-step metric by a dedicated snapshot namespace or equivalent explicit contract.
- Live current-step namespaces, including `rollout/*`, MUST retain their existing sparse-emission behavior and MUST NOT be repurposed to carry stale snapshot values on steps where the live metric was not observed.
- Operator-facing logs MUST NOT present stale carry-forward values as if they were newly observed current-step measurements.

#### Scenario: Last-seen channel metric is distinguishable from current-step output
- **GIVEN** a training step where Channel-A did not run but its last-seen metric is still surfaced for continuity
- **WHEN** the trainer emits the optimizer-step log payload
- **THEN** the carry-forward value is clearly identified as a snapshot
- **AND** operators can distinguish it from metrics computed on the current step.

### Requirement: Disabled best-effort diagnostics MUST emit explicit health signals
When a best-effort diagnostic disables itself for the remainder of a run after an unexpected failure, the trainer MUST emit an explicit health signal describing that disabled state.

Normative behavior:
- The health signal MAY be emitted as canonical diagnostic metrics, a canonical artifact field, or both.
- The signal MUST allow operators to distinguish “diagnostic observed zero issue” from “diagnostic no longer running.”
- The signal MUST NOT silently change the training objective.

#### Scenario: Diagnostic disablement is visible beyond a one-time warning
- **GIVEN** a best-effort diagnostic throws an unexpected exception and disables itself
- **WHEN** later operator-facing metrics or run artifacts are inspected
- **THEN** there is an explicit signal that the diagnostic is disabled
- **AND** the run does not rely only on the original warning line to communicate that state.
