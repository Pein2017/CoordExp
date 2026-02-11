# stage2-ab-training Spec Delta

This is a delta spec for change `refactor-src-modernization`.

## ADDED Requirements

### Requirement: Stage-2 AB consumes rollout helpers through public contracts only
The Stage-2 AB capability SHALL consume rollout parsing/matching/packing helpers only through a public rollout-matching contract module.
It MUST NOT import underscore-prefixed symbols from trainer implementation files.

#### Scenario: Private rollout helper removal does not break Stage-2 imports
- **WHEN** private underscore-prefixed helpers are removed from the rollout trainer implementation file
- **THEN** Stage-2 AB still imports successfully via public contract modules
- **AND** training initialization does not fail due to missing private symbols.

### Requirement: No-private-import boundary is regression-guarded
The Stage-2 AB capability SHALL include a regression guard that detects imports from underscore-prefixed rollout symbols and fails validation when such imports reappear.
This guard MUST use AST import inspection (test or static check) rather than regex text matching so formatting/comment changes do not create false signals.
The guard MUST run in routine validation for this capability.
The guard scope MUST cover the full Stage-2 AB capability surface:
- `src/trainers/stage2_ab_training.py`
- `src/trainers/stage2_ab/**/*.py`

#### Scenario: Validation fails when a private rollout helper import is reintroduced
- **GIVEN** any Stage-2 AB source file in the guarded surface imports an underscore-prefixed rollout helper
- **WHEN** capability validation checks execute
- **THEN** validation fails with a boundary-violation diagnostic
- **AND** the regression is caught before merge.

### Requirement: Stage-2 AB trainer is decomposed into orchestrator plus owned components
The Stage-2 AB trainer SHALL be structured as an orchestration surface that delegates scheduling, async queue management, and channel execution to dedicated components.
The decomposition MUST preserve deterministic channel selection and existing Stage-2 contract semantics.

#### Scenario: Scheduling policy changes are isolated from trainer orchestration
- **GIVEN** a change to channel scheduling policy implementation
- **WHEN** Stage-2 AB training is run with unchanged YAML semantics
- **THEN** only scheduling component modules require modification
- **AND** the top-level trainer orchestration entrypoint remains interface-compatible.

### Requirement: Stage-2 critical invariants fail fast with contextual diagnostics
Stage-2 AB SHALL classify critical runtime invariants (queue feasibility, version-window gating, sync boundaries, required batch fields) as fail-fast conditions.
Unexpected failures on critical invariants MUST raise errors with step/channel/version context.
Best-effort diagnostics MAY continue under guarded warning paths.

#### Scenario: Async queue invariant violation raises actionable error
- **GIVEN** async mode is enabled and queue state violates required invariants for a scheduled Channel-B step
- **WHEN** Stage-2 attempts to execute that step
- **THEN** training raises with contextual diagnostics including step kind and queue/version state
- **AND** the failure is not silently suppressed.
