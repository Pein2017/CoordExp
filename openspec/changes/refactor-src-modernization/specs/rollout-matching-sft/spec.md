# rollout-matching-sft Spec Delta

This is a delta spec for change `refactor-src-modernization`.

## ADDED Requirements

### Requirement: Rollout-matching exposes stable submodule contracts by concern
The rollout-matching capability SHALL expose stable public contracts for parsing, matching, packing, and backend orchestration via dedicated submodules.
The trainer-facing module MAY provide compatibility re-exports during migration, but behavior ownership MUST live in the dedicated submodules.

#### Scenario: Shared parsing contract is importable without trainer class dependency
- **WHEN** a consumer imports rollout parsing contracts for validation/testing
- **THEN** it can do so without importing the trainer class implementation
- **AND** parsed output contracts remain stable for downstream use.

### Requirement: Rollout backend orchestration uses a backend interface contract
Rollout backend selection and synchronization SHALL be mediated through a backend interface contract rather than inline trainer branches.
Supported backend implementations MUST preserve existing rollout semantics and output fields required by strict parsing/matching.

#### Scenario: Backend implementation swap preserves rollout contract
- **GIVEN** the same rollout config semantics and sample inputs
- **WHEN** backend implementation is switched through the backend interface
- **THEN** returned rollout payload fields required by parsing/matching are preserved
- **AND** trainer-side supervision construction remains valid.

### Requirement: Post-rollout packing scheduling is reusable and deterministic
Post-rollout packing/window scheduling SHALL be implemented in reusable helpers consumable by rollout and Stage-2 paths.
Given identical segment inputs and ordering, helper outputs MUST be deterministic.

#### Scenario: Shared packing helper yields deterministic selection
- **GIVEN** identical segment metadata and insertion order
- **WHEN** the shared packing scheduler runs twice
- **THEN** it yields the same selected segment set and order in both runs.
