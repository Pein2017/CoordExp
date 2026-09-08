## Purpose

Prevent one physical instruction file from being rendered repeatedly through user-global and nested project aliases while preserving intentional directory-scoped guidance.

## ADDED Requirements

### Requirement: Canonical instruction aliases are rendered once

The instruction projection SHALL render at most one instruction contribution when the user-global `AGENTS.md` and a project instruction candidate resolve to the same canonical file target. The retained contribution SHALL preserve the user-global precedence and model-facing user-global path.

#### Scenario: Nested symlink aliases the user-global file

- **WHEN** a project candidate is reached through a successful filesystem touch and resolves to the same canonical target as the loaded user-global instruction file
- **THEN** the projection SHALL NOT append a second instruction contribution for that project candidate

#### Scenario: Alias appears during initial discovery

- **WHEN** initial discovery encounters a project candidate that resolves to the same canonical target as the user-global file
- **THEN** the rendered baseline SHALL contain only the user-global contribution for that target

#### Scenario: An alias later points to a different target

- **WHEN** a previously suppressed project candidate resolves to a different canonical target
- **THEN** that candidate SHALL become eligible for its normal project-scoped instruction contribution

### Requirement: Independent directory scopes remain distinct

The instruction projection SHALL NOT collapse instruction files solely because their trimmed contents are equal when they resolve to different canonical targets or represent different directory scopes.

#### Scenario: Different files share identical content

- **WHEN** user-global or project instruction files in different scopes have identical content but resolve to different targets
- **THEN** each file SHALL retain its existing scope and precedence behavior

#### Scenario: Canonical identity is unavailable

- **WHEN** the filesystem provider cannot provide a canonical target identity for a candidate
- **THEN** the loader SHALL retain the existing logical-path behavior rather than suppressing the candidate based on an unverified alias
