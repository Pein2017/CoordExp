## Purpose

Keep local Pi CLI and Web on one official installation while preserving active work and enabling indexed access to official and maintained source.

## ADDED Requirements

### Requirement: Canonical local managed installation
CLI and Web SHALL use the same selected official release from one installation under `codex-tools/pi-core`. Shared profile, credentials, sessions and local extension selection MUST remain preserved across relocation and core updates.

#### Scenario: New CLI and Web start
- **WHEN** CLI or Web starts after the selected release changes
- **THEN** it resolves that release from the canonical installation rather than an independent package copy
- **AND** the existing shared profile and local compaction package remain selected

### Requirement: Idle Web adoption
The managed local Web service SHALL detect a changed official selection and adopt it automatically after active work finishes. It MUST defer while a session start, command, SDK run, queued input, shutdown or terminal shell is active.

#### Scenario: Update during a task
- **WHEN** the installation changes while Web is executing a task
- **THEN** Web continues that task on its loaded release
- **AND** adopts the selected release after its sessions and terminal shells become idle

#### Scenario: Work races with adoption
- **WHEN** Web begins idle adoption and a new session, mutating command or terminal creation arrives before restart
- **THEN** that new work is rejected with a retryable update-in-progress error before it mutates state or executes

#### Scenario: Restart preserves history
- **WHEN** idle Web adopts the selected release
- **THEN** the service resumes on the same address with the new SDK
- **AND** persisted session identity and history remain available without rewriting them

#### Scenario: Invalid release or shutdown failure
- **WHEN** release validation, session shutdown or binding activation fails
- **THEN** Web logs the failure and does not exit for adoption
- **AND** new work admission reopens with the retained release

### Requirement: Indexed local source reference
Each active local repository SHALL own its CodeGraph index, including official Pi reference source and maintained Web source. Indexes MUST exclude runtime, dependency and private profile data. Source reference identity MUST be explicit and independent of the selected runtime version. A Pi-only parent index MUST NOT catch queries from unrelated repositories.

#### Scenario: Cross-module lookup
- **WHEN** a user explores Pi compaction and Web session startup using the respective repository indexes
- **THEN** results come from the explicitly selected official core and maintained Web repositories
- **AND** installation artifacts and credentials are excluded
