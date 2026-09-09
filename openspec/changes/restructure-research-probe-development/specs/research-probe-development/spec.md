## Purpose

Let maintained research directions run independently from one research base while preserving searchable knowledge and recoverable historical implementations when directions retire.

## ADDED Requirements

### Requirement: Independent direction execution

Retained experimental producers SHALL have documented direction-owned package entries and explicit dependencies available in the same checkout or declared installed dependencies. Shared code MUST NOT depend on direction packages. Executable imports MUST NOT require another temporary worktree or a historical research-script namespace. Explicit external data, model and artifact locations SHALL remain supported without treating them as code dependencies.

#### Scenario: Direction starts from the research base

- **WHEN** a retained direction is checked out independently with declared dependencies and fixture inputs
- **THEN** its documented package import, offline/preflight entry and targeted tests run without sibling worktree code

#### Scenario: Source/Rweak consumes former COCO helpers

- **WHEN** the retained row-cross runner or reducer needs checkpoint validation, parsing or owner assignment
- **THEN** it uses its direction-owned scientific checks and same-checkout public operations without importing the COCO worktree

### Requirement: Direction profiles keep separate scientific meanings

The initial maintained direction set SHALL cover differentiable DORA owner learning, Source/Rweak row crossing, logit-lens intervention and Human13 finite-panel intervention. A direction SHALL be able to contain multiple protocol/configuration profiles without a package per run or global experiment registry. Each entry SHALL identify its owning research record and preserve profile-specific objectives, populations, metrics, stopping choices and artifact meaning.

Unselected historical producers SHALL remain recoverable without being automatically repackaged as maintained directions. Operational agent benchmarks MUST remain distinct from the scientific producer and its result interpretation.

#### Scenario: Two learning objectives inhabit one direction

- **WHEN** coordinate-credit and full-action profiles reuse a direction's execution operations
- **THEN** their selection and reduction remain explicit and separately testable rather than being silently unified by a common profile default

#### Scenario: Completed historical solver is not migrated

- **WHEN** a hash-bound N256 solver is selected for historical preservation rather than maintenance
- **THEN** its original sources/configs/dependencies and required evidence remain recoverable, and the default checkout does not falsely advertise a migrated compatible solver

### Requirement: Knowledge survives retirement

Before a direction is retired, its unique accepted research records SHALL be reachable from question-oriented navigation in the research base. Current synthesis SHALL distinguish observations, bounded interpretation, incomplete/invalid execution and archival status. Original facts and evidence locations MUST remain traceable; incompatible findings MUST NOT be pooled or replaced merely by recency.

#### Scenario: Two directions address the same question

- **WHEN** their results use different populations, metrics or interventions
- **THEN** the synthesis explains those differences and links original records instead of copying both branch summaries as current truth or combining incompatible outcomes

#### Scenario: Partial and selected-panel evidence is integrated

- **WHEN** an incomplete null solve, CPU-only preparation or selected-panel contrast enters navigation
- **THEN** the corresponding evidence limit remains explicit and is not promoted to a completed comparison or population result

### Requirement: Preserve content before removing worktrees

Retirement SHALL preserve relevant committed, modified, untracked and referenced ignored content, including required source/config dependencies and necessary executed evidence. Historical source identity MUST remain recoverable without rewriting receipts. The retirement check SHALL verify necessary evidence is accessible without the retiring directory and SHALL reject removal while a relevant live holder or unresolved content remains.

#### Scenario: Latest result is untracked

- **WHEN** a completed direction has relevant untracked material beyond HEAD
- **THEN** tagging HEAD alone does not satisfy preservation and retirement remains incomplete until that content is saved or explicitly dispositioned

#### Scenario: Hash-bound historical producer is refactored

- **WHEN** a retained implementation changes source layout or bytes
- **THEN** the original version and receipt keep their identity, the new implementation is identified separately, and source recovery is not reported as verified model replay

### Requirement: One permanent research base

The canonical research base SHALL remain the existing protected `research-probes` worktree. Routine shared development SHALL occur there; larger or conflicting work SHALL use temporary worktrees that return accepted changes and retire. The former `research-probe-infras` lane SHALL retire only after its valid content and dependencies are handled. Lifecycle work MUST leave the excluded bridge-cache worktree, production worktrees, shared agent/runtime configuration and remote refs untouched.

#### Scenario: Infrastructure lane is ready to retire

- **WHEN** the former infra lane satisfies preservation, integration and no-live-holder checks
- **THEN** it can be unlocked and retired under the accepted lifecycle decision without unlocking or replacing `research-probes`
