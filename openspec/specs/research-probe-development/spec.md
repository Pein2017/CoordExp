# research-probe-development Specification

## Purpose

Let maintained research directions run independently from one research base while preserving searchable knowledge and recoverable historical implementations when directions retire.

## Requirements

### Requirement: Independent direction execution

Retained experimental producers SHALL have documented direction-owned package entries and explicit dependencies available in the same checkout or declared installed dependencies. Shared code MUST NOT depend on direction packages. Executable imports MUST NOT require another temporary worktree or a historical research-script namespace. Explicit external data, model and artifact locations SHALL remain supported without treating them as code dependencies.

#### Scenario: Direction starts from the research base

- **WHEN** a retained direction is checked out independently with declared dependencies and fixture inputs
- **THEN** its documented package import, offline/preflight entry and targeted tests run without sibling worktree code

#### Scenario: Source/Rweak consumes former COCO helpers

- **WHEN** the retained row-cross runner or reducer needs checkpoint validation, parsing or owner assignment
- **THEN** it uses its direction-owned scientific checks and same-checkout public operations without importing the COCO worktree

### Requirement: Direction profiles keep separate scientific meanings

The maintained direction set SHALL be selected by supported entry points, actual
consumer dependencies and explicitly retained reusable baselines, rather than an
immutable list of all historical producers. A direction SHALL be able to contain
multiple protocol/configuration profiles without a package per run or a global
experiment registry. Each supported entry SHALL identify its research owner and
preserve profile-specific objectives, populations, metrics, stopping choices and
artifact meaning. Ordinary module files are not automatically command entries.

Unselected historical producers SHALL NOT be repackaged or kept as forwarding
aliases solely to make old launch commands executable at the latest revision.
Their necessary methods, parameters and result interpretation SHALL remain with
the research owner. Existing receipt-required source captures and frozen data
SHALL retain their identities; this does not require restoring a retired
migration-time source library. Operational agent benchmarks MUST remain distinct
from scientific producers and their result interpretation.

#### Scenario: A profile moves out of the production directory

- **WHEN** a maintained direction loads the same valid source configuration from its package
- **THEN** shared resolution and value validation preserve its effective values and fingerprint without enabling debug mode; the production loader retains its authoring restrictions

#### Scenario: Two learning objectives inhabit one direction

- **WHEN** coordinate-credit and full-action profiles reuse a direction's execution operations
- **THEN** their selection and reduction remain explicit and separately testable instead of being unified by a common default

#### Scenario: Completed historical solver is not maintained

- **WHEN** a closed solver has no supported execution consumer and is not an explicitly retained baseline
- **THEN** its disposable implementation can retire after method and reference checks, while its research record and required evidence remain discoverable without a claim that today's checkout can replay it

#### Scenario: Non-novel operation is still required

- **WHEN** a supported probe consumes a fitting, adapter or input operation
- **THEN** the operation remains supported or receives a behavior-verified replacement before deletion, irrespective of its novelty

### Requirement: Knowledge survives retirement

Before a direction is retired, its unique accepted research records SHALL be
reachable from question-oriented navigation and the complete experiment catalog.
The catalog SHALL retain every current-schema state owner, including closed and
paused units. The frontier index SHALL link the catalog and select the current
reading route without requiring a direct home-page link to every state. Missing,
duplicate or orphan state owners and invalid selected links MUST still fail
validation. The catalog SHALL NOT infer scientific outcome or execution authority.

Current synthesis SHALL distinguish observations, bounded interpretation,
incomplete/invalid execution and historical status. Original facts and evidence
locations MUST remain traceable; incompatible findings MUST NOT be pooled or
replaced merely by recency. Memory and handoff text MUST NOT own a second live
state ledger. Decision-relevant CPU verification SHALL be retained where selected
for continuing support; temporary reporting shells do not thereby acquire a
permanent execution-support promise.

#### Scenario: Two directions address the same question

- **WHEN** results use different populations, metrics or interventions
- **THEN** the synthesis explains the differences and links the original records rather than pooling incompatible outcomes or maintaining duplicate current summaries

#### Scenario: Partial evidence is integrated

- **WHEN** an incomplete null solve, CPU preparation or selected-panel contrast enters navigation
- **THEN** its evidence limit remains explicit and is not promoted to a completed comparison or population result

#### Scenario: A closed state leaves the frontier

- **WHEN** a closed unit remains completely catalogued but is no longer selected on the frontier page
- **THEN** knowledge validation accepts it and users can still reach its state, protocol and accepted result through the catalog

#### Scenario: Complete discovery route is missing

- **WHEN** the frontier has no actual local Markdown link to the experiment catalog, or a current-schema state has no catalog owner
- **THEN** knowledge validation fails instead of silently dropping discoverability

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
