# coordexp-swift-research-probe-infra-base Specification

## Purpose

Define the smallest reusable execution-mechanics base for research probes so
new directions can compose proven identity, durability, admission, and
inference capabilities without inheriting a generic runner or another
experiment's scientific semantics.

## Requirements

### Requirement: Smallest sufficient capability profile

The infra-base SHALL let each producer select only the existing mechanics its
execution actually needs.  An immutable one-shot producer MUST be able to use
strict validation and exclusive publication without creating a journal or
admission.  A multi-item producer that requires durable continuation MAY add
the existing evidence journal.  A producer that requires a mechanics-only
CPU-to-model launch gate MAY additionally use the existing probe admission.
An inference-backed producer MAY use the existing public frontend, launch,
session, request, result, and runtime-receipt contracts when those semantics
fit.

Selecting one capability MUST NOT silently enable another, create a global
coordinator, assign retry policy, or change the producer's output schema.

#### Scenario: One-shot producer chooses the leaf profile

- **WHEN** a probe has one immutable strict-JSON result and no continuation or
  model-launch admission requirement
- **THEN** it can validate and publish that result exclusively without a
  journal, admission root, phase plan, or generated runner

#### Scenario: Resumable producer adds only a journal

- **WHEN** one producer has a fixed work-item plan and needs independently
  durable progress across process attempts
- **THEN** it can add the existing journal while keeping work-item meaning,
  retry authorization, scheduling, and terminal interpretation caller-owned

#### Scenario: Custom differentiable runtime does not fit inference

- **WHEN** a probe needs direct differentiable model state rather than the
  existing deterministic decode-session contract
- **THEN** the infra-base does not wrap or mislabel that runtime as ordinary
  inference, and the probe keeps the unmatched runtime mechanics local

### Requirement: Producer topology remains caller-owned

The infra-base SHALL compose mechanics per independently scheduled producer.
It MUST NOT require an experiment-global journal, lock, stage order, worker
registry, or terminal barrier.  A caller MAY bind relationships between
producer artifacts as strict input identities, but the shared layer MUST NOT
turn those relationships into a scheduler or DAG.

#### Scenario: Independent cell producers remain independent

- **WHEN** several cells consume a common immutable plan but run as separate
  operating-system producers
- **THEN** each cell can use its own output or journal root, one failed cell
  does not mutate another cell's evidence, and no shared append order is
  required

#### Scenario: Distributed producer owns its rank barrier

- **WHEN** one probe producer uses multiple ranks and caller-defined stage
  barriers
- **THEN** rank groups, collectives, stage transitions, and terminal election
  remain caller-owned while shared artifact leaves retain their declared
  identity and publication behavior

### Requirement: Shared evidence remains mechanics-only

Every infra-base profile SHALL preserve the existing distinction between
mechanical integrity and scientific validity.  Shared validation MAY attest
strict values, exact identity, exclusive durability, plan completeness,
attempt history, or launch-gate closure.  It MUST NOT define or infer cohort,
intervention, owner matching, objective, optimizer meaning, metric, threshold,
scientific outcome, claim scope, continuation decision, or stop rule.

#### Scenario: Mechanically complete negative result

- **WHEN** a producer publishes complete, identity-valid evidence whose opaque
  payload reports a null or harmful scientific outcome
- **THEN** the shared layer may report mechanical completion but emits no
  scientific success, promotion, retry, or continuation decision

#### Scenario: Historical record is used as a specimen

- **WHEN** a retired probe tag or returned research record is inspected to
  characterize repeated mechanics
- **THEN** it remains historical evidence and is not treated as a live
  consumer, resumed execution, or completion of an unfinished route

### Requirement: Canonical public mechanics are discoverable

The stable operations needed by the capability profiles SHALL have documented
public import paths owned by the existing artifact and inference modules.  A
canonical operator guide MUST map each profile to its public owner, caller
obligations, failure meaning, and cheapest acceptance check.  New probe code
MUST NOT need to import orchestration-private helpers or copy an exclusive
publication algorithm merely to use these stable mechanics.

The guide SHALL link rather than duplicate the canonical branch/worktree
policy and the existing journal, admission, and inference contracts.

#### Scenario: New one-shot probe finds the publication owner

- **WHEN** a direction needs to publish one strict immutable result
- **THEN** the canonical guide identifies one public exclusive-publication
  operation, its occupied-path failure behavior, and the caller-owned schema
  boundary without offering an overwrite fallback

#### Scenario: Existing deep owner remains authoritative

- **WHEN** a public import is added or reorganized for discoverability
- **THEN** serialization, journaling, admission, or inference behavior remains
  implemented and tested at its existing deep owner rather than being copied
  into a pass-through execution framework

### Requirement: New shared layers require live cross-direction evidence

Any proposal to add a coordinator, phase DSL, trainable-model session,
optimizer or RNG transaction, checkpoint abstraction, monitor, or lifecycle
automation SHALL identify at least two live cross-direction consumers with the
same caller-visible contract.  It MUST state the duplicated mechanics, prove
that direct composition of existing owners is insufficient, preserve each
consumer's execution topology, and include the cheapest counterexample that
would falsify the common seam.

Repeated use within one direction and a retired historical specimen MUST NOT
alone satisfy this promotion threshold.

#### Scenario: One live direction plus one retired lineage

- **WHEN** a proposed abstraction is evidenced only by repeated callers in one
  live direction and a materially different retired lineage
- **THEN** the abstraction is deferred and the shared base exposes only the
  already proven common mechanics

#### Scenario: A later direction proves an identical seam

- **WHEN** a second live direction duplicates the same mechanics, scheduling,
  artifact, and failure contract and a bounded comparison rejects direct
  composition as insufficient
- **THEN** a separate OpenSpec change may propose the smallest new shared layer
  without changing this infra-base's scientific-neutrality or lifecycle rules

### Requirement: Integration never performs probe lifecycle actions

The infra-base and its acceptance checks SHALL NOT create, move, unlock,
retire, merge, tag, or delete research worktrees or branches.  Lifecycle
routing remains owned by the canonical branch/worktree policy and requires its
separate authorization gates.

#### Scenario: Infra-base validation runs in the integration lane

- **WHEN** the capability is implemented or verified in the fixed
  research-probe infrastructure worktree
- **THEN** validation changes only the authorized source, tests, specs, and
  documentation for this change and leaves every probe worktree, branch, tag,
  external artifact, and running process unchanged
