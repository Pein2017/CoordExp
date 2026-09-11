# coordexp-infras-natural-boundary-support-shards Specification

## Purpose
Provide durable, cost-balanced execution mechanics for the sealed
natural-boundary support-completion contexts while preserving the existing
scientific plan and downstream shard-receipt contract.

## Requirements

### Requirement: The sealed support plan remains logical authority
The capability SHALL consume the validated natural-boundary support-completion
plan as an immutable logical work-item plan. Every planned context identifier,
context identity, declared scalar-equivalent-forward cost, legacy receipt shard,
and ordered candidate set MUST remain unchanged. The capability MUST NOT change
the research question, support rule, denominator, checkpoint, prefix, candidate
set, outcome schema, calibration, or interpretation.

The capability SHALL bind the exact plan bytes, consumer bytes, adapter bytes,
model and runtime identity, inference configuration, and caller-owned execution
identity before model work. Any bound identity drift MUST reject continuation
and require a fresh journal root.

#### Scenario: Exact logical plan is admitted
- **WHEN** the accepted 200-context plan and every declared execution identity
  match their bound digests
- **THEN** the capability admits exactly those 200 context work items without
  rewriting the plan or its sealed source artifacts

#### Scenario: Consumer or execution identity changes
- **WHEN** the consumer source, adapter source, model, config, runtime policy,
  plan bytes, or another bound identity differs from an existing journal
- **THEN** continuation fails before model loading or context execution

### Requirement: Each support context is independently durable
Each support context SHALL be one journal work item. The work-item payload MUST
be the complete strict-JSON observation mapping required to reconstruct the
existing shard receipt, and publication MUST occur immediately after that
context has completed and before the next context begins. Candidate-level
forwards MAY remain in memory within the current context; they MUST NOT be
presented as independently durable work items.

An explicitly requested continuation SHALL validate the complete execution and
schedule identities, read accepted work-item records, and execute only planned
context identifiers that have no accepted record. A context with an accepted
payload, including a caller-owned failure or unmatched outcome, MUST NOT be
automatically re-executed.

#### Scenario: Process exits after several contexts
- **WHEN** three context records are durably accepted and the process exits
  before the fourth context record is published
- **THEN** a matching explicit continuation preserves the three records and
  begins with the fourth context

#### Scenario: Context payload records a non-success outcome
- **WHEN** the consumer publishes a strict observation payload whose status is
  not eligible for a completed legacy receipt
- **THEN** the journal treats the context as mechanically durable and the
  adapter does not retry or scientifically reinterpret it

### Requirement: Logical work and physical schedule are separate
The capability SHALL derive one immutable physical schedule artifact from the
logical contexts without mutating their plan records. Scheduling MUST use the
declared `scalar_equivalent_forward_count` as the cost, sort contexts by
descending cost then ascending context identifier, and assign each context to
the slot with the lexicographically smallest current total cost, context count,
and slot index. The schedule MUST bind its algorithm version, slot count,
logical-plan identity, ordered assignments, per-slot totals, and content
digest.

The physical slot identity MUST NOT replace or rewrite the legacy receipt
`shard_index`. Continuation MUST reuse the exact same schedule identity.

#### Scenario: Cost-aware schedule is regenerated
- **WHEN** the same logical plan and slot count are scheduled repeatedly
- **THEN** the schedule bytes, assignment order, per-slot counts, per-slot
  costs, and schedule digest are identical

#### Scenario: Equal-cost contexts and slots tie
- **WHEN** two contexts have equal declared cost or two slots have equal current
  load
- **THEN** ascending context identifier and then ascending slot index resolve
  the ties deterministically

### Requirement: Legacy shard receipts are materialized without semantic drift
After all required journal records validate, the capability SHALL
deterministically regroup observation payloads by each context's sealed legacy
receipt `shard_index` and materialize the existing
`natural_boundary_owner_support_completion_execution.v1.receipt.v1` schema.
Within each receipt, observations MUST follow the order expected by the sealed
plan. Receipt publication MUST be write-once and bind canonical content.

The current merger MUST accept the materialized receipts without code changes.
The current analyzer's separate census-v3 regression contract MUST remain
unchanged and pass on its owned accepted inputs; shard receipts MUST NOT be
misrepresented as direct analyzer inputs. A pre-existing incompatibility in a
prior-support-ledger to census-v3 bridge MUST be reported separately and MUST
NOT be hidden by rewriting old records or weakening the merger, census, or
analyzer contract. Attempt identifiers, process exits, physical slot
assignments, and continuation history MUST remain outside the scientific shard
receipt.

#### Scenario: Physical and legacy partitions differ
- **WHEN** cost-aware physical slots contain contexts from several legacy
  receipt shards
- **THEN** materialization emits exactly one complete receipt for every legacy
  shard with its original context denominator and observation order

#### Scenario: Journal is incomplete or has an ineligible observation
- **WHEN** any planned context record is missing, invalid, duplicated, or not
  accepted by the existing completed-receipt contract
- **THEN** the adapter refuses to publish benchmark-looking completed shard
  receipts and reports mechanics state without changing scientific meaning

### Requirement: Terminal receipts are execution-history independent
For identical logical inputs and identical deterministic context observations,
an uninterrupted execution and an interrupted execution followed by explicit
exact-identity continuation SHALL produce byte-identical canonical legacy shard
receipts and identical receipt digests. Their journal attempt records and
mechanics diagnostics MAY differ and MUST NOT be copied into those terminal
receipts.

#### Scenario: Interruption followed by continuation
- **WHEN** one execution completes without interruption and another is
  interrupted after at least one durable context then explicitly continued
- **THEN** both executions materialize byte-identical terminal shard receipts
  while retaining distinct attempt histories outside those receipts

### Requirement: Attempt and process-exit diagnostics are mechanical
Every launched worker attempt SHALL have a unique journal attempt identifier.
The launcher SHALL publish a mechanics-only exit receipt that binds the
physical slot, process return code, terminating signal when present, journal
plan and schedule identities, accepted-context count, missing-context count,
and the last durable record's sequence, work-item identifier, and digest when
present. An attempt start without an outcome SHALL remain visible as an
interrupted or unknown-exit diagnostic, not as a completed execution.

The capability MUST NOT automatically relaunch a worker, convert an exit into a
scientific outcome, authorize continuation, or hide a failed attempt after a
later continuation succeeds.

#### Scenario: Worker receives SIGTERM
- **WHEN** the launcher observes a worker exit caused by `SIGTERM`
- **THEN** it records the signal, return code, attempt identity, and last
  accepted durable context without marking the execution scientifically valid
  or automatically starting another process

#### Scenario: Operator explicitly continues
- **WHEN** the operator separately requests continuation with exact identities
- **THEN** a new attempt is created, prior diagnostics remain immutable, and
  only missing contexts are scheduled

### Requirement: Mechanics proof precedes research execution
Acceptance SHALL include a production-shaped deterministic test showing
uninterrupted versus interruption-and-continuation receipt byte equivalence and
a bounded real single-GPU smoke in which an external launcher sends `SIGTERM`
after at least one context record is durable and a fresh process continues the
same exact-identity journal.

The smoke receipt MUST be labeled mechanics-only and MUST bind source, plan,
schedule, model, config, device, process, signal, durable-record, continuation,
and terminal-materialization evidence. It MUST NOT claim a model mechanism,
support prevalence, or recovered historical result.

#### Scenario: Real signal continuation smoke passes
- **WHEN** one GPU process publishes a context, receives external `SIGTERM`, and
  a fresh exact-identity process completes the bounded smoke plan
- **THEN** prior record bytes remain unchanged, only missing work runs after
  continuation, terminal materialization succeeds, and the mechanics receipt
  contains no scientific interpretation

### Requirement: Historical roots and active research unit remain untouched
The capability SHALL write only to fresh output roots and files owned by this
change. Existing sealed roots, old v3/v4 attempts, and the active
`research-probes` worktree MUST remain unmodified. Delivery SHALL be a local
fixed commit with an interface note, an adapter patch or consumer example,
tests, and mechanics receipts; delivery MUST NOT perform merge, cherry-pick, or
push operations.

#### Scenario: Implementation is delivered from the infra worktree
- **WHEN** the change reaches implementation acceptance
- **THEN** all delivered files are fixed in the `research-probe-infras`
  worktree and no active research unit or sealed result root has changed
