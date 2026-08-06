# coordexp-swift-execution-evidence-journal Specification

## Purpose
Provide a small durable execution-evidence boundary for costly, multi-item work
without owning experiment design, scientific interpretation, or scheduling.
## Requirements
### Requirement: Strict canonical evidence values
Every persisted journal envelope, work-item record, attempt record, and terminal
record SHALL contain only recursively strict JSON values: null, booleans,
integers, finite floating-point numbers, strings, lists, and mappings with
string keys. The canonical byte representation SHALL be UTF-8 JSON with sorted
mapping keys, ASCII escaping, no insignificant whitespace, and no non-finite
values. Each persisted record MUST bind the SHA-256 digest of its canonical
content excluding its own digest field.

Unsupported values MUST fail closed. Implementations MUST NOT stringify live
objects, tensors, callbacks, paths, sets, exceptions, or other unsupported
values implicitly.

#### Scenario: Receipt mapping is accepted
- **WHEN** a caller supplies a nested finite JSON receipt mapping
- **THEN** canonical serialization, reload, and SHA-256 recomputation produce
  the same value and digest

#### Scenario: Live runtime object is rejected
- **WHEN** a planned work item contains a live callback or Python object instead
  of its explicit receipt mapping
- **THEN** journal preflight fails before any work-item execution or durable
  result publication

#### Scenario: Non-finite value is rejected
- **WHEN** an envelope or record contains NaN or positive or negative infinity
- **THEN** journal preflight or append fails with a typed artifact-contract
  error and does not publish a partial record

### Requirement: Journal publication is crash-consistent
Every accepted plan, work-item record, process-attempt record, and terminal
record SHALL be published through the same crash-consistent sequence: fully
serialize and validate the payload, write a temporary file on the target file
system, flush and `fsync` the temporary file, atomically publish it to a final
path that does not already exist, and `fsync` the containing directory before
reporting success. Accepted final paths MUST never be overwritten.

If any step fails, the journal MUST report failure without treating the target
record as accepted. Previously accepted records remain unchanged. Reload SHALL
ignore uncommitted temporary files as evidence and report them diagnostically.

#### Scenario: Failure before atomic publication
- **WHEN** a process fails after syncing a temporary record but before atomic
  publication
- **THEN** reload does not count the temporary file as accepted evidence and
  preserves all prior final records

#### Scenario: Failure before directory sync
- **WHEN** atomic publication succeeds but containing-directory sync fails
- **THEN** the append reports failure and a fresh process must reload and
  validate the final path before deciding whether the record is accepted

#### Scenario: Accepted path cannot be replaced
- **WHEN** publication targets an already accepted plan, work-item, attempt, or
  terminal path
- **THEN** publication fails without replacing the existing bytes

### Requirement: Immutable execution plan preflight
Before work-item execution, a journal SHALL atomically publish and reload an
immutable plan envelope. The envelope MUST bind a journal schema version,
execution identifier, complete execution-identity fingerprint, plan
fingerprint, ordered unique expected work-item identifiers, caller-owned opaque
context, and the context fingerprint.

Creating a journal at an occupied root, changing any bound identity, omitting a
planned work item, or supplying duplicate work-item identifiers MUST fail
before execution. The journal SHALL NOT interpret caller context or require
research-specific keys.

#### Scenario: CPU write-read preflight succeeds
- **WHEN** a new execution supplies a valid identity, expected work-item plan,
  and strict caller context
- **THEN** the journal atomically persists and reloads the envelope before the
  caller is admitted to work-item execution

#### Scenario: Duplicate planned identity is rejected
- **WHEN** two planned work items have the same identifier
- **THEN** journal creation fails without publishing an admitted plan

#### Scenario: Existing root is not reused
- **WHEN** a caller attempts to create another journal at an already populated
  journal root
- **THEN** creation fails without replacing or mutating the existing evidence

### Requirement: Independently durable work-item records
The journal SHALL publish each completed work-item record independently through
a single-writer atomic operation after fully serializing and validating it.
Each record MUST bind the execution and plan fingerprints, unique work-item
identifier, monotonically increasing sequence index, process-attempt
identifier, caller-owned opaque payload, payload digest, and complete record
digest.

A later append, attempt failure, terminal-finalization failure, or process exit
MUST NOT delete or rewrite an accepted prior record. Appending an unplanned or
already completed work-item identifier, a sequence conflict, or a mismatched
identity MUST fail closed.

#### Scenario: Late finalizer failure preserves prior work
- **WHEN** several work-item records have been accepted and terminal
  finalization then raises an error
- **THEN** every prior record remains independently readable and hash-verifiable
  while the journal is not reported as completed

#### Scenario: Duplicate work item is rejected
- **WHEN** a process attempts to append a second record for an already completed
  work-item identifier
- **THEN** the append fails without changing the accepted record

#### Scenario: Invalid payload does not damage the journal
- **WHEN** the next work-item payload is not strict JSON serializable
- **THEN** that append fails before publication and all prior records remain
  unchanged

### Requirement: Process attempts remain distinct from execution completion
Every process invocation that writes to a journal SHALL use a unique
process-attempt identifier. An attempt-failure record SHALL describe process or
mechanical failure without closing the execution or assigning scientific
meaning. Terminal completion SHALL be a separate immutable record and MUST be
admitted only when the exact expected work-item set is present once, every
record validates, and the terminal aggregate binds the ordered record digests.

An unfinished journal MAY expose its validated completed work-item identifiers
for caller-directed continuation only when execution identity and plan identity
match exactly. The capability SHALL NOT automatically retry work, decide
whether continuation is scientifically valid, or convert an attempt failure
into a completed execution.

#### Scenario: Exact-identity continuation discovery
- **WHEN** a successor process opens a non-terminal journal with the exact
  execution and plan identities
- **THEN** it can read the validated completed work-item identifiers and start a
  new process attempt without rewriting them

#### Scenario: Repaired code cannot resume old evidence
- **WHEN** source, config, runtime, model, or another bound execution identity
  changes after an attempt failure
- **THEN** continuation is rejected and the caller must use a new execution
  journal

#### Scenario: Incomplete execution cannot finalize
- **WHEN** at least one expected work-item identifier has no accepted record
- **THEN** terminal completion fails and the journal remains non-complete

### Requirement: Journal evidence is mechanics-only
The journal SHALL treat planned identifiers and caller payloads as opaque. Its
stable schema MUST NOT define cohort membership, conditioning, intervention,
arm meaning, estimand, matching, unmatched meaning, thresholds, uncertainty,
scientific validity, claim scope, or stop rules. Journal validation SHALL attest
only identity, strict serialization, durable publication, completeness, and
integrity.

#### Scenario: Mechanically complete null result
- **WHEN** all planned records are durable and caller payloads describe a null
  scientific effect
- **THEN** the journal may attest mechanical completion but does not classify or
  promote the scientific result

#### Scenario: Experiment vocabulary remains caller-owned
- **WHEN** two research callers use different arm or outcome schemas
- **THEN** both may store strict opaque payloads without adding either schema to
  the stable journal contract
