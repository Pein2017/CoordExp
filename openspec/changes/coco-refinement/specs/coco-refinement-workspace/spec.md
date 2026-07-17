## ADDED Requirements

### Requirement: Exact full-source workspace
The system SHALL expose every task from the exact max_len12000 train and
validation source contracts in separate split workspaces while preserving
source row identity, image identity, dimensions, metadata, object identity, and
split ordering. It SHALL NOT modify or copy source JSONL or source images.

#### Scenario: Full workspace is bootstrapped
- **WHEN** the operator starts a fresh runtime
- **THEN** the system validates both source fingerprints and builds compact task indexes whose counts and identities equal the complete train and validation sources

#### Scenario: Image is opened
- **WHEN** the browser requests a task image
- **THEN** the service resolves the manifest-bound shared image beneath the approved root without copying it or accepting a caller-provided path

#### Scenario: Source drifts
- **WHEN** a source hash, task identity, image binding, or row count differs from the recorded contract
- **THEN** startup fails closed before serving or mutating a workspace

### Requirement: Sparse native Draft authority
The system SHALL store a Draft only for a task whose native object semantics
differ from its committed working row. A Draft SHALL contain canonical
norm1000 `xyxy` objects, stable region keys, official COCO-80 identity,
approved provenance, a semantic hash, and a monotonic CAS revision; it SHALL
NOT contain or require Label Studio result, Annotation, or AnnotationDraft
payloads.

#### Scenario: Untouched task is read
- **WHEN** a task has never been semantically edited since its committed generation
- **THEN** the service derives its objects from the committed working row and no SQLite Draft row is required

#### Scenario: First edit is saved
- **WHEN** a valid object list is saved against the current absent-or-existing revision
- **THEN** one transaction persists the complete canonical Draft, semantic hash, result hash, mutation identity, and next monotonic revision

#### Scenario: Stale save is attempted
- **WHEN** the expected Draft revision is not current
- **THEN** the service rejects the mutation, returns the current authoritative revision and objects, and performs no automatic merge

#### Scenario: Draft returns to committed semantics
- **WHEN** a valid save exactly matches the current committed row semantics
- **THEN** the sparse Draft may be retired and later reads derive the same objects from the committed row

#### Scenario: Legacy Draft exists
- **WHEN** Label Studio contains an uncommitted Draft for the same source task
- **THEN** the standalone runtime neither imports nor treats that Draft as authority

### Requirement: Durable navigation-scale Draft operations
The service SHALL make discrete Draft saves and reads independent of working
JSONL publication and SHALL recover committed SQLite transactions after
process restart.

#### Scenario: Worker is busy
- **WHEN** a same-split batch is queued, running, or reconciling
- **THEN** valid Draft saves, task reads, and navigation remain available without waiting for the batch to publish

#### Scenario: Repeated interaction within one working generation
- **WHEN** the operator repeatedly reads or saves tasks after the working JSONL has been fully attested for the current generation
- **THEN** task navigation reads only the indexed target row, retains generation and row-hash checks, and does not rescan the complete working JSONL for each gesture

#### Scenario: Working JSONL changes behind the navigation index
- **WHEN** the working file signature changes without a matching published generation
- **THEN** navigation fails closed instead of silently rebuilding against uncommitted or externally modified bytes

#### Scenario: Service restarts after Draft response loss
- **WHEN** SQLite committed a mutation but the browser did not receive its response
- **THEN** retry or authoritative reload resolves the mutation by its idempotency identity and does not duplicate or discard objects

### Requirement: Immutable asynchronous same-split Commit
One explicit Commit SHALL transactionally capture all pending Drafts in the
selected split as one immutable batch and durably enqueue it before returning.
One worker per split SHALL publish the complete batch atomically through the
existing working store. Commit duration SHALL NOT block later Draft edits or
navigation.

#### Scenario: Pending Drafts are committed
- **WHEN** the operator invokes Commit after the active Draft save succeeds
- **THEN** the service freezes every eligible same-split Draft revision/hash/payload, fsyncs one batch queue record, and returns a queued receipt without waiting for JSONL publication

#### Scenario: Draft changes after enqueue
- **WHEN** an included task receives a later Draft revision while its batch runs
- **THEN** the worker publishes only the frozen payload and preserves the newer Draft for a later Commit

#### Scenario: Batch succeeds
- **WHEN** validation, identity allocation, and atomic file publication all succeed
- **THEN** the complete working JSONL advances by one generation, exact captured Drafts retire to the new baseline, and newer Drafts remain pending

#### Scenario: Allocated identity meets a newer Draft
- **WHEN** a successful batch allocates or reuses a `coco_ann_id` for a stable region key that remains present in a later Draft revision
- **THEN** one CAS-safe SQLite transaction merges only that ID into the matching native object, advances the Draft revision, preserves every newer semantic/provenance field and object membership, and never recreates a missing region

#### Scenario: Batch fails before publication
- **WHEN** any member, receipt, source row, or identity validation fails
- **THEN** no working row or generation advances, every Draft remains recoverable, and the batch exposes one terminal failure receipt

#### Scenario: Commit response is lost
- **WHEN** the durable enqueue response is lost and the same batch identity is retried
- **THEN** status resolves the original immutable batch without recapturing newer Drafts or applying it twice

### Requirement: Generation and recovery authority
The working store journal and published working JSONL SHALL remain the authority
for batch generation, identity allocation, tombstones, and crash recovery.
SQLite project/Draft state SHALL be a validated projection and SHALL fail
closed on cross-generation disagreement.

#### Scenario: Service starts with interrupted publication
- **WHEN** queue, candidate, file, directory, manifest, journal, or terminal projection records describe an interrupted batch
- **THEN** startup reconciliation exposes only the prior complete generation, the next complete generation, or a reconciling state and never a partial batch

#### Scenario: SQLite generation drifts
- **WHEN** a Draft or task projection references a generation/base-row hash that is not the current store authority
- **THEN** capture and mutation fail closed until the projection is explicitly refreshed

### Requirement: Explicit coord materialization
The system SHALL materialize training coord JSONL only through the existing
generation-bound materializer and SHALL preserve positive and allocated
negative object IDs, official sparse COCO identity, geometry, ordering,
metadata, and shared image resolution.

#### Scenario: Current generation is materialized
- **WHEN** the operator explicitly requests coord output for a terminal working generation
- **THEN** the materializer emits one derived coord JSONL bound to that exact generation and accepted by the current CoordExp-Swift data path

### Requirement: Legacy isolation and rollback
The standalone runtime SHALL use a distinct port, runtime root, SQLite state,
and Draft namespace from the legacy Label Studio runtime until explicit user
acceptance.

#### Scenario: Replacement is under evaluation
- **WHEN** the standalone service starts before final acceptance
- **THEN** it does not restart, rewrite, or depend on the legacy 8080 process or Label Studio state

#### Scenario: Operator rolls back before acceptance
- **WHEN** the standalone service is stopped
- **THEN** the unchanged legacy runtime remains independently usable and source data remains unchanged

### Requirement: Reproducible local service dependencies
The runtime SHALL validate and record the resolved FastAPI, Uvicorn, and
Starlette versions before binding its loopback port, SHALL run exactly one
application worker with reload disabled, and SHALL fail closed when the
supported dependency or process shape is unavailable.

#### Scenario: Supported service environment starts
- **WHEN** launch preflight resolves the accepted HTTP stack and acquires the sole runtime-root writer lock
- **THEN** the runtime receipt records exact versions and process shape before the service accepts requests

#### Scenario: Service environment is unsupported
- **WHEN** a required dependency is absent/unsupported or another writer owns the runtime root
- **THEN** startup fails before binding the port or starting any batch worker

### Requirement: Explicit local browser-proxy authority
The service SHALL use the exact numeric loopback bind authority by default. A
launcher MAY declare one canonical HTTP browser origin whose host is exactly
`localhost` or a numeric loopback and whose port is explicit and non-default.
This browser authority SHALL NOT change the listening socket or authorize
forwarded authority headers.

#### Scenario: No browser proxy origin is configured
- **WHEN** a request Host differs from the exact numeric bind authority
- **THEN** the service rejects the request before route dispatch

#### Scenario: One local browser proxy origin is configured
- **WHEN** request Host exactly matches that configured authority and a mutation Origin matches the same authority with a valid session and CSRF token
- **THEN** the request follows the ordinary route and mutation validation path

#### Scenario: Browser and mutation authorities are mixed
- **WHEN** request Host matches one accepted authority but mutation Origin names the other authority, an unconfigured port, or any non-loopback origin
- **THEN** the service rejects the mutation before route dispatch and performs no semantic write
