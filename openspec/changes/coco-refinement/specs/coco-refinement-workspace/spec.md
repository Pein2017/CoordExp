## ADDED Requirements

### Requirement: Exact full-source workspace
The system SHALL expose every task from the exact max_len12000 train and
validation bootstrap contracts in separate split workspaces while preserving
source row identity, image identity, dimensions, metadata, object identity, and
split ordering. The recorded bootstrap fingerprints SHALL remain the immutable
identity baseline. The derived max_len12000 norm/coord pair MAY later advance
only through the receipt-bound training publisher; source images and original
COCO data SHALL NOT be modified or copied.

#### Scenario: Full workspace is bootstrapped
- **WHEN** the operator starts a fresh runtime
- **THEN** the system validates both source fingerprints and builds compact task indexes whose counts and identities equal the complete train and validation sources

#### Scenario: Image is opened
- **WHEN** the browser requests a task image
- **THEN** the service resolves the manifest-bound shared image beneath the approved root without copying it or accepting a caller-provided path

#### Scenario: Source drifts
- **WHEN** a source hash, task identity, image binding, or row count differs from the recorded contract
- **THEN** startup fails closed before serving or mutating a workspace

#### Scenario: Existing runtime restarts after a terminal Commit
- **WHEN** both split workspaces already exist and a latest terminal working generation lacks an exact current training-publication receipt
- **THEN** startup validates the manifest-bound working authority, republishes that latest generation through the transactional publisher, verifies the resulting norm/coord receipt, and resumes without replacing bootstrap identity or Drafts

#### Scenario: Iterated source lacks valid restart authority
- **WHEN** neither an exact current receipt nor a valid manifest-bound working/journal authority can produce the latest derived pair
- **THEN** startup fails closed before binding the browser port

#### Scenario: Fresh runtime is requested from an iterated target
- **WHEN** no existing manifest-bound workspace can recover local object identity from the working journal
- **THEN** startup does not infer stable negative IDs from the training JSONL and fails until a separately specified identity sidecar contract exists

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

### Requirement: Iterative training-pair publication
The system SHALL automatically publish every successful ordinary or
Focus-scoped Commit as the latest derived max_len12000 split norm and coord
JSONL pair while leaving original COCO annotations and all shared images
unchanged. Later successful Commits SHALL supersede earlier pairs in commit
order; generation SHALL remain internal recovery metadata. Publication SHALL
be receipt-bound, failure-recoverable, and all-or-nothing at the application
contract boundary.

#### Scenario: Terminal generation is published
- **WHEN** a terminal-success Commit's working hash, task inventory, shared-image resolution, norm/coord equivalence, current training loader, and 12000-token budget all validate
- **THEN** that split's norm and coord targets advance together to that latest Commit and a receipt records the internal generation, paths, hashes, counts, and validation evidence

#### Scenario: Service restarts before automatic publication completes
- **WHEN** the latest terminal working state is newer than, or lacks, its training-publication receipt
- **THEN** startup republishes that latest working state before binding the browser port and does not expose an older pair as current

#### Scenario: Any refined row exceeds the budget
- **WHEN** current CoordExp-Swift encoding reports more than 12000 tokens for any candidate row
- **THEN** publication fails with the offending sample identities and neither training target advances

#### Scenario: Publication is interrupted
- **WHEN** replacement fails or a prior transaction is discovered incomplete
- **THEN** recovery restores the previous complete pair or finishes the verified new pair before reporting success

### Requirement: Persistent ordered Focus Queue
The system SHALL maintain at most one active temporary Focus Queue as a durable
ordered projection of existing task identities. Queue metadata SHALL survive
browser refresh and service restart until explicit release and SHALL NOT copy
images, JSONL rows, object payloads, or create another project.

#### Scenario: Valid image paths create a queue
- **WHEN** every supplied path exists beneath the approved shared-image root, maps uniquely to the current max_len12000 index, belongs to one split, and is not duplicated
- **THEN** one transaction creates the active queue in exact supplied order and Focus navigation exposes only those tasks

#### Scenario: Any requested member is invalid
- **WHEN** a path is absent, outside the approved root, ambiguous, unindexed, duplicated, or belongs to another split
- **THEN** the whole create request fails with member-specific errors and no active queue metadata is changed

#### Scenario: Another queue is active
- **WHEN** create is requested before the active queue is released
- **THEN** the request fails without replacing or merging the existing queue

#### Scenario: Service restarts with an active queue
- **WHEN** SQLite and project identities remain valid across restart
- **THEN** the same ordered queue resumes without reimporting tasks or annotations

#### Scenario: Queue is released
- **WHEN** the operator explicitly releases the active queue
- **THEN** only queue metadata is deleted while every Draft, batch, generation, published annotation, original file, and image remains unchanged

#### Scenario: Queue release is requested while work is active
- **WHEN** the Focus batch is nonterminal or its automatic publication is waiting or running
- **THEN** release fails Busy and preserves the complete queue/status chain

### Requirement: Focus-scoped Commit and observable training publication
The system SHALL offer a Focus Commit that captures only pending Drafts whose
stable task identities are members of the active queue. After terminal batch
success it SHALL use the same automatic latest-Commit publisher as ordinary
Commit while retaining queue-scoped publication status and retry controls.

#### Scenario: Focus queue has pending Drafts
- **WHEN** the operator invokes Focus Commit after the active Draft save completes
- **THEN** one immutable batch captures every and only eligible pending queue member and editing/navigation remain available while it runs

#### Scenario: Another same-split Commit overlaps Focus work
- **WHEN** a Focus capture, batch, or training publication is nonterminal and another Focus or ordinary Commit is requested for that split
- **THEN** the existing capture/admission barrier accepts at most one durable batch, rejects the other request Busy, and never freezes the same Draft revision/hash into two batches

#### Scenario: Focus queue has no pending Drafts
- **WHEN** no active queue member differs from its committed baseline
- **THEN** the service reports an empty capture without creating a batch or publishing training files

#### Scenario: Focus batch and publication succeed
- **WHEN** the scoped batch reaches a terminal generation and the publisher validates image bindings, norm/coord equivalence, loader compatibility, and the 12000-token ceiling
- **THEN** that split's derived norm/coord pair advances transactionally and one status chain binds queue, batch, generation, and publication receipt

#### Scenario: Training publication fails
- **WHEN** token-budget or publication validation fails after the Focus batch committed
- **THEN** the successful working generation remains authoritative, the previous training pair remains complete, the queue remains active and reserves same-split Commit admission, and status exposes a retryable publication failure without recapturing Drafts

#### Scenario: Failed publication is released instead of retried
- **WHEN** the operator releases a terminal failed Focus publication
- **THEN** only its queue/retry metadata is abandoned, annotation and generation state remain, and later same-split Commit admission becomes available

#### Scenario: Service restarts between Commit and publication
- **WHEN** a persisted Focus batch succeeded but its bound publication is absent or nonterminal after restart
- **THEN** the service resumes publication from the same queue, batch, split, and terminal generation without recapturing Drafts or allowing a newer same-split generation to overtake it

#### Scenario: Ordinary workflow is used while a queue exists
- **WHEN** the operator requests direct/full-dataset navigation or ordinary same-split Commit
- **THEN** full navigation remains available; ordinary Commit retains its full-split scope and is accepted after Focus publication succeeds or its failed retry intent is explicitly released

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
The launcher MAY separately opt into VS Code port remapping for only that
configured browser host and the exact numeric bind host. A remapped authority
SHALL still use canonical HTTP with an explicit non-default port. Neither mode
SHALL change the listening socket or authorize forwarded authority headers.

#### Scenario: No browser proxy origin is configured
- **WHEN** a request Host differs from the exact numeric bind authority
- **THEN** the service rejects the request before route dispatch

#### Scenario: One local browser proxy origin is configured
- **WHEN** request Host exactly matches that configured authority and a mutation Origin matches the same authority with a valid session and CSRF token
- **THEN** the request follows the ordinary route and mutation validation path

#### Scenario: VS Code remaps the browser port
- **WHEN** remapping is explicitly enabled and request Host is the configured browser host or exact numeric bind host with another canonical non-default port
- **THEN** the request follows the ordinary path and every mutation still requires Origin to equal that exact request-selected authority plus a valid session and CSRF token

#### Scenario: Browser and mutation authorities are mixed
- **WHEN** request Host matches one accepted exact or remapped authority but mutation Origin names another port/authority, or Host uses any other name/address
- **THEN** the service rejects the mutation before route dispatch and performs no semantic write

### Requirement: Fixed direct Gate A launcher
The operator launcher SHALL run the standalone service in the foreground on
numeric loopback port `53662`, accept `http://localhost:53662` as the one
configured browser authority, opt into canonical VS Code port remapping only
for `localhost` and `127.0.0.1`, reuse the approved Gate A runtime root, and
reject server-port overrides. It SHALL NOT require a separate forwarding
process.

#### Scenario: Operator starts Gate A
- **WHEN** the operator invokes the launcher from any working directory
- **THEN** the service directly listens on `127.0.0.1:53662`, prints the stable browser URL, and reuses `outputs/coco_refinement/gate-a-20260717`

#### Scenario: Operator requests another port
- **WHEN** the operator passes an unsupported argument or attempts to override the port
- **THEN** the launcher exits before starting the runtime and reports that Gate A is fixed at port `53662`
