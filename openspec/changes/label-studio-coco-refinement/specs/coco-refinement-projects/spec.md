## ADDED Requirements

### Requirement: Exact selected source
The system SHALL bootstrap refinement projects only from
`public_data/coco/rescale_32_1024_bbox_len12000/train.norm.jsonl` and
`public_data/coco/rescale_32_1024_bbox_len12000/val.norm.jsonl`, and SHALL reject
source path, source hash, row identity, or schema drift.

#### Scenario: Bootstrap from accepted train source
- **WHEN** the operator bootstraps the train refinement project from the exact recorded source and hashes
- **THEN** the system creates one task for every source row and records the source contract in the project manifest

#### Scenario: Bootstrap from an adjacent COCO view
- **WHEN** the operator supplies a bare, coord-token, shorter-length, or otherwise different COCO JSONL
- **THEN** the system rejects it without creating or mutating a refinement project

### Requirement: Immutable sources and shared images
The system SHALL never mutate the selected source JSONL or source images and
SHALL reference images from
`public_data/coco/rescale_32_1024_bbox/images/` without copying image bytes into
project state. Working rows SHALL rebase their relative `images[0]` locator
through a validated managed link while preserving image identity, `file_name`,
dimensions, and source metadata.

#### Scenario: Project initialization
- **WHEN** either split is initialized
- **THEN** project tasks and working rows resolve to the same allowlisted existing image bytes and the recorded source hashes remain unchanged

#### Scenario: Working JSONL moves away from the source directory
- **WHEN** the derived split is initialized under the runtime root
- **THEN** its rebased locator resolves through the managed link instead of reusing the now-invalid source-relative string

#### Scenario: Escaping the image root
- **WHEN** a task or request resolves outside the exact image root
- **THEN** the system rejects the path and exposes no file

### Requirement: Independent split projects
The system SHALL maintain one Label Studio instance with separate train and
validation projects, project-bound local-file storage records, working files,
journals, locks, generations, and task namespaces, with stable task identity
based on split and `image_id`.

#### Scenario: Same numeric image identity is addressed
- **WHEN** a task is read or committed
- **THEN** its split is part of the identity and no row in the other project can be changed

#### Scenario: Cross-split move requested
- **WHEN** a user attempts to move or commit a task into the other split
- **THEN** the system rejects the operation

### Requirement: Idempotent project bootstrap
The system SHALL record source, source/working image locators, storage identity,
adapter, category-registry, label-config, vendor, authoritative-annotation, and
task-manifest fingerprints and SHALL make bootstrap idempotent for an exact
matching project.

#### Scenario: Matching bootstrap is repeated
- **WHEN** bootstrap is run again with the same complete manifest
- **THEN** the system reuses the project without duplicating tasks or resetting Draft or committed state

#### Scenario: Existing project fingerprint differs
- **WHEN** bootstrap finds an image/storage, annotation, category, config, source, adapter, or vendor fingerprint mismatch
- **THEN** it fails closed and reports the mismatched field

### Requirement: One authoritative editable annotation per task
Bootstrap SHALL seed every source object into exactly one ordinary editable
annotation per task with stable hidden region identity and SHALL disable
alternate annotation creation/deletion, native submit/skip, and prediction-based
source state for these projects.

#### Scenario: Source task is opened
- **WHEN** a bootstrapped task is loaded
- **THEN** the selected authoritative annotation contains exactly the source objects with editable percentage rectangles and preserved hidden positive IDs

#### Scenario: Another annotation is requested
- **WHEN** the reviewer attempts to create, select, delete, submit, or skip an alternate annotation entity
- **THEN** the project blocks that path while leaving bbox region CRUD available

### Requirement: Dedicated derived runtime state
The system SHALL store mutable project state under a dedicated ignored output
root with one Label Studio state subtree, separate split data subtrees, validated
managed image links, and an ordinary `working.norm.jsonl` per split without
symlinking it over the source.

#### Scenario: Fresh split is initialized
- **WHEN** bootstrap succeeds for a split
- **THEN** its working JSONL has the same row identities and accepted row schema as the source while remaining a distinct mutable file

### Requirement: Draft and Commit have distinct authority
The system SHALL treat Label Studio saves as mutable Draft state only and SHALL
treat a successful sample-level Commit as the only operation that replaces that
sample's `objects` in `working.norm.jsonl`. Commit SHALL freeze one canonical
semantic snapshot, durably save that exact authoritative Draft revision, and
submit its hash/identity to the working store before replacement.

#### Scenario: Draft is saved
- **WHEN** the reviewer edits a task and invokes ordinary Draft save
- **THEN** Label Studio retains the edit but the working JSONL generation and row remain unchanged

#### Scenario: Sample Commit succeeds
- **WHEN** the exact saved Draft snapshot passes validation and its durable Commit completes
- **THEN** exactly that sample's objects are current in the working JSONL and the browser reports the new generation

#### Scenario: Draft save fails before Commit
- **WHEN** the authoritative Draft snapshot cannot be durably saved
- **THEN** the parent transaction does not begin, the task remains dirty, and navigation stays on the task

#### Scenario: Commit definitely fails before replacement
- **WHEN** validation, locking, or prepared-write checks fail before working-file replacement
- **THEN** the system leaves the previous working generation authoritative, keeps the Draft open, and reports the failure stage

#### Scenario: Commit response is lost after replacement
- **WHEN** the client cannot prove whether a post-replacement transaction committed
- **THEN** it enters outcome reconciliation and queries the idempotent commit ID instead of claiming the prior generation is authoritative

### Requirement: Validated working norm rows
Every committed row SHALL preserve immutable semantic row/image fields and its
validated rebased image locator, SHALL contain a non-empty object list, and each
object SHALL have strict norm1000 integer `xyxy`, canonical
`desc`/`category_name`, official sparse `category_id`, and a unique
`coco_ann_id`. The system SHALL describe this as the editing artifact, not as a
file directly accepted by the current coord-token loader.

#### Scenario: Valid edited row is committed
- **WHEN** all objects satisfy the current row and object contracts
- **THEN** the system writes a valid working norm row without changing immutable image identity or dimensions

#### Scenario: All objects are deleted
- **WHEN** the active Draft has an empty object list and the reviewer invokes Commit
- **THEN** the system rejects Commit, retains the empty Draft, and explains that the current training contract requires at least one object

#### Scenario: Invalid geometry or class is submitted
- **WHEN** an object is degenerate, outside the `0..999` lattice after clipping, unknown to COCO-80, or inconsistent in name and ID
- **THEN** the system rejects the entire sample Commit without partially changing the working row

### Requirement: Explicit current-coord materialization
The system SHALL provide an operator-invoked atomic materializer that converts
every committed norm integer into the exact current `<|coord_N|>` string while
preserving row/object identity, classes, ordering, and working image locators,
and SHALL validate the result through the actual Swift loader.

#### Scenario: Valid working split is materialized
- **WHEN** the operator requests a coord export from a valid committed generation
- **THEN** a complete `working.coord.jsonl` is atomically written and accepted by the current loader without changing `working.norm.jsonl`

#### Scenario: Working split is empty or invalid
- **WHEN** any row violates the approved non-empty, geometry, class, ID, or image contract
- **THEN** materialization fails without replacing a prior valid coord output or promoting a training config

### Requirement: Stable hidden object identity
The system SHALL preserve the `coco_ann_id` of an existing object across
geometry/class edits and SHALL allocate a stable, unique, split-local negative
integer ID for each newly committed human or inference object without exposing
ID editing in the UI. The authoritative journal/index and Draft metadata SHALL
map stable region keys to IDs idempotently across response loss and reload.

#### Scenario: Existing box is moved and relabeled
- **WHEN** a source object is edited and committed
- **THEN** its original positive `coco_ann_id` remains attached to the updated object

#### Scenario: New box is committed
- **WHEN** a region with no committed identity is first committed
- **THEN** the system assigns an unused negative ID and retains it across later edits

#### Scenario: Deleted identity is followed by another addition
- **WHEN** a committed object is deleted and a later object is added
- **THEN** the deleted ID is not reused

#### Scenario: Commit response is lost after allocating a new ID
- **WHEN** the task reloads and recommits the same stable region key
- **THEN** reconciliation restores the originally allocated negative ID and never allocates a second one

### Requirement: Deterministic object materialization
The system SHALL materialize committed objects with `desc` equal to canonical
`category_name`, official category mapping, accepted fields only, and the
existing deterministic top-left geometric ordering.

#### Scenario: UI order differs from geometry order
- **WHEN** the reviewer commits boxes selected or created in arbitrary order
- **THEN** the row is written in stable top-left order while prior committed rank resolves equal-top-left ties and creation ordinal resolves new ties

### Requirement: Atomic per-sample replacement
The system SHALL serialize each split's commits and SHALL acknowledge success
only after a validated complete working JSONL has atomically replaced the prior
generation, its directory entry and manifest are durable, and a terminal
receipt exists or can be finalized by recovery.

#### Scenario: Two commits overlap
- **WHEN** a second Commit arrives while the split lock is held or with a stale generation
- **THEN** the system rejects it for retry and never applies last-writer-wins

#### Scenario: Process stops before atomic replacement
- **WHEN** a prepared journal entry exists but the previous working file is still authoritative
- **THEN** startup recovery rolls back or completes exactly one deterministic generation before accepting new commits

#### Scenario: Process stops after replacement
- **WHEN** the candidate working file is authoritative but the terminal journal marker was not flushed
- **THEN** startup recovery recognizes the candidate hash, finalizes the receipt, and does not apply the change twice

#### Scenario: Process stops after manifest replacement
- **WHEN** working file and manifest match the prepared candidate but the terminal journal append is absent
- **THEN** recovery appends exactly one terminal record and preserves the candidate generation

### Requirement: Append-only commit journal
The system SHALL append prepared and terminal journal records containing commit
identity, canonical Draft hash/revision, task identity, generation, before/after
hashes and object payloads, identity mappings/tombstones, timestamp, and
referenced inference receipts sufficient for audit and recovery.

#### Scenario: Human-only commit is inspected
- **WHEN** an operator reads the journal entry for a completed sample
- **THEN** the entry identifies the exact before/after row state and contains no invented model provenance

#### Scenario: Inference-assisted commit is inspected
- **WHEN** committed objects include ROI results
- **THEN** the journal links the resolved profile and transform receipt that produced them

### Requirement: Immediate ordinary JSONL output
After each successful Commit, `working.norm.jsonl` SHALL be a complete ordinary
JSONL reflecting all committed samples through the acknowledged generation and
SHALL not require replay of a delta log to read current state.

#### Scenario: Editing-data consumer opens working output
- **WHEN** a successful Commit has been acknowledged
- **THEN** the consumer can stream the complete current JSONL without consulting Label Studio or the journal

### Requirement: Whole-file Commit latency gate
The implementation SHALL measure full-train Commit latency before deep UI work,
SHALL target p95 at most two seconds on intended local storage, and SHALL stop
for renewed design approval when a representative operation exceeds five
seconds rather than silently changing output authority or projection timing.

#### Scenario: Early benchmark exceeds the hard gate
- **WHEN** a representative full-train Commit takes more than five seconds
- **THEN** implementation pauses before deep vendor UI work and does not substitute a delta store or deferred JSONL projection

### Requirement: Source-safe recovery and rollback
Recovery and rollback SHALL operate only inside the dedicated runtime subtree
and SHALL never rewrite source JSONL or source images.

#### Scenario: Runtime project is abandoned
- **WHEN** the operator removes the dedicated project runtime subtree
- **THEN** all source artifacts remain intact and reusable for a clean bootstrap
