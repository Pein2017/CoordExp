## Context

The superseded `label-studio-coco-refinement` change established executable
contracts for the exact max_len12000 sources, shared images, COCO-80 identity,
norm1000 geometry, durable batch Commit, recovery, coord materialization, and
resident ROI inference. Its parent-owned core is dependency-injected: the
`RefinementRuntime` consumes a `DraftCatalog`, the `WorkingDatasetStore`
consumes immutable batch members and verifier/receipt protocols, and the ROI
runtime consumes current-target and inference-engine protocols.

The remaining vendor layer is not lightweight. Label Studio owns Task,
Annotation, AnnotationDraft, session/CSRF, autosave, Data Manager navigation,
MobX editor state, serialization, and ROI insertion/finalization. Coordinating
those authorities produced recurring races that do not exist in the intended
single-user bbox-only product.

This replacement therefore keeps the proven parent core but gives one small
local service sole ownership of projects, task indexing, Draft persistence,
HTTP, and ROI Draft mutation. The exact sources remain:

- `public_data/coco/rescale_32_1024_bbox_len12000/train.norm.jsonl`
- `public_data/coco/rescale_32_1024_bbox_len12000/val.norm.jsonl`
- shared images below `public_data/coco/rescale_32_1024_bbox/images/`

The operator is one local user. Accuracy, identity preservation, and source
immutability outrank throughput. Commit may take as long as necessary, but
editing and navigation must not wait for publication.

## Goals / Non-Goals

**Goals:**

- Serve the complete train and validation task index through localhost without
  copying source JSONL, source images, or one baseline annotation per task.
- Persist only touched task Drafts in SQLite using native norm1000 objects and
  monotonic compare-and-swap revisions.
- Provide a small SVG bbox editor with the approved COCO-80, editing,
  visibility, Undo, navigation, autosave, and status interactions.
- Reuse the existing store, worker, recovery, identity, materializer, profile,
  inference, parser, and ROI transform implementations through explicit local
  adapters.
- Preserve immutable same-split batch snapshots while allowing later Draft
  edits during background publication.
- Deliver the human-only loop for early UAT before adding the accepted real
  model, while keeping ROI contracts in the same change.
- Keep the legacy Label Studio runtime and committed source available as a
  fallback until the replacement passes user acceptance.

**Non-Goals:**

- Label Studio Draft/result compatibility or migration of its uncommitted
  Drafts.
- Multiple users, permissions, adjudication, remote/LAN serving, collaborative
  editing, or multiple browser writers as a supported workflow.
- Masks, polygons, rotated boxes, crowd controls, aliases, translations,
  arbitrary classes, arbitrary JSONL schemas, or image-only tasks.
- Synchronous JSONL publication, throughput optimization, multiple workers per
  split, distributed queues, or queued ROI jobs.
- Automatic NMS, duplicate deletion, per-box accept/reject staging, or model
  results with lower authority than human boxes.
- Automatic promotion into training. Coord materialization remains explicit.

## Decisions

### 1. Standalone FastAPI service and build-free SVG client

One single-process FastAPI application bound to `127.0.0.1` owns the SQLite
connection factory, static ES modules, image route, Draft APIs, batch workers,
and ROI orchestration. Uvicorn runs with one worker and reload disabled. The
accepted environment currently provides FastAPI `0.136.3`, Uvicorn `0.37.0`,
and Starlette `0.52.1`; launch preflight and a runtime receipt SHALL record the
resolved versions rather than relying on an ambient import. The human-only
gate uses browser-native ES modules, HTML, CSS, and SVG; it has no Node build
step and no vendor editor dependency.

SVG renders the natural image with an overlay in the same view box. Client
pointer geometry is mapped to natural-image edges, while the server validates
and stores only canonical integer norm1000 `xyxy`. The client never becomes
the geometry authority.

Alternatives rejected:

- Extending Label Studio again retains the lifecycle being removed.
- A desktop shell adds packaging without helping the localhost requirement.
- Canvas is viable but makes hit targets, handles, visibility, and browser
  assertions less direct for the expected per-image object counts.
- A React/Vite application can be introduced later only if the build-free
  client becomes materially harder to maintain; it is not needed for Gate A.

### 2. Native task index plus sparse SQLite Draft authority

The runtime root is separate from the legacy root:

```text
outputs/coco_refinement/rescale_32_1024_bbox_len12000/
  state.sqlite3
  runtime.json
  static-session.json
  train/
    working.norm.jsonl
    project.json
    task_index.jsonl
    queue.jsonl
    journal.jsonl
    images -> <approved shared image root>
  val/
    ...
```

SQLite stores compact project/task identity rows for the complete source index,
but not baseline object arrays. Each task row contains split, image ID,
zero-based source row, immutable locator/dimensions/fingerprints, and ordering
fields. Baseline objects are read from the current committed working row.

A Draft row exists only after the first semantic edit or ROI insertion. Its
canonical payload is a JSON array of:

```json
{
  "region_key": "train:coco:123 | local:<uuid> | roi:<receipt>:<result>",
  "bbox_2d": [0, 0, 999, 999],
  "category_name": "person",
  "category_id": 1,
  "coco_ann_id": 123,
  "metadata": {}
}
```

`coco_ann_id` is required for committed/source objects and omitted for new
objects until the store allocates a stable negative ID. Metadata is limited to
approved training/inference provenance. Presentation color, visibility,
selection, and Undo history never enter semantic hashes or exported JSONL.

Every save sends the full native object list plus `expected_revision`. In one
SQLite `BEGIN IMMEDIATE` transaction the service validates task identity,
COCO-80, geometry, stable-key uniqueness, positive source identity, provenance,
and CAS; then it increments a per-Draft integer revision and commits. A stale
revision returns the current authoritative Draft and does not merge silently.

Before a human create, move, resize, or relabel is saved, the browser sends the
natural-image pixel rectangle, canonical English class name, and current task
binding to a non-persistent object-projection endpoint. The service rechecks
that binding, reuses the approved pixel-to-norm1000 quantizer, derives the sparse
category ID, and either returns a deterministic server-generated `local:<uuid>`
for a create request or copies the existing stable identity and provenance for
an update. Projection never writes SQLite. The browser replaces or appends that
returned object in its in-memory full list and then uses the existing idempotent
Draft PUT; that PUT remains the only durable mutation and CAS owner. A binding
change between projection and PUT therefore fails as an ordinary Draft conflict
instead of silently overwriting newer task state.

If the saved semantic hash equals the committed row, the sparse Draft may be
removed and the task returns to its baseline-derived state. This keeps the
database proportional to edited work.

Alternatives rejected:

- Label Studio `rectanglelabels`, percent `xywh`, Annotation IDs, and Draft
  timestamps are legacy wire compatibility and recreate the removed adapter.
- Copying all baseline object arrays into SQLite duplicates a complete dataset
  and introduces a second committed-data authority.
- Browser-only local storage cannot provide durable Commit capture or safe
  JSONL publication.

### 3. Thin native adapters over the proven core

The new service provides these implementations:

- `SqliteDraftCatalog`: transactionally captures every same-split Draft whose
  semantic hash differs from its current committed row and returns immutable
  `AuthoritativeDraftSnapshot` values.
- `SqliteDraftVerifier`: proves the frozen task/draft/revision/hash identity at
  the store boundary without requiring that a later live Draft remain equal.
- `SqliteInferenceReceiptResolver`: resolves accepted ROI receipt links during
  Commit validation.
- `SqliteCurrentTargetProvider` and `SqliteRoiFinalizer`: freeze and recheck the
  task/Draft revision and append accepted ROI results atomically.
- A standalone runtime factory that binds exact source/store/profile paths and
  starts exactly one simple worker per split.

Opaque `project_id`, `task_id`, `annotation_id`, and `draft_id` fields required
by the existing core receive stable local identifiers. No Django or Label
Studio model is imported. Vendor-shaped names in reusable modules are tolerated
behind adapters during Gate A; broad renaming is deferred until executable
behavior is stable.

### 4. Simple asynchronous Commit with exact snapshot semantics

`POST /api/splits/{split}/commits` first flushes the active task through the
client navigation coordinator, then the service captures all pending Drafts in
one SQLite read transaction and calls `RefinementRuntime.capture_and_enqueue`.
The endpoint returns the durable batch receipt after queue fsync and never waits
for whole-file publication.

One in-process worker per split processes one batch at a time through the
existing `WorkingDatasetStore`. There is no throughput target. The UI polls a
status endpoint and may edit or navigate during Queued/Running/Reconciling.

On terminal success, an exact captured Draft that still matches the captured
revision/hash is retired to the new committed baseline. A newer Draft is
preserved. If the terminal result allocates or reuses a `coco_ann_id` for a
stable region key that is still present in that newer Draft, one SQLite
transaction merges only that hidden identity into the matching native object
and advances its revision. Geometry, class, membership, ordering, provenance,
and all unrelated metadata remain byte-for-byte equivalent; missing regions
are never recreated. Failure preserves every Draft and exposes a retry/status
receipt. Startup runs existing store reconciliation before accepting writes.

### 5. One navigation/save coordinator and small client state machine

Semantic writes occur after discrete gestures: pointer-up create/move/resize,
class change, delete, Undo, and ROI insertion. The client may debounce adjacent
events briefly, but task navigation always awaits the current save. Batch work
is never awaited by navigation.

Client states are intentionally small:

- task: `Committed`, `Saving`, `Draft`, or `Conflict`;
- batch overlay: `Idle`, `Queued`, `Running`, `Reconciling`, `Succeeded`, or
  `Failed`;
- ROI overlay: `Idle`, `Selected`, `Running`, `Inserted`, or `Failed`.

Next/Previous/row navigation shows a pending-Draft reminder but does not demand
a Commit. Tab close warns only while a semantic save is in flight or an
in-memory gesture has not reached a durable response.

### 6. Dense-scene interaction is presentation-only

The editor supports show-all, dim-non-selected, hide-non-selected, per-region
visibility, restore, and focused deletion. These flags exist only in browser
state and never mark a Draft dirty.

Class selection accepts exact canonical English COCO-80 names and offers
prefix, substring, and spelling-tolerant filtering over that fixed list. Only
the selected canonical name and official sparse category ID are saved.

Uncommitted inference-origin regions reuse the deterministic neighborhood
color/badge and same-class IoU advisory policy already implemented in the
parent editor policy. Colors and duplicate hints remain presentation-only.

### 7. ROI inference appends server-side under CAS

Gate B begins only after the human loop passes UAT. The client draws one
temporary ROI, chooses factor-valid width/height defaulting to `1024 x 1024`,
flushes the current Draft, and submits the exact task/revision/profile target.
Only one request is active; current-task navigation and ROI controls are locked
while it runs.

The service loads the allowlisted image, executes the existing resident engine,
records the discrete crop/letterbox/inverse transform receipt, validates
canonical COCO-80 parser results, and builds native regions. In one SQLite
transaction it rechecks the target revision and appends every valid result to
the Draft. It returns the new revision and authoritative object list. The
client applies that response as one Undo step; later edit/delete/save behavior
is identical to human boxes.

If the target changed, the service retains the inference receipt but inserts
nothing. Empty, all-rejected, malformed, cancelled, timeout, and runtime
failures follow explicit no-partial-insertion outcomes. The service never
suppresses existing or inferred boxes through NMS.

### 8. Loopback boundary and source protection

The server fails unless bound to numeric loopback. It validates `Host` and
same-origin `Origin`, issues a random SameSite session/CSRF token, derives the
fixed `local-operator` principal server-side, and rejects caller-provided paths.
Image APIs resolve only manifest-bound locators beneath the approved shared
root and prevent traversal/symlink escape.

The default request authority remains the exact numeric bind host and port.
When a local in-app browser proxy rewrites the browser-visible authority, the
launcher may accept one explicit canonical `http://localhost:<port>` or numeric
loopback browser origin with a non-default HTTP port. This option does not
change the numeric loopback bind, does not trust `Forwarded` or
`X-Forwarded-*`, and is disabled by default. Each
request `Host` must exactly match either the bind authority or that one explicit
browser authority; a mutation `Origin` must match the authority selected by its
own `Host`, so the two authorities cannot be mixed.

Source JSONL and source images are opened read-only and fingerprinted. All
mutable files live under the new ignored runtime root. Working JSONL advances
only through the existing atomic store publication. Tests bind before/after
source and image identities.

## Risks / Trade-offs

- **[Risk] The reusable package still contains Label Studio names and payload
  assumptions.** → Keep a strict native adapter boundary for Gate A, add residue
  tests prohibiting vendor imports in the new service, and refactor names only
  where a concrete native DTO cannot be expressed.
- **[Risk] A full task index could recreate heavy bootstrap.** → Store only
  compact identities/offsets, stream source validation, reuse existing task
  index receipts, and measure one real full-index bootstrap/open/restart before
  UAT.
- **[Risk] SQLite/database truth could drift from working JSONL generation.** →
  Bind every Draft save/capture to task identity plus the current store
  generation/base-row hash; fail closed and rebase explicitly after terminal
  publication.
- **[Risk] Multiple server processes could start duplicate in-process
  workers.** → Launch one uvicorn worker with reload disabled, lock the runtime
  root, and refuse a second writer process.
- **[Risk] Ambient FastAPI/Uvicorn versions could drift across environments.**
  → Validate the resolved versions at launch, record them in the runtime
  receipt, and cover unsupported/missing dependency failure before binding a
  port.
- **[Risk] Browser and server disagree after a lost save response.** → Make PUT
  idempotent by client mutation ID, expose authoritative GET, and use monotonic
  CAS revisions; never guess success.
- **[Risk] Server-side ROI insertion changes the legacy browser-first proof
  sequence.** → Preserve target/revision/receipt linkage and validate the final
  saved semantic hash in the same transaction; test response loss and restart.
- **[Trade-off] The build-free client has fewer framework conveniences.** →
  Keep components small and state explicit; introduce a build only after a
  measured maintainability problem.
- **[Trade-off] Commit performance is not optimized.** → Retain the already
  proven atomic store implementation and report status; correctness and
  continued editing are the only gates.

## Migration Plan

1. Preserve the committed legacy vendor HEAD on
   `codex/archive-label-studio-coco-refinement` and record its parent/nested
   commit identities. Do not delete or rewrite the old OpenSpec.
2. Build the standalone runtime under a distinct ignored root and port while
   8080 remains unchanged.
3. Bootstrap the exact original max_len12000 sources into compact task indexes
   and fresh working stores; do not import uncommitted Label Studio Drafts.
4. Deliver Gate A and run UAT on real train/validation tasks, including restart
   and a multi-task Commit while further edits continue.
5. Add Gate B real-profile ROI inference and commit one inserted result through
   materialized coord output.
6. After explicit user acceptance, stop treating 8080 as fallback and update
   current operator docs. Preserve the Git archive branch and runtime receipt;
   runtime-state deletion remains a separate explicit cleanup decision.

Rollback at every pre-acceptance step is simply stopping the new service and
returning to unchanged 8080. No source or legacy runtime mutation is required.

## Open Questions

No product-semantic questions remain after grilling. Framework details beyond
the selected minimal Gate A stack may change only if executed evidence shows a
blocking limitation.
