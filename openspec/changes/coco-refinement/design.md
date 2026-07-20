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
- Mutation of original COCO annotations or shared images. The derived
  max_len12000 norm/coord pair is an approved iterative training target.

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

The operator entrypoint is one foreground shell wrapper that directly binds
`127.0.0.1:53662`, accepts `http://localhost:53662` as the one explicit browser
authority, opts into VS Code remapping of only `localhost` or the exact
`127.0.0.1` bind host to another explicit non-default browser port, and always reopens
`outputs/coco_refinement/gate-a-20260717`. It rejects port overrides. This
removes the transient relay that made browser bookmarks expire while retaining
the numeric-loopback boundary and request-selected exact Origin/CSRF checks.
`Ctrl-C` remains the normal graceful shutdown path.

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

Every terminal-success batch, whether ordinary or Focus-scoped, then enters
the same background training-publication stage. The validated publisher
transactionally overwrites that split's derived max_len12000 norm/coord pair;
the later successful Commit is always the current training target. Store
`generation` remains an internal serialization, idempotency, and crash-recovery
identity and is never an operator-selected dataset branch. Draft reads, saves,
and navigation do not wait for this stage. A restart detects a terminal working
generation whose training receipt is absent or older, republishes that latest
working authority, and only then opens the browser port. The original COCO
annotations and shared images remain outside this overwrite boundary.

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

Select-mode resize uses an explicit selection-affordance layer above every
region body without raising the selected region's full body above its peers.
On pointer down, the editor first performs screen-space proximity arbitration
against only the selected region's eight handles: corner handles accept the
nearest point within 14 CSS pixels and edge-midpoint handles within 12 CSS
pixels, with corners winning an exact tie. A matched handle starts resize and
captures the pointer before any overlapping region body can change selection.
Outside those handle zones, ordinary region-body hit order remains unchanged,
so clicking another visible object still switches selection. Hover previews
the winning resize cursor/handle. Screen-space thresholds remain constant
through zoom, pan, aspect-preserving letterboxing, and image resolution; tiny
boxes with overlapping handle zones resolve by nearest distance rather than
DOM paint order. Fully coincident bodies remain selectable through the
existing right-panel inventory rather than adding click cycling or modifiers.

Class selection accepts exact canonical English COCO-80 names and offers
prefix, substring, and spelling-tolerant filtering over that fixed list. Only
the selected canonical name and official sparse category ID are saved.

The editor exposes two primary mode shortcuts: Command+1/Command+2 when the
browser delivers them, plus plain 1/2 outside text-entry controls as a reliable
fallback. Select is mode 1 and Draw bbox is mode 2. Draw mode renders
presentation-only horizontal and vertical guides through the pointer while it
is inside the natural image, including during the drag. Guides never affect
geometry, Draft state, or exported data.

Local magnification remains a presentation transform over the original image,
never a second geometry authority. `editor-geometry` owns the testable contain
transform between client CSS pixels and the SVG `viewBox`, including
aspect-preserving letterbox offsets and its exact inverse. `svg-editor` owns one
`viewBox` expressed only in natural-image pixels. Command/Meta plus wheel or
trackpad input zooms about the natural point under the pointer; plain wheel
pans vertically and Shift plus wheel pans horizontally. Wheel deltas are
normalized from pixel, line, or page units, and Shift-remapped horizontal input
may arrive through either axis. Meta takes precedence over Shift. This wheel
router is active only over the rendered image pixels, excluding SVG
letterboxing, and calls `preventDefault` only when the clamped candidate
`viewBox` actually differs from the current one. Events outside the image and
outward movement at a fit or pan/zoom boundary therefore remain available to
ordinary webpage scrolling or browser behavior. Ctrl/Alt without Meta is not
reinterpreted as image pan. Space+primary-drag and middle-drag temporarily pan
without changing Select/Draw mode, and the selected-object focus action fits
its natural rectangle with 15 percent padding while preserving image aspect
and clamping to image bounds. The application receives only natural-pixel
`pixelXYXY`; the existing server projection remains the sole norm1000
quantizer.

The UI exposes the current fit-relative zoom, Fit/reset, selected-object focus,
and a canvas-focus layout that hides both side panels without changing the
`viewBox` or Draft. Reflow refreshes screen-space handles against the unchanged
natural view. Magnification uses ordinary browser-native smooth raster
resampling of the original JPEG, equivalent to a conventional resize of the
visible natural crop. It performs no AI enhancement, creates no replacement
image artifact, and is prohibited from annotation coordinate conversion or
exported data.

The right panel renders every current authoritative-or-Draft object in the
same top-left order required by training: ascending `(y1, x1)`, with exact
anchor ties retaining prior committed rank and then new-object creation order.
The displayed `#1..#N` identifiers are transient presentation ordinals, not
`coco_ann_id`, and are recomputed after each completed authoritative create,
move, resize, delete, relabel, Undo, reload, or ROI insertion. They do not jump
during an active pointer drag. Canvas and list selection remain bidirectionally
synchronized. Activating an inventory item by pointer or keyboard enters Select
mode before selecting that stable region, so the ordinary canvas move/resize
affordances are immediately available even when Draw or Pan was previously
active. While focus remains in the inventory, Delete or Backspace deletes the
selected region through the same autosaved Draft path as canvas-focused
deletion. Mode and selection remain presentation-only; deletion remains a
semantic edit.

Class selection is sticky across image navigation. Selecting an existing
object makes its class the active Draw class; a visible `Drawing as: <class>`
cue exposes that state. Entering Draw without an active class focuses the class
search. A train/val split change clears the active class because it changes the
operator's dataset context.

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
`X-Forwarded-*`, and is disabled by default. A separate explicit remap opt-in
may accept another canonical non-default port only when request Host remains
the configured browser hostname or exact numeric bind host. Each mutation
`Origin` must match the authority selected by its own request Host, so ports and
authorities cannot be mixed. Other hostnames, other numeric loopbacks, default
port 80, and noncanonical forms remain rejected.

Source JSONL and source images are opened read-only and fingerprinted. All
mutable files live under the new ignored runtime root. Working JSONL advances
only through the existing atomic store publication. Tests bind before/after
source and image identities.

The immutable-source statement applies to the original COCO dataset and shared
image store. After every terminal committed generation, the operator-approved
training publisher automatically replaces the derived
`rescale_32_1024_bbox_len12000/{split}.norm.jsonl` and matching
`{split}.coord.jsonl` as one receipt-bound transaction. It rebases image
locators to the shared store, validates exact row/object equivalence and the
current CoordExp-Swift loader, rejects the whole publish if any row exceeds
12000 encoded tokens, and preserves the prior pair on failure. No image bytes
or original COCO annotation file are copied or changed.

The original max_len12000 hashes recorded at first bootstrap remain the
immutable identity baseline; the overwritten norm/coord pair is the mutable
active training target. Restarting that same runtime is authorized by the
manifest-bound working store and journal. If the schema-v2 publication receipt
does not yet bind their latest terminal generation, startup reruns the same
transactional publisher before source inspection; otherwise it verifies the
current published pair, object counts, loader result, token ceiling, working
hash, and journal hash. It then resumes the existing workspaces rather than
reimporting the mutable target, preserving Drafts and stable local negative
IDs.

The current training JSONL does not persist `region_key`, so it is insufficient
to reconstruct stable negative IDs in a brand-new runtime after iteration.
Fresh recovery from an iterated target therefore remains fail-closed until an
explicit identity-sidecar contract is designed; this is not part of the fixed
launcher slice.

### 9. One persistent Focus Queue projects the existing task index

The runtime stores at most one active Focus Queue in SQLite. A queue is an
ordered projection of existing stable task identities, not a project, copied
dataset, or annotation namespace. Its members contain only queue ordinal,
split, task ID, image ID, and the already indexed source-row identity. Browser
refresh and service restart preserve the active queue until explicit release.

The operator-facing CLI is intentionally small:

```text
scripts/coco_refinement_focus.py create <image paths...>
scripts/coco_refinement_focus.py status
scripts/coco_refinement_focus.py release
scripts/coco_refinement_focus.py retry-publication
```

`create` validates the complete request before one SQLite transaction: every
path must resolve beneath the approved shared-image root, exist, map uniquely
to the current max_len12000 task index, belong to one split, and appear only
once. Any invalid, absent, ambiguous, duplicate, or mixed-split member rejects
the whole request. Creating while another queue is active fails until that
queue is released. Supplied order is preserved exactly.
The CLI reaches these operations only through the loopback HTTP/session/CSRF
boundary and never opens `state.sqlite3` directly.

When a queue is active, Focus mode Next/Previous and its task list stay within
the ordered members. Direct task reads, full-dataset navigation, and the
ordinary same-split Commit remain available as separate existing operations.
Releasing a queue removes only queue metadata; it never removes or rewrites a
Draft, batch, generation, published row, or image.

`Commit focus` first flushes the active browser Draft, then captures only
pending Drafts whose stable task IDs are active queue members. Capture remains
all-or-nothing and uses the same immutable snapshot, worker, retirement, and
newer-Draft rules as ordinary Commit. An empty focus capture is reported
without creating a batch.

The existing per-split capture lock and single-active-batch store admission are
the reservation boundary: only one same-split capture can durably enqueue, so
the same Draft revision/hash cannot enter both a Focus and ordinary batch.
While a Focus batch is nonterminal or its training publication is waiting or
running, another same-split Commit returns Busy. Editing and Draft saves remain
available. Release also returns Busy until both stages are terminal.

After a Focus batch reaches terminal success, the same background workflow
invokes the accepted training publisher for that split and generation. Focus
success is not reported as fully published until the norm/coord replacement
receipt is terminal. A token-budget or publication failure leaves the previous
training pair intact, preserves annotation/generation state, exposes the
failure receipt, and does not auto-release the queue. Publication never changes
the raw split JSONL, original COCO annotations, or shared images.
While that failed publication remains retryable, the queue continues reserving
its split against newer Commit admission. Explicit release abandons that retry
intent and makes later same-split Commit available without deleting any
annotation or generation state.

The publication job is persisted before enqueue and recovered by exact
`queue_id`, `batch_id`, split, and terminal generation after restart. It holds
the store's per-split batch-process barrier while validating/publishing, but
uses the committed-file lock only for short authority snapshots and final
replacement. This prevents a newer generation from overtaking publication
without blocking ordinary task reads and Draft writes for the full token check.

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
- **[Risk] Per-task authority checks rescan the full working JSONL.** → Build an
  ephemeral byte-span index during the existing full startup/generation
  attestation, bind it to generation, working hash, line count, and file stat,
  and random-read only task-navigation rows. Keep generic restore and Commit
  paths fully attested and fail closed on same-generation file drift.
- **[Risk] Server-side ROI insertion changes the legacy browser-first proof
  sequence.** → Preserve target/revision/receipt linkage and validate the final
  saved semantic hash in the same transaction; test response loss and restart.
- **[Trade-off] The build-free client has fewer framework conveniences.** →
  Keep components small and state explicit; introduce a build only after a
  measured maintainability problem.
- **[Trade-off] Commit performance is not optimized.** → Retain the already
  proven atomic store implementation and report status; correctness and
  continued editing are the only gates.
- **[Risk] A temporary selection could become a second task authority.** →
  Persist only ordered references to indexed stable task IDs, revalidate them
  against the current project, and keep Draft/read/Commit semantics owned by
  the existing repository and working store.
- **[Risk] Automatic training publication can fail after annotation Commit.** →
  Model Focus completion as Commit followed by a separately receipted publish;
  retain the successful generation and prior training pair on publish failure,
  keep the queue active, and permit explicit retry without recapturing Drafts.
- **[Risk] A later same-split Commit could overtake an older Focus publish.** →
  Persist the publication intent before enqueue, reject new same-split Commit
  admission until that chain is terminal, and hold the existing batch-process
  barrier while publishing.

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
7. Add the persistent Focus Queue as a projection over the accepted standalone
   workspace; prove ordered navigation, scoped capture, automatic norm/coord
   publication, restart persistence, and metadata-only release before using it
   for high-frequency small-image refinement.

Rollback at every pre-acceptance step is simply stopping the new service and
returning to unchanged 8080. No source or legacy runtime mutation is required.

## Open Questions

No product-semantic questions remain after grilling. Framework details beyond
the selected minimal Gate A stack may change only if executed evidence shows a
blocking limitation.
