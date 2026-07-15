## Context

The selected source is the already normalized, nearly complete COCO pair at
`public_data/coco/rescale_32_1024_bbox_len12000/train.norm.jsonl` and
`public_data/coco/rescale_32_1024_bbox_len12000/val.norm.jsonl`.
Each row has immutable image identity and dimensions plus a non-empty `objects`
list whose boxes are integer `xyxy` coordinates on the inclusive `0..999`
lattice. Existing objects carry the official sparse COCO `category_id`, the
canonical English `category_name`/`desc`, and a positive `coco_ann_id`.

The source contains 117,266 train rows with 849,947 boxes and 4,952 validation
rows with 36,335 boxes. The selected rows contain no empty examples and no
non-positive boxes. Images already live under
`public_data/coco/rescale_32_1024_bbox/images/`; copying them into Label Studio
would add storage without adding provenance.

`label-studio/` is an upstream checkout (currently version `1.24.0.dev0`) and
already supplies image rectangle CRUD, region visibility/locking, zoom, label
filtering, tasks/projects, and ML-backend integration. CoordExp remains the
owner of data and inference semantics. The narrow UI behaviors that are not
native enough for this workflow are implemented at explicit Label Studio
extension points; project/data/inference ownership stays in the parent repo.

The pinned checkout can support the target shape through one Label Studio
instance and two projects, but its development runtime is not currently
bootstrapped: Yarn, frontend dependencies/build output, and the Python virtual
environment are absent. Source scale is also material: train alone contains
117,266 tasks, slightly above Label Studio's approximate 100k guidance. Runtime
bootstrap and progressive 1k/10k/full-project measurements are therefore Wave-0
gates rather than late launch checks.

The active inference implementation is `src/infer.py` plus
`src/inference/{pipeline,runtime,backend,prompt,parsing}.py`. The accepted
Qwen3-VL inference config and its exact prompt/parser contract remain the
authority. The older `src/infer/` package is historical and is not a service
API for this change.

The selected `.norm.jsonl` is the editing artifact, not a file the current Swift
loader can consume directly: that loader expects coordinate-token strings for
this seven-field source family. The legacy generic converter is not executable
in the live checkout because its referenced codec module is absent. This change
therefore owns a narrow, explicit norm-to-current-coord materializer and proves
its output through the actual loader; it does not introduce a new format or
change training input semantics.

This design intentionally supersedes the generic candidate/snapshot review
workflow described in `label-studio/AGENTS.md` only for this named change. The
user has selected a dynamic working dataset: successful inference results enter
the active editable annotation directly, and sample-level Commit immediately
replaces that row's objects in a derived JSONL. Raw inputs remain immutable and
the append-only journal preserves recovery and provenance.

## Goals / Non-Goals

**Goals:**

- Provide one lightweight, single-user, same-machine loopback Web workflow with
  one Label Studio instance and distinct train/validation projects over the
  exact selected source.
- Reuse source images by reference and keep both source images and source JSONL
  byte-for-byte unchanged.
- Make bbox correction fast: add, resize, move, relabel, and delete official
  COCO-80 instances with keyboard-friendly canonical-name search.
- Make dense scenes legible with overlay-only focus/hide controls and visually
  distinct nearby inference-origin instances.
- Keep Draft state freely editable inside Label Studio while making Commit the
  explicit per-sample gate into `working.norm.jsonl`.
- Run one ROI inference request at a time through a saved resident-model
  profile, invert an attested resize/letterbox transform, and insert valid
  results into the active annotation as ordinary editable boxes.
- Preserve enough receipts to recover interrupted commits and reproduce which
  model/profile/transform created an inference-origin box.
- Provide an explicit validated `working.norm.jsonl` to current coord-token
  materialization step without making it an automatic promotion.
- Keep the first delivery small enough to audit and operate locally.

**Non-Goals:**

- Arbitrary JSONL schemas, raw image-only tasks, or any input other than the
  exact `max_len12000` train/validation pair.
- COCO crowd regions, polygon/segmentation/rotated boxes, pixel masks, or model
  classes outside the official COCO-80 set.
- Chinese names, aliases, synonyms, or free-form object descriptions.
- Multi-user adjudication, permissions, remote SaaS deployment, queued/batch
  ROI requests, automatic active learning, or automatic dataset promotion.
- Automatic non-maximum suppression, duplicate deletion, replacement, or merge
  of existing and inferred boxes.
- Reviewed-empty Commit. V1 preserves an empty Draft but requires at least one
  object for Commit and coord materialization.
- LAN/remote-browser access or a desktop shell. Hard reload/tab close uses the
  browser-native Leave/Stay warning; rich asynchronous choices are in-app only.
- A new training/inference format. `.coord.jsonl` remains an explicit derived
  training surface produced from the committed working norm view.

## Decisions

### 1. Parent-owned adapter with narrow vendor UI extensions

The parent repository owns five cohesive components:

1. `Coco80Registry`: canonical English names and official sparse COCO IDs.
2. `RefinementProjectAdapter`: idempotent project/task bootstrap, fixed Label
   Studio config, image references, task identity, and manifest checks.
3. `WorkingDatasetStore`: validation, per-sample commits, journal/recovery, and
   generation receipts.
4. `WorkingCoordMaterializer`: explicit norm-int to current coord-token export
   with image-reference and loader attestation.
5. `RoiInferenceService`: profile loading, resident runtime calls, transform
   receipts, parser validation, and mapped result payloads.

The Label Studio checkout owns only presentation and interaction changes that
cannot be delivered through configuration/API: canonical fuzzy selection, the
AI Region control, direct insertion into the active annotation, inference-local
colors, focus controls, and the dirty-navigation/Commit affordance.

The browser calls the allowlisted parent service through a same-origin proxy
mounted beneath the Label Studio origin; it never sends an arbitrary filesystem
path. Label Studio's API/database owns Draft persistence, while
`WorkingDatasetStore` owns committed dataset truth. A dedicated Commit
coordinator freezes one semantic snapshot, durably saves that exact Draft, and
then submits its identity/hash to the parent store so the two authorities never
silently commit different payloads.

Alternatives considered:

- Building a new annotation UI was rejected because it would reimplement the
  mature rectangle editor, zoom, task navigation, and annotation state.
- Putting all logic inside Label Studio/Django was rejected because it would
  couple CoordExp data and model contracts to vendor internals.
- Running separate Label Studio instances per split was rejected because one
  instance with two isolated projects is simpler and already supplies project
  task/storage boundaries.
- Using read-only Label Studio `predictions` was rejected for the ROI path
  because the agreed interaction requires results to be immediately editable
  without an accept/copy step.
- Calling `src/inference/pipeline.py::run` online was rejected because that API
  owns offline JSONL planning, runtime construction, and run artifacts. The ROI
  adapter reuses owner-neutral prompt/parser/image/backend components without
  changing batch entrypoint, config, or artifact semantics.

### 2. Exact-source split projects and shared images

Bootstrap creates one Label Studio instance containing exactly two projects,
`train` and `val`, with one task per source row. Stable task identity is
`(split, image_id)` and the manifest also records source line number, source
path/hash, source and working image locator, image-root identity, storage ID,
adapter version, vendor revision, COCO-80 registry fingerprint, and label-config
fingerprint. Cross-split moves are not supported.

The runtime root is dedicated and ignored by Git:

```
outputs/label_studio_coco_refinement/rescale_32_1024_bbox_len12000/
  label-studio/
    state/
  train/
    project.json
    working.norm.jsonl
    journal.jsonl
    images -> /data/CoordExp/public_data/coco/rescale_32_1024_bbox/images
  val/
    project.json
    working.norm.jsonl
    journal.jsonl
    images -> /data/CoordExp/public_data/coco/rescale_32_1024_bbox/images
```

`working.norm.jsonl` is an ordinary derived file, not a symlink. Immutable
semantic row fields (`file_name`, `image_id`, `width`, `height`, and source
metadata) are preserved. Because source `images[0]` values are relative to the
source JSONL directory, bootstrap deliberately rebases the working locator to
`images/{train2017|val2017}/...` and creates a validated managed link at each
split root. The manifest retains both locators and verifies the link resolves to
the one allowlisted shared image root. This changes location syntax, not image
identity or bytes.

For Label Studio image access, bootstrap sets the document root to the exact
shared images directory and idempotently creates project-bound local-file
storage records for `train2017/` and `val2017/`. A task is seeded with exactly
one editable `annotations` entity containing every source object; neither
source boxes nor ROI results use the read-only `predictions` collection. Native
creation/deletion of alternate annotations and native submit/skip paths are
disabled for these projects, while region CRUD remains enabled.

Startup fails closed on source hash, row identity, schema, image-link/storage,
class-registry, authoritative-annotation, or project-manifest drift.

### 3. Label Studio percentages are an editing view of the norm1000 lattice

Source imports map each norm1000 edge directly to Label Studio percentage space:

```
percent = 100 * bin / 999
```

Each authoritative region retains its stable region key and last committed
integer bbox. If its canonical geometry is unchanged, Commit reuses those exact
integers instead of re-quantizing binary floats. Edited rectangles use a
decimal/tolerance-aware outward quantizer: starts apply
`floor(percent * 999 / 100 + 1e-9)` and ends apply
`ceil(percent * 999 / 100 - 1e-9)`, then clip to `0..999` and require strict
`x1 < x2`, `y1 < y2`. Implementations must prove all 1000 edge bins through the
actual Label Studio JSON round trip; ordinary raw `floor/ceil` over binary
floats is forbidden because it expands known untouched end bins.

This rule is intentionally distinct from the executed inference parser's
current `round(bin * extent / 1000)` conversion. ROI inference first respects
that parser contract on the model canvas, then inverts the recorded letterbox
transform into original-image coordinates, and only then quantizes into the
working dataset's norm1000 lattice. Round-trip and boundary fixtures cover both
contracts separately, including `x + width` reconstruction in Label Studio.

### 4. Hidden identity and category compatibility

The UI exposes only the 80 official English class names. `Coco80Registry`
performs all name-to-official-sparse-ID mapping and rejects any unknown name or
name/ID mismatch. IDs are never editable in the browser.

Every imported region has a hidden stable region key mapped to its positive
`coco_ann_id`. Geometry and class edits preserve that mapping. A newly drawn or
inferred region receives a stable split-local negative integer ID at its first
Commit. The per-split allocator is serialized under the project lock and never
reuses an issued ID. Region-key-to-ID mappings and tombstones are authoritative
in the journal/rebuildable store index, returned by Commit, and rehydrated into
Draft metadata on reopen; a lost response cannot cause a second allocation.
IDs and mapping metadata are never editable in the browser.

Committed objects contain only the current accepted object fields:
`bbox_2d`, `desc`, `category_id`, `category_name`, `coco_ann_id`, and optional
`metadata`. `desc` and `category_name` are the same canonical name. Output
objects preserve the current stable top-left ordering contract. The adapter
first orders known objects by their prior committed rank and new objects by
their stable creation ordinal, then applies the existing stable `(y1, x1)`
sort. Thus equal-top-left source objects retain their prior sequence rather than
being reordered by arbitrary UI order, while stable IDs carry identity across
ordinary geometry reorderings.

`working.norm.jsonl` is validated as the canonical editing output, not falsely
described as directly loader-compatible. On explicit operator request,
`WorkingCoordMaterializer` copies each committed row, replaces each norm integer
with the exact `<|coord_N|>` token, preserves IDs/classes/rebased image locators,
writes an atomic `working.coord.jsonl`, and proves it through the current Swift
loader. It refuses empty or otherwise invalid working rows and does not promote
the result automatically into a training config.

### 5. Draft, Commit, journal, and recovery

Every task opened in the editor is mutable through exactly one authoritative
annotation ID/revision. Ordinary Label Studio save behavior updates Draft state
only. A sample Commit is coordinated across Draft and working-data authorities:

1. Disable semantic editing, serialize one canonical projection `H` (stable
   region key/identity, quantized geometry, canonical class, and membership;
   excluding order noise and view metadata), and force/await durable Label
   Studio Draft save of that exact annotation revision and hash. Draft-save
   failure stops here.
2. Submit an idempotent `commit_id` with project/task/annotation/Draft IDs,
   annotation revision, `H`, base row hash, and observed project generation.
   The store rejects stale or mismatched identities before allocation/writes.
3. Validate and allocate any new negative IDs, then append/fsync a prepared
   journal record containing before/after rows, identity mappings/tombstones,
   provenance references, and the expected candidate file/manifest hashes.
4. Stream and validate the complete split JSONL into a sibling temporary file,
   fsync the file, rename over `working.norm.jsonl`, and fsync the parent
   directory.
5. Atomically replace and fsync the manifest/generation through its own sibling
   temporary file and parent-directory fsync, then append/fsync a terminal
   committed record. Rebuildable indexes are never the transaction authority.
6. Return the committed row hash/generation and region-key-to-ID mapping. The UI
   rehydrates the exact canonical committed snapshot as its new Draft/baseline
   and clears the pre-Commit undo history.

A per-project process/file lock rejects concurrent commits instead of allowing
last-writer-wins. `commit_id + H` retries are idempotent. A definite failure
before working-file replacement leaves the prior generation authoritative. A
lost response or failure after replacement is `Commit outcome unknown`, not
`Commit failed`; the browser queries commit status and startup recovery uses the
prepared record plus working/manifest hashes to finalize committed or rolled
back state exactly once before serving projects. Recovery is tested at every
append/fsync/rename/manifest/response cut and source files are never targets.

The full train JSONL is rewritten on each Commit because the requested external
contract is an immediately current, ordinary JSONL. The approved target is p95
at most 2 seconds on the intended local storage, with a hard stop if any
representative Commit exceeds 5 seconds. Failure of that early benchmark blocks
deep UI implementation and requires renewed approval before substituting a
delta store, deferred projection, or database authority.

The approved V1 behavior rejects Commit when all boxes have been deleted while
preserving that state as a Draft. Supporting reviewed-empty samples later
requires an explicit `verified_empty` working/training-projection change.

### 6. Navigation guard and version cues

Dirty state compares the active canonical semantic projection with that task's
last committed row hash/time; project generation is shown separately and a
Commit on another task cannot make this task dirty. In-app task navigation,
Previous/Next, routes, and editor-close controls present `Commit and continue`,
`Continue with saved Draft`, and `Stay`; destructive discard is secondary and
must reset the persisted Draft from the committed row. Navigation continues
only after the chosen Commit or Draft save succeeds.

Hard reload/tab/window close cannot await custom asynchronous actions. It uses
the browser-native unsaved-work Leave/Stay warning and makes no Commit claim.
The visible semantic states are `Committed`, `Draft`, `Committing`, and
`Reconciling outcome`; validation/write errors are banners on the still-dirty
Draft, not a competing state. A new semantic edit clears stale validation
errors. Successful Commit rebases the Draft and undo stack: a committed deletion
cannot be resurrected by browser Undo, and redrawing creates a new negative ID.

### 7. Canonical class search and dense-scene presentation

Class selection searches only `Coco80Registry` values. Ranking is deterministic:
exact match, prefix/substring match, then spelling distance. Case and repeated
whitespace are normalized for matching, but returned values are always the
canonical English names. There are no aliases, translated labels, synonyms, or
user-created labels.

Dense-scene controls affect overlays only; they never modify the image or
annotation payload. The reviewer can show all regions, dim non-selected
regions, hide non-selected regions, and restore all overlays. Per-region native
visibility remains available.

All uncommitted inference-origin regions in the active task participate in
versioned `visual_policy_v1`, including regions from sequential ROI requests.
Two boxes are color-neighbors when their norm1000 rectangles intersect after
each edge is expanded by 12 bins and clipped to `0..999`. Stable region keys are
sorted before deterministic greedy coloring from an accessible high-contrast
palette; palette exhaustion uses deterministic reuse plus a visible numeric
instance badge. A potential-duplicate cue is separately defined as same
canonical class with IoU at least `0.5`; it is advisory and never blocks Commit
or changes objects.

Request/region receipt IDs are persisted as non-training Draft metadata so
colors and conflict groups reproduce after save/navigation/restart. Canonical
class text remains visible. Color/group/badge state is presentation metadata,
not class or training metadata; Commit strips it from working objects, retains
journal links, and restores ordinary class colors. Insertion while other boxes
are hidden exits hide-non-selected and selects/flashes the new group; deleting
the focused region restores Show All.

### 8. Versioned resident-model profiles

An inference profile records a stable profile name, endpoint/bind target,
content-addressed fingerprints for resolved base weights, adapter/checkpoint,
embedding delta, tokenizer, model config, processor artifacts, resolved infer
config, and full system+user prompt policy; it also records parser/adapter/
ROI-transform versions, Transformers version, forced processor kwargs,
processor-derived patch/merge factor, default target width/height, axis and
total-pixel bounds, timeout/deadline policy, and optional runtime metadata.
Multiple profiles can be saved; exactly one is active per project, and the user
can switch through a compact selector. Activation/runtime matching fails before
inference if any payload changes at the same path. Journal/receipts record the
immutable resolved profile rather than only its display name. The evaluator's
contiguous category mapping is explicitly forbidden as a source for official
sparse COCO IDs.

The service assembles the current runtime once and keeps the selected backend
resident. It never invokes the offline pipeline run API. The already-letterboxed
canvas is passed through the accepted Qwen image path with `do_resize=False` and
the observed processor grid/canvas equality is asserted. Requests are
single-flight in V1 and carry a deadline/cancellation signal through generation;
the slot is released only after a terminal backend state. Cooperative stopping
and post-cancel model reuse are an early probe—failure blocks this resident
design rather than silently treating an HTTP timeout as cancellation.

### 9. One temporary ROI, explicit resolution, reversible mapping

AI Region is a temporary selection mode, not a COCO annotation class. The user:

1. draws one ROI on the original image;
2. chooses target width and height (default `1024 x 1024`, each divisible by
   the active profile's processor-derived factor, currently attested as `32`,
   and within its axis/total-pixel bounds);
3. clicks Infer and waits for the one request;
4. receives valid mapped boxes directly in the active editable annotation;
5. adjusts/deletes them normally, then continues with another ROI.

Drawing a new ROI replaces only the prior temporary ROI, never annotation
boxes. Existing and earlier inferred annotations remain. There is no implicit
context expansion: the submitted crop is exactly the selected ROI after
clipping to original-image bounds.

At submission the service freezes `request_id`, task ID/epoch, authoritative
annotation ID/revision, profile fingerprint, ROI, canvas, and pre-existing dirty
state. In-app task/annotation navigation plus ROI redraw, profile switching, and
resolution changes are locked until a terminal request state. Forced unload
marks the receipt `abandoned_before_insertion`; before insertion the browser
revalidates the frozen target, and any mismatch produces abandonment with zero
annotation mutation.

ROI percentages are converted to natural-image floating edges. The crop uses
clipped half-open integer edges
`[floor(left), floor(top), ceil(right), ceil(bottom))`. The adapter computes
`fit = min(canvas_w/crop_w, canvas_h/crop_h)`, rounds realized dimensions with a
versioned half-up rule, records the actual `scale_x` and `scale_y`, resizes once
with fixed Pillow bicubic resampling, and applies black centered letterbox
padding with the extra odd pixel on right/bottom. The same immutable transform
object prepares pixels and inverts boxes. The receipt records:

- task/image identity and immutable original width/height;
- floating and clipped integer ROI edges in original-image coordinates;
- crop edge convention, requested canvas, realized resize dimensions,
  `scale_x`/`scale_y`, each padding edge, resampler/pad value, pixel-center/edge
  convention, and pixel/processor fingerprint;
- resolved model/profile/checkpoint/template/parser identities;
- raw response text or durable content reference plus hash, parser status,
  parsed/inserted/rejected counts, and mapped result IDs.

For every parser-valid canvas box, mapping performs the inverse in this order:
remove padding, divide x/y by recorded `scale_x`/`scale_y`, clip to the ROI content rectangle,
offset by the ROI origin, clip to original-image bounds, and quantize to strict
norm1000 `xyxy`. Degenerate results after clipping are rejected and counted.
The same transform object used to prepare pixels owns the inverse operation;
parallel hand-written formulas are forbidden.

Terminal behavior is explicit:

- parser `accepted`: insert all mapped valid boxes atomically and clear ROI;
- `accepted_with_drops`: insert valid boxes atomically, show dropped reasons,
  and clear ROI;
- true parser `empty`: insert nothing, show zero, and clear ROI;
- syntactically valid parse whose classes/mappings are all rejected: insert
  nothing, show all-rejected reasons, and clear ROI;
- parser `all_spans_dropped` or `unsupported_format`: insert nothing and retain
  ROI as a response-level failure;
- transport/runtime/timeout/profile failure: insert nothing and retain ROI.

Every outcome preserves pre-existing Draft dirtiness. A successful insertion is
one undo step and marks the task dirty; zero/all-rejected/failure does not.

### 10. Direct insertion, equal status, and non-destructive conflicts

ROI output is class-validated against COCO-80 and inserted as ordinary editable
rectangle labels in the active annotation, not as read-only Label Studio
predictions and not behind a per-box accept/reject queue. Inference-origin
metadata remains in serializable result `meta` until Commit/reload for receipts,
but committed objects have the same training status and schema as human-created
objects. Direct insertion uses the existing active-annotation append seam as one
history action only after matching the frozen task/annotation epoch.

Inference appends to the active objects. It never removes, merges, relabels, or
changes an existing region. `visual_policy_v1` highlights same-class IoU>=0.5
potential duplicates and colors nearby instances for human inspection, with no
automatic NMS/replacement and no blocking conflict state. Sample Commit is the
sole human acceptance gate.

### 11. Local security and failure isolation

The approved V1 deployment is same-machine only. Label Studio and parent
services bind to loopback, while the browser reaches adapter/inference APIs
through an exact same-origin proxy namespace. State-changing requests require
Label Studio authentication, CSRF/capability protection, and expected
project/task/annotation/revision/generation; unknown origins or stale identities
are rejected. Direct credential-free cross-port mutation is forbidden.

Local image serving is rooted at the exact shared image directory, not the whole
repository or `/data`, and project-bound storage records restrict train/val
subdirectories. Paths and managed links are resolved and checked against that
root. Logs/receipts never include credentials. LAN access is out of V1 scope and
requires a later explicit bind/auth/CORS/image-serving change.

The fixed bbox project suppresses the rectangle `rotation` field/handle when
`canRotate=false`, and the parent Commit validator independently requires zero
rotation. This closes the native Info-panel path that would otherwise create an
unsupported rotated rectangle.

## Risks / Trade-offs

- **Whole-file Commit latency:** rewriting the 117,266-row train JSONL after
  every sample is simple and satisfies immediate-output semantics, but may feel
  slow. Mitigation: enforce the approved p95<=2s / hard-5s early gate, keep the
  write streaming/atomic, and return for approval before changing semantics.
- **Label Studio scale/build feasibility:** the train project exceeds approximate
  100k guidance and the checkout lacks a ready frontend/runtime. Mitigation: pin
  Node/Yarn/Python receipts and probe 1k, 10k, then full task import/open/Next/
  Draft/restart latency and RSS before deep vendor work.
- **Vendor extension maintenance:** direct active-annotation insertion and the
  navigation guard touch Label Studio frontend behavior. Mitigation: isolate
  patches, pin the vendor revision, add browser-level fixtures, and avoid
  rewriting native rectangle CRUD.
- **Coordinate off-by-one drift:** source preparation, Label Studio floats, and
  the current inference parser use different scale conventions. Mitigation:
  preserve unchanged bins, use tolerance-aware quantization, record the full
  discrete affine/raster transform, and run exhaustive/browser golden tests.
- **Draft/working divergence:** Label Studio state can be newer than the derived
  JSONL. This is intentional. Mitigation: one authoritative annotation,
  freeze-save-commit handshake, per-task semantic hash, idempotent commit status,
  and explicit outcome reconciliation.
- **Identity leakage/collision:** new boxes do not have official COCO annotation
  IDs. Mitigation: negative split-local allocator, tombstones, uniqueness checks,
  and a loader smoke over derived data.
- **Dense colors mistaken for labels:** instance colors could look semantic.
  Mitigation: keep class text visible, scope colors to uncommitted inference
  regions, and restore class colors after Commit.
- **Model/runtime coupling:** online inference could accidentally fork accepted
  offline semantics or time out while generation keeps running. Mitigation: a
  new adapter over owner-neutral components, full profile fingerprints,
  no-resize grid assertions, cancellation probes, and one real-profile smoke.
- **Local file exposure:** Label Studio local-file serving can expose too broad a
  root. Mitigation: exact project storage records, same-origin proxy protection,
  exact image-root allowlisting, and loopback-only binding.

## Migration Plan

1. Record source hashes, row/box counts, COCO-80 registry fingerprint, and the
   pinned Label Studio revision in fixtures; do not modify current data.
2. Pin/build the Label Studio runtime and run early feasibility gates: exact
   source-annotation seeding/direct insertion with a fake backend, 1k/10k/full
   project scale, full-train atomic Commit latency, and cut-point recovery.
3. Implement/test the parent registry/project/store/materializer boundary, then
   bootstrap disposable train/val projects under the ignored runtime root with
   managed image links and storage records.
4. Implement the minimal editing UI extensions, authoritative Commit handshake,
   navigation guards, and fake-backend browser flow.
5. Implement the resident ROI adapter/profile/discrete-transform boundary,
   deadline cancellation, and one real-profile smoke using the accepted
   CoordExp prompt/parser/no-resize path.
6. Validate working norm JSONL, explicitly materialize a bounded coord sample,
   load it with the current training loader, and attest source hashes are
   unchanged.
7. Perform user acceptance on a small train/val slice before opening the full
   projects. Rollback removes only the dedicated runtime root and vendor patch;
   immutable source artifacts need no restoration.

## Approved Product Decisions

- V1 rejects empty Commit but preserves an empty Draft.
- V1 is same-machine and loopback-only; no LAN browser access.
- Whole-file Commit targets p95<=2 seconds and stops for redesign/approval on a
  representative operation exceeding 5 seconds.
- In-app navigation offers the rich asynchronous guard; hard reload/tab close
  uses the browser-native unsaved-work warning.

## Open Questions

None before implementation. If cooperative model cancellation cannot be made
safe while retaining a resident backend, implementation pauses and returns for
an explicit process-isolation/runtime decision.
