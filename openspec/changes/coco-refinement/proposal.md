## Why

The current Label Studio integration proves the COCO refinement data, commit,
recovery, and ROI contracts, but its vendor Draft, Data Manager, MobX, and
navigation lifecycles dominate implementation and verification cost for a
single-user bbox-only workflow. A standalone localhost editor can retain the
validated CoordExp core while making the operator path smaller, faster to
change, and easier to reason about.

## What Changes

- Add a standalone loopback-only Web editor for the exact
  `rescale_32_1024_bbox_len12000` train and validation sources, exposing the
  complete task index without copying source JSONL or images.
- Add SQLite-backed sparse Drafts only for touched tasks. Drafts use native
  norm1000 `xyxy` COCO object records, stable region keys, monotonic CAS
  revisions, and inference provenance rather than Label Studio result or
  `AnnotationDraft` payloads.
- Reuse the existing `WorkingDatasetStore`, `RefinementRuntime`, materializer,
  COCO-80 registry, identity allocation, batch recovery, inference profiles,
  resident model runtime, and ROI transform/mapping contracts.
- Provide lightweight bbox create/move/resize/relabel/delete, exact COCO-80
  search, mode shortcuts, draw-mode pointer guides, a training-order object
  inventory, sticky visible class context, overlap-safe selected-handle resize,
  Undo, dense-scene focus/hide controls, task navigation with Draft flush,
  pending-Draft status, and one explicit same-split batch Commit.
- Keep Commit asynchronous and deliberately simple: one worker per split,
  immutable pending-Draft capture, atomic derived JSONL publication, and no
  performance target beyond remaining usable while the worker runs.
- Deliver in two user-visible gates: first the complete human annotation loop,
  then real-profile ROI inference with server-side atomic Draft insertion and
  editable returned boxes.
- Add one foreground launcher with a fixed direct `localhost:53662` endpoint
  and fixed Gate A runtime root so operator bookmarks and restart commands do
  not depend on an ephemeral relay.
- Add one persistent temporary Focus Queue over the existing task index so an
  operator can supply a small ordered list of approved image paths, navigate
  only those tasks, commit only their pending Drafts, and release the queue
  without copying images/JSONL or deleting annotation state.
- After a Focus Commit succeeds, automatically run the validated training
  publisher in the background and replace only that split's derived
  max_len12000 norm/coord pair as one transaction. Invalid paths, duplicate or
  mixed-split selections, and rows exceeding 12000 tokens fail closed.
- **BREAKING for the legacy prototype only:** do not migrate or remain wire
  compatible with uncommitted Label Studio Drafts. Source/working JSONL, shared
  images, committed object identity, materialized coord output, and inference
  model contracts remain compatible.
- Freeze `label-studio-coco-refinement` as historical implementation evidence.
  Preserve its committed vendor code on a Git archive branch, exclude its
  unfinished browser harness from new work, and retain the current 8080 runtime
  only as a fallback until the standalone editor passes user acceptance.

## Capabilities

### New Capabilities

- `coco-refinement-workspace`: Exact-source full-index projects, shared-image
  access, sparse native Draft persistence, asynchronous batch Commit,
  generation history, recovery, and current coord materialization.
- `coco-bbox-editor`: Lightweight COCO-80 norm1000 bbox editing, navigation,
  autosave, Undo, dense-scene visibility controls, and operator status cues.
- `coco-roi-assistance`: Profile-bound ROI selection, resident inference,
  reversible coordinate mapping, atomic Draft insertion, provenance, and
  direct human correction.

### Modified Capabilities

None. Current CoordExp-Swift data, inference, training, artifact, and evaluation
contracts remain unchanged.

## Impact

- New parent-owned local service, SQLite schema, HTTP API, and static SVG-based
  editor under `/data/CoordExp`; no new active dependency on the Label Studio
  frontend or ORM.
- Existing parent modules under `src/label_studio_coco_refinement/` are reused
  initially through adapter protocols; vendor-shaped names and DTOs may be
  separated only where required by the native Draft boundary.
- New ignored runtime state is isolated from
  `outputs/label_studio_coco_refinement/`; original COCO annotations and shared
  images remain immutable. The derived max_len12000 norm/coord training pair is
  an explicitly publishable iterative target with generation/hash receipts.
- Focus Queue metadata is a small SQLite projection over stable existing task
  identities. It survives browser/service restart until explicit release and
  never becomes a second annotation or image authority.
- The existing Label Studio checkout, state, and port 8080 remain untouched
  during the replacement gate and are not current authority for the new Drafts.
