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
  search, Undo, dense-scene focus/hide controls, task navigation with Draft
  flush, pending-Draft status, and one explicit same-split batch Commit.
- Keep Commit asynchronous and deliberately simple: one worker per split,
  immutable pending-Draft capture, atomic derived JSONL publication, and no
  performance target beyond remaining usable while the worker runs.
- Deliver in two user-visible gates: first the complete human annotation loop,
  then real-profile ROI inference with server-side atomic Draft insertion and
  editable returned boxes.
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
  `outputs/label_studio_coco_refinement/`; source JSONL and images remain
  immutable and shared by validated paths.
- The existing Label Studio checkout, state, and port 8080 remain untouched
  during the replacement gate and are not current authority for the new Drafts.
