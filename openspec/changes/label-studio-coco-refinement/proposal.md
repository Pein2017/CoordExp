## Why

COCO 2017 contains missed and occasionally incorrect instance boxes, while the current correction loop requires ad hoc inspection and cannot safely turn a fine-tuned detector's local suggestions into continuously maintained training JSONL. A lightweight localhost workflow built on the existing Label Studio checkout can preserve CoordExp's data contracts while making per-image correction, dense-scene review, and ROI-assisted annotation practical.

## What Changes

- Add one single-user Label Studio instance with independently governed `train` and `val` projects for the exact processed inputs `public_data/coco/rescale_32_1024_bbox_len12000/train.norm.jsonl` and `public_data/coco/rescale_32_1024_bbox_len12000/val.norm.jsonl`; reuse the existing images through validated references and never copy or mutate source images or source JSONL.
- Add an external CoordExp adapter that imports source objects into exactly one authoritative editable annotation per task, maintains a separate mutable `working.norm.jsonl` per split, and provides an explicit validated norm-to-current-coord materializer for training handoff.
- Make sample-level `Commit` the human gate: edits remain freely revisable, `Save Draft` changes only Label Studio state, and Commit freezes and durably saves the exact Draft snapshot before an idempotent journaled update of that sample's working objects. Lost responses enter reconciliation rather than claiming failure.
- Provide bbox-only COCO-80 editing with the official English class names, spelling-tolerant search limited to those names, fast create/update/delete operations, navigation guards, and dense-scene overlay focus/hide controls.
- Distinguish nearby uncommitted inference-origin instances with a versioned deterministic high-contrast coloring policy, while preserving class identity as text and returning committed regions to the ordinary class-color presentation.
- Add saved inference profiles and a simple target-bound single-request ROI flow: draw one temporary ROI, select a default-1024-by-1024 target whose dimensions follow the processor-derived patch factor, reuse the current CoordExp prompt/parser/backend components through a new resident adapter without modifying offline batch inference, map results back through a replayable discrete letterbox transform, and insert valid results directly as editable annotations.
- Preserve existing boxes during inference. Potential duplicates are highlighted for human review but are never silently suppressed, replaced, or merged.
- Keep the first implementation single-user, same-machine, and loopback-only. In-app navigation gets the full Commit/Draft/Stay guard; hard reload/tab close uses the browser-native unsaved-work warning. The first implementation does not support arbitrary JSONL schemas, image-only/no-coordinate tasks, COCO crowd regions, polygons or masks, pixel-level masking, reviewed-empty Commit, multi-user adjudication, queued/batch ROI inference, or automatic dataset promotion.

## Capabilities

### New Capabilities

- `coco-refinement-projects`: Exact-source project bootstrap, immutable input/shared-image reuse, train/val separation, authoritative Draft/Commit behavior, dynamic norm JSONL, coord materialization, journaling, and recovery.
- `coco-bbox-editing`: COCO-80 bbox CRUD, canonical-name search, dense-scene visualization, inference-origin coloring, and dirty-navigation protection.
- `coco-roi-inference`: Versioned inference profiles, temporary ROI interaction, target-resolution constraints, current-template model execution, reversible coordinate mapping, and direct editable result insertion.

### Modified Capabilities

None. Existing CoordExp data, inference, evaluation, and visualization contracts remain authoritative; this change adds adapters and UI behavior around them rather than changing their supported semantics.

## Impact

- Parent-repo adapter/service/config/test surfaces will be added outside the upstream checkout for project bootstrap, image-reference rebasing, JSONL synchronization/materialization, journaling, COCO category mapping, and ROI inference.
- A narrowly scoped Label Studio frontend extension is expected for rotation suppression, authoritative Commit coordination, AI Region controls, direct result insertion, canonical-name fuzzy selection, collision coloring, focus controls, and navigation guards. Upstream code remains vendor-owned outside those explicit extension points.
- Runtime state and mutable artifacts will live under a dedicated ignored output root, with one Label Studio state subtree, separate `train`/`val` data subtrees, and validated managed links to `public_data/coco/rescale_32_1024_bbox/images/`.
- Browser calls use a same-origin proxy to loopback services; Label Studio local-file storage records are bootstrapped per split rather than relying only on environment variables.
- The resident ROI adapter will reuse the checked-in CoordExp-Swift prompt, parser, no-resize image path, and backend components; the stable offline batch entrypoint/config/artifact behavior is unchanged.
- Implementation is gated before deep UI work on deterministic coordinate round trips, atomic-commit fault injection, fake-backend direct insertion, 1k/10k/full-project Label Studio scale probes, a full-size JSONL Commit benchmark targeting p95 at most 2 seconds with a hard 5-second stop gate, and one real-profile ROI smoke.
