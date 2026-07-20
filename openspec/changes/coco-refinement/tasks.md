## 1. Native Workspace Foundation

- [x] 1.1 Record the approved replacement decisions: native norm1000 Drafts with no Label Studio Draft migration, complete train/val indexes plus sparse Draft rows, simple asynchronous Commit, human-loop-first UAT, real ROI second, and legacy fallback isolation.
- [x] 1.2 Record the parent/nested legacy commit identities and archive branch in a durable research receipt; prove the tracked Cypress config is restored and keep the AgentGuard-blocked untracked harness outside every commit.
- [x] 1.3 Add interface-level failing tests for native object canonicalization: strict norm1000 geometry, official sparse COCO-80 IDs, source/new/ROI stable keys, positive-ID preservation, metadata allowlist, deterministic semantic hash, duplicate rejection, and presentation exclusion.
- [x] 1.4 Implement `src/coco_refinement` native object/Draft/task models and canonicalizer without Django or Label Studio imports; adapt canonical regions into the existing `AuthoritativeDraftSnapshot`/store boundary.
- [x] 1.5 Add failing transaction tests and implement the SQLite schema/repository for projects, complete compact task identities, sparse Drafts, monotonic CAS revisions, idempotent mutation IDs, current-generation/base-row bindings, and restart durability.
- [x] 1.6 Add failing bootstrap/image-boundary tests and implement exact max_len12000 train/val indexing, working-store bootstrap, validated shared-image links, allowlisted image resolution, traversal/symlink rejection, idempotent restart, and source/manifest drift failure.
- [x] 1.7 Execute a bounded then full-index bootstrap/open/restart probe; record task counts, source/image identities, runtime-root bytes, SQLite row counts, and representative task latency without imposing a throughput optimization gate.
- [x] 1.8 Add launch-preflight tests and a dependency receipt for FastAPI 0.136.3, Uvicorn 0.37.0, and Starlette 0.52.1; reject missing/unsupported versions, reload/multi-worker process shapes, and duplicate runtime-root writers before port bind.
- [x] 1.9 Gate Wave 1 with targeted tests, `openspec validate --strict`, Django/Label Studio import residue checks, and separate engineering plus intent-contract audits with no unresolved P0/P1 findings.

## 2. Local Service, Draft, and Commit Loop

- [x] 2.1 Add failing adapter tests and implement `SqliteDraftCatalog`, exact Draft verifier, inference-receipt resolver, terminal exact-Draft retirement, and CAS-safe stable-key identity-only merge into newer Drafts over the existing `RefinementRuntime` and `WorkingDatasetStore`.
- [x] 2.2 Add the single-process runtime factory and root lock: one store/worker per split, startup reconciliation before writes, reload/multi-worker refusal, graceful shutdown, and worker-health receipts.
- [x] 2.3 Add FastAPI session/security and task/image endpoints with numeric-loopback bind, exact Host/Origin/CSRF enforcement including one optional explicit local browser-proxy authority, server-derived local principal, no caller paths, no-store mutable responses, task cursor navigation, and authoritative committed-or-Draft reads.
- [x] 2.4 Add idempotent full-Draft PUT/GET APIs with strict shape/CAS validation, response-loss retry, committed-baseline retirement, conflict payloads, and source/working authority rechecks.
- [x] 2.5 Add Commit/status/project-state APIs that flush/capture all pending same-split Drafts, return only after durable enqueue, expose queued/running/reconciling/terminal states, and never block Draft reads/writes/navigation on publication.
- [x] 2.6 Execute multi-task human-only batch probes with a deliberately paused worker, later edits to captured/unrelated tasks, lost enqueue/status responses, invalid all-or-nothing members, terminal exact/newer Drafts, restart reconciliation, and unchanged source/images.
- [x] 2.7 Gate Wave 2 with targeted unit/API/recovery tests, one full-size small-member Commit, strict OpenSpec validation, vendor-import residue checks, and separate engineering plus intent-contract audits.

## 3. Lightweight Human Annotation Gate

- [x] 3.1 Add the build-free HTML/CSS/ES-module application shell with train/val project selector, full task cursor/list, image viewport, compact status panel, and no Label Studio/Node runtime dependency.
- [ ] 3.2 Add SVG image/overlay coordinate mapping and pointer-tested bbox select/create/move/resize/delete with server-returned canonical norm1000 geometry, stable keys, zero rotation, zoom/pan, and natural-aspect preservation.
- [x] 3.3 Add keyboard COCO-80 exact/prefix/substring/spelling-tolerant search, canonical sparse-ID selection, cancellation, and rejection of aliases/free-form persistence.
- [x] 3.4 Add one client Draft controller for discrete autosave, idempotent mutation IDs, save/conflict/error state, Next/Previous/row navigation flush, pending-Draft reminder, unload warning only for unsaved local state, and batch-independent navigation.
- [x] 3.5 Add local Undo grouping plus show-all, dim-non-selected, hide-non-selected, per-region visibility, restore, focused deletion recovery, and tests proving presentation never changes semantic hash or save count.
- [x] 3.6 Add Commit controls and exact per-task/batch status cues for Committed/Saving/Draft/Conflict plus Queued/Running/Reconciling/Succeeded/Failed and newer-Draft preservation.
- [x] 3.7 Run focused browser E2E against the real local service for CRUD, invalid geometry, class search, autosave/reload, navigation/save failure, visibility/Undo, multi-task Commit during later editing, restart recovery, and full-index task opening.
- [ ] 3.8 Launch an isolated Gate A experience instance on a non-8080 port and runtime root, hand the train/val URLs to the user, remove measured interaction blockers without weakening authority checks, relaunch, and wait for explicit human-loop approval before beginning real ROI implementation.
- [x] 3.9 Add tested Select/Draw Command+1/Command+2 plus plain 1/2 fallback shortcuts, bounded Draw pointer guides, visible sticky class context with split reset, and a bidirectionally selectable right-panel object inventory that recomputes transient IDs in exact training order after completed authoritative edits.
- [x] 3.10 Add one tested foreground Gate A launcher that directly binds the fixed `localhost:53662` browser endpoint and approved runtime root without an ephemeral relay.
- [x] 3.11 Add a tested topmost selection-affordance layer and screen-space selected-handle arbitration so corner/edge resize wins over overlapping bbox bodies while clicks outside handle zones retain ordinary object selection.
- [x] 3.12 Add tested edit-equivalent right-panel object activation that enters Select mode and accepts Delete/Backspace from inventory focus while preserving ordinary move/resize/autosave behavior.

## 4. Real ROI Assistance Gate

- [ ] 4.1 Add failing native target/finalization tests and implement SQLite current-target capture, target/revision CAS recheck, inference receipt resolution, and all-valid-results atomic Draft append with idempotent response-loss recovery.
- [ ] 4.2 Add profile-list/infer/status APIs over the existing allowlisted launch manager, resident engine, parser, ROI transform, cancellation/deadline, and credential-safe receipt store without importing the offline inference pipeline entrypoint.
- [ ] 4.3 Add one temporary AI Region SVG mode, replace-on-redraw behavior, independent factor-valid width/height controls defaulting to 1024 x 1024, single-flight task lock, and explicit terminal outcome messages.
- [ ] 4.4 Apply successful authoritative ROI responses as one local Undo action; preserve/edit/delete/reload inference provenance and reuse deterministic nearby-region colors, palette-exhaustion badges, and same-class overlap advisories as presentation-only state.
- [ ] 4.5 Run fake-engine browser/API tests for rectangular transforms, sequential ROI, every parser/runtime/target outcome, response loss, restart, atomic no-partial insertion, direct edit/delete, Undo, reload provenance, colors, overlaps, and later batch Commit.
- [ ] 4.6 Execute one accepted real-profile ROI smoke through current model/prompt/parser components and attest canvas/profile/transform/raw/mapped/Draft/batch/working/materialized links plus unchanged source/images.
- [ ] 4.7 Gate Wave 4 with targeted tests, strict OpenSpec validation, inference-entrypoint/source-residue checks, and separate engineering plus intent-contract audits with no unresolved P0/P1 findings.

## 5. Operator Handoff and Replacement Acceptance

- [ ] 5.1 Run representative train/val human-only and inference-assisted sessions, materialize the terminal generation, load it through the current CoordExp-Swift data path, and verify positive/negative IDs, sparse category mapping, geometry, ordering, metadata, and shared images.
- [ ] 5.1a Add and attest the operator-approved training publisher that validates a terminal generation and the 12000-token ceiling, then transactionally replaces the selected max_len12000 norm/coord pair with shared image locators while preserving original COCO data and recovering the previous pair on failure.
- [ ] 5.2 Add concise operator docs for dependencies, launch/shutdown, runtime roots, Draft versus Commit, recovery/status, output/materialization, ROI profiles, known exclusions, fallback, and protected source data.
- [ ] 5.3 Record final legacy boundaries: archived vendor branch, old OpenSpec status, retained 8080/state disposition, new runtime identity, and the manual cleanup command for AgentGuard-blocked untracked harness files.
- [ ] 5.4 Conduct final user acceptance on train and validation, require explicit approval before deactivating legacy 8080 or treating the standalone workspace as active, then sync stable specs/docs and archive `coco-refinement` only after every requirement has executed evidence.

## 6. Persistent Focus Queue Workflow

- [x] 6.1 Add transaction tests and implement one persistent ordered SQLite Focus Queue over stable existing task identities, including all-or-nothing approved-root path mapping, unique membership, single-split enforcement, one-active-queue refusal, restart recovery, and metadata-only release.
- [x] 6.2 Add Focus Queue create/status/release APIs and `scripts/coco_refinement_focus.py` CLI without accepting arbitrary server-side paths outside queue creation validation or copying images/JSONL.
- [x] 6.3 Extend Draft capture with an explicit stable-task member scope and add Focus Commit/status APIs that capture every and only pending active-queue Draft, preserve ordinary all-pending Commit, and report empty captures without a batch.
- [x] 6.4 Complete the minimum accepted publisher behavior from 5.1a, then chain terminal Focus Commit to that validated background publisher for the same split/generation; persist/recover one queue-batch-publication status chain, hold the split batch-process barrier, reject overtaking same-split Commit/release while nonterminal, preserve the prior norm/coord pair on failure, retain the active queue, and support publication retry without Draft recapture.
- [x] 6.5 Add Focus mode to the build-free client with exact ordered task list, bounded Next/Previous, current/total and pending-count cues, separate Commit/publish states, Focus Commit control, and an explicit switch back to full-index navigation.
- [x] 6.6 Run repository/API/client regression tests for valid and invalid queue creation, member identity revalidation, restart and pre-publication crash recovery, release safety, scoped versus ordinary Commit admission, later Draft preservation, automatic publish success/failure/retry, no generation overtaking, source/raw/image immutability, and responsive editing; then run strict OpenSpec validation and separate engineering plus intent-contract audits.
