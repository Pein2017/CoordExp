## 1. Contract Decisions and Feasibility Gate

- [x] 1.1 Record the approved V1 decisions: reject empty Commit while retaining Draft, same-machine loopback only, p95<=2s/hard-5s whole-file Commit gate, and rich navigation choices only for in-app transitions.
- [x] 1.2 Add immutable fixtures for both selected source paths, row/box counts, representative row schema, source hashes, image root, official COCO-80 sparse ID registry, and the pinned Label Studio revision.
- [x] 1.3 Add failing exhaustive/golden/property tests for unchanged-bin Label Studio float round trips, edited outward quantization, parser canvas semantics, discrete crop/letterbox inversion, clipping, and strict non-degenerate final geometry.
- [ ] 1.4 Pin and bootstrap accepted Node/Yarn/Python Label Studio frontend/backend runtimes, record build receipts, and execute a fake-backend spike proving source seeding and direct ROI append into the authoritative editable annotation as one undo action with result-meta save/reload.
- [ ] 1.5 Execute progressive 1k/10k/full-train Label Studio import/open/Next/Draft/restart latency/RSS probes plus the full-train streaming Commit benchmark and every durability-cut recovery probe without modifying sources; enforce p95<=2s and the hard 5-second stop.
  - 2026-07-15 gate receipt: one full-train whole-file Commit took 12.145s, exceeding the 5s hard stop. Deeper UI/service implementation is on hold; see `research/investigations/label-studio-coco-refinement-commit-gate/`.
- [ ] 1.6 Gate Wave 1 on strict OpenSpec validation and separate standards/intent audits with no unresolved P0/P1 findings; return to the user if either spike fails or requires a contract change.

## 2. Project, Category, and Working-Data Core

- [ ] 2.1 Add parent-owned module/config boundaries for `Coco80Registry`, `RefinementProjectAdapter`, `WorkingDatasetStore`, `WorkingCoordMaterializer`, manifests, same-origin service contracts, and the dedicated ignored one-instance/two-split runtime layout.
- [ ] 2.2 Implement/test canonical English-name to official sparse-ID validation (forbidding the contiguous evaluator map), hidden stable region-key identity, preservation of positive IDs, serialized non-reused negative allocation, tombstones, and idempotent Commit/reload mapping rehydration.
- [ ] 2.3 Implement/test direct norm1000-to-percentage import, exact unchanged-bin reuse, tolerance-aware edited quantization, clipping/strict geometry, accepted fields, working image-locator rebasing/managed links, and stable `(y1,x1)` ordering with prior-rank tie preservation.
- [ ] 2.4 Implement the exact-Draft Commit coordinator and per-split locked transaction with idempotent commit IDs, prepared/terminal journal records, complete JSONL temp-file fsync/rename/directory-fsync, atomic manifest replacement, stale annotation/row/generation rejection, and committed/rolled-back/outcome-unknown status lookup.
- [ ] 2.5 Implement/test startup reconciliation for every journal/file/directory/manifest/response cut, journal/hash disagreement, ID-allocation recovery, repeated recovery, and no double application.
- [ ] 2.6 Implement idempotent one-instance train/validation bootstrap with stable `(split,image_id)` tasks, exactly one editable imported annotation per task, disabled alternate/native submit paths, exact image-root document setting, project-bound split storage records, managed links, and manifest drift checks.
- [ ] 2.7 Implement the explicit atomic norm-to-current-coord materializer and verify core behavior with source hashes, norm validation, real loader smoke including a negative ID, full-size latency rerun, adjacent-family residue checks, and independent standards/intent audits before Wave 3.

## 3. Bbox Editing and Commit UX

- [ ] 3.1 Generate the bbox-only fixed COCO-80 configuration, suppress the native rotation field/handle when disabled, and enforce parent-side zero rotation while excluding free-form classes, aliases, masks, polygons, and crowd controls.
- [ ] 3.2 Add deterministic exact/prefix/substring/spelling-tolerant canonical-name search with keyboard focus, traversal, selection, cancellation, and tests that never persist aliases, translations, or synonyms.
- [ ] 3.3 Add per-task semantic-hash Committed/Draft/Committing/Reconciling state, error banners, project generation cues, exact-Draft save-before-Commit coordination, and committed snapshot/new-ID rehydration while ordinary save remains Draft-only.
- [ ] 3.4 Add/test the rich in-app guard with Commit-and-continue, Continue-with-saved-Draft, Stay, persisted-Draft reset, save/Commit failure handling, plus browser-native-only hard unload warning and post-Commit undo rebasing.
- [ ] 3.5 Add overlay-only show-all, dim-non-selected, hide-non-selected, restore, native per-region visibility, focused-deletion recovery, and insertion visibility transitions without semantic dirty changes.
- [ ] 3.6 Run targeted frontend/store/browser probes for create/move/resize/relabel/delete, zero-rotation rejection, empty-Draft policy, exact autosave/Commit/reload hashes, lost-response reconciliation, navigation/save failures, dense focus, and one-row Commit; close audits before Wave 4.

## 4. Inference Profile and ROI Mapping Service

- [ ] 4.1 Add strict profiles with content-addressed base-weight/adapter/checkpoint/embedding-delta/tokenizer/model-config/processor artifacts plus complete resolved config/prompt-policy/parser/adapter/transform/Transformers/processor-kwargs fingerprints, same-path drift rejection, processor-derived factor, axis/total-pixel bounds/defaults, deadline, one active profile per project, and credential-safe receipts.
- [ ] 4.2 Implement a pure immutable ROI transform owning float-to-half-open crop, half-up realized dimensions, `scale_x/scale_y`, Pillow bicubic resize, black centered odd-padding rule, no-resize canvas identity, ordered inverse mapping, image clipping, and strict final norm1000 quantization.
- [ ] 4.3 Add authenticated/CSRF-protected same-origin proxy and allowlisted single-flight service keyed by project/task epoch/authoritative annotation revision/profile/generation, with loopback backends, no arbitrary reads, cooperative deadline cancellation, target abandonment, and no direct annotation mutation.
- [ ] 4.4 Add a new resident ROI adapter over runtime assembly plus current prompt/parser/no-resize image/backend components without calling or modifying offline `src/inference/pipeline.py::run`; add tracked-source/AST residue tests against historical `src.infer.*` imports.
- [ ] 4.5 Validate canonical COCO-80 outputs, implement the exact parser/mapping outcome table, retain raw/per-result replay data, and return one target-bound direct-insertion payload without NMS/replacement/merge/per-box acceptance.
- [ ] 4.6 Run unit/property tests for full fingerprints, same-origin security, single-flight/deadline/post-cancel reuse, target epoch, discrete transform edges/padding, every parser outcome, raw receipt replay, and official sparse categories; close audits before Wave 5.

## 5. AI Region and Inference Review UX

- [ ] 5.1 Add one temporary AI Region overlay with replace-on-redraw behavior plus independent width/height controls defaulting to `1024 x 1024` and enforcing the active profile's processor-derived factor and axis/total-pixel bounds.
- [ ] 5.2 Add compact profile selection and request UI for running/accepted/partial/zero/all-rejected/response-failure/runtime-failure/abandoned states; lock navigation/ROI/profile/resolution while running and implement the exact clear-versus-retain table.
- [ ] 5.3 Insert valid responses into the frozen authoritative annotation as one undo action, persist request/region result metadata through Draft reload, revalidate target epoch, mark dirty without a prediction-copy step, and visibly abandon detached responses.
- [ ] 5.4 Implement `visual_policy_v1` across all uncommitted inference regions: 12-bin-expanded neighborhood, stable-key greedy colors, palette-exhaustion badges, same-class IoU>=0.5 advisory conflicts, insertion focus transition, and class-color restoration/metadata stripping after Commit.
- [ ] 5.5 Execute fake-backend browser E2E for rectangular resolutions, lock/redraw/sequential ROI behavior, every outcome, target mismatch/unload abandonment, atomic undo, direct edit/delete, Draft reload colors/provenance, overlaps, retry, navigation guard, and inference-assisted Commit.
- [ ] 5.6 Execute one accepted real-profile ROI smoke through the current prompt/parser/runtime and attest the input canvas, transform receipt, mapped boxes, direct edit, journal link, and committed working row before closing standards/intent audits.

## 6. Full-Project Validation and Operator Handoff

- [ ] 6.1 Bootstrap disposable full train and validation projects in one Label Studio instance under the ignored runtime root and verify idempotence, exact task/annotation counts, split namespaces/storage, managed shared-image reuse, and manifest drift rejection.
- [ ] 6.2 Exercise representative human-only and inference-assisted commits, restart recovery, stale commit rejection, and complete working JSONL validation while proving all source JSONL/image hashes are unchanged.
- [ ] 6.3 Materialize a bounded `working.coord.jsonl` through the new explicit materializer and load it through the current training data path to prove IDs, sparse COCO mapping, tied ordering, geometry, image resolution, and metadata compatibility.
- [ ] 6.4 Add a concise operator runbook for dependencies, launch/shutdown, project/profile configuration, Commit versus Draft, recovery, output locations, validation, known V1 exclusions, and safe runtime-only rollback.
- [ ] 6.5 Run targeted backend/frontend tests, browser E2E, real-profile smoke, full-size performance gate, strict OpenSpec validation, source-residue checks, and final independent standards plus intent-contract audits with no unresolved P0/P1 findings.
- [ ] 6.6 Conduct user acceptance on a small train and validation slice and wait for explicit approval before treating the full projects as the active COCO refinement workflow or archiving this change.
