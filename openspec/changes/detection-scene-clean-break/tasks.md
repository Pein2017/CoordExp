# Tasks

These tasks are proposal and implementation gates. Production code, configs, and
stable current-behavior docs must not be changed until the user explicitly
approves implementation after reviewing this OpenSpec change and the
Superpowers plan.

## 1. Proposal And Review

- [x] 1.1 Scaffold the OpenSpec change `detection-scene-clean-break`.
- [x] 1.2 Draft the proposal, design, capability deltas, and proposal-review
  implementation roadmap.
- [x] 1.3 Collect subagent findings for existing OpenSpec contracts, current
  docs, and naming/surface classification.
- [x] 1.4 Revise OpenSpec and plan artifacts from subagent findings.
- [x] 1.5 Tighten the OpenSpec activation boundary, scene minimality,
  Stage-2 parse ownership, and replacement-before-deletion constraints before
  implementation.
- [x] 1.6 Produce a decision packet that states changed files, convergence
  status, unresolved blockers, skipped validation, and the no-code hold point.
- [x] 1.7 Receive explicit user approval before production implementation.

## 2. Archive And Classification Gates After Approval

- [x] 2.1 Create or identify the pre-cleanup archive checkpoint commit, branch,
  or tag.
- [x] 2.2 Record the archive checkpoint in the appropriate progress or docs
  surface.
- [x] 2.3 Classify active and historical surfaces as keep/rename, temporary
  migration handle, quarantine, or delete.
- [x] 2.4 Name search gates for historical public names before deletion starts.
- [x] 2.5 Name replacement-before-deletion gates for every retained canonical
  behavior that will be renamed, rerouted, or removed from an old surface.

## 3. DetectionScene Semantic Layer After Approval

- [x] 3.1 Introduce the `DetectionScene`, `DetectionObject`, and
  `DetectionGeometry` semantic layer.
- [x] 3.2 Route raw JSONL loading into the semantic layer without making raw rows
  the in-memory authority.
- [x] 3.3 Preserve image/geometry alignment, coordinate frame, bbox/poly meaning,
  and object ordering invariants for retained canonical surfaces.

## 4. Stage-1 Projection After Approval

- [x] 4.1 Route Stage-1 detection teacher-forcing through
  `DetectionScene -> RenderedDetectionSequence -> DetectionSupervisionView`.
- [x] 4.2 Rename or replace public `recursive_detection_ce` vocabulary for the
  new canonical Stage-1 surface.
- [x] 4.3 Add characterization tests for template render/parse, token spans,
  labels, masks, and retained Stage-1 semantic parity.

## 5. Stage-2 Projection After Approval

- [ ] 5.1 Route Stage-2 rollout correction through
  `DetectionScene + RolloutPrediction -> DetectionAssignment -> CorrectionEvent
  -> DetectionSupervisionView`.
- [ ] 5.2 Consolidate new Stage-2 rollout/decode/eval policy under the
  Stage-2 rollout-correction surface rather than public `rollout_matching.*`.
- [ ] 5.3 Add characterization tests for assignment, duplicate filtering,
  correction-event construction, invalid/drop metadata, and retained Stage-2
  semantic parity.
- [ ] 5.4 Prove `RolloutPrediction` is derived from shared inference/runtime
  strict decoded output and provenance rather than a private Stage-2 parser or
  diagnostic salvage path.

## 6. Inference And Eval Projection After Approval

- [ ] 6.1 Route inference parse outputs through `DecodedDetectionResult`.
- [ ] 6.2 Route raw eval rows through `DetectionEvalRecord` and scored rows
  through `ScoredDetectionEvalRecord`.
- [ ] 6.3 Keep `gt_vs_pred.jsonl` and `gt_vs_pred_scored.jsonl` artifact
  filenames stable unless a later artifact-contract change renames them.
- [ ] 6.4 Add checks for strict metric-bearing parser status, raw/scored
  separation, score provenance, and artifact compatibility.

## 7. Rename/Delete Cleanup After Approval

- [ ] 7.1 Rename public concepts whose names conflict with the target
  vocabulary.
- [ ] 7.2 Delete or quarantine retired trainer variants, old config roots,
  compatibility facades, and stale script-only workflows that are not retained
  canonical surfaces.
- [ ] 7.3 Update docs/catalog/spec routing so current authority points to the
  new concepts and archived history remains clearly historical.
- [ ] 7.4 Tighten search gates only after the corresponding replacement surface
  exists.
- [ ] 7.5 Do not delete or rename retained canonical behavior until its
  replacement seam and characterization checks are in place.

## 8. Validation After Approval

- [ ] 8.1 Run narrow template/render/parse tests.
- [ ] 8.2 Run tokenization/span/supervision tests.
- [ ] 8.3 Run Stage-1 smoke or config-parse checks for the retained canonical
  surface.
- [ ] 8.4 Run Stage-2 target-construction and assignment tests.
- [ ] 8.5 Run inference decode tests.
- [ ] 8.6 Run eval artifact/scoring tests.
- [ ] 8.7 Run final search gates for retired names in active surfaces.
- [x] 8.8 Run `openspec validate detection-scene-clean-break --strict` before
  requesting implementation approval.
