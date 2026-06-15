# DetectionScene Clean-Break Proposal Review and Implementation Roadmap

> **Implementation approved on 2026-05-31.** This roadmap now governs the clean-break implementation sequence. Production code, configs, stable current-behavior docs, and runtime refactors must still follow the OpenSpec contract and the archive/classification gates below.

**Goal:** Converge the OpenSpec contract and implementation sequence for the clean-break `DetectionScene` architecture.

**Architecture:** OpenSpec owns the durable architecture contract for `DetectionScene`, representation ownership, Stage-1/Stage-2 projections, clean-break compatibility policy, artifact policy, config namespace direction, and verification gates. This Superpowers plan owns sequencing, review lanes, approval boundaries, and the eventual implementation roadmap.

**Tech Stack:** OpenSpec, CoordExp typed YAML configs, Stage-1 detection teacher forcing, Stage-2 rollout correction, inference/eval artifact contracts, pytest/search gates after approval.

---

## Governing Artifacts

Proposal artifacts:

- `openspec/changes/detection-scene-clean-break/proposal.md`
- `openspec/changes/detection-scene-clean-break/design.md`
- `openspec/changes/detection-scene-clean-break/specs/detection-scene-architecture/spec.md`
- `openspec/changes/detection-scene-clean-break/tasks.md`
- `docs/superpowers/plans/2026-05-31-detection-scene-clean-break.md`

Current decision context:

- `docs/architecture/INDEPENDENT_ARCHITECTURE_REVIEW.md`
- `docs/AGENT_INDEX.md`
- `docs/catalog.yaml`

Current adjacent stable specs:

- `openspec/specs/runtime-architecture-refactor-program/spec.md`
- `openspec/specs/stage1-detection-objectives/spec.md`
- `openspec/specs/stage2-rollout-correction/spec.md`
- `openspec/specs/inference-pipeline/spec.md`
- `openspec/specs/detection-evaluator/spec.md`
- `openspec/specs/training-config-hierarchy/spec.md`
- `openspec/specs/object-field-ordering/spec.md`

## Approval Boundary

- [x] User approved detection-only scope.
- [x] User approved layered representation ownership.
- [x] User approved joint Stage-1/Stage-2 design.
- [x] User approved clean break from live historical runtime compatibility.
- [x] User approved hard detection invariants.
- [x] User approved docs/OpenSpec governance for stable contracts.
- [x] User approved concept-driven renaming of misleading historical names.
- [x] User approved `DetectionScene` as the central semantic object.
- [x] User approved keeping `gt_vs_pred*` artifact filenames for now.
- [x] User approved OpenSpec for the main architecture contract and Superpowers
  for implementation sequencing.
- [x] Subagent audits have returned and been incorporated.
- [x] User has reviewed the final decision packet.
- [x] User has explicitly approved production implementation.

Production implementation is approved by direct user instruction and proceeds
through the archive/classification gates before runtime replacement work.

## Read-Only Review Lanes

- [ ] **Lane 1: OpenSpec contract audit**

Review existing specs for conflicts with the new clean-break contract.

Required output:

- existing requirements the new change must preserve or explicitly supersede;
- specs requiring migration notes;
- contradictions between current stable specs and the clean-break proposal.

Disposition: incorporated. The OpenSpec now explicitly amends
`runtime-architecture-refactor-program`, adds `shared-inference-runtime` and
`gt-vs-pred-visualization` deltas, keeps `DetectionScene` semantic-only, and
states that `gt_vs_pred*.jsonl` schemas are not silently redefined.

- [ ] **Lane 2: Current docs audit**

Review current docs for language that conflicts with `DetectionScene`,
clean-break naming, or Stage-1/Stage-2 projection ownership.

Required output:

- docs that should update only after implementation;
- docs that should remain historical;
- docs that may need routing changes once implementation starts.

Disposition: incorporated. Current docs remain current-behavior authority until
implementation. Older architecture/superpowers docs using `CoordExpRecord` or
`DetectionDocument` are treated as pre-decision context, not implementation
authority.

- [ ] **Lane 3: Naming and surface classification audit**

Review source/config/docs names enough to classify surfaces.

Required output:

- keep/rename candidates;
- temporary migration-handle candidates;
- quarantine/delete candidates;
- public names to avoid;
- search-gate seeds.

Disposition: incorporated. The OpenSpec and plan preserve the accepted target
vocabulary, keep `gt_vs_pred*` filenames as temporary/stable artifact handles,
and classify high-risk historical names for future keep/rename, migration,
quarantine, or delete decisions.

## Subagent Convergence Register

| Lane | Status | Main findings | Disposition |
| --- | --- | --- | --- |
| OpenSpec contract audit | Returned | Existing specs require explicit amendment for clean break; `DetectionScene` must not absorb prompt/decode/trainer/evaluator ownership; raw/scored/guarded artifacts and strict metric-bearing parser semantics remain protected; visualization resource overlaps scene semantics. | Added `runtime-architecture-refactor-program`, `shared-inference-runtime`, and `gt-vs-pred-visualization` deltas. Added semantic-only and artifact-schema-stability language. |
| Current docs audit | Returned | Older proposal/roadmap docs center `CoordExpRecord` or `DetectionDocument`; current system/data/training/eval docs should update only after implementation; current strongest aligned doc is the independent architecture review decision block. | Kept OpenSpec as normative target, left current docs as current-behavior authority, and deferred domain-doc updates to post-implementation slices. |
| Naming/surface classification audit | Returned | Preserve `DetectionDecodeRequest`, `CorrectionEvent`, `MetricEvent`, artifact filenames, and concrete template IDs; rename/consolidate `DetectionDocument`, `NormalizedDetectionSample`, `RenderedAssistantSequence`, `EncodedDetectionView`, assignment result names; quarantine/delete public `recursive_detection_ce`, `stage2_ab`, `stage2_two_channel`, `stage2_rollout_runtime`, `rollout_aligned`, and `rollout_matching.pipeline`. | Incorporated into target vocabulary, migration-handle policy, and future classification gates. |

## Implementation Roadmap After Approval

### Phase 0: Archive and classify

- [ ] Identify the pre-cleanup archive checkpoint.
- [ ] Record the archive checkpoint in a durable surface.
- [ ] Classify surfaces as keep/rename, temporary migration handle, quarantine,
  or delete.
- [ ] Name search gates for retired public concepts.

### Phase 1: DetectionScene semantic layer

- [ ] Introduce `DetectionScene`, `DetectionObject`, and `DetectionGeometry`.
- [ ] Route raw detection JSONL into the semantic layer.
- [ ] Preserve geometry/image alignment, coordinate frame, bbox/poly meaning,
  and object ordering invariants for retained canonical surfaces.

### Phase 2: Stage-1 projection

- [ ] Route Stage-1 detection teacher forcing through
  `DetectionScene -> RenderedDetectionSequence -> DetectionSupervisionView`.
- [ ] Replace `recursive_detection_ce` as the public target concept for new
  canonical Stage-1 configs/docs.
- [ ] Add render/parse/token-span/supervision characterization tests.

### Phase 3: Stage-2 projection

- [ ] Route Stage-2 rollout correction through
  `DetectionScene + RolloutPrediction -> DetectionAssignment -> CorrectionEvent
  -> DetectionSupervisionView`.
- [ ] Move new public rollout/decode/eval policy out of `rollout_matching.*`
  and under the Stage-2 rollout-correction surface.
- [ ] Add target-construction, assignment, duplicate-filter, correction-event,
  invalid/drop, and semantic-parity tests.

### Phase 4: Inference and eval projection

- [ ] Route parsed inference outputs through `DecodedDetectionResult`.
- [ ] Route raw eval rows through `DetectionEvalRecord`.
- [ ] Route scored eval rows through `ScoredDetectionEvalRecord`.
- [ ] Preserve `gt_vs_pred.jsonl` and `gt_vs_pred_scored.jsonl` filenames.
- [ ] Add strict parser, metric-bearing, raw/scored separation, score provenance,
  and artifact compatibility checks.

### Phase 5: Rename/delete cleanup

- [ ] Rename public concepts whose names conflict with the target vocabulary.
- [ ] Delete or quarantine retired trainer variants, old config roots,
  compatibility facades, stale script-only workflows, and unneeded bridge paths.
- [ ] Update `docs/AGENT_INDEX.md`, `docs/catalog.yaml`, domain docs, and stable
  specs only after implementation makes the new behavior current.
- [ ] Tighten search gates once replacement surfaces exist.

## Proposed Verification Commands After Approval

These commands are not approved to run by this planning file alone. They are the
verification path for the implementation phase.

```bash
python -m pytest tests -q -k "detection_template or tokenization or supervision"
python -m pytest tests -q -k "stage2 and (assignment or correction or rollout)"
python -m pytest tests -q -k "infer or decode or evaluator or gt_vs_pred"
openspec validate detection-scene-clean-break --strict
```

Exact targets should be narrowed to the files touched by each implementation
slice before execution.

## No-Code Stop Condition

This proposal-review plan is complete when:

- OpenSpec artifacts are drafted;
- subagent audit findings are incorporated or explicitly deferred;
- changed files and unresolved risks are summarized;
- validation status is reported;
- the user can approve implementation, request revisions, split the change, or
  abandon the change.

Until then, production implementation remains blocked.
