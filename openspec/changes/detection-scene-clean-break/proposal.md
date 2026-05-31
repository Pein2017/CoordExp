## Why

CoordExp's active detection stack now has enough accumulated Stage-1, Stage-2,
inference, eval, artifact, and config history that the same concept appears as
raw JSONL rows, normalized samples, detection documents, rendered strings,
token sidecars, rollout states, parsed predictions, and eval rows. The next
research direction is a clean restart: the codebase should optimize for a small
set of precise detection concepts rather than carrying historical runnable
compatibility surfaces forward.

This change establishes the durable contract for that clean break before runtime
implementation begins. It defines `DetectionScene` as the central semantic
object for active detection workflows, unifies the Stage-1 and Stage-2 design
around projections from that object, and records which compatibility burdens are
intentionally dropped after an archive checkpoint.

Until the user explicitly approves implementation and the archive checkpoint is
recorded, the current docs and stable specs remain the authority for runnable
behavior. This OpenSpec change is the target contract for the clean-break
implementation phase; it is not permission to edit production runtime code or to
claim that old public names have already been retired.

## What Changes

- Introduce `DetectionScene` as the canonical in-memory semantic object for one
  image's detection meaning: image identity, dimensions, coordinate frame,
  ground-truth objects, ordering policy, and metadata needed by Stage-1,
  Stage-2, inference, and eval projections.
- Define layered representation ownership: raw JSONL records, `DetectionScene`,
  rendered detection sequences, token supervision views, rollout predictions,
  assignment/correction events, decoded results, and eval records each have a
  distinct owner.
- Require Stage-1 and Stage-2 to be designed together:
  - Stage-1 projects `DetectionScene -> RenderedDetectionSequence ->
    DetectionSupervisionView`.
  - Stage-2 projects `DetectionScene + RolloutPrediction ->
    DetectionAssignment -> CorrectionEvent -> DetectionSupervisionView`.
- Require Stage-2 rollout predictions to come through the shared inference /
  runtime parse boundary. `RolloutPrediction` is a Stage-2 rollout-context view
  over strict decoded output, not a second parser or salvage path.
- **BREAKING**: historical execution compatibility is not required after an
  explicit archive checkpoint. Retired configs, public names, import facades,
  script-only workflows, and old namespaces may be deleted or quarantined.
- **BREAKING**: misleading historical public names are not protected. The clean
  stack should retire names such as `dense_caption`, `recursive_detection_ce`,
  `rollout_matching`, `stage2_rollout_runtime`, `rollout_aligned`, `stage2_ab`,
  `set_continuation`, `ConversationRecord`, `NormalizedDetectionSample`, and
  `DetectionDocument` when they conflict with the target vocabulary.
- Preserve hard detection invariants unless separately approved as research
  changes: image/geometry alignment, offline-prepared geometry with no silent
  runtime resize, explicit coordinate frames, explicit object ordering,
  explicit bbox/poly meaning, template-owned chat/render boundaries,
  template/tokenization-derived supervision, explicit Stage-2 correction target
  construction, explicit parse failure/drop metadata, raw/scored eval separation,
  metric/provenance interpretability, and deterministic config resolution.
- Keep `gt_vs_pred.jsonl` and `gt_vs_pred_scored.jsonl` as artifact filenames
  for now, while internal types use `DetectionEvalRecord` and
  `ScoredDetectionEvalRecord`.
- Route implementation sequencing to a Superpowers plan. OpenSpec owns the
  stable architecture contract; the plan owns task ordering and approval gates.
- Require explicit user approval before production-code implementation starts.

## Capabilities

### New Capabilities

- `detection-scene-architecture`: defines the clean-break detection semantic
  architecture, vocabulary, representation ownership, Stage-1/Stage-2
  projection model, archive policy, naming policy, and implementation-readiness
  gates.

### Modified Capabilities

- `runtime-architecture-refactor-program`: changes the runtime-refactor contract
  from compatibility-preserving cleanup to clean-break cleanup for explicitly
  retired surfaces, while preserving hard detection semantics for retained
  canonical surfaces.
- `stage1-detection-objectives`: supersedes public `recursive_detection_ce`
  vocabulary for the new canonical Stage-1 detection teacher-forcing surface.
- `stage2-rollout-correction`: removes `rollout_matching.*` as the target public
  runtime namespace for clean-break Stage-2 and consolidates rollout/decode/eval
  policy under the Stage-2 rollout-correction concept.
- `shared-inference-runtime`: integrates `DetectionScene` with the existing
  prompt/decode/backend/parser/provenance ownership model and treats this change
  as the approved future OpenSpec that may retire `rollout_matching.*` as a
  public online namespace after the archive checkpoint.
- `inference-pipeline`: aligns inference outputs with `DecodedDetectionResult`,
  `DetectionEvalRecord`, and raw/scored artifact policy while keeping
  `gt_vs_pred*` filenames stable.
- `detection-evaluator`: aligns evaluator terminology with
  `DetectionEvalRecord` and `ScoredDetectionEvalRecord` while preserving raw vs
  scored artifact interpretation.
- `gt-vs-pred-visualization`: demotes the visualization resource from a
  competing canonical scene-like schema to a derived review view over
  `DetectionScene` / eval-record concepts.
- `training-config-hierarchy`: defines clean-break config namespace direction
  and removes legacy naming as a required compatibility constraint.

## Impact

Affected future implementation areas:

- `src/detection/` semantic IR, templates, tokenization, and supervision views.
- `src/datasets/` JSONL loading and raw-record-to-scene normalization.
- Stage-1 compact / teacher-forcing detection configs and objective routing.
- Stage-2 rollout-correction target construction, rollout prediction parsing,
  assignment, correction events, and config namespaces.
- `src/infer/` decoded-result and strict parse boundary.
- `src/eval/` eval-record/scored-record concepts and artifact readers/writers.
- `src/config/`, `configs/`, docs routing, and search gates for retired names.

Affected durable artifacts:

- new OpenSpec change: `openspec/changes/detection-scene-clean-break/`
- implementation roadmap: `docs/superpowers/plans/2026-05-31-detection-scene-clean-break.md`
- architecture review context: `docs/architecture/INDEPENDENT_ARCHITECTURE_REVIEW.md`

Non-goals for this change:

- no production-code implementation before explicit user approval;
- no benchmark-improvement requirement as an OpenSpec validity gate;
- no artifact filename rename for `gt_vs_pred.jsonl` or
  `gt_vs_pred_scored.jsonl` in this slice;
- no attempt to define a universal `CoordExpRecord` for non-detection tasks;
- no preservation of historical runnable compatibility after the archive
  checkpoint unless a surface is explicitly retained as canonical;
- no cleanup slice that deletes or renames retained canonical behavior before
  its replacement seam, characterization checks, and search gates are named.
