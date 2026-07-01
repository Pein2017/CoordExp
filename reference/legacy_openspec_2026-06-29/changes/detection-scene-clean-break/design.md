## Context

The independent architecture review found that CoordExp is converging toward a
real architecture but still lacks a single semantic center for active detection
workflows. The current stack repeatedly transforms the same image/object/geometry
meaning through raw JSONL rows, dense-caption records, normalized detection
samples, detection-document adapters, rendered detection strings, token sidecars,
rollout correction inputs, decoded predictions, and eval rows.

The user has approved a clean restart direction: do not preserve live backward
compatibility for historical execution paths. Preserve research provenance
through an archive checkpoint, then make the current codebase the clean next
research stack.

The design goal is not a broad framework. The design goal is a small vocabulary
and a set of deep seams that explain the active detection stack:

```text
RawDetectionRecord
  -> DetectionScene
  -> stage-specific projection
  -> training, inference, or eval artifact
```

### Activation boundary

This change describes the target clean-break contract, not current runnable
behavior. Until the user approves production implementation and the archive
checkpoint is recorded, current docs and stable specs remain the authority for
existing workflows.

Implementation may proceed only after the approval packet identifies the
archive checkpoint, retained canonical surfaces, parity expectations, cleanup
search gates, and verification plan. The clean-break contract permits deletion
after those gates, but it does not justify deleting retained behavior before its
replacement seam exists.

## Key Decisions

### Detection-only first

The canonical semantic seam is detection-only. This change does not define a
universal `CoordExpRecord` for all possible future multimodal tasks.

The included scope is active detection JSONL, Stage-1 detection teacher forcing,
Stage-2 rollout correction, inference decode, and official eval artifacts.
Historical fusion surfaces, retired public trainer variants, stale script-only
workflows, and old compatibility facades are excluded unless explicitly
classified as retained canonical surfaces.

### Central object: DetectionScene

The central semantic object is `DetectionScene`.

`DetectionScene` represents one image's detection semantics:

- image identity and source metadata;
- exactly one resolved image reference;
- image dimensions;
- explicit coordinate frame, coordinate space, and bbox chart;
- ground-truth `DetectionObject` values;
- explicit `DetectionGeometry` values such as bbox or poly;
- object ordering policy;
- metadata required by Stage-1, Stage-2, inference, and eval projections.

`DetectionScene` must not contain rendered assistant text, token IDs, token
spans, rollout assignments, correction events, raw backend output, eval metric
fields, or visualization layout metadata. Those are projections or artifacts
owned by other layers.

The scene layer may represent canonical `xyxy` bbox semantics or explicitly
declared offline model-facing bbox branches, but it must never infer a bbox
chart from position alone. Downstream inference, eval, and visualization
boundaries remain canonical `xyxy` unless a separate artifact/schema decision
approves another outward representation.

`DetectionDocument` is rejected as the target public concept because it is too
text/document-shaped for a visual geometry stack.

`DetectionScene` is a semantic interchange object, not a broad runtime
framework. It must not absorb prompt rendering, backend decode, Stage-2 target
IR construction, duplicate-control policy, evaluator math, DDP coordination,
loss execution, or training metric projection. Those remain owned by their
existing or successor layers and consume scene/projection objects through
explicit interfaces.

### Representation ownership

No layer may treat another layer's representation as canonical.

| Representation | Owner | Canonical only for |
| --- | --- | --- |
| `RawDetectionRecord` | data/dataset loading | on-disk JSONL and intake contract |
| `DetectionScene` | detection domain code | in-memory detection semantics |
| `RenderedDetectionSequence` | detection template code | prompt/completion formatting and semantic span events |
| `DetectionSupervisionView` | detection tokenization/objective code | labels, masks, coordinate spans, sidecars, and loss alignment |
| `RolloutPrediction` | Stage-2 rollout adapter over shared runtime decode output | parsed/generated rollout prediction plus rollout context before Stage-2 assignment |
| `DetectionAssignment` | Stage-2 correction modules | GT/pred matching and assignment decisions |
| `CorrectionEvent` | Stage-2 correction modules | residual correction action derived from GT plus rollout state |
| `DecodedDetectionResult` | inference/runtime parse boundary | parsed predictions plus invalid/drop metadata |
| `DetectionEvalRecord` | eval/artifact code | raw GT/pred comparison rows |
| `ScoredDetectionEvalRecord` | eval/artifact code | scored comparison rows with metric annotations |

### Stage-1 and Stage-2 are designed together

Stage-1 is not the architecture center. Stage-2 is not an afterthought. They are
projections from the same scene semantics.

```text
Stage-1:
DetectionScene
  -> RenderedDetectionSequence
  -> DetectionSupervisionView
```

```text
Stage-2:
DetectionScene + RolloutPrediction
  -> DetectionAssignment
  -> CorrectionEvent
  -> DetectionSupervisionView
```

The first implementation slice may prove the seam through Stage-1 because it is
smaller, but the design and tests must show how Stage-2 consumes the same
semantic layer.

Stage-2 rollout prediction construction must reuse shared inference/runtime
prompt, decode, parser, trace, and provenance ownership. Stage-2 owns
assignment and correction-event construction; it must not parse raw backend
text, independently salvage predictions for metric-bearing training/eval, or
use Stage-1 rendered targets as rollout semantics.

### Clean break with archive checkpoint

Historical runnable compatibility is not required after the archive checkpoint.
The codebase may remove old configs, imports, aliases, scripts, and public names
that are not part of the new canonical stack.

Before large cleanup, create or identify a stable pre-cleanup commit, branch, or
tag and record it as the historical compatibility boundary. Old behavior is then
preserved by git history, archived artifacts, progress notes, and explicit
historical docs, not by live compatibility code.

### Behavior preservation means retained semantic parity

Under the clean-break policy, behavior-preserving means semantic parity for
retained canonical surfaces and hard detection invariants. It does not mean old
paths remain runnable.

Allowed after archive checkpoint:

- delete retired paths;
- rename misleading public concepts;
- remove compatibility facades;
- replace historical config namespaces;
- break old imports, scripts, and configs outside the retained canonical stack.

Deletion order is still constrained. A retained canonical behavior must not be
removed until its clean-break replacement path and characterization checks are
named. A historical path may be deleted earlier only after it is classified as
delete or quarantine and its absence is covered by the archive checkpoint and
search gates.

Requires separate semantic-change approval:

- coordinate frame changes;
- object ordering changes;
- bbox/poly meaning changes;
- retained template semantics changes;
- retained token label or mask changes;
- retained Stage-2 assignment/correction semantic changes;
- eval metric meaning changes;
- raw/scored artifact interpretation changes.

### Naming direction

Target public vocabulary:

| Concept | Canonical term |
| --- | --- |
| Semantic GT/image/object unit | `DetectionScene` |
| One annotated object | `DetectionObject` |
| Geometry value | `DetectionGeometry` |
| Rendered target text | `RenderedDetectionSequence` |
| Strict template owner | `DetectionSequenceTemplate` |
| Token-level training projection | `DetectionSupervisionView` |
| Model rollout parse | `RolloutPrediction` |
| GT/pred matching result | `DetectionAssignment` |
| Stage-2 target unit | `CorrectionEvent` |
| Inference output | `DecodedDetectionResult` |
| Raw eval comparison row | `DetectionEvalRecord` |
| Scored eval row | `ScoredDetectionEvalRecord` |

Target public surface direction:

- Stage-1 config/root concept: `stage1_detection_teacher_forcing`.
- Stage-2 trainer/concept: `stage2_rollout_correction` remains accurate.
- Stage-2 rollout/decode/eval policy should be owned under
  `stage2_rollout_correction` rather than public `rollout_matching.*`.
- Offline inference remains under `infer.*`.
- Artifact filenames `gt_vs_pred.jsonl` and `gt_vs_pred_scored.jsonl` remain
  stable for now.

Historical names to avoid as public concepts include `dense_caption`,
`recursive_detection_ce`, `compact_detection_sequence`, `detection_sequence` when
it hides scene/render/supervision distinctions, `rollout_matching`,
`stage2_rollout_runtime`, `rollout_aligned`, `stage2_ab`, `set_continuation`,
`BaseCaptionDataset`, `ConversationRecord`, `NormalizedDetectionSample`, and
`DetectionDocument`.

### Config direction

New canonical configs should be organized by clear stage and mechanism names:

```text
configs/stage1/detection_teacher_forcing/
configs/stage2/rollout_correction/
```

Stage-2 should consolidate rollout prompt/decode/backend/eval authoring under
the Stage-2 rollout-correction surface rather than requiring a separate public
`rollout_matching` namespace.

The exact typed schema may be introduced incrementally, but new stable config
names must not expose historical concepts as the public route.

### Artifact policy

Keep these filenames in the clean-break stack:

- `gt_vs_pred.jsonl`
- `gt_vs_pred_scored.jsonl`

Use these internal concepts:

- `DetectionEvalRecord`
- `ScoredDetectionEvalRecord`

For this change, `DetectionEvalRecord` and `ScoredDetectionEvalRecord` are
conceptual/internal row names for the existing artifact family. This proposal
does not silently change the `gt_vs_pred*.jsonl` line schema. If a later
implementation introduces a materially different scene artifact family, it must
use a distinct artifact contract, for example a new scene artifact, or receive a
separate artifact-schema decision before reusing existing filenames.

A future artifact filename rename would be a separate artifact-contract change.

### Metric-bearing and diagnostic states

Official eval, confidence post-op, COCO/LVIS/mAP, comparable reports, and
Stage-2 metric-bearing rollout eval must consume strict metric-bearing decoded
results or eval records. Diagnostic salvage may produce non-metric diagnostic
records, but it must not silently enter `DecodedDetectionResult`,
`DetectionEvalRecord`, `ScoredDetectionEvalRecord`, or guarded official eval
inputs as a metric-bearing prediction.

Scene-derived views that participate in official metrics must be able to resolve
prompt, decode, model identity, parser policy, score policy, and metric-bearing
status provenance. Historical artifacts without that provenance may remain
readable for inspection, but they are not comparable unless a migration tool
reconstructs and stamps exact provenance.

### Visualization as derived view

Existing `gt_vs_pred` visualization resources overlap with the proposed scene
concept. In the clean-break architecture, visualization resources are derived
review views over `DetectionScene`, decoded results, and eval-record concepts.
They are not a second canonical scene schema.

## Implementation Readiness Gates

The implementation phase is not approved by this proposal alone. Before runtime
code cleanup starts, the implementation plan must identify:

1. archive checkpoint gate: commit, branch, or tag to preserve old state;
2. surface classification gate: keep/rename, temporary migration handle,
   quarantine, or delete;
3. semantic parity gate: retained surfaces and invariants requiring parity;
4. Stage-1 projection gate;
5. Stage-2 projection gate;
6. inference/eval projection gate;
7. naming gate;
8. config namespace gate;
9. verification gate;
10. governance gate.

## Sequencing

Recommended implementation sequence after user approval:

1. Record the archive checkpoint.
2. Classify active/historical surfaces.
3. Introduce `DetectionScene` and scene-owned object/geometry types.
4. Route Stage-1 teacher-forcing projection through scene/render/supervision
   boundaries.
5. Route Stage-2 rollout correction through scene/prediction/assignment/event
   boundaries, with rollout prediction parsing derived from the shared
   inference/runtime decode boundary.
6. Route inference strict parse outputs to `DecodedDetectionResult`.
7. Route eval concepts to `DetectionEvalRecord` and
   `ScoredDetectionEvalRecord` while keeping `gt_vs_pred*` filenames.
8. Rename public concepts and config roots.
9. Delete or quarantine historical paths.
10. Promote implemented current behavior into domain docs and stable specs.

## Review Boundary

This change may be reviewed, revised, and validated before implementation. It
must not be treated as approval to edit production runtime code. The user must
explicitly approve implementation after reviewing the OpenSpec and Superpowers
plan.
