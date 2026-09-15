---
status: active
scope: detection-scene-clean-break
kind: phase-0-surface-classification
date: 2026-05-31
---

# DetectionScene Phase 0 Surface Classification

This note records the Phase 0 classification gates for the approved
`DetectionScene` clean-break implementation.

Archive checkpoint:

- Branch: `codex/detection-scene-clean-break`
- Checkpoint commit: `5ac77b5a9131d27fd5681b8e4c6a8d233ec72844`
- Checkpoint note:
  `progress/audits/2026-05-31_detection_scene_clean_break_archive_checkpoint.md`

This is not a current-behavior operator guide. It is a cleanup gate for the
implementation work tracked by
`openspec/changes/detection-scene-clean-break/tasks.md`.

## Classification Summary

### Keep Or Rename

| Area | Keep or rename | Direction |
| --- | --- | --- |
| Raw detection intake | `RawDetectionRow`, `RawDetectionObject`, `DetectionMetadata`, `parse_raw_detection_row()` | Keep intake responsibility; expose as `RawDetectionRecord` only where it remains public. Raw rows project into `DetectionScene`; they do not become semantic authority. |
| Semantic detection bridge | `NormalizedDetectionSample`, `NormalizedDetectionObject`, `DetectionDocument` | Replace with `DetectionScene`, `DetectionObject`, and `DetectionGeometry`. Do not promote `DetectionDocument`. |
| Geometry | `CoordinateTokenBox`, bbox/poly helpers, non-inverted `xyxy` validation | Preserve validation and geometry semantics; make bbox/poly kind, coordinate space, coordinate frame, and bbox chart explicit in `DetectionGeometry`. |
| Template owner | `DetectionSequenceTemplate` | Keep as strict render/parse owner. It should consume `DetectionScene` after migration. |
| Rendered sequence | `RenderedAssistantSequence` | Rename or wrap as `RenderedDetectionSequence`; rendered text and span events stay in the render layer, not in `DetectionScene`. |
| Supervision view | `TokenizedDetectionExample`, `TeacherForcingTargetIR`, `SupervisionAtom` | Replace or wrap with `DetectionSupervisionView`; labels, masks, token spans, sidecars, and loss alignment live here. |
| Stage-1 objective | `TeacherForcingObjectiveConfig`, `objective.id: teacher_forcing` | Keep as canonical objective vocabulary behind `stage1_detection_teacher_forcing`. |
| Stage-1 config/root concept | `stage1_compact_trie_ce`, `stage1_json_ce`, `recursive_detection_ce` configs | Keep as temporary/comparator handles until `configs/stage1/detection_teacher_forcing/` exists and parity checks cover retained behavior. |
| Stage-2 public concept | `stage2_rollout_correction`, `Stage2RolloutCorrectionTrainer` | Keep as canonical Stage-2 surface. |
| Shared inference runtime | `DetectionDecodeRequest`, prompt/decode/backend/provenance runtime, strict parser metadata | Keep and strengthen. `RolloutPrediction` and `DecodedDetectionResult` derive from this boundary. |
| Stage-2 parser output | `DetectionParserResult`, `Stage2ParsedRolloutPredictions`, `Stage2RolloutParseResult`, dict rollout views | Rename, wrap, or rehome into `DecodedDetectionResult` and `RolloutPrediction`; private helpers may remain only while classified. |
| Assignment | `GreedyIoUAssignment`, `AssignmentObject`, `AssignmentResult` | Keep behavior; rename result vocabulary toward `DetectionAssignment`. |
| Duplicate filtering | `DuplicateFilter`, `DuplicateCandidate`, `DuplicateFilterResult` | Keep policy before assignment; ensure provenance is explicit and non-survivors do not become positive correction targets. |
| Correction event | `CorrectionEvent` | Keep as canonical term. Rename surrounding residual/channel names where they leak old mechanism concepts. |
| Correction supervision adapter | `build_residual_set_target_ir` and related target-builder flow | Keep behavior; publicly frame as `CorrectionEvent -> DetectionSupervisionView`. |
| Inference/eval artifacts | `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, guarded companions, `raw_rollouts.jsonl`, token trace artifacts | Keep filenames stable. Internal row concepts become `DetectionEvalRecord` and `ScoredDetectionEvalRecord`. |
| Visualization | `vis_resources/gt_vs_pred.jsonl`, `src/vis/gt_vs_pred.py` | Keep as derived review view; stop treating it as a canonical scene schema after scene/eval-record path exists. |
| Manifests/provenance | experiment manifest, pipeline manifest, run metadata, resolved config artifacts | Keep stable until replacement code/config surfaces are current. |

### Temporary Migration Handles

| Handle | Why it may remain temporarily | Removal or rehome gate |
| --- | --- | --- |
| `NormalizedDetectionSample`, `NormalizedDetectionObject` | Current raw-row-to-training bridge. | Remove/rename after `DetectionScene` intake and object ordering/geometry parity checks exist. |
| `DetectionDocument` | Existing semantic-ish adapter, but rejected as target vocabulary. | Remove after `DetectionScene.from_raw_record` or equivalent covers all references. |
| `RenderedAssistantSequence` | Current rendered text/span carrier. | Rename after render/parse characterization covers retained templates. |
| `TokenizedDetectionExample` | Current labels/masks/token span sidecar carrier. | Replace after `DetectionSupervisionView` covers label/mask/span parity. |
| `TeacherForcingTargetIR` | Useful lower-level target IR. | Keep only as private implementation detail beneath `DetectionSupervisionView`, or retire. |
| `recursive_detection_ce` configs and `recursive_detection_ce_cfg` runtime plumbing | Current compact-full parity anchor. | Delete/quarantine only after `stage1_detection_teacher_forcing` config/runtime route exists. |
| `configs/stage1/compact_detection_sequence/` | Legacy compact bridge. | Delete/quarantine after replacement smoke config exists. |
| `configs/stage2_rollout_correction/` | Current active Stage-2 root. | Move toward `configs/stage2/rollout_correction/` only after config loader/docs/routing accept the nested root. |
| top-level `rollout_matching:` config namespace | Current Stage-2 runtime still requires it for backend/decode/eval policy. | Remove after Stage-2 rollout-correction schema owns those knobs and runtime plan no longer requires top-level rollout matching. |
| `RolloutMatchingConfig`, `RolloutEvalDetectionConfig` | Current typed config owners for rollout backend/decode/eval/materialization. | Rehome under Stage-2 rollout-correction or shared inference policy with strict validation preserved. |
| `src/trainers/rollout_matching/*` | Current contracts/parsing/preflight utilities are still referenced. | Quarantine as private adapters only until `RolloutPrediction`, `DetectionAssignment`, and shared strict decode replace them. |
| `parse_rollout_for_matching` salvage parser | Existing CoordJSON/salvage path. | Must not produce metric-bearing `RolloutPrediction`; keep only diagnostic/private until deleted. |
| `stage2_rollout_runtime.py` | Internal rollout execution still active. | Rename/delete public concept only after shared runtime owns train/eval rollout generation. |
| `rollout_aligned_evaluator.py` | Old eval-step artifact writer. | Delete only after replacement writes equivalent raw/scored eval, raw rollouts, token traces, metrics, and summaries. |
| `gt_vs_pred*.jsonl` filenames | Stable artifact handles. | Do not rename in this implementation; any rename requires a separate artifact-contract change. |
| historical docs/progress/OpenSpec references | Preserve research provenance. | Search gates must allow clearly historical references and fail only active current-authority usage. |

### Quarantine Or Delete After Replacement Exists

| Candidate | Classification |
| --- | --- |
| `DetectionDocument` | Delete/rename after `DetectionScene` exists and references are ported. |
| `NormalizedDetectionSample`, `NormalizedDetectionObject` as public concepts | Rename/delete after scene intake parity exists. |
| `RenderedAssistantSequence` as public concept | Rename/delete after `RenderedDetectionSequence` exists. |
| `TokenizedDetectionExample` as public concept | Rename/delete after `DetectionSupervisionView` exists. |
| `recursive_detection_ce` as public objective/config concept | Delete/quarantine after `stage1_detection_teacher_forcing` route and checks exist. |
| `configs/_shared/recursive_detection/` | Rehome shared snippets under teacher-forcing naming before deletion. |
| `configs/stage1/compact_detection_sequence/` | Delete/quarantine after new smoke config exists. |
| public `stage2_ab`, `stage2_two_channel`, `stage2_ab_training` | Keep rejection tests only; no live compatibility adapter required. |
| public `rollout_matching_sft`, `stage2_rollout_aligned`, `stage2_rollout_runtime` | Quarantine/delete public concepts after retained Stage-2 behavior is covered. |
| `rollout_matching.pipeline` | Delete/gate from active configs and current-authority docs; allow only historical/rejection references. |
| public `rollout_matching.*` parser/matching namespace | Rehome/delete after Stage-2 rollout-correction owns rollout/decode/eval policy. |
| `ParsedPredObject`, `RolloutParseResult`, `MatchResult`, `GTObject` | Replace with `DetectionObject`, `RolloutPrediction`, and `DetectionAssignment` concepts. |
| `_ChannelB*`, channel A/B public vocabulary | Rename after rollout-correction target context names exist. |
| metric-bearing salvage paths | Delete or force diagnostic-only `metric_bearing=false`; never feed official eval or Stage-2 metric-bearing rollout eval. |
| old duplicate/objective/loss names | Delete/quarantine active usage after replacement checks exist. |
| `dense_caption`, `BaseCaptionDataset`, `ConversationRecord` as active detection path names | Quarantine/delete if they remain active detection entrypoints after scene intake exists. |

## Search Gates

Run these gates after replacement surfaces exist. Historical/progress docs,
the active OpenSpec change text, explicit archive notes, and rejection tests may
be allowlisted when they are not current runtime authority.

```bash
rg -n "\b(DetectionDocument|NormalizedDetectionSample|NormalizedDetectionObject|RenderedAssistantSequence|TokenizedDetectionExample|EncodedDetectionView)\b" src tests configs docs openspec
```

```bash
rg -n "\b(DetectionScene|DetectionObject|DetectionGeometry|RenderedDetectionSequence|DetectionSequenceTemplate|DetectionSupervisionView)\b" src tests configs docs openspec
```

```bash
rg -n "\b(recursive_detection_ce|compact_detection_sequence|configs/_shared/recursive_detection|stage1_compact_trie_ce|stage1_json_ce)\b" src tests configs docs openspec
```

```bash
rg -n "\b(dense_caption|BaseCaptionDataset|ConversationRecord|RawDetectionRow|RawDetectionObject)\b" src tests configs docs openspec
```

```bash
rg -n "\b(stage2_ab|stage2_two_channel|stage2_ab_training|rollout_matching_sft|stage2_rollout_aligned|stage2_rollout_runtime|rollout_aligned)\b" src tests configs docs openspec
```

```bash
rg -n "rollout_matching(\.|:)|rollout_matching\.pipeline|rollout_matching\.eval_detection" src tests configs docs openspec
```

```bash
rg -n "\b(ParsedPredObject|RolloutParseResult|MatchResult|GTObject|Stage2RolloutParseResult|Stage2ParsedRolloutPredictions|Stage2RolloutObject|Stage2RolloutTarget|RolloutCorrectionTargetContext)\b" src tests configs docs openspec
```

```bash
rg -n "mode=\"salvage\"|salvage_recovered|metric_bearing|diagnostic_parser_result|fallback_gt_fn_append_only|invalid_rollout|dropped_invalid|dropped_ambiguous" src tests configs docs openspec
```

```bash
rg -n "\b(AssignmentResult|AssignmentObject|DuplicateCandidate|DuplicateFilterResult|ResidualObject|ResidualState|ChannelB|Channel B|channel_b|channel_a)\b" src tests configs docs openspec
```

```bash
rg -n "\b(DetectionDecodeResult|DetectionParserResult|DecodedDetectionResult|DetectionEvalRecord|ScoredDetectionEvalRecord)\b" src tests docs openspec
```

```bash
rg -n "gt_vs_pred(_scored)?(\.jsonl|_jsonl)|vis_resources/gt_vs_pred|canonical visualization|canonical sidecar|canonical scene|canonical review resource" src tests configs docs openspec
```

```bash
rg -n "configs/stage2_rollout_correction|configs/stage2/rollout_correction|configs/stage1/recursive_detection_ce|configs/stage1/detection_teacher_forcing|configs/stage1/teacher_forcing" configs docs openspec src
```

## Replacement-Before-Deletion Gates

### Gate A: Raw Intake To Scene

Before deleting or renaming `RawDetectionRow`, `NormalizedDetectionSample`,
`DetectionDocument`, or the current `DetectionTrainingDataset.__getitem__()`
flow:

- define `DetectionScene`, `DetectionObject`, and `DetectionGeometry`;
- add an explicit raw-record-to-scene converter;
- preserve image path/reference resolution, dimensions, image identity,
  coordinate frame, coordinate space, bbox chart, metadata, and source-object
  indices;
- preserve `sorted` and `random_permutation` object ordering semantics;
- preserve bbox validation, object-id requirements, metadata supervision
  snapshots, and no-silent-resize behavior.

### Gate B: Geometry

Before replacing current geometry carriers:

- `DetectionGeometry` must expose concrete bbox/poly kind;
- bbox chart and coordinate space must be explicit;
- outward bbox semantics remain `[x1, y1, x2, y2]` unless separately approved;
- polygon arity and vertex order remain explicit;
- bbox/poly math routes through `src/datasets/geometry.py` unless the edit is
  detection serialization code.

### Gate C: Template Render And Parse

Before deleting or renaming current render/template names:

- `RenderedDetectionSequence` contains rendered detection text and semantic
  span events but no token IDs or labels;
- `DetectionSequenceTemplate` consumes `DetectionScene`;
- `compact_full` and `stage1_json_pretty` render/parse strict behavior is
  characterized;
- compact-full terminal-close and chat-template stop-marker semantics remain
  owned by render/tokenization layers, not by `DetectionScene`.

### Gate D: Supervision View

Before deleting tokenized/target-side carriers:

- `DetectionSupervisionView` owns labels, masks, coordinate spans, sidecars,
  and loss alignment;
- assistant/object/desc/bbox/coord/terminal masks and token roles preserve
  current behavior;
- supervised label positions and next-token prediction mapping preserve current
  behavior;
- `TeacherForcingTargetIR` is either private under `DetectionSupervisionView`
  or replaced by equivalent atoms/metadata.

### Gate E: Stage-1 Public Route

Before deleting `recursive_detection_ce` configs/runtime plumbing:

- add `configs/stage1/detection_teacher_forcing/` with compact-full
  production/smoke replacements;
- keep `objective.id: teacher_forcing` as canonical;
- route `src/sft.py` and `src/detection/runtime.py` without exposing
  `recursive_detection_ce_cfg` as the public concept;
- update docs/catalog only after code/config behavior is current;
- run render/parse/token-supervision checks plus a Stage-1 config parse or
  smoke check.

### Gate F: Stage-2 Rollout Prediction And Assignment

Before deleting `rollout_matching` contracts/parsing, old Stage-2 variants, or
dict-based rollout views:

- `RolloutPrediction` derives from shared runtime strict decoded output,
  parser policy, invalid/drop metadata, prompt/decode/model provenance, and
  metric-bearing state;
- Stage-2 owns only assignment, duplicate filtering, correction events, and
  supervision projection;
- `DetectionAssignment` preserves greedy IoU semantics, stable tie-breaks,
  matched/unmatched prediction and GT handling, and finite non-degenerate box
  requirements;
- duplicate filtering still runs before assignment and non-survivors do not
  become positive targets;
- `CorrectionEvent` construction from dirty-prefix/residual continuation is
  characterized.

### Gate G: Stage-2 Config And Eval Materialization

Before removing top-level `rollout_matching:` or old eval-step writers:

- Stage-2 rollout-correction schema owns backend, decode, confidence, eval,
  monitor, materialization, and provenance policy;
- runtime plan no longer requires top-level rollout matching;
- strict unknown-key behavior remains;
- eval-step replacement writes the same artifact family:
  `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, `raw_rollouts.jsonl`,
  token traces when available, metrics, summaries, and provenance.

### Gate H: Inference/Eval Records

Before exposing new inference/eval vocabulary as current behavior:

- introduce `DecodedDetectionResult` after parser policy, distinct from
  backend-level `DetectionDecodeResult`;
- include parsed predictions, invalid/drop metadata, metric-bearing status,
  parser policy, salvage exclusion, and prompt/decode/model provenance;
- introduce `DetectionEvalRecord` and `ScoredDetectionEvalRecord`;
- preserve raw-vs-scored separation and score provenance;
- preserve `gt_vs_pred.jsonl` and `gt_vs_pred_scored.jsonl` filenames.

### Gate I: Visualization

Before changing visualization documentation or schema language:

- derive visualization resources from `DetectionScene`,
  `DecodedDetectionResult`, `DetectionEvalRecord`, or
  `ScoredDetectionEvalRecord`;
- preserve review-specific ordering, matching overlays, labels, and rendering
  metadata as visualization policy only;
- update "canonical" wording to "derived review view" after the replacement
  path exists.

### Gate J: Docs And Stable Specs

Before updating current-authority docs/specs:

- implementation and configs must exist;
- old names in current docs must be classified as retained, migration,
  rejection, private, historical, or delete;
- stable docs/specs must not claim `DetectionScene` behavior before it is
  runnable or config-parseable through the relevant canonical path.

## Immediate Implementation Implications

- Phase 1 should introduce the scene layer as a real semantic API and should
  not simply rename `DetectionDocument`.
- The first runtime route should be additive and parity-preserving: current raw
  rows can still parse, but canonical detection semantics should be available
  as `DetectionScene`.
- Stage-1 and Stage-2 slices must share the same `DetectionGeometry` and object
  ordering semantics.
- Stage-2 parser cleanup must wait for `RolloutPrediction` from shared strict
  decode/provenance.
- Artifact filenames are not refactor targets in this implementation.
