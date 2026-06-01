---
status: active
scope: detection-scene-clean-break
kind: cleanup-validation
date: 2026-06-01
---

# DetectionScene Cleanup And Validation Gate

This note records the final cleanup/validation slice for
`openspec/changes/detection-scene-clean-break/tasks.md` phases 7 and 8.

Archive/classification inputs:

- `progress/audits/2026-05-31_detection_scene_phase0_surface_classification.md`
- `progress/audits/2026-05-31_detection_scene_clean_break_archive_checkpoint.md`
- `openspec/changes/detection-scene-clean-break/design.md`

## Cleanup Performed

- Removed the retired semantic/render/token migration handles from the root
  `src.detection` package export surface:
  - `DetectionDocument`
  - `DetectionDocumentGeometry`
  - `NormalizedDetectionSample`
  - `NormalizedDetectionObject`
  - `RenderedAssistantSequence`
  - `TokenizedDetectionExample`
  - `detection_document_from_normalized_sample`
- Kept those names only in their module-local implementation/migration owners
  where current code still requires the bridges:
  - `src/detection/data.py`
  - `src/detection/ir.py`
  - `src/detection/template.py`
  - `src/detection/tokenization.py`
  - Stage-1 and Stage-2 projection adapters that still consume the migration
    bridge internally.
- Updated tests so root package characterization now proves canonical public
  names only:
  - `DetectionScene`
  - `DetectionObject`
  - `DetectionGeometry`
  - `RenderedDetectionSequence`
  - `DetectionSupervisionView`
- Updated the compact-full encoding golden to match the strict template owner:
  `CompactFullTemplate.capabilities.object_separator == ""`; compact-full rows
  are concatenated without newline separators.
- Tightened current routing docs and catalog status so old Stage-1 config roots
  and shadow IDs are classified as implementation-only or quarantined rather
  than current public authority:
  - `configs/stage1/detection_teacher_forcing/` is the canonical public
    Stage-1 detection teacher-forcing route.
  - `configs/stage1/recursive_detection_ce/` is quarantined legacy/comparator
    material only.
  - `configs/_shared/recursive_detection/` is quarantined legacy authoring
    material and is not consumed by canonical launch configs.
  - `stage1_json_ce` and `stage1_compact_trie_ce` are implementation-shadow
    resolver IDs, not public config routes.
- Classified `rollout_matching.*` prompt/decode/eval keys in current docs as
  retained temporary migration handles. They are not the target public namespace
  for new clean-break Stage-2 schema work.

## Search Gate Results

Final gates were run after cleanup using the required `rtk grep` patterns over
`src tests configs docs openspec` or the scoped path set named by the gate.

Summary:

| Gate | Result | Classification |
| --- | ---: | --- |
| Old semantic carriers | 440 matches in 58 files | No root `src.detection` public exports remain. Active source matches are module-local migration/private bridge details or analysis probes. Historical docs/OpenSpec/progress references are allowed. |
| Old Stage-1 names | 639 matches in 92 files | Canonical route exists under `configs/stage1/detection_teacher_forcing/`. `recursive_detection_ce` roots are quarantined legacy/comparator material; historical docs and rejection/legacy tests are allowed. |
| Raw/dense-caption names | 191 matches in 61 files | `RawDetectionRow`/`RawDetectionObject` and `dense_caption` remain intake/file-level implementation details. They do not define semantic authority; current semantic authority is `DetectionScene`. |
| Old Stage-2 variants | 1457 matches in 248 files | Active split variants are removed/rejected. Remaining references are historical docs/specs, strict rejection paths, private implementation file names, or search seeds. |
| `rollout_matching.*` namespace | 943 matches in 164 files | `rollout_matching.pipeline` is retired/rejected. Prompt/decode/backend/eval keys remain classified temporary migration handles until Stage-2 rollout-correction schema owns those knobs end to end. |
| Old rollout result names | 419 matches in 25 files | Active public concepts are `RolloutPrediction`, `DetectionAssignment`, and `CorrectionEvent`. Remaining old names are private adapters, analysis probes, or retired rollout-matching module internals. |
| Salvage/drop terms | 892 matches in 90 files | Strict metric-bearing decode/eval paths are covered by tests. Salvage/fallback/drop terms remain diagnostic/fallback policy names and do not become official metric-bearing predictions. |
| Assignment/channel terms | 492 matches in 69 files | Current Stage-2 assignment is greedy IoU through the rollout-correction surface. Old channel vocabulary remains only in historical specs/docs. |
| Decode/eval record terms | 206 matches in 36 files | `DecodedDetectionResult`, `DetectionEvalRecord`, and `ScoredDetectionEvalRecord` are active clean-break concepts. Backend-level `DetectionDecodeResult` remains distinct and allowed. |
| Artifact/visualization terms | 881 matches in 169 files | `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, and `vis_resources/gt_vs_pred.jsonl` are retained stable artifact/review handles by design. |
| Config route terms | 193 matches in 41 files | Canonical Stage-1 route is active; nested Stage-2 rollout-correction route exists. Old Stage-1 recursive roots are quarantined; historical route mentions are allowed. |

Allowed leftovers:

- Historical docs, progress notes, and OpenSpec archive/change text.
- Rejection tests and strict schema error messages for retired variants.
- Private implementation/migration adapters that still sit below canonical
  `DetectionScene`, `RenderedDetectionSequence`, `DetectionSupervisionView`,
  `RolloutPrediction`, `DetectionAssignment`, `CorrectionEvent`,
  `DecodedDetectionResult`, `DetectionEvalRecord`, and
  `ScoredDetectionEvalRecord` seams.
- Stable artifact filenames `gt_vs_pred.jsonl` and `gt_vs_pred_scored.jsonl`.

## Validation Results

Commands run:

```bash
python - <<'PY'
import pytest
raise SystemExit(pytest.main([
    'tests/test_detection_scene_contract.py',
    'tests/test_detection_scene_stage1_projection.py',
    'tests/test_compact_full_encoding_contract.py',
    'tests/test_compact_span_projector.py',
    '-q',
]))
PY
```

Result: `68 passed in 1.34s`.

```bash
python - <<'PY'
import pytest
raise SystemExit(pytest.main([
    'tests/test_detection_ir_contract.py',
    '-q',
]))
PY
```

Result: `7 passed in 1.03s`.

```bash
python - <<'PY'
import pytest
raise SystemExit(pytest.main([
    'tests/test_stage2_rollout_correction_target_boundary.py',
    '-q',
]))
PY
```

Result: `19 passed, 2 warnings in 3.96s`.

```bash
python - <<'PY'
import pytest
raise SystemExit(pytest.main([
    'tests/test_detection_scene_phase4_infer_eval_projection.py',
    '-q',
]))
PY
```

Result: `8 passed in 1.00s`.

```bash
python - <<'PY'
import pytest
raise SystemExit(pytest.main([
    'tests/test_detection_training_config_contract.py',
    'tests/test_stage2_rollout_correction_profile_leaf_contract.py',
    'tests/test_training_config_hierarchy_contract.py',
    'tests/test_training_surface_resolver.py',
    '-q',
]))
PY
```

Result: `50 passed, 96 skipped in 4.15s`.

```bash
python - <<'PY'
import pytest
raise SystemExit(pytest.main([
    'tests/test_confidence_postop.py',
    'tests/test_detection_eval_ingestion_diagnostics.py',
    'tests/test_detection_eval_output_parity.py',
    '-q',
]))
PY
```

Result: `40 passed in 1.38s`.

```bash
python - <<'PY'
from src.config.loader import ConfigLoader
cfg = ConfigLoader.load_materialized_training_config(
    'configs/stage1/detection_teacher_forcing/smoke/compact_full_tiny.yaml'
)
print(type(cfg).__name__)
print(cfg.detection_template.id)
print(cfg.objective.id)
print(cfg.data.train_jsonl)
PY
```

Result:

```text
DetectionTrainingConfig
compact_full
teacher_forcing
public_data/coco/rescale_32_1024_bbox_max60/train.coord.jsonl
```

```bash
openspec validate detection-scene-clean-break --strict
```

Result: `Change 'detection-scene-clean-break' is valid`.

Additional attempted Stage-1 config command:

```bash
PYTHONPATH=. python -m src.sft \
  --config configs/stage1/detection_teacher_forcing/smoke/compact_full_tiny.yaml \
  --cfg-only
```

Result: failed before config-only completion because ms-swift attempted to
resolve `model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp` through
ModelScope and raised `Invalid repo_id: model, must be of format
namespace/name`. The side-effect-free `ConfigLoader.load_materialized_training_config`
check above was used as the Stage-1 config-parse evidence.

## Task Evidence

- 7.1: root public concepts now expose canonical names only; retired names are
  private/module-local or historical.
- 7.2: retired roots/facades were deleted from root public exports or
  quarantined in docs/catalog. Risky runtime-required migration handles were
  classified rather than deleted.
- 7.3: current routing points to `stage1_detection_teacher_forcing` and
  rollout-correction surfaces; old Stage-1 roots and shadow IDs are marked
  quarantined/implementation-only.
- 7.4: final gates were run after replacement surfaces existed and this note
  records the tightened classification.
- 7.5: no retained canonical behavior, stable artifact filename, geometry
  semantic, object-ordering policy, or Stage-2 assignment/correction behavior
  was deleted or renamed.
- 8.1-8.7: validation and search-gate evidence are recorded above.

## Remaining Concerns

- `rollout_matching.*` remains a live prompt/decode/backend/eval migration
  namespace in active Stage-2 config and runtime code. It is classified as a
  temporary migration handle, not deleted, because Gate G requires Stage-2
  rollout-correction schema ownership before removal.
- Module-local old names such as `NormalizedDetectionSample`,
  `RenderedAssistantSequence`, and `TokenizedDetectionExample` remain below the
  canonical seam as private adapters. Deleting them would require a larger
  internal migration than this final cleanup slice.
