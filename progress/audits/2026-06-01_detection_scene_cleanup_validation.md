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

- Removed retired migration names from the root `src.detection` public export
  surface while preserving private/module-local implementation bridges where
  current code still needs them:
  - `DetectionDocument`
  - `DetectionDocumentGeometry`
  - `NormalizedDetectionSample`
  - `NormalizedDetectionObject`
  - `RenderedAssistantSequence`
  - `TokenizedDetectionExample`
  - `detection_document_from_normalized_sample`
  - `compute_recursive_detection_ce_batch_loss`
  - `normalize_recursive_detection_token_losses`
  - `RecursiveDetectionTargets`
  - `RecursiveDetectionLossResult`
  - `RecursiveDetectionLossWeights`
- Kept and characterized the canonical root public names:
  - `DetectionScene`
  - `DetectionObject`
  - `DetectionGeometry`
  - `RenderedDetectionSequence`
  - `DetectionSequenceTemplate`
  - `DetectionSupervisionView`
- Quarantined retired Stage-1 recursive-detection config roots under
  `configs/archive/detection_scene_clean_break/stage1/`:
  - `configs/archive/detection_scene_clean_break/stage1/recursive_detection_ce/`
  - `configs/archive/detection_scene_clean_break/stage1/shared_recursive_detection/`
- Kept the canonical Stage-1 public route under
  `configs/stage1/detection_teacher_forcing/` with
  `objective.id: teacher_forcing`.
- Moved the Stage-2 rollout-correction config root from
  `configs/stage2_rollout_correction/` to
  `configs/stage2/rollout_correction/` and updated loader, launcher, docs,
  specs, scripts, and tests that point at current Stage-2 config paths.
- Kept the retained behavior/config namespace `stage2_rollout_correction` and
  stable output/artifact strings. The directory move does not rename the trainer
  variant, metric namespace, manifest family, or historical output roots.
- Classified `rollout_matching.*` prompt/decode/backend/eval keys as retained
  migration/runtime handles. `rollout_matching.pipeline` remains retired and
  must not own Stage-2 objectives.
- Updated current docs/catalog/spec routing so active authority points to
  `stage1_detection_teacher_forcing`, `DetectionScene` projection vocabulary,
  and `configs/stage2/rollout_correction/`; historical/progress/OpenSpec archive
  references remain historical.
- Marked the legacy instance-trie recursive-detection config-diff test as
  historical/skipped because the archived `recursive_detection_ce` YAML is no
  longer a materializable current config under the clean-break parser.

## Search Gate Results

Final gates were run after cleanup using the required patterns over
`src tests configs docs scripts openspec/specs
openspec/changes/detection-scene-clean-break progress/audits`, excluding this
note itself and generated `__pycache__` files from the broad-count snapshot.

Summary:

| Gate | Result | Classification |
| --- | ---: | --- |
| Old semantic carriers | 495 matches in 56 files | Root `src.detection` exports are absent. Active source matches are private/module-local migration bridges or analysis/test characterization; historical docs/OpenSpec/progress references are allowed. |
| Canonical scene names | 543 matches in 74 files | Canonical names are present across source/tests/docs/OpenSpec, including root public exports. |
| Old Stage-1 names | 1170 matches in 121 files | Live Stage-1 route is `configs/stage1/detection_teacher_forcing/`. Recursive-detection config YAML is quarantined under `configs/archive/detection_scene_clean_break/stage1/`; remaining old names are historical docs, metrics, fixtures, or explicit legacy/rejection context. |
| Raw/dense-caption names | 195 matches in 42 files | Raw/dense-caption names remain intake/file-level implementation details and historical docs; semantic authority is `DetectionScene`. |
| Old Stage-2 variants | 730 matches in 79 files | Retired public variants remain as rejection tests, historical docs/specs, private implementation file names, and search seeds. Active trainer concept remains `stage2_rollout_correction`. |
| `rollout_matching.*` namespace | 467 matches in 82 files | `rollout_matching.pipeline` is retired/rejected. Prompt/decode/backend/eval keys remain classified migration/runtime handles until Stage-2 rollout-correction schema owns those knobs end to end. |
| Old rollout parse/assignment names | 483 matches in 28 files | Active public concepts are `RolloutPrediction`, `DetectionAssignment`, and `CorrectionEvent`; remaining old names are private adapters, legacy tests, analysis probes, or retired rollout-matching internals. |
| Salvage/drop terms | 1004 matches in 73 files | Strict metric-bearing decode/eval paths are covered by tests. Salvage/fallback/drop terms remain diagnostic/fallback policy names and do not become official metric-bearing predictions. |
| Assignment/channel terms | 342 matches in 35 files | Current Stage-2 assignment is greedy IoU through the rollout-correction surface. Old channel vocabulary remains only in historical specs/docs, private internals, or rejection/legacy tests. |
| Decode/eval record terms | 227 matches in 35 files | `DecodedDetectionResult`, `DetectionEvalRecord`, and `ScoredDetectionEvalRecord` are active clean-break concepts. Backend-level `DetectionDecodeResult` remains distinct and allowed. |
| Artifact/visualization terms | 779 matches in 129 files | `gt_vs_pred.jsonl`, `gt_vs_pred_scored.jsonl`, and `vis_resources/gt_vs_pred.jsonl` are retained stable artifact/review handles by design. |
| Config route terms | 253 matches in 65 files | Canonical Stage-1 route and nested Stage-2 route are active. Old live Stage-1/Stage-2 root directories are absent; remaining route strings are historical, archive, fixture, or current canonical mentions. |

Hard absence/presence checks:

```text
old_stage1_root_absent: True
old_shared_recursive_root_absent: True
stage2_old_root_absent: True
stage2_nested_root_present: True
stage1_canonical_smoke_present: True
stage1_archive_present: True
root_export_absent_DetectionDocument: True
root_export_absent_DetectionDocumentGeometry: True
root_export_absent_NormalizedDetectionSample: True
root_export_absent_NormalizedDetectionObject: True
root_export_absent_RenderedAssistantSequence: True
root_export_absent_TokenizedDetectionExample: True
root_export_absent_detection_document_from_normalized_sample: True
root_export_absent_compute_recursive_detection_ce_batch_loss: True
root_export_absent_normalize_recursive_detection_token_losses: True
root_export_absent_RecursiveDetectionTargets: True
root_export_absent_RecursiveDetectionLossResult: True
root_export_absent_RecursiveDetectionLossWeights: True
root_export_present_DetectionScene: True
root_export_present_DetectionObject: True
root_export_present_DetectionGeometry: True
root_export_present_RenderedDetectionSequence: True
root_export_present_DetectionSequenceTemplate: True
root_export_present_DetectionSupervisionView: True
```

Exact old active config-root search after the follow-up cleanup:

```text
rg -n "configs/stage2_rollout_correction|configs/stage1/recursive_detection_ce|configs/_shared/recursive_detection" src configs docs tests scripts openspec/specs --glob '!configs/archive/**'
Result: 130 matches in 17 files.
```

No active-authority files in `src`, `configs`, `docs/data`, `docs/architecture`,
`docs/training` runbooks, `scripts`, or `openspec/specs` point current routing at
the old roots. The remaining matches are dated historical `docs/superpowers/`
plans/handoffs/spec drafts and fixture/rejection references in:

- `tests/test_detection_training_config_contract.py`
- `tests/test_recursive_detection_ce_sft_wiring.py`

These are allowed historical/fixture/rejection leftovers, not live config roots.

Allowed leftovers:

- Historical docs, progress notes, archived OpenSpec text, and older
  superpowers plans/specs.
- Rejection tests and strict schema error messages for retired variants.
- Fixture names that intentionally describe legacy recursive-detection contract
  inputs.
- Private implementation/migration adapters below canonical seams.
- Stable artifact filenames `gt_vs_pred.jsonl` and `gt_vs_pred_scored.jsonl`.
- Stable metric/output namespaces that are not renamed by this cleanup slice.

## Validation Results

Commands run:

```bash
python - <<'PY'
import pytest
raise SystemExit(pytest.main([
    'tests/test_detection_compact_full_template.py',
    'tests/test_detection_stage1_json_pretty_template.py',
    'tests/test_detection_template_ir_contract.py',
    'tests/test_detection_template_parsing_eval.py',
    'tests/test_detection_template_registry.py',
    'tests/test_detection_scene_stage1_projection.py',
    '-q',
]))
PY
```

Result: `85 passed in 1.41s`.

```bash
python - <<'PY'
import pytest
raise SystemExit(pytest.main([
    'tests/test_detection_tokenized_view.py',
    'tests/test_detection_template_span_alignment.py',
    'tests/test_token_span_masks_from_templates.py',
    'tests/test_compact_full_encoding_contract.py',
    'tests/test_compact_span_projector.py',
    'tests/test_detection_scene_contract.py',
    '-q',
]))
PY
```

Result: `87 passed in 1.50s`.

```bash
python - <<'PY'
from src.config.loader import ConfigLoader
for path in [
    'configs/stage1/detection_teacher_forcing/smoke/compact_full_tiny.yaml',
    'configs/stage1/detection_teacher_forcing/prod/compact_full_support2.yaml',
    'configs/stage2/rollout_correction/smoke/compact_full_hf_1step.yaml',
    'configs/stage2/rollout_correction/prod/coco1024_online_residual_correction_vllm_tail_append.yaml',
]:
    cfg = ConfigLoader.load_materialized_training_config(path)
    print(path)
    print('  type=', type(cfg).__name__)
    print('  trainer=', getattr(getattr(cfg, 'custom', None), 'trainer_variant', None))
    print('  objective=', getattr(getattr(cfg, 'objective', None), 'id', None))
    print('  template=', getattr(getattr(cfg, 'detection_template', None), 'id', None))
base = ConfigLoader.load_yaml_with_extends('configs/stage2/rollout_correction/base.yaml')
print('configs/stage2/rollout_correction/base.yaml')
print('  raw_sections=', ','.join(sorted(base.keys())[:8]))
print('  has_custom_train_jsonl=', 'train_jsonl' in (base.get('custom') or {}))
PY
```

Result:

```text
configs/stage1/detection_teacher_forcing/smoke/compact_full_tiny.yaml
  type= DetectionTrainingConfig
  trainer= None
  objective= teacher_forcing
  template= compact_full
configs/stage1/detection_teacher_forcing/prod/compact_full_support2.yaml
  type= DetectionTrainingConfig
  trainer= None
  objective= teacher_forcing
  template= compact_full
configs/stage2/rollout_correction/smoke/compact_full_hf_1step.yaml
  type= TrainingConfig
  trainer= stage2_rollout_correction
  objective= None
  template= None
configs/stage2/rollout_correction/prod/coco1024_online_residual_correction_vllm_tail_append.yaml
  type= TrainingConfig
  trainer= stage2_rollout_correction
  objective= None
  template= None
configs/stage2/rollout_correction/base.yaml
  raw_sections= custom,data,debug,deepspeed,global_max_length,model,rollout_matching,stage2_rollout_correction
  has_custom_train_jsonl= False
```

```bash
python - <<'PY'
import pytest
raise SystemExit(pytest.main([
    'tests/test_training_config_hierarchy_contract.py',
    'tests/test_training_config_strict_unknown_keys.py',
    'tests/test_stage2_rollout_correction_profile_leaf_contract.py',
    'tests/test_stage2_launcher_server_template_flags.py',
    'tests/test_stage2_preflight_path_resolution.py',
    'tests/test_stage2_preflight_server_knob_plumbing.py',
    'tests/test_training_surface_resolver.py',
    '-q',
]))
PY
```

Result: `158 passed, 2 warnings in 4.36s`.

```bash
python - <<'PY'
import pytest
raise SystemExit(pytest.main([
    'tests/test_stage2_rollout_correction_target_boundary.py',
    'tests/test_stage2_assignment_greedy_iou.py',
    'tests/test_stage2_duplicate_filter.py',
    '-q',
]))
PY
```

Result: `39 passed, 2 warnings in 4.37s`.

```bash
python - <<'PY'
import pytest
raise SystemExit(pytest.main([
    'tests/test_detection_scene_phase4_infer_eval_projection.py',
    'tests/test_infer_decode_request_mapping.py',
    'tests/test_parser_policy_parity.py',
    '-q',
]))
PY
```

Result: `52 passed in 1.21s`.

```bash
python - <<'PY'
import pytest
raise SystemExit(pytest.main([
    'tests/test_confidence_postop.py',
    'tests/test_detection_eval_ingestion_diagnostics.py',
    'tests/test_detection_scene_phase4_infer_eval_projection.py',
    '-q',
]))
PY
```

Result: `25 passed in 1.12s`.

```bash
python - <<'PY'
import pytest
raise SystemExit(pytest.main([
    'tests/test_instance_trie_gaussian_config_diff.py',
    'tests/test_detection_training_config_contract.py',
    'tests/test_recursive_detection_ce_sft_wiring.py',
    '-q',
]))
PY
```

Result: `3 passed, 123 skipped in 1.46s`. The skipped test module is explicitly
historical because archived `recursive_detection_ce` YAML is no longer a
materializable current config.

```bash
openspec validate detection-scene-clean-break --strict
```

Result: `Change 'detection-scene-clean-break' is valid`.

## Task Evidence

- 7.1: root public concepts now expose canonical names only; retired semantic,
  render, token, and recursive-detection CE helper names are private/module-local
  or historical.
- 7.2: retired Stage-1 config roots and shared snippets are quarantined under an
  explicit archive path; public root compatibility exports were removed; risky
  runtime-required `rollout_matching.*` handles were classified instead of
  deleted.
- 7.3: docs/catalog/spec routing points to the new Stage-1 and Stage-2 routes,
  while historical docs/progress/OpenSpec archive references remain historical.
- 7.4: final gates were run after replacement surfaces existed; this note records
  exact counts, hard absence checks, and allowed leftovers.
- 7.5: retained canonical behavior, stable artifact filenames, geometry/object
  semantics, and Stage-2 assignment/correction semantics were not renamed or
  deleted.
- 8.1: template/render/parse tests passed.
- 8.2: tokenization/span/supervision tests passed.
- 8.3: Stage-1 canonical config leaves materialized; Stage-2 canonical leaves
  materialized; Stage-2 base remains a non-materialized stage base without hidden
  dataset identity.
- 8.4: Stage-2 target construction, assignment, and duplicate-filter tests
  passed.
- 8.5: inference decode/projection tests passed.
- 8.6: eval artifact/scoring tests passed.
- 8.7: final search gates passed with classified allowed leftovers.
