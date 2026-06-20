# SFT Pipeline Hierarchy Refactor — Implementation Audit

**Auditor:** claude-sonnet-4-6  
**Date:** 2026-06-19  
**Branch:** `codex/codebase-refactoring-program`  
**Worktree:** `/data/CoordExp/.worktrees/codebase-refactoring-program`  
**Mode:** read-only implementation-vs-contract audit

---

## Verdict

**HOLD — fix two failing tests and two P1 governance gaps before approval.**

All four originally-reported reviewer issues are confirmed fixed. However, two tests in the branch fail under the `ms` env, and two P1 architecture issues indicate the refactor is structurally incomplete as specified.

---

## Blocking Findings

### P1-A — `TrainingPipelineRegistry` Not Wired Into Training Execution Path

**Evidence:** `src/training/pipeline_registry.py` defines `TrainingPipelineRegistry.resolve()`. Grep across all `src/` confirms it is **never called** from `sft.py`, `loader.py`, `detection/runtime.py`, or any other production source.

```
grep -rn "TrainingPipelineRegistry\|pipeline_registry" src/ --include="*.py"
# Returns only the definition and test imports — zero production call sites
```

The actual Stage-2 dispatch still works through: `DetectionTrainingConfig` → `_runtime_trainer_variant_for_config()` (loader.py:607) → `train_args.trainer_variant` → `resolve_trainer_cls()` (sft.py:139). Stage-1 pipeline IDs (`stage1_standard_sft`, `stage1_research_teacher_forcing`) are mapped to `trainer_variant=None` and the distinction is carried only by the `objective` config section, not the registry.

**Impact:** The design spec (`openspec/changes/refactor-sft-pipeline-hierarchy/design.md`) states `pipeline.id` is "the public selection seam." The registry is the governance object that enforces this, but it is entirely a test artifact. Any production config that would fail registry validation still loads and trains without error. The P1 risk is that governance enforcement is untested under real loading conditions.

**Fix direction:** Add a `TrainingPipelineRegistry.resolve()` call inside `_materialize_training_config()` (loader.py:945) for `DetectionTrainingConfig` payloads, or document in OpenSpec that the registry is explicitly a validation-only artifact (not the execution seam) and rename accordingly to avoid misleading future implementers.

**Verification:** `grep -n "TrainingPipelineRegistry" src/config/loader.py` should return a call site; or the design.md should be amended to say "governance-only, not runtime dispatch."

---

### P1-B — `runtime.trainer_variant` Re-admitted Into New Pipeline Registry Schema

**Evidence:** `src/training/pipeline_registry.py`, `_validate_shared_domains()`:

```python
allowed = {"trainer", "trainer_variant", "precision", "packing", "cache"}
```

The `runtime.trainer_variant` key is explicitly permitted in the new shadow schema that `TrainingPipelineRegistry` validates against.

**Impact:** The refactoring's stated purpose is to retire `trainer_variant` as a public selector. Allowing `runtime.trainer_variant` in the new schema creates a second path to re-introduce the deprecated concept. If the registry is ever wired into execution (P1-A fix), it would silently pass configs using `trainer_variant` under `runtime:`.

**Fix direction:** Remove `"trainer_variant"` from the `runtime` domain's `allowed` set in `_validate_shared_domains()`. If the runtime domain needs a pipeline-identity field, use `pipeline.id` throughout.

**Verification:** `grep -n "trainer_variant" src/training/pipeline_registry.py` should return zero results after the fix.

---

## Non-Blocking Findings

### P2-A — `test_representative_migrated_leaves_stay_within_semantic_depth_budget` FAILS

**Evidence:** Test run under `ms` env:

```
FAILED tests/test_training_config_hierarchy_contract.py::test_representative_migrated_leaves_stay_within_semantic_depth_budget
AssertionError: configs/stage2/rollout_correction/prod/coco1024_online_residual_correction_vllm_tail_append.yaml
  is missing one of the required semantic layers: ['universal_base']
  assert {'universal_base'} <= {'specialized_leaf', 'stage_base'}
```

`configs/stage2/rollout_correction/base.yaml` has no `extends:` line, so its inheritance chain never reaches `configs/base.yaml`. Stage-1's `sft_base.yaml` correctly extends `../base.yaml`; Stage-2 does not.

**Impact:** This is an authored test that was added to enforce the refactoring's hierarchy governance. It fails on the branch, so the branch does not pass its own governance contract. No training correctness impact — the actual config parsing is unaffected — but it signals that the hierarchy restructuring is incomplete for Stage-2.

**Fix direction:** Add `extends: ../../base.yaml` to `configs/stage2/rollout_correction/base.yaml`. Verify no conflicts with Stage-2-specific overrides.

**Verification:** `pytest tests/test_training_config_hierarchy_contract.py::test_representative_migrated_leaves_stay_within_semantic_depth_budget` must pass.

---

### P2-B — `test_active_docs_do_not_recommend_legacy_set_continuation_surface` FAILS (FileNotFoundError)

**Evidence:** Test run:

```
FAILED tests/test_legacy_surface_absence.py::test_active_docs_do_not_recommend_legacy_set_continuation_surface
FileNotFoundError: [Errno 2] No such file or directory:
  '.../docs/training/STAGE1_ET_RMP_CE.md'
```

The file was deleted in commit `1f4d6bf6` ("Reorganize documentation architecture") but `ACTIVE_DOCS` in `test_legacy_surface_absence.py:33` still references it. The test crashes before checking any patterns.

**Impact:** The test has been broken since the docs reorganization commit. Its intended coverage (confirming that `docs/training/README.md`, `STAGE1_OBJECTIVE.md`, `METRICS.md`, etc. do not recommend the retired `stage1_set_continuation` surface) is not executing. This leaves a latent coverage gap.

**Fix direction:** Remove `"docs/training/STAGE1_ET_RMP_CE.md"` from `ACTIVE_DOCS` in `test_legacy_surface_absence.py`. If the patterns in `FORBIDDEN_ACTIVE_PATTERNS` were previously checked in that file, verify coverage is still met through the remaining active docs.

**Verification:** `pytest tests/test_legacy_surface_absence.py::test_active_docs_do_not_recommend_legacy_set_continuation_surface` must pass.

---

### P2-C — Plan-Required Test Files Missing

**Evidence:**

```
ls tests/test_training_pipeline_hierarchy_schema.py   # No such file
ls tests/test_training_objective_public_ids.py        # No such file
```

The 10-task implementation plan (`docs/superpowers/plans/2026-06-17-sft-pipeline-hierarchy-refactor.md`) lists these as outputs of Tasks 2 and 4.

**Impact:** Partial coverage exists in `test_detection_training_config_contract.py` (pipeline/objective pairing, retired path rejection) and `test_training_pipeline_registry.py` (registry-level). But the explicitly committed plan files are absent. An approver comparing against the plan checklist will find a gap. The missing coverage is primarily around: (a) `pipeline.id` as a stable schema selector tested in isolation, and (b) `standard_ce`/`research_teacher_forcing` as the canonical public objective ID surface.

**Fix direction:** Create the two test files (or rename/redirect to the existing coverage and update the plan to reflect the consolidation decision). Mark Tasks 2 and 4 in the plan as either complete-via-alternate-coverage or still pending.

**Verification:** Both files exist and each contains at least one passing test that directly references the plan's stated scope (`pipeline.id` selector, public objective IDs).

---

### P2-D — `cfg_only` Summary Emits `trainer_variant: ""` for New-Style Configs

**Evidence:** `sft.py:2741`:

```python
"trainer_variant": str(getattr(custom_config, "trainer_variant", "") or ""),
```

For `DetectionTrainingConfig` payloads, `custom_config` is the shim returned by `build_detection_runtime_custom_shim()` (which has no `trainer_variant` attribute), so this always emits `""`. The summary is silent about `pipeline.id`.

**Impact:** Operator tooling that reads `cfg_only` output to identify which pipeline is active will see `trainer_variant: ""` and have to infer the pipeline from other fields. Low severity, but inconsistent with the refactoring's goal of making `pipeline.id` the visible public selector.

**Fix direction:** In the `cfg_only` block, replace `"trainer_variant"` with `"pipeline_id"` and read it from `getattr(getattr(detection_config, "pipeline", None), "id", None)` when `detection_config is not None`.

---

## Confirmed-Fixed: Originally Reported Reviewer Issues

All four issues raised for this review pass verification:

| Issue | Location | Verification |
|---|---|---|
| Stage-2 entering Stage-1 dataset runtime | `src/detection/runtime.py:183` | `_is_stage2_rollout_correction_config()` guard raises on `pipeline.id=stage2_rollout_correction`. Test: `test_training_runtime_sft_integration.py` lines 141-169. |
| Public artifacts leaking `trainer_variant` | `src/utils/run_manifest.py:175` | `runtime_payload.pop("trainer_variant", None)`. Test: `test_run_manifest_files.py:92` asserts `"trainer_variant" not in effective_runtime["runtime"]`. |
| Empty default tuple/list fields in authored metadata | `src/sft.py:759` | `_clean_authored_experiment_payload()` filters zero-length sequences. Handles `DetectionExperimentConfig.key_deviations` and `runtime_settings` defaults. |
| `rollout_matching.pipeline` error references wrong selector | `src/config/schema.py:4630-4657` (new schema) and `schema.py:4976-4980` (old schema) | Both branches now say `pipeline.id=stage2_rollout_correction`, not `custom.trainer_variant=stage2_rollout_correction`. |

---

## Additional Verified-Correct Areas

The following areas were audited and match the spec contract:

- **`surfaces.py` deleted**: Confirmed absent. `test_legacy_surface_absence.py::test_legacy_trainer_and_collator_import_paths_are_gone` passes.
- **`teacher_forcing` id rejection**: `_detection_validate_pipeline_objective_pairing()` and `_detection_reject_removed_target_hierarchy_paths()` both enforce this. `test_detection_training_config_contract.py::test_detection_config_rejects_legacy_teacher_forcing_objective_id` passes.
- **`research_teacher_forcing` packing guard**: `_detection_validate_packing_runtime_contract()` (schema.py:3030) triggers when `objective.id == "research_teacher_forcing"` and rejects `training.packing`, `eval_packing`, `static_packing`, `padding_free_packed`. Tests pass.
- **Stage-2 provenance**: `src/bootstrap/stage2_policy_provenance.py` now reads from `pipeline.id`, output never contains `trainer_variant`. `test_stage2_policy_provenance.py:31-95` passes.
- **`token_embeddings_adapter` replaces `token_rows`**: Enforced via `_DETECTION_OBSOLETE_KEYS` and `_detection_reject_removed_target_hierarchy_paths()`.
- **Dispatcher correctness for unmigrated configs**: The unmigrated configs (`configs/stage1/lvis_bbox_max60_1024.yaml`, `configs/stage1/smoke/common_prodlike.yaml`) have only `model`, `training`, `custom` as top-level keys and will NOT accidentally match `legacy_detection_markers` or `target_hierarchy_sentinels`.
- **ET-RMP / recursive_detection_ce preserved**: Implementation active in `src/sft.py` and `src/config/schema.py`. `test_recursive_detection_ce_target_builder.py` (14 passing), `test_recursive_detection_ce_trainer_mixin.py`, and `test_recursive_detection_ce_sft_wiring.py` all pass. The 96 skipped tests in `test_detection_training_config_contract.py` are explicitly annotated "legacy recursive_detection_ce config contract retired" — they test old config syntax, not missing functionality.
- **Active migrated configs validated**: `stage1/profiles/2b/` configs, `stage1/detection_teacher_forcing/prod/compact_support2.yaml`, `stage2/rollout_correction/base.yaml`, and `stage2/rollout_correction/prod/coco1024_online_residual_correction_vllm_tail_append.yaml` all use the new `pipeline.id` selector.

---

## Evidence Reviewed

| File / Artifact | What Was Checked |
|---|---|
| `openspec/changes/refactor-sft-pipeline-hierarchy/design.md` | Contract for `pipeline.id`, objective IDs, packing guard, provenance identity, `surfaces.py` deletion |
| `openspec/changes/refactor-sft-pipeline-hierarchy/specs/stage2-rollout-correction/spec.md` | `pipeline.id` replacing `custom.trainer_variant`, provenance requirements |
| `docs/superpowers/plans/2026-06-17-sft-pipeline-hierarchy-refactor.md` | Task list, required test files |
| `src/training/pipeline_registry.py` | Registry definition, `_validate_shared_domains()`, `runtime.trainer_variant` allowance |
| `src/training/pipelines/stage1_json_ce.py`, `stage1_compact_trie_ce.py`, `stage2_rollout_correction.py` | Pipeline descriptor IDs |
| `src/config/schema.py` (selected sections) | `DetectionTrainingConfig.from_mapping`, `_detection_validate_packing_runtime_contract`, `_detection_reject_removed_target_hierarchy_paths`, `_detection_validate_pipeline_objective_pairing`, packing guard |
| `src/config/loader.py` (selected sections) | `_is_detection_training_config_payload`, `_runtime_trainer_variant_for_config`, `_materialize_training_config` |
| `src/sft.py` (selected sections) | `resolve_trainer_cls`, `_build_normalized_training_hierarchy_identity`, `_clean_authored_experiment_payload`, `cfg_only` block |
| `src/bootstrap/stage2_policy_provenance.py` | Provenance fields, `trainer_variant` removal |
| `src/utils/run_manifest.py` | `trainer_variant` pop |
| `src/detection/runtime.py` | Stage-2 guard, `build_detection_runtime_custom_shim` |
| `src/training_runtime/plan.py` | `resolve_training_runtime_plan`, Stage-1 pipeline-id passthrough |
| `configs/stage2/rollout_correction/base.yaml` | Missing `extends:` chain |
| `configs/base.yaml` | Universal base contents |
| Active migrated config files (Stage-1 profiles/2b, compact_support2, Stage-2 base/prod) | `pipeline.id`, `sample_factory`, objective IDs |
| Unmigrated configs (`lvis_bbox_max60_1024.yaml`, `smoke/common_prodlike.yaml`) | Top-level keys, dispatch safety |

---

## Verification Run

Commands run (read-only):

```bash
# Focused reviewer test suite
/root/miniconda3/envs/ms/bin/pytest \
  tests/test_training_pipeline_registry.py \
  tests/test_detection_training_config_contract.py \
  tests/test_stage2_rollout_correction_contract.py \
  tests/test_stage2_policy_provenance.py \
  tests/test_run_manifest_files.py \
  tests/test_teacher_forcing_config_contract.py \
  -q --tb=short
# Result: 149 passed, 96 skipped in 1.75s

# Hierarchy contract + legacy surface absence tests
/root/miniconda3/envs/ms/bin/pytest \
  tests/test_training_config_hierarchy_contract.py \
  tests/test_legacy_surface_absence.py \
  -v --tb=short
# Result: 2 FAILED (P2-A, P2-B), 9 passed

# ET-RMP / recursive_detection_ce implementation tests
/root/miniconda3/envs/ms/bin/pytest \
  tests/test_recursive_detection_ce_target_builder.py \
  -q --tb=short
# Result: 14 passed
```

Skipped: full 383-test suite (expensive, not required given targeted coverage). Skipped: training/inference smoke runs (expensive, read-only constraint).

---

## Residual Risks

1. **Old-format residue in `cfg_only` output** (P2-D): Operators using `--cfg_only` output for new-style configs will see `trainer_variant: ""` instead of `pipeline_id`. No training correctness impact.

2. **`gkd_monitor` carried forward via `trainer_variant`**: `configs/debug.yaml:92` uses `trainer_variant: gkd_monitor`. This uses the old config path (`TrainingConfig`), not the new hierarchy. It is explicitly allowed by `_GENERIC_STAGE1_EXTENSION_VARIANTS` in `training_runtime/plan.py`. But it is not migrated to `pipeline.id`. Acceptable as debug-only tooling, but worth tracking for completeness.

3. **96 skipped tests**: All are `pytest.mark.skip(reason="legacy recursive_detection_ce config contract retired by teacher_forcing objective")`. The underlying ET-RMP implementation is tested via separate active test files. No missing correctness coverage, but a future reader may be surprised by the skip count without the context that these are intentionally retired config-format tests.

4. **`_is_detection_training_config_payload` `legacy_detection_markers` path**: A future config that has `data`, `prompt`, `detection_template`, `objective`, `packing`, `evaluation`, and `validation` at top level but no `pipeline.id` would be dispatched to `DetectionTrainingConfig.from_mapping()` and fail with "Missing detection config sections." No current configs trigger this, but it's a latent failure mode for anyone writing a transitional config.

---

## Recommended Next Step

1. **Fix P2-A** (`base.yaml` extends chain for Stage-2): Add `extends: ../../base.yaml` to `configs/stage2/rollout_correction/base.yaml`, run `pytest tests/test_training_config_hierarchy_contract.py`.

2. **Fix P2-B** (remove dead file reference): Remove `"docs/training/STAGE1_ET_RMP_CE.md"` from `ACTIVE_DOCS` in `test_legacy_surface_absence.py`, run `pytest tests/test_legacy_surface_absence.py`.

3. **Resolve P1-A** (registry integration): Either (a) wire `TrainingPipelineRegistry.resolve()` into `loader.py:_materialize_training_config()` for new-style payloads, or (b) rename the class to `TrainingPipelineValidator` and update the OpenSpec design note to clarify it is governance-only. Without this, the spec claim that `pipeline.id` is "the public selection seam" enforced by the registry is not true.

4. **Fix P1-B** (remove `trainer_variant` from registry schema): Remove `"trainer_variant"` from `allowed` in `_validate_shared_domains()` in `pipeline_registry.py`.

5. **Address P2-C** (missing plan test files): Create or redirect `test_training_pipeline_hierarchy_schema.py` and `test_training_objective_public_ids.py` per the plan, or update the plan to document the consolidation decision.

After (1) and (2): branch passes authored tests. After (3) and (4): P1 governance gaps are resolved. After (5): plan completion is verifiable.
