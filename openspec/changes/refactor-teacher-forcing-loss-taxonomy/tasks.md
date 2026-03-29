## 1. OpenSpec Foundation

- [x] 1.1 Capture the refactor as a contract-preserving internal architecture change rather than a behavior-change proposal.
- [x] 1.2 Keep the delta spec set aligned with:
  - `teacher-forcing-objective-pipeline`
  - `teacher-forcing-unified-loss-registry`
- [x] 1.3 Validate the change after authoring:
  - `openspec validate refactor-teacher-forcing-loss-taxonomy --type change --strict --json --no-interactive`

## 2. Catalog-Driven Objective Module Taxonomy

- [x] 2.1 Introduce one canonical internal taxonomy layer for teacher-forcing modules.
- [x] 2.2 Store per-objective-module ownership in that catalog:
  - family
  - semantic role
  - config allowlist
  - optional config keys
  - application presets
  - module-level `emission_group`
  - projected Stage-2 atom definitions
- [x] 2.3 Store per-diagnostic-module ownership in the companion diagnostic catalog:
  - family
  - semantic role
  - config allowlist
- [x] 2.4 Derive strict registry allowlists from the catalogs instead of maintaining parallel handwritten tables.
- [x] 2.5 Preserve all current public module names in the first refactor slice.
- [x] 2.6 Define semantic-role naming as a stable lowercase snake_case internal vocabulary.

## 3. Explicit Bbox Family Separation

- [x] 3.1 Classify bbox-dependent objective modules under a shared `bbox` family.
- [x] 3.2 Keep bbox semantic roles separate:
  - `bbox_geo` -> `geometry`
  - `bbox_size_aux` -> `size_aux`
- [x] 3.3 Keep bbox atoms categorized consistently:
  - `bbox_smoothl1`
  - `bbox_ciou`
  - `bbox_log_wh`
  - `bbox_oversize`
- [x] 3.4 Ensure bbox-dependent losses are not folded into `coord_reg` or mixed with unrelated taxonomy.

## 4. Runtime Registry And Projection Refactor

- [x] 4.1 Keep objective and diagnostics registries explicit at runtime.
- [x] 4.2 Validate objective and diagnostics registries against companion catalogs rather than an implied single merged catalog.
- [x] 4.3 Preserve authored YAML execution order and module-to-module state handoff as explicit pipeline invariants.
- [x] 4.4 Refactor Stage-2 atom projection to consume module-owned atom definitions from the objective catalog.
- [x] 4.5 Preserve additive reconstruction checks for projected objective atoms.
- [x] 4.6 Define module-level `emission_group` routing and per-atom `required_state` ownership in the change artifacts.
- [x] 4.7 Require projected atom-key uniqueness within an emission provenance group.
- [x] 4.8 Allow explicitly optional projection atoms to remain absent without breaking additivity.

## 5. Regression Coverage

- [x] 5.1 Add focused tests proving the strict allowlists are derived from the shared catalog.
- [x] 5.2 Add focused tests proving bbox modules share a family while retaining separate semantic roles.
- [x] 5.3 Re-run narrow validation for the touched surfaces:
  - `tests/test_stage2_ab_config_contract.py`
  - `tests/test_training_config_strict_unknown_keys.py`
  - `tests/test_stage2_objective_atoms_projection.py`
  - `tests/test_teacher_forcing_loss_catalog.py`
- [x] 5.4 Re-run a broader Stage-2 trainer suite and leave only the GPU-dependent smoke run deferred.
  - Follow-up downstream Stage-1 smoke-readiness evidence after the refactor landed:
    - canonical LVIS Stage-1 config on `main`: `configs/stage1/lvis_bbox_max60_1024.yaml`
    - smoke leaf on `main`: `configs/stage1/smoke/lvis_bbox_max60_1024.yaml`
    - focused verification rerun on `main`:
      - `conda run -n ms python -m pytest -q tests/test_stage1_static_packing_runtime_config.py::test_lvis_stage1_smoke_config_only_overrides_runtime_limits`
      - `conda run -n ms python -m pytest -q tests/test_image_path_resolution_contract.py`
    - loader follow-up fix validated on `main`: `src/datasets/dense_caption.py` no longer shadows `Path` inside `BaseCaptionDataset.from_jsonl(...)`
    - launcher smoke command fired from `main`:
      - `config=configs/stage1/smoke/lvis_bbox_max60_1024.yaml gpus=0 conda run -n ms bash scripts/train.sh`
  - Note: this is Stage-1 smoke-readiness / launchability evidence for the refactored teacher-forcing stack; it does not replace the separate Stage-2 GPU smoke that remained deferred in this task.
- [x] 5.5 Add targeted tests for objective-registry drift fail-fast behavior.
- [x] 5.6 Add targeted tests for diagnostics-registry drift fail-fast behavior.
- [x] 5.7 Add additive projection coverage for absent optional projected atoms and missing required projected atoms.
- [x] 5.8 Add coverage or a fail-fast guard for projected atom-key collisions.
