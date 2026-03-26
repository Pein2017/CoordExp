## 1. OpenSpec Foundation

- [x] 1.1 Capture the refactor as a contract-preserving internal architecture change rather than a behavior-change proposal.
- [x] 1.2 Keep the delta spec set aligned with:
  - `teacher-forcing-objective-pipeline`
  - `teacher-forcing-unified-loss-registry`
- [x] 1.3 Validate the change after authoring:
  - `openspec validate refactor-teacher-forcing-loss-taxonomy --type change --strict --json --no-interactive`

## 2. Catalog-Driven Objective Module Taxonomy

- [x] 2.1 Introduce one canonical internal catalog for objective modules.
- [x] 2.2 Store per-module ownership in that catalog:
  - family
  - semantic role
  - config allowlist
  - optional config keys
  - application presets
  - projected Stage-2 atoms
- [x] 2.3 Derive strict registry allowlists from the catalog instead of maintaining parallel handwritten tables.
- [x] 2.4 Preserve all current public module names in the first refactor slice.

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
- [x] 4.2 Add fail-fast coverage validation so runtime registries stay in sync with the shared catalog.
- [x] 4.3 Refactor Stage-2 atom projection to consume module-owned atom definitions from the same catalog.
- [x] 4.4 Preserve additive reconstruction checks for projected objective atoms.
- [x] 4.5 Allow explicitly optional projection atoms to remain absent without breaking additivity.

## 5. Regression Coverage

- [x] 5.1 Add focused tests proving the strict allowlists are derived from the shared catalog.
- [x] 5.2 Add focused tests proving bbox modules share a family while retaining separate semantic roles.
- [x] 5.3 Re-run narrow validation for the touched surfaces:
  - `tests/test_stage2_ab_config_contract.py`
  - `tests/test_training_config_strict_unknown_keys.py`
  - `tests/test_stage2_objective_atoms_projection.py`
  - `tests/test_teacher_forcing_loss_catalog.py`
- [x] 5.4 Record that a broader Stage-2 trainer suite was sampled and the observed failures were outside the touched teacher-forcing files.
