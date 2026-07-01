## Workstream 0. Baseline And Scope Freeze

- [x] 0.1 Record the representative leaves and shared parents that define the
      current training hierarchy baseline:
  - Stage-1 LVIS production leaf
  - Stage-1 LVIS smoke leaf
  - one Stage-1 COCO leaf
  - Stage-2 LVIS production leaf
  - one canonical Stage-2 COCO prod leaf
- [x] 0.1b Record every canonical leaf family that will consume a new shared
      dataset facet or shared prompt facet so migration coverage is not limited
      to one happy-path sample per stage.
- [x] 0.1c Record all current `configs/base.yaml` consumers and make an explicit
      scope inventory that includes:
  - in-scope canonical Stage-1 families
  - in-scope canonical Stage-2 families
  - legacy fusion consumers to retire across config/schema/runtime/docs/tests
- [x] 0.1d Record repo-owned non-stage `configs/base.yaml` consumers such as
      debug/bench/helper surfaces and decide whether each one is:
  - migrated onto the new facet layout
  - kept as an intentional exception
  - or explicitly de-supported
- [x] 0.2 Capture the stable invariants this refactor must preserve:
  - strict unknown-key parsing
  - current `extends` merge semantics
  - explicit leaf run identity
  - current canonical Stage-2 profile discovery
  - prompt/rendering semantics
  - dataset path and object-ordering behavior
- [x] 0.2b Capture the artifact-path simplification invariant:
  - one shared `training.artifact_subdir`
  - separately configurable `training.output_root`
  - separately configurable `training.logging_root`
  - stable resolved `training.output_dir` / `training.logging_dir`
  - no direct duplicated authoring of `training.output_dir` /
    `training.logging_dir` in canonical repo-owned configs
- [x] 0.3 Collect a pre-refactor verification baseline with:
  - `conda run -n ms python -m pytest -q tests/test_stage1_static_packing_runtime_config.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_profile_leaf_contract.py`
  - `conda run -n ms python -m pytest -q tests/test_training_config_strict_unknown_keys.py`
- [x] 0.3b If a named baseline suite is already red, record the pre-existing
      failure and stabilize or partition it before using that suite as parity
      evidence.
  - Retroactive note: a literal pre-refactor capture is no longer recoverable
    because the hierarchy refactor was already in flight before this finishing
    pass. The retained parity evidence for this change is the stabilized
    post-refactor contract slice re-run under `6.2`, plus the added focused
    hierarchy checks in `tests/test_training_config_hierarchy_contract.py`.
- [x] 0.3c Make an explicit Stage-2 smoke decision before using Stage-2 profile
      tests as hierarchy-contract proof:
  - restore and migrate the smoke leaves
  - or retire smoke from loader/docs/tests/scripts in the same change

## Workstream 1. Canonical Hierarchy Contract

- [x] 1.1 Add the OpenSpec delta for canonical training-config hierarchy rules.
- [x] 1.2 Document the allowed concern groups for reusable facets:
  - dataset identity
  - prompt identity
  - stage-local runtime defaults
  - stage-local objective bundles
  - stage-local observability bundles
- [x] 1.2b Document that COCO/LVIS are only the initial migration exemplars and
      that the facet contract must remain reusable for future prepared
      single-dataset families.
- [x] 1.2c Document the preferred authored hierarchy depth:
  - universal base
  - stage-wise base
  - shared/common reusable package settings
  - specialized experiment leaf
- [x] 1.2d Document how derivative wrappers are counted:
  - smoke/runtime-limit overlays are allowed
  - they stay narrow
  - they do not consume an additional shared/common semantic layer
- [x] 1.2e Document the artifact-path simplification contract under the existing
      `training` section:
  - `training.output_root`
  - `training.logging_root`
  - `training.artifact_subdir`
- [x] 1.3 Explicitly document that list-valued bundles must have one owner in the
      inheritance chain because the current loader replaces lists.
- [x] 1.4 Confirm the implementation plan only changes loader/schema where
      necessary to remove legacy authoring paths and shut down fusion runtime
      authoring, while
      preserving strict parsing and deterministic merge semantics.

## Workstream 2. Global Base Cleanup

- [x] 2.0 Shut down fusion coherently while preserving dormant legacy assets:
  - keep `configs/fusion/` as dormant reference material
  - disable `custom.fusion_config` in schema/runtime
  - update docs/specs/tests so they no longer treat fusion authoring as active
- [x] 2.1 Remove dataset-specific training defaults from `configs/base.yaml`.
- [x] 2.2 Keep `configs/base.yaml` limited to repo-global model/template/training
      defaults.
- [x] 2.2b Remove shared dataset and prompt identity from the stage-local bases
      for every leaf family migrated onto the new facet layout.
- [x] 2.2c Reconcile canonical Stage-2 runtime inheritance onto
      `configs/base.yaml` rather than leaving `configs/stage2_two_channel/base.yaml`
      as a permanent standalone exception.
- [x] 2.3 Add a regression guard that the global base no longer resolves hidden
      dataset identity.
- [x] 2.4 Add a regression guard for the chosen migrated families showing their
      stage-local bases no longer act as hidden dataset/prompt selectors.

## Workstream 3. Shared Dataset And Prompt Facets

- [x] 3.1 Introduce shared dataset facets under `configs/_shared/datasets/` for
      the representative COCO and LVIS prepared datasets.
- [x] 3.1b Document the naming and ownership rules for adding future
      `configs/_shared/datasets/*.yaml` facets without loader/schema changes.
- [x] 3.2 Introduce shared prompt facets under `configs/_shared/prompts/` for:
  - `coco80_desc_first`
  - `coco80_geometry_first`
  - `lvis_stage1_federated`
  - `lvis_stage2_federated`
- [x] 3.3 Ensure dataset facets own dataset-specific path and ordering contracts,
      while prompt facets own prompt variant and field order.
- [x] 3.4 Add direct materialization tests for the new shared dataset facets and
      shared prompt facets via minimal composed fixture leaves.
- [x] 3.5 Ensure dataset-facet parity coverage includes the paired image-budget
      contract:
  - `custom.offline_max_pixels`
  - `template.max_pixels`

## Workstream 4. Stage-Local Runtime And Shared Bundle Cleanup

- [x] 4.1 Narrow `configs/stage1/sft_base.yaml` to Stage-1 runtime ownership
      only.
- [x] 4.2 Narrow `configs/stage2_two_channel/base.yaml` to Stage-2 runtime
      ownership only.
- [x] 4.3 Extract genuinely reused Stage-1 objective defaults into explicit
      stage-local shared facets.
- [x] 4.4 Extract genuinely reused Stage-2 objective or observability defaults
      into explicit stage-local shared facets without splitting list-valued
      manifests across multiple owners.
- [x] 4.5 Review mixed shared parents such as `configs/stage2_two_channel/_shared/prod_common.yaml`
      and split orthogonal concerns when that reduces hidden coupling.
- [x] 4.5b Introduce the repo-owned artifact-path composition rule so canonical
      configs can author one `training.artifact_subdir` plus separate
      `training.output_root` / `training.logging_root`.
- [x] 4.5c Remove direct duplicated `training.output_dir` /
      `training.logging_dir` authoring from canonical repo-owned configs.
- [x] 4.6 Add a guard test or validation rule that overlapping ownership of the
      same list-valued config key in one inheritance chain is rejected or
      otherwise made impossible by the repo-owned hierarchy contract.
- [x] 4.7 Make the enforcement point explicit:
  - repo-owned validation helper or preflight rule
  - plus regression coverage for duplicate list ownership in one inheritance
    chain

## Workstream 5. Representative Leaf Migration

- [x] 5.1 Repoint the representative Stage-1 LVIS production leaf to the new
      facet layout.
- [x] 5.2 Repoint the Stage-1 LVIS smoke leaf so it differs only by runtime
      smoke limits.
- [x] 5.3 Repoint one representative Stage-1 COCO leaf to the new facet layout.
- [x] 5.4 Repoint the representative Stage-2 LVIS production leaf to the new
      facet layout.
- [x] 5.5 Repoint one canonical Stage-2 COCO prod leaf to the new facet layout.
- [x] 5.6 Preserve explicit leaf identity for:
  - `model.model`
  - `training.run_name`
  - `training.artifact_subdir`
- [x] 5.6b Keep `training.output_root` / `training.logging_root` reusable while
      preserving the same resolved artifact directories as before.
- [x] 5.6c Do not keep a parallel legacy authoring path for duplicated
      `training.output_dir` / `training.logging_dir` in migrated canonical
      configs.

## Workstream 6. Verification And Docs

- [x] 6.1 Add resolved-config parity assertions for the representative leaves so
      the refactor is behavior-preserving.
- [x] 6.1b Extend parity coverage to every canonical leaf family that consumes a
      newly introduced shared dataset facet or shared prompt facet.
- [x] 6.1c Add one contract-level check or fixture showing that a future
      prepared dataset facet can be authored with the same existing typed keys
      and hierarchy rules.
- [x] 6.1d Confirm the migrated canonical configs remain within the preferred
      semantic depth budget of 4 levels from universal base to specialized leaf,
      excluding narrow smoke/runtime wrappers.
- [x] 6.1e Add raw-leaf ownership checks showing canonical migrated leaves still
      author:
  - `model.model`
  - `training.run_name`
  - `training.artifact_subdir`
- [x] 6.1f Add artifact-path parity checks for the resolved:
  - `training.output_dir`
  - `training.logging_dir`
- [x] 6.2 Re-run:
  - `conda run -n ms python -m pytest -q tests/test_stage1_static_packing_runtime_config.py`
  - `conda run -n ms python -m pytest -q tests/test_stage2_ab_profile_leaf_contract.py`
  - `conda run -n ms python -m pytest -q tests/test_training_config_strict_unknown_keys.py`
- [x] 6.3 Update operator-facing docs and routing pages for the new config facet
      layout:
  - `docs/training/README.md`
  - `docs/training/LVIS.md`
  - `docs/IMPLEMENTATION_MAP.md`
- [x] 6.4 Validate the OpenSpec change with the local CLI and address any
      structural issues before implementation begins.
