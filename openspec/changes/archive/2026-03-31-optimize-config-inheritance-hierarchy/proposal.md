## Why

CoordExp's training configs have reached the point where the main source of
friction is not missing schema strictness, but mixed ownership inside the YAML
tree.

The current loader is intentionally simple:

- `src/config/loader.py` resolves `extends` recursively,
- merges mappings with plain deep-merge semantics,
- and relies on the strict typed schema in `src/config/schema.py` to reject
  unsupported keys.

That simplicity is good for reproducibility, but it means the hierarchy itself
must stay readable and unambiguous.

Today that hierarchy has several concrete problems:

- `configs/base.yaml` is no longer a truly global base because it still carries
  dataset-specific `custom.train_jsonl` / `custom.val_jsonl` defaults.
- Stage-1 and Stage-2 parent files mix runtime defaults, dataset identity,
  prompt identity, objective bundles, and observability knobs in the same
  layers.
- shared prompt files are too narrow to act as first-class composition units,
  while dataset identity and prompt semantics remain scattered across separate
  parents.
- leaf files repeat experiment identity intentionally, but also repeat reusable
  concern groups that should be owned once.

This makes config intent harder to audit from a resolved tree and increases the
risk of precedence mistakes like hidden dataset defaults, accidental override
shadowing, or list-valued objective bundles being replaced unintentionally.

The repo does **not** need a new configuration language or a schema rewrite for
this problem. It needs a clearer hierarchy.

This change is intentionally scoped to the repo's single-dataset training
default. Legacy multi-dataset config surfaces under `configs/fusion/` are not
part of the new canonical hierarchy and may be removed rather than migrated.

## What Changes

- Introduce a dedicated training-config hierarchy optimization pass centered on
  reusable YAML facets, not on schema redesign.
- Remove or retire `configs/fusion/` as a canonical example/config-folder
  surface instead of adapting that legacy tree to the new hierarchy.
- Retire the underlying `custom.fusion_config` capability in the same change
  rather than carrying a legacy runtime/schema escape hatch forward.
- Make `configs/base.yaml` dataset-agnostic and prompt-agnostic so repo-global
  defaults only cover truly global model/template/training behavior.
- Introduce shared dataset and prompt facets under `configs/_shared/` for
  reusable training identities such as:
  - dataset source paths
  - offline image-budget contract
  - object ordering
  - prompt variant
  - object field order
  - with COCO and LVIS as the initial migration exemplars rather than as a
    closed set of supported dataset families
- Keep Stage-1 and Stage-2 runtime bases separate, but narrow their ownership:
  - Stage-1 base owns Stage-1 runtime defaults only.
  - Stage-2 base owns Stage-2 runtime defaults only.
  - Stage-local bases are not allowed to remain the hidden long-term owner of
    shared dataset or prompt identity after migration.
  - Canonical migrated Stage-1 and Stage-2 families should reconcile onto the
    same repo-global base rather than leaving Stage-2 as a permanent standalone
    exception.
- Keep the authored hierarchy shallow and auditable:
  - universal base
  - stage-wise base config
  - shared/common reusable package settings
  - specialized experiment leaf
- Canonical repo-owned training configs should therefore target a maximum
  authored depth of 4 semantic levels from universal base to specialized
  experiment leaf, while still allowing multiple shared/common parents at the
  same layer.
- Derivative wrappers such as smoke/runtime-limit overlays may sit on top of a
  specialized leaf, but they must stay narrow and do not count as a separate
  shared/common hierarchy layer.
- Move reusable objective or observability bundles into explicit stage-local
  shared facets rather than keeping them embedded in broad parent files.
- Preserve explicit leaf identity for high-signal fields such as:
  - `model.model`
  - `training.run_name`
  - `training.artifact_subdir`
  - stage-defining effective batch and sample-limit overrides
- Introduce a repo-owned artifact-path composition rule under the existing
  `training` section so canonical configs can set:
  - `training.output_root`
  - `training.logging_root`
  - `training.artifact_subdir`
  and author the shared suffix once instead of duplicating it across
  `training.output_dir` and `training.logging_dir`.
- Treat that root-plus-subdir pattern as the new canonical authoring contract
  for repo-owned training configs instead of preserving duplicated
  `training.output_dir` / `training.logging_dir` authoring.
- Preserve the current deterministic merge model and strict parsing behavior:
  - no new top-level config sections,
  - no list-merge magic,
  - no weakening of unknown-key fail-fast behavior.
- Keep the hierarchy universal for any future single-dataset family that uses
  the existing prepared-data contract; the first slice must not special-case
  COCO or LVIS in loader/schema behavior.
- Treat list-valued config bundles as single-owner surfaces in the inheritance
  chain so the current deep-merge implementation cannot silently replace an
  earlier facet's objective or augmentation list.
- Add resolved-config parity coverage for representative Stage-1 and Stage-2
  leaves so the hierarchy refactor is behavior-preserving rather than
  cosmetically reorganized.
- Update docs/runbooks so the new facet layout is discoverable and operator
  workflows remain obvious.

## Recommended First Version

The first implementation slice should stay focused on the highest-value cleanup:

- remove dataset defaults from `configs/base.yaml`,
- introduce shared dataset facets and shared prompt facets,
- migrate representative Stage-1 and Stage-2 LVIS/COCO leaves onto that layout,
- introduce root-plus-subdir artifact path composition under `training` so
  `output_dir` and `logging_dir` stop duplicating the same suffix,
- retire legacy config authoring paths instead of supporting them in parallel,
- and add resolved-config parity tests before broader migration.

That first slice should **not**:

- rewrite `CustomConfig` into new nested dataclasses,
- auto-derive run names or output paths,
- add templating/DSL features beyond the current `extends` mechanism,
- or keep backward-compatibility shims for legacy config-folder surfaces or
  duplicated artifact-path authoring.

## Assumptions

- The main maintainability problem is hierarchy/layout ownership, not a lack of
  schema strictness.
- Explicit leaf identity is valuable for reproducibility and should remain
  visible instead of being generated dynamically.
- Shared dataset and prompt facets can reduce duplication without weakening the
  typed config contract.
- The same facet pattern should scale to future prepared datasets without adding
  dataset-name conditionals to the loader or schema.
- Artifact directory duplication is a hierarchy/layout problem, not a reason to
  introduce a new top-level config system.
- Multi-dataset or fusion-first config authoring is legacy and should remain
  disabled in the canonical surface, while dormant examples/modules can remain
  in-tree for future reactivation.
- Stage-1 and Stage-2 are too semantically different to collapse into one
  universal runtime base.

## Non-Blocking Follow-Ups

- If hierarchy cleanup is still painful after facetization, consider a follow-on
  typed-schema cleanup that splits `CustomConfig` into smaller ownership groups.
- If experiment identity remains too noisy after the facet pass, consider a
  follow-on review of `run_name` / output-path conventions, but only with
  explicit operator-artifact validation.
- If multi-dataset or fusion-first training needs to return later, handle it in
  a separate change rather than stretching this single-dataset hierarchy
  contract.

## Risks To Validity

- Moving dataset identity out of `configs/base.yaml` can break consumers that
  still rely on hidden inherited defaults.
- Non-stage consumers that currently extend `configs/base.yaml` can also break
  unless they are explicitly migrated or de-supported.
- Splitting list-valued bundles across multiple facets can silently change
  resolved objectives because the current loader replaces lists instead of
  merging them.
- Over-abstracting leaf files can make runs harder, not easier, to audit if
  high-signal identity fields stop being visible at the leaf.
- Treating the 4-layer rule as raw YAML-hop counting would conflict with
  smoke/runtime overlays unless the contract is defined in semantic layers.
- A hierarchy-only refactor that lacks resolved-config parity tests could look
  cleaner while still changing actual runtime behavior.

## Required Evidence

- Resolved-config parity for representative leaves before and after migration:
  - Stage-1 LVIS leaf
  - Stage-1 smoke LVIS leaf
  - one Stage-1 COCO leaf
  - Stage-2 LVIS leaf
  - one canonical Stage-2 COCO prod leaf
- Stabilized baseline evidence for any named parity suite; pre-existing red
  suites cannot be reused as migration proof until fixed or explicitly
  partitioned.
- Minimal composed fixture checks for the new shared dataset and prompt facets so
  facet breakage is caught without requiring raw facet fragments to load as
  standalone training configs.
- Proof that introducing a future dataset facet would reuse the same typed keys
  and hierarchy rules rather than requiring new schema branches.
- Proof that `configs/base.yaml` no longer carries dataset-specific training
  defaults.
- Proof that all repo-owned `configs/base.yaml` consumers, including non-stage
  helpers such as debug/bench surfaces, are either migrated intentionally or
  explicitly de-supported.
- Proof that legacy fusion capability has been shut down coherently across
  config, schema, runtime, docs, specs, and tests, while dormant examples stay
  discoverable in-tree.
- Proof that strict unknown-key validation and canonical Stage-2 profile loading
  remain unchanged.
- Proof that canonical migrated configs derive the same resolved
  `training.output_dir` / `training.logging_dir` contract from
  `output_root` + `logging_root` + `artifact_subdir`.
- Proof that canonical repo-owned configs no longer author duplicated
  `training.output_dir` / `training.logging_dir` directly.
- Proof that dataset-facet parity preserves both dataset paths and the paired
  image-budget contract:
  - `custom.offline_max_pixels`
  - `template.max_pixels`
- Docs evidence that the new facet layout is discoverable from the current
  training routers/runbooks.

## Capabilities

### New Capabilities

- `training-config-hierarchy`: define the canonical reusable YAML hierarchy for
  training configs so shared concern groups are explicit, dataset/global base
  ownership is clear, and leaf profiles remain audit-friendly across COCO, LVIS,
  and future prepared single-dataset families.

### Modified Capabilities

- `stage2-ab-training`: preserve current profile discovery and resolved-profile
  validation while allowing a cleaner shared-facet hierarchy underneath.
- `dataset-prompt-variants`: keep prompt variants explicit and reusable as
  shared composition facets across Stage-1 and Stage-2 training families.
- `fusion-dataset`: retire legacy fusion-config training support instead of
  preserving it alongside the new canonical hierarchy.

## Impact

- Primary config surfaces expected to change:
  - `configs/base.yaml`
  - `configs/_shared/`
  - `configs/stage1/`
  - `configs/stage2_two_channel/`
  - `configs/fusion/` (removal/retirement)
- Primary code surfaces expected to change narrowly as part of the canonical
  contract cleanup:
  - `src/config/loader.py`
  - `src/config/schema.py`
- Primary verification surfaces:
  - `tests/test_stage1_static_packing_runtime_config.py`
  - `tests/test_stage2_ab_profile_leaf_contract.py`
  - `tests/test_training_config_strict_unknown_keys.py`
  - `tests/test_fusion_config.py` (retirement or removal)
- Primary docs surfaces:
  - `docs/training/README.md`
  - `docs/training/LVIS.md`
  - `docs/IMPLEMENTATION_MAP.md`
  - `docs/data/FUSION.md` (retirement or removal)
