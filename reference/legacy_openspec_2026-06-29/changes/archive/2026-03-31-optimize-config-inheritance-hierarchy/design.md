## Context

CoordExp already has strong config fundamentals:

- YAML-first operation through `extends`,
- strict typed parsing in `src/config/schema.py`,
- fail-fast unknown-key validation,
- and canonical profile checks for the active Stage-2 tree.

The weakness is not the parsing model. It is ownership inside the hierarchy.

The current loader in `src/config/loader.py` performs a simple recursive
inheritance walk and a plain deep merge. That gives the repo a deterministic and
inspectable config story, but it also means hierarchy mistakes become runtime
mistakes unless the YAML tree itself is carefully structured.

The current structure shows three concrete issues:

1. `configs/base.yaml` is carrying dataset identity even though it is supposed to
   be the repo-global base.
2. Stage-specific bases still mix reusable concern groups:
   - runtime defaults
   - dataset identity
   - prompt identity
   - objective bundles
   - observability bundles
3. list-valued sections such as objective manifests are vulnerable to accidental
   replacement if split across multiple parents without a clear single owner.

The intended end state is not a flatter tree at all costs. It is a hierarchy
whose merge order makes concern ownership obvious:

`global base`
-> `stage runtime base`
-> `dataset facet`
-> `prompt facet`
-> `objective/observability facet`
-> `explicit leaf identity`

For canonical migrated families, both Stage-1 and Stage-2 should participate in
that same repo-global hierarchy rather than leaving Stage-2 as a permanent
standalone exception to the universal base.

The hierarchy is intended to be universal within the repo's current default
training posture: single-dataset training over prepared JSONL/image assets. COCO
and LVIS are the current migration exemplars, not the only dataset families the
layout should support.

This universality is intentionally bounded to prepared single-dataset families.
Legacy multi-dataset example config authoring under `configs/fusion/` is not
part of the target hierarchy and may be removed rather than migrated. This
change intentionally treats fusion as legacy and can retire the corresponding
runtime/schema capability rather than preserving it.

## Goals / Non-Goals

**Goals**

- Make the config hierarchy easier to reason about without changing merge
  semantics or strict parsing behavior.
- Remove hidden dataset defaults from the repo-global base.
- Reuse dataset identity and prompt identity through explicit shared facets.
- Keep the hierarchy scalable to future prepared dataset families without
  schema or loader special-casing.
- Keep Stage-1 and Stage-2 runtime ownership narrow and stage-local.
- Preserve auditability by keeping high-signal leaf identity explicit.
- Add resolved-config parity tests so the refactor is behavior-preserving.
- Keep the authored inheritance chain shallow enough that config provenance
  remains easy to inspect.

**Non-Goals**

- No Hydra/Jinja/template DSL migration.
- No new top-level config sections.
- No weakening of strict unknown-key behavior.
- No automatic derivation of `run_name`.
- No opaque naming magic for artifact paths beyond repo-owned
  root-plus-`artifact_subdir` composition under the existing `training` section.
- No broad `CustomConfig` dataclass redesign in the first slice.
- No attempt to preserve legacy fusion/multi-dataset configs in this change.

## Decisions

### Decision 1: Keep the current loader and strict schema unchanged

Decision:

- Preserve `extends` plus plain deep-merge semantics as the canonical config
  composition model.
- Preserve strict typed parsing and unknown-key fail-fast behavior while
  allowing targeted schema-key changes needed for the new canonical contract.
- Preserve multi-parent composition, but keep the authored hierarchy shallow and
  auditable.

Why:

- The current loader is deterministic and already well-covered by existing
  contract tests.
- The current problem is hierarchy/layout readability, not parser capability.

Alternative considered:

- Introduce a smarter merge engine, semantic list merging, or a templating DSL.
  Rejected because it increases magic and creates a larger correctness surface
  than the current problem requires.

### Decision 2: Make the repo-global base truly global

Decision:

- `configs/base.yaml` must no longer own dataset-specific or prompt-specific
  training identity.
- The repo-global base should cover only stable global concerns such as:
  - model defaults
  - template defaults
  - shared training defaults
  - shared data-loader defaults

Why:

- Dataset identity in the repo-global base creates hidden inherited behavior and
  makes leaves less trustworthy to audit.

Alternative considered:

- Keep dataset defaults in `configs/base.yaml` and document them more clearly.
  Rejected because the hidden-precedence problem remains.

### Decision 3: Introduce shared facets for reusable concern groups

Decision:

- Add shared dataset facets under `configs/_shared/datasets/`.
- Add shared prompt facets under `configs/_shared/prompts/`.
- Keep stage-specific reusable objective/observability bundles under the
  existing stage-local `_shared/` trees.
- Treat COCO/LVIS facet files as initial concrete instances of a general pattern,
  not a closed registry baked into code.

Recommended ownership:

- dataset facet:
  - `custom.train_jsonl`
  - `custom.val_jsonl`
  - `custom.offline_max_pixels`
  - dataset-specific `custom.object_ordering` when applicable
  - related `template.max_pixels` only when tied to the same prepared dataset
- prompt facet:
  - `custom.object_field_order`
  - `custom.extra.prompt_variant`
- stage-local objective facet:
  - loss bundle or objective manifest
- stage-local observability facet:
  - eval/monitor/dump defaults when those are shared across a stage family

Why:

- These concern groups are reused across multiple leaves and currently scattered
  across different parent files.
- The same typed keys are already sufficient for future prepared datasets, so
  the hierarchy should encode a reusable pattern rather than dataset-specific
  branching in code.

Alternative considered:

- Keep all shared fields inside broad stage bases.
  Rejected because broad bases hide ownership and encourage unrelated coupling.

### Decision 4: Keep Stage-1 and Stage-2 runtime bases separate and smaller

Decision:

- `configs/stage1/sft_base.yaml` remains the Stage-1 runtime base.
- `configs/stage2_two_channel/base.yaml` remains the Stage-2 runtime base.
- For canonical migrated families, both stage bases should layer on top of
  `configs/base.yaml`; the Stage-2 base should not remain a permanent
  standalone exception to the universal-base contract.
- Both should narrow to runtime ownership only, not dataset or prompt identity.
- The migration is not complete until shared dataset and prompt identity have
  moved out of those stage-local bases for the leaf families that consume the
  new shared facets.
- The preferred authored hierarchy is:
  - universal base
  - stage-wise base
  - shared/common reusable package settings
  - specialized experiment leaf
- Canonical repo-owned configs should therefore target a maximum authored depth
  of 4 semantic levels from universal base to specialized experiment leaf.
- Derivative wrappers such as smoke/runtime-limit overlays are allowed on top
  of a specialized leaf, but they must stay narrow and are outside the 4-level
  semantic budget.

Why:

- Stage-1 and Stage-2 have different runtime semantics and should not be forced
  into a single universal base.

Alternative considered:

- Create a single shared training base for all stages.
  Rejected because it would either become too generic to help or too broad to be
  safe.

### Decision 5: Keep explicit leaf identity while deduplicating artifact paths

Decision:

- Leaf configs should continue to author:
  - `model.model`
  - `training.run_name`
  - `training.artifact_subdir`
- Roots should be reusable and separately configurable through repo-owned
  internal training keys:
  - `training.output_root`
  - `training.logging_root`
- Canonical migrated configs should derive:
  - `training.output_dir = training.output_root / training.artifact_subdir`
  - `training.logging_dir = training.logging_root / training.artifact_subdir`
  before downstream trainer initialization.
- Canonical repo-owned configs should stop authoring duplicated
  `training.output_dir` / `training.logging_dir` directly.
- Leaves should also keep any experiment-defining runtime overrides explicit,
  such as effective batch size, step cadence, or sample caps.

Why:

- Researchers and operators mostly care about the shared artifact suffix and run
  identity, not about repeating the same suffix twice with different roots.
- The loader already performs one path-composition step today by appending
  `training.run_name` to `training.output_dir`, so root-plus-subdir composition
  extends an existing repo-owned pattern instead of introducing a new config
  system.

Alternative considered:

- Keep `training.output_dir` and `training.logging_dir` fully duplicated in
  leaves forever.
  Rejected because it preserves a noisy, error-prone duplication pattern with
  no real audit benefit once `training.artifact_subdir` stays explicit.

### Decision 5b: Shut down authored fusion runs while keeping dormant legacy assets

Decision:

- This change keeps `configs/fusion/` as a dormant legacy example tree for
  future reactivation.
- This change disables the `custom.fusion_config` authored runtime surface in
  schema/runtime/tests/docs instead of preserving it as an active escape hatch.

Why:

- The canonical hierarchy should stay single-dataset-first today.
- Keeping dormant examples/modules in-tree reduces future reactivation cost
  without leaving runtime fusion authoring silently active.

Alternative considered:

- Keep `custom.fusion_config` alive after removing the example folder.
  Rejected because it preserves a hidden legacy contract and weakens the goal of
  a single canonical training hierarchy.

### Decision 6: List-valued bundles must have one facet owner

Decision:

- Any list-valued section that is merge-sensitive must have a single owner in the
  inheritance chain.
- In particular, objective manifests such as:
  - `stage2_ab.pipeline.objective`
  - `stage2_ab.pipeline.diagnostics`
  - rollout-aligned equivalents
  must be owned by one facet or one leaf, not assembled by stacking multiple
  list-providing parents.

Why:

- The current merge behavior replaces lists; it does not concatenate or merge
  them semantically.
- The implementation should treat overlapping list ownership in the same
  inheritance chain as an invalid hierarchy pattern, whether that is enforced by
  validation, tests, or both.

Alternative considered:

- Compose list fragments via multiple parent files.
  Rejected because it is too easy to produce silent replacement behavior.

### Decision 7: Migrate by resolved-config parity, not by visual similarity

Decision:

- Migration should proceed in small slices and validate resolved-config parity on
  representative leaves before broad rollout.

Representative parity set for the first migration slice:

- Stage-1 LVIS production leaf
- Stage-1 LVIS smoke leaf
- one Stage-1 COCO leaf
- Stage-2 LVIS production leaf
- one canonical Stage-2 COCO prod leaf

This representative set validates the first rollout, but the contract itself is
not limited to those families.

Verification wording refinement:

- shared dataset/prompt facets should be verified through minimal composed leaf
  fixtures that include the facet plus the surrounding required training keys,
  not by trying to load a raw facet fragment as a standalone config.
- dataset-facet parity must cover both dataset identity and image-budget
  identity:
  - `custom.train_jsonl`
  - `custom.val_jsonl`
  - `custom.offline_max_pixels`
  - `template.max_pixels`
- artifact-path parity must cover both resolved directories:
  - `training.output_dir`
  - `training.logging_dir`
- raw ownership checks should confirm the canonical migrated leaf itself still
  authors:
  - `model.model`
  - `training.run_name`
  - `training.artifact_subdir`
- a named regression suite cannot be used as migration proof while it is
  already red for unrelated reasons; such suites must be stabilized or
  partitioned first.

Why:

- Visual hierarchy improvements are not enough; the resolved config is the real
  contract.

Alternative considered:

- Bulk-migrate all configs first and rely on current tests to catch drift.
  Rejected because hidden inherited behavior is part of the current problem.

### Decision 8: Remove legacy `configs/fusion/` examples rather than stretching the new hierarchy around them

Decision:

- Remove or retire `configs/fusion/` as a canonical example/config-folder
  surface as part of this cleanup.
- Treat multi-dataset or fusion-first config authoring as out of scope for this
  hierarchy contract.
- Retire `custom.fusion_config` in the same cleanup rather than leaving a
  legacy capability behind.

Why:

- The repo's default posture is single-dataset training.
- Keeping a legacy fusion config surface alive while purifying `configs/base.yaml`
  would leave a hidden exception to the new hierarchy rules.

Alternative considered:

- Preserve `configs/fusion/` and adapt it to the new hierarchy.
  Rejected because it broadens this change into a non-default path and weakens
  the single-dataset contract.

## Target Layout

```text
configs/
  base.yaml
  _shared/
    datasets/
      coco_768_bbox_max60.yaml
      coco_1024_bbox_max60.yaml
      lvis_1024_bbox_max60.yaml
    prompts/
      coco80_desc_first.yaml
      coco80_geometry_first.yaml
      lvis_stage1_federated.yaml
      lvis_stage2_federated.yaml
  stage1/
    sft_base.yaml
    _shared/
      objectives/
      observability/
    ...
  stage2_two_channel/
    base.yaml
    _shared/
      objectives/
      observability/
    ...
```

This layout is illustrative rather than exhaustive. The key invariant is
ownership by concern, not exact folder names.

## Migration Plan

1. Create the new change-local spec and docs baseline for the hierarchy rules.
2. Inventory current `configs/base.yaml` consumers and record which families are
   in-scope for migration, including non-stage helper surfaces.
3. Stabilize any named baseline suite that is already red, especially Stage-2
   profile-contract coverage, before using it as migration proof.
4. Remove or retire `configs/fusion/` as a canonical config-folder surface and
   retire `custom.fusion_config` coherently across schema/runtime/docs/specs/tests.
5. Remove dataset-specific defaults from `configs/base.yaml`.
6. Reconcile canonical Stage-2 runtime inheritance onto `configs/base.yaml`
   rather than leaving it as a permanent standalone exception.
7. Introduce shared dataset facets for representative COCO and LVIS profiles and
   document the reusable pattern for future prepared single-dataset families.
8. Introduce shared prompt facets that own prompt variant and field order.
9. Introduce reusable artifact roots plus explicit `training.artifact_subdir`
   so canonical configs stop duplicating the same suffix across
   `training.output_dir` and `training.logging_dir`.
10. Repoint representative and in-scope canonical Stage-1/Stage-2 leaves to the
    new facet chain while keeping the authored hierarchy within the preferred
    semantic depth budget.
11. Split stage-local objective or observability bundles only where they are
    genuinely reused.
12. Add resolved-config parity tests for the representative and in-scope
    migrated leaves.
13. Add raw ownership checks and minimal composed fixture coverage for the new
    shared dataset/prompt facets and artifact-path contract so facet breakage is
    visible even outside the representative leaf set.
14. Update training docs and implementation map to reflect the new facet layout.
15. Confirm the documented facet pattern for a future dataset family uses the
    same existing typed keys and does not require loader/schema branching.

## Risks / Trade-offs

- [Consumers depend on hidden base defaults]
  - Mitigation: remove dataset defaults from `configs/base.yaml` first and cover
    representative leaves with parity tests.
- [List-valued sections are accidentally replaced]
  - Mitigation: enforce single-owner facet rules for objective/diagnostic lists.
- [Too much abstraction makes leaves harder to audit]
  - Mitigation: keep explicit leaf identity for model/run/artifact-subdir
    fields.
- [Schema creep sneaks into a layout refactor]
  - Mitigation: explicitly keep loader/schema behavior unchanged in the first
    slice and use existing strict-config tests as guardrails.

## Verification Strategy

- Config parity:
  - resolved-config comparisons for representative leaves via
    `ConfigLoader.load_materialized_training_config(...)`
- Facet coverage:
  - minimal composed fixture tests that include the new shared dataset facets and
    shared prompt facets inside a valid training-config harness
- Existing contract tests:
  - `tests/test_stage1_static_packing_runtime_config.py`
  - `tests/test_stage2_ab_profile_leaf_contract.py`
  - `tests/test_training_config_strict_unknown_keys.py`
- Legacy-surface cleanup:
  - no remaining unintended references to `configs/fusion/` as a canonical
    config-folder surface after removal/retirement
  - no remaining live `custom.fusion_config` references in canonical runtime,
    docs, specs, or tests after retirement
- Additional targeted assertions:
  - `configs/base.yaml` no longer contains dataset-specific training paths
  - representative leaves still resolve the same:
    - dataset paths
    - `custom.offline_max_pixels`
    - `template.max_pixels`
    - prompt variants
    - object ordering / field order
    - output/logging/run identity

## Open Questions

- Whether observability facets should live under stage-local `_shared/` only, or
  whether some monitoring defaults are reusable enough to merit a global shared
  area.
- Whether rollout-aligned Stage-2 configs should adopt the same facet layout in
  the first slice or follow immediately after the `stage2_two_channel` pass.
