## Why

The current teacher-forcing loss surface is correct in behavior, but its organization is harder to extend than it needs to be.

Today the same objective-module facts are repeated in multiple places:

- strict allowlists in the module registry,
- runtime objective/diagnostic registries in the pipeline,
- and Stage-2 objective-atom projection rules in the trainer logging path.

That duplication creates exactly the failure mode this refactor is meant to eliminate:

- adding or reorganizing a loss component requires touching multiple parallel tables,
- bbox-dependent terms such as `ciou` are not represented through a shared bbox taxonomy even though they are clearly bbox semantics,
- and internal naming mixes public module names, semantic roles, and logging atoms without one coherent ownership model.

The result is unnecessary drift risk when extending the loss surface.

## What Changes

- Introduce one canonical module-taxonomy layer inside the teacher-forcing loss registry with:
  - an objective-module catalog,
  - and a companion diagnostic-module catalog.
- Make those catalogs the single source of truth for their owned surfaces:
  - public objective module names,
  - family classification,
  - semantic role,
  - allowed config keys,
  - optional config keys,
  - allowed application presets,
  - module-level Stage-2 `emission_group` routing for objective modules that emit Stage-2 atoms,
  - and projected Stage-2 objective-atom definitions (`atom_name`, `state_key`, `required_state`).
- Explicitly classify bbox-dependent modules under a shared `bbox` family while preserving their distinct roles:
  - `bbox_geo` -> semantic role `geometry`
  - `bbox_size_aux` -> semantic role `size_aux`
- Keep bbox-dependent atoms consistently categorized:
  - `bbox_smoothl1`
  - `bbox_ciou`
  - `bbox_log_wh`
  - `bbox_oversize`
- Make the runtime objective registry validate against the objective catalog and the runtime diagnostics registry validate against the companion diagnostic catalog.
- Make Stage-2 objective-atom projection consume the objective catalog instead of maintaining separate module-specific projection tables.
- Preserve authored YAML execution order and module-to-module state handoff as explicit invariants of the teacher-forcing pipeline.
- Make optional projected atoms explicit:
  - required atoms fail fast when absent under additive projection,
  - explicitly optional atoms may be absent without breaking additivity.
- Require projected emitted atom keys to remain unique within an emission provenance group.
- Add focused regression tests for catalog-driven validation, bbox family separation, registry drift, and projection optionality.

## Recommended First Version

The first official refactor version is intentionally contract-preserving:

- keep the existing public YAML module names unchanged,
- keep the existing emitted objective atom names unchanged,
- keep the existing execution order semantics unchanged,
- keep objective-to-objective state handoff semantics unchanged,
- and improve only the internal organization, taxonomy, and single-source-of-truth guarantees.

That means the first version does **not** rename:

- `bbox_geo`
- `bbox_size_aux`
- `coord_reg`
- `token_ce`
- `loss_duplicate_burst_unlikelihood`

It only reorganizes how those modules are declared and reasoned about.

## Assumptions

- The immediate maintainability win comes more from unifying module identity and projection rules than from renaming public YAML keys.
- A companion diagnostic catalog is the safest first-version design because diagnostics share taxonomy concepts with objectives but do not participate in Stage-2 objective-atom projection.
- Public module-name renames are a separate compatibility question and should not be mixed into the first refactor slice.
- Bbox family classification is valuable even when the public module names remain stable.

## Non-Blocking Follow-Ups

- Move the module runner registration itself behind the same catalog shape if the team wants full registry generation rather than coverage validation.
- Consider public alias-backed name cleanup in a later compatibility change, for example `bbox_geo` -> `bbox_geometry`.
- Consider moving the catalog into a dedicated file if the repo wants a stricter separation between taxonomy and validation helpers.

## Risks To Validity

- If the shared taxonomy layer is underspecified, future modules may still need one-off logic and reintroduce drift.
- If execution order and state handoff are left implicit, a future refactor could look compliant while breaking bbox/coord dependencies.
- If the objective-vs-diagnostic catalog split is described ambiguously, future implementers may “fix” the design in incompatible directions.
- If public names are renamed too early, config compatibility risk will outweigh the refactor benefit.
- If bbox family ownership is described ambiguously, future bbox-dependent terms may still end up split across unrelated surfaces.

## Required Evidence

- Validation that config allowlists still match the authored Stage-2 contract.
- Validation that authored YAML execution order and module-to-module state handoff remain unchanged.
- Validation that objective-atom projection still reconstructs weighted module losses additively.
- Validation that required projected atoms fail fast when absent while explicitly optional projected atoms may remain absent.
- Explicit regression proof that bbox modules share a family while keeping separate semantic roles.
- Explicit regression proof that registry drift fails fast for both objective and diagnostic registries.
- Evidence that no unrelated Stage-2 trainer behavior was silently changed by the refactor.

## Capabilities

### Modified Capabilities

- `teacher-forcing-objective-pipeline`: modify module-registration and validation ownership so companion canonical catalogs define strict runtime coverage while preserving authored execution order.
- `teacher-forcing-unified-loss-registry`: modify the internal loss taxonomy so bbox-dependent modules and atoms are classified coherently by family and semantic role, and so Stage-2 projection remains catalog-owned and explicit.

## Impact

- Immediate impact is internal architecture plus tests.
- The implemented surface is centered on:
  - `src/trainers/teacher_forcing/module_registry.py`
  - `src/trainers/teacher_forcing/objective_pipeline.py`
  - `src/trainers/teacher_forcing/objective_atoms.py`
  - `tests/test_teacher_forcing_loss_catalog.py`
- Public YAML names, objective ordering, and emitted atom names remain unchanged in this first version.
