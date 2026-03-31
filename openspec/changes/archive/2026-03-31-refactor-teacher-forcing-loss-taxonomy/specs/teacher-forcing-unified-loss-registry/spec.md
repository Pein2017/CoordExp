# teacher-forcing-unified-loss-registry Specification (Delta)

## MODIFIED Requirements

### Requirement: Canonical loss component names and contexts are canonical and shared
The unified loss registry SHALL define a canonical taxonomy layer for teacher-forcing modules that separates family classification from semantic role while keeping the current public module names stable.

Normative behavior:

- in addition to canonical public module names and canonical loss component names, every objective and diagnostic module MUST have:
  - a family classification,
  - and a semantic role,
- `semantic_role` MUST be a stable lowercase snake_case internal identifier drawn from a registry-owned vocabulary,
- the objective-module catalog MUST additionally own:
  - optional config keys,
  - allowed application presets,
  - module-level `emission_group` when the module participates in Stage-2 atom emission,
  - and projected atom definitions,
- each projected atom definition MUST include:
  - `atom_name`
  - `state_key`
  - `required_state`,
- the same taxonomy layer MUST be reused across:
  - strict config validation,
  - runtime objective/diagnostic registry coverage validation,
  - and Stage-2 objective-atom projection for objective modules,
- companion diagnostic modules MUST reuse the same core taxonomy fields (`family`, `semantic_role`, `config_keys`) but MUST NOT participate in Stage-2 objective-atom projection unless a future change explicitly adds diagnostic emission semantics,
- duplicated hardcoded module-identity tables across those surfaces are not part of the intended architecture,
- public objective module names and emitted atom names remain unchanged in this first refactor slice.

#### Scenario: Module taxonomy is reused across validation and projection
- **WHEN** the codebase reasons about a teacher-forcing module
- **THEN** config validation, runtime registry coverage, and Stage-2 objective-atom projection reuse the same canonical taxonomy layer for the relevant module kind
- **AND** module identity is not re-authored independently in each surface.

### Requirement: Stage-2 objective-atom projection is module-owned and deterministic
The unified loss registry SHALL keep Stage-2 objective-atom projection explicit, module-owned, and deterministic.

Normative behavior:

- every objective module that participates in Stage-2 objective-atom emission MUST declare a module-level `emission_group`,
- within a single emission provenance group, projected emitted atom keys MUST be unique across objective modules,
- missing projected atoms with `required_state=true` MUST fail fast when additive projection is enforced,
- projected atoms with `required_state=false` MAY be absent without violating additive projection,
- explicit optionality MUST be declared in the projected atom definition rather than inferred from missing state at runtime,
- diagnostics MUST NOT emit Stage-2 objective atoms in this change.

#### Scenario: Optional projected atom absence preserves additivity
- **WHEN** an objective module declares a projected atom with `required_state=false`
- **AND** the backing state tensor is absent during additive projection
- **THEN** projection may omit that atom without failing additivity.

#### Scenario: Projected atom-key collision fails fast
- **WHEN** two objective modules attempt to project the same emitted atom key within one emission provenance group
- **THEN** projection fails fast instead of silently overwriting one contribution.

### Requirement: Geometry loss (`geo`) uses canonicalized boxes and a stable decomposition
The unified loss registry SHALL classify bbox-dependent objective modules and atoms coherently under a shared bbox family while keeping geometry and size auxiliaries semantically separate.

Normative behavior:

- bbox-dependent objective modules MUST be classified under a shared internal `bbox` family,
- within that family:
  - `bbox_geo` MUST have semantic role `geometry`,
  - `bbox_size_aux` MUST have semantic role `size_aux`,
- bbox-geometry atoms such as:
  - `bbox_smoothl1`
  - `bbox_ciou`
  MUST remain owned by the geometry role,
- bbox-size auxiliary atoms such as:
  - `bbox_log_wh`
  - `bbox_oversize`
  MUST remain owned by the size-aux role,
- bbox-dependent terms MUST NOT be folded into `coord_reg` taxonomy or treated as generic coord regularization just because they consume coord-decoded boxes,
- `ciou` remains a bbox-geometry term even when the decoded-box path reuses coord-token predictions.

#### Scenario: CIoU remains categorized as bbox geometry
- **WHEN** the system classifies the `bbox_ciou` objective atom
- **THEN** it is owned by the bbox geometry role under the shared bbox family
- **AND** it is not categorized as a coord-regularization term.
