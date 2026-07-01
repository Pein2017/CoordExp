## Context

The teacher-forcing stack currently has a healthy external contract but a fragmented internal ownership model.

Three different places need to stay aligned for every objective module:

- module-registry validation,
- runtime pipeline registration,
- and Stage-2 atom projection.

Diagnostics add a fourth alignment surface for runtime coverage validation, but diagnostics do not participate in Stage-2 objective-atom projection.

That duplication is manageable with a small fixed set of modules, but it does not scale well as the loss surface grows. It is especially awkward for bbox-dependent terms, because the codebase already treats these losses as conceptually related while the internal structure does not expose that relationship explicitly.

The concrete example the user called out is `ciou`:

- it is clearly bbox-dependent,
- it is implemented under `bbox_geo`,
- but there was no shared internal taxonomy that made `bbox_geo` and `bbox_size_aux` visibly part of one bbox family.

This refactor addresses that organization problem without changing the public Stage-2 contract.

## Terminology

- `public module name`: the stable YAML identifier for a teacher-forcing objective or diagnostic module.
- `family`: a stable internal classification that groups related modules such as `bbox`, `coord`, `text`, `rollout`, or `diagnostic`.
- `semantic_role`: a stable lowercase snake_case internal identifier from a registry-owned vocabulary. It is distinct from public module naming and captures what the module semantically does inside its family.
- `emission_group`: the module-level Stage-2 routing for projected objective atoms. In this change, objective modules may emit into `text` or `coord` groups, while diagnostics do not emit Stage-2 objective atoms.
- `projected atom definition`: a catalog-owned entry with `atom_name`, `state_key`, and `required_state` describing one Stage-2 objective atom contribution.
- `required projected atom`: a projected atom whose state tensor must exist when additive projection is enforced.
- `optional projected atom`: a projected atom whose state tensor may be absent without violating additive projection.

## Goals / Non-Goals

**Goals**

- Define one canonical taxonomy layer for teacher-forcing modules.
- Separate internal family classification from public module naming.
- Make bbox-dependent modules explicitly share a `bbox` family while keeping distinct semantic roles.
- Keep Stage-2 atom projection and config validation derived from the same catalog-owned definitions.
- Preserve current YAML names, metric names, execution ordering, and module-to-module state handoff.
- Make objective-vs-diagnostic ownership explicit rather than implied.

**Non-Goals**

- No public YAML module rename in this change.
- No metric-key rename in this change.
- No trainer-behavior change outside the teacher-forcing registry/pipeline/projection surfaces.
- No redefinition of bbox parameterization or geometry semantics.
- No attempt to make diagnostics emit Stage-2 objective atoms in this change.

## Decisions

### 1) Use one canonical taxonomy layer with companion objective and diagnostic catalogs

The refactor introduces a shared internal taxonomy layer in the registry module.

Within that layer:

- the objective catalog owns, per objective module:
  - public module name,
  - family,
  - semantic role,
  - strict config allowlist,
  - optional config keys,
  - allowed application presets,
  - module-level `emission_group`,
  - and projected atom definitions,
- the companion diagnostic catalog owns, per diagnostic module:
  - public module name,
  - family,
  - semantic role,
  - and strict config allowlist.

Why:

- this removes parallel handwritten tables,
- keeps objective and diagnostic ownership explicit,
- and avoids implying that diagnostics must participate in objective-only projection behavior.

### 2) Preserve authored YAML execution order and state handoff as first-class invariants

Objective modules continue to execute in authored YAML order.

Within a single teacher-forcing pass:

- runtime registry validation MUST NOT reorder the objective list,
- downstream objective modules MAY depend on state emitted by earlier objective modules,
- and state handoff remains part of the public behavior of the ordered objective pipeline even though the taxonomy organization is internal.

Why:

- modules such as `bbox_size_aux` and `coord_reg` depend on artifacts prepared by `bbox_geo`,
- and preserving ordering/state handoff prevents a “catalog-driven” refactor from accidentally becoming a behavior change.

### 3) Separate internal family classification from public module names and normalize semantic-role naming

The first version keeps public names stable, but internally classifies modules by family and semantic role.

This allows the codebase to answer questions such as:

- which modules are bbox-related?
- which modules are coord-related?
- which modules emit text-side vs coord-side objective atoms?
- which modules are diagnostics rather than trainable objective terms?

The semantic-role vocabulary remains small, stable, and lowercase snake_case. For this first slice it includes at least the current roles such as:

- `token_ce`
- `duplicate_burst_unlikelihood`
- `geometry`
- `size_aux`
- `regularizer`
- `diagnostic`

Why:

- the user’s readability/scalability goal is largely internal taxonomy,
- semantic roles need stronger normalization than free-form labels,
- while public renames are a separate compatibility concern.

### 4) Make bbox modules explicitly share a family and keep their semantic roles separate

The bbox family is defined as at least:

- `bbox_geo` with semantic role `geometry`
- `bbox_size_aux` with semantic role `size_aux`

Within that family:

- `bbox_smoothl1` and `bbox_ciou` remain geometry atoms,
- `bbox_log_wh` and `bbox_oversize` remain size-aux atoms.

Why:

- this makes bbox ownership explicit,
- prevents `ciou` or future bbox-dependent terms from being misclassified,
- and keeps semantically different bbox losses decoupled.

### 5) Validate runtime registries against their corresponding companion catalogs

The runtime objective and diagnostics registries remain explicit in the pipeline, but they now validate coverage against their corresponding catalogs.

That means:

- objective handlers are validated against the objective catalog,
- diagnostic handlers are validated against the companion diagnostic catalog,
- missing runners fail fast,
- stale catalog entries fail fast,
- and unregistered additions fail fast.

Why:

- it improves safety immediately,
- while avoiding a larger runner-factory refactor in the first slice,
- and it removes ambiguity about whether diagnostics share the exact same catalog object as objectives.

### 6) Drive Stage-2 atom projection from module-owned definitions

The Stage-2 objective-atom projector resolves module projection from objective-module definitions.

Those definitions own:

- module-level `emission_group`,
- projected atom `atom_name`,
- backing `state_key`,
- and `required_state` optionality.

Within a single emission provenance group:

- emitted atom keys remain unique,
- missing required projected atoms fail fast when additive projection is enforced,
- and explicitly optional projected atoms may be absent without breaking additivity.

Diagnostics do not participate in Stage-2 objective-atom projection in this change.

Why:

- objective projection is one of the easiest places for silent drift,
- catalog-driven projection makes extension much safer,
- and explicit optionality avoids forcing placeholder tensors for optional contributions.

### 7) Keep first-version compatibility strict

This change keeps the current contract stable:

- public YAML module names remain unchanged,
- emitted Stage-2 atom names remain unchanged,
- config keys remain unchanged,
- objective execution order remains unchanged,
- and module-to-module state handoff remains unchanged.

Why:

- the change is meant to improve internal coherence first,
- not to couple refactoring risk with public-surface churn.
