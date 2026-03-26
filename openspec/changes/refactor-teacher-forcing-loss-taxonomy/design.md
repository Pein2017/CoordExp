## Context

The teacher-forcing stack currently has a healthy external contract but a fragmented internal ownership model.

Three different places need to stay aligned for every objective module:

- module-registry validation,
- runtime pipeline registration,
- and Stage-2 atom projection.

That duplication is manageable with a small fixed set of modules, but it does not scale well as the loss surface grows. It is especially awkward for bbox-dependent terms, because the codebase already treats these losses as conceptually related while the internal structure does not expose that relationship explicitly.

The concrete example the user called out is `ciou`:

- it is clearly bbox-dependent,
- it is implemented under `bbox_geo`,
- but there was no shared internal taxonomy that made `bbox_geo` and `bbox_size_aux` visibly part of one bbox family.

This refactor addresses that organization problem without changing the public Stage-2 contract.

## Goals / Non-Goals

**Goals**

- Define one canonical internal catalog for objective-module identity.
- Separate internal family classification from public module naming.
- Make bbox-dependent modules explicitly share a `bbox` family while keeping distinct semantic roles.
- Keep Stage-2 atom projection and config validation derived from the same catalog.
- Preserve current YAML names, metric names, and execution ordering.

**Non-Goals**

- No public YAML module rename in this change.
- No metric-key rename in this change.
- No trainer-behavior change outside the teacher-forcing registry/pipeline/projection surfaces.
- No redefinition of bbox parameterization or geometry semantics.

## Decisions

### 1) Use one canonical catalog as the single source of truth for objective-module identity

The refactor introduces a shared internal module catalog that owns, per objective module:

- public module name,
- family,
- semantic role,
- strict config allowlist,
- optional config keys,
- allowed application presets,
- and projected Stage-2 atom mappings.

Why:

- this removes parallel handwritten tables,
- makes extension easier,
- and ensures config validation and logging projection cannot drift independently.

### 2) Separate internal family classification from public module names

The first version keeps public names stable, but internally classifies modules by family.

This allows the codebase to answer questions such as:

- which modules are bbox-related?
- which modules are coord-related?
- which modules emit text-side vs coord-side objective atoms?

without forcing a public rename.

Why:

- the user’s readability/scalability goal is largely internal taxonomy,
- while public renames are a separate compatibility concern.

### 3) Make bbox modules explicitly share a family and keep their semantic roles separate

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

### 4) Validate runtime registries against the same catalog

The runtime objective and diagnostics registries remain explicit in the pipeline, but they now validate their coverage against the shared catalog.

That means:

- missing runners fail fast,
- stale catalog entries fail fast,
- and unregistered additions fail fast.

Why:

- it improves safety immediately,
- while avoiding a larger runner-factory refactor in the first slice.

### 5) Drive Stage-2 atom projection from module-owned atom definitions

The Stage-2 objective-atom projector now resolves module projections from the same catalog.

This keeps one ownership path for:

- what atoms a module projects,
- which state keys back those atoms,
- and whether a projected state tensor is required or optional.

Why:

- objective projection is one of the easiest places for silent drift,
- and catalog-driven projection makes extension much safer.

### 6) Keep first-version compatibility strict

This change keeps the current contract stable:

- public YAML module names remain unchanged,
- emitted Stage-2 atom names remain unchanged,
- config keys remain unchanged,
- and objective execution order remains unchanged.

Why:

- the change is meant to improve internal coherence first,
- not to couple refactoring risk with public-surface churn.
