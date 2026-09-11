## Why

Two current research-probe consumers independently reimplement file and
directory identity, destination-relative data resolution, executable identity,
plan-to-manifest projection checks, reserved output-root checks, and final
write-once evidence closure.  The accepted evidence journal makes individual
work durable, but it does not prove that a costly launch used the intended
production entrypoint or that its raw output crossed the real finalizer and
downstream validator before bulk work begins.

This gap caused mechanically preventable failures after model loading or GPU
execution: a launcher shim consumed a command, relocated relative image paths
became unreadable, equivalent directory inventories disagreed, manifest
projections drifted, a live callback reached canonical metadata, and CPU
finalizers rejected otherwise valid GPU output.  A small admission owner is
needed now because the natural-boundary support and K10-H20 crossover paths are
two concrete consumers of the same mechanics.

## What Changes

- Add a mechanics-only research-probe admission capability that captures and
  live-revalidates typed regular-file, directory-tree, resolved-data-file,
  absolute-executable, strict-value, and reserved-absent-path bindings.
- Use full canonical identities, deterministic relative-path directory
  inventories, strict JSON values, and crash-consistent write-once artifacts;
  never stringify callbacks, tensors, paths, handles, or other live objects.
- Add a journal-backed two-stage admission dossier.  It requires one CPU
  production-path preflight record and one bounded vertical-smoke record under
  the same immutable binding identity before publishing an admitted receipt.
- Require stage evidence to bind the exact producer and validator artifacts,
  output artifacts, binding fingerprint, and mechanics assertions.  Admission
  validates evidence closure but does not launch work, retry it, or infer that
  a caller exercised the correct scientific condition.
- Add consumer examples and compatibility tests grounded in the existing
  natural-boundary support and K10-H20 crossover paths without editing or
  merging into the active `research-probes` worktree.
- Prove the CPU gate through real consumer validators and run one bounded
  single-GPU mechanics smoke through an existing production worker, durable
  record, terminal materializer, and the bounded mechanics validator for that
  terminal.  The smoke does not claim closure of the legacy eight-shard
  scientific merger; it remains non-scientific and writes only fresh
  change-owned roots.
- Keep logical planning, cost scheduling, process supervision, automatic
  retry, actuator semantics, owner matching, outcomes, estimands, thresholds,
  stop rules, and scientific interpretation outside this capability.

## Capabilities

### New Capabilities

- `coordexp-infras-research-probe-admission`: Typed execution-input binding and
  journal-backed CPU-plus-vertical mechanics admission for production-shaped
  research-probe launches.

### Modified Capabilities

None.

## Impact

- New stable mechanics owner under `src/artifacts/`, reusing strict JSON,
  exclusive publication, and `ExecutionEvidenceJournal` without changing their
  disk schemas.
- New focused tests under `tests/artifacts/` plus worktree-local consumer
  examples and compatibility checks under `scripts/research/` and
  `tests/research/`.
- Existing natural-boundary scheduling, support observations, legacy receipt
  schemas, merger/analyzer semantics, crossover endpoints, model forward
  behavior, checkpoint/config identities, and sealed research roots remain
  unchanged.
- No new service, queue, scheduler, retry controller, model dependency, or
  scientific-result registry is introduced.  No merge, cherry-pick, push, or
  active-unit mutation is part of this change.
