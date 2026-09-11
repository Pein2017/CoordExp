## Why

The current research-probe admission path revalidates its binding manifest when
it accepts a stage, but the production-shaped vertical launcher starts the
model worker before that vertical-stage append. A `research-probes` target can
therefore drift after CPU admission and consume model work before its evidence
is rejected. The research baseline needs a narrow, reproducible pre-model
identity gate before more probes depend on it.

## What Changes

- Add a target-tree binding to research-probe admission that identifies the
  resolved target worktree a consumer will execute from (currently
  `research-probes`, later a clean `probe/<ticket>` worktree), its exact Git
  commit, its clean status, and the declared effective source/runtime/config
  inputs used by that consumer.
- Require each production-shaped consumer to revalidate that target binding at
  one pre-model choke point immediately before its launcher can run. A stale,
  dirty, unresolved, symlinked, or otherwise mismatched target fails with a
  typed error before launcher, model-load, GPU, or vertical-evidence action.
- Preserve the existing mechanics-only admission boundary and the two current
  consumer adapters. Consumer-owned cohort, prefix, intervention, matching,
  estimand, result, claim, and stop-rule semantics remain outside the shared
  capability.
- Add focused tests and a production-shaped no-model sentinel proving that a
  post-CPU target mutation cannot reach the worker launcher.
- Keep whole-worktree cleanliness as the reusable-admission policy: planning
  dirt may exist in a shared baseline, but that checkout is not an admitted
  model-launch target until it is clean. The normal execution target is the
  clean, disposable `probe/<ticket>` worktree created from a baseline tag.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `coordexp-infras-research-probe-admission`: admission now binds and
  revalidates a clean target research tree before any model-launch boundary,
  rather than detecting a target mismatch only while recording later evidence.

## Impact

- Expected implementation surfaces are the admission binding owner, the two
  existing consumer adapters, the production-shaped vertical driver, and their
  focused CPU tests in the fixed
  `/data/CoordExp/.worktrees/research-probe-infras` integration worktree.
- The accepted change will be merged into `research-probes` and revalidated in
  that exact target tree before it can contribute to `research-base-v1`.
- No generic runner, DAG, plugin, TensorFlow environment, data-manipulation
  framework, scientific result schema, GPU launch, or raw-artifact migration
  is in scope.
