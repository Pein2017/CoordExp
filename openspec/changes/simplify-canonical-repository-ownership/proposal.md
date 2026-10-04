## Why

Main advertises stable integration but retains hundreds of tests and commands for modules deleted by e1662c2c76d387665c9706ccc75a284ccc2ee682. Current documentation and default test discovery conceal this split, increasing maintenance cost and misleading new callers.

## What Changes

- Remove unreachable legacy tests/commands together with their exclusively dependent launch surfaces; preserve real data-recovery consumers and the maintained annotation systems.
- Make maintained entrypoints, test discovery and checkout-local ownership explicit; repair stale navigation rather than building another archive.
- Record the three-branch role assessment and recommendations here. Implement the independent infrastructure configuration refactor in that checkout's `unify-config-document-loading` change.
- Initially hold research-probes writes while eight-rank work and whole-checkout source admission are active. After a fresh workload/source recheck, implement only its independent ownership/test-layout slice under clarify-research-ownership-and-test-layout.

## Capabilities

### New Capabilities

None. This is retirement of nonfunctional leftovers, test/tooling and documentation maintenance; `skip_specs: true` is explicit.

### Modified Capabilities

None. Current training, inference, geometry, data recovery, annotation and artifact contracts stay unchanged. Historical CLI/import names already refer to absent implementations; no compatibility shims will be introduced.

## Impact

Only main's inspected files are changed by this change. No datasets, outputs, models, reference/vendor trees, active workloads, agent/harness files or sibling worktrees are modified. No commit, push, merge, rebase, reset, clean or stash. Initial HEAD 52d7dd204449a5d610715666359c91ce243b758e was clean; concurrent `.codex/hooks.json` and `.codex/skills/shared-memory` changes observed later remain outside scope. Validation is offline CPU-only plus exact diff and dependency review.
