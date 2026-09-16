## 1. Release binding and first vertical slice

- [x] 1.1 Bind the isolated 0.153.4 release worktree and retain the installed-Core failure/recovery baselines; verify the base hash, clean initial worktree, proposal, and linked receipts.
- [x] 1.2 Implement explicit per-server policy parsing and disabled defaults; verify positive interval validation, required recovery kind, and unsupported-transport behavior through configuration tests and generated schema checks.
- [x] 1.3 Implement selected stdio suspension and singleflight reconstruction beneath existing prepared bindings; prove the missing behavior through a caller-facing counterexample and pass focused suspend/rebuild/no-replay tests.

## 2. Task admission and context protection

- [x] 2.1 Integrate the shared lifecycle with actual Core turn/direct-call admission and completed-child classification; verify deadline races, long calls, pending interactions, and uncertainty prevent unsafe suspension.
- [x] 2.2 Preserve startup-project semantics, noneligible clients, and passive observation; verify same-project Serena activation remains reclaimable, unsupported context mutation is retained, and status/prewarm does not recreate suspended processes.

## 3. Real entrypoint acceptance

- [x] 3.1 Run required scoped Rust checks and build the candidate app-server/CLI with one build owner; retain command exit statuses and exact source/binary identities.
- [x] 3.2 Exercise the exact candidate through isolated app-server/stdio tests covering main-task suspend/reuse, concurrent demand, another active task, and background-work preservation; verify process identities, no transport-closed regression, one replacement, and cleanup. Bound each scenario to at most two test tasks/servers, one isolated daemon where needed, and no real model/GPU work.
- [x] 3.3 Verify the completed-subagent grace path and wake-frontend independence through the nearest real consumer plus focused characterization; retain explicit evidence for daemon/record preservation without delivering test wakes to production tasks.

## 4. Delivery

- [x] 4.1 Package a versioned, reproducible runtime candidate and operator configuration/activation/rollback instructions; verify package hashes and replay the decisive entrypoint check against the packaged binary, preserving the stock runtime.
- [x] 4.2 Complete root OpenSpec validation, exact-diff review, scoped commits and verification receipts; report implementation, test, package, and shared-activation status separately, with unrelated work preserved.
