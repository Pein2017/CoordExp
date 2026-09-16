## Context

See `proposal.md` for motivation and scope. Baseline probes exercise installed
Core 0.153.4, Serena package 1.7.0, and the installed wake frontend. They prove
normal stdio teardown and explicit reconstruction, not an automatic idle policy.

The release source at `3d2ee51ca2d5db578f328aa75e20aa22c0197c9a` is the
implementation base. The previously inspected main checkout differs materially
from that release, so exact integration is re-anchored to the isolated release
worktree. Existing `codex-mcp` runtime publication, connection reuse, required
server validation, and Core activity entrypoints remain the owners.

## Goals / Non-Goals

**Goals:** one owner-controlled suspension/reconstruction lifecycle, explicit
eligibility, unchanged disabled behavior, and production-shaped evidence for
main-task continuity and child resource release.

**Non-Goals:** external PID scanners, a shared Serena backend, arbitrary MCP
state checkpointing, new Desktop APIs merely to operate a timer, subagent
history-policy redesign, wake queue repair, or changes to model/research work.

## Decisions

### Keep the lifecycle with the connection owner

Core supplies task classification and turn admission; `codex-mcp` owns eligible
connections and actual MCP operations. Idle observation uses existing runtime
lifetime/cancellation machinery and bounded timer state, not another daemon.
Cold start/resume behavior remains unchanged in this first implementation.

Alternatives considered: external termination fails the installed transport
recovery test; global reload affects every loaded task; Serena-only LSP sleep
reclaimed 52.7% of one small fixture's PSS but retained its Python frontend and
requires separate executor/cache corrections. Full-task eviction can terminate
background execution, making it broader than the requested MCP reclaim.

### Serialize activity versus suspension

Use one lifecycle admission boundary shared by the relevant turn/direct-call
paths and idle maintenance. Timer admission rechecks deadline and activity;
operation admission prevents suspension until the operation's lifetime ends.
An existing refresh mutex alone is insufficient because it does not cover
execution after a binding is obtained. Cached metadata must not hold an activity
lease forever. Track uncertain timeout/cancellation/error states conservatively;
an MCP response deadline is not proof that an underlying tool body stopped.

### Reconstruct through existing transport initialization

The release already retains transport recipes and initialize context below the
published bindings. Add explicit suspension/reconstruction there so existing
authoritative bindings can retain a managed handle without retaining a live
transport. This is smaller than replacing every prepared binding. Revalidate
the restored handshake/tool contract before publishing the replacement inside
that handle. Close only selected reconstructible stdio clients, preserve
unaffected clients, and share reconstruction across concurrent demand. Required
server validation applies before use. Do not transparently retry an operation
whose execution may already have begun.

Passive inventories use retained bounded metadata/connection state; auth/config
refresh must not start an intentionally suspended server merely to prewarm it.
Configuration changes are applied before demand reconstructs a suspended client.

### Preserve context without a generic replay framework

The first eligibility policy is explicit and local. Per-server configuration
uses optional positive `idle_timeout_sec`, optional positive
`idle_timeout_completed_subagent_sec` (falling back to the main interval), and
an explicit `idle_recovery` kind when enabled: `stateless` or
`serena_startup_project`. Absent idle fields preserve current behavior.

Wake frontend registrations are durable outside the frontend. Serena restoration
must remain bound to the same effective project. Ordinary startup-project use
and a provably identical absolute startup-project activation remain eligible;
a different/name-only activation or another unreconstructible session mutation
prevents automatic suspension. Never replay arbitrary tool calls or infer
project identity from a reused PID or a neighboring task's cwd.

### Keep activation separate from candidate verification

The code defaults to disabled outside explicit operator configuration. Initial
operator grace values are 900 seconds for main tasks and 120 seconds for
completed native subagents; isolated tests can use shorter positive values.
Do not lower global concurrency or change subagent expiry to implement this.

## Risks / Trade-offs

- Idle/call race -> one admission boundary plus deterministic concurrent tests.
- Timed-out server work still running -> preserve uncertain instances, with a
  regression based on the observed Serena executor behavior.
- Wrong project after restart -> project-context test and conservative refusal
  when restoration is unavailable.
- Status/prewarm resurrects idle processes -> real status-only observation test.
- Old bindings retain or address closed clients -> lifetime and generation tests,
  plus same-task real call after a short idle interval.
- Cold latency -> record startup/query timing without extrapolating the tiny
  fixture to large repositories.
- Release/source drift -> pinned isolated source, reproducible build identity,
  and acceptance through the exact built binary.

## Migration Plan

Implement and test in the isolated release worktree. Stage a versioned candidate
with source diff, binary hashes, configuration example, and isolated entrypoint
receipts. Preserve the installed stock runtime as rollback. Before a shared
activation, reconcile active-task ownership and use a controlled switch; do not
interrupt unrelated tasks as part of verification. Report staged versus active
status explicitly. Disable the opt-in policy to restore ordinary retention;
switch back to the preserved binary if a runtime rollback is necessary.
