# Official Serena Per-Worktree Runtime Design

> **RETIRED (2026-08-13):** This shared Streamable HTTP/stdio-bridge design was
> removed after live Codex sessions experienced transport closure during
> lifecycle handoff. The active architecture is one unmodified official Serena
> stdio process per Agent session, started with `--project-from-cwd`. This file
> is retained only as historical design evidence and is not implementation
> authority.

## Goal

Use unmodified official Serena while allowing Codex and Claude Code agents to
work concurrently across many Git worktrees. Agents in the same worktree share
one Serena language-server process; agents in different worktrees remain
isolated. Session shutdown and app-server restart must not accumulate orphaned
Serena or language-server processes.

## Non-goals

- No fork or patch of Serena, SolidLSP, Pyright, or other language servers.
- No replacement semantic implementation or Serena Light compatibility layer.
- No dashboard, GUI log window, memories, onboarding, or broad tool surface.
- No cross-host service, external listener, authentication system, or proxy use
  for localhost traffic.
- No transparent project switching inside a shared server.

## Architecture

The MCP client starts a small stdio wrapper under a Linux parent-death signal.
The wrapper resolves the nearest Git worktree root from its startup working
directory and maps the canonical absolute root to a service-owned runtime slot.

Each slot owns exactly one official Serena process started as a localhost-only
Streamable HTTP server with the existing `coordexp-minimal` context. A pinned
stdio-to-Streamable-HTTP bridge connects each MCP client to that server. The
wrapper does not inspect or transform MCP payloads.

```text
Codex or Claude Code
  -> stdio lifecycle wrapper
  -> pinned MCP transport bridge
  -> localhost Streamable HTTP
  -> official Serena
  -> project language servers
```

The official Serena executable remains `/root/.local/bin/serena`, installed and
updated through `uv tool`. Updating Serena does not rewrite the wrapper.

## Runtime ownership

Runtime state lives below `/data/CoordExp/.codex/serena/shared/` and is never
committed. A slot key is derived from the canonical Git worktree path, while the
slot metadata records the complete path to make hash or port collisions
detectable.

Each slot contains a startup lock, backend PID identity, selected localhost
port, client leases, and bounded logs. Backend identity consists of PID, Linux
process start time, expected executable, and expected project path; a PID alone
never authorizes termination.

Startup holds the slot lock while it removes stale leases, validates an existing
backend, allocates a loopback port, and waits for a newly started server to
become reachable. Concurrent clients therefore converge on one backend.

Each wrapper creates a lease containing its PID and process start time. Normal
exit or SIGTERM removes the lease. When the final lease disappears, a detached
reaper waits for a 60-second warm grace, reacquires the slot lock, removes stale
leases again, and terminates only the still-matching backend. A later client
cancels retirement simply by creating a valid lease before the reaper checks.

At every startup, stale leases and mismatched backend metadata are reconciled.
This provides eventual cleanup after SIGKILL without requiring a persistent
supervisor.

## Worktree and project rules

The startup worktree is the server's project identity. Multiple clients may
confirm or reactivate that same root. They must not use `activate_project` to
switch a shared server to a different project, because that would change state
for every client sharing the slot.

To work in another worktree, start or fork the agent task from that worktree.
Its wrapper resolves a different slot and launches or joins the corresponding
Serena instance. This preserves independent language-server state across
parallel worktrees while reusing state within one worktree.

## Hooks and client configuration

`serena-hooks remind` remains enabled and may block excessive shell or text
search use. It is independent of process lifecycle.

The MCP command for Codex and Claude Code becomes:

```text
/usr/bin/setpriv --pdeathsig TERM <shared-wrapper>
```

The wrapper receives the current working directory from the client process. It
preserves the existing Serena environment for the `ms` Conda interpreter and
removes proxy variables from localhost backend and bridge children.

`serena-hooks cleanup` may still remove Serena Hook session persistence, but it
is not treated as a backend process owner. Wrapper leases remain the lifecycle
authority because they also work when a client has no SessionEnd hook.

## Dependency policy

The MCP bridge is installed once into a service-owned environment and pinned to
an exact source revision plus an MCP SDK version known to be compatible. Per-
session `uvx latest` resolution is forbidden. Serena itself continues to follow
the user-controlled official `uv tool` installation and upgrade lifecycle.

## Failure behavior

- If the startup root is not inside a Git worktree, startup fails with an
  actionable error rather than sharing an ambiguous global instance.
- If backend metadata, port ownership, or root identity is inconsistent, the
  wrapper refuses to attach and reports the slot path.
- If Serena exits before readiness, the wrapper reports the bounded backend log
  path and removes only state created by that startup attempt.
- If the bridge exits, the wrapper releases its lease and returns the bridge's
  exit status.
- Cleanup never kills by name, wildcard, port alone, or unverified PID.
- No component changes global proxy or `NO_PROXY` settings.

## Acceptance

1. Two concurrent MCP clients from the same worktree both complete initialize,
   `initial_instructions`, project activation, symbol overview, and a Python
   symbol query while exactly one Serena and one Python language server own that
   worktree.
2. Concurrent clients in `CoordExp-swift` and `research-probes` use distinct
   slots and both complete semantic queries.
3. Restarting the Codex app-server with multiple restored tasks does not create
   duplicate same-worktree backends or failed Pyright initializations.
4. A normally closed client releases its lease; the last client causes backend
   retirement after the grace interval.
5. A killed wrapper leaves no durable false lease; the next startup reconciles
   it without touching unrelated Serena or language-server processes.
6. Poisoned ambient proxy variables do not affect localhost MCP traffic.
7. Existing fixed Serena tool allowlist and blocking `serena-hooks remind`
   behavior remain unchanged.

## Rollback

Client MCP commands can be restored to direct official Serena stdio without
altering any project configuration. Runtime slots are disposable only after
their recorded backend identities have been revalidated and stopped. The
official Serena installation and project `.serena/project.yml` files remain
valid independently of the wrapper.
