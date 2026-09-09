# Operator procedure

Status: r2 is staged and verified; shared runtime activation is pending.
Package:
`/data/CoordExp/.codex/packages/standalone/releases/0.153.4-mcp-idle-20260909-r2-x86_64-unknown-linux-gnu`.
Its `mcp-idle-manifest.json` binds the binary and source commit, and
`validation/acceptance.json` binds all seven isolated entry scenarios.
This document does not authorize or report a shared runtime restart.

## Policy

After installing a verified candidate, add the following keys to the existing
`[mcp_servers.serena]` table in `/data/CoordExp/.codex/config.toml`. Keep its
command, arguments, environment, and timeout settings intact; do not create a
second TOML table with the same name.

```toml
idle_timeout_sec = 900
idle_timeout_completed_subagent_sec = 120
idle_recovery = "serena_startup_project"
```

The second interval applies to completed native subagents. It does not expire a
task or delete its history. Omitting all three keys preserves ordinary retention.
Use `stateless` only for a local stdio server whose required session context is
fully reconstructed by startup. Do not apply either declaration to an arbitrary
stateful server to force reclamation.

An active turn, in-flight operation, or unresolved operation outcome prevents
suspension. Serena project changes that cannot be reconstructed also prevent it.
Such retention is intentional. The initial implementation does not replay tool
calls to restore arbitrary session state.

Passive status uses metadata retained by the matching task for opted-in local
servers. It does not launch a temporary instance to enumerate resources. An
empty resource inventory in this passive response means no retained inventory;
it is not evidence that a fresh server reported no resources.

Plugin-contributed stdio servers can declare the same fields in their own
`.mcp.json` server object. The existing user plugin override table cannot attach
these fields. In particular, the installed wake plugin stays resident unless
its manifest explicitly opts in; this change does not silently rewrite plugin
caches or add a second wake server registration. A later plugin package update
can opt its reconstructible frontend in with `idle_recovery: "stateless"`.

## Candidate acceptance and shared activation

1. Check the versioned package manifest, source identity, binary hashes, and
   exact-binary acceptance receipts linked from `verification.md`.
2. Record the current runtime target and preserve the original configuration.
   Reconcile the actual shared app-server process identity and active tasks
   immediately before a switch. An older PID or launch receipt is insufficient.
3. Apply the policy and perform a controlled runtime switch only within the
   authorized maintenance boundary. Do not restart another owner's active work.
   Keep the original release and configuration available for rollback.
4. Verify the running executable resolves to the candidate and the intended
   configuration loaded. Exercise one opted-in server in a disposable task,
   observe its idle suspension, and use the same task again. Record task ID,
   process replacement, successful call, and effective project.

Until those live checks pass, report the package as staged and the shared runtime
as unactivated. Isolated acceptance alone does not establish Desktop activation.

## Rollback

Remove the three idle-policy keys to restore ordinary retention after the
configuration takes effect. This does not require deleting tasks or MCP state.
For a binary rollback, restore the preserved runtime target and configuration
through the same controlled maintenance boundary, then verify the running
executable and a normal tool call. Do not terminate MCP descendants by PID as a
substitute for owner-controlled suspension.
