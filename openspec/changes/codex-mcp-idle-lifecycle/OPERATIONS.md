# Operator procedure

Status: r2 is active and live-verified on 2026-09-09, following explicit user authorization.
Package:
`/data/CoordExp/.codex/packages/standalone/releases/0.153.4-mcp-idle-20260909-r2-x86_64-unknown-linux-gnu`.
Its `mcp-idle-manifest.json` binds the binary and source commit, and
`validation/acceptance.json` binds all seven isolated entry scenarios.
The activation receipt is `operations/activation-receipt.json` in that package;
the actual shared-Core Serena test is `operations/live-20260909T112842Z/receipt.json`.

## Current mechanism

Core owns the timer and MCP connection lifecycle. Main tasks use 900 seconds of
idle grace; completed native subagents use 120 seconds. Active turns, operations,
pending interactions, and uncertain completion prevent suspension. Only the
opted-in Serena stdio instance is closed; task history and background execution
are retained. The next real MCP demand reconstructs one instance, rechecks its
handshake/catalog, and preserves the proven startup project. Passive status does
not recreate it. A different or ambiguous project/context mutation may retain
the instance instead. This is a Core patch, not an external PID-cleanup daemon
or a modified Serena distribution.

## Upgrading Codex or Serena

This patch is currently based on Codex 0.153.4. Source commit
`1208c8f07f480e3e778118a80fad99e5a62e742a` is maintained on
`codex/mcp-idle-lifecycle` in
`/data/CoordExp/external/harness/codex-mcp-idle-lifecycle-0.153.4`.
The version string alone does not distinguish this build from stock; check the
resolved executable, manifest, and binary hash.

For a future CLI/Core upgrade:

1. Preserve the currently working package and configuration. Prepare a new
   isolated checkout of the intended release and port this commit there.
2. Resolve changes at configuration parsing, actual-turn admission, MCP client
   lifecycle, and passive-status collection. A clean cherry-pick is not proof
   of behavioral compatibility.
3. Run the affected Rust checks and all three packaged harnesses in `validation/`
   (main including real Serena, native child, wake). Bind results to the new
   binary and its matching code-mode host.
4. Stage a new versioned package, then perform a controlled switch and live
   Serena smoke. Do not overwrite the working release in place.

Installing an unpatched official CLI does not carry this source patch forward;
the three configuration keys alone cannot provide reclamation. The relevant
upgrade is the executable that owns MCP, not merely the Desktop UI version.
Porting may be small or require adaptation; do not assume either in advance.

A Serena upgrade usually does not require rebuilding Core, but its startup CLI,
project behavior, handshake and tool catalog must still satisfy the reconstruction
contract. Re-run the real Serena scenario; contract changes may require a Core
adjustment. Changing only the grace intervals requires configuration reload,
not recompilation. Removing all three policy keys and reloading configuration
disables automatic reclamation.

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
