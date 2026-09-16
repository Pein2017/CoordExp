## Why

Desktop-loaded main tasks and completed native subagents retain per-task Serena
MCP/LSP processes while idle. An installed-Core probe proved that externally
terminating an idle server breaks the next call with `Transport closed`; global
MCP reload recovers it but affects unrelated tasks. Resource reclamation must
therefore live with Core's connection ownership and demand admission.

## What Changes

- Add an opt-in, task-scoped idle lifecycle for explicitly eligible local stdio
  MCP servers, initially the configured Serena server and wake-me-up frontend.
- Use a longer main-task idle grace (initial operator value 900 seconds) and a
  shorter completed-subagent grace (120 seconds), with positive configurable
  intervals and unchanged behavior when disabled.
- Atomically admit activity or suspend eligible MCP connections; never suspend
  an active turn, in-flight call, pending interaction, or unresolved uncertain
  operation. Reconstruct once before subsequent demand without replaying a
  potentially executed tool operation.
- Keep main task history/identity, background exec/code-mode work, noneligible
  MCP connections, and the independent wake daemon intact. Preserve or safely
  reconstruct the server's project/session context; if that cannot be proven,
  retain the server rather than guess.
- Keep current cold start/resume behavior and native subagent-history policy.
  Do not add a process-scanning daemon, generic supervisor, shared Serena server,
  Desktop client fork, or new subagent resurrection contract.

## Capabilities

### New Capabilities

- `codex-mcp-idle-lifecycle`: Owner-controlled MCP suspension and safe demand
  reconstruction for idle Desktop tasks and completed native subagents.

### Modified Capabilities

None. Existing CoordExp training/inference and wake delivery contracts are
unchanged.

## Impact

- Planning authority: `/data/CoordExp/openspec/changes/codex-mcp-idle-lifecycle`.
- Implementation: isolated Codex worktree
  `/data/CoordExp/external/harness/codex-mcp-idle-lifecycle-0.153.4`, branch
  `codex/mcp-idle-lifecycle`, base `3d2ee51ca2d5db578f328aa75e20aa22c0197c9a`
  (release source version 0.153.4). The existing main/native-observe checkouts
  contain unrelated changes and are preserved.
- Expected owners: `codex-mcp` connection/runtime/binding machinery and the
  narrow Core session/config/admission integration; generated configuration
  schema, focused tests, and operator documentation as required.
- Evidence already retained in
  `/data/CoordExp/.codex/scratch/serena-reclaim-20260909T072138Z/`: real stdio
  close/reopen, installed-Core failure/reload, persisted main read/resume, and
  isolated wake idle-retirement receipts. These are baseline evidence, not
  acceptance of the proposed automatic policy.
- Implementation and isolated tests are authorized by the user. Stage a
  separately verifiable runtime candidate; activation must preserve unrelated
  active tasks and report whether the shared runtime actually changed.
