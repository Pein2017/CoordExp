---
title: Codex and Pi research-probe forks
date: 2026-07-23
status: prepared
scope: harness comparison
---

# Codex and Pi Research-Probe Forks

Two parallel worktrees were prepared from `research-probes` HEAD
`41e530c016fa70f545fa230aa5209940ad153a04`:

- `/data/CoordExp/.worktrees/codex-research-probes` on
  `codex/research-probes-fork`;
- `/data/CoordExp/.worktrees/pi-research-probes` on
  `pi/research-probes-fork`.

Both received the same tracked dirty diff and untracked research and Pi-harness
sources. The named coordinate-prefix handoff was copied separately because the
`handoff/` tree is ignored. The two experiment-fork copies were then edited
identically to use the launch-time `pwd` and `$COORDEXP_ARTIFACT_ROOT`, while
the source checkout retained the historical handoff. Both experiment copies
have SHA-256
`129e753a7e446532a6b99a7b8007d6ec19527c88511f52f11c1b56949d2a4718`.

The Pi fork has an independent worktree-local Pi HOME, session directory,
authentication copy, pinned package tree, skill path, and Git safe-directory
entry. Its Serena smoke connected one server with twelve tools. The Codex fork
passes the Serena setup check and sees the configured Serena MCP server.

The first fork initialization generated `.serena/project.yml` with
`languages: []` in both experiment worktrees. Both local ignored configs were
corrected to `python`, and Serena's own system-prompt command now reports each
exact fork path with `Programming languages: python`.

Repository hook parity is now installed in the Pi fork. The Pi-owned extension
`.pi-worker/home/pi-worker/extensions/codex-rtk-hook-adapter.ts` adapts Pi's
mutable `bash` `tool_call` event to the existing Codex PreToolUse JSON protocol
and reuses `/data/CoordExp/.codex/hooks/rtk-pretooluse.py`. It registers no
model-visible tool and does not change the system prompt. Unit checks cover the
rewrite, machine-readable and disable bypasses, malformed output, and fail-open
behavior. A real Luna-medium smoke produced the compressed tool result
`Pytest: 7 passed`, and startup emits the UI-only status
`RTK hook: Codex parity`. This matches the repository RTK hook, not product
runtime policies such as AgentGuard.

Both worktrees currently link `outputs` to `/data/CoordExp/outputs`. AgentGuard
blocked replacing those newly created links with isolated targets, so concurrent
agents must use the prepared harness-specific artifact roots instead of the same
output path:

- `/data/CoordExp/outputs/research-probe-forks/codex-research-probes/`
- `/data/CoordExp/outputs/research-probe-forks/pi-research-probes/`

Before launch, set the artifact root in each shell. Both agents can now read
the handoff directly without a wrapper; the handoff itself forbids switching
worktrees and routes new outputs through the environment variable.
