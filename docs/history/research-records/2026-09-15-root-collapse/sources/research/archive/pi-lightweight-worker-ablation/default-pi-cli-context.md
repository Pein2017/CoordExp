---
title: Default Pi CLI Context Parity Setup
description: Worktree-local default Pi configuration for CoordExp skills, AGENTS.md, and the same Serena MCP server used by Codex CLI.
type: investigation
role: setup-receipt
authority: non_normative_research
architecture_promotion_status: pilot_only
topic: pi-lightweight-worker-ablation
status: complete
evidence_status: verified_bounded
updated: 2026-07-23
---

# Default Pi CLI Context Parity Setup

## Scope

This setup gives the default interactive Pi CLI the same worktree-level
starting guidance surfaces requested for a direct Pi-versus-Codex harness
comparison:

- the root `AGENTS.md`, discovered by Pi's native context-file loading;
- the worktree `.codex/skills/` directory;
- the same Serena Model Context Protocol (MCP) server command, arguments,
  environment, and project-from-current-working-directory behavior configured
  for Codex CLI.

It does not alter the isolated stateful supervisor. That worker continues to
disable ambient extensions, skills, prompt templates, and context files so its
earlier pilot contract remains reproducible.

## Local Runtime

From the worktree root:

```bash
source .pi-worker/home/.bashrc
pi --provider openai-codex --model gpt-5.6-sol --thinking xhigh
```

The shell initialization makes `.pi-worker/home/` the effective `HOME`, puts
`.pi-worker/home/pi-worker/` first on `PATH`, and keeps Pi agent and session
state under `.pi-worker/`. It also exports the required HTTP, HTTPS, and ALL
proxy variables through `127.0.0.1:9090`. A worktree-local `.gitconfig` marks
only this exact checkout as a safe directory; it does not inherit the global
Git identity or credential configuration from `/root`.

## Loaded Surfaces

Pi `0.81.1` discovered 26 skills from the worktree skill directory, including
`coordexp-vllm-mechanistic-loop`, `model-diagnosis`, and
`detection-gt-vs-pred-visualization`. The root agent guide was available as
initial context without copying or translating it into a Pi-specific prompt.

Serena is exposed through pinned `pi-mcp-adapter@2.11.0`, with npm integrity:

`sha512-4Y/eLbhbxnRih519dJUxMyQ5QASvPcdWyBlS8+dDXteAzaMuLnd4nMTWgoZw3JRIW+0r93KAQcz1Rbli4xCwEQ==`

The adapter uses the exact Codex Serena launch contract: context `coordexp`,
mode `no-memories`, project discovery from the current working directory, and
the shared worktree-aware Serena runtime. Direct-tool mode exposes the twelve
Serena tools and hides the adapter proxy tool so the comparison does not gain
an artificial token-saving indirection.

## Verification Receipt

A real Luna-medium, no-session CLI smoke verified all three surfaces without
filesystem discovery tools:

- it answered that durable repository artifacts use English from `AGENTS.md`;
- it recognized the three named worktree skills;
- it called `serena_initial_instructions` once and Serena reported the active
  project as `research-probes` at this exact worktree path.

The first model request used 10,887 input tokens, 55 output tokens, and 36
reasoning tokens. The post-tool request used 344 input tokens, 10,752 cached
input tokens, 95 output tokens, and 43 reasoning tokens. Combined reported
cost was USD 0.0132062. The 10,887-token first-request input is the initial
background baseline for the later matched Codex CLI comparison.

The installed dependency graph reports moderate npm advisories and no high or
critical advisories. The concrete transitive Hono advisory concerns Windows
encoded-backslash static-file traversal and is not on this Linux Serena stdio
path, but the pinned adapter currently has no dependency resolution that
removes all moderate findings. Keep this as a research-only local setup.

## Codex RTK Hook Parity

The Pi comparison fork also loads a Pi-owned event adapter from
`.pi-worker/home/pi-worker/extensions/codex-rtk-hook-adapter.ts`. It listens
only to Pi's `tool_call` event for the built-in `bash` tool, sends the command
through the existing `/data/CoordExp/.codex/hooks/rtk-pretooluse.py`, and
applies the returned Codex `updatedInput.command`. It does not copy the RTK
rewrite policy, register a model-visible tool, or modify the system prompt.

The adapter preserves the Codex hook's fail-open contract and both forms of
`RTK_HOOK_DISABLE=1`. Unit coverage verifies ordinary rewrite,
machine-readable bypass, command-level and process-level disable, empty or
malformed output, subprocess failure, and valid updated input. A real
Luna-medium Pi smoke issued raw `pytest -q ...` arguments and received the RTK
compressed tool result `Pytest: 7 passed`, establishing that the rewrite runs
inside Pi's actual tool-call preflight. The Pi footer displays
`RTK hook: Codex parity` when the extension is loaded.

This matches the repository RTK PreToolUse hook only. Product-level Codex
runtime policies such as AgentGuard are outside `.codex/hooks.json` and are not
claimed as matched.
