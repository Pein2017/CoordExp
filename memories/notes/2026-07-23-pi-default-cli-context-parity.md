---
title: Pi default CLI context parity
date: 2026-07-23
status: verified
scope: pi-lightweight-worker-ablation
---

# Pi Default CLI Context Parity

The default worktree-local `pi` command now starts with the root `AGENTS.md`,
the 26 skills under `.codex/skills/`, and the same Serena MCP launch contract
used by Codex CLI. `pi-mcp-adapter@2.11.0` exposes twelve direct Serena tools.

All Pi runtime state remains below `.pi-worker/`, and network-aware entry points
export the proxy at `127.0.0.1:9090`. A real Luna-medium smoke verified the
guide, skill discovery, and Serena activation for the exact `research-probes`
worktree. Its first request reported 10,887 input tokens; use that as the Pi
initial-background baseline for the matched Codex CLI comparison.

This changes only the default interactive CLI. The stateful RPC supervisor
continues to disable ambient extensions, skills, prompt templates, and context
files under its earlier isolation contract.

Full receipt:
`research/investigations/pi-lightweight-worker-ablation/default-pi-cli-context.md`
