---
name: handoff
description: Use when the user asks for a handoff, continuation prompt, compact session summary, another-agent brief, or cross-machine execution note.
---

# Handoff

Write only the state a fresh agent or another machine needs to continue correctly.

## Output Target

- If the user gives a path, write there.
- For durable or cross-machine output, use the user-named path or an explicitly agreed stable project/shared path. Never use `mktemp` or `/tmp`.
- Otherwise return the handoff inline in chat.
- Read the target path before writing if it already exists.

## CoordExp Content

Include only relevant items:

- repo root, branch if relevant, and dirty-file scope;
- exact artifact/config/checkpoint paths that matter;
- commands already run and their outcomes;
- unresolved decisions, blockers, and recommended next action;
- which skills or repo docs the next agent should use;
- verification that still needs to run.

Link exact paths instead of copying large artifacts, plans, metrics, or docs. Live repo files and artifacts remain the source of truth.

For domain-specific continuation fields, follow the owning skill: use
`coordexp-public-data-provenance` for manifests/regeneration and
`baidu-netdisk-transfer` for transfer state, tmux/log handles, mappings, and
post-transfer checks.
