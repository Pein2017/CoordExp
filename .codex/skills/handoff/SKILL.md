---
name: handoff
description: Use when the user asks for a handoff, continuation prompt, compact session summary, another-agent brief, or cross-machine execution note.
---

# Handoff

Write a concise continuation document for a fresh agent or another machine.

## Output Target

- If the user gives a path, write there.
- Otherwise save to a temporary Markdown file from `mktemp -t handoff-XXXXXX.md`.
- Read the target path before writing if it already exists.

## CoordExp Content

Include only portable, actionable state:

- repo root, branch if relevant, and dirty-file scope;
- exact artifact/config/checkpoint paths that matter;
- commands already run and their outcomes;
- unresolved decisions, blockers, and recommended next action;
- which skills or repo docs the next agent should use;
- verification that still needs to run.

Do not duplicate large artifacts, PRDs, plans, metrics, or docs. Link exact paths instead. Keep Notion and Linear references short and navigational; repo files and artifacts remain executable truth.
