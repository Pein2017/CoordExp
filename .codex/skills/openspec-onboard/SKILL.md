---
name: openspec-onboard
description: Use when the user explicitly asks to learn the OpenSpec workflow through a guided CoordExp example.
---

# OpenSpec Onboard

Requires `openspec` CLI.

Rare in this repo. OpenSpec is for stable compatibility contracts, while ordinary planning belongs in docs, progress notes, repo-local super-power plans, configs, tests, and artifacts.

## Flow

1. Preflight:
   ```bash
   openspec status --json
   ```
2. Explain the downgraded role of OpenSpec in CoordExp before choosing a task.
3. Pick a tiny compatibility-sensitive example, not a generic TODO.
4. Walk through: new change -> proposal -> delta spec -> design if needed -> tasks -> apply -> verify -> archive.
5. Pause at proposal, tasks, and archive so the user can redirect.

## Teaching Boundaries

- Use real repo files and exact paths.
- Keep narration short; teach ownership surfaces, not generic OpenSpec theory.
- If the user's task is ordinary implementation work, demonstrate why a repo-local plan is the better surface and stop.
