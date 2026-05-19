---
name: openspec-archive-change
description: Use when finalizing and archiving a completed OpenSpec change.
---

# OpenSpec Archive Change

Requires `openspec` CLI.

Archive only after implementation and spec-sync decisions are explicit.

## Flow

1. Select the change; ask if no name is provided.
2. Run `openspec status --change "<name>" --json`.
3. Check task checkboxes and artifact completion; warn before archiving incomplete work.
4. If delta specs exist, assess whether main specs need sync. Use `openspec-sync-specs` when syncing is chosen.
5. Move the change to:
   ```text
   openspec/changes/archive/YYYY-MM-DD-<name>/
   ```
6. Report archive path, sync decision, incomplete-work warnings, and remaining dirty files if relevant.

## Guardrails

- Do not guess the target change.
- Do not overwrite an existing archive directory.
- Do not use archive as a substitute for verification; call out skipped validation clearly.
