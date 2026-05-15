---
name: openspec-bulk-archive-change
description: Use when the user explicitly wants to archive multiple OpenSpec changes in one operation.
---

# OpenSpec Bulk Archive

Requires `openspec` CLI.

Batch archive is high-risk because spec deltas can conflict. Keep it interactive and evidence-backed.

## Flow

1. Run `openspec list --json` and ask the user which active changes to include.
2. For each selected change, gather status, task completion, and delta-spec capabilities.
3. Detect conflicts where multiple changes modify the same capability spec.
4. For each conflict, inspect implementation evidence and delta intent before deciding sync order.
5. Present a compact table: artifacts, tasks, delta specs, conflicts, proposed sync/archive action.
6. Ask for confirmation, then archive each selected change to the dated archive path.

## Guardrails

- Never auto-select all active changes.
- Do not sync conflicting specs without a stated resolution.
- Incomplete changes may be archived only after an explicit warning and confirmation.
- Stop on the first move or sync failure and report what already changed.
