---
name: openspec-apply-change
description: Use when implementing tasks from an existing OpenSpec change that is explicitly in scope.
---

# OpenSpec Apply Change

Requires `openspec` CLI.

Implement the checked-in OpenSpec task list, keeping code/config/docs changes narrowly aligned with the contract artifacts.

## Flow

1. Select the change; ask if ambiguous.
2. Run:
   ```bash
   openspec status --change "<name>" --json
   openspec instructions apply --change "<name>" --json
   ```
3. Read every `contextFiles` path before editing.
4. Implement pending tasks in order unless a task is blocked by a discovered design issue.
5. After each completed task, update the checkbox in the task file.
6. Run the smallest meaningful validation named by the artifacts or repo docs.

## Stop Conditions

- The task contradicts current repo docs or executable behavior.
- The implementation needs a contract change not represented in the artifacts.
- Validation exposes a root-cause issue outside the task scope.

Report completed tasks, remaining tasks, validation, and blockers. Suggest archive only when all tasks and verification are complete.
