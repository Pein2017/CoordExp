---
status: active
scope: detection-scene-clean-break
kind: archive-checkpoint
date: 2026-05-31
---

# DetectionScene Clean-Break Archive Checkpoint

This note records the pre-cleanup archive checkpoint for the approved
`DetectionScene` clean-break implementation.

## Checkpoint

- Branch: `codex/detection-scene-clean-break`
- Commit: `5ac77b5a9131d27fd5681b8e4c6a8d233ec72844`
- Commit subject: `docs: approve DetectionScene implementation`
- Worktree used for implementation: `/data/CoordExp/.worktrees/detection-scene-clean-break`

## Meaning

This checkpoint is the durable rollback/reference handle for the repository
state after the architecture proposal was approved and before production
runtime replacement, deletion, or rename cleanup began.

Historical compatibility surfaces may be renamed, quarantined, or deleted only
after the replacement-before-deletion gates in
`openspec/changes/detection-scene-clean-break/tasks.md` are satisfied.

Stable current-behavior docs and specs remain current authority until the
corresponding implementation slice makes the new `DetectionScene` behavior
current.
