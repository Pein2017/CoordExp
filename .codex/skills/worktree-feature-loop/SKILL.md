---
name: worktree-feature-loop
description: Choose, enter, verify, or retire the exact CoordExp Git worktree that owns a feature, fix, or research task.
---

# Worktree Feature Loop

Bind one task to one **exact checkout** from preflight through handoff. Use
`git-hygiene` for staging, commits, synchronization, and publication.

## Work

1. Inspect repository root, branch, status, and `git worktree list`.
2. Work in place for a narrow reversible change with no ownership conflict;
   otherwise isolate the task in a worktree from the user-named or canonical
   base.
3. Confirm the absolute worktree and branch before editing. Preserve unrelated
   dirt and keep heavy data, models, caches, and outputs outside Git.
4. If shared roots are required, add only missing, validated links; never replace
   an existing path or stage runtime links.
5. Activate Serena for the exact absolute worktree when symbol semantics matter.
   Reactivate after moving roots; use raw files to resolve any index conflict.
6. Verify the task in the same checkout and report path, branch, changes,
   evidence, remaining dirt, and merge/cleanup state.

The loop is complete when one checkout owns the work and its verification.
Remove a worktree only after work is merged or explicitly discarded, durable
artifacts are preserved, and status is clean. Stop on ambiguous base,
overlapping dirty ownership, destructive cleanup, publication, secrets, or a
new expensive run.
