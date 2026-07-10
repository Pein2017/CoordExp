---
doc_id: docs.branch-and-worktree-policy
layer: docs
doc_type: workflow
status: canonical
domain: repo
summary: Canonical branch, archive branch, worktree, and Codex-session routing for CoordExp.
tags: [git, branches, worktrees, codex, coordexp-swift]
updated: 2026-07-10
---

# Branch And Worktree Policy

CoordExp-Swift is now the canonical implementation on repository `main`.

## Canonical routing

- `main` is the active CoordExp-Swift branch.
- `/data/CoordExp/.worktrees/CoordExp-swift` is the active checkout attached
  to `main`.
- `ms-swift` is the preserved pre-promotion mainline, retained as a history
  archive and compatibility/reference branch. It is not the default target
  for new implementation work.
- `origin/coordexp-swift` is a transition and rollback alias for the current
  Swift promotion commit. New work should target `main`.

When a task asks for the current repository, current implementation, or default
branch, resolve it against `main` and the Swift checkout. Use `ms-swift` only
for historical reconstruction, old-run reproduction, or explicit archive
maintenance.

## Codex sessions and task worktrees

Codex session transcripts, archived sessions, attachments, and app-managed
session indexes are historical or runtime-owned state. They may contain old
task-branch or worktree descriptions from when Swift was a candidate worktree.
Do not rewrite those transcripts to make past conversations appear current.

For a new or continued task, verify the live checkout with:

```bash
git branch --show-current
git worktree list --porcelain
git rev-parse --show-toplevel
```

The active implementation checkout should report branch `main` and path
`/data/CoordExp/.worktrees/CoordExp-swift`. A Codex task that still displays an
older task branch is stale app-owned metadata; starting from or sending a new
message in the live `main` checkout should refresh that association. Historical
session content remains unchanged by design.

## Legacy and upstream wording

References to `ms-swift`, `/data/ms-swift`, `src.sft`, legacy config roots, and
old mainline launchers are valid only when explicitly labeled as upstream
dependency, compatibility, historical evidence, or archive material. They must
not be presented as the current CoordExp entrypoint.

The current Swift entrypoints are documented in
[`COORDEXP_SWIFT.md`](COORDEXP_SWIFT.md) and use `src/train.py`, `src/infer.py`,
`src/inference/`, and `src/eval/detection_consumer.py`.
