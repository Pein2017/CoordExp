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

- `main` is the stable CoordExp-Swift branch. `/data/CoordExp` is its
  operational checkout for official training and evaluation.
- `coordexp-swift` is the active development branch. It is checked out at
  `/data/CoordExp/.worktrees/CoordExp-swift` for feature and experiment work.
- `ms-swift` is the preserved pre-promotion mainline, retained as a history
  archive and compatibility/reference branch. It has no active worktree and
  is not a target for new implementation or launches.
- `origin/coordexp-swift` tracks the active development branch; promote
  validated work from it into `main` through an explicit merge.

When a task asks for the current repository, current implementation, or default
branch, resolve it against `main` at `/data/CoordExp`. For feature work, use
the Swift development checkout. Use `ms-swift` only for historical
reconstruction, old-run reproduction, or explicit archive maintenance.

The normal iteration is: develop and validate in `coordexp-swift`; merge the
accepted commits into `main`; launch official training/evaluation from the
root `main` checkout; then update the development branch from the promoted
`main` state before the next feature slice.

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

The stable implementation checkout should report branch `main` and path
`/data/CoordExp`; the active development checkout should report branch
`coordexp-swift` and path `/data/CoordExp/.worktrees/CoordExp-swift`. A Codex
task that still displays an older task branch is stale app-owned metadata;
starting from or sending a new message in the intended live checkout should
refresh that association. Historical session content remains unchanged by
design.

## Legacy and upstream wording

References to `ms-swift`, `/data/ms-swift`, `src.sft`, legacy config roots, and
old mainline launchers are valid only when explicitly labeled as upstream
dependency, compatibility, historical evidence, or archive material. They must
not be presented as the current CoordExp entrypoint.

The current Swift entrypoints are documented in
[`COORDEXP_SWIFT.md`](COORDEXP_SWIFT.md) and use `src/train.py`, `src/infer.py`,
`src/inference/`, and `src/eval/detection_consumer.py`.
