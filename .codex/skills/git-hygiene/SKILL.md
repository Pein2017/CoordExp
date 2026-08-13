---
name: git-hygiene
description: Choose and verify the exact CoordExp worktree, or stage, commit, split, synchronize, publish, and safely retire repository work while preserving unrelated edits and keeping credentials out of history and output.
---

# Git Hygiene

Keep commits logical and scoped. Dirty changes from parallel work are expected;
never stage, rewrite, or revert unrelated files.

## Bind The Checkout

1. Inspect the repository root, current branch and status, and `git worktree
   list` before editing or publishing.
2. Use the user-named checkout or the smallest current owner from repository
   guidance. Work in place only for narrow reversible work with no ownership
   conflict; otherwise isolate it in a worktree from the authorized base.
3. Confirm the absolute worktree and branch before changing files. Keep heavy
   data, models, caches, outputs, and runtime links outside Git. Add a shared
   link only when the target is validated and the path is absent; never replace
   an existing path or stage the link.
4. Keep symbol indexes and commands bound to that absolute checkout. Reactivate
   a code index after changing roots, and verify the task in the same checkout.

Stop on an ambiguous base, overlapping dirty ownership, or a requested move to
a different checkout until ownership is resolved.

## Preflight

1. Inspect status, current branch, worktrees, remotes, upstream, and
   ahead/behind state.
2. Resolve the user-authorized file set and intended publication boundary.
3. Confirm credential sources are ignored and untracked without printing their
   contents.
4. Stop when remote identity, branch ownership, secret status, or requested
   scope is ambiguous.

## Commit Loop

1. Inspect unstaged and staged diffs.
2. Group changes by one intent: feature, fix, test, documentation, config,
   formatting, or generated artifact.
3. Stage interactively or by explicit paths.
4. Reinspect the complete staged patch and run whitespace checks.
5. Run the smallest meaningful verification for that staged scope.
6. Commit with an imperative message, then confirm the remaining dirty set.

Completion requires a commit whose patch matches the stated intent, contains no
unrelated work or secrets, and has an explicit verification receipt.

## Synchronize And Push

Synchronization means integrate upstream changes before publication, not
push-only.

1. Confirm or establish the intended upstream.
2. Fetch, then inspect ahead/behind/diverged state.
3. If behind or diverged, choose rebase or merge according to branch policy and
   user authority; preserve local work before integration.
4. Push only when the branch is current with upstream or the user approved a
   different integration plan.

For HTTPS personal-access-token workflows, use a temporary credential helper
that reads an ignored local credential source. Disable terminal prompts and
never place credentials in remotes, command arguments, commit messages, PR
bodies, logs, or tracked files. Skip network operations that have no work.

## Worktree And Destructive Boundaries

- Before deleting branches or worktrees, compare metadata with real directories
  and inspect uncommitted work plus ahead/behind state.
- Retire a worktree only after its work is merged or explicitly discarded,
  durable artifacts are preserved, and its status is clean.
- Dry-run metadata pruning before applying it.
- A stale worktree marker or lock error is evidence to investigate, not proof
  that deletion is safe.
- History rewrite, force push, hard reset, destructive cleanup, and branch
  creation require explicit user scope.

Report the exact checkout and branch, changes and verification, commits created,
fetch/integration/push state, remaining dirt, and merge or retirement state.
