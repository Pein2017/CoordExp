---
name: git-hygiene
description: Perform explicitly authorized CoordExp Git integration, publication or worktree retirement with retained source/artifact ownership.
---

# CoordExp Git Operations

General Git safety belongs to the [user contract](../../AGENTS.md#authority-and-scope).
Use the current harness's exact commit/staging/verification contract; this skill
does not prescribe a competing staging loop or authorize commit, push or worktree
creation merely because cleanup was requested.

For checkout choice, integration or lifecycle, read the selected checkout's
`docs/BRANCH_AND_WORKTREE_POLICY.md` (the cross-checkout owner is
`/data/CoordExp/docs/BRANCH_AND_WORKTREE_POLICY.md`). Bind the physical tree,
current HEAD, dirty ownership and exact operation/destination. Do not assume
newer siblings form a release ladder. Keep changes in explicit logical batches,
with reviewed diffs and proportionate checks; preserve unrelated index/worktree
state and leave secrets, heavy assets and runtime links out of Git.

Before authorized publication, verify remote/upstream identity and divergence;
synchronize only through the authorized integration route. Keep credentials out
of argv, remotes and output. A failed check or ambiguous branch ownership is not
permission to rewrite history or silently include another file.

For retirement, also read `docs/OUTPUT_STORAGE_POLICY.md` and the applicable
retention/research convention. Merged commits or a clean status do not prove
research knowledge, untracked artifacts or active consumers are preserved. Bind
recoverable refs and verified recovery before deleting the exact authorized
worktree; inspect remaining branch refs separately. No broad pruning or automatic
worktree isolation.

Report exact checkout/branch, scoped changes, verification, any commits and
publication/integration/retirement actually performed, and intentionally retained
dirty work. An invocation of this skill is not itself a request to publish.
