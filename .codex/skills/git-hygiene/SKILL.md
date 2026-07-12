---
name: git-hygiene
description: Use when CoordExp Git work needs worktree isolation, staging, committing, dirty-tree splitting, branch cleanup, remote sync or push, or the repo HTTPS token workflow.
---

# Git Hygiene

Keep commits logical and scoped. Dirty changes from parallel work are expected; never stage or revert unrelated files.

## Preflight

Run compact discovery unless exact output matters:

```bash
rtk git status --short --branch
git branch --show-current
git remote -v
git check-ignore -v github_personal_token.txt
git ls-files --error-unmatch github_personal_token.txt 2>/dev/null && echo TRACKED || true
```

Stop if secrets are tracked, remote identity is surprising, or the requested scope is ambiguous.

## Worktree isolation

Use a linked worktree when parallel work, unrelated dirt, long-running outputs, or a multi-file/multi-commit change makes isolation valuable. Use the current checkout for a narrow change when its owned files are clear.

The user-named checkout always wins. Verify its root, branch, status, and linked-worktree record before acting; do not infer current behavior from another checkout. New branch names default to `codex/<task-slug>`.

Read [references/worktrees.md](references/worktrees.md) before creating, entering, repairing, or removing a worktree. It contains the safe lifecycle, shared-root setup, and cleanup gates.

## Commit Loop

1. Inspect `rtk git diff --stat` and `rtk git diff --name-only`.
2. Group changes by intent: feature, fix, tests, docs, config, formatting, or generated artifacts.
3. Stage narrowly with `git add -p` or explicit paths.
4. Verify staged diff with `git diff --cached --stat`, `git diff --cached`, and `git diff --cached --check`.
5. Run the smallest meaningful check for the staged scope.
6. Commit with an imperative message; use minimal messages for mechanical config/arg/default changes.
7. Repeat until only intentional leftovers remain.

## Stale worktree and cleanup checks

Before branch deletion, worktree cleanup, grouped commits, or sync:

- run `git worktree list --porcelain`;
- compare branch refs against actual directories;
- inspect ahead/behind before deleting local refs;
- use `git worktree prune --dry-run` before cleanup when metadata looks stale.

Stale `+` markers or branch lock errors are usually worktree-metadata problems, not proof that the branch is still active. Remove metadata only with explicit cleanup scope and after confirming no uncommitted work is being discarded.

## Sync with remote

Default to the current branch. Sync means integrate remote commits, then publish local commits — not push-only.

1. Confirm upstream: `git rev-parse --abbrev-ref --symbolic-full-name @{u}` (set with `git push -u origin HEAD` when missing).
2. Fetch and inspect: `git fetch origin` then `rtk git status --short --branch` (ahead/behind/diverged).
3. If behind or diverged, pull first (prefer `git pull --rebase` on feature branches when history is linear).
4. Push only after the branch is up to date with its upstream (or you have an explicit user-approved merge/rebase plan).

CoordExp's HTTPS remote is normally:

```text
https://github.com/Pein2017/CoordExp.git
```

For one-shot PAT fetch/pull/push, use a temporary credential helper that reads the ignored local token file:

```bash
_branch="$(git branch --show-current)"
_git_https() {
  GIT_TERMINAL_PROMPT=0 git \
    -c credential.helper= \
    -c "credential.helper=!f() { if [ \"$1\" = get ]; then echo username=x-access-token; printf 'password='; tr -d '\n' < github_personal_token.txt; echo; fi; }; f" \
    "$@"
}
_git_https fetch origin
_git_https pull --rebase origin "$_branch"
_git_https push origin HEAD
```

Skip `pull` when already up to date; skip `push` when there is nothing to publish. Never place the token in remotes, commit messages, PR bodies, shell output, or tracked files.

## Guardrails

- No `reset --hard`, history rewrite, `push --force`, or destructive cleanup without explicit user request.
- Do not create a branch unless asked or required by the user's workflow.
- Remove a worktree only after its work is merged or explicitly discarded, its status is clean, and its durable research evidence or artifacts have been promoted.
- If a pull/merge is needed, split local work into logical commits first when feasible.
- Final report should include commits created, checks run, sync status (fetch/pull/push), and remaining dirty files.

## CoordExp Notes

- `.codex/memories/` is local-only; verify with `git ls-files .codex/memories` before any memory-related commit.
- Before Baidu `outputs/` sync or artifact publication, align the intended branch with remote unless the user explicitly wants a local-only transfer.
- `output_remote/` is transitional. Use `scripts/absorb_output_remote_into_outputs.py --apply` for no-overwrite absorption while active writers still use the old root.
- For old benchmark CSV conflicts, prefer canonical `outputs/...` paths over stale `output_remote/...` strings unless the user is preserving historical evidence verbatim.
