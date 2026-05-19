---
name: git-sync
description: Use when staging, committing, syncing with the remote branch, splitting dirty CoordExp work, or using the repo HTTPS token workflow.
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

## Commit Loop

1. Inspect `rtk git diff --stat` and `rtk git diff --name-only`.
2. Group changes by intent: feature, fix, tests, docs, config, formatting, or generated artifacts.
3. Stage narrowly with `git add -p` or explicit paths.
4. Verify staged diff with `git diff --cached`.
5. Run the smallest meaningful check for the staged scope.
6. Commit with an imperative message; use minimal messages for mechanical config/arg/default changes.
7. Repeat until only intentional leftovers remain.

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
- If a pull/merge is needed, split local work into logical commits first when feasible.
- Final report should include commits created, checks run, sync status (fetch/pull/push), and remaining dirty files.
