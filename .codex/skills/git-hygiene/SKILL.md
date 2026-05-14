---
name: git-hygiene
description: Use when staging, committing, pushing, splitting dirty CoordExp work, or using the repo HTTPS token workflow.
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

## Push

Default to the current branch. If upstream is missing, use `git push -u origin HEAD`.

CoordExp's HTTPS remote is normally:

```text
https://github.com/Pein2017/CoordExp.git
```

For one-shot PAT pushes, use a temporary credential helper that reads the ignored local token file:

```bash
GIT_TERMINAL_PROMPT=0 git \
  -c credential.helper= \
  -c "credential.helper=!f() { if [ \"$1\" = get ]; then echo username=x-access-token; printf 'password='; tr -d '\n' < github_personal_token.txt; echo; fi; }; f" \
  push origin HEAD
```

Never place the token in remotes, commit messages, PR bodies, shell output, or tracked files.

## Guardrails

- No `reset --hard`, history rewrite, `push --force`, or destructive cleanup without explicit user request.
- Do not create a branch unless asked or required by the user's workflow.
- If a pull/merge is needed, split local work into logical commits first when feasible.
- Final report should include commits created, checks run, push status, and remaining dirty files.
