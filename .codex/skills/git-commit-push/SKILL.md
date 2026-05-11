---
name: git-hygiene
description: Use when committing, staging, pushing, or cleaning up Git work in CoordExp, especially when HTTPS push with a GitHub personal access token is needed.
---

# Git Hygiene: Commit + Push

## Identity

This directory, `.codex/skills/git-commit-push/`, is the repo-local skill
package for `git-hygiene`. Treat `SKILL.md` here as the single canonical
implementation; do not maintain a duplicate git-push skill elsewhere.

## When To Use

Use this skill whenever:
- A “non-trivial” change is complete (feature slice, refactor, bugfix, test suite update).
- You notice uncommitted changes are piling up.
- You’re about to context-switch, pull, rebase, or run risky experiments.
- You need to push CoordExp through HTTPS using `github_personal_token.txt`.

## Goals

- Keep the working tree small and readable.
- Produce a sequence of *logical commits* (each passes basic checks).
- Avoid “mega-commits”: commit in *logical/similar groups*, not “everything at once”.
- Push to a remote branch early (so work is backed up and reviewable).

## Hard Safety Rules

- **Never commit secrets** (tokens, private keys, passwords, `.env`, credentials). If suspected, stop and ask to rotate/remove.
- **No destructive commands** unless explicitly requested: no `reset --hard`, no `rebase`, no `push --force`, no history rewriting.
- Default: commit and push on the **current branch**. Do not auto-create a new branch unless the user explicitly asks.
- CoordExp pushes use HTTPS, not SSH, unless the user explicitly asks otherwise.
- Never place a token in the remote URL, a commit message, a PR body, shell output, or repo-tracked files.
- `github_personal_token.txt` is a local credential input only. Confirm it is ignored and untracked before using it.

## CoordExp HTTPS Push Default

CoordExp's normal remote shape is:

```bash
origin  https://github.com/Pein2017/CoordExp.git
```

For one-shot pushes without permanently storing credentials, use a temporary
credential helper that reads `github_personal_token.txt`:

```bash
GIT_TERMINAL_PROMPT=0 git \
  -c credential.helper= \
  -c "credential.helper=!f() { if [ \"\$1\" = get ]; then echo username=x-access-token; printf 'password='; tr -d '\n' < github_personal_token.txt; echo; fi; }; f" \
  push origin HEAD
```

Use `push origin main` instead of `push origin HEAD` when the user explicitly
asks to push `main`.

## Persistent Credential Setup

It is acceptable to make HTTPS pushes permanent and default on a trusted
machine, but Git's built-in `store` helper saves the token in plaintext under
the user's home directory. Use it only when that tradeoff is acceptable for the
node.

Set the default helper and approve the current token once:

```bash
git config --global credential.helper store
printf 'protocol=https\nhost=github.com\nusername=x-access-token\npassword=' > /tmp/coordexp_git_credential
tr -d '\n' < github_personal_token.txt >> /tmp/coordexp_git_credential
printf '\n\n' >> /tmp/coordexp_git_credential
git credential approve < /tmp/coordexp_git_credential
rm -f /tmp/coordexp_git_credential
chmod 600 ~/.git-credentials 2>/dev/null || true
```

After this, ordinary HTTPS pushes should work:

```bash
git push
```

If a token is rotated or revoked, remove the stale GitHub entry from
`~/.git-credentials` and approve the replacement token again.

## Workflow

### 0) Establish context

Run:
- `git status`
- `git branch --show-current`
- `git remote -v`
- `git check-ignore -v github_personal_token.txt`
- `git ls-files --error-unmatch github_personal_token.txt 2>/dev/null && echo TRACKED || true`

If the current branch has no upstream, plan to push with `git push -u origin HEAD`.
If the remote is not HTTPS, stop and either switch to HTTPS or ask the user.

### 1) Summarize the change pile

Collect a quick map:
- `git diff --stat`
- `git diff --name-only`
- If needed: `git diff` (spot-check key areas)

Then propose a commit plan: 2–6 commits max, grouped by intent *and similarity*.
- Prefer separating: refactor vs feature vs tests vs docs vs formatting.
- Prefer separating: unrelated areas/modules.
- Prefer separating: “mechanical” changes (config/args/format) from behavioral changes.

Example plan:
1. “refactor: extract X”
2. “feat: implement Y”
3. “test: add coverage for Y”
4. “chore: docs/formatting”

### 2) For each planned commit: stage narrowly (don’t commit everything at once)

Prefer interactive staging:
- `git add -p` (recommended)
or stage by path:
- `git add path/to/files`

Verify:
- `git diff --cached` (confirm only intended changes are staged)

Rule of thumb: if a staged diff mixes multiple intents, split it into multiple commits.

### 3) Run the smallest meaningful checks

Run the fastest checks that reduce regret:
- lint/format (if present)
- unit tests for touched modules (or the smallest test target)
If checks are slow, at least run smoke tests or a minimal subset.

### 4) Write a high-signal commit message

Default format (imperative, scoped):
- `feat(<area>): ...`
- `fix(<area>): ...`
- `refactor(<area>): ...`
- `test(<area>): ...`
- `chore(<area>): ...`

Body (optional): why, constraints, follow-ups, known gaps.

#### Minimal messages for regular/trivial changes
For standard config changes or argument tweaks (mechanical/expected/no behavioral surprise), use *minimal* commit messages, e.g.:
- `chore: config`
- `chore: args`
- `chore: flags`
- `chore: defaults`
- `chore: formatting`

Commit:
- `git commit -m "type(scope): summary"`

### 5) Repeat until the worktree is clean-ish

Aim for:
- `git status` shows either clean or only intentional leftovers (WIP experiments).

### 6) Push to remote branch

Push the **current branch**.

If upstream exists:
- `git push`

If no upstream:
- `git push -u origin HEAD`

If credentials are not yet persistent, use the one-shot HTTPS PAT command from
the CoordExp HTTPS Push Default section.

### 7) Final report

Provide:
- commit list (`git log --oneline -n <N>`)
- current status (`git status`)
- what remains uncommitted (if anything) and why

## Examples

### Example A: “I refactored + added a feature”
Plan:
1) `refactor(parser): split tokenizer`
2) `feat(parser): support new syntax`
3) `test(parser): add cases for new syntax`

Commands (typical):
- `git add -p`
- `git diff --cached`
- run targeted tests
- `git commit -m "..."`
- repeat
- `git push -u origin HEAD`

### Example B: “Too many changes, I’m lost”
Do:
- Create a commit plan based on `--name-only` and `--stat`
- Stage one directory at a time
- If truly tangled, isolate with interactive staging (`git add -p`) and leave leftovers for a follow-up commit

### Example C: “Mostly config/args tweaks”
Plan:
1) `chore: config`
2) `chore: args`

Commands:
- stage only config files first, commit with minimal message
- stage arg/flag/default changes next, commit with minimal message
