---
name: worktree-feature-loop
description: "Manage feature development in concurrent environments with an optional worktree lifecycle. Use when starting feature/fix/research tasks that may need isolation: decide whether a new worktree is needed, request user approval before creating one when not explicitly requested, choose spec path (existing OpenSpec from main, draft new OpenSpec, or no spec), implement, run audit/review rounds, merge to main, and clean up safely."
---

# Worktree Feature Loop

Run this loop for each independent feature task so multiple agents can work in parallel without stepping on each other.

## Quick Inputs

Capture these before doing work:
- `task_slug`: short unique identifier (for folder + branch naming)
- `base_branch`: usually `main`
- `spec_mode`: `existing`, `new`, or `none`
- `worktree_root`: use `.worktrees/` by default; if user explicitly requests a different root (for example `worktrees/` or `.wortrees/`), use that path

## Lifecycle

1. Decide whether to use a new worktree.
- If the user explicitly asks for a worktree, create one.
- If the request is simple editing and isolation is not required, work in the current tree.
- If unclear, ask for permission before creating a new worktree.

2. Create an isolated worktree only after explicit request or approval.
```bash
mkdir -p <worktree_root>
git worktree add <worktree_root>/<task_slug> -b <branch_name> <base_branch>
```
If `<branch_name>` already exists:
```bash
git worktree add <worktree_root>/<task_slug> <branch_name>
```

3. Choose the spec path before coding.
- `existing`: start from an existing OpenSpec already on `main`, then implement from the worktree
- `new`: draft OpenSpec in the worktree first, then implement
- `none`: implement directly, but write explicit acceptance checks in task notes/PR description

4. Implement in the approved location.
- If a worktree was created, implement inside that worktree only.
- If no worktree was approved, implement in the current tree.
- Keep changes scoped to this task
- Commit incrementally with clear messages
- Run targeted checks after each logical slice

5. Run audit and review rounds until clean.
- Self-check and fix obvious issues
- Run relevant validation/tests for changed areas
- Address external review findings and rerun checks
- Repeat until implementation artifacts and behavior match requirements/spec

6. Merge into `main` after verification passes.
If implementation used a worktree, verify it is clean first:
```bash
git -C <worktree_root>/<task_slug> status --short
```
Then merge on `main`:
```bash
git checkout main
git pull --ff-only
git merge --no-ff <branch_name>
```
Use project-preferred merge style if different.

7. Clean up the worktree only after successful merge.
```bash
git worktree remove <worktree_root>/<task_slug>
git branch -d <branch_name>
```
If cleanup might remove uncommitted work, stop and ask for explicit confirmation.

## Concurrent Development Rules

- One task per worktree and one active branch per worktree.
- Never modify another task's worktree unless explicitly asked.
- Treat dirty files in other worktrees as expected; focus only on your assigned worktree.
- Report every handoff with exact `worktree_path`, `branch_name`, spec mode, and validation status.

## OpenSpec Handling

When OpenSpec is involved:
- `existing`: continue from current spec artifacts before/while implementing
- `new`: draft the required artifacts, then implement, then verify against artifacts
- Before merge, confirm implementation and artifacts are aligned

## Output Checklist

Return this concise status block at the end of each run:
- `task_slug`
- `execution_mode` (`worktree`/`inplace`)
- `worktree_path` (`n/a` when `inplace`)
- `branch_name` (`current_branch` when `inplace`)
- `spec_mode` (`existing`/`new`/`none`)
- `validation_ran`
- `merge_state` (`not_started`/`ready`/`merged`)
- `cleanup_state` (`pending`/`done`)
