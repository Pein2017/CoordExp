---
name: worktree-feature-loop
description: "Use when starting feature, fix, or research work in a repo where isolation may matter; decides worktree vs inplace execution and owns the lifecycle from setup through cleanup"
---

# Worktree Feature Loop

Run this loop for each independent feature or research task so multiple agents can work in parallel without stepping on each other.

This is the policy owner for worktree usage.

- It decides `execution_mode=worktree|inplace`
- It chooses the preferred root, usually `.worktrees/`
- It delegates actual creation to `using-git-worktrees`
- It owns the lifecycle through merge/cleanup

## Quick Inputs

Capture these before doing work:
- `task_slug`: short unique identifier (for folder + branch naming)
- `base_branch`: usually `main`
- `spec_mode`: `existing`, `new`, or `none`
- `worktree_root`: use `.worktrees/` by default; if user explicitly requests a different root (for example `worktrees/` or `.wortrees/`), use that path

## Default Policy

Default to `execution_mode=worktree` for:

- research tasks
- multi-file changes
- tasks expected to live for more than one logical commit
- tasks that may run in parallel with other work
- tasks started while the current tree is dirty
- tasks involving long-running experiments, evaluation, or artifact generation

Use `execution_mode=inplace` only when the task is small, local, and isolation is clearly unnecessary.

For CoordExp-style work, prefer `.worktrees/` unless the user explicitly requests a different root.

## Lifecycle

1. Decide whether to use a new worktree.
- If the user explicitly asks for a worktree, create one.
- If repo policy or task type matches the default policy above, choose `worktree`.
- If the request is simple editing and isolation is not required, work in the current tree.
- If no default policy applies and the trade-off is unclear, ask for permission before creating a new worktree.

2. If `execution_mode=worktree`, invoke `using-git-worktrees`.
- Pass `worktree_root`, `branch_name`, and `base_branch`
- Let that skill handle ignore checks, creation, setup, and baseline verification
- Do not duplicate directory-selection or ignore-verification logic here

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
- When running from a worktree, prefer absolute or shared-root-anchored paths for shared outputs, datasets, checkpoints, and caches.

5. Run audit and review rounds until clean.
- Self-check and fix obvious issues
- Run relevant validation/tests for changed areas
- Address external review findings and rerun checks
- Repeat until implementation artifacts and behavior match requirements/spec

6. Finish using `finishing-a-development-branch` after verification passes.
If implementation used a worktree, verify it is clean first:
```bash
git -C <worktree_root>/<task_slug> status --short
```

Then hand off final branch resolution to `finishing-a-development-branch`:
- merge locally
- PR
- keep as-is
- discard

7. Clean up the worktree only after branch resolution is complete.
- Remove the worktree only after merge or discard is finished.
- If cleanup might remove uncommitted work, stop and ask for explicit confirmation.

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
