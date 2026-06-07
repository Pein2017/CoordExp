---
name: worktree-feature-loop
description: Use when starting CoordExp feature, fix, or research work where a dirty tree, parallel work, experiments, or long-running artifacts make isolation useful.
---

# Worktree Feature Loop

Choose `execution_mode=worktree|inplace`, then own the lifecycle through verification and cleanup. Delegate creation details to `using-git-worktrees` when a worktree is chosen.

## Preflight Snapshot

Before edits, commits, sync, or cleanup, inspect root, branch, linked worktree state, `git status --short --branch`, and `git worktree list`.

When sync, publication, or cleanup is in scope, also inspect upstream/ahead/behind state and confirm no nested `mcp/codexUI` changes would be swept into the parent repo.

Stop or ask on ambiguous scope, tracked secrets, destructive cleanup, publication not requested, high-cost training/smoke, or dirty files overlapping intended edits with unclear ownership.

## Default Decision

Use a worktree for:

- research tasks;
- multi-file or multi-commit work;
- long-running training/eval/artifact generation;
- current-tree dirty state;
- parallel-agent work;
- schema, artifact-name, metric-semantics, stable-default, or recommended-workflow changes;
- changes likely to need branch review.

Use inplace only for narrow edits with explicit verification, no stable contract change, no long-running outputs, and no unrelated dirt in the same files. If tradeoff is unclear, ask before creating.

CoordExp default root: `.worktrees/`, unless the user explicitly requests another path. Remember `mcp/codexUI` is a nested git repo; never sweep nested-repo changes into parent commits.

## Inputs

- `task_slug`
- `base_branch` (usually `main`)
- `spec_mode`: `existing`, `new`, or `none`
- `worktree_root`

Branch prefix defaults to `codex/`.

## Lifecycle

1. Decide execution mode.
2. If worktree, invoke `using-git-worktrees` with root, branch, and base branch.
3. After creating or entering a worktree, initialize local navigation state:
   - run `codegraph init -i` in the exact worktree when CodeGraph will be used;
   - confirm with `codegraph status`;
   - activate the exact worktree path with Serena MCP when Python symbol work is needed;
   - for CodeGraph MCP calls in linked worktrees, pass `projectPath=/absolute/worktree/path`.
4. Choose planning surface:
   - `existing`: continue existing `openspec-lifecycle` or super-power artifacts;
   - `new`: create only the appropriate repo-local plan/spec artifacts;
   - `none`: implement directly and record acceptance checks in final/PR text.
5. Implement only in the approved tree.
6. Use absolute or shared-root paths for heavy data, checkpoints, caches, and outputs.
7. Put one-off debug artifacts under `temp/` and clean them after durable evidence is extracted.
8. Validate the smallest realistic surface.
9. For commits or sync, delegate detailed staging, PAT, fetch/pull/push, and conflict handling to `git-hygiene`; keep this skill focused on lifecycle state.
10. For OpenSpec contract artifacts, delegate mode-specific workflow to `openspec-lifecycle`.
11. Finish with `finishing-a-development-branch` or the user's requested commit/push/merge flow.
12. Remove worktree only after merge/discard, from the main root, with provenance check and no uncommitted work.

## CoordExp Gotchas

- In worktrees, ignored data/model roots may be missing; prefer local symlinks over config path rewrites.
- CodeGraph indexes are worktree-local. A parent/root `.codegraph/` is not enough for implementation in a linked worktree.
- Serena project activation by name can point to another checkout. Prefer activation by absolute worktree path for side branches and parallel implementation lanes.
- For infer/eval fanout, preserve canonical image roots or explicitly rewrite them before launching shards.
- Dirty files in other worktrees are expected and out of scope.
- If a research worktree produced durable findings, promote canonical outputs to `progress/`, docs, or requested artifact locations before cleanup.

## Final Status Block

```text
task_slug:
request_class:
execution_mode:
worktree_path:
branch_name:
base_branch:
spec_mode:
changed_files_or_commits:
validation_ran:
validation_skipped:
codegraph_index:
serena_project:
artifact_roots:
sync_state:
merge_state:
cleanup_state:
remaining_dirty:
residual_risks:
```
