---
name: worktree-feature-loop
description: Use when starting CoordExp feature, fix, or research work where a dirty tree, parallel work, experiments, or long-running artifacts make isolation useful.
---

# Worktree Feature Loop

Choose `execution_mode=worktree|inplace`, then own the lifecycle through verification and cleanup. Delegate creation details to `using-git-worktrees` when a worktree is chosen.

This skill owns isolation, lifecycle state, checkout identity, and cleanup readiness. It does not own commit grouping, staging, remote sync, or conflict resolution; delegate those to `git-hygiene`.

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

CoordExp stable operational checkout: `/data/CoordExp` on `main`; use it for
official training and evaluation. The active development checkout is
`/data/CoordExp/.worktrees/CoordExp-swift` on `coordexp-swift`; implement and
validate features there before an explicit merge to `main`. `ms-swift` is an
unmounted history archive branch. Worktree roots still default to `.worktrees/`
unless the user explicitly requests another path. Remember
`mcp/codexUI` is a nested git repo; never sweep nested-repo changes into parent
commits.

## Inputs

- `task_slug`
- `base_branch` (usually `main`)
- `spec_mode`: `existing`, `new`, or `none`
- `worktree_root`

Branch prefix defaults to `codex/`.

## Lifecycle

1. Decide execution mode.
2. If worktree, invoke `using-git-worktrees` with root, branch, and base branch.
3. After creating or entering a worktree, restore runtime path parity with the
   main checkout by manually adding local symlinks for ignored heavy roots:
   ```bash
# Shared data/artifact root in the stable main checkout.
   main_root=/data/CoordExp
   for name in model_cache outputs; do
     target="$main_root/$name"
     link="$PWD/$name"
     if [ -e "$link" ] || [ -L "$link" ]; then
       if [ "$(readlink "$link" 2>/dev/null || true)" = "$target" ]; then
         continue
       fi
       echo "Refusing to replace existing $link; inspect manually." >&2
       exit 1
     fi
     test -e "$target" || { echo "Missing shared root: $target" >&2; exit 1; }
     ln -s "$target" "$link"
   done
   ```
   These links make worktree-relative paths such as `model_cache/...` and
   `outputs/...` behave like the main checkout. Do not stage these runtime
   symlinks unless the user explicitly requests tracking them.
4. Initialize local navigation state:
   - run `codegraph init -i` in the exact worktree when CodeGraph will be used;
   - confirm with `codegraph status`;
   - activate the exact worktree path with Serena MCP before narrowed Python symbol inspection, reference checks, diagnostics, or symbolic edits;
   - for CodeGraph MCP calls in linked worktrees, pass `projectPath=/absolute/worktree/path`.
5. Choose planning surface:
   - `existing`: continue existing `openspec-lifecycle` or super-power artifacts;
   - `new`: create only the appropriate repo-local plan/spec artifacts;
   - `none`: implement directly and record acceptance checks in final/PR text.
6. Implement only in the approved tree.
7. Use absolute or shared-root paths for heavy data, checkpoints, caches, and outputs.
8. Put one-off debug artifacts under `temp/` and clean them after durable evidence is extracted.
9. Validate the smallest realistic surface.
10. For commits or sync, invoke `git-hygiene` before staging anything and delegate detailed staging, PAT, fetch/pull/push, and conflict handling there; keep this skill focused on lifecycle state.
11. For OpenSpec contract artifacts, delegate mode-specific workflow to `openspec-lifecycle`.
12. Finish with `finishing-a-development-branch` or the user's requested commit/push/merge flow.
13. Remove worktree only after merge/discard, from the repository control
    checkout, with provenance check and no uncommitted work. Confirm that
    `/data/CoordExp/.worktrees/CoordExp-swift` remains the `coordexp-swift`
    development worktree before any cleanup.

## CoordExp Gotchas

- In worktrees, ignored data/model/output roots may be missing. Prefer local
  symlinks over config path rewrites; by default, link `model_cache` and
  `outputs` back to `/data/CoordExp/model_cache` and `/data/CoordExp/outputs`
  immediately after worktree creation.
- CodeGraph indexes are worktree-local. A parent/root `.codegraph/` is not enough for implementation in a linked worktree.
- Serena project activation by name can point to another checkout. Prefer activation by absolute worktree path for side branches and parallel implementation lanes.
- Use CodeGraph for first-pass "where is this?" maps, then Serena for exact Python symbol semantics once files/classes/functions are known; do not let CodeGraph replace Serena for reference-sensitive edits.
- For infer/eval fanout, preserve canonical image roots or explicitly rewrite them before launching shards.
- Dirty files in other worktrees are expected and out of scope.
- If a research worktree produced durable findings, promote interpretation and
  continuation context to `research/`, stable behavior to `docs/` or OpenSpec,
  and concrete artifacts to requested artifact locations before cleanup. Do not
  create new `progress/` records.

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
