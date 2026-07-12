# CoordExp Worktree Lifecycle

Read this reference only when deciding, creating, entering, repairing, or removing a linked worktree.

## Choose the execution mode

Prefer a worktree for parallel work, unrelated dirt in the current checkout, long-running training/eval/artifact generation, or multi-file/multi-commit changes. Prefer in-place work for a narrow edit with clear ownership and no contract or artifact-lifecycle change.

The checkout or worktree named by the user is authoritative. Do not redirect the task to another checkout based on historical roles or branch names.

## Snapshot before mutation

Run from the intended repository:

```bash
git rev-parse --show-toplevel
git branch --show-current
git status --short --branch
git worktree list --porcelain
```

If publication or cleanup is in scope, also inspect the upstream and ahead/behind state. Stop on overlapping dirty files with unclear ownership, tracked secrets, destructive ambiguity, or publication not requested.

## Create or enter

Branch names default to `codex/<task-slug>`. Create a worktree only when the user asked for a branch/worktree or the approved workflow requires isolation:

```bash
git worktree add -b "codex/$task_slug" "$worktree_path" "$base_branch"
```

Before editing, re-run the snapshot commands inside the new path. If the branch or path already exists, inspect it; do not replace or force it.

### Restore ignored heavy roots when needed

Linked worktrees may not contain ignored model and artifact roots. For workflows that use repository-relative `model_cache/` or `outputs/`, create local symlinks back to the canonical shared root without replacing anything:

```bash
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

Do not stage these runtime symlinks. Prefer absolute/shared-root paths for large data, checkpoints, caches, and outputs.

### Navigation tools

Initialize worktree-local indexes only when the corresponding tool is installed and needed. Pass the exact absolute worktree path to repository-aware tools; names can resolve to another checkout. Treat indexes as navigation aids, then verify behavior in live source or executed semantics.

## Work and verify

- Edit only in the approved worktree and preserve dirt in every other checkout.
- Put durable research interpretation in `research/`, stable behavior in `docs/` or the relevant spec, and requested artifacts in their canonical roots. Do not create new `progress/` records.
- Run the smallest realistic verification before staging.
- Follow the main `git-hygiene` commit and sync loop for staging, PAT use, fetch/pull/push, or conflict handling.

## Cleanup gate

Remove a worktree only when all of these are true:

- its status is clean;
- its commits are merged, or the user explicitly approved discarding them;
- durable findings and required artifacts have been promoted;
- the path and branch match the intended cleanup target;
- cleanup is explicitly in scope.

Run removal from a repository control checkout:

```bash
git worktree list --porcelain
git -C "$worktree_path" status --short --branch
git worktree remove "$worktree_path"
git worktree prune --dry-run
```

Run actual `git worktree prune` only after reviewing the dry run. Branch deletion is a separate decision: inspect ahead/behind and remote refs before deleting it.
