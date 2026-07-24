---
name: worktree-feature-loop
description: Use when deciding whether to isolate CoordExp work in a Git worktree, creating or entering one, checking checkout identity and runtime links, or preparing it for merge and cleanup.
---

# Worktree Feature Loop

Keep one task tied to one exact checkout from preflight through handoff.
Delegate staging, commits, sync, and conflict resolution to `git-hygiene`.

## Decide

Use a worktree for research, parallel work, long-running jobs, a dirty current
checkout, or changes that may need independent review. Work in place only for a
narrow reversible edit with no overlapping dirt or long-running artifacts.

Use `main` as the base unless the user names another canonical branch. Ask
before choosing when the base changes scientific meaning or merge direction.

## Start

1. Inspect `pwd`, repository root, branch, `git status --short --branch`, and
   `git worktree list`.
2. Confirm the task's exact worktree and branch before editing. Preserve
   unrelated dirty files and isolate overlapping ownership.
3. Create a worktree only when needed. Use the `codex/` branch prefix by
   default and never replace an existing path silently.
4. If the worktree needs shared heavy roots, link missing `model_cache` and
   `outputs` paths to `/data/CoordExp`. Refuse to replace an existing file,
   directory, or different symlink, and never stage these runtime links.
5. Keep checkpoints, data, caches, and generated outputs outside Git. Use
   absolute paths when a worktree-relative path could resolve differently.

## Navigate with Serena

Serena MCP uses the custom `coordexp-codex` context, not the built-in `codex`
context. Its stdio server starts with `--project-from-cwd`, so the initial
project follows the launch directory. Explicit project switching is enabled;
Serena memory, onboarding, shell, generic file operations, and broad text search
remain disabled. CoordExp research knowledge remains in `research/`, docs,
skills, and Codex memory. Serena runtime state is persisted under
`/data/CoordExp/.codex/runtime/serena`. Install or verify the portable context
and concise initial instructions with:

```bash
.codex/serena/setup.sh
.codex/serena/setup.sh --check
```

At the start of a conclusion-sensitive task, call `initial_instructions` and
confirm the reported absolute worktree path. The response is intentionally a
short routing note plus project receipt, not a general coding manual. If the
task moves to another folder or worktree, call `activate_project` with the exact
absolute path and verify its activation receipt before using symbol tools. A
server restart is not required solely because the target path changed. Avoid
ambiguous project names, and do not mix evidence from different worktrees.

Use these Serena tools when symbol semantics matter:

- inspect: `get_symbols_overview`, `find_symbol`,
  `find_referencing_symbols`, `find_declaration`, `find_implementations`, and
  `get_diagnostics_for_file`;
- refactor: `replace_symbol_body`, `insert_before_symbol`,
  `insert_after_symbol`, `rename_symbol`, and `safe_delete_symbol`;
- integration: `initial_instructions` for the startup project receipt,
  `activate_project` for an explicit absolute-path switch, and
  `get_current_config` when the active project needs another confirmation.

Use `rg` and RTK for broad text or file search, and `apply_patch` for ordinary
edits. Manual `serena project index` is optional because the language-server
index updates with source changes. When Serena conflicts with the live source,
verify the file directly and reactivate the exact absolute project path.

## Complete

1. Run the smallest realistic verification in the same worktree.
2. Preserve durable research interpretation under `research/` and report exact
   artifact roots before cleanup.
3. Report the worktree path, branch, changed files or commits, verification,
   remaining dirt, and merge or cleanup state.
4. Remove a worktree only after its work is merged or explicitly discarded,
   its durable artifacts are preserved, and `git status` is clean. Perform
   removal from another repository checkout.

Stop for user input on an ambiguous base, overlapping dirty ownership,
destructive cleanup, publication, secrets, or a newly required expensive run.
