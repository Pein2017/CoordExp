---
name: serena-mcp-navigation
description: Use when CoordExp code work needs symbol-aware navigation, references, call-site tracing, or precise edits after rg/rtk narrowing.
---

# Serena MCP Navigation

For Python exploration/editing in CoordExp, use Serena after narrowing candidates with `rg` or `rtk`. Use shell reads for Markdown, YAML, JSON, generated artifacts, and exact line snippets.

## Preflight

1. Check the active project if it may not match the current repo/worktree.
2. Activate the exact repo or worktree path, for example `/data/CoordExp` or `/data/CoordExp/.worktrees/<name>`.
3. Check onboarding before symbol exploration or edits.
4. If activation or language support fails, state that blocker before falling back.

## Symbol Workflow

1. Narrow with `rg`/`rtk grep`.
2. Use `get_symbols_overview` on candidate files.
3. Use `find_symbol(..., depth=1)` to list methods.
4. Read only needed bodies with `include_body=True`.
5. Use `find_referencing_symbols` early when callers or compatibility matter.

## Editing

- Whole function/class/method: `replace_symbol_body`.
- Insert near a known symbol: `insert_before_symbol` or `insert_after_symbol`.
- Small text-only change inside a larger body or non-code file: `apply_patch`.

## Guardrails

- Do not run Serena repo-wide pattern scans with `relative_path` unset or `"."`; narrow first with shell search.
- Activate the exact worktree being edited.
- After Serena edits, verify with targeted tests or `rtk git diff`.
