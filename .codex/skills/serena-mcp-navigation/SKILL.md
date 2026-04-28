---
name: serena-mcp-navigation
description: "Use when code work needs symbol-aware navigation, reference finding, call-site tracing, or precise edits, especially for CoordExp Python files after narrowing candidates with rg or rtk."
---

# Serena MCP Navigation

Navigate codebases with symbol-aware precision using Serena MCP tools. Superior to CLI tools for relationship discovery and targeted code analysis.

## Core Workflows

### Project Activation Preflight
1. Run `get_current_config` before Serena symbol work when the active project may not match the current repo/worktree
2. If the active project is wrong but the target is listed under available projects, run `activate_project` with the project name, for example `CoordExp`
3. If the target is not listed, run `activate_project` with the absolute project/worktree path, for example `/data/CoordExp` or `/data/CoordExp/.worktrees/<name>`
4. After activation, run `check_onboarding_performed` before symbol exploration or edits
5. Only fall back to CLI-only code navigation if activation fails or Serena lacks the needed language/tool support; state that specific blocker

### Locate Symbol Definition
1. Use CLI `rg` to narrow candidate files/dirs first (avoid Serena repo-wide search)
2. Use `search_for_pattern` only with tight, scoped `relative_path` (never `"."` / repo root, and don’t leave it unset) if file location is still unknown
3. Run `get_symbols_overview` on candidate files for symbol inventory
4. Use `find_symbol` with `depth=1` to list class methods
5. Retrieve specific method body with `find_symbol(... include_body=True)`

### Find References and Dependencies
1. Locate target symbol with `find_symbol` (no body needed)
2. Run `find_referencing_symbols` to enumerate all call sites immediately
3. Read only relevant caller bodies via `find_symbol(... include_body=True)`

### Make Precise Edits
1. Locate symbol using definition workflow above
2. For whole methods/functions/classes: use `replace_symbol_body` after retrieving current body
3. For insertions: use `insert_before_symbol` near a known symbol when available
4. For small text-only changes outside a symbol body, use `apply_patch`

## Tool Selection Guide

**Default to Serena MCP** for code analysis - superior accuracy, completeness, and better cache efficiency.

### Use Serena MCP for:
- Symbol navigation and relationship discovery
- Finding references, callers, and dependencies
- Precise code editing and refactoring
- Deep implementation analysis
- CoordExp Python exploration and editing after an initial `rg`/`rtk` narrowing pass

### Use CLI tools for:
- Documentation and prose scanning
- Bulk text search across mixed filetypes
- Config validation and log inspection
- Contiguous code block reading

## Navigation Strategy

1. **Activate target project** - Serena MCP requires project activation for symbol access; switch projects with `activate_project` instead of falling back when Serena is pointed at another root
2. **Start with symbol overview** - Use `get_symbols_overview` for structured file inventory
3. **Find references early** - Use `find_referencing_symbols` to map relationships
4. **Read selectively** - Use `find_symbol` with `include_body=True` for targeted method access

## Advanced Usage

### Reference Discovery
Use `find_referencing_symbols` to immediately surface all relationships:
1. Locate target symbol with `find_symbol`
2. Run `find_referencing_symbols` for complete call graph
3. Read specific caller implementations as needed

### Code Editing
Choose the minimal editing approach:
- **Whole symbols**: `replace_symbol_body` after retrieving current body
- **Insertions**: `insert_before_symbol`/`insert_after_symbol` near relevant symbols
- **Small text-only changes**: use `apply_patch` after Serena identifies the correct location

## Best Practices

- If Serena is pointed at a different project root, do not immediately fall back to shell reads. Use `get_current_config` to inspect active/available projects, then `activate_project` by project name or absolute directory path.
- For disposable worktrees or nested repos, prefer activating the exact absolute directory you are editing so Serena indexes the same files as the shell.
- Never set `"relative_path": "."` (or leave `relative_path` unset for a repo-wide scan) in Serena tools; use CLI `rg` to narrow scope first
- Always specify the smallest viable `relative_path` (single file or tight subdir)
- Use scoped searches to avoid overwhelming results
- Leverage `find_referencing_symbols` early for relationship mapping
- Reserve CLI tools for documentation and bulk text scanning
- For Markdown, YAML, JSON, exact line reads, and generated artifacts, prefer shell reads plus `apply_patch`; Serena is most valuable where symbol boundaries or references matter.

## Pairing With RTK

Use Serena for code semantics and RTK for shell compaction:

- Start with `rtk` when you need to read docs, inspect logs, scan the repo, or summarize `git`/test output before touching code.
- Use `rg` or `rtk grep` to narrow candidate files, then switch to Serena for Python symbol discovery and editing.
- Treat Serena as the source of truth for Python structure. Do not rely on `rtk smart`, raw full-file reads, or broad text search when the task depends on references, callers, or symbol boundaries.
- After Serena-driven edits, switch back to `rtk` for compact verification: `rtk conda run -n ms python -m pytest ...`, `rtk git diff`, `rtk log`, and similar checks.
- If exact stdout or shell semantics matter, bypass RTK deliberately and say why.
