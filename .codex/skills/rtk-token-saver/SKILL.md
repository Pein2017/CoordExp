---
name: rtk-token-saver
description: Use `rtk` as the default execution layer for shell commands when available. Automatically apply it to commands likely to produce multi-line or noisy output (e.g., git, search, file reads, tests, logs), without requiring explicit user instruction. Prefer `rtk` in exploratory or shell-heavy workflows to reduce token usage. Fall back to raw commands only when exact, unfiltered output is required, RTK adds risk (e.g., quoting/state), or provides no meaningful benefit.
---

# RTK Token Saver

Use `rtk` directly as the primary path.

Direct `rtk ...` calls are simpler, easier to reason about, and avoid path-resolution mistakes.

## Goal

- Use RTK-filtered shell output by default for shell commands whenever compaction is likely to help.
- Fall back to the raw command when RTK has no rewrite or when exact raw output matters more than compression.

## Autonomous Activation

Activate this skill proactively when:

- the task involves multiple shell commands, especially exploratory repo work
- a command is likely to emit more than a few lines of output
- the command is a good fit for RTK's rewrite/filter behavior
- saving tokens is beneficial even if the user did not ask explicitly

Do not wait for a user to say "use RTK" if the command obviously benefits from output compaction.
Treat explicit user requests as confirmation, but not as the only trigger.

## Quick Start

1. Check that RTK exists:
```bash
command -v rtk
```
2. Use direct RTK as the default path for ordinary shell work:
```bash
rtk git status
rtk grep "TODO" src
rtk pytest tests/test_example.py
rtk read docs/PROJECT_CONTEXT.md
```
3. Only if direct `rtk ...` is awkward or unsupported, fall back to the raw command.

## Default RTK Cases

RTK is the default for commands that usually emit a lot of text, and for ordinary shell work where compact output is preferable:

- `git ...`
- `rg`, `grep`, `find`, `ls`, `read`, `sed -n`
- `pytest`, `ruff`, `mypy`, `cargo`, `go test`
- `npm`, `pnpm`, `tsc`, `next`, `lint`, `format`
- `curl`, logs, and error-focused runs when RTK has a matching subcommand

## When To Skip RTK

Use the raw command instead when:

- the user explicitly asks for verbatim output
- exact machine-readable stdout matters more than compression
- the command depends on shell state, heredocs, or delicate quoting
- RTK has no useful rewrite/filter for the command
- the command is already trivially tiny and wrapping it would add more noise than value

## Working Rules

- Default order:
  1. For most shell commands, prefer RTK first.
  2. Call `rtk ...` directly when the mapping is obvious.
  3. If direct `rtk ...` is not a good fit, run the raw command instead of introducing wrapper indirection.
- If you need exact output for debugging, say why you are bypassing RTK.
- Do not claim the output is verbatim if RTK filtered it.
- In ambiguous cases, prefer saving tokens unless there is a concrete downside to filtering.

## Pairing With Serena MCP

Use RTK and Serena as complementary layers, not as a single chained tool:

- Use `rtk` first for repo orientation: `rtk read` for docs/prose, `rtk grep`/`rtk find` for coarse search, `rtk git ...` for status/diff, and `rtk pytest`/`rtk test` for compact verification.
- Once the task enters Python code understanding or editing, switch to Serena MCP for symbol-aware navigation, reference discovery, and precise edits.
- Return to `rtk` for post-edit verification such as targeted tests, diffs, logs, or other shell-heavy checks.
- Do not try to replace Serena with `rtk smart` or aggressive file filtering when you need call relationships, symbol boundaries, or safe edits.
- Do not force `rtk` into exact-output workflows; raw `sed`, raw script execution, or other unfiltered commands are still the right tool when exact stdout matters.

Typical split:

1. `rtk read docs/PROJECT_CONTEXT.md`
2. `rtk grep "TargetSymbol" src`
3. Serena MCP: `get_symbols_overview` -> `find_symbol` -> `find_referencing_symbols`
4. `rtk pytest tests/...`
5. `rtk git diff`

## Verification

- Rewrite probe:
```bash
rtk rewrite "git status"
```
- Direct execution:
```bash
rtk git status
```
- Raw fallback for unsupported commands:
```bash
pwd
```
