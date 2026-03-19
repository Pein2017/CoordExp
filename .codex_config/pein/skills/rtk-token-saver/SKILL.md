---
name: rtk-token-saver
description: Default to RTK in Codex CLI for shell commands whenever output compaction can help, so noisy output from git, file reads, search, tests, builds, logs, and diagnostics stays compact. Trigger automatically for shell-heavy or potentially noisy command work when the `rtk` binary is available; explicit user requests to save tokens or use RTK are a strong signal, not a requirement.
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
