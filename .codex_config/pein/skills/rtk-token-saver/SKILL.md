---
name: rtk-token-saver
description: Default to RTK in Codex CLI for applicable shell commands so noisy output from git, file reads, search, tests, builds, logs, and diagnostics stays compact. Trigger when the user asks to save tokens, use RTK, reduce shell noise, or compact terminal output and the `rtk` binary is available.
---

# RTK Token Saver

Codex CLI does not expose the same pre-tool hook path that RTK supports for Claude Code and OpenCode.
When this skill is active, emulate that behavior by routing shell commands through `scripts/rtk_auto.sh`, resolved relative to this skill directory.

## Goal

- Use RTK-filtered shell output by default for applicable shell commands.
- Fall back to the raw command when RTK has no rewrite or when exact raw output matters more than compression.

## Quick Start

1. Check that RTK exists:
```bash
command -v rtk
```
2. Use RTK as the default path for ordinary shell work:
```bash
scripts/rtk_auto.sh "git status"
scripts/rtk_auto.sh "rg TODO src"
scripts/rtk_auto.sh "pytest tests/test_example.py"
```
3. If the RTK form is obvious, call it directly:
```bash
rtk git status
rtk grep TODO src
rtk read docs/PROJECT_CONTEXT.md
```

## Default RTK Cases

RTK is the default for commands that usually emit a lot of text:

- `git ...`
- `rg`, `grep`, `find`, `ls`, `read`
- `pytest`, `ruff`, `mypy`, `cargo`, `go test`
- `npm`, `pnpm`, `tsc`, `next`, `lint`, `format`
- `curl`, logs, and error-focused runs when RTK has a matching subcommand

## When To Skip RTK

Use the raw command instead when:

- the user explicitly asks for verbatim output
- exact machine-readable stdout matters more than compression
- the command depends on shell state, heredocs, or delicate quoting
- RTK has no rewrite for the command and the wrapper falls back

## Working Rules

- Default order:
  1. If the RTK form is obvious, call `rtk ...` directly.
  2. Otherwise call `scripts/rtk_auto.sh "<command>"`.
  3. Only bypass RTK for the skip cases below.
- If you need exact output for debugging, say why you are bypassing RTK.
- Do not claim the output is verbatim if RTK filtered it.
- Keep the wrapper call scoped to the noisy command instead of wrapping unrelated shell setup.

## Verification

- Rewrite probe:
```bash
rtk rewrite "git status"
```
- Wrapped execution:
```bash
scripts/rtk_auto.sh "git status"
```
- Fallback execution for unsupported commands:
```bash
scripts/rtk_auto.sh "pwd"
```
