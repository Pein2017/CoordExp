---
name: rtk-token-saver
description: Default to RTK in Codex CLI for shell commands whenever output compaction can help, so noisy output from git, file reads, search, tests, builds, logs, and diagnostics stays compact. Trigger automatically for shell-heavy or potentially noisy command work when the `rtk` binary is available; explicit user requests to save tokens or use RTK are a strong signal, not a requirement.
---

# RTK Token Saver

Codex CLI does not expose the same pre-tool hook path that RTK supports for Claude Code and OpenCode.
When this skill is active, emulate that behavior by routing shell commands through `scripts/rtk_auto.sh`, resolved relative to this skill directory.

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
- RTK has no rewrite for the command and the wrapper falls back
- the command is already trivially tiny and wrapping it would add more noise than value

## Working Rules

- Default order:
  1. For most shell commands, prefer RTK first.
  2. If the RTK form is obvious, call `rtk ...` directly.
  3. Otherwise call `scripts/rtk_auto.sh "<command>"`.
  4. Only bypass RTK for the skip cases above.
- If you need exact output for debugging, say why you are bypassing RTK.
- Do not claim the output is verbatim if RTK filtered it.
- Keep the wrapper call scoped to the noisy command instead of wrapping unrelated shell setup.
- In ambiguous cases, prefer saving tokens unless there is a concrete downside to filtering.

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
