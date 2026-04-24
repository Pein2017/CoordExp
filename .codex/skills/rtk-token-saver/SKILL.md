---
name: rtk-token-saver
description: "Use when shell output is likely to be noisy and token-heavy: broad search, docs reads, git diff/status/log, tests, logs, file discovery, or daily repo orientation. Skip when exact stdout, machine-readable output, delicate shell semantics, or tiny commands matter more than compression."
---

# RTK Token Saver

Use `rtk` directly as the primary path.

Direct `rtk ...` calls are simpler, easier to reason about, and avoid path-resolution mistakes.

## Goal

- Use RTK-filtered shell output by default for shell commands whenever compaction is likely to help.
- Fall back to the raw command when RTK has no rewrite or when exact raw output matters more than compression.
- Treat RTK as a noise-control layer, not a replacement for semantic code navigation or exact-output commands.

## Measured Effect

In `/data/CoordExp`, `rtk gain --project` showed about `1.3M` tokens saved across `2265` commands, a `65.8%` overall reduction. The largest wins came from broad search and discovery:

- `rtk grep`: biggest aggregate saver
- `rtk read`: useful in aggregate for docs/progress reads
- `rtk find`, `rtk ls`, and `rtk git diff`: strong when output is large

Daily savings can be near zero when commands are already tiny or fall back to raw execution. That is fine; use RTK where output volume is the problem.

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
rtk conda run -n ms python -m pytest tests/test_example.py
rtk read docs/PROJECT_CONTEXT.md
```
3. Only if direct `rtk ...` is awkward or unsupported, fall back to the raw command.

## Default RTK Cases

RTK is the default for commands that usually emit a lot of text, and for ordinary shell work where compact output is preferable:

- `git status`, `git diff`, `git log`
- broad `rg`/`grep`, `find`, `ls`, `tree`
- docs/prose reads with `rtk read`
- `pytest`, `ruff`, `mypy`, `cargo`, `go test`
- `npm`, `pnpm`, `tsc`, `next`, `lint`, `format`
- `curl`, logs, and error-focused runs when RTK has a matching subcommand

For test and lint commands, preserve any existing project-specific wrapper:

- good: `rtk conda run -n ms python -m pytest tests/test_example.py`
- risky: `rtk pytest tests/test_example.py` when the repo expects a non-default environment

## When To Skip RTK

Use the raw command instead when:

- the user explicitly asks for verbatim output
- exact machine-readable stdout matters more than compression
- the command depends on shell state, heredocs, or delicate quoting
- the command depends on a specific interpreter, venv, conda env, or launcher and `rtk ...` would bypass that wrapper
- RTK has no useful rewrite/filter for the command
- the command is already trivially tiny and wrapping it would add more noise than value
- a narrow exact line read is needed, such as `sed -n '120,170p' file.py` or `nl -ba file.py | sed -n '120,170p'`
- downstream parsing depends on exact JSON/CSV/table output, such as `jq`, `python -c`, or a script whose stdout is consumed directly

## Working Rules

- Default order:
  1. For most shell commands, prefer RTK first.
  2. Call `rtk ...` directly when the mapping is obvious.
  3. If the repo already has a required wrapper such as `conda run -n ms`, keep that wrapper under `rtk`.
  4. If direct `rtk ...` is not a good fit, run the raw command instead of introducing wrapper indirection.
- If you need exact output for debugging, say why you are bypassing RTK.
- Do not claim the output is verbatim if RTK filtered it.
- In ambiguous cases, prefer saving tokens unless there is a concrete downside to filtering.
- Use `rtk gain`, `rtk gain --project`, and `rtk gain --history` to inspect actual savings instead of guessing about impact.
- Use `rtk rewrite "<command>"` to see whether a raw command maps cleanly to an RTK equivalent.

## Pairing With Serena MCP

Use RTK and Serena as complementary layers, not as a single chained tool:

- Use `rtk` first for repo orientation: `rtk read` for docs/prose, `rtk grep`/`rtk find` for coarse search, `rtk git ...` for status/diff, and `rtk conda run -n ms python -m pytest ...` for compact verification in this repo.
- Once the task enters Python code understanding or editing, switch to Serena MCP for symbol-aware navigation, reference discovery, and precise edits.
- Return to `rtk` for post-edit verification such as targeted tests, diffs, logs, or other shell-heavy checks.
- Do not try to replace Serena with `rtk smart` or aggressive file filtering when you need call relationships, symbol boundaries, or safe edits.
- Do not force `rtk` into exact-output workflows; raw `sed`, raw script execution, or other unfiltered commands are still the right tool when exact stdout matters.

Typical split:

1. `rtk read docs/PROJECT_CONTEXT.md`
2. `rtk grep "TargetSymbol" src`
3. Serena MCP: `get_symbols_overview` -> `find_symbol` -> `find_referencing_symbols`
4. `rtk conda run -n ms python -m pytest tests/...`
5. `rtk git diff`

Do not use RTK as a substitute for Serena. If the task depends on Python call relationships, class/method boundaries, symbol references, or precise code edits, use Serena after narrowing with `rg`/`rtk`.

## Verification

- Rewrite probe:
```bash
rtk rewrite "git status"
```
- Direct execution:
```bash
rtk git status
```
- Savings:
```bash
rtk gain --project
rtk gain --project --history
```
- Raw fallback for unsupported commands:
```bash
pwd
```
