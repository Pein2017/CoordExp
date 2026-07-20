# RTK - Rust Token Killer (Codex CLI)

**Usage**: Token-optimized CLI proxy for shell commands.

## Rule

Always prefix shell commands with `rtk`.

Examples:

```bash
rtk git status
rtk cargo test
rtk npm run build
rtk pytest -q
```

The local Codex PreToolUse hook also rewrites common shell commands and RTK-
supported compound commands through RTK automatically. If exact stdout,
shell syntax, or machine-readable output matters, prefer the raw command with
`RTK_HOOK_DISABLE=1` or use `rtk proxy`.

## CoordExp Caveats

- Use `rtk proxy python -m pytest ...` for pytest diagnosis, collection,
  version checks, or ambiguous "No tests collected" output. `rtk pytest` may
  summarize zero selected or informational pytest runs too aggressively.
- Use `RTK_HOOK_DISABLE=1 <command>` for exact Git status/diff output,
  structured JSON/YAML, NUL-delimited output, or a pipeline whose downstream
  consumer requires the producer's raw bytes.
- Do not rely on `rtk pytest --version`; upstream currently collapses that
  informational output into the pytest summary path.
- If RTK output is surprising, re-run once with `rtk proxy <command>` before
  changing code or tests.

## Meta Commands

```bash
rtk gain            # Token savings analytics
rtk gain --history  # Recent command savings history
rtk proxy <cmd>     # Run raw command without filtering
```

## Verification

```bash
rtk --version
rtk gain
which rtk
```
