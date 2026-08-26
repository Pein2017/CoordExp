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

- On RTK 0.45, normal pytest uses `rtk pytest`. The local hook keeps only
  `pytest --collect-only` on `rtk test`, because the specialized parser still
  reports `No tests collected` for a successful collection run.
- Use `rtk proxy <pytest command>` when exact pytest output matters, such as
  diagnosis, collection, or version checks.
- Use `RTK_HOOK_DISABLE=1 <command>` for exact Git name/status/diff output,
  structured JSON/YAML, NUL-delimited output, or a pipeline whose downstream
  consumer requires the producer's raw bytes.
- Shell expansion is semantic, not cosmetic. The hook preserves direct-command
  glob/parameter spelling and leaves prefixed commands such as `conda run ...
  tests/*.py` raw when safe insertion would require re-quoting the expansion.
- Independent multiline command lists are rewritten line by line while
  preserving their newlines. Continuations, control-flow blocks, variable or
  command substitutions, and here-doc style syntax remain raw; prefix each
  line with `rtk` explicitly when exact control is needed.
- `rtk pytest --version` is safe on RTK 0.45; older-version caveats do not
  apply to the current installation.
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
