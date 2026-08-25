# RTK - Rust Token Killer (Codex CLI)

**Usage**: Token-optimized CLI proxy for shell commands.

## Rule

Always prefix shell commands with `rtk`.

Examples:

```bash
rtk git status
rtk cargo test
rtk npm run build
rtk test python -m pytest -q
```

The local Codex PreToolUse hook also rewrites common shell commands and RTK-
supported compound commands through RTK automatically. If exact stdout,
shell syntax, or machine-readable output matters, prefer the raw command with
`RTK_HOOK_DISABLE=1` or use `rtk proxy`.

## CoordExp Caveats

- The local rewrite hook routes pytest through
  `rtk test <original pytest command>`, not `rtk pytest`. RTK 0.43's
  specialized pytest parser can misreport a successful quiet run as
  `No tests collected` when the normal pytest summary is absent.
- Use `rtk proxy python -m pytest ...` when exact pytest output matters, such
  as diagnosis, collection, or version checks.
- Use `RTK_HOOK_DISABLE=1 <command>` for exact Git status/diff output,
  structured JSON/YAML, NUL-delimited output, or a pipeline whose downstream
  consumer requires the producer's raw bytes.
- Shell expansion is semantic, not cosmetic. The hook preserves direct-command
  glob/parameter spelling and leaves prefixed commands such as `conda run ...
  tests/*.py` raw when safe insertion would require re-quoting the expansion.
- Independent multiline command lists are rewritten line by line while
  preserving their newlines. Continuations, control-flow blocks, variable or
  command substitutions, and here-doc style syntax remain raw; prefix each
  line with `rtk` explicitly when exact control is needed.
- Do not rely on `rtk pytest --version`; upstream currently collapses that
  informational output into the pytest summary path. The hook's generic
  `rtk test` route avoids that parser but still filters output.
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
