# Repo Lifecycle

`repo_lifecycle/` contains lightweight governance helpers for the CoordExp
refactoring program. It is intentionally separate from both `src/` and
`scripts/`:

- it is not runtime library code,
- it is not an experiment, training, inference, eval, or analysis entrypoint,
- it should stay small enough to make repo lifecycle state easier to inspect,
  not become another tool pile.

Current command:

```bash
python -m repo_lifecycle.report_lifecycle_registry
```

The default mode is report-only. Use `--strict` only after the initial
classification warnings are intentionally resolved.
