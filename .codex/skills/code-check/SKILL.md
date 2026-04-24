---
name: code-check
description: "Use when running fail-fast Python static checks with ruff and a pyright-compatible type checker on explicit paths, especially before claiming CoordExp code changes are clean."
---

# Python Code Check (fail-fast static checks)

This skill is intentionally narrow: it runs the *standard* baseline checks that should be true before deeper review:
- `ruff format` (formatter)
- `ruff check` (lint)
- `pyright`/`basedpyright` (type checking)

## Mandatory gate (always)

Do **not** consider code-check complete unless a type checker was executed.

- Required: run `basedpyright` when available, otherwise `pyright`
- Not allowed: `ruff`-only validation
- Finalization rule: if type checking was skipped or failed, report that explicitly and do not claim static checks passed

It does **not** attempt schema/architecture review.
It does **not** replace targeted behavioral tests from `docs/IMPLEMENTATION_MAP.md`.

## 0) Pick a scope (mandatory)

Always pass explicit paths. Prefer directories for reproducibility.

- Good: `src`, `tests`, `.`
- Good: `src/foo/bar.py`
- Avoid: `*.py` (breaks when the working directory changes; can silently target the wrong files)

## 1) Run standard checks directly (recommended)

Run these from the repo root (`.`) or any directory; just pass paths that exist in the repo.
For noisy runs in CoordExp, `rtk conda run -n ms ...` is acceptable when filtered output is enough; keep the `conda run -n ms` environment wrapper either way.

**Formatter drift (no writes):**
```bash
conda run -n ms ruff format --check src tests
```

**Lint:**
```bash
conda run -n ms ruff check src tests
```

**Type check (prefer basedpyright when available):**
```bash
conda run -n ms basedpyright -p pyrightconfig.json
# or:
conda run -n ms pyright -p pyrightconfig.json
```

## 2) If you need changed-files-only checks

Prefer directory-based runs (they're simplest and most reproducible). If you must target only changed Python files, generate an explicit file list and pass those paths (no globs).

Example (compare against `origin/main`):
```bash
git diff --name-only --diff-filter=ACMRT origin/main...HEAD | rg '\.py$' > /tmp/py_files.txt

cat /tmp/py_files.txt | xargs -r conda run -n ms ruff format --check
cat /tmp/py_files.txt | xargs -r conda run -n ms ruff check
```

Type check is still mandatory in changed-files mode. Run at least one of:
```bash
conda run -n ms basedpyright -p pyrightconfig.json
# or:
conda run -n ms pyright -p pyrightconfig.json
```

## 3) CoordExp follow-up

After static checks, run the smallest relevant tests named by `docs/IMPLEMENTATION_MAP.md` for the touched area, for example data/geometry, Stage-1, Stage-2, infer/eval, or artifact/provenance tests.

When reporting results, keep static checks separate from behavioral verification:

- "ruff and basedpyright passed" means static checks passed
- "targeted tests passed" requires a separate fresh test command
- if type checking is skipped or too broad/expensive to finish, say that explicitly
