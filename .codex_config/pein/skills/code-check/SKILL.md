---
name: code-check
description: "Fail-fast Python code check: run ruff (format + lint) and a pyright-compatible type checker on explicit paths. Avoid shell globs like `*.py`; prefer directories (src/tests/.) or explicit files so runs are reproducible from any working directory."
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

## 0) Pick a scope (mandatory)

Always pass explicit paths. Prefer directories for reproducibility.

- Good: `src`, `tests`, `.`
- Good: `src/foo/bar.py`
- Avoid: `*.py` (breaks when the working directory changes; can silently target the wrong files)

## 1) Run standard checks directly (recommended)

Run these from the repo root (`/data/CoordExp`) or any directory; just pass paths that exist in the repo.

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
