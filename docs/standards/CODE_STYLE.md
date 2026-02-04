# Code & Architecture Style (Transformers-Inspired, Option A)

This document captures a **medium-weight** set of style/architecture guidelines inspired by Hugging Face
Transformers (i.e. *mature, professional Python ML library* patterns), adapted to CoordExp’s
**research-grade + YAML-first** workflow.

Goals:
- Keep the repo reproducible and paper-ready.
- Keep imports light, avoid optional-dependency footguns.
- Make new features easy to locate, test, and document.
- Keep “core contracts” stable even as experiments evolve.

Non-goals:
- Enforcing a perfect, fully uniform style across all files.
- Large architectural rewrites (use `openspec/` governance for breaking/contract changes).

---

## 1) Structural principles (what to copy from Transformers)

### 1.1 Layered API surface
Keep a clear separation between:
- **Contracts / schemas** (stable): typed structures, IO formats, invariants.
- **Algorithms** (stable-ish): loss functions, decoding, matching, metrics.
- **Pipelines / entrypoints** (change-friendly): training loops, evaluation runs, CLI wrappers.

Transformers is effective because “core primitives” are stable and shared across models/tasks. In CoordExp:
- Put contracts in `src/common/` and `src/datasets/contracts.py`.
- Put reusable algorithmic building blocks in `src/coord_tokens/`, `src/metrics/`, `src/eval/`.
- Keep orchestration in `src/trainers/`, `src/infer/`, and `src/sft.py`.

### 1.2 Prefer small, composable base abstractions
When you need a “base class”, make it:
- Narrow in responsibility (e.g. decoding, schema validation, token typing).
- Easy to test without GPUs or huge checkpoints.
- Backward compatible in serialized formats and metric keys.

### 1.3 Explicit outputs over ad-hoc tuples
Transformers uses dataclass outputs (`ModelOutput` + `@dataclass` subclasses) to keep returns stable.

In CoordExp:
- Prefer returning a small `@dataclass` (or a typed dict) instead of a positional tuple.
- Document shapes/units (e.g. `norm1000` coords vs pixel coords).
- Keep optional fields defaulting to `None` (eases backwards compatibility).

---

## 2) Where code goes (CoordExp mapping)

Use these directories as “modules by responsibility”:

- `src/common/`: low-level shared contracts (schemas, IO helpers, coord standardization).
  - Avoid importing heavy training frameworks here.
  - Keep functions deterministic and dependency-light.
- `src/config/`: configuration schema + loader.
  - Prefer adding YAML knobs + schema validation here, not new CLI flags.
- `src/datasets/`: dataset builders, packing, augmentation, and geometry.
  - Geometry invariants live in `src/datasets/geometry.py` (do not reimplement elsewhere).
- `src/coord_tokens/`: coord vocab, codec, validators, and losses specific to coord tokens.
- `src/trainers/`: training loops and rollout-matching orchestration.
  - Keep “algorithm” pieces in reusable modules; trainers should mainly wire them together.
- `src/infer/`: inference pipeline utilities and visualization.
- `src/eval/`: evaluation parsing + metrics computation.
- `src/utils/`: cross-cutting utilities (logging, optional-dependency helpers, env detection).

If you are unsure, choose the simplest rule:
> Put code where a new contributor would look for it first.

---

## 3) Module and symbol naming conventions

### 3.1 File/module naming
Follow Transformers’ “role-first” naming:
- `*_utils.py` for shared utilities.
- `contracts.py`, `schemas.py` for stable IO / typed structures.
- `codec.py`, `validator.py`, `parser.py` for format transformations.
- Prefer **snake_case** for filenames and modules.

### 3.2 Class/function naming
- Classes: `PascalCase`
- Functions/vars: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Mixins (if used): suffix `Mixin` and keep them small.

### 3.3 Public vs private
- Use `__all__` in modules that intentionally define a public surface.
- Prefix private helpers with `_` and keep them close to their usage.
- Keep `src/__init__.py` thin (avoid heavy imports at import-time).

---

## 4) Optional dependencies (avoid import-time pain)

Transformers treats backends as optional and provides *actionable* errors.

In CoordExp:
- Do not import heavyweight optional deps at module import time when avoidable (e.g., `vllm`, `opencv`, large HF backends).
- Centralize availability checks in `src/utils/` (e.g., `is_vllm_available()`), and gate usage inside functions.
- When failing, raise errors that tell the user what to install and why.

Recommended error message style:
- Include the missing package name(s).
- Include the feature they tried to use.
- Provide a next action (“install X” / “use config Y instead”).

---

## 5) Logging and warnings

Transformers uses a centralized logging wrapper and “warn once” patterns to reduce noise.

In CoordExp:
- Prefer a repo-wide logger helper (see `src/utils/logger.py`) rather than configuring logging ad-hoc.
- Use warnings sparingly; a warning should be:
  - actionable,
  - non-spammy,
  - and never hide silent correctness changes.
- For deprecations, prefer:
  - a single warning per process (or per call site),
  - a concrete timeline (“deprecated now; remove after <date>/<version>”), and
  - a migration path (which YAML knob replaces it).

---

## 6) Documentation style

### 6.1 Docstrings
Keep docstrings “docs-grade” for public entrypoints and contracts:
- First line: what the function/class does.
- Then: important invariants (especially around geometry and coord normalization).
- Then: arguments/returns in simple Markdown style.

Do not over-comment obvious code; instead:
- document invariants,
- document non-obvious tradeoffs,
- and link to the relevant `docs/` page when the explanation is long.

### 6.2 Docs-as-source-of-truth
Stable behavior belongs in `docs/` (not in ad-hoc comments or chat history):
- Data contracts: `docs/data/`
- Training runbooks: `docs/training/`
- Repo standards: `docs/standards/`

If you add a new user-facing feature, add a doc entry and link it from `docs/README.md`.

---

## 7) Testing style

CoordExp tests should aim to be:
- deterministic (seeded, stable ordering),
- contract-focused (geometry, JSONL schema, packing invariants),
- runnable from repo root without special cwd assumptions.

Patterns to copy from mature libraries:
- Use `tests/conftest.py` for environment bootstrapping (path setup, dependency checks).
- Prefer small unit tests for pure functions (geometry, codec, parsing).
- Add at least one “smoke” test for any new end-to-end pipeline component.
- Gate truly expensive tests behind an explicit opt-in (env var / marker).

---

## 8) Config style (YAML-first)

Treat configs as first-class artifacts:
- Prefer adding knobs to YAML + `src/config/schema.py` over adding CLI flags.
- Keep experiment names descriptive and reproducible (dataset + model + key knobs + seed).
- Keep `prompts:` empty unless the code explicitly supports it (prompt changes should live in code, not in YAML).

When adding a new knob:
1) Add it to the schema with validation and sensible defaults.
2) Document it (short note in the relevant runbook, or a new doc if it’s a new capability).
3) Add/adjust a test if it affects contracts or output formats.

---

## 9) Hard guardrails (do not violate)

These are correctness/compatibility constraints:
- Preserve geometry: never drop/reorder coords; use `src/datasets/geometry.py`.
- Maintain Qwen3-VL chat-template compatibility.
- Do not edit upstream HF model internals (off-limits files).

If you need to change a contract or introduce a breaking change, follow `openspec/` governance.

