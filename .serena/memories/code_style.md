# CoordExp Code Style & Conventions

- Follow Transformers-inspired Option A (docs/standards/CODE_STYLE.md) with layered APIs: separate contracts/schemas, algorithms, and orchestration (training/eval entrypoints).
- Prefer small composable abstractions, explicit dataclass outputs, and narrow responsibilities (keep shared contracts in `src/common/`, dataset logic in `src/datasets/`, etc.).
- Snake case modules/functions, PascalCase classes, and UPPER_SNAKE_CASE constants; use `__all__` to define public surfaces and `_` prefix for private helpers.
- Avoid import-time optional dependencies; gate heavy imports with availability checks in `src/utils/` and raise actionable errors when missing packages.
- Logging via centralized helper (see `src/utils/logger.py`); warn sparingly with concrete actions and migration guidance.
- Documentation standards: docstrings must state invariants, arguments, and link to docs when necessary; prefer docs/ entries over ad-hoc comments.
- Testing style: deterministic, contract-focused, small unit tests for geometry/packing, smoke tests for pipeline components, and expensive tests behind opt-in flags.
- Config-first approach: strict schema validation (`src/config/schema.py`, `strict_dataclass`, `rollout_matching_schema`), no new CLI flags, prompt definitions reserved for code, unknown keys fail fast, use `custom.extra` for permitted leftovers.
