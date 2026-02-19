## Why

CoordExp currently contains try/except blocks in core training/inference/eval paths that can silently swallow unexpected exceptions (e.g., defaulting to empty outputs or “safe” numeric fallbacks). This makes runs non-auditable and can silently corrupt supervision, metrics, or artifacts; we want strict-by-default, fail-fast behavior so failures are observable and reproducibility is preserved.

## What Changes

### Definitions (normative)

- **Core execution paths**: dataset encoding, trainer steps, inference pipeline stages, and evaluation metric computation (i.e., correctness-affecting logic under `src/`).
- **Expected per-sample errors**: sample-scoped validation/parse failures that do not indicate a code bug. These MAY be recorded and skipped *per sample* (never by substituting “safe” defaults) in inference/eval and in explicitly salvage-mode training subpaths that consume model-generated outputs (e.g., rollout parsing/matching). For deterministic training inputs (dataset encoding / cooked targets / GT), such errors MUST fail fast.
- **Unexpected internal exceptions**: anything not explicitly treated as an expected per-sample error; MUST terminate the run (fail fast).
- **Observable**: recorded via structured per-sample `errors` and run-level counters; warnings may be rate-limited but are not sufficient alone.

- Enforce **strict-by-default** exception handling in core `src/` execution paths:
  - unexpected exceptions MUST fail fast (raise),
  - expected, enumerated validation/parse errors MAY be handled, but MUST be observable (structured error codes + run-level counters; warnings may be bounded but are not sufficient alone).
- Remove/replace any blanket exception handlers in core paths that continue/return defaults without observability (counters + structured errors) or without re-raising.
- Tighten optional-dependency handling by catching `ImportError`/`ModuleNotFoundError` (not blanket `Exception`) and raising actionable guidance.
- Constrain best-effort handling to explicitly-justified I/O sinks (e.g., log tee writes), not to artifact parsing, matching, metrics, or model-state updates.
- **Strip over-engineering:** remove redundant try/except wrappers that only re-raise without adding context, and remove “defensive” fallback branches that change semantics silently (prefer explicit failure with guidance).
- Add minimal, contract-focused tests/CI checks that prevent regressions of silent swallowing without introducing a complex exception-policy framework.
- Non-goals: no exception suppression registries/allowlists and no new CLI flags.

**BREAKING**: Unexpected internal exceptions that were previously masked (continue/return defaults/partial artifacts) will now terminate the run (non-zero exit) to prevent silent corruption. Best-effort behavior remains limited to explicitly-justified I/O sinks.

## Capabilities

### New Capabilities
- (none)

### Modified Capabilities
- `silent-failure-policy`: expand the definition of “silent swallowing” beyond `except Exception: pass` to include defaulting/continuing/returning in core paths without observability, and clarify strict-by-default behavior vs narrow best-effort I/O sinks.
- `inference-engine`: clarify the boundary between (a) expected per-sample validation/parse errors that should be recorded as sample-scoped `errors` and counters, and (b) unexpected internal exceptions that must terminate the run (fail fast).

## Impact

- Affected code: core error-handling blocks across `src/common/`, `src/infer/`, `src/eval/`, `src/trainers/`, and targeted test coverage under `tests/`.
- Affected behavior: fewer “quiet” partial successes; failures become explicit and earlier. No artifact key renames are intended, but unexpected-exception cases will now stop the run instead of silently degrading outputs.
- Dependencies: no new required dependencies; optional-dep errors become narrower and more actionable.
