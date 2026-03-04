## Why

CoordExp currently contains try/except blocks in core training/inference/eval paths that can silently swallow unexpected exceptions (e.g., defaulting to empty outputs or “safe” numeric fallbacks). This makes runs non-auditable and can silently corrupt supervision, metrics, or artifacts; we want strict-by-default, fail-fast behavior so failures are observable and reproducibility is preserved.

## What Changes

### Definitions (normative)

- **Core execution paths**: dataset encoding, trainer steps, inference pipeline stages, and evaluation metric computation (i.e., correctness-affecting logic under `src/`).
- **Operator-controlled inputs**: training inputs (dataset encoding, cooked targets, GT) and inference/eval inputs (JSONL + images + required metadata) that are deterministic and can be validated in advance.
- **Resolvable sample-scoped violations**: sample-scoped validation/parse/contract failures for operator-controlled inputs (invalid JSONL line, wrong schema, missing/corrupt image, malformed geometry, missing width/height, wrong format, etc.). These MUST fail fast (raise and terminate the run) so operators fix data/contracts ahead of compute.
- **Model-generated output invalidity (explicit model-output consumers only)**: invalid/truncated/partial model outputs produced during codepaths that explicitly consume model-generated text (e.g., inference prediction parsing/validation and salvage-mode training subpaths like rollout parsing/matching). Only these contexts MAY drop/skip invalid model outputs *per sample* (never by substituting “safe” defaults), and MUST be observable (structured errors + counters).
- **Unexpected internal exceptions**: anything not explicitly treated as model-generated output invalidity in an explicit model-output consumer; MUST terminate the run (fail fast).
- **Observable**: recorded via structured per-sample `errors` and run-level counters; warnings may be rate-limited but are not sufficient alone.

- Enforce **strict-by-default** exception handling in core `src/` execution paths:
  - unexpected internal exceptions MUST fail fast (raise),
  - operator-controlled input violations MUST fail fast (raise; no skip-and-continue),
  - only explicitly model-output consumer subpaths MAY continue past invalid model outputs, and MUST be observable (structured error codes + run-level counters; warnings may be bounded but are not sufficient alone).
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
- `inference-engine`: clarify that operator-controlled input violations MUST terminate inference/eval (no skip-and-continue), while model-output/prediction parse+validation failures are continue-but-observable; unexpected internal exceptions and generation failures terminate (fail fast).

## Impact

- Affected code: core error-handling blocks across `src/common/`, `src/infer/`, `src/eval/`, `src/trainers/`, and targeted test coverage under `tests/`.
- Affected behavior: fewer “quiet” partial successes; failures become explicit and earlier. No artifact key renames are intended, but unexpected-exception cases will now stop the run instead of silently degrading outputs.
- Dependencies: no new required dependencies; optional-dep errors become narrower and more actionable.
