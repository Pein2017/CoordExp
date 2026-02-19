## Context

CoordExp currently contains `try/except` blocks in core training/inference/eval paths that can silently swallow unexpected exceptions (e.g., `except Exception: pass`, `except Exception: continue`, or returning “safe” numeric defaults). This can produce partial artifacts and/or silently corrupt supervision and metrics, making runs non-auditable and undermining reproducibility.

This change enforces a strict-by-default error-handling policy:
- unexpected exceptions fail fast (terminate the run),
- only explicitly-enumerated, sample-scoped validation/parse errors may be handled (and must be observable),
- best-effort behavior is constrained to narrowly-defined I/O sinks and must not alter correctness-affecting state.

Constraints:
- Config-first: no new ad-hoc CLI flags and no exception suppression registries/allowlists.
- Preserve geometry invariants: never drop/reorder coords; continue using `src/datasets/geometry.py` utilities.
- Keep Qwen3-VL chat-template compatibility; do not modify upstream HF model internals.
- Avoid over-design: prefer minimal local refactors and contract-focused tests over a new exception-policy framework.

## Goals / Non-Goals

**Goals**
- Remove silent swallowing of unexpected exceptions in core `src/` execution paths.
- Make intentionally-handled per-sample validation/parse failures observable:
  - structured per-sample `errors` entries (where applicable),
  - run-level counters/metrics for error classes.
- Replace blanket exception handlers with narrow, enumerated exception handling or fail-fast re-raise.
- Tighten optional dependency handling to `ImportError` / `ModuleNotFoundError` with actionable messages.
- **Strip over-engineering:** remove redundant `try/except` wrappers that only re-raise without adding context, and remove semantics-changing “safe defaults” used to hide failures.

**Non-Goals**
- No global exception policy registry, allowlist, or “suppression framework”.
- No new user-facing CLI flags.
- No best-effort fallbacks in correctness-affecting logic (parsing, matching, metrics, model-state updates).

## Decisions

1) **Taxonomy: expected per-sample errors vs unexpected internal exceptions**
- Expected per-sample errors:
  - sample-scoped validation/parse failures that do not indicate a code bug,
  - may be handled only when explicitly enumerated,
  - MUST be observable (structured `errors` + counters),
  - MUST NOT be “fixed” by substituting semantics-changing defaults.
- Unexpected internal exceptions:
  - anything not explicitly treated as an expected per-sample error,
  - MUST terminate the run (fail fast).

2) **Training vs inference/eval strictness**
- Training:
  - deterministic inputs (dataset encoding, cooked targets, GT) treat validation/parse errors as contract violations and MUST fail fast,
  - explicitly salvage-mode training subpaths that consume model-generated outputs (e.g., rollout parsing/matching) MAY drop/skip invalid model outputs *per sample*, but MUST be observable (structured errors + counters) and MUST NOT suppress unexpected internal exceptions.
- Inference/eval:
  - expected per-sample validation/parse errors MAY be recorded and skipped per sample,
  - unexpected internal exceptions MUST fail fast.

3) **Observability is required when continuing**
When the system continues past an expected per-sample error, it MUST:
- record a structured error entry on that sample (where a per-sample artifact exists),
- increment a run-level counter/metric for that error class,
- avoid producing “fake success” outputs (e.g., emitting empty predictions without an error record).

4) **Best-effort is limited to non-correctness sinks**
Best-effort exception handling is allowed only for sinks that cannot affect correctness-affecting state, such as log tee mirroring and diagnostics/telemetry reporting. These handlers MUST:
- catch narrow, expected exception classes when possible (e.g., `OSError` for file I/O),
- emit explicit diagnostics (at least once; rate-limited is OK),
- never suppress exceptions outside the sink scope itself.

5) **Prefer elimination over wrapping**
Where a `try/except` block only re-raises the same exception without adding actionable context, remove it. Where context is needed, add it once at a meaningful boundary using `raise ... from e`, not via repeated wrapper layers.

## Risks / Trade-offs

- **[Risk] Previously-masked issues now abort runs** → Mitigation: keep changes narrow, add actionable error messages, and improve observability of expected per-sample errors.
- **[Risk] CI scanning becomes brittle** → Mitigation: enforce only high-signal patterns (broad exception handlers that suppress/continue/return defaults), using AST-based checks rather than regex when feasible.
- **[Risk] Partial artifact expectations in tooling** → Mitigation: clarify which artifacts are allowed to be partial (only when explicitly scoped to per-sample expected errors) and otherwise fail fast.
