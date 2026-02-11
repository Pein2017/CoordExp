## Context

CoordExp currently carries high-value training/eval semantics in a small number of large, tightly coupled modules across `src/trainers/`, `src/datasets/`, `src/infer/`, and `src/eval/`. This has enabled rapid iteration, but it now increases correctness and reproducibility risk through boundary leakage, duplicate pipelines, and inconsistent failure/diagnostic behavior.

This change is cross-cutting and contract-sensitive:
- Stage-2 AB and rollout-matching interfaces are shared across training variants and configs.
- Data and geometry handling must remain invariant across dataset, augmentation, inference, and evaluation paths.
- Inference/evaluation artifacts (`gt_vs_pred.jsonl`, metric outputs) must remain stable for paper-ready comparisons.

Primary hard constraints:
- YAML-first workflow; avoid new CLI flags.
- Preserve geometry semantics (never drop/reorder coords).
- Preserve Qwen3-VL chat-template compatibility.
- Keep training policy compatibility (`do_resize=false`).
- Upstream HF model internals remain off-limits.

Data flow for this refactor scope:

`JSONL/records -> preprocessing/transforms/packing -> trainer/infer runtime orchestration -> outputs/artifacts (logs, checkpoints, gt_vs_pred, metrics)`

The design intentionally focuses on architecture and migration strategy, not line-by-line implementation.

## Goals / Non-Goals

**Goals:**
- Isolate correctness-critical contracts into explicit, testable module boundaries.
- Reduce duplicate logic in dataset encoding and geometry/token conversion paths.
- Standardize critical-path failure semantics and diagnostics.
- Preserve output and metric contract parity while improving maintainability.
- Keep OpenSpec capability updates aligned with behavior-level requirement changes.

**Non-Goals:**
- No new model architecture families or bespoke RL framework shifts.
- No upstream HF core model file modifications.
- No intentional behavior expansion beyond declared capability deltas.
- No broad CLI surface growth; config-first remains the mechanism for behavior expression.

## Decisions

### Decision 1: Use seam-first incremental refactor sequencing
We will execute a staged sequence that prioritizes contract hardening before structural decomposition:
1. Boundary + failure semantics hardening.
2. Data pipeline consolidation.
3. Trainer/entrypoint decomposition.
4. Infer/eval modernization.
5. Metrics contract inversion + quality gate hardening.

Rationale:
- Minimizes regression risk by preserving behavior with compatibility shims while seams are extracted.
- Keeps parity checks meaningful and localized at each stage.

Alternative considered:
- Big-bang rewrite of large modules.
- Rejected due to high blast radius and weak bisectability for semantic regressions.

### Decision 2: Introduce explicit public contracts for Stage-2/rollout sharing
Shared logic currently crossing trainer boundaries will move to explicit public interfaces/modules, with temporary compatibility exports during migration.

Rationale:
- Removes fragile underscore-private coupling.
- Enables independent evolution of Stage-2 orchestration and rollout algorithms.

Alternative considered:
- Keep cross-imports and rely on convention.
- Rejected because conventions are insufficient for long-lived contract-critical code.

### Decision 3: Define and enforce critical-path exception taxonomy
Runtime paths will classify handling into:
- Critical invariant failures: fail-fast with actionable context.
- Best-effort diagnostics/telemetry: isolated catch-and-log behavior.

Rationale:
- Improves debuggability and trust in observed metrics/artifacts.
- Prevents silent suppression of correctness failures.

Alternative considered:
- Keep broad defensive catches in place.
- Rejected because it obscures true failure domains and complicates reproducibility triage.

### Decision 4: Consolidate dataset encode flow and geometry/token helper ownership
Dense and fusion dataset sample-to-encode paths will converge on shared helpers; coordinate/geometry conversion and validation helpers will be centralized for reuse across training and evaluation consumers.

Rationale:
- Eliminates drift between parallel code paths.
- Strengthens invariant enforcement consistency.

Alternative considered:
- Retain duplication with stronger review policy.
- Rejected because repeated logic already diverged and review burden scales poorly.

### Decision 5: Keep infer/eval artifact contracts stable while separating orchestration concerns
Infer/eval refactor will separate config parsing/resolution from runtime execution, and unify loader/diagnostic behavior without changing declared artifact semantics.

Rationale:
- Preserves benchmark continuity.
- Improves portability and operator-facing diagnostics.

Alternative considered:
- Refactor infer/eval and artifact semantics simultaneously.
- Rejected due to coupled risk; contract changes should be isolated and explicitly spec-governed.

### Decision 6: Invert metrics dependency direction
Metrics computation should consume neutral contracts/events rather than trainer internals.

Rationale:
- Reduces coupling and enables testable metrics components.
- Makes trainer evolution less fragile.

Alternative considered:
- Keep runtime dynamic imports from metrics into trainer internals.
- Rejected because it hides coupling and impedes local reasoning/testing.

## Risks / Trade-offs

- [Risk] Compatibility shims linger and create dual-path complexity. -> Mitigation: enforce milestone exit criteria that remove shims once downstream references are migrated.
- [Risk] Parity validation misses subtle behavior shifts under distributed/async conditions. -> Mitigation: add targeted async/queue/version-window regression tests and run scoped smoke checks with pinned configs.
- [Risk] Cross-module extraction introduces temporary naming churn and cognitive overhead. -> Mitigation: publish stable module maps and keep migration in small reviewable increments.
- [Risk] Consolidating geometry/token helpers may surface latent data-quality errors earlier. -> Mitigation: stage strictness rollout via diagnostics-first checks and explicit error messaging.
- [Risk] Large-scope refactor may stall if treated as one deliverable. -> Mitigation: gate by capability-aligned increments with measurable completion criteria per stage.

## Migration Plan

1. Establish baseline parity fixtures and validation command set for critical flows.
2. Extract Stage-2/rollout public contracts and enforce import boundary changes.
3. Apply exception taxonomy in critical runtime paths.
4. Consolidate dataset encode and geometry/token helper ownership.
5. Refactor trainer and entrypoint orchestration into composable components.
6. Refactor infer/eval internals with contract-preserving artifact outputs.
7. Invert metrics dependency direction and finalize test/lint/type quality gates.
8. Remove temporary compatibility shims after all callsites migrate and tests pass.

Rollback strategy:
- Each stage lands as isolated commits; revert at stage granularity if parity checks fail.
- Keep legacy compatibility paths until stage exit gates are satisfied.

## Open Questions

- Should infer/eval loader strictness defaults be standardized globally now, or staged per capability to reduce adoption friction?
- How much Stage-2 telemetry key compatibility must be preserved verbatim vs alias-supported during migration?
- Should schema modularization happen before or after trainer decomposition, given current coupling between config validation and runtime wiring?
- What minimal smoke matrix is required to declare parity for async server-mode workflows in CI vs manual validation?
