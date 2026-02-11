## Why

CoordExp's `src/` surface has grown rapidly around Stage-2 AB, rollout matching, dataset encoding, and infer/eval orchestration, and the current shape increases correctness and reproducibility risk (fragile cross-module boundaries, duplicated pipelines, inconsistent diagnostics). This refactor is needed now to harden contracts before further capability work, while keeping training/eval semantics stable and paper-ready.

## What Changes

- Refactor `src/` using a seam-first program that prioritizes correctness-critical boundaries before structural decomposition.
- Harden Stage-2 and rollout trainer contracts by removing private cross-module coupling and defining explicit shared interfaces.
- Standardize failure semantics and diagnostics on critical runtime paths (fail-fast for invariants, scoped best-effort handling where explicitly intended).
- Unify duplicated dataset sample-to-encode flows and coordinate/geometry conversion helpers without changing geometry or chat-template behavior.
- Improve infer/eval robustness by consolidating JSONL diagnostics, reducing implicit env-coupling, and clarifying pipeline stage responsibilities.
- Decouple metrics components from trainer internals via neutral contracts to improve maintainability and testability.
- Keep changes config-first (YAML/schema aligned) and avoid introducing new CLI flags.
- Keep external behavior compatibility targets explicit: geometry invariants, Qwen3-VL template compatibility, `do_resize=false`, and existing artifact contracts.

## Capabilities

### New Capabilities
- None.

### Modified Capabilities
- `stage2-ab-training`: clarify/refine requirements around trainer boundary ownership, async feasibility/telemetry invariants, and critical failure handling.
- `rollout-matching-sft`: refine shared contract boundaries used by Stage-2, while preserving parsing/matching and post-rollout packing semantics.
- `trainer-metrics-components`: update requirement boundaries so metrics consume neutral contracts instead of trainer-internal coupling.
- `coord-utils`: align requirement-level ownership for shared coordinate conversion/validation helpers used across training and evaluation paths.
- `inference-pipeline`: refine stage resolution, override, and artifact-resolution requirements to reduce orchestration ambiguity.
- `inference-engine`: refine runtime robustness requirements (backend portability and diagnostics) without changing output contract.
- `detection-evaluator`: refine loader/diagnostic and helper-reuse requirements while keeping evaluation metric intent and artifact structure stable.

## Impact

- **Affected code**: `src/trainers/*`, `src/datasets/*`, `src/coord_tokens/*`, `src/common/*`, `src/infer/*`, `src/eval/*`, `src/metrics/*`, `src/config/*`, and corresponding tests/docs/configs.
- **APIs/configs**: No new CLI flags; behavior changes must be expressed through YAML/schema where needed, with compatibility constraints explicitly documented.
- **Correctness/reproducibility/eval validity**: expected to improve via explicit runtime contracts, consistent diagnostics, and reduced duplicate logic; parity checks and regression tests are part of the change scope.
- **Dependencies/systems**: no required platform shift; scope includes OpenSpec deltas, test expansion, and docs synchronization for affected capabilities.
