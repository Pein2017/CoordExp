## Context

CoordExp’s `src/` surface spans dataset preprocessing/packing, training (Stage-1/Stage-2 AB + rollout matching), unified inference pipeline, evaluation, and visualization. The recent refactor modernized a number of boundaries, but a follow-up audit shows several “same concept, multiple implementations” cases remain:
- semantic description normalization + embedding (evaluator vs training gating/monitoring),
- coord-token detection (parallel regex/constants),
- geometry extraction/shape validation (datasets vs standardization vs token annotation),
- image-path resolution (infer engine/vis, evaluator overlays, preprocessors),
- JSONL load + diagnostics (generic loader vs evaluator-specific loader),
- module naming collisions that confuse ownership (e.g. `dataset_metrics` meaning different things).

The project constraints remain:
- Config-first (YAML), avoid new CLI flags.
- Preserve geometry invariants (never drop/reorder coords); do not change resize training policy (`do_resize=false`).
- Keep Qwen3-VL chat-template compatibility.
- Do not edit upstream HF model internals.

Data flow reference:
`JSONL/records -> preprocessing/transforms/packing -> training/infer runtime -> artifacts (gt_vs_pred.jsonl, metrics, overlays)`

## Goals / Non-Goals

**Goals:**
- Establish single canonical ownership for shared helpers that cross dataset/infer/eval surfaces.
- Eliminate near-duplicate implementations that can drift (semantic encoder, regex/token checks, IO/paths, geometry extraction).
- Reduce ambiguity by introducing canonical module names and leaving compatibility shims for legacy imports.
- Keep behavior stable and paper-ready: any strictness changes must be explicit, localized, and validated.

**Non-Goals:**
- No new model capabilities or bespoke RL loops.
- No changes to upstream HF model implementation files.
- No new CLI flags; any behavior configuration remains YAML-driven.
- No intentional changes to geometry semantics (ordering, meaning) or metric key naming.

## Decisions

### Decision 1: Centralize cross-surface helpers under `src/common/*`
We will place “shared across infer/eval/train/datasets” utilities under `src/common/*` where possible, and re-export from legacy locations to avoid churn.

Alternatives considered:
- Keep logic in the original surface module (e.g., evaluator-only). Rejected because drift risk is high and violates the “single source of truth” goal.

### Decision 2: Prefer compatibility shims over deletion
For modules that may have downstream import users (internal scripts/tests), we will keep a compatibility shim that re-exports canonical symbols. This keeps the refactor bisectable and reduces breakage.

Alternatives considered:
- Delete old modules and fix all imports in one pass. Rejected due to higher blast radius.

### Decision 3: Separate “strict” vs “best-effort” semantics explicitly
Some helpers intentionally differ across surfaces:
- visualization wants existence checks (skip missing images),
- inference/eval often wants best-effort path resolution and logs counters.

We will encode this as distinct shared helpers (e.g., strict vs best-effort image resolution; simple JSONL load vs diagnostic JSONL load) so behavior is explicit instead of implicit divergence.

### Decision 4: Keep geometry validation order-preserving and non-destructive
All shared geometry extraction/validation helpers MUST preserve point ordering and MUST not reorder or drop valid coordinates. Validation should only reject malformed records (wrong keys, odd arity, invalid bbox length).

### Decision 5: Resolve naming collisions by introducing canonical module names
Where two modules share a confusing name but represent different concerns, introduce a clearer canonical module name and convert the ambiguous file into a shim.

## Risks / Trade-offs

- [Risk] Subtle behavior changes in image-path resolution (ROOT_IMAGE_DIR precedence, existence checks). → Mitigation: introduce strict vs best-effort helpers; add targeted tests; keep callsites behavior-preserving.
- [Risk] Semantic encoder reuse changes pooling/normalization behavior. → Mitigation: consolidate onto a single implementation (mean pooling + L2 norm) and ensure evaluator + Stage-2 gating use the same code.
- [Risk] “Common” module becomes dependency-heavy (torch/transformers). → Mitigation: keep dependency-light helpers (`io`, `paths`) separate; isolate semantic encoder logic in a dedicated module and avoid importing it from hot paths unless configured/enabled.
- [Risk] Compatibility shims linger and reintroduce ambiguity. → Mitigation: clearly document canonical imports in module docstrings; optionally add a future lint/check to prevent new callsites from using legacy paths.
