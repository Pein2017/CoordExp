## Why

CoordExp’s `src/` tree has accumulated multiple “near-duplicate” helpers (semantic-desc encoding, coord-token detection, geometry extraction/validation, JSONL loading diagnostics, image-path resolution). These overlaps increase correctness and reproducibility risk because parallel implementations drift, and they create architectural ambiguity about canonical ownership.

This change consolidates redundant modules and clarifies ownership boundaries so training, inference, evaluation, visualization, and dataset preprocessing share the same contracts and numeric/IO fences.

## What Changes

- Consolidate semantic-description normalization + embedding encoder logic into a single canonical helper and remove duplicate implementations from the evaluator.
- Canonicalize coord-token regex/helpers so there is one source of truth for `<|coord_k|>` detection/encode/decode; keep compatibility aliases where needed.
- Consolidate single-geometry extraction/shape validation for `bbox_2d|poly` into a shared helper used by datasets, standardization, and coord-token annotation paths.
- Reduce naming ambiguity by introducing canonical module names and leaving compatibility shims for legacy import paths (e.g., collator “dataset metrics” vs trainer mixins).
- Centralize image-path resolution helpers so inference engine, vis, evaluator overlays, and dataset preprocessing resolve relative paths consistently (while preserving each surface’s strict/best-effort behavior).
- Centralize JSONL loading with evaluation-grade diagnostics (path + 1-based line, clipped snippet) so evaluator-specific loaders don’t duplicate parsing/warning behavior.
- Update callsites and add targeted tests where feasible (dependency-light; no network/model downloads).
- For overlapping active deltas, this change is the authoritative helper-contract source; other deltas (for example `refactor-src-modernization`) reference these helper contracts for integration/migration only.

## Capabilities

### New Capabilities
- None.

### Modified Capabilities
- `coord-utils`: clarify canonical coord-token helpers and shared geometry extraction/validation reuse across consumers.
- `detection-evaluator`: remove duplicated semantic-desc encoder logic and reuse canonical JSONL/image-path helpers while preserving evaluation artifact/schema behavior.
- `inference-pipeline`: reuse canonical image-path resolution helpers across infer stages (engine/vis) while preserving output contracts.
- `vl-token-type-metrics`: ensure coord-token detection logic is sourced from the canonical coord-token helper (no parallel regex).

## Impact

- **Affected code**: `src/common/*`, `src/coord_tokens/*`, `src/datasets/*`, `src/data_collators/*`, `src/infer/*`, `src/eval/*`, and small compatibility surfaces under `src/metrics/*`.
- **Behavior**: intended to be behavior-preserving; changes focus on eliminating duplicate logic and making ownership explicit. Any strictness differences must be explicitly documented in specs/tasks and validated with focused tests.
- **APIs/imports**: some internal module import paths will become canonical; legacy paths remain via compatibility shims to reduce breakage.
- **Dependencies/systems**: no new dependencies; avoid new CLI flags; keep YAML-first workflow and existing artifact contracts stable.
