## 1. Shared IO/Path Helpers

- [x] 1.1 Add canonical image-path resolution helpers under `src/common/paths.py` (strict vs best-effort) and keep behavior-preserving wrappers at callsites.
- [x] 1.2 Add a shared JSONL diagnostic loader under `src/common/io.py` (path + 1-based line + clipped snippet; strict vs non-strict).

## 2. Semantic-Desc Consolidation

- [x] 2.1 Move semantic description normalization + embedding encoder into a single canonical module and keep `src/metrics/semantic_desc.py` as compatibility surface if needed.
- [x] 2.2 Remove duplicated semantic encoder implementation from `src/eval/detection.py` by delegating to the canonical helper; preserve mapping/drop behavior and error messaging.

## 3. Coord-Token Helper Canonicalization

- [x] 3.1 Remove parallel coord-token regex/constants by routing `src/common/geometry/coord_utils.py` through `src/coord_tokens/codec.py` (keep compatibility aliases like `COORD_TOKEN_RE`).
- [x] 3.2 Ensure token-type telemetry uses the canonical coord-token helper (no parallel regex in token-type code).

## 4. Shared Geometry Extraction/Validation Helper

- [x] 4.1 Introduce a shared single-geometry extraction/shape validation helper under `src/common/geometry/` and document its invariants (bbox len=4, poly even and >=6, non-destructive).
- [x] 4.2 Update dataset geometry helpers (`src/datasets/utils.py`) and coord standardization/token annotation paths to reuse the shared helper (preserve ordering; do not drop/reorder coords).

## 5. Naming Ambiguity Cleanup (Canonical Modules + Shims)

- [x] 5.1 Introduce a canonical name for the batch-extras collator wrapper (currently `src/data_collators/dataset_metrics.py`) and keep the old module as a shim re-export.
- [x] 5.2 Introduce a canonical name/location for trainer-side “metrics mixins” (currently `src/metrics/dataset_metrics.py`) and keep the old module as a shim re-export.

## 6. Adopt Helpers in Infer/Eval/Datasets

- [x] 6.1 Route image-path resolution in infer engine/vis/evaluator overlays and dataset resize preprocessing through the shared `src/common/paths.py` helper (preserve strictness per surface).
- [x] 6.2 Route evaluator JSONL ingestion through the shared JSONL diagnostic loader helper (preserve strict vs non-strict behavior and counters).

## 7. Validation

- [x] 7.1 Add targeted unit tests for new shared helpers (dependency-light; avoid model downloads) and update any existing tests affected by import moves.
- [x] 7.2 Run focused validation: `PYTHONPATH=. conda run -n ms python -m py_compile <touched files>`, `PYTHONPATH=. conda run -n ms python -m pytest -q <target tests>`, and `openspec validate src-ambiguity-cleanup-2026-02-11 --strict`.

## 8. Reopened Contract Clarifications (2026-02-11)

- [x] 8.1 Pin this change as the authoritative helper-contract delta and sync cross-references from overlapping active changes.
- [ ] 8.2 Make evaluator strict-parse behavior fully explicit in specs (`eval.strict_parse`, default false, `warn_limit=5`, `max_snippet_len=200`) and keep implementation/tests aligned.
- [ ] 8.3 Make root-image resolution precedence/provenance explicit (`env > config > gt_parent > none`) and ensure the same resolved decision is consumed consistently across infer/eval/vis.
- [ ] 8.4 Add explicit canonical module map + coord range/shape rules in `coord-utils` spec (including nested-point opt-in boundaries).
- [x] 8.5 Re-run strict validation for both overlapping active changes after synchronization.
