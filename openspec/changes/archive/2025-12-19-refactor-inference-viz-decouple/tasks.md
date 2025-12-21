## 1. Implementation
- [x] 1.1 Add shared coord utilities module (token encode/decode, norm1000↔pixel scaling, clamp, degenerate checks, line→bbox, bbox→segm) wired through `src/common/geometry` and `src/common/schemas`, retaining raw invalid geometry for diagnostics, with unit tests.
- [x] 1.2 Introduce inference-only runner CLI (e.g., `scripts/run_infer.py`) that loads a checkpoint, runs generation with baseline decoding params, enforces single-image inputs, and writes robust JSONL (raw text, parsed preds required, coord_mode=`norm1000`, width/height, error field on failure, scores=1.0, retains flawed raw output when parsing fails).
- [x] 1.3 Update visualization tool to consume inference JSONL (parsed preds preferred; fallback reparsing via shared parser) without running inference; keep per-line robustness, scaling via coord utilities, and error-tolerant overlays.
- [x] 1.4 Wire detection evaluator to the shared coord utilities via `src/common/geometry`, honoring coord_mode to avoid double scaling; ensure bbox–poly/line IoU bridging remains correct and per-image diagnostics keep raw invalid shapes.
- [x] 1.5 Document the staged workflow (inference → eval and/or viz) and CLI usage; include guidance for comparing checkpoints/ablations; refine existing doc rather than creating a new one.
- [x] 1.6 Add/adjust fixtures and tests covering inference output schema (with errors), JSON repair robustness (aggressive repair allowed), coord scaling edge cases, degenerate drop-with-counter, and the end-to-end staged flow (smoke test consuming inference JSONL for eval+viz).

## 2. Validation
- [x] 2.1 `openspec validate refactor-inference-viz-decouple --strict` passes.
