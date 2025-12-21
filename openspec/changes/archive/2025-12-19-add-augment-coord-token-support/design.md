## Context
- Coord-token datasets (`<|coord_k|>`) are supported in loaders/codec but augmentation assumes numeric floats, causing crashes.
- Current workaround disables augmentation, losing robustness benefits.

## Goals
- Let the existing augmentation pipeline operate without changes to its internals, by converting inputs/outputs when coord tokens are enabled.
- Preserve template/output behaviour: after augmentation, geometry should still be expressed as coord tokens for chat text.
- Keep numeric datasets unaffected.

## Non-Goals
- Changing augmentation algorithms or adding new ops.
- Expanding coord vocab or altering tokenizer.

## Decisions
- **Convert at the boundary:** Preprocessor converts token geometries to ints before augmentation; post-process converts back to tokens.
- **Config gate:** Activate only when `custom.coord_tokens.enabled` is true.
- **Cache reuse:** Prefer `_coord_token_ints` / `_coord_tokens` caches when present to avoid re-parsing.
- **Error handling:** If conversion encounters out-of-range values (e.g., 1000) raise clear ValueError early.

## Risks / Trade-offs
- Minor performance cost for conversion; mitigated by cache reuse.
- Rounding/clamping changes could drift if not identical to numeric path; testing will assert round-trip for identity/no-op cases.

## Open Questions
- (Resolved by request) Keep public geometry as tokens; retain caches for losses only as needed.
