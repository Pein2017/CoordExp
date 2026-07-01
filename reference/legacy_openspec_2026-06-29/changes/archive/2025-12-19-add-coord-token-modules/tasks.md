# Tasks (ordered)
- [x] Add config flag/schema stub for `coord_tokens.enabled` and optional `skip_bbox_norm` to gate the new path without altering defaults.
- [x] Implement `src/coord_tokens/codec.py` with token↔int↔float mapping, coord-token id mask utilities, and the 0..999 default range (optional 1000 via config).
- [x] Add `src/coord_tokens/validator.py` to accept tokenized geometries while preserving existing numeric validation paths.
- [x] Add `src/coord_tokens/template_adapter.py` to bypass bbox normalization when coord-token mode is on; wire into template selection via the new flag.
- [x] Ensure the offline converter (`scripts/convert_to_coord_tokens.py`) aligns with the spec: round to norm1000, default to 0..999 tokens, configurable handling if a value would round to 1000.
- [x] Add loss helpers (`src/coord_tokens/loss.py`) for expectation decoding and CE masking using the codec.
- [x] Add minimal tests/fixtures covering: tokenized JSONL acceptance, codec round-trip (0..999 and optional 1000), template skip path, and converter output.
- [x] Update docs/config examples to show how to enable coord-token mode; keep default numeric path unchanged.
- [x] Run `openspec validate add-coord-token-modules --strict` and fix any spec issues.
