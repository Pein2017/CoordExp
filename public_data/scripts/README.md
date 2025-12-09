Offline normalization defaults
==============================

Default behavior
- All LVIS pipeline outputs are pre-normalized to norm1000 (0–999) for both numeric text JSONLs and coord-token JSONLs. No runtime normalization is required.
- Pixel-space intermediates (`*.raw.jsonl`) are temporary; final `{split}.jsonl` and `{split}.coord.jsonl` are normalized.

Script
- `convert_to_coord_tokens.py`
  - Pixel → norm ints and tokens in one pass: `--output-norm ... --output-tokens ...`
  - Already-normalized ints/tokens → tokens only: `--assume-normalized --output-tokens ...`
  - Geometry keys default: `bbox_2d poly line`
  - Clamps to [0,width-1]/[0,height-1], rounds to [0,999]; validates bounds.

Training config alignment
- Numeric JSONL (norm1000 ints): set `custom.emit_norm: none`, `coord_tokens.enabled: false`.
- Coord-token JSONL (norm1000 tokens): set `coord_tokens.enabled: true`; no runtime scaling.
- Avoid double-scaling: don’t set `emit_norm: norm1000` when using pre-normalized numeric JSONLs.

