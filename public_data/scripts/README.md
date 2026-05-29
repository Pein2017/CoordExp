Offline normalization defaults
==============================

Default behavior
- All LVIS pipeline outputs are pre-normalized to norm1000 (0–999) for both numeric text JSONLs and coord-token JSONLs. No runtime normalization is required.
- Unified canonical artifacts:
  - `{split}.jsonl`: pixel-space records after resize/filter
  - `{split}.norm.jsonl`: normalized integer coords in [0,999]
  - `{split}.coord.jsonl`: coord-token records

LVIS: bbox-only vs poly-prefer (fallback-to-bbox)
================================================

For geometry-format ablations on LVIS we export two main dataset variants:

- `bbox_only`: every instance emits `bbox_2d`
- `poly_prefer_semantic`: prefer a **single** polygon per instance when possible; fallback to bbox
  for semantic edge cases (visible mask is a tiny fragment under occlusion) or when a capped poly
  cannot be formed.

Reproducer script (train+val, max60 objects/image, cap10 + cap20, plus token-length sanity):
```bash
bash public_data/scripts/export_lvis_bbox_poly_prefer_semantic_max60.sh
```

Core builder:
- `build_lvis_hull_mix.py`
  - `--geometry-policy bbox_only|poly_prefer_semantic|mix`
  - `--poly-cap 10|20` (vertex cap for ablation)

For more context + output paths, see:
- `public_data/lvis/README.md`

Script
- `convert_to_coord_tokens.py`
  - Pixel → norm ints and tokens in one pass: `--output-norm ... --output-tokens ...`
  - Already-normalized ints/tokens → tokens only: `--assume-normalized --output-tokens ...`
  - Geometry keys default: `bbox_2d poly line`
  - Strict: raises on out-of-range / malformed coords (no clamping); rounds to [0,999] when valid.

Training config alignment
- Numeric JSONL (norm1000 ints): set `custom.emit_norm: none`, `coord_tokens.enabled: false`.
- Coord-token JSONL (norm1000 tokens): set `coord_tokens.enabled: true`; no runtime scaling.
- Avoid double-scaling: don’t set `emit_norm: norm1000` when using pre-normalized numeric JSONLs.

Phase 1 COCO views
==================

Canonical COCO views under `public_data/coco/views/**` store norm1000 integer
coordinates in strict JSON and resolve `images[]` through the shared image
store `public_data/coco/images/res-1024`. Assistant targets may still render
those integers as Qwen `<|coord_k|>` tokens; that is a builder/template
boundary, not the stored view format.

The 12k `len-12000` policy counts image patch tokens, system/user tokens, and
the rendered assistant object sequence. `max_objects` / `max-60` is retained as
legacy artifact provenance rather than current runtime filtering policy.
