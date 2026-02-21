# LVIS (BBox-only vs Poly-prefer Semantic)

This folder contains raw LVIS assets (`raw/`) and a few **dataset-fixed** JSONL
variants for ablations on geometry format.

Key constraint (CoordExp JSONL contract):
- 1 image record -> multiple instances
- each instance has exactly **one** geometry: `bbox_2d` OR `poly` (never both)

Unified runner note:
- `./public_data/run.sh lvis ...` now routes shared internals through the modular pipeline/factory.
- Canonical preset artifacts are `*.jsonl`, `*.norm.jsonl`, and `*.coord.jsonl`.
- Optional max-object filtering stays off by default; enable with `PUBLIC_DATA_MAX_OBJECTS=<N>` and preset naming uses canonical suffix `_max{N}`.

## Background: why LVIS `poly` can be tricky

LVIS follows COCO-style `segmentation`, where **one instance can contain multiple
polygon parts** (e.g. two disconnected visible regions).

Also, LVIS masks represent the **visible** part of an object. Under heavy
occlusion, the polygon can be a tiny fragment that is **not semantically
faithful** to the "full object extent" (classic example: a plate mostly covered
by pizza). In such cases, any polygon enclosure derived from the visible mask
(convex hull, LCC, etc.) can look "wrong" for a semantic grounding task.

Because our priority is **semantic correctness first**, we explicitly support
fallback-to-bbox for these edge cases.

## Two geometry modes

We export two main modes via `public_data/scripts/build_lvis_hull_mix.py`:

1) `bbox_only`
- Always emits `bbox_2d` for every instance.
- Useful for benchmarking against bbox-only detectors and for cross-dataset
  compatibility (many datasets have bbox but not polygons).

2) `poly_prefer_semantic` (poly-prefer, fallback-to-bbox)
- Build a single polygon candidate from LVIS segmentation by:
  - union all parts (multi-part instance)
  - compute **convex hull** of the union mask
  - simplify / cap vertices (for ablation, e.g. cap=10 or cap=20)
- Emit `poly` whenever it is eligible under the vertex cap.
- Fallback to `bbox_2d` when:
  - a cap-respecting poly cannot be formed, or
  - a **semantic guard** triggers (visible-mask fragment is too small / too
    unfaithful to bbox extent).

Note: this mode is intentionally **not** driven by "cost-effectiveness" (IoU per
extra point). It is driven by the ability to keep polygons semantically correct.

## Export scripts and outputs

### From scratch on a new machine / cluster node

1) Download + extract LVIS annotations and COCO2017 images:
```bash
./public_data/run.sh lvis download
```

This writes to:
- `public_data/lvis/raw/annotations/` (LVIS v1 train/val JSON)
- `public_data/lvis/raw/images/` (`train2017/` and `val2017/`)

2) Export the two geometry modes (train+val, max60 objects/image, cap10 + cap20):
```bash
bash public_data/scripts/export_lvis_bbox_poly_prefer_semantic_max60.sh
```

Alternatively, a one-command reproducer (downloads only if missing) is:
```bash
bash public_data/lvis/reproduce_max60_exports.sh
```

Notes:
- The export script materializes a real `images/` directory inside each output dataset directory
  with pre-rescaled images matching JSONL `width/height` (no symlinks; runtime resizing is forbidden).
- Token-length sanity is best-effort: it runs only if `model_cache/Qwen3-VL-8B-Instruct-coordexp` exists.

### Outputs
```bash
bash public_data/scripts/export_lvis_bbox_poly_prefer_semantic_max60.sh
```

Outputs:

- BBox-only (max60):
  - `public_data/lvis/rescale_32_768_bbox_max60/`
  - `train.bbox_only.max60.{jsonl,norm.jsonl,coord.jsonl}`
  - `val.bbox_only.max60.{jsonl,norm.jsonl,coord.jsonl}`

- Poly-prefer semantic (max60):
  - `public_data/lvis/rescale_32_768_poly_prefer_semantic_max60/`
  - `train.poly_prefer_semantic_cap10.max60.{jsonl,norm.jsonl,coord.jsonl}`
  - `train.poly_prefer_semantic_cap20.max60.{jsonl,norm.jsonl,coord.jsonl}`
  - `val.poly_prefer_semantic_cap10.max60.{jsonl,norm.jsonl,coord.jsonl}`
  - `val.poly_prefer_semantic_cap20.max60.{jsonl,norm.jsonl,coord.jsonl}`

Per-split stats are saved alongside the JSONLs:
- `*.build_stats.json`: geometry policy decisions and semantic-guard counters
- `*.filter_stats.json`: max-objects filter stats
- `length.*.assistant_tokens.json`: GT assistant token-length stats for budgeting

## Coordinate formats (raw digits vs coord tokens)

Each export produces 3 JSONLs:
- `*.jsonl`: pixel-space coords
- `*.norm.jsonl`: normalized ints in [0,999] ("raw digits")
- `*.coord.jsonl`: `<|coord_k|>` tokens with k in [0,999]

These are **offline-normalized** to avoid runtime scaling. See
`public_data/scripts/README.md` for normalization details.
