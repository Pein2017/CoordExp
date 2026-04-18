---
doc_id: docs.data.contract
layer: docs
doc_type: contract
status: canonical
domain: data
summary: Authoritative JSONL, geometry, and runtime assumptions for dataset ingestion.
updated: 2026-04-13
---

# Data JSONL Contract (Global)

This document defines the universal JSONL format consumed by all CoordExp training/eval datasets (public detection/grounding and any legacy sources). Every record MUST adhere to this contract so the shared chat-template pipeline can process all sources.

Important separation:
- This contract is for **raw JSONL files** (`*.jsonl`, `*.norm.jsonl`, `*.coord.jsonl`), which must remain strict JSON.
- Model-facing assistant text is rendered as **CoordJSON** (`{"objects": [...]}`) and may express geometry either with bare coord tokens or bare norm1000 JSON integers, depending on the authored geometry-expression mode.

## Top-Level Record
- **Provenance**: Records are typically produced by dataset-specific converters (e.g., `public_data/scripts/convert_lvis.py`) and then resized/tokenized via `public_data/scripts/rescale_jsonl.py` and `public_data/scripts/convert_to_coord_tokens.py` (see [`PREPARATION.md`](PREPARATION.md)). Regardless of source, they MUST match this contract.
- `images` (list[str], required): Relative paths to image files; resolved against the JSONL directory.
- `objects` (list[object], required): Structured annotations (see below).
- `width` (int, required): Image width in pixels (original or post-resize if applied offline).
- `height` (int, required): Image height in pixels.
- `summary` (str, optional): Single-line English summary (if provided by the dataset). When present, it should be built from the raw `desc` strings; identical entries may be merged into `desc xN`. Missing objects or empty `desc` should fail during conversion.
- `metadata` (object, optional): Free-form metadata for provenance (not automatically injected).

## Objects
Each object MUST contain exactly one geometry field plus a non-empty `desc`.
- `desc` (str, required): Plain English description / class string (no hierarchy or slash prefixes required).
- One geometry (required, mutually exclusive):
  - `bbox_2d`: `[x1, y1, x2, y2]` pixel coordinates.
  - `poly`: flat list `[x1, y1, x2, y2, ...]` (even length, ≥6 values / ≥3 points). Optional `poly_points` (int) should equal `len(poly)/2` when present.
- No additional geometry fields are allowed on the same object.

Note: only `bbox_2d` and `poly` are supported in CoordExp; `line` geometries are rejected.

### Geometry keys and coordinate space (canonical)
- Accepted geometry keys are **only** `bbox_2d` or `poly` (plus optional `poly_points`). Legacy aliases `bbox` or `polygon` must be converted during preprocessing.
- **Training coordinate space is pre-normalized norm1000**:
  - Numeric coords may be bare integers in `0..999`, OR
  - coord tokens `<|coord_k|>` where `k ∈ [0, 999]`.
  Pixel-space floats are allowed only as intermediate artifacts before conversion; do not feed them directly into training.
- Keep `custom.coord_tokens.skip_bbox_norm: true` in both geometry-expression modes to prevent double scaling.

### Raw JSONL vs assistant CoordJSON
- Raw JSONL must stay strict JSON:
  - coord-token surfaces store quoted token strings (e.g., `"<|coord_123|>"`),
  - raw-text norm1000 surfaces store bare numeric integers (e.g., `123`).
- Assistant dense outputs use top-level `{"objects": [...]}` and either:
  - bare CoordTok literals in geometry arrays (e.g., `[<|coord_123|>, <|coord_456|>, ...]`), or
  - bare norm1000 integers (e.g., `[123, 456, 789, 900]`).
- Parsing boundary for assistant-output-like text is `CoordJSON -> strict JSON` transpilation, then `json.loads`.

### Canonical preset data vs offline-prepared bbox-format branches
- Canonical raw and preset JSONL remain model-independent `xyxy` surfaces.
- Runtime training and builder code MUST NOT reinterpret canonical preset
  `bbox_2d` records into another bbox chart on the fly.
- Non-canonical model-facing bbox charts are authored offline under
  sibling preset roots such as
  `public_data/<dataset>/<preset>_cxcy_logw_logh/` and
  `public_data/<dataset>/<preset>_cxcywh/`.
- For `<preset>_cxcy_logw_logh/`:
  - `<split>.jsonl` stores norm1000 integer `bbox_2d` slots in
    `[cx, cy, logw, logh]`,
  - `<split>.norm.jsonl` mirrors the same numeric lattice in preset-compatible
    layout,
  - `<split>.coord.jsonl` stores the tokenized form of that same lattice,
  - branch records must include prepared-bbox provenance metadata,
  - the first supported branch surface is `bbox_2d`-only and fails fast on
    `poly` or mixed geometry.
- For `<preset>_cxcywh/`:
  - `<split>.jsonl` stores norm1000 integer `bbox_2d` slots in
    `[cx, cy, w, h]`,
  - `<split>.norm.jsonl` mirrors the same numeric lattice in preset-compatible
    layout,
  - `<split>.coord.jsonl` stores the tokenized form of that same lattice,
  - branch records must include prepared-bbox provenance metadata,
  - the first supported branch surface is `bbox_2d`-only and fails fast on
    `poly` or mixed geometry.
- Inference, visualization, matching, and emitted evaluation artifacts remain
  canonical `xyxy`; any non-canonical prediction surface such as
  `cxcy_logw_logh` or `cxcywh` must be inverted back to canonical `xyxy`
  before those downstream boundaries.

## Invariants
- For training, coords MUST be pre-normalized to norm1000 (ints 0..999) or pre-tokenized `<|coord_k|>` values. Width/height must always be present.
- This raw-data invariant applies to canonical raw JSONL and other
  model-independent geometry surfaces. Offline-prepared bbox-format branches may
  encode model-facing bbox slots differently, but they must declare that branch
  provenance explicitly instead of relying on runtime reinterpretation.
- Image paths remain relative in JSONL; loaders resolve them to absolute paths.
- Geometry is validated; records with multiple geometry fields per object are rejected.
- Runtime payload emission is fail-fast: builders/preprocessors reject objects with missing geometry, multiple geometry fields, invalid bbox/poly arity, or empty `desc` instead of serializing partial objects.
- Default ordering invariant: when `custom.object_ordering: sorted` (default), object sequences must already be sorted by `(minY, minX)` in the source JSONL.
- `custom.object_ordering: random` is supported for ablation-style dataset-backed training/eval only and means a deterministic per-epoch reshuffle derived from sample identity plus epoch; it does not change per-object field order inside each object payload.
- Polygon vertices should be canonicalized offline for determinism (recommended; not enforced by the runtime loader/builder):
  - drop duplicated closing point if present
  - order vertices clockwise around the centroid (angle sort)
  - rotate so the top-most (then left-most) vertex is first
  This matches the public-data converters (e.g., `public_data/scripts/convert_to_coord_tokens.py`) and the prompt spec.
- Optional fields (e.g., `summary`, `poly_points`, `metadata`) may be absent; templates and preprocessors must tolerate absence.
- Geometry-expression modes:
  - `custom.coord_tokens.enabled: true` => coord-token assistant targets backed by `*.coord.jsonl`
  - `custom.coord_tokens.enabled: false` => raw-text norm1000 assistant targets backed by `*.norm.jsonl`
  In both cases keep `custom.coord_tokens.skip_bbox_norm: true`.

## Example
```json
{
  "images": ["images/0001.jpg"],
  "objects": [
    {"poly": ["<|coord_12|>", "<|coord_34|>", "<|coord_56|>", "<|coord_34|>", "<|coord_56|>", "<|coord_78|>", "<|coord_12|>", "<|coord_78|>"], "poly_points": 4, "desc": "yellow box"},
    {"bbox_2d": ["<|coord_100|>", "<|coord_120|>", "<|coord_180|>", "<|coord_200|>"], "desc": "tool cabinet"}
  ],
  "width": 768,
  "height": 512
}
```

## Current Sources (checked)
- `public_data/*`: LVIS/COCO/Objects365-style exports; polygons include `poly_points`; descriptions are English classes/phrases.
- The cached LVIS bbox Stage-1 exports under `public_data/lvis/rescale_32_1024_bbox_max60/{train,val}.coord.jsonl`
  already satisfy the default sorted-order invariant above.
  - verification status: full-row scan, `0` unsorted rows in train and `0` unsorted rows in val
  - comparator used: `(minY, minX)` from bbox top-left, matching the runtime loader contract

All future domains MUST emit this contract to remain compatible with the shared chat template pipeline.

For an exact view of how a record plus the default prompts are rendered by the Qwen3-VL chat template, run:
```
PYTHONPATH=. conda run -n ms python scripts/tools/inspect_chat_template.py --jsonl <path/to/data.jsonl> --index 0
```
